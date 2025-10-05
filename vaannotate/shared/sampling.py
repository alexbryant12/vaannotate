"""Sampling utilities implementing the deterministic strategy described in the spec."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sqlite3

from .database import Database, ensure_schema
from .metadata import (
    MetadataField,
    MetadataFilterCondition,
    discover_corpus_metadata,
    extract_document_metadata,
)
from . import models


@dataclass
class SamplingFilters:
    metadata_filters: List[MetadataFilterCondition]
    match_any: bool = False

    def field_keys(self) -> List[str]:
        return [condition.field for condition in self.metadata_filters]


@dataclass
class ReviewerAssignment:
    reviewer_id: str
    units: List[Dict[str, object]]


def _hash_seed(seed: int, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(digest[:16], 16)


def _note_year_sort_value(value: object) -> int:
    if value is None or value == "":
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return -1


def candidate_documents(
    corpus_db: Database | sqlite3.Connection,
    level: str,
    filters: SamplingFilters,
    *,
    metadata_fields: Sequence[MetadataField] | None = None,
    stratify_keys: Sequence[str] | None = None,
) -> List[sqlite3.Row | Dict[str, object]]:
    if isinstance(corpus_db, Database):
        with corpus_db.connect() as conn:
            return _candidate_documents_from_connection(
                conn,
                level,
                filters,
                metadata_fields,
                stratify_keys,
            )
    return _candidate_documents_from_connection(
        corpus_db,
        level,
        filters,
        metadata_fields,
        stratify_keys,
    )


def _candidate_documents_from_connection(
    conn: sqlite3.Connection,
    level: str,
    filters: SamplingFilters,
    metadata_fields: Sequence[MetadataField] | None,
    stratify_keys: Sequence[str] | None,
) -> List[sqlite3.Row | Dict[str, object]]:
    available_fields: Sequence[MetadataField]
    if metadata_fields is None:
        available_fields = discover_corpus_metadata(conn)
    else:
        available_fields = metadata_fields
    field_lookup: Dict[str, MetadataField] = {field.key: field for field in available_fields}

    requested: Dict[str, MetadataField] = {}
    for condition in filters.metadata_filters:
        field = field_lookup.get(condition.field)
        if field:
            requested[field.key] = field
    if stratify_keys:
        for key in stratify_keys:
            field = field_lookup.get(str(key))
            if field:
                requested[field.key] = field
    active_fields = list(requested.values())

    select_parts = [
        "documents.doc_id AS doc_id",
        "documents.patient_icn AS patient_icn",
        "documents.note_year AS note_year",
        "documents.notetype AS notetype",
        "documents.date_note AS date_note",
        "documents.cptname AS cptname",
        "documents.sta3n AS sta3n",
        "documents.hash AS hash",
        "documents.text AS text",
    ]
    seen_aliases = {
        "doc_id",
        "patient_icn",
        "note_year",
        "notetype",
        "date_note",
        "cptname",
        "sta3n",
        "hash",
        "text",
    }
    for field in active_fields:
        if field.alias in seen_aliases:
            continue
        select_parts.append(f'{field.expression} AS "{field.alias}"')
        seen_aliases.add(field.alias)

    clauses: List[str] = []
    params: List[object] = []
    for condition in filters.metadata_filters:
        field = field_lookup.get(condition.field)
        if not field:
            continue
        local_clauses: List[str] = []
        if field.data_type == "number":
            if condition.min_value is not None:
                try:
                    params.append(float(condition.min_value))
                    local_clauses.append(f"CAST({field.expression} AS REAL) >= ?")
                except ValueError:
                    pass
            if condition.max_value is not None:
                try:
                    params.append(float(condition.max_value))
                    local_clauses.append(f"CAST({field.expression} AS REAL) <= ?")
                except ValueError:
                    pass
            if condition.values:
                numeric_values: List[float] = []
                for value in condition.values:
                    try:
                        numeric_values.append(float(value))
                    except ValueError:
                        continue
                if numeric_values:
                    placeholders = ",".join(["?"] * len(numeric_values))
                    local_clauses.append(f"CAST({field.expression} AS REAL) IN ({placeholders})")
                    params.extend(numeric_values)
        elif field.data_type == "date":
            if condition.min_value:
                local_clauses.append(f"{field.expression} >= ?")
                params.append(condition.min_value)
            if condition.max_value:
                local_clauses.append(f"{field.expression} <= ?")
                params.append(condition.max_value)
            if condition.values:
                placeholders = ",".join(["?"] * len(condition.values))
                local_clauses.append(f"{field.expression} IN ({placeholders})")
                params.extend(condition.values)
        else:
            if condition.values:
                placeholders = ",".join(["?"] * len(condition.values))
                local_clauses.append(f"{field.expression} IN ({placeholders})")
                params.extend(condition.values)
        if local_clauses:
            clauses.append(" AND ".join(local_clauses))

    query = [
        "SELECT",
        ", ".join(select_parts),
        "FROM documents",
        "JOIN patients ON patients.patient_icn = documents.patient_icn",
    ]
    if clauses:
        joiner = " OR " if filters.match_any else " AND "
        query.append("WHERE " + joiner.join(f"({clause})" for clause in clauses))
    query.append(
        "ORDER BY documents.patient_icn, documents.note_year, documents.doc_id",
    )
    sql = " ".join(query)
    rows = conn.execute(sql, params).fetchall()

    if level == "multi_doc":
        grouped: Dict[str, List[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["patient_icn"], []).append(row)
        aggregated: List[Dict[str, object]] = []
        for patient_icn, docs in grouped.items():
            ordered_docs = sorted(
                docs,
                key=lambda item: (_note_year_sort_value(item["note_year"]), item["doc_id"]),
            )
            primary_row = ordered_docs[0]
            primary_dict = dict(primary_row)
            doc_payloads = []
            for idx, doc in enumerate(ordered_docs):
                doc_dict = dict(doc)
                doc_dict.pop("documents", None)
                doc_dict["order_index"] = idx
                metadata = extract_document_metadata(doc_dict)
                doc_dict["metadata"] = metadata
                doc_payloads.append(doc_dict)
            primary_metadata = extract_document_metadata(primary_dict)
            entry: Dict[str, object] = {
                "unit_id": patient_icn,
                "patient_icn": patient_icn,
                "doc_id": None,
                "note_year": primary_dict.get("note_year"),
                "notetype": primary_dict.get("notetype"),
                "date_note": primary_dict.get("date_note"),
                "cptname": primary_dict.get("cptname"),
                "sta3n": primary_dict.get("sta3n"),
                "softlabel": primary_dict.get("patient__softlabel"),
                "note_count": len(ordered_docs),
                "documents": doc_payloads,
                "metadata": primary_metadata,
            }
            for field in active_fields:
                entry[field.alias] = primary_dict.get(field.alias)
            aggregated.append(entry)
        return aggregated
    return rows


def stratify(rows: Sequence[sqlite3.Row], keys: Sequence[str]) -> Dict[str, List[sqlite3.Row]]:
    strata: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        key = "|".join(str(row[k]) for k in keys)
        strata.setdefault(key, []).append(row)
    return strata


def allocate_units(
    rows: Sequence[sqlite3.Row],
    reviewers: Sequence[Dict[str, str]],
    overlap_n: int,
    seed: int,
    total_n: int | None = None,
    strat_keys: Sequence[str] | None = None,
) -> Dict[str, ReviewerAssignment]:
    reviewer_units: Dict[str, ReviewerAssignment] = {r["id"]: ReviewerAssignment(r["id"], []) for r in reviewers}
    if strat_keys:
        strata = stratify(rows, strat_keys)
    else:
        strata = {"__all__": list(rows)}
    if strat_keys and total_n is not None and len(strata) > total_n:
        raise ValueError(
            "The requested sample size is smaller than the number of strata. "
            "Increase the total units or adjust the stratification keys."
        )
    allocations: Dict[str, int] = {}
    total_available = sum(len(list(items)) for items in strata.values())
    target_total = total_available if total_n is None else min(total_n, total_available)
    if total_n is None:
        for strata_key, items in strata.items():
            allocations[strata_key] = len(items)
    else:
        allocations = {key: 0 for key in strata}
        remaining = target_total
        active = [key for key, items in strata.items() if len(items) > 0]
        while remaining > 0 and active:
            share = max(1, remaining // len(active))
            next_active: list[str] = []
            for strata_key in list(active):
                capacity = len(strata[strata_key]) - allocations[strata_key]
                if capacity <= 0:
                    continue
                take = min(share, capacity, remaining)
                allocations[strata_key] += take
                remaining -= take
                if allocations[strata_key] < len(strata[strata_key]):
                    next_active.append(strata_key)
                if remaining == 0:
                    break
            active = next_active if remaining > 0 else []
    remaining = target_total if total_n is not None else None
    for strata_key in sorted(strata.keys()):
        items = list(strata[strata_key])
        rng = random.Random(_hash_seed(seed, strata_key))
        rng.shuffle(items)
        take = allocations.get(strata_key, len(items))
        if remaining is not None:
            if remaining <= 0:
                break
            if take > remaining:
                take = remaining
        items = items[:take]
        overlap_size = min(overlap_n, len(items))
        overlap_pool = items[:overlap_size]
        remainder = items[overlap_size:]
        for reviewer in reviewer_units.values():
            for row in overlap_pool:
                payload = _build_unit_payload(row, strata_key, is_overlap=True)
                reviewer.units.append(payload)
        for idx, row in enumerate(remainder):
            reviewer = reviewers[idx % len(reviewers)]["id"]
            payload = _build_unit_payload(row, strata_key, is_overlap=False)
            reviewer_units[reviewer].units.append(payload)
        if remaining is not None:
            remaining -= len(items)
            if remaining <= 0:
                break
    # randomize display order per reviewer deterministically
    for reviewer in reviewer_units.values():
        rng = random.Random(_hash_seed(seed, reviewer.reviewer_id))
        rng.shuffle(reviewer.units)
        for rank, unit in enumerate(reviewer.units, start=1):
            unit["display_rank"] = rank
    return reviewer_units


def _build_unit_payload(row: sqlite3.Row | Dict[str, object], strata_key: str, is_overlap: bool) -> Dict[str, object]:
    if isinstance(row, dict):
        data = dict(row)
    else:
        data = {key: row[key] for key in row.keys()}
    documents = data.get("documents")
    doc_payloads: List[Dict[str, object]] = []
    if documents:
        for doc in documents:
            doc_copy = dict(doc)
            doc_copy.pop("documents", None)
            try:
                doc_copy["order_index"] = int(doc_copy.get("order_index", len(doc_payloads)))
            except (TypeError, ValueError):
                doc_copy["order_index"] = len(doc_payloads)
            metadata = extract_document_metadata(doc_copy)
            doc_copy["metadata"] = metadata
            doc_payloads.append(doc_copy)
    elif data.get("doc_id"):
        doc_copy = dict(data)
        doc_copy.pop("documents", None)
        try:
            doc_copy["order_index"] = int(doc_copy.get("order_index", 0))
        except (TypeError, ValueError):
            doc_copy["order_index"] = 0
        metadata = extract_document_metadata(doc_copy)
        doc_copy["metadata"] = metadata
        doc_payloads = [doc_copy]
    unit_id = data.get("unit_id") or data.get("doc_id")
    payload = {
        "unit_id": unit_id,
        "patient_icn": data.get("patient_icn"),
        "doc_id": data.get("doc_id"),
        "strata_key": strata_key,
        "is_overlap": 1 if is_overlap else 0,
        "hash": data.get("hash", ""),
        "text": data.get("text", ""),
        "note_count": data.get("note_count"),
        "documents": doc_payloads,
    }
    return payload


def write_manifest(path: Path, assignments: Dict[str, ReviewerAssignment]) -> None:
    fieldnames = ["doc_id", "patient_icn", "strata_key", "assigned_to", "is_overlap", "display_rank"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for reviewer_id, assignment in assignments.items():
            for unit in assignment.units:
                row = {key: unit.get(key, "") for key in fieldnames}
                row["assigned_to"] = reviewer_id
                writer.writerow(row)


def initialize_assignment_db(path: Path) -> Database:
    db = Database(path)
    with db.transaction() as conn:
        ensure_schema(
            conn,
            [
                models.AssignmentUnit,
                models.AssignmentUnitNote,
                models.AssignmentDocument,
                models.Annotation,
                models.Rationale,
                models.Event,
            ],
        )
    return db


def populate_assignment_db(db: Database, reviewer: str, units: Sequence[Dict[str, object]]) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    with db.transaction() as conn:
        for order, unit in enumerate(units):
            doc_id_value = unit.get("doc_id")
            note_count = unit.get("note_count")
            record = models.AssignmentUnit(
                unit_id=str(unit["unit_id"]),
                display_rank=int(unit["display_rank"]),
                patient_icn=str(unit["patient_icn"]),
                doc_id=str(doc_id_value) if doc_id_value is not None else None,
                note_count=int(note_count) if isinstance(note_count, int) else None,
                complete=0,
                opened_at=None,
                completed_at=None,
            )
            record.save(conn)
            documents = unit.get("documents") or []
            if documents:
                for doc in documents:
                    doc_id = doc.get("doc_id")
                    if not doc_id:
                        continue
                    note = models.AssignmentUnitNote(
                        unit_id=str(unit["unit_id"]),
                        doc_id=str(doc_id),
                        order_index=int(doc.get("order_index", 0)),
                    )
                    note.save(conn)
                    metadata = extract_document_metadata(doc)
                    metadata_json = (
                        json.dumps(metadata, sort_keys=True) if metadata else None
                    )
                    doc_record = models.AssignmentDocument(
                        doc_id=str(doc_id),
                        hash=str(doc.get("hash", "")),
                        text=str(doc.get("text", "")),
                        metadata_json=metadata_json,
                    )
                    doc_record.save(conn)
            elif doc_id_value:
                note = models.AssignmentUnitNote(
                    unit_id=str(unit["unit_id"]),
                    doc_id=str(doc_id_value),
                    order_index=order,
                )
                note.save(conn)
                metadata = extract_document_metadata(unit)
                metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None
                doc = models.AssignmentDocument(
                    doc_id=str(doc_id_value),
                    hash=str(unit.get("hash", "")),
                    text=str(unit.get("text", "")),
                    metadata_json=metadata_json,
                )
                doc.save(conn)
        event = models.Event(
            event_id=f"init:{reviewer}:{timestamp}",
            ts=timestamp,
            actor=reviewer,
            event_type="assignment_initialized",
            payload_json=json.dumps({"unit_count": len(units)}),
        )
        event.save(conn)
