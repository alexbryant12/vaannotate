"""Sampling utilities implementing the deterministic strategy described in the spec."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import sqlite3

from .database import Database, ensure_schema
from . import models


@dataclass
class SamplingFilters:
    patient_filters: Dict[str, object]
    note_filters: Dict[str, object]


@dataclass
class ReviewerAssignment:
    reviewer_id: str
    units: List[Dict[str, object]]


def _hash_seed(seed: int, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return int(digest[:16], 16)


def candidate_documents(corpus_db: Database, level: str, filters: SamplingFilters) -> List[sqlite3.Row]:
    with corpus_db.connect() as conn:
        base_query = [
            "SELECT documents.*, patients.softlabel FROM documents"
            " JOIN patients ON patients.patient_icn = documents.patient_icn"
        ]
        clauses = []
        params: List[object] = []
        pf = filters.patient_filters
        nf = filters.note_filters
        if year := pf.get("year_range"):
            clauses.append("patients.date_index BETWEEN ? AND ?")
            params.extend(year)
        if sta := pf.get("sta3n_in"):
            placeholders = ",".join(["?"] * len(sta))
            clauses.append(f"patients.sta3n IN ({placeholders})")
            params.extend(sta)
        if softlabel := pf.get("softlabel_gte"):
            clauses.append("(patients.softlabel IS NOT NULL AND patients.softlabel >= ?)")
            params.append(softlabel)
        if nf.get("notetype_in"):
            placeholders = ",".join(["?"] * len(nf["notetype_in"]))
            clauses.append(f"documents.notetype IN ({placeholders})")
            params.extend(nf["notetype_in"])
        if nf.get("note_year_range"):
            clauses.append("documents.note_year BETWEEN ? AND ?")
            params.extend(nf["note_year_range"])
        if nf.get("regex"):
            clauses.append("documents.text REGEXP ?")
            params.append(nf["regex"])
        if clauses:
            base_query.append("WHERE " + " AND ".join(clauses))
        base_query.append("ORDER BY documents.note_year")
        sql = " ".join(base_query)
        # register simple regexp implementation
        conn.create_function("REGEXP", 2, lambda pattern, text: 1 if pattern and text and re.search(pattern, text) else 0)
        rows = conn.execute(sql, params).fetchall()
    if level == "multi_doc":
        # for the MVP we treat documents as units for both levels; multi doc
        # callers aggregate downstream.
        return rows
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
    strat_keys: Sequence[str] | None = None,
    per_stratum: int | None = None,
) -> Dict[str, ReviewerAssignment]:
    reviewer_units: Dict[str, ReviewerAssignment] = {r["id"]: ReviewerAssignment(r["id"], []) for r in reviewers}
    if strat_keys:
        strata = stratify(rows, strat_keys)
    else:
        strata = {"__all__": list(rows)}
    for strata_key, items in strata.items():
        items = list(items)
        if per_stratum:
            items = items[:per_stratum]
        rng = random.Random(_hash_seed(seed, strata_key))
        rng.shuffle(items)
        overlap_size = min(overlap_n, len(items))
        overlap_pool = items[:overlap_size]
        remainder = items[overlap_size:]
        for reviewer in reviewer_units.values():
            for row in overlap_pool:
                row_keys = row.keys()
                reviewer.units.append({
                    "unit_id": row["doc_id"],
                    "patient_icn": row["patient_icn"],
                    "doc_id": row["doc_id"],
                    "strata_key": strata_key,
                    "is_overlap": 1,
                    "hash": row["hash"] if "hash" in row_keys else "",
                    "text": row["text"] if "text" in row_keys else "",
                })
        for idx, row in enumerate(remainder):
            reviewer = reviewers[idx % len(reviewers)]["id"]
            row_keys = row.keys()
            reviewer_units[reviewer].units.append({
                "unit_id": row["doc_id"],
                "patient_icn": row["patient_icn"],
                "doc_id": row["doc_id"],
                "strata_key": strata_key,
                "is_overlap": 0,
                "hash": row["hash"] if "hash" in row_keys else "",
                "text": row["text"] if "text" in row_keys else "",
            })
    # randomize display order per reviewer deterministically
    for reviewer in reviewer_units.values():
        rng = random.Random(_hash_seed(seed, reviewer.reviewer_id))
        rng.shuffle(reviewer.units)
        for rank, unit in enumerate(reviewer.units, start=1):
            unit["display_rank"] = rank
    return reviewer_units


def write_manifest(path: Path, assignments: Dict[str, ReviewerAssignment]) -> None:
    fieldnames = ["doc_id", "patient_icn", "strata_key", "assigned_to", "is_overlap", "display_rank"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for reviewer_id, assignment in assignments.items():
            for unit in assignment.units:
                row = dict(unit)
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
            record = models.AssignmentUnit(
                unit_id=str(unit["unit_id"]),
                display_rank=int(unit["display_rank"]),
                patient_icn=str(unit["patient_icn"]),
                doc_id=str(unit["doc_id"]),
                note_count=None,
                complete=0,
                opened_at=None,
                completed_at=None,
            )
            record.save(conn)
            note = models.AssignmentUnitNote(
                unit_id=str(unit["unit_id"]),
                doc_id=str(unit["doc_id"]),
                order_index=order,
            )
            note.save(conn)
            doc = models.AssignmentDocument(
                doc_id=str(unit["doc_id"]),
                hash=str(unit.get("hash", "")),
                text=str(unit.get("text", "")),
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
