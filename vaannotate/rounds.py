"""Round creation, sampling, and aggregation."""
from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .project import fetch_labelset
from .schema import initialize_assignment_db, initialize_round_aggregate_db
from .shared.metadata import (
    MetadataFilterCondition,
    MetadataField,
    discover_corpus_metadata,
    extract_document_metadata,
)
from .shared.sampling import SamplingFilters, candidate_documents
from .utils import (
    canonical_json,
    copy_sqlite_database,
    deterministic_choice,
    ensure_dir,
    stable_hash,
)


def _date_sort_value(value: object) -> tuple:
    if value is None:
        return ("", "")
    text = str(value)
    return (text, text)


@dataclass
class CandidateUnit:
    unit_id: str
    patient_icn: str
    doc_id: str | None
    strata_key: str
    payload: dict


@dataclass
class AssignmentUnit:
    unit_id: str
    patient_icn: str
    doc_id: str | None
    payload: dict


class RoundBuilder:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.project_db = project_root / "project.db"

    def _connect_project(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.project_db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _connect_corpus(self, corpus_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(corpus_path)
        conn.row_factory = sqlite3.Row
        return conn

    def generate_round(
        self,
        pheno_id: str,
        config_path: Path,
        created_by: str,
        preselected_units_csv: Path | str | None = None,
    ) -> dict:
        config = json.loads(Path(config_path).read_text("utf-8"))
        with self._connect_project() as project_conn:
            pheno = project_conn.execute(
                "SELECT * FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
            if not pheno:
                raise ValueError(f"Phenotype {pheno_id} not found")
            storage_path = pheno["storage_path"]
            if not storage_path:
                raise ValueError("Phenotype storage path is not defined")
            storage_dir = Path(storage_path)
            if not storage_dir.is_absolute():
                storage_dir = (self.project_root / storage_dir).resolve()
            corpus_id = config.get("corpus_id")
            corpus_row = None
            if corpus_id:
                corpus_row = project_conn.execute(
                    "SELECT * FROM project_corpora WHERE corpus_id=?",
                    (corpus_id,),
                ).fetchone()
            if corpus_row:
                corpus_path = self.project_root / corpus_row["relative_path"]
            else:
                legacy_path = config.get("corpus_path")
                if not legacy_path:
                    raise ValueError("Round configuration does not specify a corpus")
                corpus_path = Path(legacy_path)
                if not corpus_path.is_absolute():
                    corpus_path = (self.project_root / corpus_path).resolve()
            with self._connect_corpus(corpus_path) as corpus_conn:
                labelset = fetch_labelset(project_conn, config["labelset_id"])
                round_number = config.get("round_number")
                csv_override: Path | None = None
                if not preselected_units_csv:
                    csv_value = config.get("preselected_units_csv")
                    if csv_value:
                        csv_override = Path(csv_value)
                        if not csv_override.is_absolute():
                            csv_override = (Path(config_path).parent / csv_override).resolve()
                else:
                    csv_override = Path(preselected_units_csv)
                round_id = config.get("round_id") or f"{pheno_id}_r{round_number}"
                status = str(config.get("status") or "active")
                if csv_override and not csv_override.exists():
                    raise FileNotFoundError(csv_override)
            rng_seed = config.get("rng_seed", 0)
            config_hash = stable_hash(canonical_json(config))
            phenotype_dir = storage_dir
            round_dir = phenotype_dir / "rounds" / f"round_{round_number}"
            ensure_dir(round_dir)
            ensure_dir(round_dir / "assignments")
            ensure_dir(round_dir / "imports")
            ensure_dir(round_dir / "reports" / "confusion_matrices")
            ensure_dir(round_dir / "reports" / "exports")

            if csv_override:
                candidates = list(
                    self._build_preselected_candidates(corpus_conn, pheno, csv_override)
                )
            else:
                candidates = list(self._build_candidates(corpus_conn, pheno, config))
            if not candidates:
                raise ValueError("No candidates matched filters")

            reviewer_ids = [reviewer["id"] for reviewer in config["reviewers"]]
            reviewer_assignments = self._allocate_units(
                candidates,
                reviewer_ids,
                rng_seed,
                config.get("overlap_n", 0),
                config.get("total_n"),
                config.get("stratification"),
                preserve_input_order=bool(csv_override),
            )

            if csv_override:
                config["preselected_units_csv"] = str(csv_override)
            config["status"] = status

            manifest_path = round_dir / "manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as fh:
                fieldnames = ["unit_id", "patient_icn", "doc_id", "strata_key", "assigned_to", "is_overlap"]
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for reviewer_id, assignments in reviewer_assignments.items():
                    for unit in assignments:
                        writer.writerow(
                            {
                                "unit_id": unit.unit_id,
                                "patient_icn": unit.patient_icn,
                                "doc_id": unit.doc_id,
                                "strata_key": unit.payload["strata_key"],
                                "assigned_to": reviewer_id,
                                "is_overlap": 1 if unit.payload.get("is_overlap") else 0,
                            }
                        )

            (round_dir / "round_config.json").write_text(canonical_json(config), encoding="utf-8")

            project_conn.execute(
                """
                INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    round_id,
                    pheno_id,
                    round_number,
                    labelset["labelset_id"],
                    config_hash,
                    rng_seed,
                    status,
                    datetime.utcnow().isoformat(),
                ),
            )
            project_conn.execute(
                "INSERT OR REPLACE INTO round_configs(round_id, config_json) VALUES (?, ?)",
                (round_id, canonical_json(config)),
            )

            for reviewer in config["reviewers"]:
                assign_dir = ensure_dir(round_dir / "assignments" / reviewer["id"])
                assignment_db = assign_dir / "assignment.db"
                with initialize_assignment_db(assignment_db) as assign_conn:
                    units = reviewer_assignments[reviewer["id"]]
                    for display_rank, unit in enumerate(units):
                        assign_conn.execute(
                            "INSERT OR REPLACE INTO units(unit_id, display_rank, patient_icn, doc_id, note_count) VALUES (?,?,?,?,?)",
                            (
                                unit.unit_id,
                                display_rank,
                                unit.patient_icn,
                                unit.doc_id,
                                unit.payload.get("note_count"),
                            ),
                        )
                        documents = unit.payload.get("documents", [])
                        assign_conn.execute(
                            "DELETE FROM unit_notes WHERE unit_id=?",
                            (unit.unit_id,),
                        )
                        for order_index, doc in enumerate(documents):
                            doc_id = doc.get("doc_id")
                            if not doc_id:
                                continue
                            metadata = extract_document_metadata(doc)
                            metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None
                            assign_conn.execute(
                                "INSERT OR REPLACE INTO documents(doc_id, hash, text, metadata_json) VALUES (?,?,?,?)",
                                (
                                    doc_id,
                                    doc.get("hash", ""),
                                    doc.get("text", ""),
                                    metadata_json,
                                ),
                            )
                            assign_conn.execute(
                                "INSERT OR REPLACE INTO unit_notes(unit_id, doc_id, order_index) VALUES (?,?,?)",
                                (unit.unit_id, doc_id, order_index),
                            )
                    for label in labelset["labels"]:
                        for unit in units:
                            assign_conn.execute(
                                "INSERT OR IGNORE INTO annotations(unit_id, label_id, na) VALUES (?, ?, 0)",
                                (unit.unit_id, label["label_id"]),
                            )
                (assign_dir / "client.exe").write_text("Annotator client placeholder", encoding="utf-8")
                (assign_dir / "assignment.lock").write_text("", encoding="utf-8")
                ensure_dir(assign_dir / "logs")
                project_conn.execute(
                    """
                    INSERT OR REPLACE INTO assignments(assign_id, round_id, reviewer_id, sample_size, overlap_n, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{round_id}_{reviewer['id']}",
                        round_id,
                        reviewer["id"],
                        len(reviewer_assignments[reviewer["id"]]),
                        config.get("overlap_n", 0),
                        datetime.utcnow().isoformat(),
                        "open",
                    ),
                )

            project_conn.commit()
            return {
                "round_id": round_id,
                "round_dir": str(round_dir),
                "assignment_counts": {rid: len(units) for rid, units in reviewer_assignments.items()},
            }

    def _build_candidates(self, corpus_conn: sqlite3.Connection, pheno_row: sqlite3.Row, config: dict) -> Iterator[CandidateUnit]:
        level = pheno_row["level"]
        filters_config = config.get("filters", {})
        strat_config = config.get("stratification") or {}
        metadata_filters_config = filters_config.get("metadata")
        strat_field_keys: list[str] = []
        use_metadata_sampling = isinstance(metadata_filters_config, (list, dict))
        raw_metadata_filters: list[Mapping[str, object]] = []
        metadata_logic = "all"
        if isinstance(metadata_filters_config, dict):
            metadata_logic = str(metadata_filters_config.get("logic") or "all").lower()
            raw_conditions = metadata_filters_config.get("conditions")
            if isinstance(raw_conditions, list):
                raw_metadata_filters = [
                    entry for entry in raw_conditions if isinstance(entry, Mapping)
                ]
        elif isinstance(metadata_filters_config, list):
            raw_metadata_filters = [
                entry for entry in metadata_filters_config if isinstance(entry, Mapping)
            ]
        if isinstance(strat_config, dict):
            raw_fields = strat_config.get("fields")
            if isinstance(raw_fields, list):
                strat_field_keys = [str(field) for field in raw_fields if str(field)]
                if strat_field_keys:
                    use_metadata_sampling = True
            else:
                legacy_keys = strat_config.get("keys")
                if isinstance(legacy_keys, list):
                    strat_field_keys = [str(key) for key in legacy_keys]
                elif legacy_keys:
                    strat_field_keys = [str(legacy_keys)]

        if use_metadata_sampling:
            metadata_fields: Sequence[MetadataField] = discover_corpus_metadata(corpus_conn)
            field_lookup = {field.key: field for field in metadata_fields}

            conditions: list[MetadataFilterCondition] = []
            missing_filter_fields: list[str] = []
            for entry in raw_metadata_filters:
                try:
                    condition = MetadataFilterCondition.from_payload(entry)
                except Exception:  # noqa: BLE001
                    continue
                field = field_lookup.get(condition.field)
                if not field:
                    missing_filter_fields.append(condition.field or condition.label)
                    continue
                conditions.append(condition)
            if missing_filter_fields:
                raise ValueError(
                    "Unknown metadata fields in filters: "
                    + ", ".join(sorted({field for field in missing_filter_fields if field})),
                )

            sampling_filters = SamplingFilters(
                metadata_filters=conditions,
                match_any=metadata_logic == "any",
            )

            filtered_strat_keys: list[str] = []
            strat_aliases: list[str] = []
            missing_strat_fields: list[str] = []
            for key in strat_field_keys:
                field = field_lookup.get(key)
                if not field:
                    missing_strat_fields.append(key)
                    continue
                filtered_strat_keys.append(field.key)
                strat_aliases.append(field.alias)
            if missing_strat_fields:
                raise ValueError(
                    "Unknown metadata fields requested for stratification: "
                    + ", ".join(sorted(set(missing_strat_fields))),
                )

            rows = candidate_documents(
                corpus_conn,
                level,
                sampling_filters,
                metadata_fields=metadata_fields,
                stratify_keys=filtered_strat_keys,
            )
            if level == "single_doc":
                for row in rows:
                    row_dict = dict(row)
                    strata_key = self._compute_strata_key(row_dict, strat_aliases)
                    document_metadata = extract_document_metadata(row_dict)
                    documents = [
                        {
                            "doc_id": row_dict["doc_id"],
                            "hash": row_dict.get("hash"),
                            "text": row_dict.get("text"),
                            "order_index": 0,
                            "metadata_json": row_dict.get("metadata_json"),
                            "metadata": document_metadata,
                            "date_note": row_dict.get("date_note"),
                        }
                    ]
                    payload = {
                        "date_note": row_dict.get("date_note"),
                        "hash": row_dict.get("hash"),
                        "metadata_json": row_dict.get("metadata_json"),
                        "metadata": document_metadata,
                        "strata_key": strata_key,
                        "note_count": 1,
                        "documents": documents,
                        "display_label": row_dict["doc_id"],
                    }
                    yield CandidateUnit(row_dict["doc_id"], row_dict["patient_icn"], row_dict["doc_id"], strata_key, payload)
            else:
                for entry in rows:
                    strata_key = self._compute_strata_key(entry, strat_aliases)
                    payload = dict(entry)
                    payload["strata_key"] = strata_key
                    payload.setdefault("display_label", entry.get("unit_id") or entry.get("patient_icn"))
                    yield CandidateUnit(
                        str(entry.get("unit_id") or entry.get("patient_icn")),
                        str(entry.get("patient_icn")),
                        entry.get("doc_id"),
                        strata_key,
                        payload,
                    )
            return

        params: list = []
        where_clauses = []
        regex_expr = None
        regex_flags = ""
        if "note" in filters_config:
            nf = filters_config["note"]
            regex_expr = nf.get("regex")
            regex_flags = nf.get("regex_flags", "")
            if regex_expr:
                where_clauses.append("documents.text REGEXP ?")
                params.append(regex_expr)
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        if regex_expr:
            flags = 0
            if "i" in regex_flags.lower():
                flags |= re.IGNORECASE
            pattern = re.compile(regex_expr, flags)
            corpus_conn.create_function("REGEXP", 2, lambda expr, item, compiled=pattern: 1 if compiled.search(item or "") else 0)
        else:
            corpus_conn.create_function("REGEXP", 2, lambda expr, item: 1)
        strat_keys_config = strat_config.get("keys") or []
        if isinstance(strat_keys_config, list):
            legacy_aliases = [str(key) for key in strat_keys_config]
        elif strat_keys_config:
            legacy_aliases = [str(strat_keys_config)]
        else:
            legacy_aliases = []
        cursor = corpus_conn.execute(
            f"""
            SELECT documents.doc_id, documents.patient_icn, documents.date_note,
                   documents.hash, documents.text, documents.metadata_json
            FROM documents
            WHERE {where_sql}
            """,
            params,
        )
        if level == "single_doc":
            for row in cursor:
                row_dict = dict(row)
                strata_key = self._compute_strata_key(row, legacy_aliases)
                document_metadata = extract_document_metadata(row_dict)
                documents = [
                    {
                        "doc_id": row["doc_id"],
                        "hash": row["hash"],
                        "text": row["text"],
                        "order_index": 0,
                        "metadata_json": row_dict.get("metadata_json"),
                        "metadata": document_metadata,
                        "date_note": row_dict.get("date_note"),
                    }
                ]
                payload = {
                    "date_note": row["date_note"],
                    "hash": row["hash"],
                    "metadata_json": row_dict.get("metadata_json"),
                    "metadata": document_metadata,
                    "strata_key": strata_key,
                    "note_count": 1,
                    "documents": documents,
                    "display_label": row["doc_id"],
                }
                yield CandidateUnit(row["doc_id"], row["patient_icn"], row["doc_id"], strata_key, payload)
        else:
            patient_docs: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for row in cursor:
                patient_docs[row["patient_icn"]].append(row)
            for patient_icn, docs in patient_docs.items():
                primary_row = docs[0]
                strata_key = self._compute_strata_key(primary_row, legacy_aliases)
                ordered_docs = sorted(
                    docs,
                    key=lambda item: (_date_sort_value(item["date_note"]), item["doc_id"]),
                )
                doc_payloads = []
                for idx, doc in enumerate(ordered_docs):
                    doc_dict = dict(doc)
                    metadata = extract_document_metadata(doc_dict)
                    doc_payloads.append(
                        {
                            "doc_id": doc_dict["doc_id"],
                            "hash": doc_dict.get("hash"),
                            "text": doc_dict.get("text"),
                            "order_index": idx,
                            "metadata_json": doc_dict.get("metadata_json"),
                            "metadata": metadata,
                            "date_note": doc_dict.get("date_note"),
                        }
                    )
                primary_dict = dict(primary_row)
                primary_metadata = extract_document_metadata(primary_dict)
                payload = {
                    "date_note": primary_dict.get("date_note"),
                    "metadata_json": primary_dict.get("metadata_json"),
                    "metadata": primary_metadata,
                    "strata_key": strata_key,
                    "note_count": len(docs),
                    "documents": doc_payloads,
                    "display_label": patient_icn,
                }
                yield CandidateUnit(patient_icn, patient_icn, None, strata_key, payload)

    def _build_preselected_candidates(
        self,
        corpus_conn: sqlite3.Connection,
        pheno_row: sqlite3.Row,
        csv_path: Path,
    ) -> Iterator[CandidateUnit]:
        level = str(pheno_row["level"] or "single_doc")
        rows_by_unit: dict[str, dict] = {}
        unit_order: list[str] = []
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for position, row in enumerate(reader):
                if not isinstance(row, Mapping):
                    continue
                normalized: dict[str, str] = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    normalized[key.strip().lower()] = (value or "").strip()
                raw_unit = (
                    normalized.get("unit_id")
                    or normalized.get("doc_id")
                    or normalized.get("patient_icn")
                    or normalized.get("patienticn")
                )
                if not raw_unit:
                    continue
                unit_id = str(raw_unit)
                entry = rows_by_unit.get(unit_id)
                if not entry:
                    entry = {
                        "unit_id": unit_id,
                        "patient_icn": normalized.get("patient_icn")
                        or normalized.get("patienticn"),
                        "doc_ids": [],
                        "strata_key": normalized.get("strata_key") or "ai_import",
                        "selection_reasons": [],
                        "doc_order": [],
                        "first_index": position,
                    }
                    rows_by_unit[unit_id] = entry
                    unit_order.append(unit_id)
                if not entry.get("patient_icn"):
                    patient_value = normalized.get("patient_icn") or normalized.get("patienticn")
                    if patient_value:
                        entry["patient_icn"] = patient_value
                doc_id = normalized.get("doc_id") or normalized.get("document_id")
                if doc_id:
                    doc_id = str(doc_id)
                    entry["doc_ids"].append(doc_id)
                    entry["doc_order"].append(doc_id)
                reason = normalized.get("selection_reason") or normalized.get("reason")
                if reason:
                    entry["selection_reasons"].append(reason)
                strata_value = normalized.get("strata_key")
                if strata_value:
                    entry["strata_key"] = str(strata_value)

        if not rows_by_unit:
            return

        doc_ids: list[str] = []
        for entry in rows_by_unit.values():
            doc_ids.extend(entry.get("doc_ids", []))
        doc_map = self._fetch_documents_by_ids(corpus_conn, doc_ids)

        for unit_id in unit_order:
            entry = rows_by_unit[unit_id]
            strata_key = str(entry.get("strata_key") or "ai_import")
            selection_reason = next(
                (reason for reason in entry.get("selection_reasons", []) if reason),
                None,
            )
            if level == "single_doc":
                doc_id = entry["doc_ids"][0] if entry["doc_ids"] else unit_id
                doc_row = doc_map.get(doc_id)
                if not doc_row:
                    raise ValueError(
                        f"Document {doc_id} referenced in {csv_path} was not found in the corpus"
                    )
                row_dict = dict(doc_row)
                patient_icn = str(
                    entry.get("patient_icn")
                    or row_dict.get("patient_icn")
                    or row_dict.get("patienticn")
                    or ""
                )
                metadata = extract_document_metadata(row_dict)
                documents = [
                    {
                        "doc_id": row_dict.get("doc_id"),
                        "hash": row_dict.get("hash"),
                        "text": row_dict.get("text"),
                        "order_index": 0,
                        "metadata_json": row_dict.get("metadata_json"),
                        "metadata": metadata,
                        "date_note": row_dict.get("date_note"),
                    }
                ]
                payload = {
                    "date_note": row_dict.get("date_note"),
                    "hash": row_dict.get("hash"),
                    "metadata_json": row_dict.get("metadata_json"),
                    "metadata": metadata,
                    "strata_key": strata_key,
                    "note_count": 1,
                    "documents": documents,
                    "display_label": row_dict.get("doc_id") or unit_id,
                }
                if selection_reason:
                    payload["selection_reason"] = selection_reason
                yield CandidateUnit(str(unit_id), patient_icn, row_dict.get("doc_id"), strata_key, payload)
                continue

            patient_icn = entry.get("patient_icn")
            doc_rows: list[sqlite3.Row] = []
            if entry["doc_ids"]:
                for doc_id in entry["doc_ids"]:
                    doc_row = doc_map.get(doc_id)
                    if not doc_row:
                        raise ValueError(
                            f"Document {doc_id} referenced in {csv_path} was not found in the corpus"
                        )
                    doc_rows.append(doc_row)
            else:
                if not patient_icn:
                    raise ValueError(
                        f"Preselected unit {unit_id} is missing patient identifier and document IDs"
                    )
                doc_rows = self._fetch_patient_documents(corpus_conn, str(patient_icn))
                if not doc_rows:
                    raise ValueError(
                        f"No documents found for patient {patient_icn} referenced by {unit_id}"
                    )

            ordered = []
            for index, doc_row in enumerate(doc_rows):
                doc_dict = dict(doc_row)
                metadata = extract_document_metadata(doc_dict)
                ordered.append(
                    {
                        "doc_id": doc_dict.get("doc_id"),
                        "hash": doc_dict.get("hash"),
                        "text": doc_dict.get("text"),
                        "order_index": index,
                        "metadata_json": doc_dict.get("metadata_json"),
                        "metadata": metadata,
                        "date_note": doc_dict.get("date_note"),
                    }
                )
                if not patient_icn and doc_dict.get("patient_icn"):
                    patient_icn = doc_dict.get("patient_icn")
            if not patient_icn:
                raise ValueError(
                    f"Could not determine patient identifier for preselected unit {unit_id}"
                )
            payload = {
                "metadata_json": ordered[0].get("metadata_json") if ordered else None,
                "metadata": ordered[0].get("metadata") if ordered else None,
                "strata_key": strata_key,
                "note_count": len(ordered),
                "documents": ordered,
                "display_label": unit_id,
            }
            if selection_reason:
                payload["selection_reason"] = selection_reason
            yield CandidateUnit(str(unit_id), str(patient_icn), None, strata_key, payload)

    def _fetch_documents_by_ids(
        self, corpus_conn: sqlite3.Connection, doc_ids: Sequence[str]
    ) -> dict[str, sqlite3.Row]:
        unique_ids = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id]
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        cursor = corpus_conn.execute(
            f"""
            SELECT doc_id, patient_icn, date_note, hash, text, metadata_json
            FROM documents
            WHERE doc_id IN ({placeholders})
            """,
            unique_ids,
        )
        return {str(row["doc_id"]): row for row in cursor.fetchall()}

    def _fetch_patient_documents(
        self, corpus_conn: sqlite3.Connection, patient_icn: str
    ) -> list[sqlite3.Row]:
        cursor = corpus_conn.execute(
            """
            SELECT doc_id, patient_icn, date_note, hash, text, metadata_json
            FROM documents
            WHERE patient_icn=?
            """,
            (patient_icn,),
        )
        rows = cursor.fetchall()
        return sorted(
            rows,
            key=lambda item: (_date_sort_value(item["date_note"]), item["doc_id"]),
        )

    def _compute_strata_key(self, row: sqlite3.Row | dict, aliases: Sequence[str]) -> str:
        if not aliases:
            return "default"
        keys: list[str] = []
        for alias in aliases:
            value: object | None
            if isinstance(row, dict):
                value = row.get(alias)
                if value is None and alias in row:
                    value = row[alias]
            else:
                try:
                    value = row[alias]
                except Exception:
                    value = None
            keys.append("" if value is None else str(value))
        return "|".join(keys) if keys else "default"

    def _allocate_units(
        self,
        candidates: list[CandidateUnit],
        reviewers: list[str],
        rng_seed: int,
        overlap_n: int,
        total_n: int | None,
        strat_config: dict | None,
        preserve_input_order: bool = False,
    ) -> dict[str, list[AssignmentUnit]]:
        by_strata: dict[str, list[CandidateUnit]] = defaultdict(list)
        strata_order: list[str] = []
        for candidate in candidates:
            by_strata[candidate.strata_key].append(candidate)
            if preserve_input_order and candidate.strata_key not in strata_order:
                strata_order.append(candidate.strata_key)
        assignments: dict[str, list[AssignmentUnit]] = {rid: [] for rid in reviewers}
        stratified = self._stratification_requested(strat_config)
        if stratified and total_n is not None and len(by_strata) > total_n:
            raise ValueError(
                "The requested sample size is smaller than the number of strata. "
                "Increase the total units or adjust the stratification keys."
            )
        total_available = sum(len(items) for items in by_strata.values())
        target_total = total_available if total_n is None else min(total_n, total_available)
        if total_n is None:
            allocations = {key: len(items) for key, items in by_strata.items()}
        else:
            allocations = {key: 0 for key in by_strata}
            remaining = target_total
            active = [key for key, items in by_strata.items() if items]
            while remaining > 0 and active:
                share = max(1, remaining // len(active))
                next_active: list[str] = []
                for strata_key in list(active):
                    capacity = len(by_strata[strata_key]) - allocations[strata_key]
                    if capacity <= 0:
                        continue
                    take = min(share, capacity, remaining)
                    allocations[strata_key] += take
                    remaining -= take
                    if allocations[strata_key] < len(by_strata[strata_key]):
                        next_active.append(strata_key)
                    if remaining == 0:
                        break
                active = next_active if remaining > 0 else []
        assigned_total = 0
        selected_by_strata: dict[str, list[CandidateUnit]] = {}
        strata_keys = strata_order if preserve_input_order and strata_order else sorted(by_strata.keys())
        for strata_key in strata_keys:
            strata_candidates = by_strata[strata_key]
            if preserve_input_order:
                shuffled = list(strata_candidates)
            else:
                shuffled = deterministic_choice(strata_candidates, stable_hash(rng_seed, strata_key))
            take = allocations.get(strata_key, len(strata_candidates))
            if total_n is not None and assigned_total + take > target_total:
                take = max(0, target_total - assigned_total)
            selected = list(shuffled[:take])
            assigned_total += len(selected)
            selected_by_strata[strata_key] = selected

        overlap_allocations: dict[str, int] = {key: 0 for key in selected_by_strata}
        if overlap_n:
            remaining_overlap = min(
                overlap_n, sum(len(items) for items in selected_by_strata.values())
            )
            active = [key for key, items in selected_by_strata.items() if items]
            while remaining_overlap > 0 and active:
                share = max(1, remaining_overlap // len(active))
                next_active: list[str] = []
                for strata_key in list(active):
                    capacity = len(selected_by_strata[strata_key]) - overlap_allocations[strata_key]
                    if capacity <= 0:
                        continue
                    take = min(share, capacity, remaining_overlap)
                    overlap_allocations[strata_key] += take
                    remaining_overlap -= take
                    if (
                        overlap_allocations[strata_key] < len(selected_by_strata[strata_key])
                        and remaining_overlap > 0
                    ):
                        next_active.append(strata_key)
                    if remaining_overlap == 0:
                        break
                active = next_active if remaining_overlap > 0 else []

        for strata_key in sorted(selected_by_strata.keys()):
            selected = selected_by_strata.get(strata_key, [])
            overlap_count = overlap_allocations.get(strata_key, 0)
            overlap_units = selected[:overlap_count]
            remainder = selected[overlap_count:]
            per_reviewer = self._split_among_reviewers(remainder, reviewers, rng_seed, strata_key)
            for reviewer_id in reviewers:
                combined = []
                for unit in overlap_units:
                    combined.append(
                        AssignmentUnit(
                            unit.unit_id,
                            unit.patient_icn,
                            unit.doc_id,
                            {**unit.payload, "is_overlap": True},
                        )
                    )
                combined.extend(per_reviewer.get(reviewer_id, []))
                assignments[reviewer_id].extend(combined)
        for reviewer_id, units in assignments.items():
            seed = stable_hash(rng_seed, reviewer_id)
            shuffled = deterministic_choice(units, stable_hash(seed, "display"))
            assignments[reviewer_id] = [
                AssignmentUnit(unit.unit_id, unit.patient_icn, unit.doc_id, unit.payload)
                for unit in shuffled
            ]
        return assignments

    def _split_among_reviewers(
        self,
        units: list[CandidateUnit],
        reviewers: list[str],
        rng_seed: int,
        strata_key: str,
    ) -> dict[str, list[AssignmentUnit]]:
        shuffled = deterministic_choice(units, stable_hash(rng_seed, strata_key, "split"))
        buckets: dict[str, list[AssignmentUnit]] = {rid: [] for rid in reviewers}
        for idx, unit in enumerate(shuffled):
            reviewer_id = reviewers[idx % len(reviewers)]
            buckets[reviewer_id].append(
                AssignmentUnit(
                    unit.unit_id,
                    unit.patient_icn,
                    unit.doc_id,
                    {**unit.payload, "is_overlap": False},
                )
            )
        return buckets

    def _stratification_requested(self, strat_config: dict | None) -> bool:
        if not strat_config or not isinstance(strat_config, dict):
            return False
        fields = strat_config.get("fields")
        if isinstance(fields, list) and fields:
            return True
        keys = strat_config.get("keys")
        if isinstance(keys, list):
            return bool(keys)
        return bool(keys)

    # Aggregation & import helpers
    def import_assignment(self, pheno_id: str, round_number: int, reviewer_id: str) -> Path:
        with self._connect_project() as conn:
            pheno = conn.execute(
                "SELECT storage_path FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
            if not pheno:
                raise ValueError(f"Phenotype {pheno_id} not found")
            storage_path = Path(pheno["storage_path"])
            if not storage_path.is_absolute():
                storage_path = (self.project_root / storage_path).resolve()
            phenotype_dir = storage_path
            round_dir = phenotype_dir / "rounds" / f"round_{round_number}"
            assignment_dir = round_dir / "assignments" / reviewer_id
            assignment_db = assignment_dir / "assignment.db"
            imports_dir = ensure_dir(round_dir / "imports")
            target_path = imports_dir / f"{reviewer_id}_assignment.db"
            copy_sqlite_database(assignment_db, target_path)
            round_id = conn.execute(
                "SELECT round_id FROM rounds WHERE pheno_id=? AND round_number=?",
                (pheno_id, round_number),
            ).fetchone()
            if not round_id:
                raise ValueError("Round metadata missing")
            conn.execute(
                "UPDATE assignments SET status='imported' WHERE round_id=? AND reviewer_id=?",
                (round_id["round_id"], reviewer_id),
            )
            conn.commit()
        return target_path

    def build_round_aggregate(self, pheno_id: str, round_number: int) -> Path:
        with self._connect_project() as conn:
            pheno = conn.execute(
                "SELECT storage_path FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
            if not pheno:
                raise ValueError(f"Phenotype {pheno_id} not found")
            storage_path = Path(pheno["storage_path"])
            if not storage_path.is_absolute():
                storage_path = (self.project_root / storage_path).resolve()
            phenotype_dir = storage_path
        round_dir = phenotype_dir / "rounds" / f"round_{round_number}"
        imports_dir = ensure_dir(round_dir / "imports")
        aggregate_db = round_dir / "round_aggregate.db"
        with initialize_round_aggregate_db(aggregate_db) as agg_conn:
            agg_conn.execute("DELETE FROM unit_annotations")
            agg_conn.execute("DELETE FROM unit_summary")
            for assignment_path in imports_dir.glob("*_assignment.db"):
                reviewer_id = assignment_path.stem
                if reviewer_id.endswith("_assignment"):
                    reviewer_id = reviewer_id[: -len("_assignment")]
                with sqlite3.connect(assignment_path) as assign_conn:
                    assign_conn.row_factory = sqlite3.Row
                    for unit_row in assign_conn.execute("SELECT * FROM units"):
                        agg_conn.execute(
                            "INSERT OR IGNORE INTO unit_summary(round_id, unit_id, patient_icn, doc_id) VALUES (?,?,?,?)",
                            (
                                f"{pheno_id}_r{round_number}",
                                unit_row["unit_id"],
                                unit_row["patient_icn"],
                                unit_row["doc_id"],
                            ),
                        )
                    for ann_row in assign_conn.execute("SELECT * FROM annotations"):
                        agg_conn.execute(
                            """
                            INSERT INTO unit_annotations(round_id, unit_id, reviewer_id, label_id, value, value_num, value_date, na, notes)
                            VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                f"{pheno_id}_r{round_number}",
                                ann_row["unit_id"],
                                reviewer_id,
                                ann_row["label_id"],
                                ann_row["value"],
                                ann_row["value_num"],
                                ann_row["value_date"],
                                ann_row["na"],
                                ann_row["notes"],
                            ),
                        )
            agg_conn.commit()
        return aggregate_db
