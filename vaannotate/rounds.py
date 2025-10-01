"""Round generation and manifest management."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .project import (
    create_assignment_record,
    init_assignment_db,
    ProjectPaths,
    record_round,
    register_reviewers,
    save_round_config,
)
from .utils import deterministic_seed, ensure_dir, shuffled, text_hash


@dataclass
class CandidateUnit:
    patient_icn: str
    doc_id: str | None
    strata_key: str
    payload: Mapping[str, object]


class CorpusView:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    def iter_documents(self, where: str = "", params: Sequence[object] | None = None) -> Iterator[sqlite3.Row]:
        query = "SELECT * FROM documents"
        if where:
            query += " WHERE " + where
        for row in self.conn.execute(query, params or []):
            yield row

    def iter_patients(self, where: str = "", params: Sequence[object] | None = None) -> Iterator[sqlite3.Row]:
        query = "SELECT * FROM patients"
        if where:
            query += " WHERE " + where
        for row in self.conn.execute(query, params or []):
            yield row


# -- filtering -----------------------------------------------------------------------


def apply_regex_filter(rows: Iterable[sqlite3.Row], pattern: str, flags: str | None) -> Iterator[sqlite3.Row]:
    compiled = re.compile(pattern, flags=re.IGNORECASE if flags and "i" in flags else 0)
    for row in rows:
        if compiled.search(row["text"]):
            yield row


def candidate_documents(conn: sqlite3.Connection, config: Mapping[str, object]) -> list[sqlite3.Row]:
    view = CorpusView(conn)
    filters = config.get("filters", {})
    patient_filters = filters.get("patient", {})
    note_filters = filters.get("note", {})

    where_clauses = []
    params: list[object] = []
    if note_filters.get("notetype_in"):
        placeholders = ",".join(["?"] * len(note_filters["notetype_in"]))
        where_clauses.append(f"notetype IN ({placeholders})")
        params.extend(note_filters["notetype_in"])
    year_range = note_filters.get("note_year_range") or note_filters.get("year_range")
    if year_range:
        start, end = year_range
        where_clauses.append("note_year BETWEEN ? AND ?")
        params.extend([start, end])
    if patient_filters.get("sta3n_in"):
        placeholders = ",".join(["?"] * len(patient_filters["sta3n_in"]))
        where_clauses.append(f"sta3n IN ({placeholders})")
        params.extend(patient_filters["sta3n_in"])

    doc_rows = list(view.iter_documents(" AND ".join(where_clauses), params))
    if note_filters.get("regex"):
        doc_rows = list(apply_regex_filter(doc_rows, note_filters["regex"], note_filters.get("regex_flags")))

    if patient_filters:
        logic = (patient_filters.get("logic") or "AND").upper()
        candidate_sets: list[set[str]] = []

        if patient_filters.get("year_range"):
            start, end = patient_filters["year_range"]
            where = "CAST(strftime('%Y', date_index) AS INTEGER) BETWEEN ? AND ?"
            candidate_sets.append(
                {
                    row["patient_icn"]
                    for row in view.iter_patients(where, [start, end])
                }
            )
        if patient_filters.get("sta3n_in"):
            placeholders = ",".join(["?"] * len(patient_filters["sta3n_in"]))
            where = f"sta3n IN ({placeholders})"
            candidate_sets.append(
                {
                    row["patient_icn"]
                    for row in view.iter_patients(where, patient_filters["sta3n_in"])
                }
            )
        if patient_filters.get("softlabel_gte") is not None:
            value = float(patient_filters["softlabel_gte"])
            candidate_sets.append(
                {
                    row["patient_icn"]
                    for row in view.iter_patients("softlabel >= ?", [value])
                }
            )

        if candidate_sets:
            if logic == "OR":
                allowed_patients = set().union(*candidate_sets)
            else:
                allowed_patients = set(candidate_sets[0])
                for extra in candidate_sets[1:]:
                    allowed_patients &= extra
            doc_rows = [row for row in doc_rows if row["patient_icn"] in allowed_patients]

    return doc_rows


# -- manifest creation ---------------------------------------------------------------


def build_single_doc_units(conn: sqlite3.Connection, config: Mapping[str, object], *, strat_keys: Sequence[str]) -> list[CandidateUnit]:
    docs = candidate_documents(conn, config)
    units: list[CandidateUnit] = []
    for row in docs:
        strata_values = [str(row[key]) for key in strat_keys] if strat_keys else ["__default__"]
        strata_key = "|".join(strata_values)
        units.append(
            CandidateUnit(
                patient_icn=row["patient_icn"],
                doc_id=row["doc_id"],
                strata_key=strata_key,
                payload={"doc_id": row["doc_id"], "patient_icn": row["patient_icn"], "hash": row["hash"], "sta3n": row["sta3n"], "note_year": row["note_year"]},
            )
        )
    return units


def build_multi_doc_units(conn: sqlite3.Connection, config: Mapping[str, object], *, strat_keys: Sequence[str]) -> list[CandidateUnit]:
    view = CorpusView(conn)
    filters = config.get("filters", {})
    patient_filters = filters.get("patient", {})
    note_filters = filters.get("note", {})

    patient_where = []
    patient_params: list[object] = []
    if patient_filters.get("sta3n_in"):
        placeholders = ",".join(["?"] * len(patient_filters["sta3n_in"]))
        patient_where.append(f"sta3n IN ({placeholders})")
        patient_params.extend(patient_filters["sta3n_in"])
    if patient_filters.get("softlabel_gte") is not None:
        patient_where.append("softlabel >= ?")
        patient_params.append(patient_filters["softlabel_gte"])

    patients = list(view.iter_patients(" AND ".join(patient_where), patient_params))
    units: list[CandidateUnit] = []

    for patient in patients:
        docs = list(view.iter_documents("patient_icn = ?", [patient["patient_icn"]]))
        docs = apply_note_filters(docs, note_filters, patient)
        if not docs:
            continue
        strata_values = [str(patient[key]) for key in strat_keys] if strat_keys else ["__default__"]
        strata_key = "|".join(strata_values)
        units.append(
            CandidateUnit(
                patient_icn=patient["patient_icn"],
                doc_id=None,
                strata_key=strata_key,
                payload={
                    "patient_icn": patient["patient_icn"],
                    "note_count": len(docs),
                    "doc_ids": [doc["doc_id"] for doc in docs],
                },
            )
        )
    return units


def apply_note_filters(docs: Iterable[sqlite3.Row], note_filters: Mapping[str, object], patient: sqlite3.Row | None = None) -> list[sqlite3.Row]:
    filtered = list(docs)
    if note_filters.get("notetype_in"):
        allowed = set(note_filters["notetype_in"])
        filtered = [doc for doc in filtered if doc["notetype"] in allowed]
    year_range = note_filters.get("note_year_range") or note_filters.get("year_range")
    if year_range:
        start, end = year_range
        filtered = [doc for doc in filtered if start <= doc["note_year"] <= end]
    if patient is not None and note_filters.get("window_days_from_index") and patient["date_index"]:
        start_days, end_days = note_filters["window_days_from_index"]
        target = patient["date_index"]
        filtered = [doc for doc in filtered if within_window(target, doc["date_note"], start_days, end_days)]
    if note_filters.get("regex"):
        filtered = list(apply_regex_filter(filtered, note_filters["regex"], note_filters.get("regex_flags")))
    return filtered


def within_window(index_date: str, note_date: str, start_days: int, end_days: int) -> bool:
    from datetime import datetime

    if not index_date:
        return True
    idx = datetime.fromisoformat(index_date)
    note = datetime.fromisoformat(note_date)
    delta = (note - idx).days
    return start_days <= delta <= end_days


# -- assignment creation -------------------------------------------------------------


def distribute_units(
    units: Sequence[CandidateUnit],
    reviewers: Sequence[Mapping[str, object]],
    *,
    overlap_n: int,
    rng_seed: int,
    sample_per_stratum: int | None = None,
    total_sample: int | None = None,
) -> dict[str, list[CandidateUnit]]:
    if not reviewers:
        raise ValueError("No reviewers configured")

    reviewer_ids = [reviewer["id"] for reviewer in reviewers]
    grouped: dict[str, list[CandidateUnit]] = {rid: [] for rid in reviewer_ids}

    per_stratum: dict[str, list[CandidateUnit]] = defaultdict(list)
    for unit in units:
        per_stratum[unit.strata_key].append(unit)

    # deterministic shuffle within each stratum
    for strata_key in list(per_stratum.keys()):
        seed = deterministic_seed("stratum", strata_key, base_seed=rng_seed)
        ordered = shuffled(per_stratum[strata_key], seed)
        if sample_per_stratum:
            ordered = ordered[:sample_per_stratum]
        per_stratum[strata_key] = ordered

    if total_sample is not None:
        ordered_keys = sorted(per_stratum)
        flattened = [unit for key in ordered_keys for unit in per_stratum[key]]
        if len(flattened) > total_sample:
            selection_seed = deterministic_seed("total_sample", base_seed=rng_seed)
            selected_keys = {
                make_unit_id(unit)
                for unit in shuffled(flattened, selection_seed)[:total_sample]
            }
            trimmed: dict[str, list[CandidateUnit]] = defaultdict(list)
            for key in ordered_keys:
                trimmed[key] = [
                    unit for unit in per_stratum[key] if make_unit_id(unit) in selected_keys
                ]
            per_stratum = trimmed

    overlap_remaining = max(overlap_n, 0)

    for strata_key in sorted(per_stratum):
        strata_units = per_stratum[strata_key]
        if not strata_units:
            continue
        overlap_take = min(overlap_remaining, len(strata_units))
        overlap_units = strata_units[:overlap_take]
        remainder = strata_units[overlap_take:]

        for rid in reviewer_ids:
            grouped[rid].extend(overlap_units)
        overlap_remaining -= overlap_take

        chunked = list(chunks_round_robin(remainder, reviewer_ids))
        for rid, chunk in zip(reviewer_ids, chunked):
            grouped[rid].extend(chunk)

    if overlap_remaining > 0:
        raise ValueError("Not enough units to satisfy overlap requirement")

    # randomize display order per reviewer
    for rid in reviewer_ids:
        seed = deterministic_seed(rid, base_seed=rng_seed)
        grouped[rid] = shuffled(grouped[rid], seed)
    return grouped


def chunks_round_robin(units: Sequence[CandidateUnit], reviewer_ids: Sequence[str]) -> Iterator[list[CandidateUnit]]:
    if not reviewer_ids:
        return iter([])  # type: ignore[return-value]
    buckets = {rid: [] for rid in reviewer_ids}
    for index, unit in enumerate(units):
        rid = reviewer_ids[index % len(reviewer_ids)]
        buckets[rid].append(unit)
    return buckets.values()


# -- public API ----------------------------------------------------------------------


def generate_round(paths: ProjectPaths, *, config_path: Path, pheno_id: str, round_number: int, created_by: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    rng_seed = config.get("rng_seed")
    if rng_seed is None:
        raise ValueError("Round config missing rng_seed")
    labelset_id = config.get("labelset_id") or config.get("labelset")
    if not labelset_id:
        raise ValueError("Round config must specify labelset_id")
    reviewers = config.get("reviewers", [])
    if not reviewers:
        raise ValueError("Round config must include reviewers")

    register_reviewers(paths, reviewers)

    ensure_dir(paths.phenotypes / pheno_id / "rounds")
    round_dir = paths.round_dir(pheno_id, round_number)
    save_round_config(round_dir, config)
    ensure_dir(round_dir / "imports")
    reports_dir = ensure_dir(round_dir / "reports")
    ensure_dir(reports_dir / "confusion_matrices")
    ensure_dir(reports_dir / "exports")

    with sqlite3.connect(paths.corpus_db) as conn:
        conn.row_factory = sqlite3.Row
        level = config.get("level", "single_doc")
        strat_keys = config.get("stratification", {}).get("keys", [])
        if level == "single_doc":
            units = build_single_doc_units(conn, config, strat_keys=strat_keys)
        else:
            units = build_multi_doc_units(conn, config, strat_keys=strat_keys)

    strat_config = config.get("stratification", {})
    sample_per_stratum = strat_config.get("sample_per_stratum")
    if sample_per_stratum is not None:
        sample_per_stratum = int(sample_per_stratum)
    total_sample = config.get("sample_size")
    if total_sample is not None:
        total_sample = int(total_sample)

    assignments = distribute_units(
        units,
        reviewers,
        overlap_n=int(config.get("overlap_n", 0)),
        rng_seed=rng_seed,
        sample_per_stratum=sample_per_stratum,
        total_sample=total_sample,
    )

    round_id = f"{pheno_id}_round_{round_number}"
    record_round(
        paths,
        config,
        round_id=round_id,
        pheno_id=pheno_id,
        round_number=round_number,
        labelset_id=labelset_id,
        rng_seed=rng_seed,
    )

    manifest_path = round_dir / "manifest.csv"
    write_manifest(manifest_path, assignments, level)

    assignments_dir = ensure_dir(round_dir / "assignments")
    for reviewer in reviewers:
        rid = reviewer["id"]
        reviewer_dir = ensure_dir(assignments_dir / rid)
        assignment_db = reviewer_dir / "assignment.db"
        init_assignment_db(assignment_db)
        populate_assignment_db(assignment_db, assignments[rid], level)
        ensure_dir(reviewer_dir / "logs")
        lock_path = reviewer_dir / "assignment.lock"
        if not lock_path.exists():
            lock_path.write_text("placeholder lock file", encoding="utf-8")
        placeholder = reviewer_dir / "CLIENT_PLACEHOLDER.txt"
        if not placeholder.exists():
            placeholder.write_text(
                "Place the portable annotator client executable in this directory.",
                encoding="utf-8",
            )
        create_assignment_record(
            paths,
            assign_id=f"{round_id}_{rid}",
            round_id=round_id,
            reviewer_id=rid,
            sample_size=len(assignments[rid]),
            overlap_n=config.get("overlap_n", 0),
        )


def write_manifest(path: Path, assignments: Mapping[str, Sequence[CandidateUnit]], level: str) -> None:
    ensure_dir(path.parent)
    fieldnames = ["assigned_to", "is_overlap", "strata_key", "patient_icn"]
    if level == "single_doc":
        fieldnames.append("doc_id")
    else:
        fieldnames.append("note_count")
    counts: dict[str, int] = defaultdict(int)
    for units in assignments.values():
        for unit in units:
            counts[make_unit_id(unit)] += 1
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for reviewer_id, units in assignments.items():
            for unit in units:
                unit_id = make_unit_id(unit)
                payload = {
                    "assigned_to": reviewer_id,
                    "strata_key": unit.strata_key,
                    "patient_icn": unit.patient_icn,
                }
                payload["is_overlap"] = 1 if counts[unit_id] > 1 else 0
                if level == "single_doc":
                    payload["doc_id"] = unit.doc_id
                else:
                    payload["note_count"] = unit.payload.get("note_count")
                writer.writerow(payload)


def populate_assignment_db(path: Path, units: Sequence[CandidateUnit], level: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM units")
        conn.execute("DELETE FROM unit_notes")
        for display_rank, unit in enumerate(units, start=1):
            conn.execute(
                """
                INSERT INTO units(unit_id, display_rank, patient_icn, doc_id, note_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    make_unit_id(unit),
                    display_rank,
                    unit.patient_icn,
                    unit.doc_id,
                    unit.payload.get("note_count"),
                ),
            )
            if level != "single_doc":
                for order_index, doc_id in enumerate(unit.payload.get("doc_ids", []), start=1):
                    conn.execute(
                        "INSERT INTO unit_notes(unit_id, doc_id, order_index) VALUES (?, ?, ?)",
                        (make_unit_id(unit), doc_id, order_index),
                    )


def make_unit_id(unit: CandidateUnit) -> str:
    base = unit.doc_id or unit.patient_icn
    digest = text_hash(json.dumps(unit.payload, sort_keys=True))[:8]
    return f"{base}_{digest}"

