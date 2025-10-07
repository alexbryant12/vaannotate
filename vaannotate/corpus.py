"""Corpus ingestion and querying."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd

from .schema import initialize_corpus_db
from .utils import ensure_dir


def normalize_text(text: str) -> str:
    """Canonicalize note text for hashing."""
    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return normalized


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_patients(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    prepared = []
    for row in rows:
        prepared.append(
            {
                "patient_icn": row["patient_icn"],
                "sta3n": row.get("sta3n"),
                "date_index": row.get("date_index"),
                "softlabel": row.get("softlabel"),
            }
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO patients(patient_icn, sta3n, date_index, softlabel)
        VALUES (:patient_icn, :sta3n, :date_index, :softlabel)
        """,
        prepared,
    )


def import_documents(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    prepared = []
    for row in rows:
        text = normalize_text(row["text"])
        note_year = row.get("note_year")
        if note_year is not None:
            try:
                note_year = int(note_year)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid note_year value: {note_year!r}") from None
        raw_metadata = row.get("metadata")
        if not raw_metadata and "__metadata__" in row:
            raw_metadata = row.get("__metadata__")
        metadata: dict[str, object] = {}
        if isinstance(raw_metadata, dict):
            metadata = {key: value for key, value in raw_metadata.items() if value not in (None, "")}
        metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None
        prepared.append(
            {
                "doc_id": row["doc_id"],
                "patient_icn": row["patient_icn"],
                "notetype": row.get("notetype"),
                "note_year": note_year,
                "date_note": row.get("date_note"),
                "cptname": row.get("cptname"),
                "sta3n": row.get("sta3n"),
                "hash": hash_text(text),
                "text": text,
                "metadata_json": metadata_json,
            }
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO documents(doc_id, patient_icn, notetype, note_year, date_note, cptname, sta3n, hash, text, metadata_json)
        VALUES (:doc_id,:patient_icn,:notetype,:note_year,:date_note,:cptname,:sta3n,:hash,:text,:metadata_json)
        """,
        prepared,
    )


def load_patients_csv(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {
                "patient_icn": row["patient_icn"],
                "sta3n": row.get("sta3n"),
                "date_index": row.get("date_index"),
                "softlabel": float(row["softlabel"]) if row.get("softlabel") else None,
            }


def load_documents_csv(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return
        rename: dict[str, str] = {}
        for column in reader.fieldnames:
            canonical = _canonical_column(str(column))
            if canonical:
                rename[column] = canonical
        recognized = REQUIRED_COLUMNS | OPTIONAL_COLUMNS | {"doc_id"}
        for row in reader:
            normalized: dict[str, object] = {}
            metadata: dict[str, object] = {}
            for column, value in row.items():
                if column is None:
                    continue
                key = rename.get(column, column)
                cell = value
                if isinstance(cell, str):
                    if key == "text":
                        cell = cell
                    else:
                        cell = cell.strip() or None
                if key in recognized:
                    normalized[key] = cell
                else:
                    metadata[key] = cell
            if "patient_icn" not in normalized:
                raise ValueError("Corpus is missing required column 'patienticn'")
            if "text" not in normalized:
                raise ValueError("Corpus is missing required column 'text'")
            if metadata:
                cleaned_metadata = {key: val for key, val in metadata.items() if val not in (None, "")}
                if cleaned_metadata:
                    normalized["__metadata__"] = cleaned_metadata
            yield normalized


def bulk_import_from_csv(corpus_db: Path, patients_csv: Path, documents_csv: Path) -> None:
    ensure_dir(corpus_db.parent)
    with initialize_corpus_db(corpus_db) as conn:
        import_patients(conn, load_patients_csv(patients_csv))
        import_documents(conn, load_documents_csv(documents_csv))
        conn.commit()


TABULAR_EXTENSIONS = {".csv", ".parquet", ".pq"}

REQUIRED_COLUMNS = {"patient_icn", "text"}

OPTIONAL_COLUMNS = {
    "sta3n",
    "date_index",
    "notetype",
    "note_year",
    "date_note",
    "cptname",
    "doc_id",
    "softlabel",
}

_COLUMN_ALIAS_MAP = {
    "patienticn": "patient_icn",
    "patient_icn": "patient_icn",
    "text": "text",
    "sta3n": "sta3n",
    "dateindex": "date_index",
    "notetype": "notetype",
    "noteyear": "note_year",
    "datenote": "date_note",
    "cptname": "cptname",
    "docid": "doc_id",
    "softlabel": "softlabel",
}


def _canonical_column(column: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", column.lower())
    return _COLUMN_ALIAS_MAP.get(normalized)


def load_tabular_corpus(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported corpus format: {path.suffix}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Corpus data could not be loaded into a table")
    rename: dict[str, str] = {}
    for column in df.columns:
        canonical = _canonical_column(str(column))
        if canonical:
            rename[column] = canonical
    df = df.rename(columns=rename)
    df = df.where(pd.notnull(df), None)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {', '.join(sorted(missing))}")
    recognized = REQUIRED_COLUMNS | OPTIONAL_COLUMNS | {"doc_id"}
    rows: List[dict] = []
    for record in df.to_dict(orient="records"):
        normalized: dict[str, object] = {}
        metadata: dict[str, object] = {}
        for column, value in record.items():
            if column is None:
                continue
            cell = value
            if isinstance(cell, str):
                if column == "text":
                    cell = cell
                else:
                    cell = cell.strip() or None
            if column in recognized:
                normalized[column] = cell
            else:
                metadata[column] = cell
        if metadata:
            cleaned_metadata = {key: val for key, val in metadata.items() if val not in (None, "")}
            if cleaned_metadata:
                normalized["__metadata__"] = cleaned_metadata
        rows.append(normalized)
    return rows


def _coerce_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value: {value!r}") from None


def _infer_note_year(note_year: object | None, date_note: str | None) -> int | None:
    if note_year in (None, ""):
        if date_note:
            match = re.search(r"(\d{4})", date_note)
            if match:
                return int(match.group(1))
        return None
    try:
        return int(note_year)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid note_year value: {note_year!r}") from None


def import_tabular_corpus(source: Path, corpus_db: Path) -> None:
    rows = load_tabular_corpus(source)
    if not rows:
        raise ValueError("Corpus file is empty")
    ensure_dir(corpus_db.parent)
    patient_map: dict[str, dict] = {}
    doc_counter: dict[str, int] = {}
    documents: List[dict] = []
    for index, row in enumerate(rows, start=1):
        patient_icn_raw = row.get("patient_icn")
        patient_icn = _coerce_str(patient_icn_raw)
        if not patient_icn:
            raise ValueError(f"Row {index} is missing a patienticn value")
        metadata = row.get("__metadata__") or {}
        text_value = row.get("text")
        if text_value is None or (isinstance(text_value, str) and text_value.strip() == ""):
            raise ValueError(f"Row {index} is missing note text")
        doc_id = _coerce_str(row.get("doc_id"))
        if not doc_id:
            seq = doc_counter.get(patient_icn, 0) + 1
            doc_counter[patient_icn] = seq
            doc_id = f"{patient_icn}_{seq:05d}"
        sta3n = _coerce_str(row.get("sta3n"))
        date_note = _coerce_str(row.get("date_note"))
        note_year = _infer_note_year(row.get("note_year"), date_note)
        patient_entry = patient_map.setdefault(
            patient_icn,
            {
                "patient_icn": patient_icn,
                "sta3n": sta3n,
                "date_index": _coerce_str(row.get("date_index")),
                "softlabel": _coerce_float(row.get("softlabel")),
            },
        )
        if sta3n and not patient_entry.get("sta3n"):
            patient_entry["sta3n"] = sta3n
        if row.get("date_index") and not patient_entry.get("date_index"):
            patient_entry["date_index"] = _coerce_str(row.get("date_index"))
        if row.get("softlabel") is not None:
            patient_entry["softlabel"] = _coerce_float(row.get("softlabel"))
        documents.append(
            {
                "doc_id": doc_id,
                "patient_icn": patient_icn,
                "notetype": _coerce_str(row.get("notetype")),
                "note_year": note_year,
                "date_note": date_note,
                "cptname": _coerce_str(row.get("cptname")),
                "sta3n": sta3n or patient_entry.get("sta3n"),
                "text": str(text_value),
                "metadata": metadata,
            }
        )
    with initialize_corpus_db(corpus_db) as conn:
        import_patients(conn, patient_map.values())
        import_documents(conn, documents)
        conn.commit()
