"""Corpus ingestion and querying."""
from __future__ import annotations

import csv
import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

from .schema import initialize_corpus_db
from .utils import ensure_dir


def normalize_text(text: str) -> str:
    """Canonicalize note text for hashing."""
    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return normalized


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_patients(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO patients(patient_icn, sta3n, date_index, softlabel)
        VALUES (:patient_icn, :sta3n, :date_index, :softlabel)
        """,
        rows,
    )


def import_documents(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    prepared = []
    for row in rows:
        text = normalize_text(row["text"])
        prepared.append(
            {
                "doc_id": row["doc_id"],
                "patient_icn": row["patient_icn"],
                "notetype": row["notetype"],
                "note_year": int(row["note_year"]),
                "date_note": row.get("date_note"),
                "cptname": row.get("cptname"),
                "sta3n": row["sta3n"],
                "hash": hash_text(text),
                "text": text,
            }
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO documents(doc_id, patient_icn, notetype, note_year, date_note, cptname, sta3n, hash, text)
        VALUES (:doc_id,:patient_icn,:notetype,:note_year,:date_note,:cptname,:sta3n,:hash,:text)
        """,
        prepared,
    )


def load_patients_csv(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {
                "patient_icn": row["patient_icn"],
                "sta3n": row["sta3n"],
                "date_index": row.get("date_index"),
                "softlabel": float(row["softlabel"]) if row.get("softlabel") else None,
            }


def load_documents_csv(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {
                "doc_id": row["doc_id"],
                "patient_icn": row["patient_icn"],
                "notetype": row["notetype"],
                "note_year": int(row["note_year"]),
                "date_note": row.get("date_note"),
                "cptname": row.get("cptname"),
                "sta3n": row["sta3n"],
                "text": row["text"],
            }


def bulk_import_from_csv(corpus_db: Path, patients_csv: Path, documents_csv: Path) -> None:
    ensure_dir(corpus_db.parent)
    with initialize_corpus_db(corpus_db) as conn:
        import_patients(conn, load_patients_csv(patients_csv))
        import_documents(conn, load_documents_csv(documents_csv))
        conn.commit()
