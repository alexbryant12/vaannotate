"""Corpus management utilities."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

from .project import init_corpus_db, ProjectPaths
from .utils import canonicalize_text, text_hash


def load_patients(paths: ProjectPaths, rows: Iterable[Mapping[str, str]]) -> None:
    with sqlite3.connect(paths.corpus_db) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO patients(patient_icn, sta3n, date_index, softlabel)
            VALUES (:patient_icn, :sta3n, :date_index, :softlabel)
            """,
            rows,
        )


def load_documents(paths: ProjectPaths, rows: Iterable[Mapping[str, str]]) -> None:
    with sqlite3.connect(paths.corpus_db) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO documents(
                doc_id, patient_icn, notetype, note_year, date_note, cptname, sta3n, hash, text
            ) VALUES (:doc_id, :patient_icn, :notetype, :note_year, :date_note, :cptname, :sta3n, :hash, :text)
            """,
            rows,
        )


def load_patients_csv(paths: ProjectPaths, csv_path: Path) -> None:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        load_patients(paths, reader)


def load_documents_csv(paths: ProjectPaths, csv_path: Path) -> None:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            text = canonicalize_text(row["text"])
            rows.append({
                "doc_id": row["doc_id"],
                "patient_icn": row["patient_icn"],
                "notetype": row["notetype"],
                "note_year": int(row["note_year"]),
                "date_note": row["date_note"],
                "cptname": row.get("cptname"),
                "sta3n": row["sta3n"],
                "hash": text_hash(text),
                "text": text,
            })
        load_documents(paths, rows)


def ensure_corpus(paths: ProjectPaths) -> None:
    init_corpus_db(paths.corpus_db)

