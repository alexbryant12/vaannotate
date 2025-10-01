"""Project level operations for VAAnnotate."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

from .utils import ensure_dir, stable_json_dump, utcnow_ts

PROJECT_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS phenotypes (
        pheno_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        name TEXT NOT NULL,
        level TEXT CHECK(level IN ('single_doc','multi_doc')) NOT NULL,
        description TEXT,
        UNIQUE(project_id, name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS label_sets (
        labelset_id TEXT PRIMARY KEY,
        pheno_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL,
        notes TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS labels (
        label_id TEXT PRIMARY KEY,
        labelset_id TEXT NOT NULL,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        required INTEGER NOT NULL,
        order_index INTEGER NOT NULL,
        rules TEXT,
        gating_expr TEXT,
        na_allowed INTEGER DEFAULT 0,
        unit TEXT,
        min REAL,
        max REAL,
        FOREIGN KEY(labelset_id) REFERENCES label_sets(labelset_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS label_options (
        option_id TEXT PRIMARY KEY,
        label_id TEXT NOT NULL,
        value TEXT NOT NULL,
        display TEXT NOT NULL,
        order_index INTEGER NOT NULL,
        weight REAL,
        FOREIGN KEY(label_id) REFERENCES labels(label_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS rounds (
        round_id TEXT PRIMARY KEY,
        pheno_id TEXT NOT NULL,
        round_number INTEGER NOT NULL,
        labelset_id TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        rng_seed INTEGER NOT NULL,
        status TEXT CHECK(status IN ('draft','active','closed','adjudicating','finalized')) NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS round_configs (
        round_id TEXT PRIMARY KEY,
        config_json TEXT NOT NULL,
        FOREIGN KEY(round_id) REFERENCES rounds(round_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reviewers (
        reviewer_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        windows_account TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS assignments (
        assign_id TEXT PRIMARY KEY,
        round_id TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        sample_size INTEGER NOT NULL,
        overlap_n INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT CHECK(status IN ('open','submitted','imported')) NOT NULL,
        FOREIGN KEY(round_id) REFERENCES rounds(round_id),
        FOREIGN KEY(reviewer_id) REFERENCES reviewers(reviewer_id)
    );
    """
]


ASSIGNMENT_SCHEMA = [
    """
    PRAGMA journal_mode=WAL;
    """,
    """
    CREATE TABLE IF NOT EXISTS units (
        unit_id TEXT PRIMARY KEY,
        display_rank INTEGER NOT NULL,
        patient_icn TEXT NOT NULL,
        doc_id TEXT,
        note_count INTEGER,
        complete INTEGER DEFAULT 0,
        opened_at TEXT,
        completed_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS unit_notes (
        unit_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        order_index INTEGER NOT NULL,
        PRIMARY KEY(unit_id, doc_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS annotations (
        unit_id TEXT NOT NULL,
        label_id TEXT NOT NULL,
        value TEXT,
        value_num REAL,
        value_date TEXT,
        na INTEGER DEFAULT 0,
        notes TEXT,
        PRIMARY KEY(unit_id, label_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS rationales (
        rationale_id TEXT PRIMARY KEY,
        unit_id TEXT NOT NULL,
        label_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        start_offset INTEGER NOT NULL,
        end_offset INTEGER NOT NULL,
        snippet TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        ts TEXT NOT NULL,
        actor TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL
    );
    """
]


CORPUS_SCHEMA = [
    """
    PRAGMA journal_mode=WAL;
    """,
    """
    CREATE TABLE IF NOT EXISTS patients (
        patient_icn TEXT PRIMARY KEY,
        sta3n TEXT NOT NULL,
        date_index TEXT,
        softlabel REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        patient_icn TEXT NOT NULL,
        notetype TEXT NOT NULL,
        note_year INTEGER NOT NULL,
        date_note TEXT NOT NULL,
        cptname TEXT,
        sta3n TEXT NOT NULL,
        hash TEXT NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY(patient_icn) REFERENCES patients(patient_icn)
    );
    """,
    """CREATE INDEX IF NOT EXISTS idx_documents_patient ON documents(patient_icn);""",
    """CREATE INDEX IF NOT EXISTS idx_documents_notetype ON documents(notetype);""",
    """CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(note_year);""",
    """CREATE INDEX IF NOT EXISTS idx_documents_sta ON documents(sta3n);"""
]


class ProjectPaths:
    """Helper object exposing canonical project folder locations."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.corpus = self.root / "corpus"
        self.phenotypes = self.root / "phenotypes"
        self.admin_tools = self.root / "admin_tools"
        self.project_db = self.root / "project.db"
        self.corpus_db = self.corpus / "corpus.db"

    def round_dir(self, pheno_id: str, round_number: int | str) -> Path:
        return self.phenotypes / pheno_id / "rounds" / str(round_number)


# -- initialization -----------------------------------------------------------------


def init_project(root: Path, project_id: str, name: str, created_by: str) -> ProjectPaths:
    paths = ProjectPaths(root)
    ensure_dir(paths.root)
    ensure_dir(paths.corpus)
    ensure_dir(paths.phenotypes)
    ensure_dir(paths.admin_tools)

    with sqlite3.connect(paths.project_db) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        for stmt in PROJECT_SCHEMA:
            conn.executescript(stmt)
        conn.execute(
            "INSERT OR REPLACE INTO projects(project_id, name, created_at, created_by) VALUES (?, ?, ?, ?)",
            (project_id, name, utcnow_ts(), created_by),
        )
    init_corpus_db(paths.corpus_db)
    return paths


# -- helpers -------------------------------------------------------------------------


def ensure_pheno(paths: ProjectPaths, pheno_id: str, project_id: str, name: str, level: str, description: str) -> None:
    with sqlite3.connect(paths.project_db) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(
            """
            INSERT OR REPLACE INTO phenotypes(pheno_id, project_id, name, level, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pheno_id, project_id, name, level, description),
        )


def register_labelset(paths: ProjectPaths, *, labelset_id: str, pheno_id: str, version: int, created_by: str, notes: str | None, labels: Iterable[Mapping[str, object]]) -> None:
    with sqlite3.connect(paths.project_db) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(
            "INSERT OR REPLACE INTO label_sets(labelset_id, pheno_id, version, created_at, created_by, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (labelset_id, pheno_id, version, utcnow_ts(), created_by, notes),
        )
        for label in labels:
            conn.execute(
                """
                INSERT OR REPLACE INTO labels(
                    label_id, labelset_id, name, type, required, order_index, rules, gating_expr, na_allowed, unit, min, max
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label["label_id"],
                    labelset_id,
                    label["name"],
                    label["type"],
                    1 if label.get("required", False) else 0,
                    label.get("order_index", 0),
                    label.get("rules"),
                    label.get("gating_expr"),
                    1 if label.get("na_allowed") else 0,
                    label.get("unit"),
                    label.get("min"),
                    label.get("max"),
                ),
            )
            for option in label.get("options", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO label_options(option_id, label_id, value, display, order_index, weight)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        option["option_id"],
                        label["label_id"],
                        option["value"],
                        option.get("display", option["value"]),
                        option.get("order_index", 0),
                        option.get("weight"),
                    ),
                )


def read_project_config(paths: ProjectPaths) -> Mapping[str, object]:
    with sqlite3.connect(paths.project_db) as conn:
        conn.row_factory = sqlite3.Row
        project = conn.execute("SELECT * FROM projects LIMIT 1").fetchone()
        phenotypes = conn.execute("SELECT * FROM phenotypes").fetchall()
        return {
            "project": dict(project) if project else None,
            "phenotypes": [dict(row) for row in phenotypes],
        }


def save_round_config(round_dir: Path, config: Mapping[str, object]) -> None:
    ensure_dir(round_dir)
    stable_json_dump(round_dir / "round_config.json", config)


def record_round(paths: ProjectPaths, config: Mapping[str, object], *, round_id: str, pheno_id: str, round_number: int, labelset_id: str, rng_seed: int, status: str = "active") -> None:
    config_json = json.dumps(config, sort_keys=True)
    config_hash = sqlite3.Connection("file::memory:")  # placeholder for hashing via sqlite? not good
    import hashlib

    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    with sqlite3.connect(paths.project_db) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(
            """
            INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, utcnow_ts()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO round_configs(round_id, config_json) VALUES (?, ?)",
            (round_id, config_json),
        )


def register_reviewers(paths: ProjectPaths, reviewers: Iterable[Mapping[str, object]]) -> None:
    with sqlite3.connect(paths.project_db) as conn:
        for reviewer in reviewers:
            conn.execute(
                """
                INSERT OR IGNORE INTO reviewers(reviewer_id, name, email, windows_account)
                VALUES (?, ?, ?, ?)
                """,
                (
                    reviewer["id"],
                    reviewer.get("name", reviewer["id"]),
                    reviewer.get("email"),
                    reviewer.get("windows_account"),
                ),
            )


def create_assignment_record(paths: ProjectPaths, *, assign_id: str, round_id: str, reviewer_id: str, sample_size: int, overlap_n: int) -> None:
    with sqlite3.connect(paths.project_db) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO assignments(assign_id, round_id, reviewer_id, sample_size, overlap_n, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (assign_id, round_id, reviewer_id, sample_size, overlap_n, utcnow_ts(), "open"),
        )


def init_assignment_db(path: Path) -> None:
    ensure_dir(path.parent)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        for stmt in ASSIGNMENT_SCHEMA:
            conn.executescript(stmt)


def init_corpus_db(path: Path) -> None:
    ensure_dir(path.parent)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        for stmt in CORPUS_SCHEMA:
            conn.executescript(stmt)

