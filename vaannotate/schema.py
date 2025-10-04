"""Database schema helpers."""
from __future__ import annotations

from pathlib import Path
import sqlite3

from .utils import ensure_dir


PROJECT_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS projects(
        project_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS phenotypes(
        pheno_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        name TEXT NOT NULL,
        level TEXT NOT NULL CHECK(level IN ('single_doc','multi_doc')),
        description TEXT,
        storage_path TEXT NOT NULL,
        UNIQUE(project_id, name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS project_corpora(
        corpus_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        name TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(project_id, name),
        FOREIGN KEY(project_id) REFERENCES projects(project_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS label_sets(
        labelset_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        pheno_id TEXT,
        version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        created_by TEXT NOT NULL,
        notes TEXT,
        FOREIGN KEY(project_id) REFERENCES projects(project_id),
        FOREIGN KEY(pheno_id) REFERENCES phenotypes(pheno_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS labels(
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
        max REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS label_options(
        option_id TEXT PRIMARY KEY,
        label_id TEXT NOT NULL,
        value TEXT NOT NULL,
        display TEXT NOT NULL,
        order_index INTEGER NOT NULL,
        weight REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS rounds(
        round_id TEXT PRIMARY KEY,
        pheno_id TEXT NOT NULL,
        round_number INTEGER NOT NULL,
        labelset_id TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        rng_seed INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('draft','active','closed','adjudicating','finalized')),
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS round_configs(
        round_id TEXT PRIMARY KEY,
        config_json TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reviewers(
        reviewer_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        windows_account TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS assignments(
        assign_id TEXT PRIMARY KEY,
        round_id TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        sample_size INTEGER NOT NULL,
        overlap_n INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('open','submitted','imported'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS adjudications(
        adjudication_id TEXT PRIMARY KEY,
        round_id TEXT NOT NULL,
        unit_id TEXT NOT NULL,
        label_id TEXT NOT NULL,
        value TEXT,
        value_num REAL,
        value_date TEXT,
        na INTEGER DEFAULT 0,
        notes TEXT,
        adjudicator_id TEXT,
        adjudicated_at TEXT NOT NULL
    );
    """,
];


CORPUS_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS patients(
        patient_icn TEXT PRIMARY KEY,
        sta3n TEXT,
        date_index TEXT,
        softlabel REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS documents(
        doc_id TEXT PRIMARY KEY,
        patient_icn TEXT NOT NULL,
        notetype TEXT,
        note_year INTEGER,
        date_note TEXT,
        cptname TEXT,
        sta3n TEXT,
        hash TEXT NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY(patient_icn) REFERENCES patients(patient_icn)
    );
    """,
    """CREATE INDEX IF NOT EXISTS idx_documents_patient ON documents(patient_icn);""",
    """CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(note_year);""",
    """CREATE INDEX IF NOT EXISTS idx_documents_sta ON documents(sta3n);""",
    """CREATE INDEX IF NOT EXISTS idx_documents_notetype ON documents(notetype);""",
];


ASSIGNMENT_SCHEMA = [
    """
    PRAGMA journal_mode=WAL;
    """,
    """
    CREATE TABLE IF NOT EXISTS units(
        unit_id TEXT PRIMARY KEY,
        display_rank INTEGER NOT NULL,
        patient_icn TEXT,
        doc_id TEXT,
        note_count INTEGER,
        complete INTEGER DEFAULT 0,
        opened_at TEXT,
        completed_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS unit_notes(
        unit_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        order_index INTEGER NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS documents(
        doc_id TEXT PRIMARY KEY,
        hash TEXT NOT NULL,
        text TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS annotations(
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
    CREATE TABLE IF NOT EXISTS rationales(
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
    CREATE TABLE IF NOT EXISTS events(
        event_id TEXT PRIMARY KEY,
        ts TEXT NOT NULL,
        actor TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ui_state(
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
];


ROUND_AGG_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS unit_annotations(
        round_id TEXT NOT NULL,
        unit_id TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        label_id TEXT NOT NULL,
        value TEXT,
        value_num REAL,
        value_date TEXT,
        na INTEGER,
        notes TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS unit_summary(
        round_id TEXT NOT NULL,
        unit_id TEXT NOT NULL,
        patient_icn TEXT,
        doc_id TEXT,
        PRIMARY KEY(round_id, unit_id)
    );
    """,
];


def initialize_db(path: Path, schema: list[str]) -> sqlite3.Connection:
    """Create SQLite file with provided schema."""
    ensure_dir(path.parent)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    for statement in schema:
        conn.executescript(statement)
    conn.commit()
    return conn


def initialize_project_db(path: Path) -> sqlite3.Connection:
    return initialize_db(path, PROJECT_SCHEMA)


def initialize_corpus_db(path: Path) -> sqlite3.Connection:
    return initialize_db(path, CORPUS_SCHEMA)


def initialize_assignment_db(path: Path) -> sqlite3.Connection:
    return initialize_db(path, ASSIGNMENT_SCHEMA)


def initialize_round_aggregate_db(path: Path) -> sqlite3.Connection:
    return initialize_db(path, ROUND_AGG_SCHEMA)
