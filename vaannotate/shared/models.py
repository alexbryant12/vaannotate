"""Dataclass style records representing the persistent schema.

The models follow the specification outlined in the project brief.  Only the
columns that are part of the MVP are implemented, but the structure leaves
space for future expansion.  Each model inherits from :class:`Record` which
provides helper utilities for creating tables and saving rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .database import Record


# --------------------------- corpus.db tables ------------------------------- #


@dataclass
class Patient(Record):
    patient_icn: str
    sta3n: str
    date_index: Optional[str] = None
    softlabel: Optional[float] = None

    __tablename__ = "patients"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_icn TEXT PRIMARY KEY,
            sta3n TEXT NOT NULL,
            date_index TEXT NULL,
            softlabel REAL NULL
        )
        """
    )


@dataclass
class Document(Record):
    doc_id: str
    patient_icn: str
    notetype: str
    note_year: int
    date_note: str
    cptname: Optional[str]
    sta3n: str
    hash: str
    text: str

    __tablename__ = "documents"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            patient_icn TEXT NOT NULL,
            notetype TEXT NOT NULL,
            note_year INTEGER NOT NULL,
            date_note TEXT NOT NULL,
            cptname TEXT NULL,
            sta3n TEXT NOT NULL,
            hash TEXT NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(patient_icn) REFERENCES patients(patient_icn)
        )
        """
    )


# --------------------------- project.db tables ------------------------------ #


@dataclass
class Project(Record):
    project_id: str
    name: str
    created_at: str
    created_by: str

    __tablename__ = "projects"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL
        )
        """
    )


@dataclass
class Phenotype(Record):
    pheno_id: str
    project_id: str
    name: str
    level: str
    description: str
    storage_path: str

    __tablename__ = "phenotypes"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS phenotypes (
            pheno_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            level TEXT CHECK(level IN ('single_doc','multi_doc')) NOT NULL,
            description TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            UNIQUE(project_id, name),
            FOREIGN KEY(project_id) REFERENCES projects(project_id)
        )
        """
    )


@dataclass
class ProjectCorpus(Record):
    corpus_id: str
    project_id: str
    name: str
    relative_path: str
    created_at: str

    __tablename__ = "project_corpora"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS project_corpora (
            corpus_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(project_id, name),
            FOREIGN KEY(project_id) REFERENCES projects(project_id)
        )
        """
    )


@dataclass
class LabelSet(Record):
    labelset_id: str
    project_id: str
    pheno_id: Optional[str]
    version: int
    created_at: str
    created_by: str
    notes: str

    __tablename__ = "label_sets"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS label_sets (
            labelset_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            pheno_id TEXT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            notes TEXT NOT NULL,
            FOREIGN KEY(pheno_id) REFERENCES phenotypes(pheno_id),
            FOREIGN KEY(project_id) REFERENCES projects(project_id)
        )
        """
    )


@dataclass
class Label(Record):
    label_id: str
    labelset_id: str
    name: str
    type: str
    required: int
    order_index: int
    rules: str
    gating_expr: Optional[str]
    na_allowed: int
    unit: Optional[str]
    min: Optional[float]
    max: Optional[float]

    __tablename__ = "labels"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS labels (
            label_id TEXT PRIMARY KEY,
            labelset_id TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            required INTEGER NOT NULL,
            order_index INTEGER NOT NULL,
            rules TEXT NOT NULL,
            gating_expr TEXT NULL,
            na_allowed INTEGER NOT NULL,
            unit TEXT NULL,
            min REAL NULL,
            max REAL NULL,
            FOREIGN KEY(labelset_id) REFERENCES label_sets(labelset_id)
        )
        """
    )


@dataclass
class LabelOption(Record):
    option_id: str
    label_id: str
    value: str
    display: str
    order_index: int
    weight: Optional[float]

    __tablename__ = "label_options"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS label_options (
            option_id TEXT PRIMARY KEY,
            label_id TEXT NOT NULL,
            value TEXT NOT NULL,
            display TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            weight REAL NULL,
            FOREIGN KEY(label_id) REFERENCES labels(label_id)
        )
        """
    )


@dataclass
class Round(Record):
    round_id: str
    pheno_id: str
    round_number: int
    labelset_id: str
    config_hash: str
    rng_seed: int
    status: str
    created_at: str

    __tablename__ = "rounds"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS rounds (
            round_id TEXT PRIMARY KEY,
            pheno_id TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            labelset_id TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            rng_seed INTEGER NOT NULL,
            status TEXT CHECK(status IN ('draft','active','closed','adjudicating','finalized')) NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(pheno_id) REFERENCES phenotypes(pheno_id),
            FOREIGN KEY(labelset_id) REFERENCES label_sets(labelset_id)
        )
        """
    )


@dataclass
class RoundConfig(Record):
    round_id: str
    config_json: str

    __tablename__ = "round_configs"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS round_configs (
            round_id TEXT PRIMARY KEY,
            config_json TEXT NOT NULL,
            FOREIGN KEY(round_id) REFERENCES rounds(round_id)
        )
        """
    )


@dataclass
class Reviewer(Record):
    reviewer_id: str
    name: str
    email: str
    windows_account: Optional[str]

    __tablename__ = "reviewers"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS reviewers (
            reviewer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            windows_account TEXT NULL
        )
        """
    )


@dataclass
class Assignment(Record):
    assign_id: str
    round_id: str
    reviewer_id: str
    sample_size: int
    overlap_n: int
    created_at: str
    status: str

    __tablename__ = "assignments"
    __schema__ = (
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
        )
        """
    )


# ------------------------ assignment.db tables ------------------------------ #


@dataclass
class AssignmentUnit(Record):
    unit_id: str
    display_rank: int
    patient_icn: str
    doc_id: Optional[str]
    note_count: Optional[int]
    complete: int
    opened_at: Optional[str]
    completed_at: Optional[str]

    __tablename__ = "units"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS units (
            unit_id TEXT PRIMARY KEY,
            display_rank INTEGER NOT NULL,
            patient_icn TEXT NOT NULL,
            doc_id TEXT NULL,
            note_count INTEGER NULL,
            complete INTEGER DEFAULT 0,
            opened_at TEXT NULL,
            completed_at TEXT NULL
        )
        """
    )


@dataclass
class AssignmentUnitNote(Record):
    unit_id: str
    doc_id: str
    order_index: int

    __tablename__ = "unit_notes"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS unit_notes (
            unit_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            PRIMARY KEY(unit_id, doc_id)
        )
        """
    )


@dataclass
class AssignmentDocument(Record):
    doc_id: str
    hash: str
    text: str
    metadata_json: Optional[str] = None

    __tablename__ = "documents"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NULL
        )
        """
    )


@dataclass
class Annotation(Record):
    unit_id: str
    label_id: str
    value: Optional[str]
    value_num: Optional[float]
    value_date: Optional[str]
    na: int
    notes: Optional[str]

    __tablename__ = "annotations"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS annotations (
            unit_id TEXT NOT NULL,
            label_id TEXT NOT NULL,
            value TEXT NULL,
            value_num REAL NULL,
            value_date TEXT NULL,
            na INTEGER DEFAULT 0,
            notes TEXT NULL,
            PRIMARY KEY(unit_id, label_id)
        )
        """
    )


@dataclass
class Rationale(Record):
    rationale_id: str
    unit_id: str
    label_id: str
    doc_id: str
    start_offset: int
    end_offset: int
    snippet: str
    created_at: str

    __tablename__ = "rationales"
    __schema__ = (
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
        )
        """
    )


@dataclass
class Event(Record):
    event_id: str
    ts: str
    actor: str
    event_type: str
    payload_json: str

    __tablename__ = "events"
    __schema__ = (
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
