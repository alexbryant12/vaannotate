#!/usr/bin/env python
"""Seed a fully-populated toy project for demonstrations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vaannotate.project import (
    add_labelset,
    add_phenotype,
    fetch_labelset,
    get_connection,
    init_project,
    register_reviewer,
)
from vaannotate.rounds import RoundBuilder
from vaannotate.utils import canonical_json, ensure_dir


@dataclass
class Note:
    doc_id: str
    patient_icn: str
    sta3n: str
    notetype: str
    date: str
    text: str

    @property
    def note_year(self) -> int:
        return int(self.date.split("-")[0])


PATIENTS: List[Dict[str, str]] = [
    {"patient_icn": f"20{index:02d}", "sta3n": "506" if index % 2 else "515"}
    for index in range(1, 11)
]

NOTE_THEMES = [
    ("PRIMARY CARE NOTE", "Primary care follow-up summarizing comprehensive chronic disease management."),
    ("ENDOCRINOLOGY NOTE", "Endocrinology consultation focused on medication titration and long-term planning."),
    ("PHARMACY NOTE", "Pharmacy medication therapy management review concentrating on adherence and refills."),
    ("NUTRITION NOTE", "Nutrition counseling session highlighting meal planning and lifestyle reinforcement."),
    ("TELEHEALTH NOTE", "Telehealth touchpoint documenting remote symptom monitoring and coaching."),
]


def generate_notes(patients: Sequence[Dict[str, str]], total_notes: int) -> List[Note]:
    notes: List[Note] = []
    theme_count = len(NOTE_THEMES)
    for index in range(total_notes):
        patient = patients[index % len(patients)]
        notetype, theme = NOTE_THEMES[index % theme_count]
        year = 2015 + (index % 10)
        month = (index % 12) + 1
        day = (index % 28) + 1
        date = f"{year}-{month:02d}-{day:02d}"
        doc_id = f"{patient['patient_icn']}_{date.replace('-', '')}_{index:03d}"
        paragraphs = [
            f"{theme} Visit date {date} at facility {patient['sta3n']}. Clinicians reviewed vitals, symptoms, and chart history while confirming medication reconciliation for patient {patient['patient_icn']}.",
            (
                "Medication adjustments included refreshed metformin guidance, individualized insulin teaching, and"
                f" reinforcement of home glucose monitoring. Documented HbA1c trend measured {6.4 + (index % 9) * 0.2:.1f}"
                " with comparison to prior labs to demonstrate progress."
            ),
            "Care planning emphasized social support, nutrition, and physical activity goals. The note records resources offered for transportation, pharmacy follow-up, and nursing outreach to close care gaps.",
            "Structured assessment captured review of systems, motivational interviewing highlights, and next steps for laboratory surveillance with clear thresholds for escalation.",
        ]
        text = "\n\n".join(paragraphs)
        notes.append(
            Note(
                doc_id=doc_id,
                patient_icn=patient["patient_icn"],
                sta3n=patient["sta3n"],
                notetype=notetype,
                date=date,
                text=text,
            )
        )
    return notes


NOTES: List[Note] = generate_notes(PATIENTS, 100)

REVIEWERS = [
    {"reviewer_id": "r_alex", "name": "Alex Reviewer", "email": "alex@example.test"},
    {"reviewer_id": "r_blake", "name": "Blake Reviewer", "email": "blake@example.test"},
]

LABEL_RULES = {
    "Has_phenotype": (
        "Mark Yes when the chart shows diabetes (diagnosis, diabetes meds, or A1c ≥ 6.5). "
        "Select No when diabetes is explicitly ruled out. Choose Unknown when evidence is inconclusive."
    ),
    "Evidence_type": (
        "Check all supporting evidence types when Has_phenotype is Yes. "
        "Medication = diabetes medications noted; Lab = abnormal HbA1c; Radiology = imaging describing diabetes complications."
    ),
    "HbA1c_value": (
        "Record the most recent HbA1c value mentioned (3.0–20.0). Use decimals exactly as written. "
        "Leave blank or mark N/A if not provided."
    ),
    "Notes": "Add any clarifying comments for adjudication or context.",
    "HTN_Has_phenotype": (
        "Mark Yes when hypertension or elevated blood pressure management is evident. "
        "Choose No when hypertension is ruled out and Unknown when information is insufficient."
    ),
    "HTN_Controlled": (
        "Select the best description of blood pressure control. Controlled = goals met; Uncontrolled = persistently high; Unknown = not documented."
    ),
    "HTN_Notes": "Optional comments specific to hypertension findings.",
}

ROUND_CONFIGS = [
    {
        "pheno_id": "ph_diabetes",
        "labelset_id": "ls_diabetes_v1",
        "round_number": 1,
        "round_id": "ph_diabetes_r1",
        "created_by": "toy_seed",
        "filters": {
            "patient": {
                "year_range": [2018, 2024],
                "sta3n_in": ["506", "515"],
            },
            "note": {
                "notetype_in": ["PRIMARY CARE NOTE", "ENDOCRINOLOGY NOTE"],
                "regex": r"(metformin|insulin|hba1c\\s*\\d+(\\.\\d+)?)",
                "regex_flags": "i",
            },
        },
        "stratification": {"keys": ["note_year"], "sample_per_stratum": 2},
        "reviewers": [
            {"id": "r_alex", "name": "Alex Reviewer"},
            {"id": "r_blake", "name": "Blake Reviewer"},
        ],
        "overlap_n": 2,
        "independent": True,
        "rng_seed": 133742,
    },
    {
        "pheno_id": "ph_diabetes",
        "labelset_id": "ls_diabetes_v1",
        "round_number": 2,
        "round_id": "ph_diabetes_r2",
        "created_by": "toy_seed",
        "filters": {
            "patient": {"sta3n_in": ["506", "515"]},
            "note": {
                "notetype_in": ["PHARMACY NOTE", "ENDOCRINOLOGY NOTE"],
                "note_year": [2016, 2020],
            },
        },
        "stratification": {"keys": ["sta3n"], "sample_per_stratum": 3},
        "reviewers": [
            {"id": "r_alex", "name": "Alex Reviewer"},
            {"id": "r_blake", "name": "Blake Reviewer"},
        ],
        "overlap_n": 2,
        "independent": True,
        "rng_seed": 133743,
    },
    {
        "pheno_id": "ph_diabetes",
        "labelset_id": "ls_diabetes_v1",
        "round_number": 3,
        "round_id": "ph_diabetes_r3",
        "created_by": "toy_seed",
        "filters": {
            "patient": {
                "year_range": [2015, 2022],
            },
            "note": {
                "notetype_in": ["PRIMARY CARE NOTE", "TELEHEALTH NOTE"],
            },
        },
        "stratification": {"keys": ["note_year", "sta3n"], "sample_per_stratum": 2},
        "reviewers": [
            {"id": "r_alex", "name": "Alex Reviewer"},
            {"id": "r_blake", "name": "Blake Reviewer"},
        ],
        "overlap_n": 2,
        "independent": True,
        "rng_seed": 133744,
    },
    {
        "pheno_id": "ph_hypertension",
        "labelset_id": "ls_htn_v1",
        "round_number": 1,
        "round_id": "ph_hypertension_r1",
        "created_by": "toy_seed",
        "filters": {
            "patient": {"sta3n_in": ["506", "515"]},
            "note": {
                "notetype_in": ["PRIMARY CARE NOTE", "PHARMACY NOTE"],
                "regex": r"blood pressure|bp\\s*:?\\s*\\d{2,3}/\\d{2,3}",
                "regex_flags": "i",
            },
        },
        "stratification": {"keys": ["note_year"], "sample_per_stratum": 3},
        "reviewers": [
            {"id": "r_alex", "name": "Alex Reviewer"},
            {"id": "r_blake", "name": "Blake Reviewer"},
        ],
        "overlap_n": 2,
        "independent": True,
        "rng_seed": 133750,
    },
]


def normalize_text(text: str) -> str:
    """Return normalized text for hashing and storage."""
    stripped = "\n".join(line.strip() for line in text.strip().splitlines())
    return stripped


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_corpus(corpus_db: Path, notes: Sequence[Note]) -> None:
    with get_connection(corpus_db) as conn:
        for patient in PATIENTS:
            conn.execute(
                "INSERT OR REPLACE INTO patients(patient_icn, sta3n, date_index, softlabel) VALUES (?,?,?,?)",
                (patient["patient_icn"], patient["sta3n"], None, None),
            )
        for note in notes:
            normalized = normalize_text(note.text)
            conn.execute(
                """
                INSERT OR REPLACE INTO documents(
                    doc_id, patient_icn, notetype, note_year, date_note,
                    cptname, sta3n, hash, text
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    note.doc_id,
                    note.patient_icn,
                    note.notetype,
                    note.note_year,
                    note.date,
                    None,
                    note.sta3n,
                    compute_hash(normalized),
                    normalized,
                ),
            )
        conn.commit()


def seed_metadata(project_db: Path, corpus_paths: Dict[str, str]) -> None:
    with get_connection(project_db) as conn:
        for reviewer in REVIEWERS:
            register_reviewer(conn, **reviewer)
        def corpus_for(pheno_id: str) -> str:
            path = corpus_paths.get(pheno_id)
            if not path:
                raise RuntimeError(f"Missing corpus path for {pheno_id}")
            return path
        add_phenotype(
            conn,
            pheno_id="ph_diabetes",
            project_id="Project_Toy",
            name="Diabetes Phenotyping",
            level="multi_doc",
            description="Toy diabetes phenotype for demonstrations",
            corpus_path=corpus_for("ph_diabetes"),
        )
        add_labelset(
            conn,
            labelset_id="ls_diabetes_v1",
            pheno_id="ph_diabetes",
            version=1,
            created_by="toy_seed",
            notes="Initial diabetes labelset for toy project",
            labels=[
                {
                    "label_id": "Has_phenotype",
                    "name": "Has phenotype",
                    "type": "categorical_single",
                    "required": True,
                    "order_index": 0,
                    "rules": LABEL_RULES["Has_phenotype"],
                    "options": [
                        {"value": "yes", "display": "Yes"},
                        {"value": "no", "display": "No"},
                        {"value": "unknown", "display": "Unknown"},
                    ],
                },
                {
                    "label_id": "Evidence_type",
                    "name": "Evidence type",
                    "type": "categorical_multi",
                    "required": False,
                    "order_index": 1,
                    "rules": LABEL_RULES["Evidence_type"],
                    "gating_expr": "Has_phenotype == 'yes'",
                    "options": [
                        {"value": "Medication", "display": "Medication"},
                        {"value": "Lab", "display": "Lab"},
                        {"value": "Radiology", "display": "Radiology"},
                    ],
                },
                {
                    "label_id": "HbA1c_value",
                    "name": "HbA1c value",
                    "type": "float",
                    "required": False,
                    "order_index": 2,
                    "rules": LABEL_RULES["HbA1c_value"],
                    "min": 3.0,
                    "max": 20.0,
                    "na_allowed": True,
                    "unit": None,
                },
                {
                    "label_id": "Notes",
                    "name": "Notes",
                    "type": "text",
                    "required": False,
                    "order_index": 3,
                    "rules": LABEL_RULES["Notes"],
                    "na_allowed": False,
                },
            ],
        )
        add_phenotype(
            conn,
            pheno_id="ph_hypertension",
            project_id="Project_Toy",
            name="Hypertension Phenotyping",
            level="single_doc",
            description="Toy hypertension phenotype for demonstrations",
            corpus_path=corpus_for("ph_hypertension"),
        )
        add_labelset(
            conn,
            labelset_id="ls_htn_v1",
            pheno_id="ph_hypertension",
            version=1,
            created_by="toy_seed",
            notes="Initial hypertension labelset for toy project",
            labels=[
                {
                    "label_id": "HTN_Has_phenotype",
                    "name": "Has hypertension",
                    "type": "categorical_single",
                    "required": True,
                    "order_index": 0,
                    "rules": LABEL_RULES["HTN_Has_phenotype"],
                    "options": [
                        {"value": "yes", "display": "Yes"},
                        {"value": "no", "display": "No"},
                        {"value": "unknown", "display": "Unknown"},
                    ],
                },
                {
                    "label_id": "HTN_Controlled",
                    "name": "Blood pressure control",
                    "type": "categorical_single",
                    "required": False,
                    "order_index": 1,
                    "rules": LABEL_RULES["HTN_Controlled"],
                    "gating_expr": "HTN_Has_phenotype == 'yes'",
                    "options": [
                        {"value": "controlled", "display": "Controlled"},
                        {"value": "uncontrolled", "display": "Uncontrolled"},
                        {"value": "unknown", "display": "Unknown"},
                    ],
                },
                {
                    "label_id": "HTN_Notes",
                    "name": "Hypertension notes",
                    "type": "text",
                    "required": False,
                    "order_index": 2,
                    "rules": LABEL_RULES["HTN_Notes"],
                    "na_allowed": False,
                },
            ],
        )
        conn.commit()


def write_label_schema(project_db: Path, assignment_dir: Path, labelset_id: str) -> None:
    labelset = None
    with get_connection(project_db) as conn:
        labelset = fetch_labelset(conn, labelset_id)
    schema_labels: List[Dict[str, object]] = []
    for label in labelset["labels"]:
        schema_labels.append(
            {
                "label_id": label["label_id"],
                "name": label["name"],
                "type": label["type"],
                "required": bool(label["required"]),
                "na_allowed": bool(label.get("na_allowed")),
                "rules": label.get("rules"),
                "unit": label.get("unit"),
                "range": {"min": label.get("min"), "max": label.get("max")},
                "gating_expr": label.get("gating_expr"),
                "options": [
                    {
                        "value": option["value"],
                        "display": option["display"],
                        "order_index": option.get("order_index", idx),
                        "weight": option.get("weight"),
                    }
                    for idx, option in enumerate(label.get("options", []))
                ],
            }
        )
    payload = {"labelset_id": labelset_id, "labels": schema_labels}
    (assignment_dir / "label_schema.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def augment_assignment_db(
    assignment_path: Path,
    patient_docs: Dict[str, List[sqlite3.Row]],
) -> None:
    with sqlite3.connect(assignment_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, hash TEXT NOT NULL, text TEXT NOT NULL)"
        )
        units = conn.execute("SELECT unit_id, patient_icn FROM units").fetchall()
        for unit_id, patient_icn in units:
            docs = patient_docs.get(patient_icn)
            if not docs:
                continue
            conn.execute("DELETE FROM unit_notes WHERE unit_id=?", (unit_id,))
            for order_index, doc_row in enumerate(docs):
                conn.execute(
                    "INSERT OR REPLACE INTO unit_notes(unit_id, doc_id, order_index) VALUES (?,?,?)",
                    (unit_id, doc_row["doc_id"], order_index),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO documents(doc_id, hash, text) VALUES (?,?,?)",
                    (doc_row["doc_id"], doc_row["hash"], doc_row["text"]),
                )
        conn.commit()


def copy_client_binary(repo_root: Path, assignment_dir: Path) -> None:
    source = repo_root / "dist" / "ClientApp.exe"
    ensure_dir(source.parent)
    if not source.exists():
        source.write_text("ClientApp placeholder\n", encoding="utf-8")
    shutil.copy2(source, assignment_dir / "client.exe")


def copy_client_script(repo_root: Path, assignment_dir: Path) -> None:
    script_source = repo_root / "scripts" / "run_client.ps1"
    if script_source.exists():
        scripts_dir = ensure_dir(assignment_dir / "scripts")
        shutil.copy2(script_source, scripts_dir / "run_client.ps1")


def seed_disagreement(assignment_path: Path, unit_id: str, reviewer: str) -> None:
    value_map = {
        "r_alex": {"Has_phenotype": "yes", "Evidence_type": "Medication,Lab", "HbA1c_value": ("7.9", 7.9)},
        "r_blake": {"Has_phenotype": "no"},
    }
    payload = value_map.get(reviewer)
    if not payload:
        return
    with sqlite3.connect(assignment_path) as conn:
        if "Has_phenotype" in payload:
            conn.execute(
                "UPDATE annotations SET value=? WHERE unit_id=? AND label_id='Has_phenotype'",
                (payload["Has_phenotype"], unit_id),
            )
        if reviewer == "r_alex":
            conn.execute(
                "UPDATE annotations SET value=? WHERE unit_id=? AND label_id='Evidence_type'",
                (payload["Evidence_type"], unit_id),
            )
            conn.execute(
                "UPDATE annotations SET value=?, value_num=? WHERE unit_id=? AND label_id='HbA1c_value'",
                (payload["HbA1c_value"][0], payload["HbA1c_value"][1], unit_id),
            )
            conn.execute(
                "UPDATE annotations SET value=? WHERE unit_id=? AND label_id='Notes'",
                ("Metformin start with elevated A1c.", unit_id),
            )
        else:
            conn.execute(
                "UPDATE annotations SET value=NULL, value_num=NULL WHERE unit_id=? AND label_id='HbA1c_value'",
                (unit_id,),
            )
            conn.execute(
                "UPDATE annotations SET value=NULL WHERE unit_id=? AND label_id='Evidence_type'",
                (unit_id,),
            )
        conn.commit()


def load_patient_docs(corpus_db: Path) -> Dict[str, List[sqlite3.Row]]:
    mapping: Dict[str, List[sqlite3.Row]] = {}
    with get_connection(corpus_db) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT doc_id, patient_icn, hash, text, date_note FROM documents ORDER BY patient_icn, date_note"
        )
        for row in cursor:
            mapping.setdefault(row["patient_icn"], []).append(row)
    return mapping


def find_overlap_units(manifest_path: Path) -> List[str]:
    units: List[str] = []
    if not manifest_path.exists():
        return units
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            flag = row.get("is_overlap", "")
            if str(flag) in {"1", "true", "True"}:
                unit_id = row.get("unit_id")
                if unit_id and unit_id not in units:
                    units.append(unit_id)
    return units


def mark_round_final(project_db: Path, round_id: str) -> None:
    with get_connection(project_db) as conn:
        conn.execute("UPDATE rounds SET status='finalized' WHERE round_id=?", (round_id,))
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the demo toy project")
    parser.add_argument("--project", default="demo/Project_Toy", help="Relative path to project root")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    project_root = (repo_root / args.project).resolve()
    if project_root.exists():
        shutil.rmtree(project_root)

    paths = init_project(project_root, "Project_Toy", "Project Toy", "toy_seed")
    phenotype_corpus_dbs: Dict[str, Path] = {}
    for pheno_id in ("ph_diabetes", "ph_hypertension"):
        corpus_dir = ensure_dir(project_root / "phenotypes" / pheno_id / "corpus")
        corpus_db = corpus_dir / "corpus.db"
        seed_corpus(corpus_db, NOTES)
        phenotype_corpus_dbs[pheno_id] = corpus_db
    relative_corpus = {
        pheno_id: corpus_db.relative_to(project_root).as_posix()
        for pheno_id, corpus_db in phenotype_corpus_dbs.items()
    }
    seed_metadata(paths.project_db, relative_corpus)

    builder = RoundBuilder(project_root)
    patient_docs_map = {pheno_id: load_patient_docs(db_path) for pheno_id, db_path in phenotype_corpus_dbs.items()}
    config_dir = ensure_dir(project_root / "config")
    for config in ROUND_CONFIGS:
        pheno_dir = ensure_dir(project_root / "phenotypes" / config["pheno_id"])
        ensure_dir(pheno_dir / "rounds")
        config_path = config_dir / f"{config['round_id']}.json"
        config_path.write_text(canonical_json(config), encoding="utf-8")
        result = builder.generate_round(config["pheno_id"], config_path, created_by=config.get("created_by", "toy_seed"))
        round_dir = Path(result["round_dir"])
        overlap_unit_ids = find_overlap_units(round_dir / "manifest.csv")
        if len(overlap_unit_ids) < 2:
            raise RuntimeError(
                f"Round {config['round_id']} generated only {len(overlap_unit_ids)} overlapping units;"
                " expected at least two for the toy example."
            )
        for reviewer in config["reviewers"]:
            assign_dir = round_dir / "assignments" / reviewer["id"]
            assignment_db = assign_dir / "assignment.db"
            augment_assignment_db(assignment_db, patient_docs_map.get(config["pheno_id"], {}))
            write_label_schema(paths.project_db, assign_dir, config["labelset_id"])
            copy_client_binary(repo_root, assign_dir)
            copy_client_script(repo_root, assign_dir)
            if (
                overlap_unit_ids
                and config["pheno_id"] == "ph_diabetes"
                and config["round_number"] == 1
            ):
                seed_disagreement(assignment_db, overlap_unit_ids[0], reviewer["id"])
        for reviewer in config["reviewers"]:
            builder.import_assignment(config["pheno_id"], config["round_number"], reviewer["id"])
        builder.build_round_aggregate(config["pheno_id"], config["round_number"])
        mark_round_final(paths.project_db, config["round_id"])

    print(f"Seeded toy project at {project_root}")


if __name__ == "__main__":
    main()
