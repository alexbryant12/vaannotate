"""Tests for reviewer assignment import and aggregation pipeline."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from vaannotate.project import (
    add_labelset,
    add_phenotype,
    get_connection,
    init_project,
    register_reviewer,
)
from vaannotate.rounds import RoundBuilder
from vaannotate.schema import initialize_corpus_db


@pytest.fixture()
def seeded_project(tmp_path: Path) -> tuple[RoundBuilder, Path]:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    with get_connection(paths.project_db) as conn:
        register_reviewer(conn, "rev_one", "Reviewer One")
        register_reviewer(conn, "rev_two", "Reviewer Two")
        add_phenotype(
            conn,
            pheno_id="ph_test",
            project_id="proj",
            name="Test phenotype",
            level="single_doc",
            corpus_path="phenotypes/ph_test/corpus/corpus.db",
        )
        add_labelset(
            conn,
            labelset_id="ls_test",
            pheno_id="ph_test",
            version=1,
            created_by="tester",
            notes=None,
            labels=[
                {
                    "label_id": "Flag",
                    "name": "Flag",
                    "type": "categorical_single",
                    "required": False,
                    "options": [
                        {"value": "yes", "display": "Yes"},
                        {"value": "no", "display": "No"},
                    ],
                },
                {
                    "label_id": "Score",
                    "name": "Score",
                    "type": "float",
                    "required": False,
                    "na_allowed": True,
                    "options": [],
                },
            ],
        )
        conn.commit()

    corpus_db = project_root / "phenotypes" / "ph_test" / "corpus" / "corpus.db"
    with initialize_corpus_db(corpus_db) as corpus_conn:
        for idx in range(3):
            patient_id = f"p{idx}"
            corpus_conn.execute(
                "INSERT INTO patients(patient_icn, sta3n, date_index, softlabel) VALUES (?,?,?,?)",
                (patient_id, "506", None, None),
            )
            corpus_conn.execute(
                """
                INSERT INTO documents(
                    doc_id, patient_icn, notetype, note_year, date_note,
                    cptname, sta3n, hash, text
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"doc_{idx}",
                    patient_id,
                    "NOTE",
                    2020,
                    "2020-01-01",
                    "",
                    "506",
                    f"hash{idx}",
                    f"Example text {idx}",
                ),
            )
        corpus_conn.commit()

    config_dir = paths.admin_dir / "round_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "round_number": 1,
        "round_id": "ph_test_r1",
        "labelset_id": "ls_test",
        "reviewers": [
            {"id": "rev_one", "name": "Reviewer One"},
            {"id": "rev_two", "name": "Reviewer Two"},
        ],
        "overlap_n": 1,
        "rng_seed": 123,
        "filters": {},
    }
    config_path = config_dir / "ph_test_r1.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    builder = RoundBuilder(project_root)
    builder.generate_round("ph_test", config_path, created_by="tester")

    return builder, project_root


def _annotate_assignment(db_path: Path, flag_value: str | None, score_factory) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        unit_ids = [row[0] for row in conn.execute("SELECT unit_id FROM units ORDER BY display_rank").fetchall()]
        for idx, unit_id in enumerate(unit_ids):
            conn.execute(
                "UPDATE annotations SET value=?, value_num=?, na=? WHERE unit_id=? AND label_id='Flag'",
                (
                    flag_value,
                    None,
                    0 if flag_value is not None else 1,
                    unit_id,
                ),
            )
            score_value = score_factory(idx)
            if score_value is None:
                conn.execute(
                    "UPDATE annotations SET value=NULL, value_num=NULL, na=1 WHERE unit_id=? AND label_id='Score'",
                    (unit_id,),
                )
            else:
                conn.execute(
                    "UPDATE annotations SET value=?, value_num=?, na=0 WHERE unit_id=? AND label_id='Score'",
                    (f"{score_value}", float(score_value), unit_id),
                )
        conn.commit()
    return unit_ids


def test_import_preserves_all_reviewer_annotations(seeded_project: tuple[RoundBuilder, Path]) -> None:
    builder, project_root = seeded_project
    round_dir = project_root / "phenotypes" / "ph_test" / "rounds" / "round_1"

    reviewer_updates: dict[str, list[str]] = {}
    reviewer_configs = {
        "rev_one": ("yes", lambda idx: 1.5 + idx),
        "rev_two": ("no", lambda idx: None),
    }
    for reviewer_id, (flag_value, score_factory) in reviewer_configs.items():
        assignment_db = round_dir / "assignments" / reviewer_id / "assignment.db"
        reviewer_updates[reviewer_id] = _annotate_assignment(assignment_db, flag_value, score_factory)
        builder.import_assignment("ph_test", 1, reviewer_id)

    aggregate_db = builder.build_round_aggregate("ph_test", 1)
    with sqlite3.connect(aggregate_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT reviewer_id, unit_id, label_id, value, value_num, na FROM unit_annotations"
        ).fetchall()

    assert {row["reviewer_id"] for row in rows} == {"rev_one", "rev_two"}

    observed = {
        (row["reviewer_id"], row["unit_id"], row["label_id"]): row for row in rows
    }

    for reviewer_id, unit_ids in reviewer_updates.items():
        for idx, unit_id in enumerate(unit_ids):
            flag_row = observed[(reviewer_id, unit_id, "Flag")]
            if reviewer_id == "rev_one":
                assert flag_row["value"] == "yes"
                assert flag_row["na"] == 0
            else:
                assert flag_row["value"] == "no"
                assert flag_row["na"] == 0

            score_row = observed[(reviewer_id, unit_id, "Score")]
            if reviewer_id == "rev_one":
                expected_score = pytest.approx(1.5 + idx)
                assert float(score_row["value"]) == expected_score
                assert score_row["value_num"] == pytest.approx(1.5 + idx)
                assert score_row["na"] == 0
            else:
                assert score_row["value"] is None
                assert score_row["value_num"] is None
                assert score_row["na"] == 1

