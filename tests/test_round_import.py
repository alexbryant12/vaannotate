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
from vaannotate.shared.database import Database
from vaannotate.shared.sampling import (
    SamplingFilters,
    allocate_units,
    candidate_documents,
    initialize_assignment_db,
    populate_assignment_db,
)


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
            project_id="proj",
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


def test_multi_doc_round_uses_patient_display_unit(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    with get_connection(paths.project_db) as conn:
        register_reviewer(conn, "rev_one", "Reviewer One")
        register_reviewer(conn, "rev_two", "Reviewer Two")
        add_phenotype(
            conn,
            pheno_id="ph_multi",
            project_id="proj",
            name="Multi Doc", 
            level="multi_doc",
            corpus_path="phenotypes/ph_multi/corpus/corpus.db",
        )
        add_labelset(
            conn,
            labelset_id="ls_multi",
            project_id="proj",
            pheno_id="ph_multi",
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
                }
            ],
        )
        conn.commit()

    docs_by_patient: dict[str, list[tuple[str, str]]] = {
        "p1": [("p1_doc1", "Patient 1 doc A"), ("p1_doc2", "Patient 1 doc B")],
        "p2": [("p2_doc1", "Patient 2 doc A"), ("p2_doc2", "Patient 2 doc B")],
        "p3": [("p3_doc1", "Patient 3 doc A"), ("p3_doc2", "Patient 3 doc B")],
    }
    corpus_db = project_root / "phenotypes" / "ph_multi" / "corpus" / "corpus.db"
    with initialize_corpus_db(corpus_db) as corpus_conn:
        for idx, (patient_icn, docs) in enumerate(docs_by_patient.items()):
            corpus_conn.execute(
                "INSERT INTO patients(patient_icn, sta3n, date_index, softlabel) VALUES (?,?,?,?)",
                (patient_icn, "506", None, None),
            )
            for doc_index, (doc_id, text) in enumerate(docs):
                corpus_conn.execute(
                    """
                    INSERT INTO documents(
                        doc_id, patient_icn, notetype, note_year, date_note,
                        cptname, sta3n, hash, text
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        doc_id,
                        patient_icn,
                        "NOTE",
                        2020 + doc_index,
                        f"202{doc_index}-01-01",
                        "",
                        "506",
                        f"hash_{doc_id}",
                        text,
                    ),
                )
        corpus_conn.commit()

    config_dir = paths.admin_dir / "round_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "round_number": 1,
        "round_id": "ph_multi_r1",
        "labelset_id": "ls_multi",
        "reviewers": [
            {"id": "rev_one", "name": "Reviewer One"},
            {"id": "rev_two", "name": "Reviewer Two"},
        ],
        "overlap_n": 0,
        "rng_seed": 7,
        "filters": {},
    }
    config_path = config_dir / "ph_multi_r1.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    builder = RoundBuilder(project_root)
    builder.generate_round("ph_multi", config_path, created_by="tester")

    round_dir = project_root / "phenotypes" / "ph_multi" / "rounds" / "round_1"
    assignment_db = round_dir / "assignments" / "rev_one" / "assignment.db"
    assert assignment_db.exists()

    with sqlite3.connect(assignment_db) as conn:
        conn.row_factory = sqlite3.Row
        units = conn.execute(
            "SELECT unit_id, patient_icn, doc_id, note_count FROM units ORDER BY display_rank"
        ).fetchall()
        assert units
        for unit in units:
            patient_icn = unit["patient_icn"]
            assert unit["doc_id"] in (None, "")
            expected_docs = [doc_id for doc_id, _text in docs_by_patient[patient_icn]]
            assert unit["note_count"] == len(expected_docs)
            doc_rows = conn.execute(
                "SELECT doc_id FROM unit_notes WHERE unit_id=? ORDER BY order_index",
                (unit["unit_id"],),
            ).fetchall()
            observed_docs = [row["doc_id"] for row in doc_rows]
            assert observed_docs == expected_docs
            for doc_id in observed_docs:
                doc_row = conn.execute(
                    "SELECT text FROM documents WHERE doc_id=?",
                    (doc_id,),
                ).fetchone()
                assert doc_row is not None
                assert doc_row["text"]


def test_admin_sampling_creates_multi_doc_units(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.db"
    docs_by_patient: dict[str, list[tuple[str, str]]] = {
        "p1": [("p1_doc1", "P1 doc one"), ("p1_doc2", "P1 doc two")],
        "p2": [("p2_doc1", "P2 doc one"), ("p2_doc2", "P2 doc two")],
    }
    with initialize_corpus_db(corpus_path) as corpus_conn:
        for patient_icn, docs in docs_by_patient.items():
            corpus_conn.execute(
                "INSERT INTO patients(patient_icn, sta3n, date_index, softlabel) VALUES (?,?,?,?)",
                (patient_icn, "506", None, None),
            )
            for order_idx, (doc_id, text) in enumerate(docs):
                corpus_conn.execute(
                    """
                    INSERT INTO documents(
                        doc_id, patient_icn, notetype, note_year, date_note,
                        cptname, sta3n, hash, text
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        doc_id,
                        patient_icn,
                        "NOTE",
                        2020 + order_idx,
                        f"202{order_idx}-01-01",
                        "",
                        "506",
                        f"hash_{doc_id}",
                        text,
                    ),
                )
        corpus_conn.commit()

    corpus_db = Database(corpus_path)
    filters = SamplingFilters({}, {})
    candidates = candidate_documents(corpus_db, "multi_doc", filters)
    assert len(candidates) == len(docs_by_patient)

    reviewers = [{"id": "rev_a", "name": "Reviewer"}]
    assignments = allocate_units(candidates, reviewers, overlap_n=0, seed=7)

    assignment_db = initialize_assignment_db(tmp_path / "assignment.db")
    populate_assignment_db(assignment_db, "rev_a", assignments["rev_a"].units)

    with assignment_db.connect() as conn:
        conn.row_factory = sqlite3.Row
        unit_rows = conn.execute(
            "SELECT unit_id, patient_icn, doc_id, note_count FROM units ORDER BY display_rank"
        ).fetchall()
        assert len(unit_rows) == len(docs_by_patient)
        for unit in unit_rows:
            unit_id = unit["unit_id"]
            assert unit["patient_icn"] == unit_id
            assert unit["doc_id"] is None
            expected_docs = docs_by_patient[unit_id]
            assert unit["note_count"] == len(expected_docs)
            doc_rows = conn.execute(
                "SELECT doc_id FROM unit_notes WHERE unit_id=? ORDER BY order_index",
                (unit_id,),
            ).fetchall()
            observed = [row["doc_id"] for row in doc_rows]
            assert observed == [doc_id for doc_id, _ in expected_docs]
            text_map = {doc_id: text for doc_id, text in expected_docs}
            for doc_id in observed:
                doc_row = conn.execute(
                    "SELECT text FROM documents WHERE doc_id=?",
                    (doc_id,),
                ).fetchone()
                assert doc_row is not None
                assert doc_row["text"] == text_map[doc_id]


def test_allocate_units_respects_total_n() -> None:
    rows = [
        {
            "unit_id": f"unit_{idx}",
            "patient_icn": f"pat_{idx}",
            "doc_id": f"doc_{idx}",
            "hash": f"hash_{idx}",
            "text": f"text {idx}",
        }
        for idx in range(20)
    ]
    reviewers = [
        {"id": "rev_a", "name": "Reviewer A"},
        {"id": "rev_b", "name": "Reviewer B"},
        {"id": "rev_c", "name": "Reviewer C"},
    ]
    overlap_n = 2
    total_n = 10

    assignments = allocate_units(rows, reviewers, overlap_n=overlap_n, seed=42, total_n=total_n)

    unique_units = {
        unit["unit_id"]
        for assignment in assignments.values()
        for unit in assignment.units
    }
    assert len(unique_units) == total_n

    non_overlap_total = total_n - overlap_n
    per_reviewer_base = non_overlap_total // len(reviewers)
    remainder = non_overlap_total % len(reviewers)

    for idx, reviewer in enumerate(reviewers):
        reviewer_id = reviewer["id"]
        count = len(assignments[reviewer_id].units)
        expected = overlap_n + per_reviewer_base + (1 if idx < remainder else 0)
        assert count == expected

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

