"""Tests for reviewer assignment import and aggregation pipeline."""

from __future__ import annotations

import csv
import os
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
    add_project_corpus,
    fetch_labelset,
    get_connection,
    init_project,
    register_reviewer,
)
from vaannotate.rounds import AssignmentUnit, CandidateUnit, RoundBuilder
from vaannotate.schema import initialize_corpus_db
from vaannotate.shared.database import Database
from vaannotate.shared.sampling import (
    SamplingFilters,
    allocate_units,
    candidate_documents,
    initialize_assignment_db,
    populate_assignment_db,
)
from vaannotate.shared.metadata import MetadataFilterCondition


@pytest.fixture()
def seeded_project(tmp_path: Path) -> tuple[RoundBuilder, Path]:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    with get_connection(paths.project_db) as conn:
        register_reviewer(conn, "rev_one", "Reviewer One")
        register_reviewer(conn, "rev_two", "Reviewer Two")
        add_project_corpus(
            conn,
            corpus_id="cor_ph_test",
            project_id="proj",
            name="Test corpus",
            relative_path="corpora/ph_test/corpus.db",
        )
        add_phenotype(
            conn,
            pheno_id="ph_test",
            project_id="proj",
            name="Test phenotype",
            level="single_doc",
            storage_path="phenotypes/ph_test",
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

    corpus_db = project_root / "corpora" / "ph_test" / "corpus.db"
    corpus_db.parent.mkdir(parents=True, exist_ok=True)
    with initialize_corpus_db(corpus_db) as corpus_conn:
        for idx in range(3):
            patient_id = f"p{idx}"
            corpus_conn.execute(
                "INSERT INTO patients(patient_icn) VALUES (?)",
                (patient_id,),
            )
            metadata = json.dumps(
                {
                    "notetype": "NOTE",
                    "note_year": 2020,
                    "sta3n": "506",
                },
                sort_keys=True,
            )
            corpus_conn.execute(
                """
                INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    f"doc_{idx}",
                    patient_id,
                    "2020-01-01",
                    f"hash{idx}",
                    f"Example text {idx}",
                    metadata,
                ),
            )
        corpus_conn.commit()

    config_dir = paths.admin_dir / "round_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "round_number": 1,
        "round_id": "ph_test_r1",
        "labelset_id": "ls_test",
        "corpus_id": "cor_ph_test",
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


def test_generate_round_with_preselected_csv(seeded_project: tuple[RoundBuilder, Path], tmp_path: Path) -> None:
    builder, _ = seeded_project
    config = {
        "round_number": 2,
        "round_id": "ph_test_r2",
        "labelset_id": "ls_test",
        "corpus_id": "cor_ph_test",
        "reviewers": [
            {"id": "rev_one", "name": "Reviewer One"},
            {"id": "rev_two", "name": "Reviewer Two"},
        ],
        "overlap_n": 0,
        "rng_seed": 321,
        "total_n": 2,
        "status": "draft",
    }
    config_path = tmp_path / "ph_test_r2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    csv_path = tmp_path / "ai_next_batch.csv"
    csv_path.write_text(
        "unit_id,doc_id,patient_icn,selection_reason\n"
        "doc_0,doc_0,p0,seeded\n"
        "doc_2,doc_2,p2,seeded\n",
        encoding="utf-8",
    )

    result = builder.generate_round(
        "ph_test",
        config_path,
        created_by="tester",
        preselected_units_csv=csv_path,
    )

    round_dir = Path(result["round_dir"])
    manifest_path = round_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    unit_ids = [row["unit_id"] for row in rows]
    assert unit_ids == ["doc_0", "doc_2"]
    stored_config = json.loads((round_dir / "round_config.json").read_text("utf-8"))
    assert stored_config.get("preselected_units_csv") == str(csv_path)


def test_generate_round_applies_env_overrides(
    seeded_project: tuple[RoundBuilder, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, _ = seeded_project
    config = {
        "round_number": 3,
        "round_id": "ph_test_r3",
        "labelset_id": "ls_test",
        "corpus_id": "cor_ph_test",
        "reviewers": [
            {"id": "rev_one", "name": "Reviewer One"},
            {"id": "rev_two", "name": "Reviewer Two"},
        ],
        "overlap_n": 0,
        "rng_seed": 42,
        "assisted_review": {"enabled": True, "top_snippets": 1},
    }
    config_path = tmp_path / "ph_test_r3.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    captured: dict[str, str | None] = {}

    def _fake_assisted(*args, **kwargs):
        captured["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        return {}

    monkeypatch.setattr(RoundBuilder, "_generate_assisted_review_snippets", _fake_assisted)

    assert os.getenv("AZURE_OPENAI_API_KEY") is None

    builder.generate_round(
        "ph_test",
        config_path,
        created_by="tester",
        env_overrides={"AZURE_OPENAI_API_KEY": "test-key"},
    )

    assert captured.get("api_key") == "test-key"
    assert os.getenv("AZURE_OPENAI_API_KEY") is None
def test_multi_doc_round_uses_patient_display_unit(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    with get_connection(paths.project_db) as conn:
        register_reviewer(conn, "rev_one", "Reviewer One")
        register_reviewer(conn, "rev_two", "Reviewer Two")
        add_project_corpus(
            conn,
            corpus_id="cor_ph_multi",
            project_id="proj",
            name="Multi corpus",
            relative_path="corpora/ph_multi/corpus.db",
        )
        add_phenotype(
            conn,
            pheno_id="ph_multi",
            project_id="proj",
            name="Multi Doc",
            level="multi_doc",
            storage_path="phenotypes/ph_multi",
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
    corpus_db = project_root / "corpora" / "ph_multi" / "corpus.db"
    corpus_db.parent.mkdir(parents=True, exist_ok=True)
    with initialize_corpus_db(corpus_db) as corpus_conn:
        for idx, (patient_icn, docs) in enumerate(docs_by_patient.items()):
            corpus_conn.execute(
                "INSERT INTO patients(patient_icn) VALUES (?)",
                (patient_icn,),
            )
            for doc_index, (doc_id, text) in enumerate(docs):
                metadata = json.dumps(
                    {
                        "notetype": "NOTE",
                        "note_year": 2020 + doc_index,
                        "sta3n": "506",
                    },
                    sort_keys=True,
                )
                corpus_conn.execute(
                    """
                    INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        doc_id,
                        patient_icn,
                        f"202{doc_index}-01-01",
                        f"hash_{doc_id}",
                        text,
                        metadata,
                    ),
                )
        corpus_conn.commit()

    config_dir = paths.admin_dir / "round_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "round_number": 1,
        "round_id": "ph_multi_r1",
        "labelset_id": "ls_multi",
        "corpus_id": "cor_ph_multi",
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


def test_generate_round_with_preselected_multi_doc(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    with get_connection(paths.project_db) as conn:
        register_reviewer(conn, "rev_one", "Reviewer One")
        add_project_corpus(
            conn,
            corpus_id="cor_multi",
            project_id="proj",
            name="Multi corpus",
            relative_path="corpora/multi/corpus.db",
        )
        add_phenotype(
            conn,
            pheno_id="ph_multi",
            project_id="proj",
            name="Multi phenotype",
            level="multi_doc",
            storage_path="phenotypes/ph_multi",
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
                    "type": "boolean",
                    "required": False,
                    "options": [
                        {"value": "yes", "display": "Yes"},
                        {"value": "no", "display": "No"},
                    ],
                }
            ],
        )
        conn.commit()

    corpus_db = project_root / "corpora" / "multi" / "corpus.db"
    corpus_db.parent.mkdir(parents=True, exist_ok=True)
    with initialize_corpus_db(corpus_db) as corpus_conn:
        corpus_conn.execute(
            "INSERT INTO patients(patient_icn) VALUES (?)",
            ("multi_p0",),
        )
        for idx in range(2):
            corpus_conn.execute(
                """
                INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    f"doc_m_{idx}",
                    "multi_p0",
                    f"2020-01-0{idx+1}",
                    f"hash{idx}",
                    f"Example text {idx}",
                    json.dumps({"note_idx": idx}, sort_keys=True),
                ),
            )
        corpus_conn.commit()

    builder = RoundBuilder(project_root)
    config = {
        "round_number": 1,
        "labelset_id": "ls_multi",
        "corpus_id": "cor_multi",
        "reviewers": [{"id": "rev_one", "name": "Reviewer One"}],
        "overlap_n": 0,
        "rng_seed": 5,
        "total_n": 1,
    }
    config_path = tmp_path / "ph_multi_round.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    csv_path = tmp_path / "ai_multi.csv"
    csv_path.write_text(
        "unit_id,doc_id,patient_icn\n"
        "multi_p0,doc_m_0,multi_p0\n"
        "multi_p0,doc_m_1,multi_p0\n",
        encoding="utf-8",
    )

    result = builder.generate_round(
        "ph_multi",
        config_path,
        created_by="tester",
        preselected_units_csv=csv_path,
    )

    round_dir = Path(result["round_dir"])
    assignment_db = round_dir / "assignments" / "rev_one" / "assignment.db"
    with sqlite3.connect(assignment_db) as conn:
        conn.row_factory = sqlite3.Row
        note_count = conn.execute(
            "SELECT note_count FROM units WHERE unit_id=?",
            ("multi_p0",),
        ).fetchone()
        assert note_count is not None and note_count["note_count"] == 2
        doc_rows = conn.execute(
            "SELECT doc_id FROM unit_notes WHERE unit_id=? ORDER BY order_index",
            ("multi_p0",),
        ).fetchall()
        assert [row["doc_id"] for row in doc_rows] == ["doc_m_0", "doc_m_1"]


def test_round_builder_metadata_filters_single_doc(seeded_project: tuple[RoundBuilder, Path]) -> None:
    builder, project_root = seeded_project
    corpus_path = project_root / "corpora" / "ph_test" / "corpus.db"
    with sqlite3.connect(corpus_path) as conn:
        row = conn.execute("SELECT metadata_json FROM documents WHERE doc_id='doc_2'").fetchone()
        metadata = json.loads(row[0] or "{}") if row else {}
        metadata["notetype"] = "OTHER"
        conn.execute(
            "UPDATE documents SET metadata_json=? WHERE doc_id='doc_2'",
            (json.dumps(metadata, sort_keys=True),),
        )
        conn.commit()

    config_dir = project_root / "admin_tools" / "round_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "ph_test_r2.json"
    config = {
        "round_number": 2,
        "round_id": "ph_test_r2",
        "labelset_id": "ls_test",
        "corpus_id": "cor_ph_test",
        "reviewers": [
            {"id": "rev_one", "name": "Reviewer One"},
            {"id": "rev_two", "name": "Reviewer Two"},
        ],
        "overlap_n": 0,
        "rng_seed": 999,
        "filters": {
            "metadata": [
                {
                    "field": "metadata.notetype",
                    "label": "Notetype",
                    "scope": "document",
                    "type": "text",
                    "values": ["OTHER"],
                }
            ]
        },
        "stratification": {"fields": ["metadata.sta3n"]},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    builder.generate_round("ph_test", config_path, created_by="tester")

    round_dir = project_root / "phenotypes" / "ph_test" / "rounds" / "round_2"
    manifest_path = round_dir / "manifest.csv"
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    doc_ids = {row["doc_id"] for row in rows}
    assert doc_ids == {"doc_2"}
    strata_keys = {row["strata_key"] for row in rows}
    assert strata_keys == {"506"}


def test_candidate_documents_match_any_filters(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.db"
    with initialize_corpus_db(corpus_path) as conn:
        conn.execute(
            "INSERT INTO patients(patient_icn) VALUES (?)",
            ("p_any",),
        )
        metadata_alpha = json.dumps(
            {"notetype": "ALPHA", "note_year": 2020, "sta3n": "506"}, sort_keys=True
        )
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_alpha", "p_any", "2020-01-01", "hash_a", "Alpha text", metadata_alpha),
        )
        metadata_beta = json.dumps(
            {"notetype": "BETA", "note_year": 2022, "sta3n": "506"}, sort_keys=True
        )
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_beta", "p_any", "2022-06-15", "hash_b", "Beta text", metadata_beta),
        )
        conn.commit()

    filters = SamplingFilters(
        metadata_filters=[
            MetadataFilterCondition(
                field="metadata.notetype",
                label="Notetype",
                scope="document",
                data_type="text",
                values=["ALPHA"],
            ),
            MetadataFilterCondition(
                field="metadata.note_year",
                label="Note year",
                scope="document",
                data_type="number",
                min_value="2022",
            ),
        ],
        match_any=True,
    )

    rows = candidate_documents(Database(corpus_path), "single_doc", filters)
    doc_ids = {row["doc_id"] for row in rows}
    assert doc_ids == {"doc_alpha", "doc_beta"}


def test_admin_sampling_creates_multi_doc_units(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.db"
    docs_by_patient: dict[str, list[tuple[str, str]]] = {
        "p1": [("p1_doc1", "P1 doc one"), ("p1_doc2", "P1 doc two")],
        "p2": [("p2_doc1", "P2 doc one"), ("p2_doc2", "P2 doc two")],
    }
    with initialize_corpus_db(corpus_path) as corpus_conn:
        for patient_icn, docs in docs_by_patient.items():
            corpus_conn.execute(
                "INSERT INTO patients(patient_icn) VALUES (?)",
                (patient_icn,),
            )
            for order_idx, (doc_id, text) in enumerate(docs):
                metadata = json.dumps(
                    {
                        "notetype": "NOTE",
                        "note_year": 2020 + order_idx,
                        "sta3n": "506",
                    },
                    sort_keys=True,
                )
                corpus_conn.execute(
                    """
                    INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        doc_id,
                        patient_icn,
                        f"202{order_idx}-01-01",
                        f"hash_{doc_id}",
                        text,
                        metadata,
                    ),
                )
        corpus_conn.commit()

    corpus_db = Database(corpus_path)
    filters = SamplingFilters(metadata_filters=[])
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


def test_round_builder_allows_document_stratification_for_multi_doc(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    with get_connection(paths.project_db) as conn:
        register_reviewer(conn, "rev_one", "Reviewer One")
        add_project_corpus(
            conn,
            corpus_id="cor_ph_multi",
            project_id="proj",
            name="Multi corpus",
            relative_path="corpora/ph_multi/corpus.db",
        )
        add_phenotype(
            conn,
            pheno_id="ph_multi",
            project_id="proj",
            name="Multi Doc",
            level="multi_doc",
            storage_path="phenotypes/ph_multi",
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
                    "options": [],
                }
            ],
        )
        conn.commit()

    docs_by_patient = {
        "p1": [("p1_doc1", "P1 doc one"), ("p1_doc2", "P1 doc two")],
    }
    corpus_db = project_root / "corpora" / "ph_multi" / "corpus.db"
    corpus_db.parent.mkdir(parents=True, exist_ok=True)
    with initialize_corpus_db(corpus_db) as corpus_conn:
        for patient_icn, docs in docs_by_patient.items():
            corpus_conn.execute(
                "INSERT INTO patients(patient_icn) VALUES (?)",
                (patient_icn,),
            )
            for idx, (doc_id, text) in enumerate(docs):
                metadata = json.dumps(
                    {
                        "notetype": "NOTE",
                        "note_year": 2020 + idx,
                        "sta3n": "506",
                    },
                    sort_keys=True,
                )
                corpus_conn.execute(
                    """
                    INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        doc_id,
                        patient_icn,
                        f"202{idx}-01-01",
                        f"hash_{doc_id}",
                        text,
                        metadata,
                    ),
                )
        corpus_conn.commit()

    builder = RoundBuilder(project_root)
    config_dir = paths.admin_dir / "round_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "ph_multi_r_invalid.json"
    config = {
        "round_number": 1,
        "round_id": "ph_multi_r_invalid",
        "labelset_id": "ls_multi",
        "corpus_id": "cor_ph_multi",
        "reviewers": [{"id": "rev_one", "name": "Reviewer One"}],
        "overlap_n": 0,
        "rng_seed": 1,
        "filters": {"metadata": []},
        "stratification": {"fields": ["metadata.note_year"]},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    builder.generate_round("ph_multi", config_path, created_by="tester")

    manifest_path = (
        project_root
        / "phenotypes"
        / "ph_multi"
        / "rounds"
        / "round_1"
        / "manifest.csv"
    )
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["strata_key"] for row in rows] == ["2020"]


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


def test_allocate_units_stratified_overlap_matches_target() -> None:
    rows = []
    for strata in range(3):
        for idx in range(5):
            rows.append(
                {
                    "unit_id": f"unit_{strata}_{idx}",
                    "patient_icn": f"pat_{strata}_{idx}",
                    "doc_id": f"doc_{strata}_{idx}",
                    "strata": strata,
                }
            )
    reviewers = [
        {"id": "rev_a", "name": "Reviewer A"},
        {"id": "rev_b", "name": "Reviewer B"},
    ]

    overlap_n = 4
    assignments = allocate_units(
        rows,
        reviewers,
        overlap_n=overlap_n,
        seed=13,
        strat_keys=["strata"],
    )

    for reviewer in reviewers:
        reviewer_id = reviewer["id"]
        overlap_units = [
            unit
            for unit in assignments[reviewer_id].units
            if unit.get("is_overlap")
        ]
        assert len(overlap_units) == overlap_n


def test_round_builder_overlap_respects_target_total() -> None:
    builder = RoundBuilder(Path("/tmp"))
    candidates: list[CandidateUnit] = []
    for strata in range(3):
        for idx in range(4):
            candidates.append(
                CandidateUnit(
                    unit_id=f"unit_{strata}_{idx}",
                    patient_icn=f"pat_{strata}_{idx}",
                    doc_id=f"doc_{strata}_{idx}",
                    strata_key=str(strata),
                    payload={"strata_key": str(strata)},
                )
            )
    reviewers = ["rev_a", "rev_b", "rev_c"]

    assignments = builder._allocate_units(
        candidates,
        reviewers,
        rng_seed=7,
        overlap_n=5,
        total_n=None,
        strat_config={"fields": ["strata"]},
    )

    overlap_counts = [
        sum(1 for unit in units if unit.payload.get("is_overlap"))
        for units in assignments.values()
    ]
    assert overlap_counts == [5, 5, 5]

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


def test_run_final_llm_labeling_forwards_logs(
    seeded_project: tuple[RoundBuilder, Path],
    monkeypatch,
) -> None:
    builder, project_root = seeded_project
    round_dir = project_root / "phenotypes" / "ph_test" / "rounds" / "round_test"
    round_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(project_root / "project.db") as conn:
        conn.row_factory = sqlite3.Row
        pheno_row = conn.execute(
            "SELECT * FROM phenotypes WHERE pheno_id=?",
            ("ph_test",),
        ).fetchone()
        assert pheno_row is not None
        labelset = fetch_labelset(conn, "ls_test")

    assignments = {
        "rev_one": [
            AssignmentUnit(
                unit_id="doc_0",
                patient_icn="p0",
                doc_id="doc_0",
                payload={
                    "unit_id": "doc_0",
                    "patient_icn": "p0",
                    "doc_id": "doc_0",
                    "documents": [{"doc_id": "doc_0", "text": "Example text 0"}],
                    "strata_key": "random_sampling",
                },
            )
        ]
    }

    messages: list[str] = []

    def _fake_apply(self, **kwargs):  # type: ignore[no-untyped-def]
        from vaannotate.vaannotate_ai_backend import engine

        engine.LOGGER.info("[FinalLLM] 1/1 complete")
        return {"final_llm_labels": "dummy"}

    monkeypatch.setattr(RoundBuilder, "_apply_final_llm_labeling", _fake_apply)

    outputs = builder.run_final_llm_labeling(
        pheno_row=pheno_row,
        labelset=labelset,
        round_dir=round_dir,
        reviewer_assignments=assignments,
        config={"ai_backend": {}, "final_llm_labeling": True},
        config_base=round_dir,
        log_callback=messages.append,
    )

    assert outputs == {"final_llm_labels": "dummy"}
    assert any("[FinalLLM]" in message for message in messages)


def test_run_final_llm_labeling_applies_env_overrides(
    seeded_project: tuple[RoundBuilder, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, project_root = seeded_project
    round_dir = project_root / "phenotypes" / "ph_test" / "rounds" / "round_env"
    round_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(project_root / "project.db") as conn:
        conn.row_factory = sqlite3.Row
        pheno_row = conn.execute(
            "SELECT * FROM phenotypes WHERE pheno_id=?",
            ("ph_test",),
        ).fetchone()
        assert pheno_row is not None
        labelset = fetch_labelset(conn, "ls_test")

    assignments = {
        "rev_one": [
            AssignmentUnit(
                unit_id="doc_0",
                patient_icn="p0",
                doc_id="doc_0",
                payload={
                    "unit_id": "doc_0",
                    "patient_icn": "p0",
                    "doc_id": "doc_0",
                    "documents": [{"doc_id": "doc_0", "text": "Example text 0"}],
                    "strata_key": "random_sampling",
                },
            )
        ]
    }

    captured: dict[str, str | None] = {}

    def _fake_apply(self, **kwargs):  # type: ignore[no-untyped-def]
        captured["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        captured["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION")
        return {}

    monkeypatch.setattr(RoundBuilder, "_apply_final_llm_labeling", _fake_apply)

    assert os.getenv("AZURE_OPENAI_API_KEY") is None
    assert os.getenv("AZURE_OPENAI_API_VERSION") is None

    builder.run_final_llm_labeling(
        pheno_row=pheno_row,
        labelset=labelset,
        round_dir=round_dir,
        reviewer_assignments=assignments,
        config={"ai_backend": {}, "final_llm_labeling": True},
        config_base=round_dir,
        env_overrides={
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_API_VERSION": "2024-05-01",
        },
    )

    assert captured["api_key"] == "env-key"
    assert captured["api_version"] == "2024-05-01"
    assert os.getenv("AZURE_OPENAI_API_KEY") is None
    assert os.getenv("AZURE_OPENAI_API_VERSION") is None

