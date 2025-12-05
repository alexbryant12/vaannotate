from __future__ import annotations

import json
import sqlite3
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class _Unavailable:  # pragma: no cover - helper for import stubbing
    def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401
        raise RuntimeError("Optional dependency is not available in tests")


if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = _Unavailable
    stub.CrossEncoder = _Unavailable
    sys.modules["sentence_transformers"] = stub

if "langchain_text_splitters" not in sys.modules:
    stub = types.ModuleType("langchain_text_splitters")
    stub.RecursiveCharacterTextSplitter = _Unavailable
    sys.modules["langchain_text_splitters"] = stub

if "langchain" not in sys.modules:
    langchain_stub = types.ModuleType("langchain")
    text_splitter_stub = types.ModuleType("langchain.text_splitter")
    text_splitter_stub.RecursiveCharacterTextSplitter = _Unavailable
    sys.modules["langchain"] = langchain_stub
    sys.modules["langchain.text_splitter"] = text_splitter_stub

from vaannotate.project import add_phenotype, get_connection, init_project
from vaannotate.schema import initialize_assignment_db, initialize_corpus_db
from vaannotate.vaannotate_ai_backend.adapters import export_inputs_from_repo
from vaannotate.vaannotate_ai_backend.engine import DataRepository, _contexts_for_unit_label


def test_export_inputs_uses_storage_path(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_test"
    storage_relative = Path("phenotypes") / "custom_slug"
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Test phenotype",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.commit()

    corpus_dir = project_root / "Corpora" / pheno_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_db = corpus_dir / "corpus.db"
    with initialize_corpus_db(corpus_db) as conn:
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p1",))
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_1", "p1", "2020-01-01", "hash1", "Sample text", "{}"),
        )
        conn.execute("ALTER TABLE documents ADD COLUMN patienticn TEXT")
        conn.execute("UPDATE documents SET patienticn = patient_icn")
        conn.execute("ALTER TABLE documents ADD COLUMN notetype TEXT")
        conn.execute("UPDATE documents SET notetype = ''")
        conn.commit()

    with get_connection(paths.project_db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("ph_test_r1", pheno_id, 1, "ls_test", "hash", 0, "finalized", "2020-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO round_configs(round_id, config_json) VALUES (?, ?)",
            ("ph_test_r1", json.dumps({"corpus_id": "cor_ph_test", "corpus_path": "Corpora/ph_test/corpus.db"})),
        )
        conn.commit()

    round_dir = phenotype_dir / "rounds" / "round_1"
    imports_dir = round_dir / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    assignment_db = imports_dir / "rev1_assignment.db"
    with initialize_assignment_db(assignment_db) as conn:
        conn.execute(
            "INSERT INTO units(unit_id, display_rank, patient_icn, doc_id, note_count) VALUES (?,?,?,?,?)",
            ("unit_1", 1, "p1", "doc_1", 1),
        )
        conn.execute(
            "INSERT INTO documents(doc_id, hash, text, metadata_json) VALUES (?,?,?,?)",
            ("doc_1", "hash1", "Sample text", "{}"),
        )
        conn.execute(
            "INSERT INTO unit_notes(unit_id, doc_id, order_index) VALUES (?,?,?)",
            ("unit_1", "doc_1", 0),
        )
        conn.execute(
            "INSERT INTO annotations(unit_id, label_id, value, value_num, na, notes) VALUES (?,?,?,?,?,?)",
            ("unit_1", "LabelA", "yes", None, 0, "note"),
        )
        conn.commit()

    notes_df, ann_df = export_inputs_from_repo(project_root, pheno_id, [1])

    assert isinstance(notes_df, pd.DataFrame)
    assert isinstance(ann_df, pd.DataFrame)
    assert {"doc_id", "patient_icn", "text"}.issubset(notes_df.columns)
    assert notes_df.loc[0, "doc_id"] == "doc_1"
    assert {"round_id", "label_value", "patient_icn", "labelset_id"}.issubset(ann_df.columns)
    assert ann_df.loc[0, "round_id"] == "ph_test_r1"
    assert ann_df.loc[0, "reviewer_id"] == "rev1"
    assert ann_df.loc[0, "document_text"] == "Sample text"


def test_export_inputs_filters_rounds_by_labelset(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_labelset_filter"
    storage_relative = Path("phenotypes") / pheno_id
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Phenotype",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            ("ph_labelset_filter_r1", pheno_id, 1, "ls_primary", "hash", 0, "finalized", "2020-01-01T00:00:00"),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            ("ph_labelset_filter_r2", pheno_id, 2, "ls_secondary", "hash", 0, "finalized", "2020-01-02T00:00:00"),
        )
        conn.commit()

    corpus_dir = project_root / "Corpora" / pheno_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_db = corpus_dir / "corpus.db"
    with initialize_corpus_db(corpus_db) as conn:
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p1",))
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p2",))
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_primary", "p1", "2020-01-01", "hash1", "Primary text", "{}"),
        )
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_secondary", "p2", "2020-01-02", "hash2", "Secondary text", "{}"),
        )
        conn.commit()

    for round_number, label, doc_id in [(1, "ls_primary", "doc_primary"), (2, "ls_secondary", "doc_secondary")]:
        round_dir = phenotype_dir / "rounds" / f"round_{round_number}"
        imports_dir = round_dir / "imports"
        imports_dir.mkdir(parents=True, exist_ok=True)
        assignment_db = imports_dir / "rev_assignment.db"
        with initialize_assignment_db(assignment_db) as conn:
            conn.execute(
                "INSERT INTO units(unit_id, display_rank, patient_icn, doc_id, note_count) VALUES (?,?,?,?,?)",
                (f"unit_{round_number}", round_number, f"p{round_number}", doc_id, 1),
            )
            conn.execute(
                "INSERT INTO documents(doc_id, hash, text, metadata_json) VALUES (?,?,?,?)",
                (doc_id, f"hash{round_number}", f"{doc_id} text", "{}"),
            )
            conn.execute(
                "INSERT INTO unit_notes(unit_id, doc_id, order_index) VALUES (?,?,?)",
                (f"unit_{round_number}", doc_id, 0),
            )
            conn.execute(
                "INSERT INTO annotations(unit_id, label_id, value, value_num, na, notes) VALUES (?,?,?,?,?,?)",
                (f"unit_{round_number}", "LabelA", "yes", None, 0, "note"),
            )
            conn.commit()

    messages: list[str] = []
    notes_df, ann_df = export_inputs_from_repo(
        project_root,
        pheno_id,
        [1, 2],
        corpus_id=f"cor_{pheno_id}",
        corpus_path=str(corpus_db.relative_to(project_root)),
        labelset_id="ls_primary",
        log_callback=messages.append,
    )

    assert not ann_df.empty
    assert set(ann_df["round_id"]) == {"ph_labelset_filter_r1"}
    assert set(ann_df["labelset_id"]) == {"ls_primary"}
    assert "doc_secondary" not in ann_df.get("doc_id", []).tolist()
    assert any("label set mismatches" in msg for msg in messages)


def test_export_inputs_filters_rounds_before_corpus_lookup(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_labelset_corpus"
    storage_relative = Path("phenotypes") / pheno_id
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    corpus_dir = project_root / "corpora" / pheno_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    primary_corpus = corpus_dir / "primary.db"
    secondary_corpus = corpus_dir / "secondary.db"

    for path, doc_id, patient_icn in [
        (primary_corpus, "doc_primary", "p1"),
        (secondary_corpus, "doc_secondary", "p2"),
    ]:
        with initialize_corpus_db(path) as conn:
            conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", (patient_icn,))
            conn.execute(
                """
                INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
                VALUES (?,?,?,?,?,?)
                """,
                (doc_id, patient_icn, "2020-01-01", "hash", f"{doc_id} text", "{}"),
            )
            conn.execute("ALTER TABLE documents ADD COLUMN patienticn TEXT")
            conn.execute("UPDATE documents SET patienticn = patient_icn")
            conn.execute("ALTER TABLE documents ADD COLUMN notetype TEXT")
            conn.execute("UPDATE documents SET notetype = ''")
            conn.commit()

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Phenotype",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            ("ph_labelset_corpus_r1", pheno_id, 1, "ls_primary", "hash", 0, "finalized", "2020-01-01T00:00:00"),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            ("ph_labelset_corpus_r2", pheno_id, 2, "ls_secondary", "hash", 0, "finalized", "2020-01-02T00:00:00"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO round_configs(round_id, config_json) VALUES (?, ?)",
            (
                "ph_labelset_corpus_r1",
                json.dumps({"corpus_path": str(primary_corpus.relative_to(project_root))}),
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO round_configs(round_id, config_json) VALUES (?, ?)",
            (
                "ph_labelset_corpus_r2",
                json.dumps({"corpus_path": str(secondary_corpus.relative_to(project_root))}),
            ),
        )
        conn.commit()

    round_dir = phenotype_dir / "rounds" / "round_1"
    imports_dir = round_dir / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    assignment_db = imports_dir / "rev_assignment.db"
    with initialize_assignment_db(assignment_db) as conn:
        conn.execute(
            "INSERT INTO units(unit_id, display_rank, patient_icn, doc_id, note_count) VALUES (?,?,?,?,?)",
            ("unit_1", 1, "p1", "doc_primary", 1),
        )
        conn.execute(
            "INSERT INTO documents(doc_id, hash, text, metadata_json) VALUES (?,?,?,?)",
            ("doc_primary", "hash", "doc_primary text", "{}"),
        )
        conn.execute(
            "INSERT INTO unit_notes(unit_id, doc_id, order_index) VALUES (?,?,?)",
            ("unit_1", "doc_primary", 0),
        )
        conn.execute(
            "INSERT INTO annotations(unit_id, label_id, value, value_num, na, notes) VALUES (?,?,?,?,?,?)",
            ("unit_1", "LabelA", "yes", None, 0, "note"),
        )
        conn.commit()

    notes_df, ann_df = export_inputs_from_repo(
        project_root,
        pheno_id,
        [1, 2],
        labelset_id="ls_primary",
    )

    assert set(notes_df["doc_id"]) == {"doc_primary"}
    assert set(ann_df.get("round_id", [])) == {"ph_labelset_corpus_r1"}
    assert "doc_secondary" not in notes_df.get("doc_id", []).tolist()


def test_export_inputs_cold_start_uses_selected_corpus(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_cold"
    storage_relative = Path("phenotypes") / pheno_id
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    corpus_dir = project_root / "corpora" / pheno_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_db = corpus_dir / "corpus.db"
    with initialize_corpus_db(corpus_db) as conn:
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p1",))
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_1", "p1", "2020-01-01", "hash1", "Sample text", "{}"),
        )
        conn.execute("ALTER TABLE documents ADD COLUMN patienticn TEXT")
        conn.execute("UPDATE documents SET patienticn = patient_icn")
        conn.execute("ALTER TABLE documents ADD COLUMN notetype TEXT")
        conn.execute("UPDATE documents SET notetype = ''")
        conn.commit()

    relative_path = Path("corpora") / pheno_id / "corpus.db"

    with get_connection(paths.project_db) as conn:
        conn.row_factory = sqlite3.Row
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Cold phenotype",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.execute(
            """
            INSERT INTO project_corpora(corpus_id, project_id, name, relative_path, created_at)
            VALUES (?,?,?,?,?)
            """,
            ("cor_cold", "proj", "Cold corpus", str(relative_path.as_posix()), datetime.utcnow().isoformat()),
        )
        conn.commit()
        corpus_row = conn.execute(
            "SELECT * FROM project_corpora WHERE corpus_id=?",
            ("cor_cold",),
        ).fetchone()

    notes_df, ann_df = export_inputs_from_repo(
        project_root,
        pheno_id,
        [],
        corpus_record=corpus_row,
        corpus_id="cor_cold",
    )

    assert not notes_df.empty
    assert notes_df.loc[0, "doc_id"] == "doc_1"
    assert ann_df.empty


def test_export_inputs_requires_explicit_corpus(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_explicit"
    storage_relative = Path("phenotypes") / pheno_id
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Phenotype with explicit corpus",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.commit()

    # Create a default corpus that would otherwise be used as a fallback.
    default_corpus = phenotype_dir / "corpus.db"
    with initialize_corpus_db(default_corpus) as conn:
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p1",))
        conn.commit()

    missing_corpus = "Corpora/nonexistent/corpus.db"

    with pytest.raises(FileNotFoundError):
        export_inputs_from_repo(
            project_root,
            pheno_id,
            [],
            corpus_path=missing_corpus,
        )


def test_export_inputs_requires_existing_scoped_csv_when_allowed(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_missing_scoped"
    storage_relative = Path("phenotypes") / pheno_id
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Phenotype missing scoped CSV",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.commit()

    default_corpus = phenotype_dir / "corpus.db"
    with initialize_corpus_db(default_corpus) as conn:
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p1",))
        conn.commit()

    missing_csv = project_root / "scoped_missing.csv"

    with pytest.raises(FileNotFoundError):
        export_inputs_from_repo(
            project_root,
            pheno_id,
            [],
            corpus_path=str(missing_csv),
            allow_scoped_corpus_csv=True,
        )


def test_export_inputs_prefers_scoped_csv_when_allowed(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_scoped_preferred"
    storage_relative = Path("phenotypes") / pheno_id
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Phenotype with scoped CSV",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.commit()

    corpus_db = phenotype_dir / "corpus.db"
    with initialize_corpus_db(corpus_db) as conn:
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p1",))
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_a", "p1", "2020-01-01", "hash1", "Doc A", "{}"),
        )
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_b", "p1", "2020-01-02", "hash2", "Doc B", "{}"),
        )
        conn.commit()

    scoped_csv = project_root / "scoped.csv"
    pd.DataFrame(
        [
            {
                "doc_id": "doc_a",
                "patient_icn": "p1",
                "text": "Doc A scoped",
            }
        ]
    ).to_csv(scoped_csv, index=False)

    notes_df, ann_df = export_inputs_from_repo(
        project_root,
        pheno_id,
        [],
        corpus_path=str(scoped_csv),
        allow_scoped_corpus_csv=True,
    )

    assert list(notes_df["doc_id"]) == ["doc_a"]
    assert list(notes_df["text"]) == ["Doc A scoped"]
    assert ann_df.empty


def test_scoped_corpus_csv_ignored_for_round_builder(tmp_path: Path) -> None:
    project_root = tmp_path / "Project"
    paths = init_project(project_root, "proj", "Project", "tester")

    pheno_id = "ph_no_scoped_csv"
    storage_relative = Path("phenotypes") / pheno_id
    phenotype_dir = project_root / storage_relative
    (phenotype_dir / "rounds").mkdir(parents=True, exist_ok=True)

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id=pheno_id,
            project_id="proj",
            name="Phenotype without scoped CSV",
            level="single_doc",
            storage_path=str(storage_relative.as_posix()),
        )
        conn.commit()

    corpus_db = phenotype_dir / "corpus.db"
    with initialize_corpus_db(corpus_db) as conn:
        conn.execute("INSERT INTO patients(patient_icn) VALUES (?)", ("p1",))
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_a", "p1", "2020-01-01", "hash1", "Doc A", "{}"),
        )
        conn.execute(
            """
            INSERT INTO documents(doc_id, patient_icn, date_note, hash, text, metadata_json)
            VALUES (?,?,?,?,?,?)
            """,
            ("doc_b", "p1", "2020-01-02", "hash2", "Doc B", "{}"),
        )
        conn.execute("ALTER TABLE documents ADD COLUMN patienticn TEXT")
        conn.execute("UPDATE documents SET patienticn = patient_icn")
        conn.execute("ALTER TABLE documents ADD COLUMN notetype TEXT")
        conn.execute("UPDATE documents SET notetype = ''")
        conn.commit()

    scoped_csv = project_root / "scoped.csv"
    pd.DataFrame(
        [
            {
                "doc_id": "doc_a",
                "patient_icn": "p1",
                "text": "Doc A scoped",
            }
        ]
    ).to_csv(scoped_csv, index=False)

    notes_df, ann_df = export_inputs_from_repo(
        project_root,
        pheno_id,
        [],
        corpus_path=str(scoped_csv),
    )

    assert set(notes_df["doc_id"]) == {"doc_a", "doc_b"}
    assert ann_df.empty


def test_contexts_for_unit_label_full_mode() -> None:
    notes_df = pd.DataFrame(
        [
            {
                "patient_icn": "p1",
                "doc_id": "doc_full",
                "text": "Full document context text",
                "unit_id": "assign-1",
            },
        ]
    )
    ann_df = pd.DataFrame(
        {
            "round_id": pd.Series(dtype="object"),
            "unit_id": pd.Series(dtype="object"),
            "doc_id": pd.Series(dtype="object"),
            "label_id": pd.Series(dtype="object"),
            "reviewer_id": pd.Series(dtype="object"),
            "label_value": pd.Series(dtype="object"),
        }
    )

    repo = DataRepository(notes_df, ann_df, phenotype_level="single_doc")

    class DummyStore:
        def __init__(self) -> None:
            self.chunk_meta = [
                {"unit_id": "doc_full", "doc_id": "doc_full", "chunk_id": "0", "text": "chunk"}
            ]

        def get_patient_chunk_indices(self, unit_id: str) -> list[int]:
            if str(unit_id) == "doc_full":
                return [0]
            return []

    class DummyRetriever:
        def __init__(self) -> None:
            self.store = DummyStore()
            self.called = False

        def _extract_meta(self, meta: dict) -> dict:
            return {"note_type": "demo", "other_meta": "meta"}

        def retrieve_for_patient_label(self, *args: object, **kwargs: object) -> list[dict]:
            self.called = True
            return [{"text": "fallback"}]

    retriever = DummyRetriever()

    contexts = _contexts_for_unit_label(
        retriever,
        repo,
        "doc_full",
        "label1",
        "",
        single_doc_context_mode="full",
        full_doc_char_limit=50,
    )

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["source"] == "full_doc"
    assert ctx["doc_id"] == "doc_full"
    assert ctx["text"] == "Full document context text"[:50]
    assert ctx["metadata"].get("other_meta") == "meta"
    assert not retriever.called


def test_contexts_for_unit_label_maps_assignment_ids() -> None:
    notes_df = pd.DataFrame(
        [
            {
                "patient_icn": "p1",
                "doc_id": "doc_full",
                "text": "Some text",
                "unit_id": "assign-1",
            }
        ]
    )
    ann_df = pd.DataFrame(
        [
            {
                "round_id": "r1",
                "unit_id": "assign-1",
                "doc_id": "doc_full",
                "label_id": "lab",
                "reviewer_id": "rev",
                "label_value": "",
            }
        ]
    )
    repo = DataRepository(notes_df, ann_df, phenotype_level="single_doc")

    class DummyStore:
        def get_patient_chunk_indices(self, unit_id: str) -> list[int]:
            assert unit_id == "doc_full"
            return []

        chunk_meta: list[dict] = []

    class DummyRetriever:
        def __init__(self) -> None:
            self.store = DummyStore()
            self.calls: list[str] = []

        def _extract_meta(self, meta: dict) -> dict:
            return {}

        def retrieve_for_patient_label(self, unit_id: str, *args: object, **kwargs: object) -> list[dict]:
            self.calls.append(unit_id)
            return [{"text": "rag"}]

    retriever = DummyRetriever()

    contexts = _contexts_for_unit_label(
        retriever,
        repo,
        "assign-1",
        "lab",
        "",
        single_doc_context_mode="rag",
    )

    assert contexts == [{"text": "rag"}]
    assert retriever.calls == ["doc_full"]

