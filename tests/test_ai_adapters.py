from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pandas as pd

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
from vaannotate.schema import initialize_corpus_db
from vaannotate.vaannotate_ai_backend.adapters import export_inputs_from_repo


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
    round_dir.mkdir(parents=True, exist_ok=True)
    round_db = round_dir / "round_aggregate.db"
    with sqlite3.connect(round_db) as conn:
        conn.execute(
            """
            CREATE TABLE annotations(
                round_id TEXT,
                unit_id TEXT,
                doc_id TEXT,
                label_id TEXT,
                reviewer_id TEXT,
                label_value TEXT,
                reviewer_notes TEXT,
                rationales_json TEXT,
                document_text TEXT,
                document_metadata_json TEXT,
                label_rules TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO annotations(round_id, unit_id, doc_id, label_id, reviewer_id, label_value,
                                     reviewer_notes, rationales_json, document_text, document_metadata_json, label_rules)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "ph_test_r1",
                "unit_1",
                "doc_1",
                "LabelA",
                "rev1",
                "yes",
                "note",
                "[]",
                "Sample text",
                "{}",
                "",
            ),
        )
        conn.commit()

    notes_df, ann_df = export_inputs_from_repo(project_root, pheno_id, [1])

    assert isinstance(notes_df, pd.DataFrame)
    assert isinstance(ann_df, pd.DataFrame)
    assert {"doc_id", "patient_icn", "text"}.issubset(notes_df.columns)
    assert notes_df.loc[0, "doc_id"] == "doc_1"
    assert {"round_id", "label_value", "patient_icn"}.issubset(ann_df.columns)
    assert ann_df.loc[0, "round_id"] == "ph_test_r1"

