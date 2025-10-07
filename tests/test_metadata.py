import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.corpus import import_tabular_corpus
from vaannotate.schema import initialize_corpus_db
from vaannotate.shared.database import Database
from vaannotate.shared.metadata import (
    MetadataFilterCondition,
    discover_corpus_metadata,
    normalize_date_value,
)
from vaannotate.shared.sampling import SamplingFilters, candidate_documents


def test_discover_metadata_identifies_dates(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.db"
    with initialize_corpus_db(corpus_path) as conn:
        conn.execute("ALTER TABLE documents ADD COLUMN custom_date TEXT")
        conn.execute(
            "INSERT INTO patients(patient_icn, sta3n, date_index, softlabel) VALUES (?,?,?,?)",
            ("p1", "506", None, None),
        )
        conn.execute(
            """
            INSERT INTO documents(
                doc_id, patient_icn, notetype, note_year, date_note,
                cptname, sta3n, hash, text, custom_date
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "doc1",
                "p1",
                "NOTE",
                2024,
                "2024-02-15",
                "",
                "506",
                "hash1",
                "Example text",
                "02/15/2024",
            ),
        )
        conn.commit()

    with initialize_corpus_db(corpus_path) as conn:
        conn.row_factory = sqlite3.Row
        fields = discover_corpus_metadata(conn)

    assert any(field.key == "document.date_note" and field.data_type == "date" for field in fields)
    assert any(field.key == "document.custom_date" and field.data_type == "date" for field in fields)


def test_discover_metadata_ignores_empty_columns(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.db"
    with initialize_corpus_db(corpus_path) as conn:
        conn.execute(
            "INSERT INTO patients(patient_icn, sta3n, date_index, softlabel) VALUES (?,?,?,?)",
            ("p_empty", "506", None, None),
        )
        conn.execute(
            """
            INSERT INTO documents(
                doc_id, patient_icn, notetype, note_year, date_note,
                cptname, sta3n, hash, text
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                "doc_empty",
                "p_empty",
                "NOTE",
                2024,
                "2024-03-01",
                None,
                "506",
                "hash-empty",
                "Example text",
            ),
        )
        conn.commit()

    with initialize_corpus_db(corpus_path) as conn:
        conn.row_factory = sqlite3.Row
        fields = discover_corpus_metadata(conn)

    keys = {field.key for field in fields}
    assert "document.cptname" not in keys
    assert "patient.softlabel" not in keys


def test_candidate_documents_supports_date_range_filters(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.db"
    with initialize_corpus_db(corpus_path) as conn:
        conn.execute(
            "INSERT INTO patients(patient_icn, sta3n, date_index, softlabel) VALUES (?,?,?,?)",
            ("p2", "506", None, None),
        )
        conn.execute(
            """
            INSERT INTO documents(
                doc_id, patient_icn, notetype, note_year, date_note,
                cptname, sta3n, hash, text
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                "doc_range",
                "p2",
                "NOTE",
                2024,
                "2024-02-15",
                "",
                "506",
                "hash_range",
                "Range test",
            ),
        )
        conn.commit()

    filters = SamplingFilters(
        metadata_filters=[
            MetadataFilterCondition(
                field="document.date_note",
                label="Date note",
                scope="document",
                data_type="date",
                min_value="02/01/2024",
                max_value="02/20/2024",
            )
        ]
    )

    rows = candidate_documents(Database(corpus_path), "single_doc", filters)
    assert [row["doc_id"] for row in rows] == ["doc_range"]


def test_import_tabular_corpus_preserves_custom_columns(tmp_path: Path) -> None:
    source = tmp_path / "notes.csv"
    source.write_text("patienticn,text,risk_score\n123,Example note,high\n", encoding="utf-8")
    corpus_path = tmp_path / "corpus.db"

    import_tabular_corpus(source, corpus_path)

    with sqlite3.connect(corpus_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT metadata_json FROM documents").fetchone()
        assert row is not None
        metadata = json.loads(row["metadata_json"])
        assert metadata == {"risk_score": "high"}
        fields = discover_corpus_metadata(conn)

    assert any(field.key == "metadata.risk_score" for field in fields)
    risk_field = next(field for field in fields if field.key == "metadata.risk_score")
    assert risk_field.label == "risk_score"


def test_normalize_date_value_formats() -> None:
    assert normalize_date_value("02/01/2024") == "2024-02-01"
    assert normalize_date_value("2024-02-01") == "2024-02-01"
    assert normalize_date_value("  ") is None
