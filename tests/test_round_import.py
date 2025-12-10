"""Tests for reviewer assignment import and aggregation pipeline."""

from __future__ import annotations

import csv
import os
import json
import sqlite3
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

def _install_pyside_stub() -> None:
    qt_module = types.ModuleType("PySide6")
    qtcore = types.ModuleType("PySide6.QtCore")
    qtgui = types.ModuleType("PySide6.QtGui")
    qtwidgets = types.ModuleType("PySide6.QtWidgets")

    class _StubSignal:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def connect(self, *args, **kwargs) -> None:
            pass

        def emit(self, *args, **kwargs) -> None:
            pass

    class _StubQObject:
        def __init__(self, *args, **kwargs) -> None:
            pass

    qtcore.Signal = lambda *args, **kwargs: _StubSignal()
    qtcore.Slot = lambda *args, **kwargs: (lambda func: func)
    qtcore.QObject = _StubQObject
    qtcore.QThread = _StubQObject
    qtcore.QTimer = _StubQObject
    qtcore.QEventLoop = type(
        "QEventLoop",
        (),
        {"processEvents": staticmethod(lambda *args, **kwargs: None)},
    )

    qtcore.QDateTime = type(
        "QDateTime",
        (),
        {
            "currentDateTimeUtc": staticmethod(
                lambda: types.SimpleNamespace(toString=lambda *args, **kwargs: "")
            )
        },
    )

    qtcore.Qt = types.SimpleNamespace(
        AlignmentFlag=type(
            "AlignmentFlag",
            (),
            {
                "AlignLeft": 0,
                "AlignRight": 0,
                "AlignHCenter": 0,
                "AlignVCenter": 0,
            },
        ),
        Orientation=types.SimpleNamespace(Horizontal=0, Vertical=1),
        SortOrder=type(
            "SortOrder", (), {"AscendingOrder": 0, "DescendingOrder": 1}
        ),
        ItemDataRole=types.SimpleNamespace(UserRole=0, DisplayRole=0),
        ItemFlag=types.SimpleNamespace(
            NoItemFlags=0,
            ItemIsEnabled=0,
            ItemIsSelectable=0,
        ),
        MatchFlag=type("MatchFlag", (), {"MatchExactly": 0}),
    )

    class _StubWidget(_StubQObject):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    def _stub_class(name: str) -> type:
        return type(name, (_StubWidget,), {})

    for cls_name in [
        "QDialog",
        "QWidget",
        "QTreeWidget",
        "QTreeWidgetItem",
        "QComboBox",
        "QLineEdit",
        "QLabel",
        "QCheckBox",
        "QPushButton",
        "QSpinBox",
        "QDoubleSpinBox",
        "QPlainTextEdit",
        "QDialogButtonBox",
        "QGroupBox",
        "QFormLayout",
        "QHBoxLayout",
        "QVBoxLayout",
        "QGridLayout",
        "QListWidget",
        "QListWidgetItem",
        "QStackedWidget",
        "QProgressBar",
        "QAction",
        "QMenu",
        "QMenuBar",
        "QToolBar",
        "QStatusBar",
        "QMainWindow",
        "QTabWidget",
        "QRadioButton",
        "QTableWidget",
        "QTableWidgetItem",
        "QHeaderView",
    ]:
        setattr(qtwidgets, cls_name, _stub_class(cls_name))

    qtwidgets.QApplication = type(
        "QApplication",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "exec": lambda self: 0,
            "instance": staticmethod(lambda: None),
            "quit": staticmethod(lambda: None),
        },
    )

    qtwidgets.QFileDialog = type(
        "QFileDialog",
        (),
        {"getExistingDirectory": staticmethod(lambda *args, **kwargs: "")},
    )

    qtwidgets.QMessageBox = type(
        "QMessageBox",
        (),
        {
            "information": staticmethod(lambda *args, **kwargs: None),
            "warning": staticmethod(lambda *args, **kwargs: None),
            "critical": staticmethod(lambda *args, **kwargs: None),
        },
    )

    qtgui.QCloseEvent = _StubWidget

    qt_module.QtCore = qtcore
    qt_module.QtGui = qtgui
    qt_module.QtWidgets = qtwidgets

    sys.modules["PySide6"] = qt_module
    sys.modules["PySide6.QtCore"] = qtcore
    sys.modules["PySide6.QtGui"] = qtgui
    sys.modules["PySide6.QtWidgets"] = qtwidgets


try:  # Fall back to a stub if Qt bindings are not importable in the test environment
    from PySide6 import QtCore as _QtCore, QtGui as _QtGui, QtWidgets as _QtWidgets  # type: ignore
except Exception:
    for mod_name in ["PySide6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"]:
        sys.modules.pop(mod_name, None)
    _install_pyside_stub()

import vaannotate.AdminApp.main as admin_main
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
    ReviewerAssignment,
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
    monkeypatch.setattr(
        RoundBuilder,
        "_normalize_for_json",
        lambda self, value: value,
    )

    assert os.getenv("AZURE_OPENAI_API_KEY") is None

    builder.generate_round(
        "ph_test",
        config_path,
        created_by="tester",
        env_overrides={"AZURE_OPENAI_API_KEY": "test-key"},
    )

    assert captured.get("api_key") == "test-key"
    assert os.getenv("AZURE_OPENAI_API_KEY") is None


def test_assisted_snippets_path_is_relative(
    seeded_project: tuple[RoundBuilder, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, _ = seeded_project
    config = {
        "round_number": 4,
        "round_id": "ph_test_r4",
        "labelset_id": "ls_test",
        "corpus_id": "cor_ph_test",
        "reviewers": [
            {"id": "rev_one", "name": "Reviewer One"},
            {"id": "rev_two", "name": "Reviewer Two"},
        ],
        "overlap_n": 0,
        "rng_seed": 77,
        "assisted_review": {"enabled": True, "top_snippets": 1},
    }
    config_path = tmp_path / "ph_test_r4.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    snippet_payload = {
        "generated_at": "2024-05-01T00:00:00Z",
        "top_snippets": 1,
        "unit_snippets": {
            "doc_0": {
                "label_1": [
                    {
                        "score": 0.5,
                        "text": "Example",
                    }
                ]
            }
        },
    }

    dummy_numpy = types.ModuleType("numpy")
    dummy_numpy.generic = float
    monkeypatch.setitem(sys.modules, "numpy", dummy_numpy)

    dummy_pandas = types.ModuleType("pandas")

    class _DummyTimestamp(datetime):
        pass

    dummy_pandas.Timestamp = _DummyTimestamp
    dummy_pandas.NA = None

    def _notnull(value: object) -> bool:
        return value is not None

    dummy_pandas.notnull = _notnull
    monkeypatch.setitem(sys.modules, "pandas", dummy_pandas)

    def _fake_assisted(*args, **kwargs):
        return snippet_payload

    monkeypatch.setattr(RoundBuilder, "_generate_assisted_review_snippets", _fake_assisted)
    monkeypatch.setattr(
        RoundBuilder,
        "_normalize_for_json",
        lambda self, value: value,
    )

    result = builder.generate_round(
        "ph_test",
        config_path,
        created_by="tester",
    )

    round_dir = Path(result["round_dir"])
    stored_config = json.loads((round_dir / "round_config.json").read_text("utf-8"))
    assist_cfg = stored_config.get("assisted_review", {})
    snippets_value = assist_cfg.get("snippets_json")
    assert snippets_value == "reports/assisted_review/snippets.json"
    assert (round_dir / snippets_value).exists()

    result_payload = result.get("assisted_review", {})
    assert result_payload.get("snippets_json") == "reports/assisted_review/snippets.json"


def test_build_round_assignment_units_normalizes_payload() -> None:
    assignments = {
        "rev_one": ReviewerAssignment(
            reviewer_id="rev_one",
            units=[
                {
                    "unit_id": "doc_0",
                    "patient_icn": "p0",
                    "doc_id": "doc_0",
                },
                {
                    "doc_id": "doc_1",
                    "patient_icn": "p1",
                    "strata": "custom",
                },
            ],
        )
    }

    reviewer_units = admin_main.build_round_assignment_units(assignments)

    assert set(reviewer_units.keys()) == {"rev_one"}
    units = reviewer_units["rev_one"]
    assert [unit.unit_id for unit in units] == ["doc_0", "doc_1"]
    assert units[0].patient_icn == "p0"
    assert units[0].doc_id == "doc_0"
    assert units[0].payload.get("strata_key") == "random_sampling"
    assert units[1].payload.get("strata_key") == "custom"


def test_random_assisted_review_generates_snippets(monkeypatch: pytest.MonkeyPatch, seeded_project: tuple[RoundBuilder, Path]) -> None:
    _, project_root = seeded_project

    class DummyContext:
        def __init__(self, root: Path) -> None:
            self.project_root = root
            self.project_db = Database(root / "project.db")
            self.registered: Dict[Path, str] = {}

        def require_db(self) -> Database:
            return self.project_db

        def register_text_file(self, path: Path, content: str) -> None:
            self.registered[path] = content

    ctx = DummyContext(project_root)

    with ctx.project_db.connect() as conn:
        conn.row_factory = sqlite3.Row
        pheno_row = conn.execute(
            "SELECT * FROM phenotypes WHERE pheno_id=?",
            ("ph_test",),
        ).fetchone()

    assignments = {
        "rev_one": ReviewerAssignment(
            reviewer_id="rev_one",
            units=[
                {
                    "unit_id": "doc_0",
                    "patient_icn": "p0",
                    "doc_id": "doc_0",
                    "documents": [{"doc_id": "doc_0", "text": "Example"}],
                }
            ],
        )
    }

    config_payload: Dict[str, object] = {
        "pheno_id": "ph_test",
        "labelset_id": "ls_test",
        "round_id": "round_random",
        "assisted_review": {"enabled": True, "top_snippets": 1},
    }

    round_dir = project_root / "phenotypes" / "ph_test" / "rounds" / "round_random"
    round_dir.mkdir(parents=True, exist_ok=True)

    captured: Dict[str, object] = {}

    class DummyRoundBuilder:
        def __init__(self, project_root: Path) -> None:
            captured["project_root"] = project_root

        def _generate_assisted_review_snippets(self, **kwargs: object) -> Dict[str, object]:
            captured["env_key"] = os.getenv("AZURE_OPENAI_API_KEY")
            captured["top_n"] = kwargs.get("top_n")
            captured["config"] = kwargs.get("config")
            return {
                "generated_at": "2024-07-01T00:00:00Z",
                "top_snippets": kwargs.get("top_n", 0),
                "unit_snippets": {
                    "doc_0": {
                        "Flag": [
                            {
                                "text": "Example",
                                "score": 0.5,
                            }
                        ]
                    }
                },
            }

        def _json_dumps(self, payload: object) -> str:
            return json.dumps(payload, sort_keys=True)

    monkeypatch.setattr(admin_main, "RoundBuilder", DummyRoundBuilder)

    embed_dir = project_root / "embed"
    embed_dir.mkdir()
    rerank_dir = project_root / "rerank"
    rerank_dir.mkdir()
    local_model_dir = project_root / "llm"
    local_model_dir.mkdir()

    class DummyLineEdit:
        def __init__(self, value: str = "") -> None:
            self._value = value

        def text(self) -> str:  # noqa: D401
            return self._value

    class DummySpinBox:
        def __init__(self, value: int = 0) -> None:
            self._value = value

        def value(self) -> int:  # noqa: D401
            return self._value

    class DummyCombo:
        def __init__(self, value: str) -> None:
            self._value = value

        def currentData(self) -> str:  # noqa: D401
            return self._value

    dummy_dialog = types.SimpleNamespace(
        ctx=ctx,
        pheno_row=pheno_row,
        random_backend_combo=DummyCombo("azure"),
        random_embedding_path_edit=DummyLineEdit(str(embed_dir)),
        random_reranker_path_edit=DummyLineEdit(str(rerank_dir)),
        random_azure_key_edit=DummyLineEdit("test-azure-key"),
        random_azure_version_edit=DummyLineEdit("2024-06-01"),
        random_azure_endpoint_edit=DummyLineEdit("https://example.azure.com"),
        random_local_model_path_edit=DummyLineEdit(str(local_model_dir)),
        random_local_max_seq_spin=DummySpinBox(0),
        random_local_max_new_tokens_spin=DummySpinBox(0),
    )

    result = admin_main.RoundBuilderDialog._generate_random_assisted_review(
        dummy_dialog,
        config_payload=config_payload,
        assignments=assignments,
        round_dir=round_dir,
        top_snippets=2,
    )

    assert captured.get("project_root") == project_root
    assert captured.get("env_key") == "test-azure-key"
    assert captured.get("top_n") == 2
    assert os.getenv("AZURE_OPENAI_API_KEY") is None

    assist_cfg = config_payload.get("assisted_review", {})
    assert assist_cfg.get("snippets_json") == "reports/assisted_review/snippets.json"
    assert assist_cfg.get("generated_at") == "2024-07-01T00:00:00Z"

    expected_path = round_dir / "reports" / "assisted_review" / "snippets.json"
    assert expected_path in ctx.registered
    stored_payload = json.loads(ctx.registered[expected_path])
    assert stored_payload.get("unit_snippets")

    assert result == {"snippets_json": "reports/assisted_review/snippets.json"}


def test_assisted_review_respects_local_backend_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vaannotate.vaannotate_ai_backend.services import contexts as backend_contexts
    from vaannotate.vaannotate_ai_backend.services import family_labeler as backend_family_labeler

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    project_root = tmp_path / "Project"
    project_root.mkdir()
    builder = RoundBuilder(project_root)
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    local_model_dir = tmp_path / "LocalLLM"
    local_model_dir.mkdir()

    from huggingface_hub import file_download, utils as hf_utils

    monkeypatch.setattr(
        file_download,
        "hf_hub_download",
        lambda *args, **kwargs: str(local_model_dir / "dummy.bin"),
    )
    monkeypatch.setattr(
        hf_utils._http,
        "http_backoff",
        lambda *args, **kwargs: types.SimpleNamespace(status_code=200, headers={}, url=""),
    )

    import requests

    class _DummyResponse:
        def __init__(self, url: str) -> None:
            self.url = url
            self.status_code = 200
            self.headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        lambda self, method, url, **kwargs: _DummyResponse(url),
    )

    class DummyPipeline:
        def __init__(self) -> None:
            self.repo = types.SimpleNamespace(notes=None)
            self.store = types.SimpleNamespace(build_chunk_index=lambda *a, **k: None)
            self.cfg = types.SimpleNamespace(
                rag=types.SimpleNamespace(),
                index=types.SimpleNamespace(),
                scjitter=types.SimpleNamespace(),
                llmfirst=types.SimpleNamespace(),
            )
            self.retriever = types.SimpleNamespace()
            self.llm = object()
            self.label_config = {}

        def _label_maps(self):  # noqa: D401
            return {}, {}, {"Flag": "rule"}, {"Flag": "categorical_single"}

    def _dummy_build_runner(*args, **kwargs):
        env_snapshot["LLM_BACKEND"] = os.getenv("LLM_BACKEND")
        env_snapshot["LOCAL_LLM_MODEL_DIR"] = os.getenv("LOCAL_LLM_MODEL_DIR")
        env_snapshot["LOCAL_LLM_MAX_SEQ_LEN"] = os.getenv("LOCAL_LLM_MAX_SEQ_LEN")
        env_snapshot["LOCAL_LLM_MAX_NEW_TOKENS"] = os.getenv("LOCAL_LLM_MAX_NEW_TOKENS")
        return DummyPipeline()

    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.orchestration.build_active_learning_runner",
        _dummy_build_runner,
    )

    env_snapshot: Dict[str, object] = {}

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            env_snapshot["LLM_BACKEND"] = os.getenv("LLM_BACKEND")
            env_snapshot["LOCAL_LLM_MODEL_DIR"] = os.getenv("LOCAL_LLM_MODEL_DIR")
            env_snapshot["LOCAL_LLM_MAX_SEQ_LEN"] = os.getenv("LOCAL_LLM_MAX_SEQ_LEN")
            env_snapshot["LOCAL_LLM_MAX_NEW_TOKENS"] = os.getenv("LOCAL_LLM_MAX_NEW_TOKENS")
            self.repo = types.SimpleNamespace(notes=None)
            self.store = types.SimpleNamespace(build_chunk_index=lambda *args, **kwargs: None)
            llmfirst = types.SimpleNamespace(
                single_doc_context="rag",
                single_doc_full_context_max_chars=None,
            )
            self.cfg = types.SimpleNamespace(
                rag=types.SimpleNamespace(),
                index=types.SimpleNamespace(),
                scjitter=types.SimpleNamespace(),
                llmfirst=llmfirst,
            )
            self.llm = object()
            self.rag = object()
            self.label_config = kwargs.get("label_config", {})

        def _label_maps(self):  # noqa: D401
            return {}, {}, {"Flag": "rule"}, {"Flag": {"type": "categorical_single"}}

    class DummyFamilyLabeler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def ensure_label_exemplars(self, *args, **kwargs) -> None:
            return None

    def dummy_contexts(*args, **kwargs):
        return [
            {
                "doc_id": "doc_local",
                "chunk_id": 1,
                "score": 0.9,
                "source": "test",
                "text": "Example context",
                "metadata": {"note": "example"},
            }
        ]

    monkeypatch.setattr(backend_family_labeler, "FamilyLabeler", DummyFamilyLabeler)
    monkeypatch.setattr(backend_contexts, "_contexts_for_unit_label", dummy_contexts)

    assignments = {
        "rev_one": [
            AssignmentUnit(
                unit_id="unit_local",
                patient_icn="p1",
                doc_id="doc_local",
                payload={
                    "documents": [
                        {
                            "doc_id": "doc_local",
                            "patient_icn": "p1",
                            "text": "Source note",
                        }
                    ]
                },
            )
        ]
    }

    labelset = {
        "labelset_id": "ls_local",
        "labels": [
            {
                "label_id": "Flag",
                "name": "Flag",
                "type": "categorical_single",
                "required": False,
                "options": [{"value": "yes", "display": "Yes"}],
            }
        ],
    }

    config = {
        "assisted_review": {"enabled": True, "top_snippets": 1},
        "ai_backend": {
            "backend": "exllamav2",
            "local_model_dir": str(local_model_dir),
            "local_max_seq_len": 4096,
            "local_max_new_tokens": 256,
        },
    }

    pheno_row = {"level": "single_doc"}

    result = builder._generate_assisted_review_snippets(
        pheno_row=pheno_row,
        labelset=labelset,
        round_dir=round_dir,
        reviewer_assignments=assignments,
        config=config,
        config_base=round_dir,
        top_n=1,
    )

    unit_map = result.get("unit_snippets", {}).get("unit_local", {})
    assert unit_map.get("Flag")
    assert env_snapshot.get("LLM_BACKEND") == "exllamav2"
    assert env_snapshot.get("LOCAL_LLM_MODEL_DIR") == str(local_model_dir)
    assert env_snapshot.get("LOCAL_LLM_MAX_SEQ_LEN") == "4096"
    assert env_snapshot.get("LOCAL_LLM_MAX_NEW_TOKENS") == "256"
    assert os.getenv("LOCAL_LLM_MODEL_DIR") is None
    assert os.getenv("LLM_BACKEND") is None
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
        from vaannotate.vaannotate_ai_backend.utils.runtime import LOGGER

        LOGGER.info("[FinalLLM] 1/1 complete")
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


def test_generate_round_auto_submits_llm_reviewer(
    seeded_project: tuple[RoundBuilder, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, project_root = seeded_project
    labels_payload = [
        {"unit_id": "doc_0", "Flag_llm": "yes", "Score_llm": 1.0},
        {"unit_id": "doc_1", "Flag_llm": "no", "Score_llm": 2.5},
        {"unit_id": "doc_2", "Flag_llm": "yes", "Score_llm": None},
    ]

    def _fake_apply(self, **_kwargs):  # type: ignore[no-untyped-def]
        round_dir = _kwargs["round_dir"]
        exports_dir = round_dir / "reports" / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        labels_json = exports_dir / "final_llm_labels.json"
        labels_json.write_text(json.dumps(labels_payload), encoding="utf-8")
        return {"final_llm_labels_json": str(labels_json)}

    monkeypatch.setattr(RoundBuilder, "_apply_final_llm_labeling", _fake_apply)

    config = {
        "round_number": 3,
        "round_id": "ph_test_r3",
        "labelset_id": "ls_test",
        "corpus_id": "cor_ph_test",
        "reviewers": [
            {"id": RoundBuilder.LLM_REVIEWER_ID, "name": "LLM"},
        ],
        "rng_seed": 5,
        "total_n": 2,
        "final_llm_labeling": True,
    }
    config_path = tmp_path / "ph_test_r3.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = builder.generate_round("ph_test", config_path, created_by="tester")

    round_dir = Path(result["round_dir"])
    assignment_dir = round_dir / "assignments" / RoundBuilder.LLM_REVIEWER_ID
    receipt = json.loads((assignment_dir / "submitted.json").read_text("utf-8"))
    assert receipt["unit_count"] == receipt["completed"]
    assert receipt["unit_count"] > 0

    with sqlite3.connect(assignment_dir / "assignment.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT unit_id, label_id, value, value_num, na FROM annotations"
        ).fetchall()

    assert rows
    observed = {(row["unit_id"], row["label_id"]): row for row in rows}
    expected_flags = {entry["unit_id"]: entry["Flag_llm"] for entry in labels_payload}
    expected_scores = {entry["unit_id"]: entry["Score_llm"] for entry in labels_payload}

    for unit_id, expected_flag in expected_flags.items():
        key = (unit_id, "Flag")
        if key not in observed:
            continue
        assert observed[key]["value"] == expected_flag
        score_key = (unit_id, "Score")
        if score_key not in observed:
            continue
        expected_score = expected_scores[unit_id]
        if expected_score is None:
            assert observed[score_key]["na"] == 1
        else:
            assert observed[score_key]["value_num"] == pytest.approx(float(expected_score))

    with sqlite3.connect(project_root / "project.db") as conn:
        conn.row_factory = sqlite3.Row
        status_row = conn.execute(
            "SELECT status FROM assignments WHERE round_id=? AND reviewer_id=?",
            ("ph_test_r3", RoundBuilder.LLM_REVIEWER_ID),
        ).fetchone()
    assert status_row is not None and status_row["status"] == "submitted"


def test_generate_round_auto_submit_llm_missing_predictions(
    seeded_project: tuple[RoundBuilder, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, _ = seeded_project
    labels_payload = [
        {"unit_id": "doc_0", "Flag_llm": "yes", "Score_llm": 1.0},
    ]

    def _fake_apply(self, **_kwargs):  # type: ignore[no-untyped-def]
        round_dir = _kwargs["round_dir"]
        exports_dir = round_dir / "reports" / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        labels_json = exports_dir / "final_llm_labels.json"
        labels_json.write_text(json.dumps(labels_payload), encoding="utf-8")
        return {"final_llm_labels_json": str(labels_json)}

    monkeypatch.setattr(RoundBuilder, "_apply_final_llm_labeling", _fake_apply)

    config = {
        "round_number": 3,
        "round_id": "ph_test_r3",
        "labelset_id": "ls_test",
        "corpus_id": "cor_ph_test",
        "reviewers": [
            {"id": RoundBuilder.LLM_REVIEWER_ID, "name": "LLM"},
        ],
        "rng_seed": 5,
        "total_n": 2,
        "final_llm_labeling": True,
    }
    config_path = tmp_path / "ph_test_r3.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = builder.generate_round("ph_test", config_path, created_by="tester")

    round_dir = Path(result["round_dir"])
    assignment_dir = round_dir / "assignments" / RoundBuilder.LLM_REVIEWER_ID
    with sqlite3.connect(assignment_dir / "assignment.db") as conn:
        conn.row_factory = sqlite3.Row
        completions = {
            row["unit_id"]: row["complete"]
            for row in conn.execute(
                "SELECT unit_id, complete FROM units ORDER BY unit_id"
            ).fetchall()
        }
        annotations = conn.execute(
            "SELECT unit_id, label_id, value, value_num, na FROM annotations"
        ).fetchall()

    assert completions["doc_0"] == 1
    assert completions["doc_1"] == 0
    observed = {(row["unit_id"], row["label_id"]): row for row in annotations}

    assert observed[("doc_0", "Flag")]["value"] == "yes"
    assert observed[("doc_0", "Score")]["value_num"] == pytest.approx(1.0)

    for label_id in ("Flag", "Score"):
        missing_row = observed[("doc_1", label_id)]
        assert missing_row["value"] in (None, "")
        assert missing_row["value_num"] is None

