from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend.core.data import DataRepository
from vaannotate.vaannotate_ai_backend.testing import DummyLLMLabeler, make_dummy_llm_labeler


DATA_DIR = Path(__file__).resolve().parent / "data" / "ai_backend"


def _load_repo() -> DataRepository:
    notes = pd.read_csv(DATA_DIR / "notes.csv")
    annotations = pd.read_csv(DATA_DIR / "annotations.csv")
    return DataRepository(notes, annotations)


def _load_label_config() -> dict:
    with open(DATA_DIR / "label_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _stubs(repo: DataRepository) -> tuple[SimpleNamespace, SimpleNamespace]:
    context_builder = SimpleNamespace(repo=repo)
    retriever = SimpleNamespace(_repo=repo, get_last_diagnostics=lambda *_args, **_kwargs: {})
    return context_builder, retriever


def test_dummy_llm_labeler_uses_keyword_rules() -> None:
    repo = _load_repo()
    context_builder, retriever = _stubs(repo)
    label_config = _load_label_config()
    label_types = {"pneumonitis": "categorical"}
    per_label_rules = {"pneumonitis": ""}

    labeler = DummyLLMLabeler(label_config)
    rows = labeler.label_unit(
        "1001",
        ["pneumonitis"],
        label_types=label_types,
        per_label_rules=per_label_rules,
        context_builder=context_builder,
        retriever=retriever,
        llmfirst_cfg=SimpleNamespace(),
        json_only=True,
        json_n_consistency=1,
        json_jitter=False,
    )

    assert rows and rows[0]["prediction"] == "pneumonitis_yes"
    assert labeler.calls == ["1001"]


def test_dummy_llm_labeler_handles_negative_context() -> None:
    repo = _load_repo()
    context_builder, retriever = _stubs(repo)
    label_types = {"pneumonitis": "categorical"}
    per_label_rules = {"pneumonitis": ""}

    labeler = make_dummy_llm_labeler(_load_label_config())
    rows = labeler.label_unit(
        "1002",
        ["pneumonitis"],
        label_types=label_types,
        per_label_rules=per_label_rules,
        context_builder=context_builder,
        retriever=retriever,
        llmfirst_cfg=SimpleNamespace(),
        json_only=True,
        json_n_consistency=1,
        json_jitter=False,
    )

    assert rows and rows[0]["prediction"] == "pneumonitis_no"
    assert labeler.calls == ["1002"]
