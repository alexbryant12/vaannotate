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


from vaannotate.vaannotate_ai_backend.engine import ActiveLearningLLMFirst, DataRepository
from vaannotate.vaannotate_ai_backend.label_configs import LabelConfigBundle


def test_label_config_overlays_new_labels_for_non_disagreement(tmp_path: Path) -> None:
    notes_df = pd.DataFrame(
        [
            {"patient_icn": "p1", "doc_id": "d1", "text": "note one"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {
                "round_id": "r1",
                "unit_id": "p1",
                "doc_id": "d1",
                "label_id": "legacy_label",
                "reviewer_id": "rev1",
                "label_value": "yes",
                "label_rules": "legacy rule",
            }
        ]
    )

    repo = DataRepository(notes_df, ann_df)

    label_config = {
        "_meta": {"labelset_id": "ls"},
        "legacy_label": {"label_id": "legacy_label", "type": "text", "rules": "legacy rule override"},
        "new_label": {"label_id": "new_label", "type": "boolean", "rules": "fresh rule"},
    }

    orchestrator = ActiveLearningLLMFirst.__new__(ActiveLearningLLMFirst)
    orchestrator.repo = repo
    orchestrator.label_config = label_config
    orchestrator.label_config_bundle = LabelConfigBundle(current=label_config)

    legacy_rules, legacy_types, current_rules, current_types = orchestrator._label_maps()

    assert "legacy_label" in legacy_rules
    assert "new_label" not in legacy_rules  # legacy annotations exclude the new label

    assert legacy_types.get("legacy_label") == "binary"
    assert "new_label" not in legacy_types

    assert current_types["legacy_label"] == "categorical"
    assert current_rules["new_label"] == "fresh rule"
    assert current_types["new_label"] == "binary"

    unseen_pairs = orchestrator.build_unseen_pairs()
    assert ("p1", "new_label") in set(unseen_pairs)

    empty_sel = pd.DataFrame(columns=["unit_id", "label_id", "label_type", "selection_reason"])
    topoff = orchestrator.top_off_random(
        current_sel=empty_sel,
        unseen_pairs=[("p1", "new_label")],
        label_types=current_types,
        target_n=1,
    )
    assert list(topoff["label_id"]) == ["new_label"]
    assert topoff.loc[topoff.index[0], "label_type"] == "binary"


def test_current_round_rules_ignore_legacy_when_not_selected() -> None:
    notes_df = pd.DataFrame(
        [
            {"patient_icn": "p1", "doc_id": "d1", "text": "note one"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {
                "round_id": "r1",
                "unit_id": "p1",
                "doc_id": "d1",
                "label_id": "legacy_label",
                "reviewer_id": "rev1",
                "label_value": "yes",
                "label_rules": "legacy rule",
            }
        ]
    )

    repo = DataRepository(notes_df, ann_df)

    label_config = {
        "_meta": {"labelset_id": "ls"},
        "new_label": {"label_id": "new_label", "type": "boolean", "rules": "fresh rule"},
    }

    orchestrator = ActiveLearningLLMFirst.__new__(ActiveLearningLLMFirst)
    orchestrator.repo = repo
    orchestrator.label_config = label_config
    orchestrator.label_config_bundle = LabelConfigBundle(current=label_config)

    legacy_rules, _, current_rules, current_types = orchestrator._label_maps()

    assert "legacy_label" in legacy_rules
    assert "legacy_label" not in current_rules
    assert current_types == {"new_label": "binary"}

    current_label_ids = set(current_rules.keys())
    unseen_pairs = orchestrator.build_unseen_pairs(label_ids=current_label_ids)
    assert unseen_pairs == [("p1", "new_label")]


def test_label_config_bundle_strips_meta_entries() -> None:
    bundle = LabelConfigBundle(
        current={
            "_meta": {"labelset_id": "ls_new"},
            "current_label": {"label_id": "current_label"},
        },
        legacy={
            "legacy": {
                "_meta": {"labelset_id": "ls_legacy"},
                "legacy_label": {"label_id": "legacy_label"},
            }
        },
    )

    assert "_meta" not in bundle.current
    legacy_config = bundle.config_for_labelset("legacy")
    assert "_meta" not in legacy_config
    assert "legacy_label" in legacy_config


def test_attach_metadata_for_multi_doc_units() -> None:
    notes_df = pd.DataFrame(
        [
            {"patient_icn": "p1", "doc_id": "d1", "text": "note one"},
            {"patient_icn": "p1", "doc_id": "d2", "text": "note two"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {
                "round_id": "r1",
                "unit_id": "p1",
                "doc_id": "d1",
                "label_id": "legacy_label",
                "reviewer_id": "rev1",
                "label_value": "yes",
            }
        ]
    )

    repo = DataRepository(notes_df, ann_df, phenotype_level="multi_doc")
    orchestrator = ActiveLearningLLMFirst.__new__(ActiveLearningLLMFirst)
    orchestrator.repo = repo

    selection = pd.DataFrame(
        [{"unit_id": "p1", "label_id": "legacy_label", "selection_reason": "bucket"}]
    )
    augmented = orchestrator._attach_unit_metadata(selection)

    assert list(augmented.columns).count("patient_icn") == 1
    assert "doc_id" not in augmented.columns
    assert augmented.loc[0, "patient_icn"] == "p1"


def test_attach_metadata_for_single_doc_units() -> None:
    notes_df = pd.DataFrame(
        [
            {"patient_icn": "p1", "doc_id": "d1", "text": "note one"},
            {"patient_icn": "p2", "doc_id": "d2", "text": "note two"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {
                "round_id": "r1",
                "unit_id": "d1",
                "doc_id": "d1",
                "label_id": "legacy_label",
                "reviewer_id": "rev1",
                "label_value": "yes",
            }
        ]
    )

    repo = DataRepository(notes_df, ann_df, phenotype_level="single_doc")
    orchestrator = ActiveLearningLLMFirst.__new__(ActiveLearningLLMFirst)
    orchestrator.repo = repo

    selection = pd.DataFrame(
        [{"unit_id": "d1", "label_id": "legacy_label", "selection_reason": "bucket"}]
    )
    augmented = orchestrator._attach_unit_metadata(selection)

    assert "patient_icn" in augmented.columns
    assert "doc_id" in augmented.columns
    assert augmented.loc[0, "patient_icn"] == "p1"
    assert augmented.loc[0, "doc_id"] == "d1"

