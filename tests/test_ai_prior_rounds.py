from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Unavailable:  # pragma: no cover - helper for optional deps
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


from vaannotate.AdminApp.main import RoundBuilderDialog  # noqa: E402
from vaannotate.vaannotate_ai_backend.engine import (  # noqa: E402
    DataRepository,
    DisagreementConfig,
    DisagreementExpander,
)
from vaannotate.vaannotate_ai_backend.label_configs import (  # noqa: E402
    LabelConfigBundle,
)


def _seed_repo() -> DataRepository:
    notes_df = pd.DataFrame(
        [
            {"patient_icn": "p1", "doc_id": "d1", "text": "note"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {
                "round_id": "1",
                "unit_id": "p1",
                "doc_id": "d1",
                "label_id": "L",
                "reviewer_id": "rev1",
                "label_value": "yes",
                "labelset_id": "ls1",
                "label_rules": "rule old",
            },
            {
                "round_id": "1",
                "unit_id": "p1",
                "doc_id": "d1",
                "label_id": "L",
                "reviewer_id": "rev2",
                "label_value": "no",
                "labelset_id": "ls1",
                "label_rules": "rule old",
            },
            {
                "round_id": "2",
                "unit_id": "p1",
                "doc_id": "d1",
                "label_id": "L",
                "reviewer_id": "rev1",
                "label_value": "yes",
                "labelset_id": "ls2",
                "label_rules": "rule new",
            },
            {
                "round_id": "2",
                "unit_id": "p1",
                "doc_id": "d1",
                "label_id": "L",
                "reviewer_id": "rev2",
                "label_value": "yes",
                "labelset_id": "ls2",
                "label_rules": "rule new",
            },
        ]
    )
    return DataRepository(notes_df, ann_df)


def test_reviewer_disagreement_aggregates_all_rounds() -> None:
    repo = _seed_repo()

    last_only = repo.reviewer_disagreement(round_policy="last")
    assert last_only.loc[last_only.index[0], "disagreement_score"] == 0.0

    combined = repo.reviewer_disagreement(round_policy="all")
    assert combined.loc[combined.index[0], "disagreement_score"] > 0.0
    # Most recent metadata should be retained when collating rounds
    assert combined.loc[combined.index[0], "round_id"] == "2"
    assert combined.loc[combined.index[0], "labelset_id"] == "ls2"


def test_disagreement_seeds_include_prior_rounds() -> None:
    repo = _seed_repo()
    bundle = LabelConfigBundle(
        current={"_meta": {"labelset_id": "ls2"}, "L": {"label_id": "L", "type": "boolean"}},
        legacy={
            "ls1": {"L": {"label_id": "L", "type": "boolean"}},
        },
        round_labelsets={"1": "ls1", "2": "ls2"},
    )

    cold_cfg = DisagreementConfig(round_policy="last", high_entropy_threshold=0.01)
    hot_cfg = DisagreementConfig(round_policy="all", high_entropy_threshold=0.01)

    cold_expander = DisagreementExpander(cold_cfg, repo, retriever=None, label_config_bundle=bundle)
    hot_expander = DisagreementExpander(hot_cfg, repo, retriever=None, label_config_bundle=bundle)

    cold = cold_expander.high_entropy_seeds()
    assert cold.empty

    hot = hot_expander.high_entropy_seeds()
    assert not hot.empty
    assert set(hot["round_id"]) == {"2"}
    assert set(hot["labelset_id"]) == {"ls2"}


class _SpinStub:
    def __init__(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value


class _CheckStub:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:  # noqa: N802 - mimic Qt API
        return self._checked


def test_collect_ai_overrides_sets_round_policy_for_multi_round() -> None:
    dialog = RoundBuilderDialog.__new__(RoundBuilderDialog)
    dialog._using_ai_backend = lambda: True  # type: ignore[attr-defined]
    dialog.total_n_spin = _SpinStub(10)
    dialog.ai_disagreement_pct = _SpinStub(0.2)
    dialog.ai_uncertain_pct = _SpinStub(0.2)
    dialog.ai_easy_pct = _SpinStub(0.2)
    dialog.ai_diversity_pct = _SpinStub(0.2)
    dialog.ai_final_llm_checkbox = _CheckStub(True)

    overrides_multi = dialog._collect_ai_overrides(prior_rounds=[1, 2])
    assert overrides_multi["disagree"]["round_policy"] == "all"

    overrides_single = dialog._collect_ai_overrides(prior_rounds=[3])
    assert overrides_single["disagree"]["round_policy"] == "last"
