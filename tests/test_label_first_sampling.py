import types
from pathlib import Path

import pandas as pd

from vaannotate.vaannotate_ai_backend.pipelines.active_learning import ActiveLearningPipeline
from vaannotate.vaannotate_ai_backend.services.label_first import parse_label_first_targets


class _Repo:
    def __init__(self, notes: pd.DataFrame, ann: pd.DataFrame) -> None:
        self.notes = notes
        self.ann = ann

    def exclude_units(self, excluded_ids) -> int:  # pragma: no cover - trivial
        _ = excluded_ids
        return 0

    def notes_by_doc(self):  # pragma: no cover - trivial
        return {}


class _Store:
    def build_chunk_index(self, *_args, **_kwargs):  # pragma: no cover - trivial
        return None


class _Selector:
    def __init__(self) -> None:
        self.label_types = {}
        self.current_label_ids = set()
        self.seen_units = set()
        self.unseen_pairs = set()

    def _empty_unit_frame(self) -> pd.DataFrame:  # pragma: no cover - unused for label-first
        return pd.DataFrame(columns=["unit_id", "label_id", "label_type", "selection_reason"])


class _Diversity:
    def select_diverse_units(self, *_, **__):  # pragma: no cover - unused for label-first
        return pd.DataFrame(columns=["unit_id", "label_id", "label_type", "selection_reason"])


class _FamilyLabeler:
    def __init__(self, *_args, **_kwargs):
        pass

    def label_family_for_unit(self, unit_id, label_types, per_label_rules, **_kwargs):
        _ = label_types
        _ = per_label_rules
        prediction = "yes" if str(unit_id) in {"u1", "u3"} else "no"
        return [{"unit_id": unit_id, "label_id": "label_a", "prediction": prediction}]


class _CtxBuilder:
    def __init__(self, bundle):
        self.label_config_bundle = bundle

    def build_context_for_label(self, unit_id, *_args, **_kwargs):
        score = {"u1": 0.95, "u2": 0.10, "u3": 0.90, "u4": 0.20}.get(str(unit_id), 0.0)
        return [{"score": score}]

    def build_context_for_family(self, *_args, **_kwargs):  # pragma: no cover - unused in this test
        return []


class _Bundle:
    def legacy_rules_map(self):
        return {}

    def legacy_label_types(self):
        return {}

    def current_rules_map(self, *_args, **_kwargs):
        return {"label_a": "rule"}

    def current_label_types(self, *_args, **_kwargs):
        return {"label_a": "categorical"}

    def label_maps(self, label_config=None, ann_df=None):
        _ = label_config
        _ = ann_df
        return (
            self.legacy_rules_map(),
            self.legacy_label_types(),
            self.current_rules_map(),
            self.current_label_types(),
        )


def test_parse_label_first_targets_filters_invalid_entries() -> None:
    parsed = parse_label_first_targets(
        [
            {"label_id": "a", "values": ["yes", "possible"], "quota": 2, "operator": "in"},
            {"label_id": "", "values": ["x"], "quota": 1},
            {"label_id": "b", "value": "ok", "quota": 0},
        ]
    )
    assert len(parsed) == 1
    assert parsed[0].label_id == "a"
    assert parsed[0].values == ("yes", "possible")


def test_active_learning_pipeline_label_first_sampling_hits_quota(tmp_path: Path, monkeypatch) -> None:
    notes = pd.DataFrame(
        [
            {"unit_id": "u1", "doc_id": "d1", "text": "alpha"},
            {"unit_id": "u2", "doc_id": "d2", "text": "beta"},
            {"unit_id": "u3", "doc_id": "d3", "text": "gamma"},
            {"unit_id": "u4", "doc_id": "d4", "text": "delta"},
        ]
    )
    ann = pd.DataFrame(columns=["unit_id", "label_id"])

    select_cfg = types.SimpleNamespace(
        batch_size=2,
        pct_disagreement=0.0,
        pct_diversity=0.0,
        pct_uncertain=0.0,
        pct_easy_qc=0.0,
        sampling_mode="label_first",
        label_first_targets=[
            {"label_id": "label_a", "values": ["yes"], "quota": 2, "operator": "in"}
        ],
        label_first_pathway="enriched",
        label_first_pool_initial_size=4,
        label_first_pool_growth_step=2,
        label_first_pool_max_size=8,
    )
    cfg = types.SimpleNamespace(
        select=select_cfg,
        rag=types.SimpleNamespace(),
        index=types.SimpleNamespace(),
        llm=types.SimpleNamespace(model_name="stub"),
        diversity=types.SimpleNamespace(),
        llmfirst=types.SimpleNamespace(
            enrich=True,
            progress_min_interval_s=0.0,
            final_llm_label_consistency=1,
            inference_labeling_mode="family",
        ),
        scjitter=types.SimpleNamespace(),
        final_llm_labeling=False,
    )

    repo = _Repo(notes=notes, ann=ann)
    bundle = _Bundle()
    pipeline = ActiveLearningPipeline(
        data_repo=repo,
        emb_store=_Store(),
        ctx_builder=_CtxBuilder(bundle),
        llm_labeler=types.SimpleNamespace(family_labeler_cls=_FamilyLabeler),
        disagreement_scorer=types.SimpleNamespace(),
        uncertainty_scorer=types.SimpleNamespace(),
        diversity_selector=_Diversity(),
        selector=_Selector(),
        config=cfg,
        paths=types.SimpleNamespace(outdir=str(tmp_path)),
        label_config={"label_a": {"label_id": "label_a"}},
        label_config_bundle=bundle,
        disagreement_builder_fn=lambda *_args, **_kwargs: pd.DataFrame(),
        uncertain_builder_fn=lambda *_args, **_kwargs: pd.DataFrame(),
        certain_builder_fn=lambda *_args, **_kwargs: pd.DataFrame(),
    )

    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *a, **k: None)

    result = pipeline.run()
    assert not result.empty
    assert set(result["unit_id"].tolist()) >= {"u1", "u3"}
    assert set(result["selection_reason"].tolist()) == {"label_first"}
