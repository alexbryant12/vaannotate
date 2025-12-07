import types
from pathlib import Path

import pandas as pd

from vaannotate.vaannotate_ai_backend.pipelines.active_learning import ActiveLearningPipeline
from vaannotate.vaannotate_ai_backend.services import LLM_RECORDER


class _StubRepo:
    def __init__(self, notes: pd.DataFrame, ann: pd.DataFrame) -> None:
        self.notes = notes
        self.ann = ann

    def exclude_units(self, excluded_ids) -> int:  # pragma: no cover - trivial
        return 0

    def notes_by_doc(self) -> dict:  # pragma: no cover - trivial
        return {}


class _StubStore:
    def build_chunk_index(self, notes, rag_cfg, index_cfg) -> None:  # pragma: no cover - trivial
        return None


class _StubSelector:
    def __init__(self) -> None:
        self.label_types = {}
        self.current_label_ids = set()
        self.seen_units = set()
        self.unseen_pairs = set()

    def _empty_unit_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["unit_id", "label_id", "label_type", "selection_reason"])

    def select_disagreement(self, pairs, selected_units):  # pragma: no cover - trivial
        return self._empty_unit_frame()

    def _filter_units(self, df, units):  # pragma: no cover - trivial
        return df

    def _to_unit_only(self, df):  # pragma: no cover - trivial
        return df

    def _head_units(self, df, n):  # pragma: no cover - trivial
        return df.head(n)

    def build_next_batch(self, disagree_df, uncertainty_df, easy_df, diversity_df, prefiltered):
        return pd.DataFrame(
            [
                {
                    "unit_id": "u1",
                    "label_id": "label_a",
                    "label_type": "binary",
                    "selection_reason": "test",
                }
            ]
        )


class _StubDiversitySelector:
    def select_diverse_units(self, *_, **__):  # pragma: no cover - trivial
        return pd.DataFrame(columns=["unit_id", "label_id", "label_type", "selection_reason"])


class _StubFamilyLabeler:
    def __init__(self, *_, **__):  # pragma: no cover - trivial
        pass


def test_active_learning_run_preserves_recorder_meta(tmp_path: Path, monkeypatch) -> None:
    notes_df = pd.DataFrame([{"unit_id": "u1", "doc_id": "d1", "text": "note"}])
    ann_df = pd.DataFrame(columns=["round_id", "unit_id", "label_id"])

    repo = _StubRepo(notes_df, ann_df)
    store = _StubStore()

    selector = _StubSelector()
    diversity_selector = _StubDiversitySelector()

    label_config = {
        "_meta": {"labelset_id": "ls"},
        "label_a": {"label_id": "label_a", "rules": "rule", "type": "boolean"},
    }

    class _LabelBundle:
        def legacy_rules_map(self):  # pragma: no cover - trivial
            return {}

        def legacy_label_types(self):  # pragma: no cover - trivial
            return {}

        def current_rules_map(self, *_, **__):  # pragma: no cover - trivial
            return {"label_a": "rule"}

        def current_label_types(self, *_, **__):  # pragma: no cover - trivial
            return {"label_a": "binary"}

        def label_maps(self, label_config=None, ann_df=None):  # pragma: no cover - trivial
            """Minimal shim to match LabelConfigBundle.label_maps signature."""
            return (
                self.legacy_rules_map(),
                self.legacy_label_types(),
                self.current_rules_map(label_config),
                self.current_label_types(label_config),
            )

    select_cfg = types.SimpleNamespace(
        batch_size=1,
        pct_disagreement=0,
        pct_diversity=0,
        pct_uncertain=0,
        pct_easy_qc=0,
    )

    diversity_cfg = types.SimpleNamespace(
        rag_topk=4,
        min_rel_quantile=0.3,
        mmr_lambda=0.7,
        sample_cap=10,
        adaptive_relax=True,
        relax_steps=(0.2,),
        pool_factor=4.0,
        use_proto=False,
    )

    cfg = types.SimpleNamespace(
        select=select_cfg,
        rag=None,
        index=None,
        llm=types.SimpleNamespace(model_name="test-model"),
        diversity=diversity_cfg,
        llmfirst=types.SimpleNamespace(enrich=None, final_llm_label_consistency=1),
        scjitter=None,
        final_llm_labeling=False,
    )

    paths = types.SimpleNamespace(outdir=str(tmp_path))

    llm_labeler = types.SimpleNamespace(family_labeler_cls=_StubFamilyLabeler, model_name="test-model")

    pipeline = ActiveLearningPipeline(
        repo,
        store,
        ctx_builder=types.SimpleNamespace(label_config_bundle=_LabelBundle()),
        llm_labeler=llm_labeler,
        disagreement_scorer=types.SimpleNamespace(),
        uncertainty_scorer=types.SimpleNamespace(),
        diversity_selector=diversity_selector,
        selector=selector,
        config=cfg,
        paths=paths,
        pooler=None,
        retriever=types.SimpleNamespace(rerank_rule_overrides={}),
        label_config=label_config,
        label_config_bundle=_LabelBundle(),
        disagreement_builder_fn=lambda *_, **__: selector._empty_unit_frame(),
        uncertain_builder_fn=lambda *_, **__: selector._empty_unit_frame(),
        certain_builder_fn=lambda *_, **__: selector._empty_unit_frame(),
    )

    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *_, **__: None)

    try:
        pipeline.run()
    finally:
        LLM_RECORDER.flush()

    assert LLM_RECORDER.run_meta == {
        "model": "test-model",
        "project": tmp_path.name,
    }
