import types

import pandas as pd


def test_llm_probe_jsonifies_llm_runs(tmp_path, monkeypatch):
    from vaannotate.vaannotate_ai_backend.pipelines.active_learning import ActiveLearningPipeline
    from vaannotate.vaannotate_ai_backend.utils.jsonish import _jsonify_cols

    # Stub out the family labeler to return a normalized probe DataFrame that includes
    # structured llm_runs values.
    probe_df = pd.DataFrame(
        {
            "unit_id": ["u1", "u2"],
            "label_id": ["L1", "L1"],
            "llm_runs": [[{"raw": {"reasoning": "x"}}], None],
            "U": [0.1, 0.2],
        }
    )

    class DummyFamilyLabeler:
        def probe_units_label_tree(self, *args, **kwargs):
            return probe_df

    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.services.family_labeler.build_family_labeler",
        lambda *a, **k: DummyFamilyLabeler(),
    )

    class DummyUncertainty:
        def score_probe_results(self, df):
            return df

    cfg = types.SimpleNamespace(
        llmfirst=types.SimpleNamespace(enrich=False),
        scjitter=types.SimpleNamespace(),
        select=types.SimpleNamespace(batch_size=2, pct_uncertain=0.5),
    )
    paths = types.SimpleNamespace(outdir=str(tmp_path))

    pipeline = ActiveLearningPipeline(
        data_repo=None,
        emb_store=None,
        ctx_builder=None,
        llm_labeler=None,
        disagreement_scorer=None,
        uncertainty_scorer=DummyUncertainty(),
        diversity_selector=None,
        selector=None,
        config=cfg,
        paths=paths,
        jsonify_cols_fn=_jsonify_cols,
    )

    result = pipeline._build_llm_uncertain_bucket(label_types={}, rules_map={}, exclude_units=None)

    parquet_path = tmp_path / "llm_probe.parquet"
    assert parquet_path.exists()

    stored = pd.read_parquet(parquet_path)
    # Structured runs should have been serialized to JSON strings before writing parquet.
    assert isinstance(stored.loc[0, "llm_runs"], str)
    assert isinstance(result, pd.DataFrame) and not result.empty
