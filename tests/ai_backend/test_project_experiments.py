from __future__ import annotations

import json

import pandas as pd
from pandas.testing import assert_frame_equal

from vaannotate.vaannotate_ai_backend import experiments, project_experiments


class _DummyResult:
    def __init__(self, df: pd.DataFrame):
        self.dataframe = df
        self.artifacts = {}
        self.outdir = None


def test_run_project_inference_experiments_applies_configs(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    notes_df = pd.DataFrame(
        [
            {"doc_id": "1", "patient_icn": "p1", "text": "Note A"},
            {"doc_id": "2", "patient_icn": "p2", "text": "Note B"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {"unit_id": "1", "label_id": "l1", "label_value": "yes"},
            {"unit_id": "2", "label_id": "l1", "label_value": "no"},
        ]
    )

    def _fake_export_inputs_from_repo(
        project_root, pheno_id, prior_rounds, *, corpus_record, corpus_id, corpus_path
    ):
        captured["export_inputs"] = {
            "project_root": project_root,
            "pheno_id": pheno_id,
            "prior_rounds": list(prior_rounds),
            "corpus_id": corpus_id,
            "corpus_path": corpus_path,
        }
        return notes_df, ann_df

    def _fake_load_label_config_bundle(
        project_root, pheno_id, labelset_id, prior_rounds, *, overrides=None
    ):
        captured["label_config_bundle"] = {
            "project_root": project_root,
            "pheno_id": pheno_id,
            "labelset_id": labelset_id,
            "prior_rounds": list(prior_rounds),
            "overrides": overrides,
        }
        return {"bundle": True}

    def _fake_session_from_env(paths, config):
        captured["session_config"] = {
            "backend": config.llm.backend,
            "llm_temperature": config.llm.temperature,
            "rag_chunk_size": config.rag.chunk_size,
            "embed_model_name": getattr(config.models, "embed_model_name", None),
            "rerank_model_name": getattr(config.models, "rerank_model_name", None),
        }
        captured["session_paths"] = paths
        return "session"

    def _fake_run_inference_experiments(**kwargs):
        captured["run_kwargs"] = kwargs
        df = pd.DataFrame(
            [
                {"unit_id": "1", "label_id": "l1", "prediction_value": "yes"},
                {"unit_id": "2", "label_id": "l1", "prediction_value": "no"},
            ]
        )
        return {"baseline": _DummyResult(df)}

    monkeypatch.setattr(
        project_experiments, "export_inputs_from_repo", _fake_export_inputs_from_repo
    )
    monkeypatch.setattr(
        project_experiments, "_load_label_config_bundle", _fake_load_label_config_bundle
    )
    monkeypatch.setattr(
        project_experiments.BackendSession, "from_env", staticmethod(_fake_session_from_env)
    )
    monkeypatch.setattr(
        project_experiments, "run_inference_experiments", _fake_run_inference_experiments
    )

    base_overrides = {
        "llm": {"backend": "azure", "temperature": 0.9},
        "rag": {"chunk_size": 321},
        "label_config": {"prompt": "custom"},
        "embedding_model_dir": "/models/embed",
        "reranker_model_dir": "/models/rerank",
    }
    sweeps = {"baseline": {"llm": {"temperature": 0.2}}}

    results, gold_df = project_experiments.run_project_inference_experiments(
        project_root=tmp_path,
        pheno_id="pheno1",
        prior_rounds=[1],
        labelset_id="ls1",
        phenotype_level="single_doc",
        sweeps=sweeps,
        base_outdir=tmp_path / "out",
        corpus_id="corp1",
        corpus_path=None,
        cfg_overrides_base=base_overrides,
    )

    assert gold_df.shape == (2, 3)
    assert isinstance(results.get("baseline"), _DummyResult)

    assert captured["export_inputs"] == {
        "project_root": tmp_path,
        "pheno_id": "pheno1",
        "prior_rounds": [1],
        "corpus_id": "corp1",
        "corpus_path": None,
    }
    assert captured["label_config_bundle"]["overrides"] == {"prompt": "custom"}

    assert captured["session_config"] == {
        "backend": "azure",
        "llm_temperature": 0.9,
        "rag_chunk_size": 321,
        "embed_model_name": "/models/embed",
        "rerank_model_name": "/models/rerank",
    }

    merged_sweep = captured["run_kwargs"]["sweeps"]["baseline"]
    assert merged_sweep["llm"]["temperature"] == 0.2
    assert merged_sweep["llm"]["backend"] == "azure"
    assert merged_sweep["rag"]["chunk_size"] == 321
    assert merged_sweep["label_config"] == {"prompt": "custom"}
    assert merged_sweep["models"]["embed_model_name"] == "/models/embed"
    assert merged_sweep["models"]["rerank_model_name"] == "/models/rerank"

    normalized_sweep = captured["run_kwargs"]["normalized_sweeps"]["baseline"]
    assert normalized_sweep["llm"]["temperature"] == 0.2
    assert "rag" not in normalized_sweep

    sweep_cfg = captured["run_kwargs"]["sweep_cfgs"]["baseline"]
    assert sweep_cfg.llm.backend == "azure"
    assert sweep_cfg.llm.temperature == 0.2
    assert sweep_cfg.rag.chunk_size == 321
    assert getattr(sweep_cfg.models, "embed_model_name") == "/models/embed"
    assert getattr(sweep_cfg.models, "rerank_model_name") == "/models/rerank"

    assert captured["run_kwargs"]["unit_ids"] == ["1", "2"]
    metrics_path = tmp_path / "out" / "baseline" / "metrics.json"
    assert metrics_path.exists()

    base_out = tmp_path / "out"
    gold_path = base_out / "gold_labels.parquet"
    collated_path = base_out / "experiments_predictions.parquet"
    summary_path = base_out / "experiments_metrics.json"

    assert gold_path.exists()
    assert collated_path.exists()
    assert summary_path.exists()

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "global" in data
    assert "labels" in data
    assert isinstance(data["labels"], dict)
    assert "l1" in data["labels"]
    assert 0.0 <= data["global"].get("overall_accuracy", 0.0) <= 1.0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "sweeps" in summary
    assert "baseline" in summary["sweeps"]


def test_run_project_inference_experiments_passes_prior_annotations(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    notes_df = pd.DataFrame(
        [
            {"doc_id": "1", "patient_icn": "p1", "text": "Note A"},
            {"doc_id": "2", "patient_icn": "p2", "text": "Note B"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {"unit_id": "1", "label_id": "l1", "label_value": "yes", "labelset_id": "ls1"},
            {"unit_id": "2", "label_id": "l1", "label_value": "no", "labelset_id": "ls1"},
        ]
    )

    def _fake_export_inputs_from_repo(*args, **kwargs):
        captured["export_inputs"] = {"args": args, "kwargs": kwargs}
        return notes_df.copy(), ann_df.copy()

    def _fake_load_label_config_bundle(*_args, **_kwargs):
        return {"bundle": True}

    def _fake_session_from_env(paths, config):
        captured["session_paths"] = paths
        captured["session_config"] = config
        return "session"

    def _fake_run_inference_experiments(**kwargs):
        captured["run_kwargs"] = kwargs
        df = pd.DataFrame(
            [
                {"unit_id": "1", "label_id": "l1", "prediction_value": "yes"},
                {"unit_id": "2", "label_id": "l1", "prediction_value": "no"},
            ]
        )
        return {"baseline": _DummyResult(df)}

    monkeypatch.setattr(
        project_experiments, "export_inputs_from_repo", _fake_export_inputs_from_repo
    )
    monkeypatch.setattr(
        project_experiments, "_load_label_config_bundle", _fake_load_label_config_bundle
    )
    monkeypatch.setattr(
        project_experiments.BackendSession, "from_env", staticmethod(_fake_session_from_env)
    )
    monkeypatch.setattr(
        project_experiments, "run_inference_experiments", _fake_run_inference_experiments
    )

    results, gold_df = project_experiments.run_project_inference_experiments(
        project_root=tmp_path,
        pheno_id="pheno1",
        prior_rounds=[1, 2],
        labelset_id="ls1",
        phenotype_level="single_doc",
        sweeps={"baseline": {}},
        base_outdir=tmp_path / "out",
        corpus_id=None,
        corpus_path=None,
        cfg_overrides_base=None,
    )

    assert isinstance(results.get("baseline"), _DummyResult)
    assert gold_df.shape == (2, 3)
    assert list(gold_df["unit_id"]) == ["1", "2"]
    assert_frame_equal(captured["run_kwargs"]["ann_df"].reset_index(drop=True), ann_df)
    assert captured["run_kwargs"]["unit_ids"] == ["1", "2"]


def test_run_project_inference_experiments_rebuilds_sessions_for_backend_overrides(
    monkeypatch, tmp_path
):
    notes_df = pd.DataFrame(
        [
            {"doc_id": "1", "patient_icn": "p1", "text": "Note A"},
            {"doc_id": "2", "patient_icn": "p2", "text": "Note B"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {"unit_id": "1", "label_id": "l1", "label_value": "yes"},
            {"unit_id": "2", "label_id": "l1", "label_value": "no"},
        ]
    )

    monkeypatch.setattr(
        project_experiments, "export_inputs_from_repo", lambda *_, **__: (notes_df, ann_df)
    )
    monkeypatch.setattr(
        project_experiments, "_load_label_config_bundle", lambda *_, **__: {"bundle": True}
    )

    def _fail_shared_session(*_args, **_kwargs):
        raise AssertionError("Shared session should not be created for backend overrides")

    monkeypatch.setattr(
        project_experiments.BackendSession, "from_env", staticmethod(_fail_shared_session)
    )

    captured: dict[str, object] = {}

    def _stub_run_inference_experiments(**kwargs):
        captured.update(kwargs)
        df = pd.DataFrame(
            [
                {"unit_id": "1", "label_id": "l1", "prediction_value": "yes"},
                {"unit_id": "2", "label_id": "l1", "prediction_value": "no"},
            ]
        )
        return {"raggy": _DummyResult(df)}

    monkeypatch.setattr(
        project_experiments, "run_inference_experiments", _stub_run_inference_experiments
    )

    sweeps = {"raggy": {"rag": {"chunk_size": 11}}}

    results, gold_df = project_experiments.run_project_inference_experiments(
        project_root=tmp_path,
        pheno_id="pheno1",
        prior_rounds=[1],
        labelset_id="ls1",
        phenotype_level="single_doc",
        sweeps=sweeps,
        base_outdir=tmp_path / "out",
        corpus_id=None,
        corpus_path=None,
        cfg_overrides_base=None,
    )

    assert isinstance(results.get("raggy"), _DummyResult)
    assert gold_df.shape == (2, 3)
    assert captured.get("session") is None


def test_inference_experiments_rag_overrides_disable_shared_session(
    monkeypatch, tmp_path
):
    notes_df = pd.DataFrame(
        [
            {"doc_id": "1", "patient_icn": "p1", "text": "Note A"},
            {"doc_id": "2", "patient_icn": "p2", "text": "Note B"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {"unit_id": "1", "label_id": "l1", "label_value": "yes"},
            {"unit_id": "2", "label_id": "l1", "label_value": "no"},
        ]
    )

    monkeypatch.setattr(
        project_experiments, "export_inputs_from_repo", lambda *_, **__: (notes_df, ann_df)
    )
    monkeypatch.setattr(
        project_experiments, "_load_label_config_bundle", lambda *_, **__: {"bundle": True}
    )

    def _fail_shared_session(*_args, **_kwargs):
        raise AssertionError("Shared session should not be created")

    monkeypatch.setattr(
        project_experiments.BackendSession, "from_env", staticmethod(_fail_shared_session)
    )

    captured: dict[str, object] = {}

    def _stub_run_inference_experiments(**kwargs):
        captured.update(kwargs)
        df = pd.DataFrame(
            [
                {"unit_id": "1", "label_id": "l1", "prediction_value": "yes"},
                {"unit_id": "2", "label_id": "l1", "prediction_value": "no"},
            ]
        )
        return {
            "baseline": _DummyResult(df),
            "rag_small": _DummyResult(df),
        }

    monkeypatch.setattr(
        project_experiments, "run_inference_experiments", _stub_run_inference_experiments
    )

    base_overrides = {"rag": {"chunk_size": 256}}
    sweeps = {
        "baseline": {},
        "rag_small": {"rag": {"chunk_size": 64}},
    }

    results, gold_df = project_experiments.run_project_inference_experiments(
        project_root=tmp_path,
        pheno_id="pheno1",
        prior_rounds=[1],
        labelset_id="ls1",
        phenotype_level="single_doc",
        sweeps=sweeps,
        base_outdir=tmp_path / "out",
        corpus_id=None,
        corpus_path=None,
        cfg_overrides_base=base_overrides,
    )

    assert isinstance(results.get("baseline"), _DummyResult)
    assert isinstance(results.get("rag_small"), _DummyResult)
    assert gold_df.shape == (2, 3)

    assert captured.get("session") is None

    forwarded_sweeps = captured["sweeps"]
    assert forwarded_sweeps["baseline"]["rag"]["chunk_size"] == 256
    assert forwarded_sweeps["rag_small"]["rag"]["chunk_size"] == 64

    sweep_cfgs = captured["sweep_cfgs"]
    assert sweep_cfgs["baseline"].rag.chunk_size == 256
    assert sweep_cfgs["rag_small"].rag.chunk_size == 64


def test_inference_experiments_merge_baseline_and_delta_configs(monkeypatch, tmp_path):
    notes_df = pd.DataFrame(
        [
            {"doc_id": "1", "patient_icn": "p1", "text": "Note A"},
        ]
    )
    ann_df = pd.DataFrame(
        [
            {"unit_id": "1", "label_id": "l1", "label_value": "yes"},
        ]
    )

    monkeypatch.setattr(
        project_experiments, "export_inputs_from_repo", lambda *_, **__: (notes_df, ann_df)
    )
    monkeypatch.setattr(
        project_experiments, "_load_label_config_bundle", lambda *_, **__: {"bundle": True}
    )

    captured: dict[str, object] = {"sessions": []}

    def _record_session(paths, config):
        captured["sessions"].append({
            "paths": paths,
            "backend": config.llm.backend,
            "chunk_size": config.rag.chunk_size,
            "temperature": config.llm.temperature,
        })
        return "session"

    monkeypatch.setattr(
        project_experiments.BackendSession, "from_env", staticmethod(_record_session)
    )

    def _stub_run_inference_experiments(**kwargs):
        captured.update(kwargs)
        df = pd.DataFrame(
            [
                {"unit_id": "1", "label_id": "l1", "prediction_value": "yes"},
            ]
        )
        return {
            "cool": _DummyResult(df),
            "warm": _DummyResult(df),
        }

    monkeypatch.setattr(
        project_experiments, "run_inference_experiments", _stub_run_inference_experiments
    )

    base_overrides = {
        "llm": {"backend": "azure", "temperature": 0.4},
        "rag": {"chunk_size": 128},
    }
    sweeps = {
        "cool": {"llm": {"temperature": 0.1}},
        "warm": {"llm": {"temperature": 0.9}},
    }

    results, gold_df = project_experiments.run_project_inference_experiments(
        project_root=tmp_path,
        pheno_id="pheno1",
        prior_rounds=[1],
        labelset_id="ls1",
        phenotype_level="single_doc",
        sweeps=sweeps,
        base_outdir=tmp_path / "out",
        corpus_id=None,
        corpus_path=None,
        cfg_overrides_base=base_overrides,
    )

    assert isinstance(results.get("cool"), _DummyResult)
    assert isinstance(results.get("warm"), _DummyResult)
    assert gold_df.shape == (1, 3)

    assert len(captured.get("sessions", [])) == 1

    sweeps_with_base = captured["sweeps"]
    assert sweeps_with_base["cool"]["llm"]["temperature"] == 0.1
    assert sweeps_with_base["cool"]["rag"]["chunk_size"] == 128
    assert sweeps_with_base["warm"]["llm"]["temperature"] == 0.9
    assert sweeps_with_base["warm"]["rag"]["chunk_size"] == 128

    sweep_cfgs = captured["sweep_cfgs"]
    assert sweep_cfgs["cool"].llm.backend == "azure"
    assert sweep_cfgs["cool"].llm.temperature == 0.1
    assert sweep_cfgs["cool"].rag.chunk_size == 128
    assert sweep_cfgs["warm"].llm.backend == "azure"
    assert sweep_cfgs["warm"].llm.temperature == 0.9
    assert sweep_cfgs["warm"].rag.chunk_size == 128
def test_inference_sweeps_forward_final_topk(monkeypatch, tmp_path):
    def _fake_export_inputs_from_repo(*args, **kwargs):
        notes_df = pd.DataFrame(
            {"unit_id": ["1"], "patient_icn": ["p"], "doc_id": ["d"]}
        )
        ann_df = pd.DataFrame({"unit_id": ["1"], "label_id": ["0"], "label_value": ["y"]})
        return notes_df, ann_df

    def _fake_load_label_config_bundle(*args, **kwargs):
        class _DummyBundle:
            def with_current_fallback(self, label_config):
                return self

        return _DummyBundle()

    def _fake_session_from_env(*args, **kwargs):
        class _DummySession:
            models = None
            store = None

        return _DummySession()

    captured = {}

    def _fake_run_inference_experiments(**kwargs):
        captured["sweeps"] = kwargs.get("sweeps") or {}

        class _DummyResult:
            dataframe = pd.DataFrame()
            artifacts = {}
            outdir = tmp_path / "out"
            cfg_overrides = {}
            name = "topk"

        return {"topk": _DummyResult()}

    monkeypatch.setattr(
        project_experiments, "export_inputs_from_repo", _fake_export_inputs_from_repo
    )
    monkeypatch.setattr(
        project_experiments, "_load_label_config_bundle", _fake_load_label_config_bundle
    )
    monkeypatch.setattr(
        project_experiments.BackendSession, "from_env", staticmethod(_fake_session_from_env)
    )
    monkeypatch.setattr(
        project_experiments, "run_inference_experiments", _fake_run_inference_experiments
    )

    project_experiments.run_project_inference_experiments(
        project_root=tmp_path,
        pheno_id="p1",
        prior_rounds=[1],
        labelset_id="ls",
        phenotype_level="single_doc",
        sweeps={"topk": {"rag": {"top_k_final": 11}}},
        base_outdir=tmp_path / "out",
        corpus_id=None,
        corpus_path=None,
        cfg_overrides_base=None,
    )

    rag_cfg = captured.get("sweeps", {}).get("topk", {}).get("rag", {})
    assert rag_cfg.get("top_k_final") == 11
    assert rag_cfg.get("per_label_topk") == 11


def test_baseline_topk_final_used_with_normalized_sweeps(monkeypatch, tmp_path):
    def _fake_export_inputs_from_repo(*args, **kwargs):
        notes_df = pd.DataFrame(
            {"unit_id": ["1"], "patient_icn": ["p"], "doc_id": ["d"], "text": ["note"]}
        )
        ann_df = pd.DataFrame({"unit_id": ["1"], "label_id": ["0"], "label_value": ["y"]})
        return notes_df, ann_df

    def _fake_load_label_config_bundle(*args, **kwargs):
        class _DummyBundle:
            def with_current_fallback(self, label_config):
                return self

        return _DummyBundle()

    class _DummySession:
        models = None
        store = None

    captured: dict[str, list[dict]] = {}

    def _fake_run_inference(**kwargs):
        captured.setdefault("cfg_overrides", []).append(kwargs.get("cfg_overrides") or {})
        df = pd.DataFrame(
            [
                {"unit_id": "1", "label_id": "l1", "prediction_value": "yes"},
            ]
        )
        return df, {}

    monkeypatch.setattr(
        project_experiments, "export_inputs_from_repo", _fake_export_inputs_from_repo
    )
    monkeypatch.setattr(
        project_experiments, "_load_label_config_bundle", _fake_load_label_config_bundle
    )
    monkeypatch.setattr(
        project_experiments.BackendSession, "from_env", staticmethod(lambda *_, **__: _DummySession())
    )
    monkeypatch.setattr(experiments.BackendSession, "from_env", staticmethod(lambda *_, **__: _DummySession()))
    monkeypatch.setattr(experiments, "run_inference", _fake_run_inference)

    project_experiments.run_project_inference_experiments(
        project_root=tmp_path,
        pheno_id="p1",
        prior_rounds=[1],
        labelset_id="ls",
        phenotype_level="single_doc",
        sweeps={"topk": {}},
        base_outdir=tmp_path / "out",
        corpus_id=None,
        corpus_path=None,
        cfg_overrides_base={"rag": {"top_k_final": 15}},
    )

    assert captured.get("cfg_overrides")
    rag_cfg = captured["cfg_overrides"][0].get("rag", {})
    assert rag_cfg.get("top_k_final") == 15


def test_build_gold_uses_date_values_for_consensus():
    ann_df = pd.DataFrame(
        [
            {
                "unit_id": "u1",
                "label_id": "onset",
                "label_value": "date provided",
                "label_value_date": pd.Timestamp("2024-01-01"),
            },
            {
                "unit_id": "u1",
                "label_id": "onset",
                "label_value": "date provided",
                "label_value_date": pd.Timestamp("2024-01-01"),
            },
        ]
    )

    gold_df = project_experiments.build_gold_from_ann(ann_df)

    assert gold_df.shape == (1, 3)
    assert gold_df.loc[0, "gold_value"].startswith("2024-01-01")

    pred_df = pd.DataFrame(
        [
            {"unit_id": "u1", "label_id": "onset", "prediction_value": "2024-01-03"},
        ]
    )

    metrics = project_experiments.compute_experiment_metrics(
        gold_df,
        pred_df,
        label_config_bundle=None,
        labelset_id=None,
        ann_df=ann_df,
    )

    onset_metrics = metrics["labels"].get("onset", {})
    assert onset_metrics.get("n") == 1
    assert onset_metrics.get("within_3d") == 1.0
    assert onset_metrics.get("exact_match") == 0.0
