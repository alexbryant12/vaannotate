from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from vaannotate.vaannotate_ai_backend import project_experiments


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

    assert captured["run_kwargs"]["unit_ids"] == ["1", "2"]
    metrics_path = tmp_path / "out" / "baseline" / "metrics.json"
    assert metrics_path.exists()


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
