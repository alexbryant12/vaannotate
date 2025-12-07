from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend import (
    InferenceExperimentResult,
    run_inference_experiments,
)
from vaannotate.vaannotate_ai_backend.label_configs import LabelConfigBundle


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ai_backend"


def _load_label_config() -> dict:
    with open(DATA_DIR / "label_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    notes = pd.read_csv(DATA_DIR / "notes.csv")
    annotations = pd.read_csv(DATA_DIR / "annotations.csv")
    return notes, annotations


def test_run_inference_experiments_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Basic smoke test for the experiment runner.

    Verifies that:
    - we get one result per experiment name
    - each experiment writes its predictions files
    - the manifest JSON contains the expected keys
    """

    label_config = _load_label_config()
    bundle = LabelConfigBundle(
        current=label_config,
        current_labelset_id=label_config.get("_meta", {}).get("labelset_id"),
    )

    notes_df, ann_df = _load_inputs()
    captured_backends: list[str | None] = []

    class _StubSession:
        models: None = None
        store: None = None

    class _StubInference:
        def __init__(self, paths, cfg=None, **_kwargs):
            self.paths = paths
            self.cfg = cfg

        def run(self, unit_ids=None):  # noqa: ANN001 - match real signature
            parquet_notes = pd.read_parquet(self.paths.notes_path)
            outdir = Path(self.paths.outdir)
            df = pd.DataFrame(
                {
                    "unit_id": parquet_notes["unit_id"].astype(str),
                    "doc_id": parquet_notes.get("doc_id", parquet_notes.get("note_id")),
                    "label_id": "pneumonitis",
                    "label_option_id": "yes",
                }
            )
            df.to_parquet(outdir / "inference_predictions.parquet", index=False)
            df.to_json(outdir / "inference_predictions.json", orient="records", lines=True)
            if self.cfg is not None:
                captured_backends.append(self.cfg.llm.backend)
            else:
                captured_backends.append(None)
            return df

    def _build_stub_runner(*args, **kwargs):
        paths = kwargs.get("paths") if "paths" in kwargs else args[0]
        cfg = kwargs.get("cfg")
        return _StubInference(paths, cfg=cfg)

    # Ensure our experiments use the stubbed inference pipeline instead of
    # actually loading models or calling an LLM backend.
    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.orchestrator.build_inference_runner",
        _build_stub_runner,
    )

    sweeps = {
        "local_backend": {"llm": {"backend": "local"}},
        "azure_backend": {"llm": {"backend": "azure"}},
    }

    results = run_inference_experiments(
        notes_df=notes_df,
        ann_df=ann_df,
        base_outdir=tmp_path,
        sweeps=sweeps,
        label_config_bundle=bundle,
        session=_StubSession(),
    )

    # One result per experiment
    assert set(results.keys()) == set(sweeps.keys())

    for name, result in results.items():
        assert isinstance(result, InferenceExperimentResult)
        assert result.name == name
        # Predictions files should exist under each experiment's outdir
        parquet_path = result.outdir / "inference_predictions.parquet"
        json_path = result.outdir / "inference_predictions.json"
        assert parquet_path.exists()
        assert json_path.exists()
        assert result.artifacts["predictions"].endswith("inference_predictions.parquet")
        assert result.artifacts["predictions_json"].endswith("inference_predictions.json")

        df = result.dataframe
        assert not df.empty
        assert set(df["unit_id"].unique()) >= {"1001", "1002"}

    # We should have one captured backend per experiment, with the expected values
    assert sorted(captured_backends) == ["azure", "local"]

    # Manifest should list all experiments
    manifest_path = tmp_path / "experiments.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest.keys()) == set(sweeps.keys())


def test_run_inference_experiments_scopes_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit_ids = ["u1", "u2"]
    notes_df = pd.DataFrame(
        {
            "unit_id": ["u1", "u2", "u3"],
            "note_id": [1, 2, 3],
            "text": ["a", "b", "c"],
        }
    )
    ann_df = pd.DataFrame(
        {
            "unit_id": ["u1", "u2", "u3"],
            "annotation": ["x", "y", "z"],
        }
    )

    class _StubSession:
        def __init__(self) -> None:
            self.models = None
            self.store = None

    stub_session = _StubSession()
    captured_calls: list[dict[str, Any]] = []

    def _stub_run_inference(**kwargs):
        captured_calls.append(
            {
                "notes_units": set(kwargs["notes_df"]["unit_id"].astype(str).unique()),
                "ann_units": set(kwargs["ann_df"]["unit_id"].astype(str).unique()),
                "unit_ids": kwargs.get("unit_ids"),
                "session": kwargs.get("session"),
            }
        )
        df = kwargs["notes_df"].copy()
        return df, {"predictions": "pred.parquet", "predictions_json": "pred.json"}

    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.experiments.run_inference",
        _stub_run_inference,
    )

    sweeps = {"exp_a": {}, "exp_b": {}}

    results = run_inference_experiments(
        notes_df=notes_df,
        ann_df=ann_df,
        base_outdir=tmp_path,
        sweeps=sweeps,
        unit_ids=unit_ids,
        session=stub_session,
    )

    assert set(results.keys()) == set(sweeps.keys())
    assert len(captured_calls) == len(sweeps)
    for call in captured_calls:
        assert call["notes_units"] == set(unit_ids)
        assert call["ann_units"] == set(unit_ids)
        assert call["unit_ids"] == unit_ids
        assert call["session"] is stub_session


def test_run_inference_experiments_applies_rag_overrides_to_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes_df = pd.DataFrame({"unit_id": ["a"], "note_id": [1], "text": ["t"]})
    ann_df = pd.DataFrame({"unit_id": ["a"], "annotation": ["z"]})

    created_normalize_flags: list[bool] = []

    class _StubSession:
        def __init__(self, normalize: bool) -> None:
            self.normalize = normalize
            self.models = None
            self.store = None

    def _stub_from_env(_paths, cfg):
        session = _StubSession(cfg.rag.normalize_embeddings)
        created_normalize_flags.append(session.normalize)
        return session

    def _stub_run_inference(**kwargs: Any):
        df = kwargs["notes_df"].copy()
        return df, {"predictions": "pred.parquet", "predictions_json": "pred.json"}

    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.experiments.BackendSession.from_env",
        _stub_from_env,
    )
    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.experiments.run_inference",
        _stub_run_inference,
    )

    sweeps = {
        "normalized": {},
        "unnormalized": {"rag": {"normalize_embeddings": False}},
    }

    results = run_inference_experiments(
        notes_df=notes_df,
        ann_df=ann_df,
        base_outdir=tmp_path,
        sweeps=sweeps,
    )

    assert set(results.keys()) == set(sweeps.keys())
    assert created_normalize_flags == [True, False]
