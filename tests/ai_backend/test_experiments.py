from __future__ import annotations

import json
from pathlib import Path
import sys

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
