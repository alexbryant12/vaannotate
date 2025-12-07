from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend import build_next_batch, run_inference
from vaannotate.vaannotate_ai_backend.label_configs import LabelConfigBundle


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ai_backend"


def _load_label_config() -> dict:
    with open(DATA_DIR / "label_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    notes = pd.read_csv(DATA_DIR / "notes.csv")
    annotations = pd.read_csv(DATA_DIR / "annotations.csv")
    return notes, annotations


def test_active_learning_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    label_config = _load_label_config()
    bundle = LabelConfigBundle(
        current=label_config,
        current_labelset_id=label_config.get("_meta", {}).get("labelset_id"),
    )

    notes_df, ann_df = _load_inputs()

    class _StubPipeline:
        def __init__(self, paths):
            self.paths = paths

        def run(self):
            parquet_notes = pd.read_parquet(self.paths.notes_path)
            outdir = Path(self.paths.outdir)

            buckets = {
                "bucket_disagreement.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
                "bucket_llm_uncertain.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
                "bucket_llm_certain.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
                "bucket_diversity.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
            }
            for name, df in buckets.items():
                df.to_parquet(outdir / name, index=False)

            return pd.DataFrame(
                {
                    "unit_id": parquet_notes["unit_id"].astype(str),
                    "doc_id": parquet_notes.get("doc_id", parquet_notes.get("note_id")),
                    "label_id": "pneumonitis",
                    "selection_reason": "dummy",
                }
            )

    def _build_stub_runner(*_args, **_kwargs):
        paths = _kwargs.get("paths") if "paths" in _kwargs else _args[0]
        return _StubPipeline(paths)

    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.orchestrator.build_active_learning_runner",
        _build_stub_runner,
    )

    final_df, artifacts = build_next_batch(
        notes_df,
        ann_df,
        tmp_path,
        label_config_bundle=bundle,
    )

    assert not final_df.empty
    assert set(final_df["unit_id"].unique()) == {"1001", "1002", "1003"}
    assert (tmp_path / "ai_next_batch.csv").exists()

    for bucket in (
        "bucket_disagreement.parquet",
        "bucket_llm_uncertain.parquet",
        "bucket_llm_certain.parquet",
        "bucket_diversity.parquet",
    ):
        assert (tmp_path / bucket).exists()

    assert artifacts["ai_next_batch_csv"].endswith("ai_next_batch.csv")


@pytest.mark.parametrize(
    "backend, assisted_review, disagreement_enabled, diversity_enabled",
    [
        ("local", False, False, False),
        ("local", True, True, False),
        ("azure", False, True, True),
    ],
)
def test_active_learning_config_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    assisted_review: bool,
    disagreement_enabled: bool,
    diversity_enabled: bool,
) -> None:
    label_config = _load_label_config()
    bundle = LabelConfigBundle(
        current=label_config,
        current_labelset_id=label_config.get("_meta", {}).get("labelset_id"),
    )

    notes_df, ann_df = _load_inputs()
    captured = {}

    class _StubPipeline:
        def __init__(self, paths):
            self.paths = paths

        def run(self):
            parquet_notes = pd.read_parquet(self.paths.notes_path)
            outdir = Path(self.paths.outdir)

            (outdir / "bucket_disagreement.parquet").touch()
            (outdir / "bucket_llm_uncertain.parquet").touch()
            (outdir / "bucket_llm_certain.parquet").touch()
            (outdir / "bucket_diversity.parquet").touch()

            return pd.DataFrame(
                {
                    "unit_id": parquet_notes["unit_id"].astype(str),
                    "doc_id": parquet_notes.get("doc_id", parquet_notes.get("note_id")),
                    "label_id": "pneumonitis",
                    "selection_reason": "matrix",
                }
            )

    def _build_stub_runner(*_args, **_kwargs):
        paths = _kwargs.get("paths") if "paths" in _kwargs else _args[0]
        cfg = _kwargs.get("cfg")
        captured["backend"] = cfg.llm.backend
        captured["assisted_review"] = getattr(cfg, "assisted_review", {})
        captured["pct_disagreement"] = cfg.select.pct_disagreement
        captured["pct_diversity"] = cfg.select.pct_diversity
        return _StubPipeline(paths)

    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.orchestrator.build_active_learning_runner",
        _build_stub_runner,
    )

    overrides = {
        "llm": {"backend": backend},
        "assisted_review": {"enabled": assisted_review, "top_snippets": 1},
        "select": {
            "pct_disagreement": 0.3 if disagreement_enabled else 0.0,
            "pct_diversity": 0.3 if diversity_enabled else 0.0,
        },
    }

    final_df, _ = build_next_batch(
        notes_df,
        ann_df,
        tmp_path,
        label_config_bundle=bundle,
        cfg_overrides=overrides,
    )

    assert not final_df.empty
    assert (tmp_path / "ai_next_batch.csv").exists()
    assert captured["backend"] == backend
    assert captured["assisted_review"].get("enabled") is assisted_review
    assert (
        captured["pct_disagreement"] == (0.3 if disagreement_enabled else 0.0)
    )
    assert captured["pct_diversity"] == (0.3 if diversity_enabled else 0.0)


@pytest.mark.parametrize(
    "backend, rag_enabled, include_reasoning",
    [
        ("local", False, False),
        ("local", True, False),
        ("azure", True, True),
    ],
)
def test_inference_config_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    rag_enabled: bool,
    include_reasoning: bool,
) -> None:
    label_config = _load_label_config()
    bundle = LabelConfigBundle(
        current=label_config,
        current_labelset_id=label_config.get("_meta", {}).get("labelset_id"),
    )

    notes_df, ann_df = _load_inputs()
    captured = {}

    class _StubInference:
        def __init__(self, paths):
            self.paths = paths

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
            return df

    def _build_stub_runner(*_args, **_kwargs):
        paths = _kwargs.get("paths") if "paths" in _kwargs else _args[0]
        cfg = _kwargs.get("cfg")
        captured["backend"] = cfg.llm.backend
        captured["rag_use_keywords"] = cfg.rag.use_keywords
        captured["rag_use_mmr"] = cfg.rag.use_mmr
        captured["include_reasoning"] = cfg.llm.include_reasoning
        return _StubInference(paths)

    monkeypatch.setattr(
        "vaannotate.vaannotate_ai_backend.orchestrator.build_inference_runner",
        _build_stub_runner,
    )

    overrides = {
        "llm": {"backend": backend, "include_reasoning": include_reasoning},
        "rag": {"use_keywords": rag_enabled, "use_mmr": rag_enabled},
    }

    predictions, artifacts = run_inference(
        notes_df,
        ann_df,
        tmp_path,
        label_config_bundle=bundle,
        cfg_overrides=overrides,
    )

    assert not predictions.empty
    assert (tmp_path / "inference_predictions.parquet").exists()
    assert (tmp_path / "inference_predictions.json").exists()
    assert set(predictions["unit_id"].unique()) == {"1001", "1002", "1003"}
    assert artifacts["predictions"].endswith("inference_predictions.parquet")
    assert captured["backend"] == backend
    assert captured["rag_use_keywords"] is rag_enabled
    assert captured["rag_use_mmr"] is rag_enabled
    assert captured["include_reasoning"] is include_reasoning
