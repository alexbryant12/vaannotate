from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from . import engine


def _default_paths(outdir: Path) -> engine.Paths:
    notes_path = outdir / "notes.parquet"
    annotations_path = outdir / "annotations.parquet"
    return engine.Paths(
        notes_path=str(notes_path),
        annotations_path=str(annotations_path),
        outdir=str(outdir),
    )


def _apply_overrides(target: object, overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if key == "phenotype_level":
            # Not currently consumed directly by the engine config.
            continue
        if isinstance(value, Mapping):
            current = getattr(target, key, None)
            if current is not None and not isinstance(current, (str, bytes, int, float, bool)):
                _apply_overrides(current, value)
            else:
                setattr(target, key, dict(value))
        else:
            setattr(target, key, value)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_next_batch(
    notes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    outdir: Path,
    label_config: Optional[dict] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """High-level entrypoint consumed by AdminApp/RoundBuilder.
    Writes intermediates under outdir and returns (final_df, artifacts).
    """
    outdir = Path(outdir)
    _ensure_dir(outdir)
    paths = _default_paths(outdir)
    # Persist inputs to the files expected by the engine
    notes_path = Path(paths.notes_path)
    ann_path = Path(paths.annotations_path)
    notes_df.to_parquet(notes_path, index=False)
    ann_df.to_parquet(ann_path, index=False)

    # Build config
    cfg = engine.OrchestratorConfig()
    if cfg_overrides:
        _apply_overrides(cfg, cfg_overrides)

    orch = engine.ActiveLearningLLMFirst(paths=paths, cfg=cfg, label_config=label_config or {})
    final_df = orch.run()  # engine returns DataFrame

    normalized = final_df.copy()
    rename_map = {}
    if "patienticn" in normalized.columns:
        rename_map["patienticn"] = "patient_icn"
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    csv_path = outdir / "ai_next_batch.csv"
    csv_columns = [
        col
        for col in [
            "unit_id",
            "patient_icn",
            "doc_id",
            "label_id",
            "selection_reason",
            "strata_key",
        ]
        if col in normalized.columns
    ]
    if csv_columns:
        normalized.to_csv(csv_path, columns=csv_columns, index=False)
    else:
        normalized.to_csv(csv_path, index=False)

    artifacts = {
        "ai_next_batch_csv": str(csv_path),
        "buckets": {
            "disagreement": str(Path(outdir) / "bucket_disagreement.parquet"),
            "llm_uncertain": str(Path(outdir) / "bucket_llm_uncertain.parquet"),
            "llm_certain": str(Path(outdir) / "bucket_llm_certain.parquet"),
            "diversity": str(Path(outdir) / "bucket_diversity.parquet"),
        },
        "final_labels": str(Path(outdir) / "final_llm_labels.parquet")
        if (Path(outdir) / "final_llm_labels.parquet").exists()
        else None,
    }
    return normalized, artifacts
