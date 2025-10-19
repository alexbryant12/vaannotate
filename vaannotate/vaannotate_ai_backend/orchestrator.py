
from __future__ import annotations
import os, json, uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from . import engine

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
    paths = engine.Paths(outdir=str(outdir))
    # Persist inputs to the files expected by the engine
    notes_path = Path(paths.notes_path)
    ann_path = Path(paths.annotations_path)
    notes_df.to_parquet(notes_path, index=False)
    ann_df.to_parquet(ann_path, index=False)

    # Build config
    cfg = engine.OrchestratorConfig.default(outdir=str(outdir))
    if cfg_overrides:
        # Shallow merge for convenience; nested dataclasses have their own defaults
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    orch = engine.ActiveLearningLLMFirst(paths=paths, cfg=cfg, label_config=label_config or {})
    final_df = orch.run()  # engine returns DataFrame

    # Also write ai_next_batch.csv to a standard location
    ai_dir = Path(outdir) / "imports" / "ai"
    _ensure_dir(ai_dir)
    csv_path = ai_dir / "ai_next_batch.csv"
    # Minimal CSV for RoundBuilder
    cols = [c for c in ["unit_id","label_id","doc_id","patienticn","selection_reason"] if c in final_df.columns]
    final_df[cols].to_csv(csv_path, index=False)

    artifacts = {
        "ai_next_batch_csv": str(csv_path),
        "buckets": {
            "disagreement": str(Path(outdir)/"bucket_disagreement.parquet"),
            "llm_uncertain": str(Path(outdir)/"bucket_llm_uncertain.parquet"),
            "llm_certain": str(Path(outdir)/"bucket_llm_certain.parquet"),
            "diversity": str(Path(outdir)/"bucket_diversity.parquet"),
        },
        "final_labels": str(Path(outdir)/"final_llm_labels.parquet") if (Path(outdir)/"final_llm_labels.parquet").exists() else None
    }
    return final_df, artifacts
