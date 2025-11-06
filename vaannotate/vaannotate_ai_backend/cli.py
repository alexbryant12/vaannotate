
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from .label_configs import LabelConfigBundle
from .orchestrator import build_next_batch

def main():
    ap = argparse.ArgumentParser(description="VAAnnotate AI Backend (LLM-first)")
    ap.add_argument("--notes", required=True, help="notes.parquet/.csv/.jsonl")
    ap.add_argument("--annotations", required=True, help="annotations.parquet/.csv/.jsonl")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--label-config", default=None, help="optional label_config.json")
    ap.add_argument("--cfg", default=None, help="optional cfg_overrides.json")
    args = ap.parse_args()

    def _read(p: str):
        pth = Path(p); ext = pth.suffix.lower()
        if ext in (".parquet",".pq"): return pd.read_parquet(pth)
        if ext == ".csv": return pd.read_csv(pth)
        if ext == ".jsonl": return pd.read_json(pth, orient="records", lines=True)
        raise SystemExit(f"Unsupported input: {p}")

    notes_df = _read(args.notes); ann_df = _read(args.annotations)
    label_config = json.loads(Path(args.label_config).read_text()) if args.label_config else None
    cfg_overrides = json.loads(Path(args.cfg).read_text()) if args.cfg else None

    bundle = LabelConfigBundle(current=label_config) if label_config else None
    final_df, artifacts = build_next_batch(
        notes_df,
        ann_df,
        outdir=Path(args.outdir),
        label_config_bundle=bundle,
        cfg_overrides=cfg_overrides,
    )
    print(str(artifacts["ai_next_batch_csv"]))

if __name__ == "__main__":
    main()
