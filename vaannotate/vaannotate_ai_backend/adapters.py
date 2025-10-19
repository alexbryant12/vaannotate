
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import os, json, sqlite3
import pandas as pd

from .orchestrator import build_next_batch

def _read_corpus_db(corpus_db: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(corpus_db))
    try:
        # Documents table assumed: doc_id TEXT, patienticn TEXT, text TEXT, metadata_json TEXT, date_note TEXT, notetype TEXT
        df = pd.read_sql_query(            """
            SELECT d.patienticn, d.doc_id, d.text,
                   COALESCE(d.date_note,'') as date_note,
                   COALESCE(d.notetype,'') as notetype,
                   COALESCE(d.metadata_json,'{}') as document_metadata_json
            FROM documents d
            """, con)
        return df
    finally:
        con.close()

def _read_round_aggregate(round_db: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(round_db))
    try:
        # Long-format annotations
        df = pd.read_sql_query(            """
            SELECT round_id, unit_id, doc_id, label_id, reviewer_id,
                   label_value,
                   COALESCE(reviewer_notes,'') AS reviewer_notes,
                   COALESCE(rationales_json,'[]') AS rationales_json,
                   COALESCE(document_text,'') AS document_text,
                   COALESCE(document_metadata_json,'{}') AS document_metadata_json,
                   COALESCE(label_rules,'') AS label_rules
            FROM annotations
            """, con)
        return df
    finally:
        con.close()

def export_inputs_from_repo(project_root: Path, pheno_id: str, prior_rounds: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root)
    # Corpus DB at: phenotypes/<pheno_id>/corpus/corpus.db
    corpus_db = root / "phenotypes" / pheno_id / "corpus" / "corpus.db"
    notes_df = _read_corpus_db(corpus_db)

    # Aggregate from selected rounds: phenotypes/<pheno_id>/rounds/round_<n>/round_aggregate.db
    ann_frames = []
    for r in prior_rounds:
        round_db = root / "phenotypes" / pheno_id / "rounds" / f"round_{r}" / "round_aggregate.db"
        if round_db.exists():
            ann_frames.append(_read_round_aggregate(round_db))
    ann_df = pd.concat(ann_frames, ignore_index=True) if ann_frames else pd.DataFrame(columns=[
        "round_id","unit_id","doc_id","label_id","reviewer_id","label_value",
        "reviewer_notes","rationales_json","document_text","document_metadata_json","label_rules"
    ])
    return notes_df, ann_df

def run_ai_backend_and_collect(project_root: Path, pheno_id: str, prior_rounds: List[int], round_dir: Path, cfg_overrides: Optional[Dict[str,Any]]=None) -> Path:
    notes_df, ann_df = export_inputs_from_repo(project_root, pheno_id, prior_rounds)
    outdir = Path(round_dir) / "imports" / "ai"
    outdir.mkdir(parents=True, exist_ok=True)
    # Optional: load per-phenotype label_config.json if present
    label_config_path = Path(project_root) / "phenotypes" / pheno_id / "ai" / "label_config.json"
    label_config = None
    if label_config_path.exists():
        try:
            label_config = json.loads(label_config_path.read_text(encoding="utf-8"))
        except Exception:
            label_config = None
    final_df, artifacts = build_next_batch(notes_df, ann_df, outdir=outdir, label_config=label_config, cfg_overrides=cfg_overrides or {})
    return Path(artifacts["ai_next_batch_csv"])
