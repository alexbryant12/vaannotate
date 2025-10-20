
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import json
import sqlite3

import pandas as pd

from . import __version__
from .orchestrator import build_next_batch


@dataclass
class BackendResult:
    csv_path: Path
    dataframe: pd.DataFrame
    artifacts: Dict[str, Any]
    params_path: Path

def _resolve_phenotype_dir(project_root: Path, pheno_id: str) -> Path:
    project_db = Path(project_root) / "project.db"
    if not project_db.exists():
        raise FileNotFoundError(f"Project database missing at {project_db}")

    con = sqlite3.connect(str(project_db))
    try:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT storage_path FROM phenotypes WHERE pheno_id=?",
            (pheno_id,),
        ).fetchone()
    finally:
        con.close()

    if not row:
        raise ValueError(f"Phenotype {pheno_id} not found in project database")

    storage_path = row["storage_path"]
    if not storage_path:
        raise ValueError(f"Phenotype {pheno_id} is missing a storage_path")

    storage = Path(storage_path)
    if not storage.is_absolute():
        storage = (Path(project_root) / storage).resolve()
    return storage


def _candidate_corpus_paths(
    project_root: Path,
    phenotype_dir: Path,
    pheno_id: str,
    hints: Optional[List[str]] = None,
) -> List[Path]:
    candidates: List[Path] = [
        phenotype_dir / "corpus" / "corpus.db",
        phenotype_dir / "corpus.db",
        phenotype_dir / "corpus.sqlite",
        phenotype_dir / f"{pheno_id}.db",
    ]

    for corpora_folder in (project_root / "corpora", project_root / "Corpora"):
        candidates.extend(
            [
                corpora_folder / pheno_id / "corpus.db",
                corpora_folder / pheno_id / "corpus.sqlite",
                corpora_folder / pheno_id / f"{pheno_id}.db",
                corpora_folder / f"{pheno_id}.db",
                corpora_folder / f"{pheno_id}.sqlite",
            ]
        )

    for hint in hints or []:
        path = Path(hint)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        candidates.append(path)

    seen: set[Path] = set()
    unique: List[Path] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (project_root / candidate).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _find_corpus_db(project_root: Path, pheno_id: str, prior_rounds: List[int]) -> Path:
    phenotype_dir = _resolve_phenotype_dir(project_root, pheno_id)
    project_db = Path(project_root) / "project.db"
    hints: List[str] = []

    if project_db.exists():
        con = sqlite3.connect(str(project_db))
        try:
            con.row_factory = sqlite3.Row
            round_ids: List[str] = []
            if prior_rounds:
                round_ids.extend(f"{pheno_id}_r{number}" for number in sorted(prior_rounds, reverse=True))
            rows = con.execute(
                "SELECT round_id FROM rounds WHERE pheno_id=? ORDER BY round_number DESC",
                (pheno_id,),
            ).fetchall()
            round_ids.extend(row["round_id"] for row in rows)

            seen_rounds: set[str] = set()
            for round_id in round_ids:
                if round_id in seen_rounds:
                    continue
                seen_rounds.add(round_id)
                cfg_row = con.execute(
                    "SELECT config_json FROM round_configs WHERE round_id=?",
                    (round_id,),
                ).fetchone()
                if not cfg_row:
                    continue
                try:
                    config = json.loads(cfg_row["config_json"])
                except Exception:  # noqa: BLE001
                    continue
                corpus_id = config.get("corpus_id")
                if corpus_id:
                    corpus_row = con.execute(
                        "SELECT relative_path FROM project_corpora WHERE corpus_id=?",
                        (corpus_id,),
                    ).fetchone()
                    if corpus_row and corpus_row["relative_path"]:
                        hints.append(corpus_row["relative_path"])
                corpus_path = config.get("corpus_path")
                if corpus_path:
                    hints.append(corpus_path)
        finally:
            con.close()

    for candidate in _candidate_corpus_paths(project_root, phenotype_dir, pheno_id, hints=hints):
        if candidate.exists():
            return candidate

    legacy = Path(project_root) / "phenotypes" / pheno_id / "corpus" / "corpus.db"
    if legacy.exists():
        return legacy

    checked = [str(p) for p in _candidate_corpus_paths(project_root, phenotype_dir, pheno_id, hints=hints)]
    checked.append(str(legacy))
    raise FileNotFoundError(
        f"Could not locate corpus.db for phenotype {pheno_id}; checked {checked}"
    )


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
    phenotype_dir = _resolve_phenotype_dir(root, pheno_id)
    corpus_db = _find_corpus_db(root, pheno_id, prior_rounds)
    notes_df = _read_corpus_db(corpus_db)

    # Aggregate from selected rounds: <storage_path>/rounds/round_<n>/round_aggregate.db
    ann_frames = []
    for r in prior_rounds:
        round_db = phenotype_dir / "rounds" / f"round_{r}" / "round_aggregate.db"
        if round_db.exists():
            ann_frames.append(_read_round_aggregate(round_db))
    ann_df = pd.concat(ann_frames, ignore_index=True) if ann_frames else pd.DataFrame(columns=[
        "round_id","unit_id","doc_id","label_id","reviewer_id","label_value",
        "reviewer_notes","rationales_json","document_text","document_metadata_json","label_rules"
    ])
    return notes_df, ann_df

def run_ai_backend_and_collect(
    project_root: Path,
    pheno_id: str,
    prior_rounds: List[int],
    round_dir: Path,
    level: str,
    user: str,
    timestamp: Optional[str] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> BackendResult:
    log = log_callback or (lambda message: None)
    log("Preparing AI backend inputs…")
    notes_df, ann_df = export_inputs_from_repo(project_root, pheno_id, prior_rounds)
    ai_dir = Path(round_dir) / "imports" / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    log(f"Exported {len(notes_df)} corpus rows and {len(ann_df)} prior annotations")

    label_config_path = Path(project_root) / "phenotypes" / pheno_id / "ai" / "label_config.json"
    label_config = None
    if label_config_path.exists():
        try:
            label_config = json.loads(label_config_path.read_text(encoding="utf-8"))
            log("Loaded label_config.json overrides")
        except Exception as exc:  # noqa: BLE001
            log(f"Warning: failed to parse label_config.json ({exc})")

    overrides: Dict[str, Any] = dict(cfg_overrides or {})
    overrides.setdefault("phenotype_level", level)

    final_df, artifacts = build_next_batch(
        notes_df,
        ann_df,
        outdir=ai_dir,
        label_config=label_config,
        cfg_overrides=overrides,
    )
    csv_path = Path(artifacts["ai_next_batch_csv"])
    log(f"AI backend produced {len(final_df)} candidate units")

    params = {
        "phenotype_id": pheno_id,
        "prior_rounds": list(sorted(prior_rounds)),
        "phenotype_level": level,
        "invoked_by": user,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "backend_version": __version__,
    }
    params_path = ai_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    log(f"Wrote parameters to {params_path}")

    return BackendResult(csv_path=csv_path, dataframe=final_df, artifacts=artifacts, params_path=params_path)
