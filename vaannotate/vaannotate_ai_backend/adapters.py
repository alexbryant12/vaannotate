
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
        con.row_factory = sqlite3.Row
        doc_columns = {row["name"] for row in con.execute("PRAGMA table_info(documents)")}

        def select_expr(target: str, candidates: List[str], *, default: str) -> str:
            for candidate in candidates:
                if candidate in doc_columns:
                    return f"{candidate} AS {target}"
            return f"{default} AS {target}"

        select_clauses = [
            select_expr("patient_icn", ["patient_icn", "patienticn"], default="''"),
            select_expr("doc_id", ["doc_id"], default="''"),
            select_expr("text", ["text", "document_text"], default="''"),
            select_expr("date_note", ["date_note"], default="''"),
            select_expr("notetype", ["notetype"], default="''"),
            select_expr(
                "document_metadata_json",
                ["document_metadata_json", "metadata_json"],
                default="'{}'",
            ),
        ]

        query = f"SELECT {', '.join(select_clauses)} FROM documents"
        df = pd.read_sql_query(query, con)
        return df
    finally:
        con.close()


def _iter_assignment_dbs(round_dir: Path) -> List[Path]:
    imports_dir = round_dir / "imports"
    if not imports_dir.exists():
        return []
    return sorted(p for p in imports_dir.glob("*_assignment.db") if p.is_file())


def _normalize_reviewer_id(assignment_path: Path) -> str:
    name = assignment_path.stem
    if name.endswith("_assignment"):
        return name[: -len("_assignment")]
    return name


def _load_label_rules(project_root: Path, pheno_id: str, round_number: int) -> Dict[str, str]:
    project_db = Path(project_root) / "project.db"
    if not project_db.exists():
        return {}

    con = sqlite3.connect(str(project_db))
    try:
        con.row_factory = sqlite3.Row
        labelset_row = con.execute(
            "SELECT labelset_id FROM rounds WHERE pheno_id=? AND round_number=?",
            (pheno_id, round_number),
        ).fetchone()
        if not labelset_row:
            return {}

        labelset_id = labelset_row["labelset_id"]
        if not labelset_id:
            return {}

        rows = con.execute(
            "SELECT label_id, rules FROM labels WHERE labelset_id=?",
            (labelset_id,),
        ).fetchall()
    finally:
        con.close()

    return {
        str(row["label_id"] or ""): str(row["rules"] or "")
        for row in rows
        if row["label_id"]
    }


def _load_change_history(round_dir: Path, round_id: str) -> Dict[Tuple[str, str, str], str]:
    agg_path = round_dir / "round_aggregate.db"
    if not agg_path.exists():
        return {}

    con = sqlite3.connect(str(agg_path))
    try:
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT unit_id, reviewer_id, label_id, history FROM annotation_change_history WHERE round_id=?",
                (round_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
    finally:
        con.close()

    history: Dict[Tuple[str, str, str], str] = {}
    for row in rows:
        unit_id = str(row["unit_id"] or "")
        reviewer_id = str(row["reviewer_id"] or "")
        label_id = str(row["label_id"] or "")
        if not unit_id or not reviewer_id or not label_id:
            continue
        history[(unit_id, reviewer_id, label_id)] = str(row["history"] or "")
    return history


def _read_round_annotations(
    round_dir: Path, pheno_id: str, round_number: int, project_root: Path
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    round_id = f"{pheno_id}_r{round_number}"
    label_rules = _load_label_rules(project_root, pheno_id, round_number)
    change_history = _load_change_history(round_dir, round_id)

    for assignment_path in _iter_assignment_dbs(round_dir):
        reviewer_id = _normalize_reviewer_id(assignment_path)
        con = sqlite3.connect(str(assignment_path))
        try:
            con.row_factory = sqlite3.Row
            units = {
                str(row["unit_id"]): {
                    "patient_icn": str(row["patient_icn"] or ""),
                    "doc_id": str(row["doc_id"] or ""),
                }
                for row in con.execute("SELECT unit_id, patient_icn, doc_id FROM units")
            }

            rationale_rows = con.execute(
                "SELECT unit_id, label_id, doc_id, start_offset, end_offset, snippet FROM rationales"
            ).fetchall()
            rationales: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for row in rationale_rows:
                unit_id = str(row["unit_id"] or "")
                label_id = str(row["label_id"] or "")
                if not unit_id or not label_id:
                    continue
                entry = {
                    "doc_id": str(row["doc_id"] or ""),
                    "start_offset": row["start_offset"],
                    "end_offset": row["end_offset"],
                    "snippet": row["snippet"],
                }
                rationales.setdefault((unit_id, label_id), []).append(entry)

            for ann in con.execute(
                "SELECT unit_id, label_id, value, value_num, value_date, na, notes FROM annotations"
            ):
                unit_id = str(ann["unit_id"] or "")
                label_id = str(ann["label_id"] or "")
                if not unit_id or not label_id:
                    continue
                value = ann["value"]
                value_str = "" if value is None else str(value)
                value_num = ann["value_num"]
                if value_num is None:
                    value_num_str = ""
                else:
                    try:
                        value_num_str = format(value_num, "g")
                    except Exception:  # noqa: BLE001
                        value_num_str = str(value_num)
                value_date = "" if ann["value_date"] is None else str(ann["value_date"])
                notes = "" if ann["notes"] is None else str(ann["notes"])
                na_flag = 1 if ann["na"] else 0
                unit_meta = units.get(unit_id, {})
                patient_icn = unit_meta.get("patient_icn", "")
                doc_id = unit_meta.get("doc_id", "")
                normalized_rationales = rationales.get((unit_id, label_id), [])
                rationale_json = (
                    json.dumps(normalized_rationales, ensure_ascii=False)
                    if normalized_rationales
                    else ""
                )

                rows.append(
                    {
                        "round_id": round_id,
                        "phenotype_id": pheno_id,
                        "unit_id": unit_id,
                        "doc_id": doc_id,
                        "patient_icn": patient_icn,
                        "reviewer_id": reviewer_id,
                        "reviewer_name": "",
                        "label_id": label_id,
                        "label_name": "",
                        "label_value": value_str,
                        "label_value_num": value_num_str,
                        "label_value_date": value_date,
                        "label_na": na_flag,
                        "reviewer_notes": notes,
                        "rationales_json": rationale_json,
                        "document_text": "",
                        "document_metadata_json": "",
                        "label_rules": label_rules.get(label_id, ""),
                        "label_change_history": change_history.get((unit_id, reviewer_id, label_id), ""),
                    }
                )
        finally:
            con.close()

    columns = [
        "round_id",
        "phenotype_id",
        "unit_id",
        "doc_id",
        "patient_icn",
        "reviewer_id",
        "reviewer_name",
        "label_id",
        "label_name",
        "label_value",
        "label_value_num",
        "label_value_date",
        "label_na",
        "reviewer_notes",
        "rationales_json",
        "document_text",
        "document_metadata_json",
        "label_rules",
        "label_change_history",
    ]

    return pd.DataFrame(rows, columns=columns)

def export_inputs_from_repo(project_root: Path, pheno_id: str, prior_rounds: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root)
    phenotype_dir = _resolve_phenotype_dir(root, pheno_id)
    corpus_db = _find_corpus_db(root, pheno_id, prior_rounds)
    notes_df = _read_corpus_db(corpus_db)

    ann_frames = []
    for r in prior_rounds:
        round_dir = phenotype_dir / "rounds" / f"round_{r}"
        if not round_dir.exists():
            continue
        frame = _read_round_annotations(round_dir, pheno_id, r, root)
        if not frame.empty:
            ann_frames.append(frame)

    ann_df = (
        pd.concat(ann_frames, ignore_index=True)
        if ann_frames
        else pd.DataFrame(
            columns=[
                "round_id",
                "phenotype_id",
                "unit_id",
                "doc_id",
                "patient_icn",
                "reviewer_id",
                "reviewer_name",
                "label_id",
                "label_name",
                "label_value",
                "label_value_num",
                "label_value_date",
                "label_na",
                "reviewer_notes",
                "rationales_json",
                "document_text",
                "document_metadata_json",
                "label_rules",
                "label_change_history",
            ]
        )
    )
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
    label_config: Optional[Dict[str, Any]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> BackendResult:
    log = log_callback or (lambda message: None)
    log("Preparing AI backend inputs…")
    notes_df, ann_df = export_inputs_from_repo(project_root, pheno_id, prior_rounds)
    ai_dir = Path(round_dir) / "imports" / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    log(f"Exported {len(notes_df)} corpus rows and {len(ann_df)} prior annotations")

    label_config_payload: Optional[Dict[str, Any]] = label_config
    if label_config_payload is None:
        label_config_path = Path(project_root) / "phenotypes" / pheno_id / "ai" / "label_config.json"
        if label_config_path.exists():
            try:
                label_config_payload = json.loads(label_config_path.read_text(encoding="utf-8"))
                log("Loaded label_config.json overrides")
            except Exception as exc:  # noqa: BLE001
                log(f"Warning: failed to parse label_config.json ({exc})")

    overrides: Dict[str, Any] = dict(cfg_overrides or {})
    select_overrides: Dict[str, Any] = dict(overrides.get("select", {}))
    if not prior_rounds:
        # Cold start: no reviewer disagreements yet, skip the bucket entirely.
        if select_overrides.get("pct_disagreement") is None:
            select_overrides["pct_disagreement"] = 0.0
        log("No prior rounds detected; skipping disagreement bucket for round zero.")
    overrides["select"] = select_overrides
    overrides.setdefault("phenotype_level", level)

    final_df, artifacts = build_next_batch(
        notes_df,
        ann_df,
        outdir=ai_dir,
        label_config=label_config_payload,
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
