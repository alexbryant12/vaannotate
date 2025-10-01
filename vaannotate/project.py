"""Project level helpers."""
from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .schema import initialize_project_db, initialize_corpus_db
from .utils import ensure_dir, canonical_json


@dataclass
class ProjectPaths:
    root: Path
    project_db: Path
    corpus_dir: Path
    corpus_db: Path
    admin_dir: Path


def build_project_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        root=root,
        project_db=root / "project.db",
        corpus_dir=root / "corpus",
        corpus_db=root / "corpus" / "corpus.db",
        admin_dir=root / "admin_tools",
    )


def init_project(root: Path, project_id: str, name: str, created_by: str) -> ProjectPaths:
    paths = build_project_paths(root)
    ensure_dir(paths.root)
    ensure_dir(paths.corpus_dir)
    ensure_dir(paths.admin_dir)
    with initialize_project_db(paths.project_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO projects(project_id, name, created_at, created_by) VALUES (?,?,?,?)",
            (project_id, name, datetime.utcnow().isoformat(), created_by),
        )
    with initialize_corpus_db(paths.corpus_db):
        pass
    metadata = {
        "project_id": project_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "created_by": created_by,
    }
    (paths.root / "project_metadata.json").write_text(canonical_json(metadata), encoding="utf-8")
    return paths


def get_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def register_reviewer(conn: sqlite3.Connection, reviewer_id: str, name: str, email: str | None = None, windows_account: str | None = None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO reviewers(reviewer_id, name, email, windows_account) VALUES (?,?,?,?)",
        (reviewer_id, name, email, windows_account),
    )


def add_phenotype(conn: sqlite3.Connection, pheno_id: str, project_id: str, name: str, level: str, description: str | None = None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO phenotypes(pheno_id, project_id, name, level, description) VALUES (?,?,?,?,?)",
        (pheno_id, project_id, name, level, description),
    )


def add_labelset(
    conn: sqlite3.Connection,
    *,
    labelset_id: str,
    pheno_id: str,
    version: int,
    created_by: str,
    notes: str | None,
    labels: Iterable[dict],
) -> None:
    created_at = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO label_sets(labelset_id, pheno_id, version, created_at, created_by, notes) VALUES (?,?,?,?,?,?)",
        (labelset_id, pheno_id, version, created_at, created_by, notes),
    )
    for idx, label in enumerate(labels):
        label_id = label["label_id"]
        conn.execute(
            """
            INSERT OR REPLACE INTO labels(
                label_id,labelset_id,name,type,required,order_index,rules,gating_expr,na_allowed,unit,min,max
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                label_id,
                labelset_id,
                label["name"],
                label["type"],
                1 if label.get("required", False) else 0,
                label.get("order_index", idx),
                label.get("rules"),
                label.get("gating_expr"),
                1 if label.get("na_allowed") else 0,
                label.get("unit"),
                label.get("min"),
                label.get("max"),
            ),
        )
        for o_idx, option in enumerate(label.get("options", [])):
            option_id = option.get("option_id") or f"{label_id}_opt{o_idx}"
            conn.execute(
                """
                INSERT OR REPLACE INTO label_options(option_id,label_id,value,display,order_index,weight)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    option_id,
                    label_id,
                    option["value"],
                    option.get("display", option["value"]),
                    option.get("order_index", o_idx),
                    option.get("weight"),
                ),
            )


def fetch_labelset(conn: sqlite3.Connection, labelset_id: str) -> dict:
    labelset_row = conn.execute(
        "SELECT * FROM label_sets WHERE labelset_id=?",
        (labelset_id,),
    ).fetchone()
    if not labelset_row:
        raise ValueError(f"Label set {labelset_id} not found")
    labels = conn.execute(
        "SELECT * FROM labels WHERE labelset_id=? ORDER BY order_index",
        (labelset_id,),
    ).fetchall()
    options_rows = conn.execute(
        "SELECT * FROM label_options WHERE label_id IN (SELECT label_id FROM labels WHERE labelset_id=?) ORDER BY order_index",
        (labelset_id,),
    ).fetchall()
    options_map: dict[str, list[sqlite3.Row]] = {}
    for row in options_rows:
        options_map.setdefault(row["label_id"], []).append(row)
    label_dicts = []
    for label in labels:
        label_dicts.append(
            {
                "label_id": label["label_id"],
                "name": label["name"],
                "type": label["type"],
                "required": bool(label["required"]),
                "order_index": label["order_index"],
                "rules": label["rules"],
                "gating_expr": label["gating_expr"],
                "na_allowed": bool(label["na_allowed"]),
                "unit": label["unit"],
                "min": label["min"],
                "max": label["max"],
                "options": [dict(opt) for opt in options_map.get(label["label_id"], [])],
            }
        )
    labelset = dict(labelset_row)
    labelset["labels"] = label_dicts
    return labelset
