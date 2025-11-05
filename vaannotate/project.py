"""Project level helpers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import initialize_project_db
from .utils import ensure_dir, canonical_json


@dataclass
class ProjectPaths:
    root: Path
    project_db: Path
    admin_dir: Path
    phenotypes_dir: Path
    corpora_dir: Path


def build_project_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        root=root,
        project_db=root / "project.db",
        admin_dir=root / "admin_tools",
        phenotypes_dir=root / "phenotypes",
        corpora_dir=root / "corpora",
    )


def init_project(root: Path, project_id: str, name: str, created_by: str) -> ProjectPaths:
    paths = build_project_paths(root)
    ensure_dir(paths.root)
    ensure_dir(paths.admin_dir)
    ensure_dir(paths.phenotypes_dir)
    ensure_dir(paths.corpora_dir)
    with initialize_project_db(paths.project_db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO projects(project_id, name, created_at, created_by) VALUES (?,?,?,?)",
            (project_id, name, datetime.utcnow().isoformat(), created_by),
        )
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


def add_project_corpus(
    conn: sqlite3.Connection,
    *,
    corpus_id: str,
    project_id: str,
    name: str,
    relative_path: str,
    created_at: str | None = None,
) -> None:
    timestamp = created_at or datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO project_corpora(corpus_id, project_id, name, relative_path, created_at)
        VALUES (?,?,?,?,?)
        """,
        (corpus_id, project_id, name, relative_path, timestamp),
    )


def add_phenotype(
    conn: sqlite3.Connection,
    pheno_id: str,
    project_id: str,
    name: str,
    level: str,
    description: str | None = None,
    *,
    storage_path: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO phenotypes(pheno_id, project_id, name, level, description, storage_path) VALUES (?,?,?,?,?,?)",
        (pheno_id, project_id, name, level, description, storage_path),
    )


def add_labelset(
    conn: sqlite3.Connection,
    *,
    labelset_id: str,
    project_id: str,
    pheno_id: str | None,
    version: int,
    created_by: str,
    notes: str | None,
    labels: Iterable[dict],
) -> None:
    created_at = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO label_sets(
            labelset_id, project_id, pheno_id, version, created_at, created_by, notes
        ) VALUES (?,?,?,?,?,?,?)
        """,
        (labelset_id, project_id, pheno_id, version, created_at, created_by, notes),
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


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _extract_condition_value(raw: str) -> Optional[str]:
    text = raw.strip()
    if not text:
        return None
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        return text[1:-1]
    return text


def build_label_config(labelset: dict) -> Dict[str, object]:
    labels: Sequence[Dict[str, object]] = labelset.get("labels", [])  # type: ignore[assignment]
    label_lookup: Dict[str, Dict[str, object]] = {}
    name_lookup: Dict[str, str] = {}
    order: List[str] = []
    for entry in labels:
        label_id = str(entry.get("label_id", "")).strip()
        if not label_id:
            continue
        label_lookup[label_id] = dict(entry)
        order.append(label_id)
        name = str(entry.get("name", "")).strip()
        if name:
            name_lookup[_normalize_name(name)] = label_id

    parent_map: Dict[str, List[Dict[str, object]]] = {}
    children_map: Dict[str, List[Dict[str, object]]] = {}
    for entry in labels:
        label_id = str(entry.get("label_id", "")).strip()
        if not label_id:
            continue
        gating_expr = str(entry.get("gating_expr") or "").strip()
        if "==" not in gating_expr:
            continue
        lhs, _, rhs = gating_expr.partition("==")
        field_key = lhs.strip()
        if not field_key:
            continue
        parent_id: Optional[str] = None
        if field_key in label_lookup:
            parent_id = field_key
        else:
            normalized = _normalize_name(field_key)
            parent_id = name_lookup.get(normalized)
        if not parent_id or parent_id == label_id:
            continue
        condition_value = _extract_condition_value(rhs)
        relationship = {"label_id": parent_id, "expression": gating_expr}
        if condition_value is not None:
            relationship["value"] = condition_value
        parent_map.setdefault(label_id, []).append(relationship)
        child_rel = {"label_id": label_id, "expression": gating_expr}
        if condition_value is not None:
            child_rel["value"] = condition_value
        children_map.setdefault(parent_id, []).append(child_rel)

    def _gating_type_for(label_id: str) -> str:
        raw_type = str(label_lookup.get(label_id, {}).get("type") or "").casefold()
        if raw_type in {"integer", "int", "float", "double", "number", "numeric"}:
            return "numeric"
        if raw_type in {"date", "datetime"}:
            return "date"
        return "categorical"

    def build_branch(label_id: str, visited: Tuple[str, ...]) -> Dict[str, object]:
        node: Dict[str, object] = {
            "label_id": label_id,
            "name": label_lookup.get(label_id, {}).get("name"),
            "type": label_lookup.get(label_id, {}).get("type"),
            "children": [],
        }
        for rel in children_map.get(label_id, []):
            child_id = str(rel.get("label_id"))
            if not child_id or child_id in visited:
                continue
            branch = build_branch(child_id, visited + (child_id,))
            branch["condition"] = rel.get("expression")
            if rel.get("value") is not None:
                branch["condition_value"] = rel.get("value")
            node.setdefault("children", []).append(branch)
        return node

    root_candidates = [label_id for label_id in order if label_id not in parent_map]
    if not root_candidates:
        root_candidates = list(order)
    dependency_tree = [build_branch(label_id, (label_id,)) for label_id in root_candidates]

    config: Dict[str, object] = {
        "_meta": {
            "labelset_id": labelset.get("labelset_id"),
            "labelset_name": labelset.get("labelset_name") or labelset.get("labelset_id"),
            "notes": labelset.get("notes"),
            "created_by": labelset.get("created_by"),
            "created_at": labelset.get("created_at"),
            "generated_at": datetime.utcnow().isoformat(),
            "dependency_tree": dependency_tree,
        }
    }

    for label_id in order:
        label = label_lookup[label_id]
        option_details = [
            {
                "value": opt.get("value"),
                "display": opt.get("display"),
                "weight": opt.get("weight"),
                "order_index": opt.get("order_index"),
            }
            for opt in label.get("options", [])  # type: ignore[assignment]
            if isinstance(opt, Mapping)
        ]
        options = [str(opt.get("value")) for opt in option_details if opt.get("value") is not None]
        parents_payload = [
            {k: rel[k] for k in ("label_id", "expression", "value") if k in rel}
            for rel in parent_map.get(label_id, [])
        ]
        children_payload = [
            {k: rel[k] for k in ("label_id", "expression", "value") if k in rel}
            for rel in children_map.get(label_id, [])
        ]
        entry_payload: Dict[str, object] = {
            "name": label.get("name"),
            "type": label.get("type"),
            "required": bool(label.get("required")),
            "na_allowed": bool(label.get("na_allowed")),
            "gating_expr": label.get("gating_expr"),
            "rules": label.get("rules"),
            "unit": label.get("unit"),
            "range": {"min": label.get("min"), "max": label.get("max")},
            "options": options,
            "option_details": option_details,
        }
        gating_parents = [
            str(rel.get("label_id"))
            for rel in parent_map.get(label_id, [])
            if rel.get("label_id")
        ]
        if gating_parents:
            unique_parents: List[str] = []
            for parent_id in gating_parents:
                if parent_id not in unique_parents:
                    unique_parents.append(parent_id)
            if len(unique_parents) == 1:
                entry_payload["gated_by"] = unique_parents[0]
            else:
                entry_payload["gated_by"] = unique_parents
            gating_rules: List[Dict[str, object]] = []
            for rel in parent_map.get(label_id, []):
                parent_id = str(rel.get("label_id") or "").strip()
                if not parent_id:
                    continue
                value = rel.get("value")
                if value is None:
                    continue
                parent_type = _gating_type_for(parent_id)
                rule: Dict[str, object] = {
                    "parent": parent_id,
                    "type": parent_type,
                    "op": "in",
                    "values": [value],
                }
                gating_rules.append(rule)
            if gating_rules:
                entry_payload["gating_rules"] = gating_rules
        if parents_payload:
            entry_payload["parents"] = parents_payload
        if children_payload:
            entry_payload["children"] = children_payload
        config[label_id] = entry_payload

    return config
