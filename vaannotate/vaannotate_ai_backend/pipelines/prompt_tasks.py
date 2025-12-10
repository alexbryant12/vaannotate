from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import pandas as pd

from ..utils.jsonish import _jsonify_cols, _maybe_parse_jsonish


@dataclass
class SinglePromptTask:
    job_id: str
    prompt_id: str  # e.g. f"{job_id}:unit:{unit_id}"
    unit_id: str
    label_ids: list[str]
    ctx_snippets: list[dict]  # RAG snippets for all labels in the task
    rules_map: dict[str, str]  # label_id -> rule text
    label_types: dict[str, str]  # label_id -> type ("binary", "categorical", "date", etc.)
    rag_fingerprint: str
    meta: dict  # pheno_id, labelset_id, phenotype_level, etc.


@dataclass
class FamilyPromptTask:
    job_id: str
    prompt_id: str  # e.g. f"{job_id}:unit:{unit_id}:label:{label_id}"
    unit_id: str
    label_id: str
    label_type: str
    label_rules: str
    ctx_snippets: list[dict]  # RAG snippets for this (unit,label) only
    rag_fingerprint: str
    meta: dict


def single_prompt_tasks_to_df(tasks: list[SinglePromptTask]) -> pd.DataFrame:
    """Convert a list of ``SinglePromptTask`` objects to a DataFrame."""

    df = pd.DataFrame([asdict(t) for t in tasks])
    if df.empty:
        return df

    return _jsonify_cols(df, ["ctx_snippets", "rules_map", "label_types", "meta"])


def family_prompt_tasks_to_df(tasks: list[FamilyPromptTask]) -> pd.DataFrame:
    """Convert a list of ``FamilyPromptTask`` objects to a DataFrame."""

    df = pd.DataFrame([asdict(t) for t in tasks])
    if df.empty:
        return df

    return _jsonify_cols(df, ["ctx_snippets", "meta"])


def _parse_jsonish(value, default):
    parsed = _maybe_parse_jsonish(value)
    if parsed is not None:
        return parsed
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return value


def df_to_single_prompt_tasks(df: pd.DataFrame) -> List[SinglePromptTask]:
    """Hydrate ``SinglePromptTask`` objects from a DataFrame."""

    if df is None or df.empty:
        return []

    tasks: list[SinglePromptTask] = []
    for _, row in df.iterrows():
        tasks.append(
            SinglePromptTask(
                job_id=str(row.get("job_id")),
                prompt_id=str(row.get("prompt_id")),
                unit_id=str(row.get("unit_id")),
                label_ids=_parse_jsonish(row.get("label_ids"), []),
                ctx_snippets=_parse_jsonish(row.get("ctx_snippets"), []),
                rules_map=_parse_jsonish(row.get("rules_map"), {}),
                label_types=_parse_jsonish(row.get("label_types"), {}),
                rag_fingerprint=row.get("rag_fingerprint"),
                meta=_parse_jsonish(row.get("meta"), {}),
            )
        )

    return tasks


def df_to_family_prompt_tasks(df: pd.DataFrame) -> List[FamilyPromptTask]:
    """Hydrate ``FamilyPromptTask`` objects from a DataFrame."""

    if df is None or df.empty:
        return []

    tasks: list[FamilyPromptTask] = []
    for _, row in df.iterrows():
        tasks.append(
            FamilyPromptTask(
                job_id=str(row.get("job_id")),
                prompt_id=str(row.get("prompt_id")),
                unit_id=str(row.get("unit_id")),
                label_id=str(row.get("label_id")),
                label_type=str(row.get("label_type")),
                label_rules=str(row.get("label_rules")),
                ctx_snippets=_parse_jsonish(row.get("ctx_snippets"), []),
                rag_fingerprint=row.get("rag_fingerprint"),
                meta=_parse_jsonish(row.get("meta"), {}),
            )
        )

    return tasks
