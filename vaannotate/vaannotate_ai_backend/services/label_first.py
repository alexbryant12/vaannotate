from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import pandas as pd

from .label_dependencies import build_label_dependencies, evaluate_gating


@dataclass(frozen=True)
class LabelFirstTarget:
    label_id: str
    values: tuple[str, ...]
    quota: int
    operator: str = "in"


def parse_label_first_targets(payload: object) -> list[LabelFirstTarget]:
    if not isinstance(payload, Sequence):
        return []
    targets: list[LabelFirstTarget] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        label_id = str(item.get("label_id") or "").strip()
        if not label_id:
            continue
        values_raw = item.get("values")
        if isinstance(values_raw, Sequence) and not isinstance(values_raw, (str, bytes)):
            values = tuple(str(v).strip().lower() for v in values_raw if str(v).strip())
        else:
            single = str(item.get("value") or "").strip().lower()
            values = (single,) if single else tuple()
        try:
            quota = int(item.get("quota") or 0)
        except (TypeError, ValueError):
            quota = 0
        if quota <= 0:
            continue
        operator = str(item.get("operator") or "in").strip().lower()
        targets.append(
            LabelFirstTarget(
                label_id=label_id,
                values=values,
                quota=quota,
                operator=operator if operator in {"in", "equals"} else "in",
            )
        )
    return targets


def collect_required_labels(
    *,
    label_config: Mapping[str, object],
    label_types: Mapping[str, str],
    target_label_ids: Iterable[str],
) -> set[str]:
    parent_to_children, child_to_parents, _roots = build_label_dependencies(dict(label_config))
    _ = parent_to_children
    required: set[str] = set()
    queue: list[str] = [str(lid) for lid in target_label_ids if str(lid)]
    seen: set[str] = set()
    while queue:
        lid = queue.pop(0)
        if lid in seen:
            continue
        seen.add(lid)
        required.add(lid)
        for parent in child_to_parents.get(lid, []):
            parent_id = str(parent)
            if parent_id and parent_id not in seen:
                queue.append(parent_id)
    # Keep only labels that are known to the current label config maps.
    return {lid for lid in required if lid in set(label_types.keys())}


def build_target_counts(targets: Sequence[LabelFirstTarget]) -> dict[str, int]:
    return {f"{t.label_id}::{','.join(t.values)}::{t.operator}": 0 for t in targets}


def quota_key(target: LabelFirstTarget) -> str:
    return f"{target.label_id}::{','.join(target.values)}::{target.operator}"


def match_target(prediction: object, target: LabelFirstTarget) -> bool:
    pred = str(prediction or "").strip().lower()
    if not pred:
        return False
    if target.operator == "equals":
        return bool(target.values) and pred == target.values[0]
    if not target.values:
        return bool(pred)
    return pred in set(target.values)


def evaluate_target_hits(
    *,
    unit_id: str,
    predictions: Mapping[str, object],
    targets: Sequence[LabelFirstTarget],
    counts: dict[str, int],
) -> list[LabelFirstTarget]:
    hits: list[LabelFirstTarget] = []
    for target in targets:
        key = quota_key(target)
        if counts.get(key, 0) >= target.quota:
            continue
        value = predictions.get(target.label_id)
        if match_target(value, target):
            hits.append(target)
    return hits


def unresolved_targets(
    targets: Sequence[LabelFirstTarget],
    counts: Mapping[str, int],
) -> list[LabelFirstTarget]:
    return [target for target in targets if counts.get(quota_key(target), 0) < target.quota]


def random_pool(
    *,
    candidates: Sequence[str],
    size: int,
    rng: random.Random,
) -> list[str]:
    if size <= 0:
        return []
    if len(candidates) <= size:
        return [str(uid) for uid in candidates]
    sampled = list(candidates)
    rng.shuffle(sampled)
    return sampled[:size]


def enriched_pool(
    *,
    candidates: Sequence[str],
    size: int,
    unresolved: Sequence[LabelFirstTarget],
    context_builder: object,
    rules_map: Mapping[str, str],
    rng: random.Random,
    max_scored_units: int = 500,
) -> list[str]:
    if size <= 0:
        return []
    if not unresolved:
        return random_pool(candidates=candidates, size=size, rng=rng)
    score_subset = list(candidates)
    rng.shuffle(score_subset)
    score_subset = score_subset[: min(len(score_subset), max_scored_units)]
    rows: list[tuple[str, float]] = []
    for uid in score_subset:
        best_score = 0.0
        for target in unresolved:
            rules = str(rules_map.get(target.label_id) or "")
            try:
                ctx = context_builder.build_context_for_label(uid, target.label_id, rules, topk_override=6)
            except Exception:
                ctx = None
            if not ctx:
                continue
            try:
                top = float(ctx[0].get("score", 0.0))
            except Exception:
                top = 0.0
            if top > best_score:
                best_score = top
        rows.append((str(uid), float(best_score)))
    ranked = [uid for uid, _ in sorted(rows, key=lambda item: item[1], reverse=True)]
    if len(ranked) >= size:
        return ranked[:size]
    leftovers = [str(uid) for uid in candidates if str(uid) not in set(ranked)]
    return ranked + random_pool(candidates=leftovers, size=max(0, size - len(ranked)), rng=rng)


def label_unit_single_prompt(
    *,
    unit_id: str,
    label_ids: Sequence[str],
    label_types: Mapping[str, str],
    rules_map: Mapping[str, str],
    llm_labeler: object,
    context_builder: object,
    label_config: Mapping[str, object],
    max_chars: int = 16000,
) -> dict[str, object]:
    ctx = context_builder.build_context_for_family(
        unit_id,
        label_ids=label_ids,
        rules_map=rules_map,
        topk_per_label=6,
        max_snippets=None,
        max_chars=max_chars,
    )
    res = llm_labeler.annotate_multi(
        unit_id=unit_id,
        label_ids=list(label_ids),
        label_types=label_types,
        rules_map=rules_map,
        ctx_snippets=ctx,
    )
    preds = res.get("predictions") if isinstance(res, Mapping) else {}
    parent_preds: dict[tuple[str, str], object] = {}
    if isinstance(preds, Mapping):
        for lid, info in preds.items():
            prediction = info.get("prediction") if isinstance(info, Mapping) else None
            parent_preds[(str(unit_id), str(lid))] = prediction

    output: dict[str, object] = {}
    for lid in label_ids:
        pred_obj = preds.get(lid) if isinstance(preds, Mapping) else {}
        prediction = pred_obj.get("prediction") if isinstance(pred_obj, Mapping) else None
        gated = evaluate_gating(
            child_id=str(lid),
            unit_id=str(unit_id),
            parent_preds=parent_preds,
            label_types=dict(label_types),
            label_config=dict(label_config),
        )
        output[str(lid)] = prediction if gated else None
    return output


def finalize_rows(
    *,
    selected_units: Sequence[str],
    matched_labels: Mapping[str, str],
    label_types: Mapping[str, str],
    selection_reason: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for uid in selected_units:
        lid = str(matched_labels.get(uid) or "")
        rows.append(
            {
                "unit_id": str(uid),
                "label_id": lid,
                "label_type": str(label_types.get(lid, "categorical")),
                "selection_reason": selection_reason,
            }
        )
    return pd.DataFrame(rows)
