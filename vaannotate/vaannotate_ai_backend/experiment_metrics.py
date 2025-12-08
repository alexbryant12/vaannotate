"""Utilities for computing experiment-level metrics for AI labeling pipelines."""

import dataclasses
import math
import numpy as np
import re
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd

from vaannotate.vaannotate_ai_backend.label_configs import LabelConfigBundle
from vaannotate.vaannotate_ai_backend.services.label_dependencies import build_label_dependencies


def _parse_multi_select(value: object) -> set[str]:
    """Normalize multi-select predictions to a canonical set of strings.

    Accepts strings, iterables, and mapping objects representing selected options.
    Splits string values on commas or semicolons, lowercases and trims whitespace,
    and ignores empty items.
    """

    def _add_item(item: object, results: set[str]) -> None:
        if item is None:
            return
        if isinstance(item, float) and math.isnan(item):
            return
        if isinstance(item, str):
            for piece in re.split(r"[;,]", item):
                normalized = piece.strip().lower()
                if normalized:
                    results.add(normalized)
            return
        normalized = str(item).strip().lower()
        if normalized:
            results.add(normalized)

    selections: set[str] = set()

    if value is None:
        return selections

    if isinstance(value, Mapping):
        for key, is_selected in value.items():
            if is_selected:
                _add_item(key, selections)
        return selections

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _add_item(item, selections)
        return selections

    _add_item(value, selections)
    return selections


def _parse_date(value: object) -> Optional[pd.Timestamp]:
    """Parse a datetime-like value into a pandas ``Timestamp``.

    Returns ``None`` when parsing fails or the input corresponds to ``NaT``.
    """

    if value is None:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed

    try:
        return pd.Timestamp(parsed)
    except (TypeError, ValueError):
        return None


def _parse_numeric(value: object) -> Optional[float]:
    """Parse a numeric value from strings or primitive numbers.

    Returns ``None`` if parsing fails or the value is missing/NaN.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            try:
                return float(cleaned.replace(",", ""))
            except ValueError:
                return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_experiment_metrics(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    label_config_bundle: LabelConfigBundle | Mapping[str, object] | None = None,
    labelset_id: str | None = None,
    ann_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """
    Compute structured evaluation metrics for inference experiments.

    Assumes:
      - gold_df has: unit_id, label_id, gold_value
      - pred_df has: unit_id, label_id, prediction_value
      - ann_df is the raw annotations DF used to build gold_df (optional)
      - label_config_bundle provides label types and families when available.

    Returns a dict with keys:
      - "global": overall summary metrics
      - "labels": per-label metrics
      - "families": family/gating-aware metrics
    """
    gold_df = gold_df.copy()
    pred_df = pred_df.copy()

    required_gold_cols = {"unit_id", "label_id", "gold_value"}
    required_pred_cols = {"unit_id", "label_id", "prediction_value"}

    if not required_gold_cols.issubset(gold_df.columns) or not required_pred_cols.issubset(
        pred_df.columns
    ):
        return {}

    label_types = _resolve_label_types(label_config_bundle, labelset_id, ann_df)

    merged = pd.merge(gold_df, pred_df, on=["unit_id", "label_id"], how="inner")

    per_label = _compute_per_label_metrics(merged, label_types)
    global_summary = _compute_global_summary(gold_df, pred_df)

    label_config = _resolve_label_config(label_config_bundle, labelset_id)
    try:
        families = _compute_family_metrics(gold_df, pred_df, label_types, label_config)
    except Exception:
        families = {}

    return {
        "global": global_summary,
        "labels": per_label,
        "families": families,
    }


def _resolve_label_types(
    label_config_bundle: LabelConfigBundle | Mapping[str, object] | None,
    labelset_id: str | None,
    ann_df: pd.DataFrame | None,
) -> dict[str, str]:
    label_types: dict[str, str] = {}

    if isinstance(label_config_bundle, LabelConfigBundle):
        label_config = (
            label_config_bundle.config_for_labelset(labelset_id)
            if labelset_id is not None
            else label_config_bundle.current or {}
        )
        *_, current_label_types = label_config_bundle.label_maps(
            label_config=label_config, ann_df=ann_df
        )
        label_types = dict(current_label_types)

    if not label_types and ann_df is not None:
        has_num = "label_value_num" in ann_df.columns
        has_date = "label_value_date" in ann_df.columns
        bin_tokens = {
            "0",
            "1",
            "true",
            "false",
            "present",
            "absent",
            "yes",
            "no",
            "neg",
            "pos",
            "positive",
            "negative",
            "unknown",
        }

        for lid, group in ann_df.groupby("label_id", sort=False):
            if has_num and group["label_value_num"].notna().any():
                label_types[str(lid)] = "numeric"
                continue
            if has_date and group["label_value_date"].notna().any():
                label_types[str(lid)] = "date"
                continue

            vals = group.get("label_value")
            if vals is not None:
                uniq = {
                    v
                    for v in vals.astype(str).str.lower().str.strip()
                    if v not in {"", "nan", "none"}
                }
                if uniq and uniq.issubset(bin_tokens):
                    label_types[str(lid)] = "binary"
                    continue

            label_types[str(lid)] = "categorical"

    normalized: dict[str, str] = {}
    allowed_types = {
        "binary",
        "categorical",
        "categorical_multi",
        "numeric",
        "ordinal",
        "date",
    }

    for lid, raw_type in label_types.items():
        key = str(lid)
        normalized_type = LabelConfigBundle._normalize_type(raw_type) or "categorical"
        if normalized_type == "categorical_single":
            normalized_type = "categorical"
        if normalized_type not in allowed_types:
            normalized_type = "categorical"
        normalized[key] = normalized_type

    return normalized


def _resolve_label_config(
    label_config_bundle: LabelConfigBundle | Mapping[str, object] | None,
    labelset_id: str | None,
) -> dict:
    if isinstance(label_config_bundle, LabelConfigBundle):
        label_config = label_config_bundle.config_for_labelset(labelset_id)
        if not label_config:
            return label_config_bundle.current or {}
        return label_config

    if isinstance(label_config_bundle, Mapping):
        return dict(label_config_bundle)

    return {}


def _compute_per_label_metrics(
    merged: pd.DataFrame,
    label_types: dict[str, str],
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}

    def _infer_label_type_from_values(gold_vals: pd.Series, pred_vals: pd.Series) -> str:
        values = pd.concat([gold_vals, pred_vals], ignore_index=True)

        for val in values:
            if isinstance(val, (list, tuple, set, Mapping)):
                return "categorical_multi"
            if isinstance(val, str) and re.search(r"[;,]", val):
                return "categorical_multi"

        date_count = sum(_parse_date(v) is not None for v in values)
        numeric_count = sum(_parse_numeric(v) is not None for v in values)

        if date_count > 0 and date_count >= numeric_count:
            return "date"
        if numeric_count > 0:
            return "numeric"
        return "categorical"

    for label_id, group in merged.groupby("label_id"):
        lid = str(label_id)
        label_type = label_types.get(lid) or _infer_label_type_from_values(
            group["gold_value"], group["prediction_value"]
        )
        result = _compute_label_metrics_for_type(
            lid, label_type, group["gold_value"], group["prediction_value"]
        )
        result.setdefault("type", label_type)
        metrics[lid] = result

    return metrics


def _compute_global_summary(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> dict[str, float]:
    merged = pd.merge(gold_df, pred_df, on=["unit_id", "label_id"], how="inner")

    if merged.empty:
        return {
            "n_pairs": 0,
            "overall_accuracy": 0.0,
            "micro_precision_yes": 0.0,
            "micro_recall_yes": 0.0,
            "micro_f1_yes": 0.0,
        }

    y_true = merged["gold_value"].astype(str)
    y_pred = merged["prediction_value"].astype(str)

    n = int(len(merged))
    overall_accuracy = float((y_true == y_pred).mean()) if n else 0.0

    def _canon_str(val: object) -> str:
        return re.sub(r"\s+", " ", str(val)).strip().lower()

    def _canon_cat(val: object) -> str:
        s = _canon_str(val)
        if s in {"y", "yes", "true", "1", "present", "positive", "pos"}:
            return "yes"
        if s in {"n", "no", "false", "0", "absent", "negative", "neg"}:
            return "no"
        return s

    gold_canon = y_true.apply(_canon_cat)
    pred_canon = y_pred.apply(_canon_cat)

    gold_positive = gold_canon == "yes"
    pred_positive = pred_canon == "yes"

    tp = int((gold_positive & pred_positive).sum())
    fp = int((~gold_positive & pred_positive).sum())
    fn = int((gold_positive & ~pred_positive).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    micro_f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "n_pairs": n,
        "overall_accuracy": overall_accuracy,
        "micro_precision_yes": precision,
        "micro_recall_yes": recall,
        "micro_f1_yes": micro_f1,
    }


def _compute_family_metrics(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_types: dict[str, str],
    label_config: dict,
) -> dict[str, dict[str, object]]:
    if not label_config:
        return {}

    try:
        parent_to_children, _child_to_parents, _roots = build_label_dependencies(label_config)
    except Exception:
        return {}

    def _canon_str(val: object) -> str:
        return re.sub(r"\s+", " ", str(val)).strip().lower()

    def _canon_cat(val: object) -> str:
        s = _canon_str(val)
        if s in {"y", "yes", "true", "1", "present", "positive", "pos"}:
            return "yes"
        if s in {"n", "no", "false", "0", "absent", "negative", "neg"}:
            return "no"
        return s

    family_metrics: dict[str, dict[str, object]] = {}

    for parent_id, children in parent_to_children.items():
        parent_gold = gold_df[gold_df["label_id"] == parent_id]
        parent_pred = pred_df[pred_df["label_id"] == parent_id]

        parent_merged = pd.merge(parent_gold, parent_pred, on=["unit_id", "label_id"], how="inner")

        if parent_merged.empty:
            parent_metrics = {
                "type": "binary",
                "n": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            family_metrics[parent_id] = {
                "parent_label_id": parent_id,
                "children": children,
                "parent_metrics": parent_metrics,
                "children_cond_parent_tp": {},
                "lost_due_to_parent_fn": 0.0,
                "spurious_families": 0,
                "end_to_end_recall": 0.0,
            }
            continue

        gold_canon = parent_merged["gold_value"].apply(_canon_cat)
        pred_canon = parent_merged["prediction_value"].apply(_canon_cat)

        gold_positive = gold_canon == "yes"
        pred_positive = pred_canon == "yes"

        tp_mask = gold_positive & pred_positive
        fp_mask = ~gold_positive & pred_positive
        fn_mask = gold_positive & ~pred_positive
        tn_mask = ~gold_positive & ~pred_positive

        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        fn = int(fn_mask.sum())
        tn = int(tn_mask.sum())

        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = float((tp + tn) / len(parent_merged)) if len(parent_merged) else 0.0

        parent_metrics = {
            "type": "binary",
            "n": int(len(parent_merged)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

        gold_event_ids = set(parent_merged.loc[gold_positive, "unit_id"].tolist())
        pred_event_ids = set(parent_merged.loc[pred_positive, "unit_id"].tolist())
        tp_units = set(parent_merged.loc[tp_mask, "unit_id"].tolist())
        fn_units = set(parent_merged.loc[fn_mask, "unit_id"].tolist())
        fp_units = set(parent_merged.loc[fp_mask, "unit_id"].tolist())

        children_cond_parent_tp: dict[str, dict[str, object]] = {}
        for child_id in children:
            child_gold = gold_df[
                (gold_df["label_id"] == child_id) & (gold_df["unit_id"].isin(tp_units))
            ]
            child_pred = pred_df[
                (pred_df["label_id"] == child_id) & (pred_df["unit_id"].isin(tp_units))
            ]

            child_merged = pd.merge(child_gold, child_pred, on=["unit_id", "label_id"], how="inner")
            child_type = label_types.get(str(child_id), "categorical")
            child_metrics = _compute_label_metrics_for_type(
                str(child_id), child_type, child_merged.get("gold_value", pd.Series()), child_merged.get("prediction_value", pd.Series())
            )
            child_metrics.setdefault("type", child_type)
            children_cond_parent_tp[str(child_id)] = child_metrics

        # End-to-end recall
        fully_correct = 0
        for uid in tp_units:
            unit_correct = True
            for child_id in children:
                child_type = label_types.get(str(child_id), "categorical")
                gold_row = gold_df[(gold_df["label_id"] == child_id) & (gold_df["unit_id"] == uid)]
                pred_row = pred_df[(pred_df["label_id"] == child_id) & (pred_df["unit_id"] == uid)]

                if gold_row.empty or pred_row.empty:
                    unit_correct = False
                    break

                gold_val = gold_row.iloc[0]["gold_value"]
                pred_val = pred_row.iloc[0]["prediction_value"]

                if child_type == "categorical_multi":
                    if _parse_multi_select(gold_val) != _parse_multi_select(pred_val):
                        unit_correct = False
                        break
                elif child_type == "date":
                    g_parsed = _parse_date(gold_val)
                    p_parsed = _parse_date(pred_val)
                    if g_parsed is None or p_parsed is None or g_parsed != p_parsed:
                        unit_correct = False
                        break
                elif child_type == "numeric":
                    g_num = _parse_numeric(gold_val)
                    p_num = _parse_numeric(pred_val)
                    if g_num is None or p_num is None or g_num != p_num:
                        unit_correct = False
                        break
                else:
                    if _canon_str(gold_val) != _canon_str(pred_val):
                        unit_correct = False
                        break

            if unit_correct:
                fully_correct += 1

        gold_events_count = len(gold_event_ids)
        end_to_end_recall = float(fully_correct / gold_events_count) if gold_events_count else 0.0
        lost_due_to_parent_fn = float(len(fn_units) / max(gold_events_count, 1))
        spurious_families = int(len(fp_units))

        family_metrics[parent_id] = {
            "parent_label_id": parent_id,
            "children": children,
            "parent_metrics": parent_metrics,
            "children_cond_parent_tp": children_cond_parent_tp,
            "lost_due_to_parent_fn": lost_due_to_parent_fn,
            "spurious_families": spurious_families,
            "end_to_end_recall": end_to_end_recall,
        }

    return family_metrics


def _compute_label_metrics_for_type(
    label_id: str,
    label_type: str,
    gold_vals: pd.Series,
    pred_vals: pd.Series,
) -> dict[str, object]:
    """
    Compute metrics for a single label, routed by normalized label_type.
    """
    normalized_type = (label_type or "categorical").lower()

    gold_series = pd.Series(gold_vals, dtype=object).copy()
    pred_series = pd.Series(pred_vals, dtype=object).copy()

    def _canon_str(val: object) -> str:
        return re.sub(r"\s+", " ", str(val)).strip().lower()

    def _canon_cat(val: object) -> str:
        s = _canon_str(val)
        if s in {"y", "yes", "true", "1", "present", "positive", "pos"}:
            return "yes"
        if s in {"n", "no", "false", "0", "absent", "negative", "neg"}:
            return "no"
        return s

    missing_tokens = {"", "nan", "none"}

    # Categorical (including binary/ordinal treated similarly)
    if normalized_type in {"binary", "categorical", "categorical_single", "ordinal"}:
        gold_clean = gold_series.apply(_canon_str)
        pred_clean = pred_series.apply(_canon_str)

        mask = (~gold_clean.isin(missing_tokens)) & (~pred_clean.isin(missing_tokens))
        if not mask.any():
            return {
                "type": normalized_type,
                "n": 0,
                "accuracy": 0.0,
                "precision": None,
                "recall": None,
                "f1": 0.0,
                "macro_f1": None,
            }

        gold_clean = gold_clean[mask]
        pred_clean = pred_clean[mask]

        n = int(len(gold_clean))
        accuracy = float((gold_clean == pred_clean).mean()) if n else 0.0

        binary_tokens = {
            "yes",
            "no",
            "present",
            "absent",
            "true",
            "false",
            "1",
            "0",
            "positive",
            "negative",
            "pos",
            "neg",
        }

        gold_canon = gold_clean.apply(_canon_cat)
        pred_canon = pred_clean.apply(_canon_cat)

        is_binary = set(gold_canon.unique()).issubset(binary_tokens)

        if is_binary:
            gold_positive = gold_canon == "yes"
            pred_positive = pred_canon == "yes"

            tp = int((gold_positive & pred_positive).sum())
            fp = int((~gold_positive & pred_positive).sum())
            fn = int((gold_positive & ~pred_positive).sum())
            tn = int((~gold_positive & ~pred_positive).sum())

            precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
            f1 = (
                float(2 * precision * recall / (precision + recall))
                if (precision + recall)
                else 0.0
            )

            return {
                "type": normalized_type,
                "n": n,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "macro_f1": None,
            }

        # Multi-class macro-F1
        classes = [c for c in gold_canon.unique() if c not in missing_tokens]
        per_class_f1 = []
        for cls in classes:
            gold_pos = gold_canon == cls
            pred_pos = pred_canon == cls
            tp = int((gold_pos & pred_pos).sum())
            fp = int((~gold_pos & pred_pos).sum())
            fn = int((gold_pos & ~pred_pos).sum())

            precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
            if (tp + fn) > 0:
                f1_cls = (
                    float(2 * precision * recall / (precision + recall))
                    if (precision + recall)
                    else 0.0
                )
                per_class_f1.append(f1_cls)

        macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

        return {
            "type": normalized_type,
            "n": n,
            "accuracy": accuracy,
            "precision": None,
            "recall": None,
            "f1": macro_f1,
            "macro_f1": macro_f1,
        }

    # Multi-select categorical
    if normalized_type == "categorical_multi":
        gold_sets = gold_series.apply(_parse_multi_select)
        pred_sets = pred_series.apply(_parse_multi_select)

        rows = []
        for g, p in zip(gold_sets, pred_sets):
            if not g and not p:
                continue
            rows.append((g, p))

        if not rows:
            return {
                "type": "categorical_multi",
                "n": 0,
                "set_exact_accuracy": 0.0,
                "mean_jaccard": 0.0,
                "mean_over_predict": 0.0,
                "mean_under_predict": 0.0,
                "items": {},
            }

        exact_matches: list[int] = []
        jaccards: list[float] = []
        over_preds: list[int] = []
        under_preds: list[int] = []
        item_counts: dict[str, dict[str, int]] = {}

        for g, p in rows:
            exact_matches.append(1 if g == p else 0)
            union = g | p
            inter = g & p
            j = float(len(inter) / len(union)) if union else 1.0
            jaccards.append(j)
            over = p - g
            under = g - p
            over_preds.append(len(over))
            under_preds.append(len(under))

            for token in g | p:
                stats = item_counts.setdefault(token, {"tp": 0, "fp": 0, "fn": 0})
                if token in g and token in p:
                    stats["tp"] += 1
                elif token in p and token not in g:
                    stats["fp"] += 1
                elif token in g and token not in p:
                    stats["fn"] += 1

        n = len(rows)
        item_metrics: dict[str, dict[str, float]] = {}
        for token, counts in item_counts.items():
            tp = counts.get("tp", 0)
            fp = counts.get("fp", 0)
            fn = counts.get("fn", 0)
            precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
            f1 = (
                float(2 * precision * recall / (precision + recall))
                if (precision + recall)
                else 0.0
            )
            item_metrics[token] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(tp + fn),
            }

        return {
            "type": "categorical_multi",
            "n": n,
            "set_exact_accuracy": float(np.mean(exact_matches)),
            "mean_jaccard": float(np.mean(jaccards)),
            "mean_over_predict": float(np.mean(over_preds)),
            "mean_under_predict": float(np.mean(under_preds)),
            "items": item_metrics,
        }

    # Date metrics
    if normalized_type == "date":
        gold_parsed = gold_series.apply(_parse_date)
        pred_parsed = pred_series.apply(_parse_date)

        paired = [
            (g, p)
            for g, p in zip(gold_parsed, pred_parsed)
            if g is not None and p is not None
        ]

        if not paired:
            return {
                "type": "date",
                "n": 0,
                "exact_match": 0.0,
                "within_3d": 0.0,
                "within_7d": 0.0,
                "within_30d": 0.0,
                "mae_days": 0.0,
                "p90_abs_error": 0.0,
            }

        abs_diff_days = [abs((p - g).total_seconds()) / 86400.0 for g, p in paired]
        abs_diff_series = pd.Series(abs_diff_days, dtype=float)

        n = len(abs_diff_days)
        return {
            "type": "date",
            "n": n,
            "exact_match": float((abs_diff_series == 0).mean()),
            "within_3d": float((abs_diff_series <= 3).mean()),
            "within_7d": float((abs_diff_series <= 7).mean()),
            "within_30d": float((abs_diff_series <= 30).mean()),
            "mae_days": float(abs_diff_series.mean()),
            "p90_abs_error": float(abs_diff_series.quantile(0.9)) if n else 0.0,
        }

    # Numeric metrics
    if normalized_type == "numeric":
        paired = [
            (g, p)
            for g, p in zip(gold_series, pred_series)
            if _parse_numeric(g) is not None and _parse_numeric(p) is not None
        ]

        if not paired:
            return {
                "type": "numeric",
                "n": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "within_10pct": 0.0,
            }

        gold_clean = []
        pred_clean = []
        for g, p in paired:
            g_parsed = _parse_numeric(g)
            p_parsed = _parse_numeric(p)
            if g_parsed is None or p_parsed is None:
                continue
            gold_clean.append(g_parsed)
            pred_clean.append(p_parsed)

        if not gold_clean:
            return {
                "type": "numeric",
                "n": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "within_10pct": 0.0,
            }

        gold_arr = np.array(gold_clean, dtype=float)
        pred_arr = np.array(pred_clean, dtype=float)
        diff = pred_arr - gold_arr
        abs_diff = np.abs(diff)

        eps = 1e-6
        tol = np.maximum(0.1 * np.abs(gold_arr), eps)

        return {
            "type": "numeric",
            "n": len(gold_arr),
            "mae": float(abs_diff.mean()),
            "rmse": float(np.sqrt(np.mean(diff**2))) if len(diff) else 0.0,
            "within_10pct": float(np.mean(abs_diff <= tol)),
        }

    # Fallback
    return {
        "type": normalized_type,
        "n": 0,
    }
