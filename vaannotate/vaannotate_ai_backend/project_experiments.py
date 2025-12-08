"""Helpers for project-level inference experiments."""

import copy
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import pandas as pd

from .adapters import _load_label_config_bundle, export_inputs_from_repo
from .config import OrchestratorConfig, Paths
from .experiment_metrics import compute_experiment_metrics
from .experiments import (
    InferenceExperimentResult,
    _normalize_local_model_overrides,
    run_inference_experiments,
)
from .orchestration import BackendSession
from .orchestrator import _apply_overrides


def merge_cfg_overrides(base: dict | None, overrides: dict | None) -> dict:
    """Deep-merge overrides onto a baseline using ``_apply_overrides`` semantics."""

    combined = copy.deepcopy(base) if base is not None else {}
    _apply_overrides(combined, overrides or {})
    return combined


def _infer_unit_id_column(notes_df: pd.DataFrame, phenotype_level: str) -> pd.Series:
    """
    Return a Series of unit_id strings aligned with the phenotype level.

    Behavior:
    - If notes_df already has a 'unit_id' column, return it as strings.
    - Else if phenotype_level is 'multi_doc', use 'patient_icn' as unit_id.
    - Else (single_doc), use 'doc_id' as unit_id.
    """
    if "unit_id" in notes_df.columns:
        return notes_df["unit_id"].astype(str)

    if phenotype_level == "multi_doc":
        return notes_df["patient_icn"].astype(str)

    return notes_df["doc_id"].astype(str)


def build_gold_from_ann(
    ann_df: pd.DataFrame,
    *,
    labelset_id: str | None = None,
    min_reviewers: int = 1,
) -> pd.DataFrame:
    """
    Return one row per (unit_id, label_id) with a consensus gold label.

    Output columns:
    - unit_id (str)
    - label_id (str)
    - gold_value (chosen label_value)
    """

    ann_df = ann_df.copy()

    for column in ("unit_id", "label_id"):
        if column not in ann_df.columns:
            raise KeyError(f"Expected column '{column}' in annotations")
        ann_df[column] = ann_df[column].astype(str)

    if labelset_id is not None and "labelset_id" in ann_df.columns:
        ann_df = ann_df[ann_df["labelset_id"].astype(str) == str(labelset_id)]

    if "label_na" in ann_df.columns:
        label_na_mask = ann_df["label_na"].fillna(False).astype(bool)
        ann_df = ann_df[~label_na_mask]

    if "label_value" not in ann_df.columns:
        raise KeyError("Expected column 'label_value' in annotations")

    ann_df = ann_df[ann_df["label_value"].notna()]

    records: List[Mapping[str, str]] = []

    for (unit_id, label_id), group in ann_df.groupby(["unit_id", "label_id"]):
        if len(group) < min_reviewers:
            continue

        gold_value = group["label_value"].value_counts().idxmax()
        records.append(
            {
                "unit_id": str(unit_id),
                "label_id": str(label_id),
                "gold_value": gold_value,
            }
        )

    return pd.DataFrame(records, columns=["unit_id", "label_id", "gold_value"])


def compute_classification_metrics(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Compute per-label metrics.

    Assumes:
    - gold_df has: unit_id, label_id, gold_value
    - pred_df has: unit_id, label_id, prediction_value

    Returns:
      { label_id: { "n": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ... }, ... }
    }
    """

    merged = gold_df.merge(pred_df, on=["unit_id", "label_id"], how="inner")
    metrics: dict[str, dict[str, float]] = {}

    for label_id, group in merged.groupby("label_id"):
        gold_vals = group["gold_value"].astype(str)
        pred_vals = group["prediction_value"].astype(str)

        n = int(len(group))
        accuracy = float((gold_vals == pred_vals).mean()) if n else 0.0

        gold_positive = gold_vals == "yes"
        pred_positive = pred_vals == "yes"

        tp = int((gold_positive & pred_positive).sum())
        fp = int((~gold_positive & pred_positive).sum())
        fn = int((gold_positive & ~pred_positive).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics[str(label_id)] = {
            "n": n,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics


def _collate_experiment_predictions(
    results: Dict[str, InferenceExperimentResult],
) -> pd.DataFrame:
    """
    Return a tall DataFrame with predictions from all sweeps.

    Columns:
      - experiment_name
      - unit_id
      - label_id
      - prediction_value (best-effort normalized)
      - plus any other prediction/metadata columns preserved from the
        per-experiment inference_predictions.* files.
    """

    rows: list[pd.DataFrame] = []

    for name, result in (results or {}).items():
        df = getattr(result, "dataframe", None)
        if df is None or df.empty:
            # Fallback to inference_predictions.parquet / .csv in result.outdir
            exp_outdir = Path(result.outdir)
            parquet_path = exp_outdir / "inference_predictions.parquet"
            csv_path = exp_outdir / "inference_predictions.csv"
            artifact_path: Path | None = None
            if parquet_path.exists():
                artifact_path = parquet_path
            elif csv_path.exists():
                artifact_path = csv_path
            if artifact_path is not None and artifact_path.exists():
                if artifact_path.suffix == ".parquet":
                    df = pd.read_parquet(artifact_path)
                else:
                    df = pd.read_csv(artifact_path)
            else:
                df = pd.DataFrame()

        if df is None or df.empty:
            continue

        df = df.copy()
        # Ensure unit_id / label_id exist and are strings.
        if "unit_id" in df.columns:
            df["unit_id"] = df["unit_id"].astype(str)
        if "label_id" in df.columns:
            df["label_id"] = df["label_id"].astype(str)

        # Best-effort detection of the main prediction column.
        if "prediction_value" not in df.columns:
            for candidate in ("llm_prediction", "label_value", "label_option_id", "prediction"):
                if candidate in df.columns:
                    df["prediction_value"] = df[candidate].astype(str)
                    break

        # Skip experiments that somehow still do not have the basics.
        if not {"unit_id", "label_id", "prediction_value"} <= set(df.columns):
            continue

        df["experiment_name"] = str(name)

        # Keep cfg_overrides as a JSON blob for downstream inspection.
        overrides = getattr(result, "cfg_overrides", {}) or {}
        try:
            df["cfg_overrides_json"] = json.dumps(overrides)
        except Exception:  # noqa: BLE001
            df["cfg_overrides_json"] = None

        rows.append(df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "experiment_name",
                "unit_id",
                "label_id",
                "prediction_value",
            ]
        )

    collated = pd.concat(rows, ignore_index=True)
    # Ensure key columns are strings
    for col in ("experiment_name", "unit_id", "label_id", "prediction_value"):
        if col in collated.columns:
            collated[col] = collated[col].astype(str)
    return collated


def _compute_metrics_for_all_sweeps(
    gold_df: pd.DataFrame,
    collated_preds: pd.DataFrame,
    *,
    label_config_bundle,
    labelset_id: str | None,
    ann_df: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    """
    Compute experiment_metrics for each sweep.

    Returns:
      { experiment_name: metrics_dict_from_compute_experiment_metrics, ... }
    """

    metrics_by_sweep: dict[str, dict[str, object]] = {}

    if gold_df is None or gold_df.empty:
        return metrics_by_sweep
    if collated_preds is None or collated_preds.empty:
        return metrics_by_sweep
    if "experiment_name" not in collated_preds.columns:
        return metrics_by_sweep

    # Ensure canonical columns
    base_cols = {"unit_id", "label_id", "prediction_value"}
    missing = [c for c in base_cols if c not in collated_preds.columns]
    if missing:
        return metrics_by_sweep

    for name, group in collated_preds.groupby("experiment_name"):
        # Only the minimal columns are needed for metrics; everything else
        # is kept in the collated parquet for inspection.
        pred_df = group[list(base_cols)].copy()
        for col in base_cols:
            pred_df[col] = pred_df[col].astype(str)

        try:
            sweep_metrics = compute_experiment_metrics(
                gold_df,
                pred_df,
                label_config_bundle=label_config_bundle,
                labelset_id=labelset_id,
                ann_df=ann_df,
            )
        except Exception as exc:  # noqa: BLE001
            # Fail-soft: record the error instead of blowing up the entire sweep.
            sweep_metrics = {"error": f"{type(exc).__name__}: {exc}"}
        metrics_by_sweep[str(name)] = sweep_metrics

    return metrics_by_sweep


def run_project_inference_experiments(
    project_root: Path,
    pheno_id: str,
    prior_rounds: list[int],
    *,
    labelset_id: str | None,
    phenotype_level: str,
    sweeps: dict[str, dict],
    base_outdir: Path,
    corpus_record: Mapping[str, object] | None = None,
    corpus_id: str | None = None,
    corpus_path: str | None = None,
    cfg_overrides_base: dict | None = None,
) -> Tuple[Dict[str, InferenceExperimentResult], pd.DataFrame]:
    """
    Use prior rounds to build a gold-standard set and run a sweep of
    inference experiments. Returns (results_dict, gold_df).

    The ``cfg_overrides_base`` configuration is deep-merged with each sweep's
    overrides (via :func:`merge_cfg_overrides` and ``_apply_overrides``
    semantics) so sweeps can specify only the deltas from a tuned baseline.
    Label configurations are loaded via ``_load_label_config_bundle`` just like
    the main AI backend path, so label rules/types/gating and prior-round gold
    construction remain aligned for experiment metrics; only a
    ``cfg_overrides_base['label_config']`` payload is treated as a label
    configuration override to avoid stomping production labelsets with
    unrelated sweep parameters.
    Sweeps that alter ``models.embed_model_name`` or ``models.rerank_model_name``
    should avoid reusing a shared :class:`BackendSession`, because the
    embedding store is specific to the embedder; sweeps that only tweak
    RAG/LLM knobs can safely share the session for speed.
    Sweeps or inference experiments can switch to single-prompt label inference
    with a cfg override like ``{"llmfirst": {"inference_labeling_mode": "single_prompt"}}``.
    """

    notes_df, ann_df = export_inputs_from_repo(
        project_root,
        pheno_id,
        prior_rounds,
        corpus_record=corpus_record,
        corpus_id=corpus_id,
        corpus_path=corpus_path,
    )

    notes_df = notes_df.copy()
    notes_df["unit_id"] = _infer_unit_id_column(notes_df, phenotype_level)

    gold_df = build_gold_from_ann(ann_df, labelset_id=labelset_id)
    eval_unit_ids = sorted(
        {str(uid) for uid in gold_df["unit_id"].unique() if pd.notna(uid) and str(uid)}
    )

    label_config_override = None
    if cfg_overrides_base and isinstance(cfg_overrides_base.get("label_config"), Mapping):
        label_config_override = cfg_overrides_base["label_config"]

    label_config_bundle = _load_label_config_bundle(
        project_root,
        pheno_id,
        labelset_id,
        prior_rounds,
        overrides=label_config_override,
    )

    base_overrides = _normalize_local_model_overrides(cfg_overrides_base or {})

    base_cfg = OrchestratorConfig()
    if base_overrides:
        _apply_overrides(base_cfg, dict(base_overrides))

    session_paths = Paths(
        notes_path=str(base_outdir / "_session_notes.parquet"),
        annotations_path=str(base_outdir / "_session_annotations.parquet"),
        outdir=str(base_outdir / "_session"),
        cache_dir_override=str(base_outdir / "cache"),
    )
    session = BackendSession.from_env(session_paths, base_cfg)

    sweeps_with_base = {
        name: _normalize_local_model_overrides(
            merge_cfg_overrides(base_overrides, dict(overrides))
        )
        for name, overrides in sweeps.items()
    }

    results = run_inference_experiments(
        notes_df=notes_df,
        ann_df=ann_df,
        base_outdir=base_outdir,
        sweeps=sweeps_with_base,
        unit_ids=eval_unit_ids,
        label_config_bundle=label_config_bundle,
        session=session,
    )

    for name, result in results.items():
        pred_df = getattr(result, "dataframe", None)

        if pred_df is None or pred_df.empty:
            artifact_path = result.artifacts.get("predictions") if result.artifacts else None
            fallback_dir = result.outdir if result.outdir else base_outdir / name
            fallback_parquet = Path(fallback_dir) / "inference_predictions.parquet"
            fallback_csv = Path(fallback_dir) / "inference_predictions.csv"

            if artifact_path:
                artifact_path = Path(artifact_path)
            elif fallback_parquet.exists():
                artifact_path = fallback_parquet
            elif fallback_csv.exists():
                artifact_path = fallback_csv
            else:
                artifact_path = None

            if artifact_path is not None and artifact_path.exists():
                if artifact_path.suffix == ".parquet":
                    pred_df = pd.read_parquet(artifact_path)
                else:
                    pred_df = pd.read_csv(artifact_path)
            else:
                pred_df = pd.DataFrame()

        pred_df = pred_df.copy()
        column_map = {}
        if "prediction_value" in pred_df.columns:
            column_map["prediction_value"] = "prediction_value"
        elif "llm_prediction" in pred_df.columns:
            column_map["llm_prediction"] = "prediction_value"
        elif "label_value" in pred_df.columns:
            column_map["label_value"] = "prediction_value"
        elif "prediction" in pred_df.columns:
            column_map["prediction"] = "prediction_value"

        pred_df.rename(columns=column_map, inplace=True)

        required_cols = ["unit_id", "label_id", "prediction_value"]
        missing = [c for c in required_cols if c not in pred_df.columns]
        if missing:
            pred_df = pd.DataFrame(columns=required_cols)
        else:
            pred_df = pred_df[required_cols].copy()
            for col in ["unit_id", "label_id", "prediction_value"]:
                pred_df[col] = pred_df[col].astype(str)

        metrics = compute_experiment_metrics(
            gold_df,
            pred_df,
            label_config_bundle=label_config_bundle,
            labelset_id=labelset_id,
            ann_df=ann_df,
        )

        exp_outdir = result.outdir or (base_outdir / name)
        exp_outdir = Path(exp_outdir)
        exp_outdir.mkdir(parents=True, exist_ok=True)

        metrics_path = exp_outdir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    base_outdir.mkdir(parents=True, exist_ok=True)

    gold_out_path = base_outdir / "gold_labels.parquet"
    try:
        gold_df.to_parquet(gold_out_path, index=False)
    except Exception:  # noqa: BLE001
        gold_df.to_csv(gold_out_path, index=False)

    collated_preds = _collate_experiment_predictions(results)
    collated_path = base_outdir / "experiments_predictions.parquet"
    try:
        collated_preds.to_parquet(collated_path, index=False)
    except Exception:  # noqa: BLE001
        collated_preds.to_csv(collated_path, index=False)

    summary_metrics = _compute_metrics_for_all_sweeps(
        gold_df,
        collated_preds,
        label_config_bundle=label_config_bundle,
        labelset_id=labelset_id,
        ann_df=ann_df,
    )
    summary = {"sweeps": summary_metrics}
    summary_path = base_outdir / "experiments_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return results, gold_df
