import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vaannotate.vaannotate_ai_backend.experiment_metrics import compute_experiment_metrics


def test_binary_metrics_simple_perfect_match():
    gold_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "p", "gold_value": "yes"},
            {"unit_id": 2, "label_id": "p", "gold_value": "no"},
        ]
    )
    pred_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "p", "prediction_value": "yes"},
            {"unit_id": 2, "label_id": "p", "prediction_value": "no"},
        ]
    )

    metrics = compute_experiment_metrics(
        gold_df,
        pred_df,
        label_config_bundle=None,
        labelset_id=None,
        ann_df=None,
    )

    assert metrics["global"]["overall_accuracy"] == 1.0
    assert metrics["labels"]["p"]["accuracy"] == 1.0


def test_multi_select_exact_match():
    gold_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "procs", "gold_value": "PSA,Biopsy,MRI"},
        ]
    )
    pred_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "procs", "prediction_value": "Biopsy,MRI,PSA"},
        ]
    )

    metrics = compute_experiment_metrics(
        gold_df,
        pred_df,
        label_config_bundle=None,
        labelset_id=None,
        ann_df=None,
    )

    assert metrics["labels"]["procs"]["set_exact_accuracy"] == 1.0
    assert metrics["labels"]["procs"]["mean_jaccard"] == 1.0


def test_date_within_window():
    gold_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "onset", "gold_value": "2024-01-01"},
        ]
    )
    pred_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "onset", "prediction_value": "2024-01-03"},
        ]
    )

    metrics = compute_experiment_metrics(
        gold_df,
        pred_df,
        label_config_bundle=None,
        labelset_id=None,
        ann_df=None,
    )

    assert metrics["labels"]["onset"].get("within_3d") == 1.0
    assert metrics["labels"]["onset"].get("exact_match") == 0.0


def test_numeric_mae():
    gold_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "psa", "gold_value": "10.0"},
        ]
    )
    pred_df = pd.DataFrame(
        [
            {"unit_id": 1, "label_id": "psa", "prediction_value": "11.0"},
        ]
    )

    metrics = compute_experiment_metrics(
        gold_df,
        pred_df,
        label_config_bundle=None,
        labelset_id=None,
        ann_df=None,
    )

    assert metrics["labels"]["psa"].get("mae") == 1.0
