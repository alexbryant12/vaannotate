import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vaannotate.project import build_label_config
from vaannotate.vaannotate_ai_backend.services.label_dependencies import evaluate_gating


def test_evaluate_gating_categorical_contains_with_or_logic() -> None:
    label_config = {
        "Child": {
            "gating_logic": "OR",
            "gating_rules": [
                {"parent": "Parent", "type": "categorical", "op": "contains", "values": ["X"]},
                {"parent": "Parent", "type": "categorical", "op": "contains", "values": ["Y"]},
            ],
        }
    }
    label_types = {"Parent": "categorical_multi", "Child": "text"}
    parent_preds = {("u1", "Parent"): "A,Y,Z"}

    assert evaluate_gating("Child", "u1", parent_preds, label_types, label_config) is True
    assert evaluate_gating("Child", "u2", {("u2", "Parent"): "A,Z"}, label_types, label_config) is False


def test_evaluate_gating_uses_build_label_config_rich_expr() -> None:
    labelset = {
        "labelset_id": "ls_or_contains",
        "labelset_name": "Demo OR/contains",
        "labels": [
            {
                "label_id": "Parent",
                "name": "Parent Label",
                "type": "categorical_multi",
                "required": False,
                "options": [
                    {"value": "X", "display": "X"},
                    {"value": "Y", "display": "Y"},
                    {"value": "Z", "display": "Z"},
                ],
            },
            {
                "label_id": "Child",
                "name": "Child Label",
                "type": "text",
                "required": False,
                "gating_expr": "Parent Label contains 'X' or Parent Label contains 'Y'",
                "options": [],
            },
        ],
    }
    label_config = build_label_config(labelset)
    label_types = {"Parent": "categorical_multi", "Child": "text"}
    assert evaluate_gating("Child", "u1", {("u1", "Parent"): "A,Z"}, label_types, label_config) is False
    assert evaluate_gating("Child", "u1", {("u1", "Parent"): "A,Y,Z"}, label_types, label_config) is True
