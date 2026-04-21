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
