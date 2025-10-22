import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vaannotate.project import build_label_config


def test_build_label_config_dependency_tree() -> None:
    labelset = {
        "labelset_id": "ls_demo",
        "labelset_name": "Demo",
        "notes": "Example label set",
        "created_by": "tester",
        "created_at": "2024-01-01T00:00:00",
        "labels": [
            {
                "label_id": "Root",
                "name": "Root",
                "type": "boolean",
                "required": True,
                "na_allowed": False,
                "options": [
                    {"value": "yes", "display": "Yes"},
                    {"value": "no", "display": "No"},
                ],
            },
            {
                "label_id": "ChildA",
                "name": "Child A",
                "type": "text",
                "required": False,
                "gating_expr": "Root == 'yes'",
                "options": [],
            },
            {
                "label_id": "ChildB",
                "name": "Child B",
                "type": "text",
                "required": False,
                "gating_expr": "Child A == 'positive'",
                "options": [],
            },
        ],
    }

    config = build_label_config(labelset)

    meta = config.get("_meta")
    assert isinstance(meta, dict)
    assert meta.get("labelset_id") == "ls_demo"

    tree = meta.get("dependency_tree")
    assert isinstance(tree, list) and tree
    root_node = tree[0]
    assert root_node.get("label_id") == "Root"
    children = root_node.get("children")
    assert isinstance(children, list) and len(children) == 1
    assert children[0].get("label_id") == "ChildA"

    child_entry = config.get("ChildA")
    assert isinstance(child_entry, dict)
    assert child_entry.get("parents")[0]["label_id"] == "Root"
    assert child_entry.get("children")[0]["label_id"] == "ChildB"

    grandchild_entry = config.get("ChildB")
    assert isinstance(grandchild_entry, dict)
    assert grandchild_entry.get("parents")[0]["label_id"] == "ChildA"
    assert grandchild_entry.get("gating_expr") == "Child A == 'positive'"
