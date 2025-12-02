import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from vaannotate.project import (
    add_labelset,
    add_phenotype,
    build_label_config,
    fetch_labelset,
    get_connection,
    init_project,
    resolve_label_config_path,
)


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
    assert child_entry.get("gated_by") == "Root"
    rules = child_entry.get("gating_rules")
    assert isinstance(rules, list) and rules
    assert rules[0]["parent"] == "Root"
    assert rules[0]["type"] == "categorical"
    assert rules[0]["op"] == "in"
    assert rules[0]["values"] == ["yes"]

    grandchild_entry = config.get("ChildB")
    assert isinstance(grandchild_entry, dict)
    assert grandchild_entry.get("parents")[0]["label_id"] == "ChildA"
    assert grandchild_entry.get("gating_expr") == "Child A == 'positive'"
    assert grandchild_entry.get("gated_by") == "ChildA"
    grules = grandchild_entry.get("gating_rules")
    assert isinstance(grules, list) and grules
    assert grules[0]["parent"] == "ChildA"
    assert grules[0]["values"] == ["positive"]


def test_resolve_label_config_path_in_labelsets_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    paths = init_project(project_root, "proj", "Project", "tester")

    target = resolve_label_config_path(paths.root, "ls_demo")

    assert target == paths.root / "label_sets" / "ls_demo" / "label_config.json"


def test_add_labelset_enforces_uniqueness_within_set(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    paths = init_project(project_root, "proj", "Project", "tester")
    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id="phen",
            project_id="proj",
            name="Phenotype",
            level="single_doc",
            storage_path="phenotypes/phen",
        )
        with pytest.raises(ValueError):
            add_labelset(
                conn,
                labelset_id="ls_dup",
                project_id="proj",
                pheno_id="phen",
                version=1,
                created_by="tester",
                notes=None,
                labels=[
                    {"label_id": "dup", "name": "A", "type": "text", "required": False},
                    {"label_id": "dup", "name": "B", "type": "text", "required": False},
                ],
            )


def test_add_labelset_allows_duplicate_ids_across_sets(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    paths = init_project(project_root, "proj", "Project", "tester")
    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id="phen",
            project_id="proj",
            name="Phenotype",
            level="single_doc",
            storage_path="phenotypes/phen",
        )
        add_labelset(
            conn,
            labelset_id="ls_one",
            project_id="proj",
            pheno_id="phen",
            version=1,
            created_by="tester",
            notes=None,
            labels=[
                {"label_id": "shared", "name": "Shared", "type": "text", "required": False}
            ],
        )
        add_labelset(
            conn,
            labelset_id="ls_two",
            project_id="proj",
            pheno_id="phen",
            version=1,
            created_by="tester",
            notes=None,
            labels=[
                {"label_id": "shared", "name": "Shared", "type": "text", "required": False}
            ],
        )
        rows = conn.execute(
            "SELECT labelset_id, label_id FROM labels ORDER BY labelset_id"
        ).fetchall()
        assert [(row["labelset_id"], row["label_id"]) for row in rows] == [
            ("ls_one", "shared"),
            ("ls_two", "shared"),
        ]


def test_label_keywords_and_examples_round_trip(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    paths = init_project(project_root, "proj", "Project", "tester")

    labelset_payload = {
        "labelset_id": "ls_features",
        "project_id": "proj",
        "pheno_id": "phen",
        "version": 1,
        "created_by": "tester",
        "notes": None,
        "labels": [
            {
                "label_id": "kw1",
                "name": "Keywords",
                "type": "text",
                "required": False,
                "keywords": ["alpha", "beta"],
                "few_shot_examples": [
                    {"context": "example context", "answer": "answer text"}
                ],
            }
        ],
    }

    with get_connection(paths.project_db) as conn:
        add_phenotype(
            conn,
            pheno_id="phen",
            project_id="proj",
            name="Phenotype",
            level="single_doc",
            storage_path="phenotypes/phen",
        )
        add_labelset(conn, **labelset_payload)

        fetched = fetch_labelset(conn, "ls_features")

    labels = fetched.get("labels", [])
    assert isinstance(labels, list) and labels
    label = labels[0]
    assert label["keywords"] == ["alpha", "beta"]
    assert label["few_shot_examples"] == [
        {"context": "example context", "answer": "answer text"}
    ]

    config = build_label_config(fetched)
    entry = config.get("kw1")
    assert isinstance(entry, dict)
    assert entry.get("keywords") == ["alpha", "beta"]
    assert entry.get("few_shot_examples") == [
        {"context": "example context", "answer": "answer text"}
    ]
