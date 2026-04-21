import sys
from pathlib import Path

import pytest

try:
    from PySide6 import QtWidgets  # noqa: F401
except ImportError:  # pragma: no cover - skip if Qt not available
    pytest.skip("PySide6 is not available", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.AdminApp.main import ProjectContext
from vaannotate.project import init_project


def test_project_context_defers_file_deletion(tmp_path: Path) -> None:
    ctx = ProjectContext()
    file_path = tmp_path / "rounds" / "round_1" / "manifest.csv"
    ctx.register_manifest(file_path, {})

    # Scheduling deletion should remove pending artifacts immediately.
    ctx._schedule_deletion(file_path, mode="file")
    assert file_path.resolve() not in ctx._pending_manifests

    ctx.save_all()

    assert not file_path.exists()
    assert not ctx._pending_deletions


def test_registering_artifact_clears_pending_deletion(tmp_path: Path) -> None:
    ctx = ProjectContext()
    round_dir = tmp_path / "rounds" / "round_1"
    manifest_path = round_dir / "manifest.csv"

    ctx._schedule_deletion(round_dir, mode="tree")
    assert ctx._pending_deletions

    ctx.register_manifest(manifest_path, {})
    assert not ctx._pending_deletions

    text_path = round_dir / "notes.txt"
    ctx._schedule_deletion(text_path, mode="file")
    assert ctx._pending_deletions

    ctx.register_text_file(text_path, "sample")
    assert not ctx._pending_deletions


def test_project_context_allows_same_label_ids_across_labelsets(tmp_path: Path) -> None:
    paths = init_project(tmp_path / "proj", "proj", "Project", "tester")
    ctx = ProjectContext()
    ctx.open_project(paths.root)
    pheno = ctx.create_phenotype(name="Phen", level="single_doc", description="")

    for labelset_id in ("ls_a", "ls_b"):
        ctx.create_labelset(
            labelset_id=labelset_id,
            created_by="tester",
            notes="",
            include_reasoning=False,
            pheno_id=pheno.pheno_id,
            labels=[
                {
                    "label_id": "colitis",
                    "name": "Colitis",
                    "type": "boolean",
                    "required": False,
                    "na_allowed": False,
                    "rules": "",
                    "options": [{"value": "yes", "display": "Yes"}],
                }
            ],
        )

    details_a = ctx.load_labelset_details("ls_a")
    details_b = ctx.load_labelset_details("ls_b")
    assert details_a and details_b
    assert details_a["labels"][0]["label_id"] == "colitis"
    assert details_b["labels"][0]["label_id"] == "colitis"


def test_project_context_blocks_update_delete_for_labelsets_used_by_rounds(tmp_path: Path) -> None:
    paths = init_project(tmp_path / "proj", "proj", "Project", "tester")
    ctx = ProjectContext()
    ctx.open_project(paths.root)
    pheno = ctx.create_phenotype(name="Phen", level="single_doc", description="")
    ctx.create_labelset(
        labelset_id="ls_used",
        created_by="tester",
        notes="",
        include_reasoning=False,
        pheno_id=pheno.pheno_id,
        labels=[
            {
                "label_id": "colitis",
                "name": "Colitis",
                "type": "boolean",
                "required": False,
                "na_allowed": False,
                "rules": "",
                "options": [{"value": "yes", "display": "Yes"}],
            }
        ],
    )
    db = ctx.require_db()
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO rounds(round_id,pheno_id,round_number,labelset_id,config_hash,rng_seed,status,created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            ("r1", pheno.pheno_id, 1, "ls_used", "hash", 1, "draft", "2026-01-01T00:00:00"),
        )
    assert ctx.labelset_round_usage("ls_used") == 1
    with pytest.raises(RuntimeError):
        ctx.update_labelset(
            labelset_id="ls_used",
            created_by="tester",
            notes="updated",
            include_reasoning=False,
            labels=[],
        )
    with pytest.raises(RuntimeError):
        ctx.delete_labelset("ls_used")
