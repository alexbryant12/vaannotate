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
