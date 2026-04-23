from __future__ import annotations

import pytest

pytest.importorskip(
    "PySide6.QtWidgets",
    reason="PySide6 QtWidgets bindings unavailable (missing libGL)",
    exc_type=ImportError,
)
from PySide6 import QtCore, QtWidgets

from vaannotate.AdminApp.main import LabelSetWizardDialog, _truncate_for_display


@pytest.fixture(scope="module")
def qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _DialogContext:
    project_row = {"created_by": "admin"}

    def list_label_sets(self):
        return [
            {
                "labelset_id": "set_a",
                "created_at": "2026-04-20",
                "notes": "This is an intentionally very long note " * 10,
            }
        ]

    def load_labelset_details(self, labelset_id: str):  # pragma: no cover - not exercised here
        return {"labelset_id": labelset_id, "labels": []}


def test_truncate_for_display_clamps_length():
    result = _truncate_for_display("word " * 100, max_length=32)
    assert len(result) <= 32
    assert result.endswith("…")


def test_copy_combo_uses_safe_sizing_policy(qt_app):
    dialog = LabelSetWizardDialog(_DialogContext())
    assert (
        dialog.copy_combo.sizeAdjustPolicy()
        == QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert dialog.copy_combo.minimumContentsLength() == 36
    assert dialog.copy_combo.view().textElideMode() == QtCore.Qt.TextElideMode.ElideRight

    item_text = dialog.copy_combo.itemText(1)
    assert len(item_text) < 220
    assert "…" in item_text
