from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip(
    "PySide6.QtWidgets",
    reason="PySide6 QtWidgets bindings unavailable (missing libGL)",
    exc_type=ImportError,
)
from PySide6 import QtWidgets

from vaannotate.AdminApp.main import LabelSetWizardDialog


@pytest.fixture(scope="module")
def qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@dataclass
class _FakeProjectContext:
    project_row: dict[str, object] | None = None

    def list_label_sets(self) -> list[dict[str, object]]:
        return [
            {
                "labelset_id": "baseline",
                "created_at": "2026-04-23",
                "notes": "x" * 1000,
            }
        ]


def test_copy_combo_is_width_capped(qt_app: QtWidgets.QApplication) -> None:
    dialog = LabelSetWizardDialog(_FakeProjectContext())
    assert (
        dialog.copy_combo.sizeAdjustPolicy()
        == QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert dialog.copy_combo.minimumContentsLength() == 30


def test_resources_tabs_use_scroll_areas(qt_app: QtWidgets.QApplication) -> None:
    dialog = LabelSetWizardDialog(_FakeProjectContext())
    dialog.labels = [
        {"label_id": "very_long_identifier", "name": "Very long display name", "type": "text"},
    ]
    dialog._refresh_label_resources()

    keyword_scrolls = dialog.keywords_tab.findChildren(QtWidgets.QScrollArea)
    fewshot_scrolls = dialog.few_shot_tab.findChildren(QtWidgets.QScrollArea)

    assert keyword_scrolls
    assert fewshot_scrolls
    assert keyword_scrolls[0].widgetResizable()
    assert fewshot_scrolls[0].widgetResizable()
