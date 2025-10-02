from __future__ import annotations

import pytest

try:
    from PySide6 import QtGui, QtWidgets
except ImportError:  # pragma: no cover - handled via skip
    pytest.skip("PySide6 is not available", allow_module_level=True)

from vaannotate.ClientApp.main import AnnotationForm, AssignmentContext, LabelDefinition


@pytest.fixture(scope="module")
def qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _dummy_cursor() -> QtGui.QTextCursor:
    document = QtGui.QTextDocument()
    return QtGui.QTextCursor(document)


def test_boolean_selection_clears_between_units(qt_app: QtWidgets.QApplication) -> None:
    ctx = AssignmentContext()
    form = AnnotationForm(ctx, _dummy_cursor, lambda: None)

    label = LabelDefinition(
        label_id="has_pheno",
        name="Has phenotype",
        type="boolean",
        required=False,
        na_allowed=False,
        rules="",
        unit=None,
        value_range=None,
        gating_expr=None,
        options=[
            {"value": "yes", "display": "Yes"},
            {"value": "no", "display": "No"},
        ],
    )

    form.set_schema([label])
    form.load_unit("unit-1", {label.label_id: {"value": "yes"}})
    widgets = form.label_widgets[label.label_id]
    group: QtWidgets.QButtonGroup = widgets["button_group"]  # type: ignore[index]
    checked_buttons = [button for button in group.buttons() if button.isChecked()]
    assert len(checked_buttons) == 1
    assert checked_buttons[0].property("option_value") == "yes"

    form.load_unit("unit-2", {})
    assert all(not button.isChecked() for button in group.buttons())

