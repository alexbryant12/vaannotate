from __future__ import annotations

import pytest

try:
    from PySide6 import QtWidgets
except ImportError:  # pragma: no cover - handled via skip
    pytest.skip("PySide6 is not available", allow_module_level=True)

from vaannotate.AdminApp.main import AgreementSample, IaaWidget, ProjectContext


@pytest.fixture(scope="module")
def qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_prepare_agreement_samples_detects_discord(qt_app: QtWidgets.QApplication) -> None:
    widget = IaaWidget(ProjectContext())
    widget.round_manifest = {
        "unit-1": {"r1": True, "r2": True},
        "unit-2": {"r1": True, "r2": True},
        "unit-3": {"r1": False, "r2": False},
    }
    values_by_unit = {
        "unit-1": {"r1": "Yes", "r2": "No"},
        "unit-2": {"r1": "No", "r2": "No"},
        "unit-3": {"r1": "Yes"},
    }

    samples, discordant, reviewers = widget._prepare_agreement_samples(values_by_unit)

    assert {sample.unit_id for sample in samples} == {"unit-1", "unit-2"}
    assert discordant == {"unit-1"}
    assert reviewers == ["r1", "r2"]

    # Samples preserve reviewer ordering for downstream metrics
    assert all(isinstance(sample, AgreementSample) for sample in samples)
    assert all(sample.reviewer_ids == ("r1", "r2") for sample in samples)

    widget.deleteLater()

