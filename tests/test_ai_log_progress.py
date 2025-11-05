from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "PySide6.QtWidgets",
    reason="PySide6 QtWidgets bindings unavailable (missing libGL)",
    exc_type=ImportError,
)
from PySide6 import QtWidgets

from vaannotate.AdminApp.main import RoundBuilderDialog


@pytest.fixture(scope="module")
def qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _LogCollector:
    _append_ai_log = RoundBuilderDialog._append_ai_log
    _write_ai_log_line = RoundBuilderDialog._write_ai_log_line
    _replace_last_ai_log_line = RoundBuilderDialog._replace_last_ai_log_line

    def __init__(self) -> None:
        self.ai_log_output = QtWidgets.QTextEdit()
        self._ai_progress_active = False
        self._ai_progress_stamp = ""
        self._ai_progress_text = ""
        self._ai_progress_block_number = None


def _datetime_generator(*stamps: str):
    iterator = iter(stamps)

    def _utcnow() -> SimpleNamespace:
        stamp = next(iterator)

        class _DateTime:
            def __init__(self, value: str) -> None:
                self._value = value

            def strftime(self, fmt: str) -> str:
                return self._value

        return _DateTime(stamp)

    return SimpleNamespace(utcnow=_utcnow)


def test_progress_updates_preserve_prior_logs(monkeypatch, qt_app):
    collector = _LogCollector()

    fake_datetime = _datetime_generator("12:00:00", "12:00:01", "12:00:02")
    monkeypatch.setattr("vaannotate.AdminApp.main.datetime", fake_datetime)

    collector._append_ai_log("Initial message")
    collector._append_ai_log("\rProgress update 1/3")
    collector._append_ai_log("\rProgress update 2/3")

    lines = collector.ai_log_output.toPlainText().splitlines()

    assert len(lines) == 2
    assert lines[0].endswith("Initial message")
    assert lines[1].endswith("Progress update 2/3")


def test_progress_followed_by_new_entries(monkeypatch, qt_app):
    collector = _LogCollector()

    fake_datetime = _datetime_generator(
        "12:10:00",
        "12:10:01",
        "12:10:02",
        "12:10:03",
        "12:10:04",
    )
    monkeypatch.setattr("vaannotate.AdminApp.main.datetime", fake_datetime)

    collector._append_ai_log("First message")
    collector._append_ai_log("\rProgress 1/2")
    collector._append_ai_log("\rProgress 2/2")
    collector._append_ai_log("Second message")
    collector._append_ai_log("\rFinal step")

    lines = collector.ai_log_output.toPlainText().splitlines()

    assert len(lines) == 4
    assert lines[0].endswith("First message")
    assert lines[1].endswith("Progress 2/2")
    assert lines[2].endswith("Second message")
    assert lines[3].endswith("Final step")
