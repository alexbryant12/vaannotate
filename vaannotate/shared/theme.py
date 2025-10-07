"""Shared helpers for applying application-wide themes."""
from __future__ import annotations

from PySide6 import QtGui, QtWidgets


def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """Configure the application with a high-contrast dark palette."""

    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(42, 42, 42))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(60, 60, 60))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor(185, 185, 185))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 85, 85))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(100, 100, 150))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(130, 160, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor(110, 110, 110))
    palette.setColor(QtGui.QPalette.ColorRole.Dark, QtGui.QColor(80, 80, 80))
    palette.setColor(QtGui.QPalette.ColorRole.Shadow, QtGui.QColor(70, 70, 70))
    app.setPalette(palette)

    existing_stylesheet = app.styleSheet() or ""
    snippets = [existing_stylesheet.strip()] if existing_stylesheet.strip() else []
    snippets.extend(
        [
            "QTableView, QTreeView {",
            "    gridline-color: rgb(110, 110, 110);",
            "}",
            "QToolTip { color: #f0f0f0; background-color: #353535; border: 1px solid #767676; }",
        ]
    )
    app.setStyleSheet("\n".join(snippets))
