"""Annotator client implemented with PySide6."""
from __future__ import annotations

import json
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..shared import models
from ..shared.database import Database, ensure_schema


@dataclass
class LabelDefinition:
    label_id: str
    name: str
    type: str
    required: bool
    na_allowed: bool
    rules: str
    unit: Optional[str]
    value_range: Optional[Dict[str, float]]
    gating_expr: Optional[str]
    options: List[Dict[str, object]]


class AssignmentContext(QtCore.QObject):
    assignment_loaded = QtCore.Signal()
    unit_changed = QtCore.Signal(dict)
    save_state_changed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.assignment_db: Optional[Database] = None
        self.assignment_path: Optional[Path] = None
        self.units: List[Dict[str, object]] = []
        self.labels: List[LabelDefinition] = []
        self.current_unit: Optional[Dict[str, object]] = None
        self.current_unit_id: Optional[str] = None
        self.current_annotations: Dict[str, Dict[str, object]] = {}

    def open_assignment(self, directory: Path) -> None:
        directory = directory.resolve()
        db_path = directory / "assignment.db"
        if not db_path.exists():
            raise FileNotFoundError(db_path)
        self.assignment_path = directory
        self.assignment_db = Database(db_path)
        self.current_unit_id = None
        self.current_annotations = {}
        with self.assignment_db.connect() as conn:
            ensure_schema(
                conn,
                [
                    models.AssignmentUnit,
                    models.AssignmentUnitNote,
                    models.AssignmentDocument,
                    models.Annotation,
                    models.Rationale,
                    models.Event,
                ],
            )
            self.units = [dict(row) for row in conn.execute("SELECT * FROM units ORDER BY display_rank").fetchall()]
        schema_path = directory / "label_schema.json"
        if schema_path.exists():
            raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self.labels = [
                LabelDefinition(
                    label_id=label["label_id"],
                    name=label["name"],
                    type=label["type"],
                    required=label.get("required", False),
                    na_allowed=label.get("na_allowed", False),
                    rules=label.get("rules", ""),
                    unit=label.get("unit"),
                    value_range=label.get("range"),
                    gating_expr=label.get("gating_expr"),
                    options=label.get("options", []),
                )
                for label in raw_schema.get("labels", [])
            ]
        else:
            self.labels = []
        self.assignment_loaded.emit()
        if self.units:
            self.set_current_unit(self.units[0])

    def set_current_unit(self, unit: Dict[str, object]) -> None:
        self.current_unit = unit
        unit_id = str(unit.get("unit_id", "")) if unit else None
        self.current_unit_id = unit_id or None
        self.current_annotations = (
            self.load_annotations(self.current_unit_id) if self.current_unit_id else {}
        )
        self.unit_changed.emit(unit)

    # Database helpers -----------------------------------------------------

    def fetch_document(self, doc_id: str) -> str:
        if not self.assignment_db:
            return ""
        with self.assignment_db.connect() as conn:
            row = conn.execute("SELECT text FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return row[0] if row else ""

    def fetch_unit_documents(self, unit_id: str) -> List[Dict[str, object]]:
        if not self.assignment_db:
            return []
        with self.assignment_db.connect() as conn:
            rows = conn.execute(
                "SELECT unit_notes.order_index, documents.doc_id, documents.text FROM unit_notes "
                "JOIN documents ON documents.doc_id = unit_notes.doc_id "
                "WHERE unit_notes.unit_id=? ORDER BY unit_notes.order_index",
                (unit_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_annotations(self, unit_id: str) -> Dict[str, Dict[str, object]]:
        if not self.assignment_db:
            return {}
        with self.assignment_db.connect() as conn:
            rows = conn.execute("SELECT * FROM annotations WHERE unit_id=?", (unit_id,)).fetchall()
        annotations: Dict[str, Dict[str, object]] = {}
        for row in rows:
            annotations[row["label_id"]] = dict(row)
        return annotations

    def save_annotation(self, unit_id: str, label_id: str, payload: Dict[str, object]) -> None:
        if not self.assignment_db:
            return
        def _normalized_state(data: Dict[str, object]) -> Dict[str, object]:
            """Prepare a dictionary that mirrors the annotation schema."""

            state: Dict[str, object] = {}
            if "value" in data:
                state["value"] = data.get("value")
            if "value_num" in data:
                state["value_num"] = data.get("value_num")
            if "value_date" in data:
                state["value_date"] = data.get("value_date")
            if "na" in data:
                state["na"] = bool(data.get("na"))
            if "notes" in data:
                notes = data.get("notes")
                state["notes"] = notes if notes is None else str(notes)
            return state

        with self.assignment_db.transaction() as conn:
            row = conn.execute(
                "SELECT value, value_num, value_date, na, notes FROM annotations WHERE unit_id=? AND label_id=?",
                (unit_id, label_id),
            ).fetchone()
            base_state: Dict[str, object] = {
                "value": row["value"] if row else None,
                "value_num": row["value_num"] if row else None,
                "value_date": row["value_date"] if row else None,
                "na": bool(row["na"]) if row else False,
                "notes": row["notes"] if row else None,
            }
            base_state.update(_normalized_state(payload))
            record = models.Annotation(
                unit_id=unit_id,
                label_id=label_id,
                value=base_state.get("value"),
                value_num=base_state.get("value_num"),
                value_date=base_state.get("value_date"),
                na=1 if base_state.get("na") else 0,
                notes=base_state.get("notes"),
            )
            record.save(conn)
            event = models.Event(
                event_id=str(uuid.uuid4()),
                ts=QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
                actor="annotator",
                event_type="annotation_saved",
                payload_json=json.dumps(
                    {
                        "unit_id": unit_id,
                        "label_id": label_id,
                        "payload": {
                            "value": record.value,
                            "value_num": record.value_num,
                            "value_date": record.value_date,
                            "na": bool(record.na),
                            "notes": record.notes,
                        },
                    }
                ),
            )
            event.save(conn)
        if self.current_unit_id == unit_id:
            self.current_annotations[label_id] = {
                "unit_id": unit_id,
                "label_id": label_id,
                "value": record.value,
                "value_num": record.value_num,
                "value_date": record.value_date,
                "na": record.na,
                "notes": record.notes,
            }
        self.save_state_changed.emit("Saved")

    def save_rationale(self, unit_id: str, label_id: str, doc_id: str, start: int, end: int, snippet: str) -> None:
        if not self.assignment_db:
            return
        record = models.Rationale(
            rationale_id=str(uuid.uuid4()),
            unit_id=unit_id,
            label_id=label_id,
            doc_id=doc_id,
            start_offset=start,
            end_offset=end,
            snippet=snippet,
            created_at=QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
        )
        with self.assignment_db.transaction() as conn:
            record.save(conn)
            event = models.Event(
                event_id=str(uuid.uuid4()),
                ts=record.created_at,
                actor="annotator",
                event_type="rationale_added",
                payload_json=json.dumps({
                    "unit_id": unit_id,
                    "label_id": label_id,
                    "doc_id": doc_id,
                    "start": start,
                    "end": end,
                }),
            )
            event.save(conn)
        self.save_state_changed.emit("Rationale saved")

    def mark_unit_complete(self, unit_id: str, complete: bool) -> None:
        if not self.assignment_db:
            return
        with self.assignment_db.transaction() as conn:
            conn.execute(
                "UPDATE units SET complete=?, completed_at=CASE WHEN ?=1 THEN ? ELSE completed_at END WHERE unit_id=?",
                (
                    1 if complete else 0,
                    1 if complete else 0,
                    QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
                    unit_id,
                ),
            )
        self.save_state_changed.emit("Progress updated")


class AnnotationForm(QtWidgets.QScrollArea):
    def __init__(
        self,
        ctx: AssignmentContext,
        get_selection: Callable[[], QtGui.QTextCursor],
        get_active_doc_id: Callable[[], Optional[str]],
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.get_selection = get_selection
        self.get_active_doc_id = get_active_doc_id
        self.container = QtWidgets.QWidget()
        self.setWidgetResizable(True)
        self.setWidget(self.container)
        self.layout = QtWidgets.QFormLayout(self.container)
        self.label_widgets: Dict[str, Dict[str, object]] = {}
        self.current_unit_id: Optional[str] = None
        self.current_annotations: Dict[str, Dict[str, object]] = {}

    def clear(self) -> None:
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.label_widgets.clear()

    def set_schema(self, labels: List[LabelDefinition]) -> None:
        self.clear()
        for label in labels:
            row_widget = self._create_row(label)
            label_widget = QtWidgets.QLabel(label.name)
            label_widget.setProperty("label_id", label.label_id)
            row_widget.setProperty("label_id", label.label_id)
            self.layout.addRow(label_widget, row_widget)

    def load_unit(self, unit_id: str, annotations: Dict[str, Dict[str, object]]) -> None:
        self.current_unit_id = unit_id
        self.current_annotations = annotations
        with self._suspend_widget_signals():
            for label_id, widgets in self.label_widgets.items():
                self._apply_annotation(label_id, widgets, annotations.get(label_id, {}))
        self._update_gating()
        self._update_completion()

    # internal helpers -----------------------------------------------------

    def _create_row(self, label: LabelDefinition) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        v_layout = QtWidgets.QVBoxLayout(wrapper)
        value_widget: QtWidgets.QWidget
        state: Dict[str, object] = {"definition": label}
        if label.type in {"boolean", "categorical_single", "ordinal"}:
            button_group = QtWidgets.QButtonGroup(wrapper)
            button_group.setExclusive(True)
            value_layout = QtWidgets.QHBoxLayout()
            for option in label.options:
                btn = QtWidgets.QRadioButton(option["display"])
                btn.setProperty("option_value", option["value"])
                button_group.addButton(btn)
                value_layout.addWidget(btn)
                btn.toggled.connect(lambda checked, opt=option, lid=label.label_id: self._on_radio(checked, lid, opt))
            value_widget = QtWidgets.QWidget()
            value_widget.setLayout(value_layout)
            state["button_group"] = button_group
        elif label.type == "categorical_multi":
            value_layout = QtWidgets.QVBoxLayout()
            checkboxes = []
            for option in label.options:
                cb = QtWidgets.QCheckBox(option["display"])
                cb.setProperty("option_value", option["value"])
                cb.stateChanged.connect(lambda _state, lid=label.label_id: self._on_multi_changed(lid))
                value_layout.addWidget(cb)
                checkboxes.append(cb)
            value_widget = QtWidgets.QWidget()
            value_widget.setLayout(value_layout)
            state["checkboxes"] = checkboxes
        elif label.type in {"integer", "float"}:
            line = QtWidgets.QLineEdit()
            validator = QtGui.QIntValidator() if label.type == "integer" else QtGui.QDoubleValidator()
            line.setValidator(validator)
            line.editingFinished.connect(lambda lid=label.label_id, widget=line: self._on_numeric(lid, widget))
            value_widget = line
            state["line_edit"] = line
        elif label.type == "date":
            date = QtWidgets.QDateEdit()
            date.setCalendarPopup(True)
            date.dateChanged.connect(lambda _date, lid=label.label_id, widget=date: self._on_date(lid, widget))
            value_widget = date
            state["date_edit"] = date
        else:
            text = QtWidgets.QTextEdit()
            text.textChanged.connect(lambda lid=label.label_id, widget=text: self._on_text(lid, widget))
            value_widget = text
            state["text_edit"] = text
        v_layout.addWidget(value_widget)

        info_layout = QtWidgets.QHBoxLayout()
        if label.na_allowed:
            na_box = QtWidgets.QCheckBox("N/A")
            na_box.stateChanged.connect(lambda _state, lid=label.label_id, widget=na_box: self._on_na(lid, widget))
            info_layout.addWidget(na_box)
            state["na_box"] = na_box
        notes = QtWidgets.QLineEdit()
        notes.setPlaceholderText("Notes")
        notes.editingFinished.connect(lambda lid=label.label_id, widget=notes: self._on_notes(lid, widget))
        info_layout.addWidget(notes)
        state["notes"] = notes
        highlight_btn = QtWidgets.QPushButton("Add highlight")
        highlight_btn.clicked.connect(lambda _checked, lid=label.label_id: self._add_highlight(lid))
        info_layout.addWidget(highlight_btn)
        v_layout.addLayout(info_layout)

        if label.rules:
            rules_label = QtWidgets.QLabel(label.rules)
            rules_label.setWordWrap(True)
            v_layout.addWidget(rules_label)
        wrapper.setProperty("label_id", label.label_id)
        self.label_widgets[label.label_id] = state
        return wrapper

    def _apply_annotation(self, label_id: str, widgets: Dict[str, object], annotation: Dict[str, object]) -> None:
        if not annotation:
            # reset controls
            self._reset_widgets(widgets)
            return
        if "button_group" in widgets:
            value = annotation.get("value")
            group: QtWidgets.QButtonGroup = widgets["button_group"]  # type: ignore[assignment]
            for button in group.buttons():
                button.setChecked(button.property("option_value") == value)
        if "checkboxes" in widgets:
            values = set((annotation.get("value") or "").split(","))
            for cb in widgets["checkboxes"]:  # type: ignore[index]
                cb.setChecked(cb.property("option_value") in values)
        if "line_edit" in widgets:
            widgets["line_edit"].setText(annotation.get("value") or "")  # type: ignore[index]
        if "date_edit" in widgets:
            date_widget: QtWidgets.QDateEdit = widgets["date_edit"]  # type: ignore[assignment]
            if annotation.get("value_date"):
                date_widget.setDate(QtCore.QDate.fromString(annotation["value_date"], QtCore.Qt.ISODate))
        if "text_edit" in widgets:
            widgets["text_edit"].setPlainText(annotation.get("value") or "")  # type: ignore[index]
        if "na_box" in widgets:
            widgets["na_box"].setChecked(bool(annotation.get("na")))  # type: ignore[index]
        if "notes" in widgets:
            widgets["notes"].setText(annotation.get("notes") or "")  # type: ignore[index]

    def _reset_widgets(self, widgets: Dict[str, object]) -> None:
        if "button_group" in widgets:
            group: QtWidgets.QButtonGroup = widgets["button_group"]  # type: ignore[assignment]
            was_exclusive = group.exclusive()
            group.setExclusive(False)
            for button in group.buttons():
                if hasattr(button, "autoExclusive"):
                    was_auto = button.autoExclusive()
                    button.setAutoExclusive(False)
                    button.setChecked(False)
                    button.setAutoExclusive(was_auto)
                else:
                    button.setChecked(False)
            group.setExclusive(was_exclusive)
        if "checkboxes" in widgets:
            for cb in widgets["checkboxes"]:  # type: ignore[index]
                cb.setChecked(False)
        if "line_edit" in widgets:
            widgets["line_edit"].clear()  # type: ignore[index]
        if "date_edit" in widgets:
            widgets["date_edit"].setDate(QtCore.QDate.currentDate())  # type: ignore[index]
        if "text_edit" in widgets:
            widgets["text_edit"].clear()  # type: ignore[index]
        if "na_box" in widgets:
            widgets["na_box"].setChecked(False)  # type: ignore[index]
        if "notes" in widgets:
            widgets["notes"].clear()  # type: ignore[index]

    @contextmanager
    def _suspend_widget_signals(self):
        widgets: List[QtCore.QObject] = []
        for state in self.label_widgets.values():
            group = state.get("button_group")
            if isinstance(group, QtWidgets.QButtonGroup):
                widgets.append(group)
                widgets.extend(group.buttons())
            checkbox_list = state.get("checkboxes")
            if isinstance(checkbox_list, list):
                widgets.extend(cb for cb in checkbox_list if isinstance(cb, QtWidgets.QCheckBox))
            line_edit = state.get("line_edit")
            if isinstance(line_edit, QtWidgets.QLineEdit):
                widgets.append(line_edit)
            date_edit = state.get("date_edit")
            if isinstance(date_edit, QtWidgets.QDateEdit):
                widgets.append(date_edit)
            text_edit = state.get("text_edit")
            if isinstance(text_edit, QtWidgets.QTextEdit):
                widgets.append(text_edit)
            na_box = state.get("na_box")
            if isinstance(na_box, QtWidgets.QCheckBox):
                widgets.append(na_box)
            notes = state.get("notes")
            if isinstance(notes, QtWidgets.QLineEdit):
                widgets.append(notes)
        try:
            for widget in widgets:
                widget.blockSignals(True)
            yield
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    # value change handlers ------------------------------------------------

    def _on_radio(self, checked: bool, label_id: str, option: Dict[str, object]) -> None:
        if not checked or not self.current_unit_id:
            return
        payload = {"value": option["value"]}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_multi_changed(self, label_id: str) -> None:
        if not self.current_unit_id:
            return
        widgets = self.label_widgets[label_id]
        selected = [cb.property("option_value") for cb in widgets["checkboxes"] if cb.isChecked()]  # type: ignore[index]
        payload = {"value": ",".join(selected)}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_numeric(self, label_id: str, widget: QtWidgets.QLineEdit) -> None:
        if not self.current_unit_id:
            return
        text = widget.text().strip()
        payload = {"value": text, "value_num": float(text) if text else None}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_date(self, label_id: str, widget: QtWidgets.QDateEdit) -> None:
        if not self.current_unit_id:
            return
        payload = {"value_date": widget.date().toString(QtCore.Qt.ISODate)}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_text(self, label_id: str, widget: QtWidgets.QTextEdit) -> None:
        if not self.current_unit_id:
            return
        payload = {"value": widget.toPlainText()}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_na(self, label_id: str, widget: QtWidgets.QCheckBox) -> None:
        if not self.current_unit_id:
            return
        payload = {"na": widget.isChecked()}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_completion()

    def _on_notes(self, label_id: str, widget: QtWidgets.QLineEdit) -> None:
        if not self.current_unit_id:
            return
        payload = {"notes": widget.text()}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)

    def _add_highlight(self, label_id: str) -> None:
        if not self.current_unit_id or not self.ctx.current_unit:
            return
        doc_id = self.get_active_doc_id()
        if not doc_id:
            QtWidgets.QMessageBox.information(self, "Highlight", "Select a note before adding highlights")
            return
        cursor = self.get_selection()
        if cursor.isNull() or cursor.selectionStart() == cursor.selectionEnd():
            QtWidgets.QMessageBox.information(self, "Highlight", "Select text in the note first")
            return
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        snippet = cursor.selectedText()
        self.ctx.save_rationale(self.current_unit_id, label_id, doc_id, start, end, snippet)

    def _update_gating(self) -> None:
        values = self._current_values()
        for i in range(self.layout.rowCount()):
            label_item = self.layout.itemAt(i, QtWidgets.QFormLayout.ItemRole.LabelRole)
            field_item = self.layout.itemAt(i, QtWidgets.QFormLayout.ItemRole.FieldRole)
            if not label_item or not field_item:
                continue
            field_widget = field_item.widget()
            label_widget = label_item.widget()
            if not field_widget or not label_widget:
                continue
            label_id = field_widget.property("label_id") or label_widget.property("label_id")
            if not label_id:
                continue
            definition: LabelDefinition = self.label_widgets[label_id]["definition"]  # type: ignore[index]
            visible = self._is_label_visible(definition, values)
            field_widget.setVisible(visible)
            label_widget.setVisible(visible)
        self._update_completion()

    def _current_values(self) -> Dict[str, object]:
        values: Dict[str, object] = {}
        for label_id, widgets in self.label_widgets.items():
            definition: LabelDefinition = widgets["definition"]  # type: ignore[index]
            if "button_group" in widgets:
                group: QtWidgets.QButtonGroup = widgets["button_group"]  # type: ignore[assignment]
                for button in group.buttons():
                    if button.isChecked():
                        values[definition.name] = button.property("option_value")
                        break
            if "checkboxes" in widgets:
                selected = [cb.property("option_value") for cb in widgets["checkboxes"] if cb.isChecked()]  # type: ignore[index]
                values[definition.name] = selected
            if "line_edit" in widgets:
                values[definition.name] = widgets["line_edit"].text()  # type: ignore[index]
            if "text_edit" in widgets:
                values[definition.name] = widgets["text_edit"].toPlainText()  # type: ignore[index]
        return values

    def _is_label_visible(self, definition: LabelDefinition, values: Dict[str, object]) -> bool:
        expr = definition.gating_expr
        if not expr:
            return True
        try:
            field, expected = expr.split("==")
            field = field.strip()
            expected = expected.strip().strip("'\"")
            return str(values.get(field, "")) == expected
        except ValueError:
            return True

    def _has_value(self, widgets: Dict[str, object]) -> bool:
        if "na_box" in widgets and widgets["na_box"].isChecked():  # type: ignore[index]
            return True
        if "button_group" in widgets:
            group: QtWidgets.QButtonGroup = widgets["button_group"]  # type: ignore[assignment]
            return any(button.isChecked() for button in group.buttons())
        if "checkboxes" in widgets:
            return any(cb.isChecked() for cb in widgets["checkboxes"])  # type: ignore[index]
        if "line_edit" in widgets:
            return bool(widgets["line_edit"].text().strip())  # type: ignore[index]
        if "date_edit" in widgets:
            return widgets["date_edit"].date().isValid()  # type: ignore[index]
        if "text_edit" in widgets:
            return bool(widgets["text_edit"].toPlainText().strip())  # type: ignore[index]
        return False

    def _update_completion(self) -> None:
        if not self.current_unit_id:
            return
        values = self._current_values()
        complete = True
        for label_id, widgets in self.label_widgets.items():
            definition: LabelDefinition = widgets["definition"]  # type: ignore[index]
            if not self._is_label_visible(definition, values):
                continue
            if definition.required and not self._has_value(widgets):
                complete = False
                break
        self.ctx.mark_unit_complete(self.current_unit_id, complete)


class ClientMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ctx = AssignmentContext()
        self.setWindowTitle("VAAnnotate Client")
        self.resize(1400, 900)
        self.current_documents: List[Dict[str, object]] = []
        self.active_doc_id: Optional[str] = None
        self._setup_menu()
        self._setup_ui()
        self.ctx.assignment_loaded.connect(self._on_assignment_loaded)
        self.ctx.unit_changed.connect(self._load_unit)
        self.ctx.save_state_changed.connect(self.statusBar().showMessage)

    def _setup_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        open_action = file_menu.addAction("Open assignment…")
        open_action.triggered.connect(self._open_assignment)
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def _setup_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.unit_list = QtWidgets.QListWidget()
        self.unit_list.itemSelectionChanged.connect(self._unit_selected)
        splitter.addWidget(self.unit_list)

        middle_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.notes_table = QtWidgets.QTableWidget()
        self.notes_table.setColumnCount(3)
        self.notes_table.setHorizontalHeaderLabels(["#", "Document ID", "Preview"])
        self.notes_table.verticalHeader().setVisible(False)
        self.notes_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.notes_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.notes_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.notes_table.itemSelectionChanged.connect(self._on_document_selected)
        self.notes_table.horizontalHeader().setStretchLastSection(True)
        middle_splitter.addWidget(self.notes_table)

        note_panel = QtWidgets.QWidget()
        note_panel_layout = QtWidgets.QVBoxLayout(note_panel)
        note_panel_layout.setContentsMargins(0, 0, 0, 0)
        note_panel_layout.setSpacing(6)
        self.note_view = QtWidgets.QTextEdit()
        self.note_view.setReadOnly(True)
        self.note_view.setFontPointSize(12)
        note_panel_layout.addWidget(self.note_view)

        info_widget = QtWidgets.QWidget()
        info_layout = QtWidgets.QHBoxLayout(info_widget)
        self.progress_label = QtWidgets.QLabel("Progress: 0/0")
        info_layout.addWidget(self.progress_label)
        note_panel_layout.addWidget(info_widget)
        middle_splitter.addWidget(note_panel)
        splitter.addWidget(middle_splitter)

        self.form = AnnotationForm(
            self.ctx,
            lambda: self.note_view.textCursor(),
            lambda: self.active_doc_id,
        )
        splitter.addWidget(self.form)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        self.setCentralWidget(splitter)

        nav_toolbar = self.addToolBar("Navigation")
        prev_action = QtGui.QAction("Previous", self)
        prev_action.triggered.connect(lambda: self._navigate(-1))
        next_action = QtGui.QAction("Next", self)
        next_action.triggered.connect(lambda: self._navigate(1))
        submit_action = QtGui.QAction("Submit", self)
        submit_action.triggered.connect(self._submit_assignment)
        nav_toolbar.addAction(prev_action)
        nav_toolbar.addAction(next_action)
        nav_toolbar.addAction(submit_action)

    def _open_assignment(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select assignment folder")
        if not directory:
            return
        self.ctx.open_assignment(Path(directory))

    def _format_unit_label(self, unit: Dict[str, object]) -> str:
        doc_id = str(unit.get("doc_id") or "").strip()
        note_count_raw = unit.get("note_count")
        try:
            note_count = int(note_count_raw) if note_count_raw is not None else None
        except (TypeError, ValueError):
            note_count = None
        patient = str(unit.get("patient_icn") or "").strip()
        if doc_id and (not note_count or note_count <= 1):
            return doc_id
        if note_count and note_count > 1:
            base = patient or str(unit.get("unit_id"))
            return f"{base} ({note_count} notes)"
        return patient or str(unit.get("unit_id"))

    def _on_assignment_loaded(self) -> None:
        self.unit_list.clear()
        self.notes_table.setRowCount(0)
        self.note_view.clear()
        self.current_documents = []
        self.active_doc_id = None
        for unit in self.ctx.units:
            label = self._format_unit_label(unit)
            item = QtWidgets.QListWidgetItem(f"{unit['display_rank']}: {label}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, unit)
            self.unit_list.addItem(item)
        self.form.set_schema(self.ctx.labels)
        if self.ctx.units:
            self.unit_list.setCurrentRow(0)
        self._update_progress()

    def _unit_selected(self) -> None:
        items = self.unit_list.selectedItems()
        if not items:
            return
        unit = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        self.ctx.set_current_unit(unit)

    def _load_unit(self, unit: Dict[str, object]) -> None:
        unit_id = str(unit["unit_id"])
        documents = self.ctx.fetch_unit_documents(unit_id)
        if documents:
            self._populate_notes_table(documents)
        else:
            fallback_doc_id = str(unit.get("doc_id", ""))
            fallback_text = self.ctx.fetch_document(fallback_doc_id)
            documents = [
                {
                    "order_index": 1,
                    "doc_id": fallback_doc_id,
                    "text": fallback_text,
                }
            ]
            self._populate_notes_table(documents)
        if self.notes_table.rowCount():
            self.notes_table.setCurrentCell(0, 0)
        annotations = self.ctx.load_annotations(str(unit["unit_id"]))
        self.form.load_unit(unit_id, annotations)
        self._update_progress()

    def _populate_notes_table(self, documents: List[Dict[str, object]]) -> None:
        self.current_documents = documents
        self.active_doc_id = None
        self.notes_table.blockSignals(True)
        self.notes_table.setRowCount(len(documents))
        for row_index, doc in enumerate(documents):
            order_item = QtWidgets.QTableWidgetItem(str(doc.get("order_index", row_index + 1)))
            order_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc)
            doc_item = QtWidgets.QTableWidgetItem(str(doc.get("doc_id", "")))
            doc_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc)
            preview_text = str(doc.get("text", ""))[:200].replace("\n", " ")
            preview_item = QtWidgets.QTableWidgetItem(preview_text)
            self.notes_table.setItem(row_index, 0, order_item)
            self.notes_table.setItem(row_index, 1, doc_item)
            self.notes_table.setItem(row_index, 2, preview_item)
        self.notes_table.blockSignals(False)
        self.notes_table.resizeColumnsToContents()

    def _on_document_selected(self) -> None:
        row_index = self.notes_table.currentRow()
        if row_index < 0:
            self.note_view.clear()
            self.active_doc_id = None
            return
        item = self.notes_table.item(row_index, 0)
        doc = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        if not doc:
            if 0 <= row_index < len(self.current_documents):
                doc = self.current_documents[row_index]
        if doc:
            self._set_active_document(doc)
        else:
            self.note_view.clear()
            self.active_doc_id = None

    def _set_active_document(self, doc: Dict[str, object]) -> None:
        self.active_doc_id = str(doc.get("doc_id", "")) or None
        self.note_view.setPlainText(str(doc.get("text", "")))

    def _navigate(self, step: int) -> None:
        row = self.unit_list.currentRow()
        target = row + step
        if 0 <= target < self.unit_list.count():
            self.unit_list.setCurrentRow(target)

    def _submit_assignment(self) -> None:
        if not self.ctx.assignment_path:
            return
        receipt = {
            "unit_count": len(self.ctx.units),
            "completed": sum(1 for unit in self.ctx.units if unit.get("complete")),
            "submitted_at": QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
        }
        (self.ctx.assignment_path / "submitted.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        QtWidgets.QMessageBox.information(self, "Submission", "Assignment marked as submitted.")

    def _update_progress(self) -> None:
        total = len(self.ctx.units)
        completed = 0
        status_map: Dict[str, int] = {}
        if self.ctx.assignment_db:
            with self.ctx.assignment_db.connect() as conn:
                rows = conn.execute("SELECT unit_id, complete FROM units").fetchall()
            status_map = {row["unit_id"]: row["complete"] for row in rows}
            completed = sum(1 for row in rows if row["complete"])
        for idx in range(self.unit_list.count()):
            item = self.unit_list.item(idx)
            unit = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not unit:
                continue
            unit_id = str(unit["unit_id"])
            if unit_id in status_map:
                unit["complete"] = status_map[unit_id]
            suffix = " ✓" if unit.get("complete") else ""
            item.setText(f"{unit['display_rank']}: {self._format_unit_label(unit)}{suffix}")
        self.progress_label.setText(f"Progress: {completed}/{total}")


def run(path: Optional[str] = None) -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = ClientMainWindow()
    window.show()
    if path:
        window.ctx.open_assignment(Path(path))
    sys.exit(app.exec())


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
