"""Annotator client implemented with PySide6."""
from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Set, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from vaannotate.shared import models
from vaannotate.shared.theme import apply_dark_palette
from vaannotate.shared.database import Database, ensure_schema
from vaannotate.project import get_connection


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


@dataclass
class ReviewerInfo:
    reviewer_id: str
    name: str
    email: Optional[str]

    def display_label(self) -> str:
        display_name = (self.name or "").strip()
        if display_name and display_name != self.reviewer_id:
            return f"{display_name} ({self.reviewer_id})"
        return self.reviewer_id


@dataclass
class AssignmentSummary:
    round_id: str
    phenotype_id: str
    phenotype_name: str
    round_number: Optional[int]
    assignment_dir: Path
    submitted: bool

    def round_label(self) -> str:
        if self.round_number is None:
            return self.round_id
        return f"Round {self.round_number}"


class ProjectBrowser:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.project_db = self.project_root / "project.db"
        if not self.project_db.exists():
            raise FileNotFoundError(self.project_db)

    def list_reviewers(self) -> List[ReviewerInfo]:
        with get_connection(self.project_db) as conn:
            rows = conn.execute(
                "SELECT reviewer_id, name, email FROM reviewers ORDER BY name COLLATE NOCASE, reviewer_id"
            ).fetchall()
        return [
            ReviewerInfo(
                reviewer_id=str(row["reviewer_id"]),
                name=str(row["name"] or ""),
                email=str(row["email"]) if row["email"] is not None else None,
            )
            for row in rows
        ]

    def list_assignments(self, reviewer_id: str) -> Tuple[List[AssignmentSummary], List[str]]:
        assignments: List[AssignmentSummary] = []
        warnings: List[str] = []
        with get_connection(self.project_db) as conn:
            rows = conn.execute(
                """
                SELECT a.round_id, r.pheno_id, r.round_number, p.name AS phenotype_name, p.storage_path
                FROM assignments AS a
                JOIN rounds AS r ON a.round_id = r.round_id
                JOIN phenotypes AS p ON r.pheno_id = p.pheno_id
                WHERE a.reviewer_id = ?
                ORDER BY p.name COLLATE NOCASE, r.round_number
                """,
                (reviewer_id,),
            ).fetchall()
        for row in rows:
            pheno_id = str(row["pheno_id"])
            round_id = str(row["round_id"])
            round_number_raw = row["round_number"]
            round_number: Optional[int]
            try:
                round_number = int(round_number_raw)
            except (TypeError, ValueError):
                round_number = None
            round_dir = self._resolve_round_dir(
                pheno_id,
                round_number,
                round_id,
                str(row["storage_path"] or ""),
            )
            if not round_dir:
                warnings.append(
                    f"Round directory not found for {pheno_id} ({round_id})."
                )
                continue
            assignment_dir = round_dir / "assignments" / reviewer_id
            assignment_db = assignment_dir / "assignment.db"
            if not assignment_db.exists():
                warnings.append(
                    f"Assignment database missing for {pheno_id} {round_dir.name}."
                )
                continue
            submitted = (assignment_dir / "submitted.json").exists()
            assignments.append(
                AssignmentSummary(
                    round_id=round_id,
                    phenotype_id=pheno_id,
                    phenotype_name=str(row["phenotype_name"] or pheno_id),
                    round_number=round_number,
                    assignment_dir=assignment_dir,
                    submitted=submitted,
                )
            )
        return assignments, warnings

    def _resolve_round_dir(
        self,
        pheno_id: str,
        round_number: Optional[int],
        round_id: str,
        storage_path: str,
    ) -> Optional[Path]:
        rounds_root = self._resolve_rounds_root(pheno_id, storage_path)
        if rounds_root is None:
            return None
        if round_number is not None:
            default_dir = rounds_root / f"round_{round_number}"
            if default_dir.exists():
                return default_dir
        if not rounds_root.exists():
            return None
        for candidate in sorted(rounds_root.iterdir()):
            if not candidate.is_dir():
                continue
            config_path = candidate / "round_config.json"
            if not config_path.exists():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            config_round_id = str(config.get("round_id") or "")
            if config_round_id == round_id:
                return candidate
        return None

    def _resolve_rounds_root(self, pheno_id: str, storage_path: str) -> Optional[Path]:
        rounds_root: Optional[Path]
        if storage_path:
            storage = Path(storage_path)
            if not storage.is_absolute():
                storage = (self.project_root / storage).resolve()
            rounds_root = storage / "rounds"
        else:
            rounds_root = None
        # Fallback for legacy directory structures where the phenotype ID was used
        if rounds_root is None or not rounds_root.exists():
            candidate = self.project_root / "phenotypes" / pheno_id / "rounds"
            rounds_root = candidate if candidate.exists() else rounds_root
        return rounds_root


class AssignmentPickerDialog(QtWidgets.QDialog):
    def __init__(self, browser: ProjectBrowser, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.browser = browser
        self.setWindowTitle("Open project assignment")
        self.resize(500, 400)
        self.reviewers: List[ReviewerInfo] = self.browser.list_reviewers()
        self._current_assignments: List[AssignmentSummary] = []
        self._current_warnings: List[str] = []

        layout = QtWidgets.QVBoxLayout(self)

        reviewer_row = QtWidgets.QHBoxLayout()
        reviewer_label = QtWidgets.QLabel("Reviewer:")
        reviewer_row.addWidget(reviewer_label)
        self.reviewer_combo = QtWidgets.QComboBox()
        for reviewer in self.reviewers:
            self.reviewer_combo.addItem(reviewer.display_label(), reviewer.reviewer_id)
        reviewer_row.addWidget(self.reviewer_combo, 1)
        layout.addLayout(reviewer_row)

        self.assignment_tree = QtWidgets.QTreeWidget()
        self.assignment_tree.setColumnCount(3)
        self.assignment_tree.setHeaderLabels(["Phenotype", "Round", "Submission"])
        self.assignment_tree.setRootIsDecorated(False)
        self.assignment_tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.assignment_tree.itemSelectionChanged.connect(self._update_button_state)
        self.assignment_tree.itemActivated.connect(lambda _item, _column: self.accept())
        layout.addWidget(self.assignment_tree, 1)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        if self.reviewers:
            self.reviewer_combo.currentIndexChanged.connect(self._refresh_assignments)
            self._refresh_assignments()
        else:
            self.reviewer_combo.setEnabled(False)
            ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
            if ok_button:
                ok_button.setEnabled(False)
            self.status_label.setText("No reviewers found in this project.")

    def selected_assignment(self) -> Optional[AssignmentSummary]:
        item = self.assignment_tree.currentItem()
        if not item:
            return None
        assignment = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return assignment if isinstance(assignment, AssignmentSummary) else None

    def selected_reviewer(self) -> Optional[ReviewerInfo]:
        index = self.reviewer_combo.currentIndex()
        if index < 0 or index >= len(self.reviewers):
            return None
        return self.reviewers[index]

    def _refresh_assignments(self) -> None:
        reviewer_id = self.reviewer_combo.currentData()
        self.assignment_tree.clear()
        self._current_assignments = []
        self._current_warnings = []
        if not reviewer_id:
            self._update_button_state()
            return
        assignments, warnings = self.browser.list_assignments(str(reviewer_id))
        self._current_assignments = assignments
        self._current_warnings = warnings
        for assignment in assignments:
            submission_text = "Submitted" if assignment.submitted else "In progress"
            item = QtWidgets.QTreeWidgetItem(
                [assignment.phenotype_name, assignment.round_label(), submission_text]
            )
            if assignment.submitted:
                item.setIcon(
                    0,
                    self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton),
                )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, assignment)
            self.assignment_tree.addTopLevelItem(item)
        if assignments:
            self.assignment_tree.setCurrentItem(self.assignment_tree.topLevelItem(0))
        self._update_status_label()
        self._update_button_state()

    def _update_status_label(self) -> None:
        if self._current_assignments:
            warning_text = "\n".join(self._current_warnings)
            self.status_label.setText(warning_text)
        else:
            if self._current_warnings:
                message = "\n".join(self._current_warnings)
            else:
                message = "No assignments found for the selected reviewer."
            self.status_label.setText(message)

    def _update_button_state(self) -> None:
        ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_button:
            ok_button.setEnabled(bool(self.assignment_tree.currentItem()))

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
        self.current_rationales: Dict[str, List[Dict[str, object]]] = {}
        self._annotation_cache: Dict[str, Dict[str, Dict[str, object]]] = {}
        self._annotation_baseline: Dict[str, Dict[str, Dict[str, object]]] = {}
        self._rationale_cache: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        self._rationale_baseline: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        self._unit_completion_baseline: Dict[str, Dict[str, object]] = {}
        self._pending_annotations: Dict[Tuple[str, str], models.Annotation] = {}
        self._pending_annotation_events: Dict[Tuple[str, str], models.Event] = {}
        self._pending_rationale_inserts: Dict[str, models.Rationale] = {}
        self._pending_rationale_updates: Dict[str, models.Rationale] = {}
        self._pending_rationale_deletes: Set[str] = set()
        self._pending_events: List[models.Event] = []
        self._pending_completion: Dict[str, Dict[str, object]] = {}
        self._dirty: bool = False
        self._document_cache: Dict[str, Dict[str, object]] = {}
        self._unit_document_cache: Dict[str, List[Dict[str, object]]] = {}
        self.assisted_review_enabled: bool = False
        self.assisted_review_top_n: int = 0
        self._assisted_review_snippets: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        self.final_llm_enabled: bool = False
        self._final_llm_reasoning: bool = True
        self._final_llm_labels: Dict[str, Dict[str, Dict[str, object]]] = {}

    @staticmethod
    def _parse_bool(value: object, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "f", "no", "n", "off"}:
                return False
        return default

    def open_assignment(self, directory: Path) -> None:
        directory = directory.resolve()
        db_path = directory / "assignment.db"
        if not db_path.exists():
            raise FileNotFoundError(db_path)
        self.assignment_path = directory
        self.assignment_db = Database(db_path)
        self.current_unit_id = None
        self.current_annotations = {}
        self.current_rationales = {}
        self._annotation_cache = {}
        self._annotation_baseline = {}
        self._rationale_cache = {}
        self._rationale_baseline = {}
        self._unit_completion_baseline = {}
        self._pending_annotations.clear()
        self._pending_annotation_events.clear()
        self._pending_rationale_inserts.clear()
        self._pending_rationale_updates.clear()
        self._pending_rationale_deletes.clear()
        self._pending_events.clear()
        self._pending_completion.clear()
        self._document_cache = {}
        self._unit_document_cache = {}
        self.assisted_review_enabled = False
        self.assisted_review_top_n = 0
        self._assisted_review_snippets = {}
        self.final_llm_enabled = False
        self._final_llm_reasoning = True
        self._final_llm_labels = {}
        round_dir: Optional[Path] = None
        try:
            round_dir = directory.parent.parent
        except Exception:  # noqa: BLE001
            round_dir = None
        if round_dir:
            config_path = round_dir / "round_config.json"
            if config_path.exists():
                try:
                    round_config = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    round_config = {}
                assisted_cfg = None
                if isinstance(round_config, Mapping):
                    assisted_cfg = round_config.get("assisted_review") or round_config.get("assisted_chart_review")
                if isinstance(assisted_cfg, Mapping) and assisted_cfg.get("enabled"):
                    self.assisted_review_enabled = True
                    raw_top = assisted_cfg.get("top_snippets")
                    try:
                        self.assisted_review_top_n = int(raw_top)
                    except (TypeError, ValueError):
                        self.assisted_review_top_n = 0
                    snippets_path_value = assisted_cfg.get("snippets_json")
                    if snippets_path_value:
                        candidate = Path(str(snippets_path_value))
                        if not candidate.is_absolute():
                            candidate = (round_dir / candidate).resolve()
                        if candidate.exists():
                            try:
                                snippet_payload = json.loads(candidate.read_text(encoding="utf-8"))
                            except Exception:  # noqa: BLE001
                                snippet_payload = {}
                            unit_payload = (
                                snippet_payload.get("unit_snippets")
                                if isinstance(snippet_payload, Mapping)
                                else {}
                            )
                            if isinstance(unit_payload, Mapping):
                                normalized: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
                                for unit_id, labels in unit_payload.items():
                                    if not isinstance(labels, Mapping):
                                        continue
                                    label_entries: Dict[str, List[Dict[str, object]]] = {}
                                    for label_id, entries in labels.items():
                                        if not isinstance(entries, list):
                                            continue
                                        cleaned: List[Dict[str, object]] = []
                                        for entry in entries:
                                            if isinstance(entry, Mapping):
                                                cleaned_entry: Dict[str, object] = {
                                                    str(k): entry[k] for k in entry
                                                }
                                                metadata_val = cleaned_entry.get("metadata")
                                                if isinstance(metadata_val, Mapping):
                                                    cleaned_entry["metadata"] = {
                                                        str(mk): metadata_val[mk]
                                                        for mk in metadata_val
                                                    }
                                                cleaned.append(cleaned_entry)
                                        if cleaned:
                                            label_entries[str(label_id)] = cleaned
                                    if label_entries:
                                        normalized[str(unit_id)] = label_entries
                                if normalized:
                                    self._assisted_review_snippets = normalized
                            if not self.assisted_review_top_n and isinstance(snippet_payload, Mapping):
                                payload_top = snippet_payload.get("top_snippets")
                                try:
                                    self.assisted_review_top_n = int(payload_top)
                                except (TypeError, ValueError):
                                    pass
                final_enabled = False
                if isinstance(round_config, Mapping):
                    final_enabled = self._parse_bool(round_config.get("final_llm_labeling"), False)
                    outputs_cfg = round_config.get("final_llm_outputs")
                    include_reasoning_value = round_config.get("final_llm_include_reasoning")
                    if isinstance(outputs_cfg, Mapping):
                        if not final_enabled:
                            final_enabled = True
                        if include_reasoning_value is None:
                            include_reasoning_value = outputs_cfg.get("final_llm_include_reasoning")
                        self._final_llm_reasoning = self._parse_bool(include_reasoning_value, True)
                        by_unit_value = outputs_cfg.get("final_llm_labels_by_unit")
                        if by_unit_value:
                            try:
                                candidate = Path(str(by_unit_value))
                            except Exception:  # noqa: BLE001
                                candidate = None
                            if candidate is not None and not candidate.is_absolute():
                                candidate = (round_dir / candidate).resolve()
                            if candidate is not None and candidate.exists():
                                try:
                                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                                except Exception:  # noqa: BLE001
                                    payload = {}
                                if isinstance(payload, Mapping):
                                    normalized_llm: Dict[str, Dict[str, Dict[str, object]]] = {}
                                    for unit_key, entries in payload.items():
                                        if not isinstance(entries, list):
                                            continue
                                        label_entries: Dict[str, Dict[str, object]] = {}
                                        for entry in entries:
                                            if not isinstance(entry, Mapping):
                                                continue
                                            label_key = str(entry.get("label_id") or "").strip()
                                            if not label_key:
                                                continue
                                            cleaned = {str(k): entry[k] for k in entry}
                                            label_entries[label_key] = cleaned
                                        if label_entries:
                                            normalized_llm[str(unit_key)] = label_entries
                                    if normalized_llm:
                                        self._final_llm_labels = normalized_llm
                else:
                    final_enabled = False
                self.final_llm_enabled = final_enabled or bool(self._final_llm_labels)
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
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN metadata_json TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass
            unit_rows = conn.execute("SELECT * FROM units ORDER BY display_rank").fetchall()
            self.units = [dict(row) for row in unit_rows]
            self._unit_completion_baseline = {
                str(unit["unit_id"]): {
                    "complete": bool(unit.get("complete")),
                    "completed_at": unit.get("completed_at"),
                }
                for unit in self.units
            }
            document_rows = conn.execute("SELECT * FROM documents").fetchall()
            self._document_cache = {}
            for row in document_rows:
                data = dict(row)
                metadata_json = data.get("metadata_json")
                metadata: Dict[str, object]
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except Exception:  # noqa: BLE001
                        metadata = {}
                else:
                    metadata = {}
                data["metadata"] = metadata
                self._document_cache[str(row["doc_id"])] = data
            note_rows = conn.execute(
                "SELECT unit_id, doc_id, order_index FROM unit_notes ORDER BY unit_id, order_index"
            ).fetchall()
            unit_documents: Dict[str, List[Dict[str, object]]] = {}
            for row in note_rows:
                unit_id = str(row["unit_id"])
                doc_id = str(row["doc_id"])
                order_index_raw = row["order_index"]
                try:
                    order_index = int(order_index_raw)
                except (TypeError, ValueError):
                    order_index = 0
                doc_record = self._document_cache.get(doc_id, {})
                text = str(doc_record.get("text", ""))
                metadata = dict(doc_record.get("metadata") or {})
                doc_entry: Dict[str, object] = {
                    "order_index": order_index,
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": metadata,
                }
                if "hash" in doc_record:
                    doc_entry["hash"] = doc_record.get("hash")
                unit_documents.setdefault(unit_id, []).append(
                    doc_entry
                )
            for unit_id, docs in unit_documents.items():
                docs.sort(key=lambda entry: (entry.get("order_index", 0), entry.get("doc_id")))
            self._unit_document_cache = {}
            for unit_id, docs in unit_documents.items():
                copies: List[Dict[str, object]] = []
                for doc in docs:
                    doc_copy = dict(doc)
                    metadata = doc_copy.get("metadata")
                    if isinstance(metadata, dict):
                        doc_copy["metadata"] = dict(metadata)
                    copies.append(doc_copy)
                self._unit_document_cache[unit_id] = copies
            annotation_rows = conn.execute("SELECT * FROM annotations").fetchall()
            baseline_annotations: Dict[str, Dict[str, Dict[str, object]]] = {}
            for row in annotation_rows:
                unit_id = str(row["unit_id"])
                label_id = str(row["label_id"])
                baseline_annotations.setdefault(unit_id, {})[label_id] = dict(row)
            self._annotation_baseline = {
                unit_id: {label_id: dict(data) for label_id, data in labels.items()}
                for unit_id, labels in baseline_annotations.items()
            }
            self._annotation_cache = {
                unit_id: {label_id: dict(data) for label_id, data in labels.items()}
                for unit_id, labels in self._annotation_baseline.items()
            }
            rationale_rows = conn.execute("SELECT * FROM rationales").fetchall()
            baseline_rationales: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
            for row in rationale_rows:
                unit_id = str(row["unit_id"])
                label_id = str(row["label_id"])
                entry = {
                    "rationale_id": str(row["rationale_id"]),
                    "unit_id": unit_id,
                    "label_id": label_id,
                    "doc_id": str(row["doc_id"]),
                    "start_offset": int(row["start_offset"]),
                    "end_offset": int(row["end_offset"]),
                    "snippet": str(row["snippet"] or ""),
                    "created_at": str(row["created_at"] or ""),
                }
                baseline_rationales.setdefault(unit_id, {}).setdefault(label_id, []).append(entry)
            for label_map in baseline_rationales.values():
                for entries in label_map.values():
                    self._sort_rationales(entries)
            self._rationale_baseline = {
                unit_id: {
                    label_id: [dict(entry) for entry in entries]
                    for label_id, entries in labels.items()
                }
                for unit_id, labels in baseline_rationales.items()
            }
            self._rationale_cache = {
                unit_id: {
                    label_id: [dict(entry) for entry in entries]
                    for label_id, entries in labels.items()
                }
                for unit_id, labels in self._rationale_baseline.items()
            }
            for unit in self.units:
                unit_id = str(unit.get("unit_id"))
                self._annotation_baseline.setdefault(unit_id, {})
                self._annotation_cache.setdefault(unit_id, {})
                self._rationale_baseline.setdefault(unit_id, {})
                self._rationale_cache.setdefault(unit_id, {})
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
        self._refresh_dirty_state("Assignment loaded")

    def set_current_unit(self, unit: Dict[str, object]) -> None:
        self.current_unit = unit
        unit_id = str(unit.get("unit_id", "")) if unit else None
        self.current_unit_id = unit_id or None
        self.current_annotations = (
            self.load_annotations(self.current_unit_id) if self.current_unit_id else {}
        )
        self.current_rationales = (
            self.load_rationales(self.current_unit_id) if self.current_unit_id else {}
        )
        self.unit_changed.emit(unit)

    # Database helpers -----------------------------------------------------

    def fetch_document(self, doc_id: str, unit_id: Optional[str] = None) -> str:
        if unit_id:
            cached_docs = self._unit_document_cache.get(unit_id)
            if cached_docs:
                for doc in cached_docs:
                    if str(doc.get("doc_id")) == doc_id:
                        return str(doc.get("text", ""))
        doc_record = self._document_cache.get(doc_id)
        text = str(doc_record.get("text", "")) if doc_record else ""
        if unit_id:
            existing_docs = self._unit_document_cache.get(unit_id)
            if not existing_docs:
                self.cache_unit_documents(
                    unit_id,
                    [
                        {
                            "order_index": 1,
                            "doc_id": doc_id,
                            "text": text,
                        }
                    ],
                )
            else:
                for doc in existing_docs:
                    if str(doc.get("doc_id")) == doc_id:
                        doc["text"] = text
                        break
        return text

    def document_metadata(self, doc_id: str) -> Dict[str, object]:
        record = self._document_cache.get(doc_id)
        if not record:
            return {}
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            return dict(metadata)
        metadata_json = record.get("metadata_json")
        if isinstance(metadata_json, str) and metadata_json:
            try:
                parsed = json.loads(metadata_json)
            except Exception:  # noqa: BLE001
                return {}
            record["metadata"] = parsed if isinstance(parsed, dict) else {}
            if isinstance(parsed, dict):
                return dict(parsed)
        return {}

    def fetch_unit_documents(self, unit_id: str) -> List[Dict[str, object]]:
        cached = self._unit_document_cache.get(unit_id)
        if cached is not None:
            copies: List[Dict[str, object]] = []
            for doc in cached:
                doc_copy = dict(doc)
                metadata = doc_copy.get("metadata")
                if isinstance(metadata, dict):
                    doc_copy["metadata"] = dict(metadata)
                copies.append(doc_copy)
            return copies
        return []

    def invalidate_unit_documents(self, unit_id: str) -> None:
        self._unit_document_cache.pop(unit_id, None)

    def cache_unit_documents(
        self, unit_id: str, documents: List[Dict[str, object]]
    ) -> None:
        cached_docs: List[Dict[str, object]] = []
        for doc in documents:
            doc_copy = dict(doc)
            metadata_value = doc_copy.get("metadata")
            metadata_json: Optional[str] = None
            if isinstance(metadata_value, dict):
                doc_copy["metadata"] = dict(metadata_value)
                try:
                    metadata_json = json.dumps(metadata_value, sort_keys=True)
                except Exception:  # noqa: BLE001
                    metadata_json = None
            doc_id = str(doc_copy.get("doc_id", ""))
            if doc_id:
                existing = dict(self._document_cache.get(doc_id, {}))
                existing.setdefault("doc_id", doc_id)
                if "text" in doc_copy:
                    existing["text"] = doc_copy["text"]
                if metadata_value:
                    existing["metadata"] = dict(metadata_value)
                    if metadata_json is not None:
                        existing["metadata_json"] = metadata_json
                self._document_cache[doc_id] = existing
            cached_docs.append(doc_copy)
        self._unit_document_cache[unit_id] = cached_docs

    def has_assisted_review(self) -> bool:
        return bool(self.assisted_review_enabled)

    def assisted_review_top_k(self) -> int:
        return int(self.assisted_review_top_n)

    def get_assisted_snippets(self, unit_id: str, label_id: str) -> List[Dict[str, object]]:
        unit_map = self._assisted_review_snippets.get(str(unit_id), {})
        entries = unit_map.get(str(label_id), [])
        return [dict(entry) for entry in entries]

    def has_final_llm_labels(self) -> bool:
        return bool(self.final_llm_enabled)

    def final_llm_reasoning_enabled(self) -> bool:
        return bool(self._final_llm_reasoning)

    def get_final_llm_label(self, unit_id: str, label_id: str) -> Dict[str, object]:
        unit_map = self._final_llm_labels.get(str(unit_id), {})
        entry = unit_map.get(str(label_id))
        if not isinstance(entry, Mapping):
            return {}
        return {str(key): entry[key] for key in entry}

    def load_annotations(self, unit_id: str) -> Dict[str, Dict[str, object]]:
        self._annotation_baseline.setdefault(unit_id, {})
        cache = self._annotation_cache.setdefault(unit_id, {})
        return {label_id: dict(data) for label_id, data in cache.items()}

    @staticmethod
    def _sort_rationales(entries: List[Dict[str, object]]) -> None:
        entries.sort(
            key=lambda entry: (
                str(entry.get("doc_id", "")),
                int(entry.get("start_offset", 0)),
                int(entry.get("end_offset", 0)),
                str(entry.get("created_at", "")),
            )
        )

    def load_rationales(self, unit_id: str) -> Dict[str, List[Dict[str, object]]]:
        cache = self._rationale_cache.setdefault(unit_id, {})
        return {label_id: [dict(entry) for entry in entries] for label_id, entries in cache.items()}

    def _remove_rationale_events(self, rationale_id: str, event_types: Optional[Set[str]] = None) -> None:
        if not self._pending_events:
            return
        remaining: List[models.Event] = []
        for event in self._pending_events:
            if not event.event_type.startswith("rationale_"):
                remaining.append(event)
                continue
            if event_types is not None and event.event_type not in event_types:
                remaining.append(event)
                continue
            try:
                payload = json.loads(event.payload_json)
            except Exception:  # noqa: BLE001
                payload = {}
            if payload.get("rationale_id") == rationale_id:
                continue
            remaining.append(event)
        self._pending_events = remaining

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

        existing_annotation = (
            self._annotation_cache.get(unit_id, {}).get(label_id)
            or self._annotation_baseline.get(unit_id, {}).get(label_id)
            or {}
        )
        base_state: Dict[str, object] = {
            "value": existing_annotation.get("value"),
            "value_num": existing_annotation.get("value_num"),
            "value_date": existing_annotation.get("value_date"),
            "na": bool(existing_annotation.get("na")),
            "notes": existing_annotation.get("notes"),
        }
        base_state.update(_normalized_state(payload))
        normalized_state = {
            "value": base_state.get("value"),
            "value_num": base_state.get("value_num"),
            "value_date": base_state.get("value_date"),
            "na": 1 if base_state.get("na") else 0,
            "notes": base_state.get("notes"),
        }
        baseline_annotation = self._annotation_baseline.get(unit_id, {}).get(label_id, {})
        baseline_state = {
            "value": baseline_annotation.get("value"),
            "value_num": baseline_annotation.get("value_num"),
            "value_date": baseline_annotation.get("value_date"),
            "na": baseline_annotation.get("na", 0),
            "notes": baseline_annotation.get("notes"),
        }
        key = (unit_id, label_id)
        if normalized_state == baseline_state:
            self._pending_annotations.pop(key, None)
            self._pending_annotation_events.pop(key, None)
        else:
            record = models.Annotation(
                unit_id=unit_id,
                label_id=label_id,
                value=normalized_state["value"],
                value_num=normalized_state["value_num"],
                value_date=normalized_state["value_date"],
                na=normalized_state["na"],
                notes=normalized_state["notes"],
            )
            self._pending_annotations[key] = record
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
            self._pending_annotation_events[key] = event
        annotation_dict = {
            "unit_id": unit_id,
            "label_id": label_id,
            "value": normalized_state["value"],
            "value_num": normalized_state["value_num"],
            "value_date": normalized_state["value_date"],
            "na": normalized_state["na"],
            "notes": normalized_state["notes"],
        }
        self._annotation_cache.setdefault(unit_id, {})[label_id] = dict(annotation_dict)
        if self.current_unit_id == unit_id:
            self.current_annotations[label_id] = dict(annotation_dict)
        self._refresh_dirty_state()

    def save_rationale(self, unit_id: str, label_id: str, doc_id: str, start: int, end: int, snippet: str) -> str:
        if not self.assignment_db:
            return ""
        created_at = QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)
        record = models.Rationale(
            rationale_id=str(uuid.uuid4()),
            unit_id=unit_id,
            label_id=label_id,
            doc_id=doc_id,
            start_offset=start,
            end_offset=end,
            snippet=snippet,
            created_at=created_at,
        )
        self._pending_rationale_inserts[record.rationale_id] = record
        self._pending_rationale_deletes.discard(record.rationale_id)
        cache = self._rationale_cache.setdefault(unit_id, {})
        highlights = cache.setdefault(label_id, [])
        highlights.append(
            {
                "rationale_id": record.rationale_id,
                "unit_id": unit_id,
                "label_id": label_id,
                "doc_id": doc_id,
                "start_offset": start,
                "end_offset": end,
                "snippet": snippet,
                "created_at": created_at,
            }
        )
        self._sort_rationales(highlights)
        event = models.Event(
            event_id=str(uuid.uuid4()),
            ts=created_at,
            actor="annotator",
            event_type="rationale_added",
            payload_json=json.dumps(
                {
                    "rationale_id": record.rationale_id,
                    "unit_id": unit_id,
                    "label_id": label_id,
                    "doc_id": doc_id,
                    "start": start,
                    "end": end,
                }
            ),
        )
        self._pending_events.append(event)
        if self.current_unit_id == unit_id:
            self.current_rationales = self.load_rationales(unit_id)
        self._refresh_dirty_state("Unsaved changes – click Save to persist")
        return record.rationale_id

    def update_rationale(
        self,
        unit_id: str,
        label_id: str,
        rationale_id: str,
        doc_id: str,
        start: int,
        end: int,
        snippet: str,
    ) -> bool:
        if not self.assignment_db:
            return False
        cache = self._rationale_cache.setdefault(unit_id, {})
        highlights = cache.get(label_id, [])
        target: Optional[Dict[str, object]] = None
        for entry in highlights:
            if entry.get("rationale_id") == rationale_id:
                target = entry
                break
        if not target:
            return False
        created_at = str(target.get("created_at") or "")
        target.update(
            {
                "doc_id": doc_id,
                "start_offset": start,
                "end_offset": end,
                "snippet": snippet,
            }
        )
        self._sort_rationales(highlights)
        if rationale_id in self._pending_rationale_inserts:
            record = self._pending_rationale_inserts[rationale_id]
            record.doc_id = doc_id
            record.start_offset = start
            record.end_offset = end
            record.snippet = snippet
        else:
            record = models.Rationale(
                rationale_id=rationale_id,
                unit_id=unit_id,
                label_id=label_id,
                doc_id=doc_id,
                start_offset=start,
                end_offset=end,
                snippet=snippet,
                created_at=created_at or QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
            )
            self._pending_rationale_updates[rationale_id] = record
        self._pending_rationale_deletes.discard(rationale_id)
        self._remove_rationale_events(rationale_id, {"rationale_updated"})
        event = models.Event(
            event_id=str(uuid.uuid4()),
            ts=QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
            actor="annotator",
            event_type="rationale_updated",
            payload_json=json.dumps(
                {
                    "rationale_id": rationale_id,
                    "unit_id": unit_id,
                    "label_id": label_id,
                    "doc_id": doc_id,
                    "start": start,
                    "end": end,
                }
            ),
        )
        self._pending_events.append(event)
        if self.current_unit_id == unit_id:
            self.current_rationales = self.load_rationales(unit_id)
        self._refresh_dirty_state("Unsaved changes – click Save to persist")
        return True

    def delete_rationale(self, unit_id: str, label_id: str, rationale_id: str) -> bool:
        if not self.assignment_db:
            return False
        cache = self._rationale_cache.setdefault(unit_id, {})
        highlights = cache.get(label_id, [])
        index = -1
        for idx, entry in enumerate(highlights):
            if entry.get("rationale_id") == rationale_id:
                index = idx
                break
        if index < 0:
            return False
        highlights.pop(index)
        self._sort_rationales(highlights)
        if rationale_id in self._pending_rationale_inserts:
            self._pending_rationale_inserts.pop(rationale_id, None)
            self._remove_rationale_events(rationale_id, {"rationale_added", "rationale_updated"})
        else:
            self._pending_rationale_updates.pop(rationale_id, None)
            self._pending_rationale_deletes.add(rationale_id)
            self._remove_rationale_events(rationale_id, {"rationale_updated"})
            event = models.Event(
                event_id=str(uuid.uuid4()),
                ts=QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
                actor="annotator",
                event_type="rationale_deleted",
                payload_json=json.dumps(
                    {
                        "rationale_id": rationale_id,
                        "unit_id": unit_id,
                        "label_id": label_id,
                    }
                ),
            )
            self._pending_events.append(event)
        if self.current_unit_id == unit_id:
            self.current_rationales = self.load_rationales(unit_id)
        self._refresh_dirty_state("Unsaved changes – click Save to persist")
        return True

    def mark_unit_complete(self, unit_id: str, complete: bool) -> None:
        if not self.assignment_db:
            return
        baseline_entry = self._unit_completion_baseline.get(unit_id, {})
        baseline_complete = bool(baseline_entry.get("complete", False))
        baseline_completed_at = baseline_entry.get("completed_at")
        pending_state = self._pending_completion.get(unit_id)
        current_state = bool(pending_state["complete"]) if pending_state else baseline_complete
        if current_state == complete:
            return
        if complete == baseline_complete:
            self._pending_completion.pop(unit_id, None)
            completed_at = baseline_completed_at
        else:
            completed_at = (
                QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)
                if complete
                else baseline_completed_at
            )
            self._pending_completion[unit_id] = {
                "complete": 1 if complete else 0,
                "completed_at": completed_at,
            }
        for unit in self.units:
            if str(unit.get("unit_id")) == unit_id:
                unit["complete"] = 1 if complete else 0
                unit["completed_at"] = completed_at
                break
        if self.current_unit and str(self.current_unit.get("unit_id")) == unit_id:
            self.current_unit["complete"] = 1 if complete else 0
            self.current_unit["completed_at"] = completed_at
        self._refresh_dirty_state()

    def flush_pending_writes(self) -> bool:
        if not self.assignment_db or not self._has_pending_changes():
            self._refresh_dirty_state()
            return False
        annotations_to_save = list(self._pending_annotations.items())
        annotation_events = list(self._pending_annotation_events.values())
        rationale_inserts = list(self._pending_rationale_inserts.values())
        rationale_updates = list(self._pending_rationale_updates.values())
        rationale_deletes = list(self._pending_rationale_deletes)
        other_events = list(self._pending_events)
        completion_updates = list(self._pending_completion.items())
        with self.assignment_db.transaction() as conn:
            if annotations_to_save:
                models.Annotation.insert_many(conn, [record for _, record in annotations_to_save])
            if rationale_inserts:
                models.Rationale.insert_many(conn, rationale_inserts)
            for record in rationale_updates:
                conn.execute(
                    "UPDATE rationales SET doc_id=?, start_offset=?, end_offset=?, snippet=? WHERE rationale_id=?",
                    (
                        record.doc_id,
                        record.start_offset,
                        record.end_offset,
                        record.snippet,
                        record.rationale_id,
                    ),
                )
            if rationale_deletes:
                placeholders = ",".join(["?"] * len(rationale_deletes))
                conn.execute(
                    f"DELETE FROM rationales WHERE rationale_id IN ({placeholders})",
                    rationale_deletes,
                )
            all_events = annotation_events + other_events
            if all_events:
                models.Event.insert_many(conn, all_events)
            for unit_id, payload in completion_updates:
                complete_value = int(payload.get("complete", 0))
                completed_at = payload.get("completed_at")
                conn.execute(
                    "UPDATE units SET complete=?, completed_at=CASE WHEN ?=1 THEN ? ELSE completed_at END WHERE unit_id=?",
                    (complete_value, complete_value, completed_at, unit_id),
                )
        for (unit_id, label_id), record in annotations_to_save:
            annotation_dict = {
                "unit_id": record.unit_id,
                "label_id": record.label_id,
                "value": record.value,
                "value_num": record.value_num,
                "value_date": record.value_date,
                "na": record.na,
                "notes": record.notes,
            }
            self._annotation_baseline.setdefault(unit_id, {})[label_id] = dict(annotation_dict)
            self._annotation_cache.setdefault(unit_id, {})[label_id] = dict(annotation_dict)
            if self.current_unit_id == unit_id:
                self.current_annotations[label_id] = dict(annotation_dict)
        for unit_id, payload in completion_updates:
            baseline_entry = self._unit_completion_baseline.setdefault(unit_id, {})
            baseline_entry["complete"] = bool(payload.get("complete", 0))
            baseline_entry["completed_at"] = payload.get("completed_at")
        affected_units: Set[str] = set()
        for record in rationale_inserts:
            entry = {
                "rationale_id": record.rationale_id,
                "unit_id": record.unit_id,
                "label_id": record.label_id,
                "doc_id": record.doc_id,
                "start_offset": record.start_offset,
                "end_offset": record.end_offset,
                "snippet": record.snippet,
                "created_at": record.created_at,
            }
            baseline_list = self._rationale_baseline.setdefault(record.unit_id, {}).setdefault(
                record.label_id, []
            )
            baseline_list.append(dict(entry))
            self._sort_rationales(baseline_list)
            cache_list = self._rationale_cache.setdefault(record.unit_id, {}).setdefault(
                record.label_id, []
            )
            cache_list.append(dict(entry))
            self._sort_rationales(cache_list)
            affected_units.add(record.unit_id)
        for record in rationale_updates:
            entry = {
                "doc_id": record.doc_id,
                "start_offset": record.start_offset,
                "end_offset": record.end_offset,
                "snippet": record.snippet,
            }
            baseline_list = self._rationale_baseline.setdefault(record.unit_id, {}).setdefault(
                record.label_id, []
            )
            cache_list = self._rationale_cache.setdefault(record.unit_id, {}).setdefault(
                record.label_id, []
            )
            for target in baseline_list:
                if target.get("rationale_id") == record.rationale_id:
                    target.update(entry)
                    break
            else:
                baseline_list.append({"rationale_id": record.rationale_id, **entry})
            self._sort_rationales(baseline_list)
            for target in cache_list:
                if target.get("rationale_id") == record.rationale_id:
                    target.update(entry)
                    break
            else:
                cache_list.append({"rationale_id": record.rationale_id, **entry})
            self._sort_rationales(cache_list)
            affected_units.add(record.unit_id)
        for rationale_id in rationale_deletes:
            for container in (self._rationale_baseline, self._rationale_cache):
                for unit_id, labels in container.items():
                    changed = False
                    for label_id, entries in labels.items():
                        before = len(entries)
                        entries[:] = [entry for entry in entries if entry.get("rationale_id") != rationale_id]
                        if len(entries) != before:
                            self._sort_rationales(entries)
                            changed = True
                    if changed:
                        affected_units.add(unit_id)
        self._pending_annotations.clear()
        self._pending_annotation_events.clear()
        self._pending_rationale_inserts.clear()
        self._pending_rationale_updates.clear()
        self._pending_rationale_deletes.clear()
        self._pending_events.clear()
        self._pending_completion.clear()
        if self.current_unit_id and self.current_unit_id in affected_units:
            self.current_rationales = self.load_rationales(self.current_unit_id)
        self._refresh_dirty_state("All changes saved")
        return True

    def discard_pending_changes(self) -> None:
        if not self.has_unsaved_changes():
            return
        self._pending_annotations.clear()
        self._pending_annotation_events.clear()
        self._pending_rationale_inserts.clear()
        self._pending_rationale_updates.clear()
        self._pending_rationale_deletes.clear()
        self._pending_events.clear()
        self._pending_completion.clear()
        self._annotation_cache = {
            unit_id: {label_id: dict(data) for label_id, data in annotations.items()}
            for unit_id, annotations in self._annotation_baseline.items()
        }
        self._rationale_cache = {
            unit_id: {
                label_id: [dict(entry) for entry in entries]
                for label_id, entries in labels.items()
            }
            for unit_id, labels in self._rationale_baseline.items()
        }
        for unit in self.units:
            unit_id = str(unit.get("unit_id"))
            baseline_entry = self._unit_completion_baseline.get(unit_id, {})
            unit["complete"] = 1 if baseline_entry.get("complete") else 0
            unit["completed_at"] = baseline_entry.get("completed_at")
        if self.current_unit:
            self.set_current_unit(self.current_unit)
        self._refresh_dirty_state()

    def has_unsaved_changes(self) -> bool:
        return self._dirty

    def _has_pending_changes(self) -> bool:
        return any(
            [
                bool(self._pending_annotations),
                bool(self._pending_annotation_events),
                bool(self._pending_rationale_inserts),
                bool(self._pending_rationale_updates),
                bool(self._pending_rationale_deletes),
                bool(self._pending_events),
                bool(self._pending_completion),
            ]
        )

    def _refresh_dirty_state(self, message: Optional[str] = None) -> None:
        dirty_before = self._dirty
        self._dirty = self._has_pending_changes()
        if message is None:
            if self._dirty and not dirty_before:
                message = "Unsaved changes – click Save to persist"
            elif not self._dirty and dirty_before:
                message = "All changes saved"
        if message:
            self.save_state_changed.emit(message)


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
        self.layout = QtWidgets.QVBoxLayout(self.container)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(12)
        self.label_widgets: Dict[str, Dict[str, object]] = {}
        self.current_unit_id: Optional[str] = None
        self.current_annotations: Dict[str, Dict[str, object]] = {}
        self.current_rationales: Dict[str, List[Dict[str, object]]] = {}

    def clear(self) -> None:
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.label_widgets.clear()
        self.current_rationales = {}

    def set_schema(self, labels: List[LabelDefinition]) -> None:
        self.clear()
        for label in labels:
            row_widget = self._create_row(label)
            row_widget.setProperty("label_id", label.label_id)
            self.layout.addWidget(row_widget)
            self._update_snippet_button(label.label_id)
            self._update_llm_button(label.label_id)
        self.layout.addStretch()

    def load_unit(
        self,
        unit_id: str,
        annotations: Dict[str, Dict[str, object]],
        rationales: Dict[str, List[Dict[str, object]]],
    ) -> None:
        self.current_unit_id = unit_id
        self.current_annotations = annotations
        self.current_rationales = rationales
        with self._suspend_widget_signals():
            for widgets in self.label_widgets.values():
                self._reset_widgets(widgets)
            for label_id, widgets in self.label_widgets.items():
                annotation = annotations.get(label_id)
                if annotation:
                    self._apply_annotation(label_id, widgets, annotation)
                self._refresh_highlights(label_id)
        self._update_gating()
        self._update_completion()

    # internal helpers -----------------------------------------------------

    def _create_row(self, label: LabelDefinition) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QWidget()
        wrapper.setProperty("label_id", label.label_id)
        v_layout = QtWidgets.QVBoxLayout(wrapper)
        v_layout.setContentsMargins(0, 0, 12, 12)
        v_layout.setSpacing(6)
        value_widget: QtWidgets.QWidget
        state: Dict[str, object] = {"definition": label, "row_widget": wrapper}

        header_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(label.name)
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        title_label.setProperty("label_id", label.label_id)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        snippet_btn = QtWidgets.QPushButton("Show relevant snippets")
        snippet_btn.setAutoDefault(False)
        snippet_btn.clicked.connect(
            lambda _checked, lid=label.label_id: self._show_assisted_snippets(lid)
        )
        snippet_btn.setVisible(self.ctx.has_assisted_review())
        snippet_btn.setEnabled(False)
        header_layout.addWidget(snippet_btn)
        state["snippet_btn"] = snippet_btn
        llm_btn = QtWidgets.QPushButton("View LLM label")
        llm_btn.setAutoDefault(False)
        llm_btn.clicked.connect(lambda _checked, lid=label.label_id: self._show_llm_label(lid))
        llm_btn.setVisible(self.ctx.has_final_llm_labels())
        llm_btn.setEnabled(False)
        header_layout.addWidget(llm_btn)
        state["llm_btn"] = llm_btn
        if label.na_allowed:
            na_box = QtWidgets.QCheckBox("N/A")
            na_box.stateChanged.connect(
                lambda _state, lid=label.label_id, widget=na_box: self._on_na(lid, widget)
            )
            header_layout.addWidget(na_box)
            state["na_box"] = na_box
        v_layout.addLayout(header_layout)

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
            unknown_box = QtWidgets.QCheckBox("Unknown")
            unknown_box.stateChanged.connect(
                lambda _state, lid=label.label_id, widget=line, toggle=unknown_box: self._on_numeric_unknown(
                    lid, widget, toggle
                )
            )
            numeric_layout = QtWidgets.QHBoxLayout()
            numeric_layout.setContentsMargins(0, 0, 0, 0)
            numeric_layout.addWidget(line)
            numeric_layout.addWidget(unknown_box)
            numeric_layout.addStretch()
            numeric_widget = QtWidgets.QWidget()
            numeric_widget.setLayout(numeric_layout)
            value_widget = numeric_widget
            state["line_edit"] = line
            state["unknown_box"] = unknown_box
        elif label.type == "date":
            date = QtWidgets.QDateEdit()
            date.setCalendarPopup(True)
            date.dateChanged.connect(lambda _date, lid=label.label_id, widget=date: self._on_date(lid, widget))
            unknown_box = QtWidgets.QCheckBox("Unknown")
            unknown_box.stateChanged.connect(
                lambda _state, lid=label.label_id, widget=date, toggle=unknown_box: self._on_date_unknown(
                    lid, widget, toggle
                )
            )
            date_layout = QtWidgets.QHBoxLayout()
            date_layout.setContentsMargins(0, 0, 0, 0)
            date_layout.addWidget(date)
            date_layout.addWidget(unknown_box)
            date_layout.addStretch()
            date_widget = QtWidgets.QWidget()
            date_widget.setLayout(date_layout)
            value_widget = date_widget
            state["date_edit"] = date
            state["unknown_box"] = unknown_box
        else:
            text = QtWidgets.QTextEdit()
            text.textChanged.connect(lambda lid=label.label_id, widget=text: self._on_text(lid, widget))
            value_widget = text
            state["text_edit"] = text
        main_section = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_section)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        main_layout.addWidget(value_widget)

        notes = QtWidgets.QLineEdit()
        notes.setPlaceholderText("Notes")
        notes.setClearButtonEnabled(True)
        notes.editingFinished.connect(lambda lid=label.label_id, widget=notes: self._on_notes(lid, widget))
        main_layout.addWidget(notes)
        state["notes"] = notes

        if label.rules:
            rules_label = QtWidgets.QLabel(label.rules)
            rules_label.setWordWrap(True)
            main_layout.addWidget(rules_label)

        v_layout.addWidget(main_section)

        highlight_section = QtWidgets.QWidget()
        highlight_layout = QtWidgets.QVBoxLayout(highlight_section)
        highlight_layout.setContentsMargins(0, 0, 0, 0)
        highlight_layout.setSpacing(4)

        highlight_controls = QtWidgets.QHBoxLayout()
        highlight_btn = QtWidgets.QPushButton("Add highlight")
        highlight_btn.clicked.connect(lambda _checked, lid=label.label_id: self._add_highlight(lid))
        highlight_controls.addWidget(highlight_btn)
        update_btn = QtWidgets.QPushButton("Replace from selection")
        update_btn.setEnabled(False)
        update_btn.clicked.connect(
            lambda _checked, lid=label.label_id: self._update_highlight_from_selection(lid)
        )
        highlight_controls.addWidget(update_btn)
        delete_btn = QtWidgets.QPushButton("Delete highlight")
        delete_btn.setEnabled(False)
        delete_btn.clicked.connect(lambda _checked, lid=label.label_id: self._delete_highlight(lid))
        highlight_controls.addWidget(delete_btn)
        highlight_controls.addStretch()
        highlight_layout.addLayout(highlight_controls)
        state["highlight_update_btn"] = update_btn
        state["highlight_delete_btn"] = delete_btn

        highlight_list = QtWidgets.QTreeWidget()
        highlight_list.setColumnCount(3)
        highlight_list.setHeaderLabels(["Document", "Range", "Text"])
        highlight_list.setRootIsDecorated(False)
        highlight_list.setIndentation(0)
        highlight_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        highlight_list.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        highlight_list.setUniformRowHeights(True)
        highlight_list.setAlternatingRowColors(True)
        highlight_list.setMinimumHeight(40)
        highlight_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )
        header = highlight_list.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        highlight_list.itemSelectionChanged.connect(
            lambda lid=label.label_id: self._on_highlight_selection_changed(lid)
        )
        highlight_layout.addWidget(highlight_list)
        state["highlight_list"] = highlight_list

        highlight_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        highlight_splitter.setChildrenCollapsible(False)
        highlight_splitter.setHandleWidth(6)
        highlight_splitter.addWidget(highlight_section)

        highlight_spacer = QtWidgets.QWidget()
        highlight_spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        highlight_splitter.addWidget(highlight_spacer)
        highlight_splitter.setStretchFactor(0, 0)
        highlight_splitter.setStretchFactor(1, 1)
        highlight_splitter.setCollapsible(0, True)
        raw_height = max(1, highlight_section.sizeHint().height())
        initial_height = max(40, raw_height // 3)
        highlight_splitter.setSizes([initial_height, max(40, raw_height - initial_height)])

        v_layout.addWidget(highlight_splitter)
        self.label_widgets[label.label_id] = state
        return wrapper

    @staticmethod
    def _format_highlight_snippet(snippet: object) -> str:
        text = str(snippet or "")
        text = text.replace("\u2029", " ").replace("\n", " ")
        text = " ".join(text.split())
        if len(text) > 120:
            return text[:117] + "…"
        return text

    def _refresh_highlights(self, label_id: str) -> None:
        widgets = self.label_widgets.get(label_id)
        if not widgets:
            return
        highlight_list = widgets.get("highlight_list")
        if not isinstance(highlight_list, QtWidgets.QTreeWidget):
            return
        highlights = self.current_rationales.get(label_id, [])
        highlight_list.blockSignals(True)
        highlight_list.clear()
        for entry in highlights:
            doc_id = str(entry.get("doc_id", ""))
            start = int(entry.get("start_offset", 0))
            end = int(entry.get("end_offset", 0))
            range_display = f"{start}-{end}" if end >= start else str(start)
            snippet = self._format_highlight_snippet(entry.get("snippet"))
            item = QtWidgets.QTreeWidgetItem([doc_id, range_display, snippet])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, dict(entry))
            highlight_list.addTopLevelItem(item)
        highlight_list.blockSignals(False)
        self._on_highlight_selection_changed(label_id)
        self._update_snippet_button(label_id)
        self._update_llm_button(label_id)

    def _update_snippet_button(self, label_id: str) -> None:
        widgets = self.label_widgets.get(label_id)
        if not widgets:
            return
        button = widgets.get("snippet_btn")
        if not isinstance(button, QtWidgets.QPushButton):
            return
        visible = self.ctx.has_assisted_review()
        has_data = False
        if visible and self.current_unit_id:
            snippets = self.ctx.get_assisted_snippets(self.current_unit_id, label_id)
            has_data = bool(snippets)
        button.setVisible(visible)
        button.setEnabled(has_data)

    def _update_llm_button(self, label_id: str) -> None:
        widgets = self.label_widgets.get(label_id)
        if not widgets:
            return
        button = widgets.get("llm_btn")
        if not isinstance(button, QtWidgets.QPushButton):
            return
        visible = self.ctx.has_final_llm_labels()
        has_value = False
        if visible and self.current_unit_id:
            entry = self.ctx.get_final_llm_label(self.current_unit_id, label_id)
            has_value = bool(entry)
        button.setVisible(visible)
        button.setEnabled(has_value)

    def _show_assisted_snippets(self, label_id: str) -> None:
        if not self.ctx.has_assisted_review():
            QtWidgets.QMessageBox.information(
                self,
                "Assisted chart review",
                "Assisted chart review snippets are not available for this assignment.",
            )
            return
        if not self.current_unit_id:
            QtWidgets.QMessageBox.information(
                self,
                "Assisted chart review",
                "Select a unit before viewing assisted chart review snippets.",
            )
            return
        snippets = self.ctx.get_assisted_snippets(self.current_unit_id, label_id)
        if not snippets:
            QtWidgets.QMessageBox.information(
                self,
                "Assisted chart review",
                "No relevant snippets are available for this label.",
            )
            return
        widgets = self.label_widgets.get(label_id) or {}
        definition = widgets.get("definition")
        label_name = label_id
        if isinstance(definition, LabelDefinition):
            label_name = definition.name
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Relevant snippets • {label_name}")
        dialog.resize(520, 400)
        layout = QtWidgets.QVBoxLayout(dialog)
        header_parts = [f"Unit {self.current_unit_id}"]
        limit = self.ctx.assisted_review_top_k()
        if limit:
            header_parts.append(f"configured top {limit}")
        header_label = QtWidgets.QLabel(" • ".join(header_parts))
        header_label.setWordWrap(True)
        layout.addWidget(header_label)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(12)
        for index, entry in enumerate(snippets, 1):
            score = entry.get("score")
            source = entry.get("source") or ""
            doc_id = entry.get("doc_id") or ""
            chunk = entry.get("chunk_id")
            title_bits = [f"Snippet {index}"]
            if doc_id:
                title_bits.append(f"Doc {doc_id}")
            if chunk is not None:
                title_bits.append(f"Chunk {chunk}")
            if score is not None:
                try:
                    title_bits.append(f"Score {float(score):.3f}")
                except (TypeError, ValueError):
                    pass
            if source:
                title_bits.append(str(source))
            group = QtWidgets.QGroupBox(" • ".join(title_bits))
            group_layout = QtWidgets.QVBoxLayout(group)
            metadata = entry.get("metadata") if isinstance(entry, Mapping) else {}
            if isinstance(metadata, Mapping) and metadata:
                meta_parts = [f"{str(k)}: {metadata[k]}" for k in metadata.keys()]
                meta_label = QtWidgets.QLabel("; ".join(meta_parts))
                meta_label.setWordWrap(True)
                meta_label.setStyleSheet("color: palette(mid);")
                group_layout.addWidget(meta_label)
            text_label = QtWidgets.QLabel(str(entry.get("text") or ""))
            text_label.setWordWrap(True)
            text_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            group_layout.addWidget(text_label)
            container_layout.addWidget(group)
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_llm_label(self, label_id: str) -> None:
        if not self.ctx.has_final_llm_labels():
            QtWidgets.QMessageBox.information(
                self,
                "LLM label",
                "Final LLM labels are not available for this assignment.",
            )
            return
        if not self.current_unit_id:
            QtWidgets.QMessageBox.information(
                self,
                "LLM label",
                "Select a unit before viewing the LLM label.",
            )
            return
        entry = self.ctx.get_final_llm_label(self.current_unit_id, label_id)
        if not entry:
            QtWidgets.QMessageBox.information(
                self,
                "LLM label",
                "No LLM label is available for this unit and label.",
            )
            return
        widgets = self.label_widgets.get(label_id) or {}
        definition = widgets.get("definition")
        label_name = label_id
        if isinstance(definition, LabelDefinition) and definition.name:
            label_name = definition.name
        prediction_value = entry.get("llm_prediction") or entry.get("prediction")
        prediction_text = str(prediction_value) if prediction_value is not None else "(none)"
        reasoning_text = None
        if self.ctx.final_llm_reasoning_enabled():
            raw_reasoning = entry.get("llm_reasoning")
            if raw_reasoning is not None:
                reasoning_text = str(raw_reasoning)
        consistency_value = entry.get("llm_consistency") or entry.get("consistency")
        if consistency_value is not None:
            try:
                consistency_float = float(consistency_value)
                consistency_display = f"{consistency_float:.3f}"
            except (TypeError, ValueError):
                consistency_display = str(consistency_value)
        else:
            consistency_display = None

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"LLM label • {label_name}")
        dialog.resize(460, 260)
        layout = QtWidgets.QVBoxLayout(dialog)
        summary = QtWidgets.QLabel(f"Prediction: {prediction_text}")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if consistency_display:
            consistency_label = QtWidgets.QLabel(
                f"Self-consistency agreement: {consistency_display}"
            )
            consistency_label.setWordWrap(True)
            layout.addWidget(consistency_label)
        if reasoning_text:
            reasoning_label = QtWidgets.QLabel("Reasoning:")
            layout.addWidget(reasoning_label)
            reasoning_box = QtWidgets.QPlainTextEdit()
            reasoning_box.setPlainText(reasoning_text)
            reasoning_box.setReadOnly(True)
            reasoning_box.setMinimumHeight(140)
            layout.addWidget(reasoning_box)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        dialog.exec()

    def _selected_highlight(self, label_id: str) -> Optional[Dict[str, object]]:
        widgets = self.label_widgets.get(label_id)
        if not widgets:
            return None
        highlight_list = widgets.get("highlight_list")
        if not isinstance(highlight_list, QtWidgets.QTreeWidget):
            return None
        items = highlight_list.selectedItems()
        if not items:
            return None
        data = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        return dict(data) if isinstance(data, dict) else None

    def _select_highlight(self, label_id: str, rationale_id: str) -> None:
        widgets = self.label_widgets.get(label_id)
        if not widgets:
            return
        highlight_list = widgets.get("highlight_list")
        if not isinstance(highlight_list, QtWidgets.QTreeWidget):
            return
        for index in range(highlight_list.topLevelItemCount()):
            item = highlight_list.topLevelItem(index)
            data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and data.get("rationale_id") == rationale_id:
                highlight_list.setCurrentItem(item)
                break

    def _on_highlight_selection_changed(self, label_id: str) -> None:
        widgets = self.label_widgets.get(label_id)
        if not widgets:
            return
        has_selection = bool(self._selected_highlight(label_id))
        for key in ("highlight_update_btn", "highlight_delete_btn"):
            widget = widgets.get(key)
            if isinstance(widget, QtWidgets.QPushButton):
                widget.setEnabled(has_selection)

    def _update_highlight_from_selection(self, label_id: str) -> None:
        if not self.current_unit_id:
            return
        highlight = self._selected_highlight(label_id)
        if not highlight:
            return
        doc_id = self.get_active_doc_id()
        if not doc_id:
            QtWidgets.QMessageBox.information(
                self,
                "Highlight",
                "Select a note before updating highlights",
            )
            return
        cursor = self.get_selection()
        if cursor.isNull() or cursor.selectionStart() == cursor.selectionEnd():
            QtWidgets.QMessageBox.information(
                self,
                "Highlight",
                "Select text in the note first",
            )
            return
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        snippet = cursor.selectedText()
        rationale_id = str(highlight.get("rationale_id", ""))
        if not rationale_id:
            return
        updated = self.ctx.update_rationale(
            self.current_unit_id,
            label_id,
            rationale_id,
            doc_id,
            start,
            end,
            snippet,
        )
        if not updated:
            QtWidgets.QMessageBox.warning(
                self,
                "Highlight",
                "Unable to update the selected highlight.",
            )
            return
        self.current_rationales = self.ctx.load_rationales(self.current_unit_id)
        self._refresh_highlights(label_id)
        self._select_highlight(label_id, rationale_id)

    def _delete_highlight(self, label_id: str) -> None:
        if not self.current_unit_id:
            return
        highlight = self._selected_highlight(label_id)
        if not highlight:
            return
        rationale_id = str(highlight.get("rationale_id", ""))
        if not rationale_id:
            return
        response = QtWidgets.QMessageBox.question(
            self,
            "Delete highlight",
            "Remove the selected highlight?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if response != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        removed = self.ctx.delete_rationale(self.current_unit_id, label_id, rationale_id)
        if not removed:
            QtWidgets.QMessageBox.warning(
                self,
                "Highlight",
                "Unable to delete the selected highlight.",
            )
            return
        self.current_rationales = self.ctx.load_rationales(self.current_unit_id)
        self._refresh_highlights(label_id)

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
        if "unknown_box" in widgets and annotation.get("value") == "unknown":
            unknown_box: QtWidgets.QCheckBox = widgets["unknown_box"]  # type: ignore[assignment]
            unknown_box.setChecked(True)
            line_widget = widgets.get("line_edit")
            if isinstance(line_widget, QtWidgets.QLineEdit):
                line_widget.clear()
                line_widget.setEnabled(False)
            date_widget = widgets.get("date_edit")
            if isinstance(date_widget, QtWidgets.QDateEdit):
                date_widget.setEnabled(False)
        if "date_edit" in widgets:
            date_widget: QtWidgets.QDateEdit = widgets["date_edit"]  # type: ignore[assignment]
            if annotation.get("value_date"):
                date_widget.setDate(QtCore.QDate.fromString(annotation["value_date"], QtCore.Qt.ISODate))
            elif annotation.get("value") != "unknown":
                date_widget.setEnabled(True)
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
            widgets["line_edit"].setEnabled(True)  # type: ignore[index]
        if "date_edit" in widgets:
            widgets["date_edit"].setDate(QtCore.QDate.currentDate())  # type: ignore[index]
            widgets["date_edit"].setEnabled(True)  # type: ignore[index]
        if "text_edit" in widgets:
            widgets["text_edit"].clear()  # type: ignore[index]
        if "na_box" in widgets:
            widgets["na_box"].setChecked(False)  # type: ignore[index]
        if "unknown_box" in widgets:
            widgets["unknown_box"].setChecked(False)  # type: ignore[index]
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
            unknown_box = state.get("unknown_box")
            if isinstance(unknown_box, QtWidgets.QCheckBox):
                widgets.append(unknown_box)
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
        widgets = self.label_widgets.get(label_id, {})
        unknown_box = widgets.get("unknown_box")
        if isinstance(unknown_box, QtWidgets.QCheckBox) and unknown_box.isChecked():
            return
        text = widget.text().strip()
        payload = {"value": text, "value_num": float(text) if text else None}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_date(self, label_id: str, widget: QtWidgets.QDateEdit) -> None:
        if not self.current_unit_id:
            return
        widgets = self.label_widgets.get(label_id, {})
        unknown_box = widgets.get("unknown_box")
        if isinstance(unknown_box, QtWidgets.QCheckBox) and unknown_box.isChecked():
            return
        payload = {"value_date": widget.date().toString(QtCore.Qt.ISODate)}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_numeric_unknown(
        self,
        label_id: str,
        line_widget: QtWidgets.QLineEdit,
        checkbox: QtWidgets.QCheckBox,
    ) -> None:
        checked = checkbox.isChecked()
        line_widget.setEnabled(not checked)
        if not self.current_unit_id:
            return
        if checked:
            line_widget.clear()
            payload = {"value": "unknown", "value_num": None}
        else:
            text = line_widget.text().strip()
            payload = {"value": text, "value_num": float(text) if text else None}
        self.ctx.save_annotation(self.current_unit_id, label_id, payload)
        self._update_gating()
        self._update_completion()

    def _on_date_unknown(
        self,
        label_id: str,
        date_widget: QtWidgets.QDateEdit,
        checkbox: QtWidgets.QCheckBox,
    ) -> None:
        checked = checkbox.isChecked()
        date_widget.setEnabled(not checked)
        if not self.current_unit_id:
            return
        if checked:
            payload = {"value": "unknown", "value_date": None}
        else:
            payload = {"value": None, "value_date": date_widget.date().toString(QtCore.Qt.ISODate)}
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
        new_id = self.ctx.save_rationale(self.current_unit_id, label_id, doc_id, start, end, snippet)
        self.current_rationales = self.ctx.load_rationales(self.current_unit_id)
        self._refresh_highlights(label_id)
        if new_id:
            self._select_highlight(label_id, new_id)

    def _update_gating(self) -> None:
        values = self._current_values()
        for label_id, widgets in self.label_widgets.items():
            definition: LabelDefinition = widgets["definition"]  # type: ignore[index]
            row_widget = widgets.get("row_widget")
            visible = self._is_label_visible(definition, values)
            if isinstance(row_widget, QtWidgets.QWidget):
                row_widget.setVisible(visible)
        self._update_completion()

    def _current_values(self) -> Dict[str, object]:
        values: Dict[str, object] = {}

        def _record_value(definition: LabelDefinition, value: object) -> None:
            keys = {definition.name}
            if definition.label_id:
                keys.add(definition.label_id)
            for key in keys:
                values[key] = value

        for label_id, widgets in self.label_widgets.items():
            definition: LabelDefinition = widgets["definition"]  # type: ignore[index]
            if "button_group" in widgets:
                group: QtWidgets.QButtonGroup = widgets["button_group"]  # type: ignore[assignment]
                for button in group.buttons():
                    if button.isChecked():
                        _record_value(definition, button.property("option_value"))
                        break
            if "checkboxes" in widgets:
                selected = [
                    cb.property("option_value")
                    for cb in widgets["checkboxes"]
                    if cb.isChecked()
                ]  # type: ignore[index]
                _record_value(definition, selected)
            if "line_edit" in widgets:
                _record_value(definition, widgets["line_edit"].text())  # type: ignore[index]
            if "text_edit" in widgets:
                _record_value(definition, widgets["text_edit"].toPlainText())  # type: ignore[index]
        return values

    def _is_label_visible(self, definition: LabelDefinition, values: Dict[str, object]) -> bool:
        expr = definition.gating_expr
        if not expr:
            return True
        try:
            field, expected = expr.split("==")
        except ValueError:
            return True
        field = field.strip()
        expected = expected.strip().strip("'\"")
        value = values.get(field, "")
        if isinstance(value, list):
            return expected in value
        return str(value) == expected

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
    DOCUMENT_METADATA_PRIORITY = [
        "patient_icn",
        "date_note",
        "note_year",
        "notetype",
        "sta3n",
        "cptname",
        "softlabel",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.ctx = AssignmentContext()
        self.setWindowTitle("VAAnnotate Client")
        self.resize(1400, 900)
        self.current_documents: List[Dict[str, object]] = []
        self.active_doc_id: Optional[str] = None
        self._last_project_dir: Optional[Path] = None
        self._current_project_root: Optional[Path] = None
        self._current_assignment: Optional[AssignmentSummary] = None
        self._note_font_size = 12
        self._setup_menu()
        self._setup_ui()
        self.ctx.assignment_loaded.connect(self._on_assignment_loaded)
        self.ctx.unit_changed.connect(self._load_unit)
        self.ctx.save_state_changed.connect(self.statusBar().showMessage)

    def _setup_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        open_action = file_menu.addAction("Open project…")
        open_action.triggered.connect(self._open_project)
        self.open_assignment_action = file_menu.addAction("Open assignment…")
        self.open_assignment_action.setEnabled(False)
        self.open_assignment_action.triggered.connect(self._open_assignment_within_project)
        self.save_action = QtGui.QAction("Save changes", self)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._save_pending_changes)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def _setup_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.unit_list = QtWidgets.QListWidget()
        self.unit_list.itemSelectionChanged.connect(self._unit_selected)
        splitter.addWidget(self.unit_list)

        middle_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.notes_table = QtWidgets.QTableWidget()
        self.notes_table.setColumnCount(0)
        self.notes_table.verticalHeader().setVisible(False)
        self.notes_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.notes_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.notes_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.notes_table.itemSelectionChanged.connect(self._on_document_selected)
        self.notes_table.setSortingEnabled(True)
        self.notes_table.horizontalHeader().setStretchLastSection(True)
        middle_splitter.addWidget(self.notes_table)

        note_panel = QtWidgets.QWidget()
        note_panel_layout = QtWidgets.QVBoxLayout(note_panel)
        note_panel_layout.setContentsMargins(0, 0, 0, 0)
        note_panel_layout.setSpacing(6)
        self.note_view = QtWidgets.QTextEdit()
        self.note_view.setReadOnly(True)
        note_font = self.note_view.font()
        note_font.setPointSize(self._note_font_size)
        self.note_view.setFont(note_font)
        self.note_view.document().setDefaultFont(note_font)
        note_panel_layout.addWidget(self.note_view)

        font_controls = QtWidgets.QHBoxLayout()
        font_label = QtWidgets.QLabel("Text size:")
        font_controls.addWidget(font_label)
        self.note_font_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.note_font_slider.setRange(8, 32)
        self.note_font_slider.setSingleStep(1)
        self.note_font_slider.setValue(self._note_font_size)
        self.note_font_slider.valueChanged.connect(self._on_note_font_size_changed)
        font_controls.addWidget(self.note_font_slider, 1)
        self.note_font_value = QtWidgets.QLabel(f"{self._note_font_size} pt")
        font_controls.addWidget(self.note_font_value)
        font_controls.addStretch()
        note_panel_layout.addLayout(font_controls)

        info_widget = QtWidgets.QWidget()
        info_layout = QtWidgets.QHBoxLayout(info_widget)
        self.progress_label = QtWidgets.QLabel("Progress: 0/0")
        info_layout.addWidget(self.progress_label)
        self.assignment_label = QtWidgets.QLabel("")
        self.assignment_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        info_layout.addWidget(self.assignment_label, 1)
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
        nav_toolbar.addAction(self.save_action)
        nav_toolbar.addAction(submit_action)

    def _on_note_font_size_changed(self, value: int) -> None:
        self._note_font_size = int(value)
        font = self.note_view.font()
        font.setPointSize(self._note_font_size)
        self.note_view.setFont(font)
        self.note_view.document().setDefaultFont(font)
        if hasattr(self, "note_font_value"):
            self.note_font_value.setText(f"{self._note_font_size} pt")

    def _open_project(self) -> None:
        start_dir = str(self._last_project_dir) if self._last_project_dir else str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select project folder",
            start_dir,
        )
        if not directory:
            return
        project_root = Path(directory)
        self._launch_assignment_picker_for_project(project_root)

    def _open_assignment_within_project(self) -> None:
        if not self._current_project_root:
            return
        self._launch_assignment_picker_for_project(self._current_project_root)

    def _launch_assignment_picker_for_project(self, project_root: Path) -> None:
        try:
            browser = ProjectBrowser(project_root)
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self,
                "Open project",
                "The selected folder does not appear to be a VAAnnotate project (project.db not found).",
            )
            return
        picker = AssignmentPickerDialog(browser, self)
        if picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        assignment = picker.selected_assignment()
        if not assignment:
            return
        reviewer = picker.selected_reviewer()
        reviewer_label = reviewer.display_label() if reviewer else ""
        self._load_assignment(project_root, assignment, reviewer_label)

    def _load_assignment(
        self,
        project_root: Path,
        assignment: AssignmentSummary,
        reviewer_label: str,
    ) -> None:
        if self.ctx.has_unsaved_changes():
            if not self._prompt_save_changes():
                return
        try:
            self.ctx.open_assignment(assignment.assignment_dir)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self,
                "Open assignment",
                f"Failed to open assignment: {exc}",
            )
            return
        self._last_project_dir = project_root
        self._current_project_root = project_root
        self._current_assignment = assignment
        self.open_assignment_action.setEnabled(True)
        self.save_action.setEnabled(True)
        self._update_assignment_info_label()
        status_parts = [assignment.phenotype_name, assignment.round_label()]
        if reviewer_label:
            status_parts.append(f"Reviewer: {reviewer_label}")
        self.statusBar().showMessage(" – ".join(status_parts))

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

    def _collect_document_metadata_keys(self, documents: List[Dict[str, object]]) -> List[str]:
        discovered: List[str] = []
        for doc in documents:
            metadata = doc.get("metadata")
            if isinstance(metadata, dict):
                for key in metadata.keys():
                    if key not in discovered:
                        discovered.append(str(key))
        priority = [key for key in self.DOCUMENT_METADATA_PRIORITY if key in discovered]
        remaining = sorted(
            [key for key in discovered if key not in self.DOCUMENT_METADATA_PRIORITY],
            key=str.lower,
        )
        return priority + remaining

    def _format_metadata_header(self, key: str) -> str:
        parts = key.split("_")
        formatted: List[str] = []
        for part in parts:
            token = part.strip()
            if not token:
                continue
            upper_token = token.upper()
            if upper_token in {"ICN", "ID", "VA", "HIN"}:
                formatted.append(upper_token)
            elif token.lower() == "sta3n":
                formatted.append("STA3N")
            elif token.isupper():
                formatted.append(token)
            else:
                formatted.append(token.capitalize())
        return " ".join(formatted) if formatted else key

    def _format_metadata_value(self, value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else "—"
        if isinstance(value, (list, tuple, set)):
            items = [self._format_metadata_value(item) for item in value]
            return ", ".join(item for item in items if item != "—") or "—"
        return str(value)

    def _on_assignment_loaded(self) -> None:
        self.save_action.setEnabled(True)
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
        self._update_assignment_info_label()

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
            fallback_text = self.ctx.fetch_document(fallback_doc_id, unit_id)
            fallback_metadata = self.ctx.document_metadata(fallback_doc_id)
            documents = [
                {
                    "order_index": 1,
                    "doc_id": fallback_doc_id,
                    "text": fallback_text,
                    "metadata": fallback_metadata,
                }
            ]
            self.ctx.cache_unit_documents(unit_id, documents)
            self._populate_notes_table(documents)
        if self.notes_table.rowCount():
            self.notes_table.setCurrentCell(0, 0)
        annotations = self.ctx.load_annotations(str(unit["unit_id"]))
        rationales = self.ctx.load_rationales(str(unit["unit_id"]))
        self.form.load_unit(unit_id, annotations, rationales)
        self._update_progress()

    def _populate_notes_table(self, documents: List[Dict[str, object]]) -> None:
        self.current_documents = documents
        self.active_doc_id = None
        metadata_keys = self._collect_document_metadata_keys(documents)
        headers = ["#", "Document ID"] + [
            self._format_metadata_header(key) for key in metadata_keys
        ] + ["Preview"]
        self.notes_table.blockSignals(True)
        self.notes_table.setSortingEnabled(False)
        self.notes_table.clear()
        self.notes_table.setColumnCount(len(headers))
        self.notes_table.setHorizontalHeaderLabels(headers)
        self.notes_table.setRowCount(len(documents))
        metadata_offset = 2
        preview_column = len(headers) - 1
        for row_index, doc in enumerate(documents):
            order_value = doc.get("order_index", row_index + 1)
            order_item = QtWidgets.QTableWidgetItem(str(order_value))
            order_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc)
            order_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.notes_table.setItem(row_index, 0, order_item)

            doc_id_item = QtWidgets.QTableWidgetItem(str(doc.get("doc_id", "")))
            doc_id_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc)
            doc_id_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.notes_table.setItem(row_index, 1, doc_id_item)

            metadata = doc.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            for offset, key in enumerate(metadata_keys, start=metadata_offset):
                value = metadata.get(key)
                display_value = self._format_metadata_value(value)
                item = QtWidgets.QTableWidgetItem(display_value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, doc)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.notes_table.setItem(row_index, offset, item)

            preview_text = str(doc.get("text", ""))[:200].replace("\n", " ")
            preview_item = QtWidgets.QTableWidgetItem(preview_text)
            preview_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc)
            preview_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.notes_table.setItem(row_index, preview_column, preview_item)
        self.notes_table.setSortingEnabled(True)
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

    def _save_pending_changes(self) -> None:
        if not self.ctx.assignment_db:
            return
        if not self.ctx.has_unsaved_changes():
            self.statusBar().showMessage("No changes to save.")
            return
        if self._attempt_save_changes():
            self.statusBar().showMessage("All changes saved.")

    def _attempt_save_changes(self) -> bool:
        try:
            if not self.ctx.has_unsaved_changes():
                return True
            return self.ctx.flush_pending_writes()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self,
                "Save changes",
                f"Failed to save changes: {exc}",
            )
            return False

    def _prompt_save_changes(self) -> bool:
        response = QtWidgets.QMessageBox.question(
            self,
            "Unsaved changes",
            "You have unsaved changes. What would you like to do?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if response == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if response == QtWidgets.QMessageBox.StandardButton.Discard:
            self.ctx.discard_pending_changes()
            self._update_progress()
            return True
        return self._attempt_save_changes()

    def _submit_assignment(self) -> None:
        if not self.ctx.assignment_path:
            return
        if self.ctx.has_unsaved_changes() and not self._attempt_save_changes():
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
        status_map: Dict[str, int] = {
            str(unit["unit_id"]): 1 if unit.get("complete") else 0 for unit in self.ctx.units
        }
        completed = sum(1 for value in status_map.values() if value)
        for idx in range(self.unit_list.count()):
            item = self.unit_list.item(idx)
            unit = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not unit:
                continue
            unit_id = str(unit["unit_id"])
            unit_complete = bool(status_map.get(unit_id))
            unit["complete"] = 1 if unit_complete else 0
            suffix = " ✓" if unit_complete else ""
            item.setText(f"{unit['display_rank']}: {self._format_unit_label(unit)}{suffix}")
        self.progress_label.setText(f"Progress: {completed}/{total}")
        self._update_assignment_info_label()

    def _update_assignment_info_label(self) -> None:
        if not self._current_assignment:
            self.assignment_label.clear()
            return
        phenotype = self._current_assignment.phenotype_name
        round_label = self._current_assignment.round_label()
        self.assignment_label.setText(f"Phenotype: {phenotype} | {round_label}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self.ctx.has_unsaved_changes():
            if not self._prompt_save_changes():
                event.ignore()
                return
        super().closeEvent(event)


def run(path: Optional[str] = None) -> None:
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)
    window = ClientMainWindow()
    window.show()
    if path:
        window.ctx.open_assignment(Path(path))
    sys.exit(app.exec())


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
