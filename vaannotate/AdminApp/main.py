"""PySide6 based Admin application for VAAnnotate."""
from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets

from ..shared import models
from ..shared.database import Database, ensure_schema
from ..shared.sampling import (
    SamplingFilters,
    allocate_units,
    candidate_documents,
    initialize_assignment_db,
    populate_assignment_db,
    write_manifest,
)
from ..utils import copy_sqlite_database, ensure_dir
from ..shared.statistics import cohens_kappa, fleiss_kappa, percent_agreement


PROJECT_MODELS = [
    models.Project,
    models.Phenotype,
    models.LabelSet,
    models.Label,
    models.LabelOption,
    models.Round,
    models.RoundConfig,
    models.Reviewer,
    models.Assignment,
]


class ProjectContext(QtCore.QObject):
    project_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.project_root: Optional[Path] = None
        self.project_db: Optional[Database] = None
        self.corpus_db: Optional[Database] = None

    def open_project(self, directory: Path) -> None:
        directory = directory.resolve()
        project_db = Database(directory / "project.db")
        with project_db.transaction() as conn:
            ensure_schema(conn, PROJECT_MODELS)
        corpus_dir = directory / "corpus"
        corpus_dir.mkdir(exist_ok=True)
        corpus_db = Database(corpus_dir / "corpus.db")
        with corpus_db.transaction() as conn:
            ensure_schema(conn, [models.Patient, models.Document])
        self.project_root = directory
        self.project_db = project_db
        self.corpus_db = corpus_db
        self.project_changed.emit()

    def require_project(self) -> Path:
        if not self.project_root:
            raise RuntimeError("No project opened")
        return self.project_root

    def require_db(self) -> Database:
        if not self.project_db:
            raise RuntimeError("Project database not initialized")
        return self.project_db

    def require_corpus_db(self) -> Database:
        if not self.corpus_db:
            raise RuntimeError("Corpus database not initialized")
        return self.corpus_db


class ProjectsPage(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext) -> None:
        super().__init__()
        self.ctx = ctx
        self._setup_ui()
        self.ctx.project_changed.connect(self.refresh)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.project_list = QtWidgets.QListWidget()
        self.project_list.currentItemChanged.connect(self._on_project_selected)
        splitter.addWidget(self.project_list)

        detail_widget = QtWidgets.QWidget()
        splitter.addWidget(detail_widget)
        form = QtWidgets.QFormLayout(detail_widget)

        self.name_edit = QtWidgets.QLineEdit()
        self.created_by_edit = QtWidgets.QLineEdit()
        self.create_btn = QtWidgets.QPushButton("Create Project")
        self.create_btn.clicked.connect(self._create_project)

        form.addRow("Project name", self.name_edit)
        form.addRow("Created by", self.created_by_edit)
        form.addRow(self.create_btn)

        self.meta_view = QtWidgets.QTextEdit()
        self.meta_view.setReadOnly(True)
        form.addRow("Details", self.meta_view)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

    def refresh(self) -> None:
        self.project_list.clear()
        try:
            db = self.ctx.require_db()
        except RuntimeError:
            return
        with db.connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        for row in rows:
            item = QtWidgets.QListWidgetItem(f"{row['name']} ({row['project_id']})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.project_list.addItem(item)

    def _create_project(self) -> None:
        if not self.name_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "Missing data", "Name is required")
            return
        created_by = self.created_by_edit.text().strip() or "unknown"
        project = models.Project(
            project_id=str(uuid.uuid4()),
            name=self.name_edit.text().strip(),
            created_at=QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate),
            created_by=created_by,
        )
        db = self.ctx.require_db()
        with db.transaction() as conn:
            project.save(conn)
        self.name_edit.clear()
        self.refresh()

    def _on_project_selected(self, current: QtWidgets.QListWidgetItem) -> None:
        if not current:
            self.meta_view.clear()
            return
        row = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self.meta_view.setPlainText(json.dumps({k: row[k] for k in row.keys()}, indent=2))


class PhenotypePage(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext) -> None:
        super().__init__()
        self.ctx = ctx
        self._setup_ui()
        self.ctx.project_changed.connect(self.refresh)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.list_widget = QtWidgets.QListWidget()
        splitter.addWidget(self.list_widget)

        right = QtWidgets.QWidget()
        splitter.addWidget(right)
        form = QtWidgets.QFormLayout(right)

        self.project_combo = QtWidgets.QComboBox()
        self.name_edit = QtWidgets.QLineEdit()
        self.level_combo = QtWidgets.QComboBox()
        self.level_combo.addItems(["single_doc", "multi_doc"])
        self.description_edit = QtWidgets.QPlainTextEdit()
        self.save_btn = QtWidgets.QPushButton("Create phenotype")
        self.save_btn.clicked.connect(self._create_phenotype)

        form.addRow("Project", self.project_combo)
        form.addRow("Name", self.name_edit)
        form.addRow("Level", self.level_combo)
        form.addRow("Description", self.description_edit)
        form.addRow(self.save_btn)

    def refresh(self) -> None:
        db = self.ctx.require_db()
        with db.connect() as conn:
            projects = conn.execute("SELECT project_id, name FROM projects ORDER BY name").fetchall()
        self.project_combo.clear()
        for project in projects:
            self.project_combo.addItem(project["name"], project["project_id"])
        self._reload_phenotypes()

    def _reload_phenotypes(self) -> None:
        db = self.ctx.require_db()
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT phenotypes.*, projects.name as project_name FROM phenotypes "
                "JOIN projects ON projects.project_id = phenotypes.project_id"
            ).fetchall()
        self.list_widget.clear()
        for row in rows:
            label = f"{row['name']} ({row['level']}) - {row['project_name']}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.list_widget.addItem(item)

    def _create_phenotype(self) -> None:
        if self.project_combo.currentIndex() < 0:
            QtWidgets.QMessageBox.warning(self, "Missing project", "Select a project first")
            return
        pheno = models.Phenotype(
            pheno_id=str(uuid.uuid4()),
            project_id=self.project_combo.currentData(),
            name=self.name_edit.text().strip(),
            level=self.level_combo.currentText(),
            description=self.description_edit.toPlainText().strip() or "",
        )
        db = self.ctx.require_db()
        with db.transaction() as conn:
            pheno.save(conn)
        self.name_edit.clear()
        self.description_edit.clear()
        self._reload_phenotypes()


class CorpusOverviewPage(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext) -> None:
        super().__init__()
        self.ctx = ctx
        self._setup_ui()
        self.ctx.project_changed.connect(self.refresh)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel("Open a project to preview corpus contents.")
        layout.addWidget(self.summary_label)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Doc ID",
            "Patient ICN",
            "Note type",
            "Date",
            "Preview",
        ])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def refresh(self) -> None:
        try:
            db = self.ctx.require_corpus_db()
        except RuntimeError:
            self.summary_label.setText("Open a project to preview corpus contents.")
            self.table.setRowCount(0)
            return

        with db.connect() as conn:
            patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            rows = conn.execute(
                "SELECT doc_id, patient_icn, notetype, date_note, substr(text, 1, 200) AS preview "
                "FROM documents ORDER BY date_note DESC LIMIT 50"
            ).fetchall()

        self.summary_label.setText(
            f"Patients: {patient_count:,} • Documents: {document_count:,} • Showing {len(rows)} most recent notes"
        )
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["doc_id"],
                row["patient_icn"],
                row["notetype"],
                row["date_note"],
                (row["preview"] or "").replace("\n", " ") + ("…" if row["preview"] and len(row["preview"]) == 200 else ""),
            ]
            for col_index, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.table.setItem(row_index, col_index, item)
        self.table.resizeColumnsToContents()


class RoundPage(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext) -> None:
        super().__init__()
        self.ctx = ctx
        self._setup_ui()
        self.ctx.project_changed.connect(self.refresh)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.round_list = QtWidgets.QListWidget()
        self.round_list.itemSelectionChanged.connect(self._load_round_config)
        splitter.addWidget(self.round_list)

        right = QtWidgets.QWidget()
        splitter.addWidget(right)
        right_layout = QtWidgets.QVBoxLayout(right)

        round_group = QtWidgets.QGroupBox("Round setup")
        round_form = QtWidgets.QFormLayout(round_group)

        self.pheno_combo = QtWidgets.QComboBox()
        self.labelset_edit = QtWidgets.QLineEdit()
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setMaximum(2**31 - 1)
        self.overlap_spin = QtWidgets.QSpinBox()
        self.overlap_spin.setRange(0, 1000)
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setRange(1, 10000)
        self.status_combo = QtWidgets.QComboBox()
        self.status_combo.addItems(["draft", "active", "closed", "adjudicating", "finalized"])
        self.create_btn = QtWidgets.QPushButton("Generate round")
        self.create_btn.clicked.connect(self._create_round)

        round_form.addRow("Phenotype", self.pheno_combo)
        round_form.addRow("Label set ID", self.labelset_edit)
        round_form.addRow("Seed", self.seed_spin)
        round_form.addRow("Overlap N", self.overlap_spin)
        round_form.addRow("Sample per reviewer", self.sample_spin)
        round_form.addRow("Status", self.status_combo)
        round_form.addRow(self.create_btn)
        right_layout.addWidget(round_group)

        filter_group = QtWidgets.QGroupBox("Sampling filters")
        filter_form = QtWidgets.QFormLayout(filter_group)
        self.patient_sta3n_edit = QtWidgets.QLineEdit()
        self.patient_sta3n_edit.setPlaceholderText("Comma-separated STA3N codes")
        patient_years_layout = QtWidgets.QHBoxLayout()
        self.patient_year_start = QtWidgets.QSpinBox()
        self.patient_year_start.setRange(0, 2100)
        self.patient_year_start.setSpecialValueText("Any")
        self.patient_year_start.setValue(0)
        self.patient_year_end = QtWidgets.QSpinBox()
        self.patient_year_end.setRange(0, 2100)
        self.patient_year_end.setSpecialValueText("Any")
        self.patient_year_end.setValue(0)
        patient_years_layout.addWidget(self.patient_year_start)
        patient_years_layout.addWidget(QtWidgets.QLabel("to"))
        patient_years_layout.addWidget(self.patient_year_end)
        self.patient_softlabel_spin = QtWidgets.QDoubleSpinBox()
        self.patient_softlabel_spin.setRange(-1.0, 100.0)
        self.patient_softlabel_spin.setDecimals(2)
        self.patient_softlabel_spin.setSpecialValueText("Disabled")
        self.patient_softlabel_spin.setValue(-1.0)
        self.note_type_edit = QtWidgets.QLineEdit()
        self.note_type_edit.setPlaceholderText("Comma-separated note types")
        note_years_layout = QtWidgets.QHBoxLayout()
        self.note_year_start = QtWidgets.QSpinBox()
        self.note_year_start.setRange(0, 2100)
        self.note_year_start.setSpecialValueText("Any")
        self.note_year_start.setValue(0)
        self.note_year_end = QtWidgets.QSpinBox()
        self.note_year_end.setRange(0, 2100)
        self.note_year_end.setSpecialValueText("Any")
        self.note_year_end.setValue(0)
        note_years_layout.addWidget(self.note_year_start)
        note_years_layout.addWidget(QtWidgets.QLabel("to"))
        note_years_layout.addWidget(self.note_year_end)
        self.note_regex_edit = QtWidgets.QLineEdit()
        self.note_regex_edit.setPlaceholderText("Python regex applied to note text")

        filter_form.addRow("Patient STA3N", self.patient_sta3n_edit)
        filter_form.addRow("Patient year range", patient_years_layout)
        filter_form.addRow("Patient softlabel ≥", self.patient_softlabel_spin)
        filter_form.addRow("Note types", self.note_type_edit)
        filter_form.addRow("Note year range", note_years_layout)
        filter_form.addRow("Note regex", self.note_regex_edit)
        right_layout.addWidget(filter_group)

        strat_group = QtWidgets.QGroupBox("Stratification")
        strat_form = QtWidgets.QFormLayout(strat_group)
        self.strat_keys_edit = QtWidgets.QLineEdit()
        self.strat_keys_edit.setPlaceholderText("Comma-separated document fields (e.g. note_year, sta3n)")
        self.strat_sample_spin = QtWidgets.QSpinBox()
        self.strat_sample_spin.setRange(0, 10000)
        self.strat_sample_spin.setSpecialValueText("Use full strata")
        self.strat_sample_spin.setValue(0)
        strat_form.addRow("Stratify by", self.strat_keys_edit)
        strat_form.addRow("Sample per stratum", self.strat_sample_spin)
        right_layout.addWidget(strat_group)

        right_layout.addWidget(QtWidgets.QLabel("Selected round configuration"))
        self.round_config_view = QtWidgets.QTextEdit()
        self.round_config_view.setReadOnly(True)
        self.round_config_view.setPlaceholderText("Select an existing round to inspect its configuration")
        right_layout.addWidget(self.round_config_view)
        right_layout.addStretch()

    def refresh(self) -> None:
        db = self.ctx.require_db()
        with db.connect() as conn:
            phenos = conn.execute("SELECT pheno_id, name FROM phenotypes ORDER BY name").fetchall()
            rounds = conn.execute("SELECT * FROM rounds ORDER BY created_at DESC").fetchall()
        self.pheno_combo.clear()
        for pheno in phenos:
            self.pheno_combo.addItem(pheno["name"], pheno["pheno_id"])
        self.round_list.clear()
        for row in rounds:
            item = QtWidgets.QListWidgetItem(f"{row['round_number']} - {row['round_id']}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.round_list.addItem(item)
        if self.round_list.count():
            self.round_list.setCurrentRow(0)
        else:
            self.round_config_view.clear()

    def _create_round(self) -> None:
        if self.pheno_combo.currentIndex() < 0:
            QtWidgets.QMessageBox.warning(self, "Phenotype missing", "Select a phenotype")
            return
        ctx = self.ctx
        db = ctx.require_db()
        pheno_id = self.pheno_combo.currentData()
        seed = self.seed_spin.value()
        overlap = self.overlap_spin.value()
        reviewers = self._prompt_reviewers()
        if not reviewers:
            return
        labelset_id = self.labelset_edit.text().strip() or f"auto_{pheno_id}"
        created_at = QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)
        default_labels: List[Dict[str, object]] = []
        with db.connect() as conn:
            exists = conn.execute("SELECT 1 FROM label_sets WHERE labelset_id=?", (labelset_id,)).fetchone()
        if not exists:
            default_labels.append(
                {
                    "label_id": str(uuid.uuid4()),
                    "name": "Has_phenotype",
                    "type": "boolean",
                    "required": 1,
                    "options": [
                        {"value": "yes", "display": "Yes"},
                        {"value": "no", "display": "No"},
                        {"value": "unknown", "display": "Unknown"},
                    ],
                }
            )
        filters = self._collect_filters()
        corpus_rows = candidate_documents(ctx.require_corpus_db(), "single_doc", filters)
        if not corpus_rows:
            QtWidgets.QMessageBox.warning(self, "No corpus", "The corpus database has no documents to sample")
            return
        strat_keys = [key.strip() for key in self.strat_keys_edit.text().split(",") if key.strip()]
        strat_sample = self.strat_sample_spin.value()
        if strat_sample <= 0:
            strat_sample = None
        default_sample = self.sample_spin.value() * len(reviewers)
        per_stratum = strat_sample
        if not strat_keys and strat_sample is None and default_sample:
            per_stratum = default_sample
        assignments = allocate_units(
            corpus_rows,
            reviewers,
            overlap,
            seed,
            strat_keys=strat_keys or None,
            per_stratum=per_stratum,
        )
        target_sample = self.sample_spin.value()
        missing = [
            reviewer["id"]
            for reviewer in reviewers
            if target_sample and len(assignments[reviewer["id"]].units) < target_sample
        ]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Sample size",
                "Not enough candidate documents were found to satisfy the desired sample size for:"
                f" {', '.join(missing)}",
            )
        round_id = str(uuid.uuid4())
        round_number = self._next_round_number(pheno_id)
        round_record = models.Round(
            round_id=round_id,
            pheno_id=pheno_id,
            round_number=round_number,
            labelset_id=labelset_id,
            config_hash=str(uuid.uuid4()),
            rng_seed=seed,
            status=self.status_combo.currentText(),
            created_at=created_at,
        )
        manifest_dir = ctx.require_project() / "phenotypes" / pheno_id / f"rounds/{round_number}"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest_dir / "manifest.csv", assignments)
        with db.transaction() as conn:
            if default_labels:
                labelset = models.LabelSet(
                    labelset_id=labelset_id,
                    pheno_id=pheno_id,
                    version=1,
                    created_at=created_at,
                    created_by="system",
                    notes="Auto-generated",
                )
                labelset.save(conn)
                for label in default_labels:
                    label_record = models.Label(
                        label_id=label["label_id"],
                        labelset_id=labelset_id,
                        name=label["name"],
                        type=label["type"],
                        required=label["required"],
                        order_index=0,
                        rules="",
                        gating_expr=None,
                        na_allowed=0,
                        unit=None,
                        min=None,
                        max=None,
                    )
                    label_record.save(conn)
                    for idx, option in enumerate(label["options"]):
                        option_record = models.LabelOption(
                            option_id=str(uuid.uuid4()),
                            label_id=label_record.label_id,
                            value=option["value"],
                            display=option["display"],
                            order_index=idx,
                            weight=None,
                        )
                        option_record.save(conn)
            round_record.save(conn)
            config_payload: Dict[str, object] = {
                "pheno_id": pheno_id,
                "labelset_id": labelset_id,
                "round_number": round_number,
                "round_id": round_id,
                "rng_seed": seed,
                "overlap_n": overlap,
                "sample_per_reviewer": self.sample_spin.value(),
                "status": self.status_combo.currentText(),
                "reviewers": reviewers,
            }
            filters_payload: Dict[str, Dict[str, object]] = {}
            if filters.patient_filters:
                filters_payload["patient"] = filters.patient_filters
            if filters.note_filters:
                filters_payload["note"] = filters.note_filters
            if filters_payload:
                config_payload["filters"] = filters_payload
            if strat_keys or strat_sample is not None:
                strat_payload: Dict[str, object] = {}
                if strat_keys:
                    strat_payload["keys"] = strat_keys
                if strat_sample is not None:
                    strat_payload["sample_per_stratum"] = strat_sample
                config_payload["stratification"] = strat_payload
            config = models.RoundConfig(round_id=round_id, config_json=json.dumps(config_payload, indent=2))
            config.save(conn)
            for reviewer in reviewers:
                reviewer_record = models.Reviewer(
                    reviewer_id=reviewer["id"],
                    name=reviewer.get("name", reviewer["id"]),
                    email=reviewer.get("email", ""),
                    windows_account=None,
                )
                reviewer_record.save(conn)
            for reviewer in reviewers:
                assignment = models.Assignment(
                    assign_id=str(uuid.uuid4()),
                    round_id=round_id,
                    reviewer_id=reviewer["id"],
                    sample_size=len(assignments[reviewer["id"]].units),
                    overlap_n=overlap,
                    created_at=round_record.created_at,
                    status="open",
                )
                assignment.save(conn)
        for reviewer in reviewers:
            assignment_dir = manifest_dir / "assignments" / reviewer["id"]
            assignment_dir.mkdir(parents=True, exist_ok=True)
            db_path = assignment_dir / "assignment.db"
            assignment_db = initialize_assignment_db(db_path)
            populate_assignment_db(assignment_db, reviewer["id"], assignments[reviewer["id"]].units)
            label_schema = self._build_label_schema(labelset_id, db)
            schema_path = assignment_dir / "label_schema.json"
            schema_path.write_text(json.dumps(label_schema, indent=2), encoding="utf-8")
        self.refresh()

    def _collect_filters(self) -> SamplingFilters:
        patient_filters: Dict[str, object] = {}
        note_filters: Dict[str, object] = {}
        sta_text = self.patient_sta3n_edit.text().strip()
        if sta_text:
            values = [value.strip() for value in sta_text.split(",") if value.strip()]
            if values:
                patient_filters["sta3n_in"] = values
        start = self.patient_year_start.value()
        end = self.patient_year_end.value()
        if start and end:
            if end < start:
                start, end = end, start
            patient_filters["year_range"] = [start, end]
        softlabel = self.patient_softlabel_spin.value()
        if softlabel >= 0:
            patient_filters["softlabel_gte"] = softlabel
        note_types = [value.strip() for value in self.note_type_edit.text().split(",") if value.strip()]
        if note_types:
            note_filters["notetype_in"] = note_types
        note_start = self.note_year_start.value()
        note_end = self.note_year_end.value()
        if note_start and note_end:
            if note_end < note_start:
                note_start, note_end = note_end, note_start
            note_filters["note_year_range"] = [note_start, note_end]
        regex = self.note_regex_edit.text().strip()
        if regex:
            note_filters["regex"] = regex
        return SamplingFilters(patient_filters=patient_filters, note_filters=note_filters)

    def _load_round_config(self) -> None:
        current = self.round_list.currentItem()
        if not current:
            self.round_config_view.clear()
            return
        data = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            self.round_config_view.clear()
            return
        try:
            db = self.ctx.require_db()
        except RuntimeError:
            self.round_config_view.setPlainText("No project loaded.")
            return
        config_text = ""
        with db.connect() as conn:
            row = conn.execute(
                "SELECT config_json FROM round_configs WHERE round_id=?",
                (data["round_id"],),
            ).fetchone()
            if row and row["config_json"]:
                config_text = row["config_json"]
        if not config_text:
            try:
                project_root = self.ctx.require_project()
            except RuntimeError:
                self.round_config_view.setPlainText("Configuration not available.")
                return
            round_number = data["round_number"]
            round_dir = project_root / "phenotypes" / data["pheno_id"] / "rounds" / f"round_{round_number}"
            config_path = round_dir / "round_config.json"
            if config_path.exists():
                config_text = config_path.read_text(encoding="utf-8")
        if config_text:
            try:
                parsed = json.loads(config_text)
                pretty = json.dumps(parsed, indent=2)
                self.round_config_view.setPlainText(pretty)
            except json.JSONDecodeError:
                self.round_config_view.setPlainText(config_text)
        else:
            self.round_config_view.setPlainText("Configuration not found for the selected round.")

    def _prompt_reviewers(self) -> Optional[List[Dict[str, str]]]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Reviewers")
        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Reviewer ID", "Name", "Email"])
        layout.addWidget(table)

        def add_row() -> None:
            row = table.rowCount()
            table.insertRow(row)
            for col in range(3):
                table.setItem(row, col, QtWidgets.QTableWidgetItem(""))

        add_row()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        add_btn = QtWidgets.QPushButton("Add reviewer")
        add_btn.clicked.connect(add_row)
        layout.addWidget(add_btn)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        reviewers: List[Dict[str, str]] = []
        for row in range(table.rowCount()):
            reviewer_id = table.item(row, 0).text().strip()
            name = table.item(row, 1).text().strip()
            email = table.item(row, 2).text().strip()
            if reviewer_id:
                reviewers.append({"id": reviewer_id, "name": name, "email": email})
        return reviewers

    def _next_round_number(self, pheno_id: str) -> int:
        db = self.ctx.require_db()
        with db.connect() as conn:
            row = conn.execute("SELECT MAX(round_number) FROM rounds WHERE pheno_id=?", (pheno_id,)).fetchone()
        return (row[0] or 0) + 1

    def _build_label_schema(self, labelset_id: str, db: Database) -> Dict[str, object]:
        with db.connect() as conn:
            labels = conn.execute("SELECT * FROM labels WHERE labelset_id=? ORDER BY order_index", (labelset_id,)).fetchall()
            options = conn.execute("SELECT * FROM label_options WHERE label_id IN (SELECT label_id FROM labels WHERE labelset_id=?)", (labelset_id,)).fetchall()
        option_map: Dict[str, List[Dict[str, object]]] = {}
        for opt in options:
            option_map.setdefault(opt["label_id"], []).append(
                {
                    "value": opt["value"],
                    "display": opt["display"],
                    "order_index": opt["order_index"],
                    "weight": opt["weight"],
                }
            )
        schema_labels = []
        for label in labels:
            schema_labels.append(
                {
                    "label_id": label["label_id"],
                    "name": label["name"],
                    "type": label["type"],
                    "required": bool(label["required"]),
                    "na_allowed": bool(label["na_allowed"]),
                    "rules": label["rules"],
                    "unit": label["unit"],
                    "range": {"min": label["min"], "max": label["max"]},
                    "gating_expr": label["gating_expr"],
                    "options": sorted(option_map.get(label["label_id"], []), key=lambda o: o["order_index"]),
                }
            )
        return {
            "labelset_id": labelset_id,
            "labels": schema_labels,
        }


class IaaPage(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.current_round: Optional[Dict[str, object]] = None
        self.current_reviewer_names: Dict[str, str] = {}
        self._setup_ui()
        self.ctx.project_changed.connect(self.refresh)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel("Phenotypes"))
        self.pheno_list = QtWidgets.QListWidget()
        self.pheno_list.itemSelectionChanged.connect(self._on_pheno_selected)
        left_layout.addWidget(self.pheno_list)
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self.round_table = QtWidgets.QTableWidget()
        self.round_table.setColumnCount(3)
        self.round_table.setHorizontalHeaderLabels(["Round", "Status", "Reviewers"])
        self.round_table.verticalHeader().setVisible(False)
        self.round_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.round_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.round_table.itemSelectionChanged.connect(self._on_round_selected)
        self.round_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.round_table)

        controls = QtWidgets.QHBoxLayout()
        self.metric_selector = QtWidgets.QComboBox()
        self.metric_selector.addItems(["Percent agreement", "Cohen's kappa", "Fleiss' kappa"])
        self.label_selector = QtWidgets.QComboBox()
        self.label_selector.setPlaceholderText("Select label")
        self.compute_btn = QtWidgets.QPushButton("Calculate agreement")
        self.compute_btn.clicked.connect(self._compute_agreement)
        controls.addWidget(QtWidgets.QLabel("Label:"))
        controls.addWidget(self.label_selector)
        controls.addWidget(QtWidgets.QLabel("Metric:"))
        controls.addWidget(self.metric_selector)
        controls.addWidget(self.compute_btn)
        right_layout.addLayout(controls)

        import_layout = QtWidgets.QHBoxLayout()
        self.auto_import_btn = QtWidgets.QPushButton("Import submitted assignments")
        self.auto_import_btn.clicked.connect(self._on_auto_import_clicked)
        self.manual_reviewer_combo = QtWidgets.QComboBox()
        self.manual_reviewer_combo.setPlaceholderText("Select reviewer")
        self.manual_import_btn = QtWidgets.QPushButton("Import reviewer DB…")
        self.manual_import_btn.clicked.connect(self._manual_import_assignment)
        self.manual_import_btn.setEnabled(False)
        self.auto_import_btn.setEnabled(False)
        import_layout.addWidget(self.auto_import_btn)
        import_layout.addStretch()
        import_layout.addWidget(QtWidgets.QLabel("Reviewer:"))
        import_layout.addWidget(self.manual_reviewer_combo)
        import_layout.addWidget(self.manual_import_btn)
        right_layout.addLayout(import_layout)

        self.import_status_label = QtWidgets.QLabel()
        self.import_status_label.setWordWrap(True)
        right_layout.addWidget(self.import_status_label)

        self.round_summary = QtWidgets.QLabel("Select a round to review agreement metrics")
        self.round_summary.setWordWrap(True)
        right_layout.addWidget(self.round_summary)

        self.results_view = QtWidgets.QTextEdit()
        self.results_view.setReadOnly(True)
        self.results_view.setPlaceholderText("Agreement results will appear here once calculated")
        right_layout.addWidget(self.results_view)

        discord_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.discord_table = QtWidgets.QTableWidget()
        self.discord_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.discord_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.discord_table.itemSelectionChanged.connect(self._on_discord_selected)
        discord_splitter.addWidget(self.discord_table)

        self.discord_note = QtWidgets.QTextEdit()
        self.discord_note.setReadOnly(True)
        self.discord_note.setPlaceholderText("Select a discordant unit to review the source note text")
        discord_splitter.addWidget(self.discord_note)
        discord_splitter.setStretchFactor(0, 3)
        discord_splitter.setStretchFactor(1, 2)

        right_layout.addWidget(discord_splitter)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 3)

    def refresh(self) -> None:
        self.pheno_list.clear()
        self.round_table.setRowCount(0)
        self.label_selector.clear()
        self.results_view.clear()
        self.discord_table.setRowCount(0)
        self.discord_note.clear()
        self.manual_reviewer_combo.clear()
        self.manual_import_btn.setEnabled(False)
        self.auto_import_btn.setEnabled(False)
        self.import_status_label.clear()
        self.current_round = None
        self.current_reviewer_names = {}
        self.round_summary.setText("Select a round to review agreement metrics")
        try:
            db = self.ctx.require_db()
        except RuntimeError:
            return
        with db.connect() as conn:
            rows = conn.execute("SELECT pheno_id, name FROM phenotypes ORDER BY name").fetchall()
        for row in rows:
            item = QtWidgets.QListWidgetItem(f"{row['name']} ({row['pheno_id']})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(row))
            self.pheno_list.addItem(item)

    def _on_pheno_selected(self) -> None:
        items = self.pheno_list.selectedItems()
        self.round_table.setRowCount(0)
        self.label_selector.clear()
        self.results_view.clear()
        self.discord_table.setRowCount(0)
        self.discord_note.clear()
        self.manual_reviewer_combo.clear()
        self.manual_import_btn.setEnabled(False)
        self.auto_import_btn.setEnabled(False)
        self.import_status_label.clear()
        self.current_round = None
        self.current_reviewer_names = {}
        if not items:
            return
        pheno = items[0].data(QtCore.Qt.ItemDataRole.UserRole) or {}
        pheno_id = pheno.get("pheno_id")
        if not pheno_id:
            return
        db = self.ctx.require_db()
        with db.connect() as conn:
            rounds = conn.execute(
                "SELECT round_id, round_number, status, labelset_id FROM rounds WHERE pheno_id=? ORDER BY round_number DESC",
                (pheno_id,),
            ).fetchall()
        self.round_table.setRowCount(len(rounds))
        for row_index, round_row in enumerate(rounds):
            with db.connect() as conn:
                reviewers = conn.execute(
                    """
                    SELECT reviewers.reviewer_id, reviewers.name
                    FROM assignments
                    JOIN reviewers ON reviewers.reviewer_id = assignments.reviewer_id
                    WHERE assignments.round_id=?
                    ORDER BY reviewers.name
                    """,
                    (round_row["round_id"],),
                ).fetchall()
            reviewer_names = ", ".join(r["name"] for r in reviewers) or "No reviewers"
            round_item = QtWidgets.QTableWidgetItem(f"Round {round_row['round_number']}")
            round_item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                {
                    "pheno_id": pheno_id,
                    "round_id": round_row["round_id"],
                    "round_number": round_row["round_number"],
                    "status": round_row["status"],
                    "labelset_id": round_row["labelset_id"],
                    "reviewers": [dict(r) for r in reviewers],
                },
            )
            status_item = QtWidgets.QTableWidgetItem(round_row["status"])
            reviewers_item = QtWidgets.QTableWidgetItem(reviewer_names)
            self.round_table.setItem(row_index, 0, round_item)
            self.round_table.setItem(row_index, 1, status_item)
            self.round_table.setItem(row_index, 2, reviewers_item)
        self.round_table.resizeColumnsToContents()
        if self.round_table.rowCount():
            self.round_table.setCurrentCell(0, 0)

    def _on_round_selected(self) -> None:
        items = self.round_table.selectedItems()
        self.label_selector.clear()
        self.results_view.clear()
        self.discord_table.setRowCount(0)
        self.discord_note.clear()
        if not items:
            self.current_round = None
            self.round_summary.setText("Select a round to review agreement metrics")
            return
        round_data = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if not round_data:
            self.current_round = None
            self.round_summary.setText("Select a round to review agreement metrics")
            return
        self.current_round = round_data
        reviewers = round_data.get("reviewers", [])
        self.current_reviewer_names = {rev["reviewer_id"]: rev["name"] for rev in reviewers if "reviewer_id" in rev}
        self.manual_reviewer_combo.clear()
        for reviewer in reviewers:
            reviewer_id = reviewer.get("reviewer_id")
            if not reviewer_id:
                continue
            display = reviewer.get("name") or reviewer_id
            self.manual_reviewer_combo.addItem(display, reviewer_id)
        has_reviewers = self.manual_reviewer_combo.count() > 0
        self.manual_import_btn.setEnabled(has_reviewers)
        self.auto_import_btn.setEnabled(has_reviewers)
        summary_parts = [
            f"Round {round_data.get('round_number')}",
            round_data.get("status", ""),
            ", ".join(self.current_reviewer_names.values()) or "No reviewers",
        ]
        self.round_summary.setText(" • ".join(part for part in summary_parts if part))
        db = self.ctx.require_db()
        with db.connect() as conn:
            labels = conn.execute(
                "SELECT label_id, name FROM labels WHERE labelset_id=? ORDER BY order_index",
                (round_data.get("labelset_id"),),
            ).fetchall()
        for label in labels:
            display = label["name"]
            self.label_selector.addItem(display, label["label_id"])
        if self.label_selector.count():
            self.label_selector.setCurrentIndex(0)
        if has_reviewers:
            self._auto_import_submissions(silent=True)

    def _on_auto_import_clicked(self) -> None:
        if not self.current_round:
            QtWidgets.QMessageBox.information(
                self, "Assignment import", "Select a round to import submissions."
            )
            return
        self._auto_import_submissions(silent=False)

    def _auto_import_submissions(self, silent: bool) -> None:
        if not self.current_round:
            if not silent:
                QtWidgets.QMessageBox.information(
                    self, "Assignment import", "Select a round to import submissions."
                )
            return
        try:
            project_root = self.ctx.require_project()
        except RuntimeError:
            message = "Project directory is not available."
            self.import_status_label.setText(message)
            if not silent:
                QtWidgets.QMessageBox.warning(self, "Assignment import", message)
            return
        pheno_id = self.current_round.get("pheno_id")
        round_number = self.current_round.get("round_number")
        if not pheno_id or round_number is None:
            message = "Round metadata is incomplete."
            self.import_status_label.setText(message)
            if not silent:
                QtWidgets.QMessageBox.warning(self, "Assignment import", message)
            return
        round_dir = project_root / "phenotypes" / pheno_id / "rounds" / f"round_{round_number}"
        statuses: List[str] = []
        errors = 0
        for reviewer in self.current_round.get("reviewers", []):
            reviewer_id = reviewer.get("reviewer_id")
            display_name = reviewer.get("name") or reviewer_id or "Unknown reviewer"
            if not reviewer_id:
                continue
            assignment_dir = round_dir / "assignments" / reviewer_id
            submitted_path = assignment_dir / "submitted.json"
            assignment_db = assignment_dir / "assignment.db"
            if not assignment_dir.exists():
                statuses.append(f"{display_name}: assignment folder missing")
                errors += 1
                continue
            if not submitted_path.exists():
                statuses.append(f"{display_name}: awaiting submission")
                continue
            if not assignment_db.exists():
                statuses.append(f"{display_name}: assignment.db missing")
                errors += 1
                continue
            try:
                self._copy_assignment_to_imports(reviewer_id, assignment_db)
            except Exception as exc:  # noqa: BLE001
                statuses.append(f"{display_name}: import failed ({exc})")
                errors += 1
            else:
                statuses.append(f"{display_name}: imported")
        summary = "\n".join(statuses) if statuses else "No reviewers for this round."
        self.import_status_label.setText(summary)
        if not silent:
            if errors:
                QtWidgets.QMessageBox.warning(self, "Assignment import", summary)
            else:
                QtWidgets.QMessageBox.information(self, "Assignment import", summary)

    def _manual_import_assignment(self) -> None:
        if not self.current_round:
            QtWidgets.QMessageBox.information(
                self, "Assignment import", "Select a round before importing."
            )
            return
        reviewer_id = self.manual_reviewer_combo.currentData()
        if not reviewer_id:
            QtWidgets.QMessageBox.information(
                self, "Assignment import", "Select a reviewer to import."
            )
            return
        start_dir = str(self.ctx.project_root or Path.home())
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select assignment database",
            start_dir,
            "SQLite databases (*.db);;All files (*)",
        )
        if not path_str:
            return
        display_name = self.current_reviewer_names.get(reviewer_id, reviewer_id)
        try:
            self._copy_assignment_to_imports(reviewer_id, Path(path_str))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self,
                "Assignment import",
                f"Failed to import assignment for {display_name}: {exc}",
            )
            return
        self.import_status_label.setText(f"{display_name}: imported from manual selection")
        QtWidgets.QMessageBox.information(
            self,
            "Assignment import",
            f"Imported assignment for {display_name}.",
        )
        self._auto_import_submissions(silent=True)

    def _copy_assignment_to_imports(self, reviewer_id: str, source: Path) -> Path:
        if not self.current_round:
            raise RuntimeError("Round context missing")
        pheno_id = self.current_round.get("pheno_id")
        round_number = self.current_round.get("round_number")
        if not pheno_id or round_number is None:
            raise RuntimeError("Round metadata incomplete")
        project_root = self.ctx.require_project()
        round_dir = project_root / "phenotypes" / pheno_id / "rounds" / f"round_{round_number}"
        imports_dir = ensure_dir(round_dir / "imports")
        target_path = imports_dir / f"{reviewer_id}_assignment.db"
        copy_sqlite_database(source, target_path)
        db = self.ctx.require_db()
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT round_id FROM rounds WHERE pheno_id=? AND round_number=?",
                (pheno_id, round_number),
            ).fetchone()
            if not row:
                raise RuntimeError("Round metadata missing in project database")
            conn.execute(
                "UPDATE assignments SET status='imported' WHERE round_id=? AND reviewer_id=?",
                (row["round_id"], reviewer_id),
            )
        return target_path

    def _compute_agreement(self) -> None:
        if not self.current_round:
            QtWidgets.QMessageBox.information(self, "IAA", "Select a round before calculating agreement")
            return
        label_id = self.label_selector.currentData()
        if not label_id:
            QtWidgets.QMessageBox.information(self, "IAA", "Select a label to evaluate")
            return
        project_root = self.ctx.require_project()
        pheno_id = self.current_round["pheno_id"]
        round_number = self.current_round["round_number"]
        round_dir = project_root / "phenotypes" / pheno_id / f"rounds/round_{round_number}"
        aggregate_path = round_dir / "round_aggregate.db"
        if not aggregate_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "IAA",
                "Round aggregate not found. Import assignments and build the aggregate before calculating agreement.",
            )
            return
        round_key = f"{pheno_id}_r{round_number}"
        with sqlite3.connect(aggregate_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT unit_id, reviewer_id, value, value_num, value_date, na
                FROM unit_annotations
                WHERE round_id=? AND label_id=?
                ORDER BY unit_id, reviewer_id
                """,
                (round_key, label_id),
            ).fetchall()
            summaries = conn.execute(
                "SELECT unit_id, patient_icn, doc_id FROM unit_summary WHERE round_id=?",
                (round_key,),
            ).fetchall()
        summary_map = {row["unit_id"]: dict(row) for row in summaries}
        if not rows:
            self.results_view.setPlainText("No annotations found for the selected label.")
            self.discord_table.setRowCount(0)
            return
        values_by_unit: Dict[str, Dict[str, str]] = {}
        for row in rows:
            unit_id = row["unit_id"]
            reviewer_id = row["reviewer_id"]
            value = self._format_value(row)
            values_by_unit.setdefault(unit_id, {})[reviewer_id] = value
        reviewer_ids = sorted({rid for ratings in values_by_unit.values() for rid in ratings.keys()})
        complete_samples = []
        for unit_id, ratings in values_by_unit.items():
            sample_row = []
            missing = False
            for reviewer_id in reviewer_ids:
                if reviewer_id not in ratings:
                    missing = True
                    break
                sample_row.append(ratings[reviewer_id])
            if not missing and len(sample_row) >= 2:
                complete_samples.append(sample_row)
        metric = self.metric_selector.currentText()
        result_lines = []
        if metric == "Percent agreement":
            value = percent_agreement(complete_samples)
            result_lines.append(f"Percent agreement: {value:.3%} across {len(complete_samples)} overlapped units")
        elif metric == "Cohen's kappa":
            if len(reviewer_ids) != 2:
                QtWidgets.QMessageBox.warning(self, "IAA", "Cohen's kappa requires exactly two reviewers")
                return
            rater_a = [row[0] for row in complete_samples]
            rater_b = [row[1] for row in complete_samples]
            value = cohens_kappa(rater_a, rater_b)
            result_lines.append(f"Cohen's kappa: {value:.3f} across {len(complete_samples)} overlapped units")
        else:
            categories = sorted({val for ratings in values_by_unit.values() for val in ratings.values()})
            matrix = []
            for ratings in values_by_unit.values():
                if len(ratings) < 2:
                    continue
                counts = [sum(1 for val in ratings.values() if val == category) for category in categories]
                matrix.append(counts)
            value = fleiss_kappa(matrix)
            result_lines.append(f"Fleiss' kappa: {value:.3f} across {len(matrix)} overlapped units")
        discordant_units = []
        for unit_id, ratings in values_by_unit.items():
            if len(ratings) < 2:
                continue
            if len(set(ratings.values())) > 1:
                discordant_units.append((unit_id, summary_map.get(unit_id, {}), ratings))
        result_lines.append(f"Discordant units: {len(discordant_units)}")
        self.results_view.setPlainText("\n".join(result_lines))
        self._populate_discord_table(discordant_units, reviewer_ids)

    def _populate_discord_table(
        self,
        rows: List[Tuple[str, Dict[str, object], Dict[str, str]]],
        reviewer_ids: List[str],
    ) -> None:
        headers = ["Unit", "Patient", "Document"] + [self.current_reviewer_names.get(rid, rid) for rid in reviewer_ids]
        self.discord_table.clear()
        self.discord_table.setColumnCount(len(headers))
        self.discord_table.setHorizontalHeaderLabels(headers)
        self.discord_table.setRowCount(len(rows))
        for row_index, (unit_id, summary, ratings) in enumerate(rows):
            unit_item = QtWidgets.QTableWidgetItem(unit_id)
            unit_item.setData(QtCore.Qt.ItemDataRole.UserRole, {"doc_id": summary.get("doc_id"), "unit_id": unit_id})
            patient_item = QtWidgets.QTableWidgetItem(str(summary.get("patient_icn", "")))
            doc_item = QtWidgets.QTableWidgetItem(str(summary.get("doc_id", "")))
            self.discord_table.setItem(row_index, 0, unit_item)
            self.discord_table.setItem(row_index, 1, patient_item)
            self.discord_table.setItem(row_index, 2, doc_item)
            for col_offset, reviewer_id in enumerate(reviewer_ids, start=3):
                value = ratings.get(reviewer_id, "—")
                self.discord_table.setItem(row_index, col_offset, QtWidgets.QTableWidgetItem(value))
        self.discord_table.resizeColumnsToContents()

    def _on_discord_selected(self) -> None:
        items = self.discord_table.selectedItems()
        if not items:
            self.discord_note.clear()
            return
        payload = items[0].data(QtCore.Qt.ItemDataRole.UserRole) or {}
        doc_id = payload.get("doc_id")
        if not doc_id:
            self.discord_note.clear()
            return
        try:
            corpus_db = self.ctx.require_corpus_db()
        except RuntimeError:
            self.discord_note.setPlainText("Corpus database is not available.")
            return
        with corpus_db.connect() as conn:
            row = conn.execute("SELECT text FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if row and row["text"]:
            self.discord_note.setPlainText(row["text"])
        else:
            self.discord_note.setPlainText("Document text not found in corpus.")

    @staticmethod
    def _format_value(row: sqlite3.Row) -> str:
        if row["na"]:
            return "N/A"
        if row["value"] is not None and row["value"] != "":
            return str(row["value"])
        if row["value_num"] is not None:
            return str(row["value_num"])
        if row["value_date"]:
            return row["value_date"]
        return ""


class AdminMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ctx = ProjectContext()
        self.setWindowTitle("VAAnnotate Admin")
        self.resize(1200, 800)
        self._setup_menu()
        self._setup_central()

    def _setup_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        open_action = file_menu.addAction("Open project folder…")
        open_action.triggered.connect(self._open_project)
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def _setup_central(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        nav_widget = QtWidgets.QListWidget()
        nav_widget.addItems(["Projects", "Phenotypes", "Rounds", "Corpus", "IAA"])
        nav_widget.currentRowChanged.connect(self._switch_page)
        splitter.addWidget(nav_widget)

        self.stack = QtWidgets.QStackedWidget()
        self.pages = [
            ProjectsPage(self.ctx),
            PhenotypePage(self.ctx),
            RoundPage(self.ctx),
            CorpusOverviewPage(self.ctx),
            IaaPage(self.ctx),
        ]
        for page in self.pages:
            self.stack.addWidget(page)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 5)
        nav_widget.setCurrentRow(0)
        self.setCentralWidget(splitter)

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.pages[index].refresh()

    def _open_project(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select project folder")
        if not directory:
            return
        self.ctx.open_project(Path(directory))
        for page in self.pages:
            page.refresh()


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = AdminMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
