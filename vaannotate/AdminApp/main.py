"""PySide6 based Admin application for VAAnnotate."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from PySide6 import QtCore, QtGui, QtWidgets

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
from ..shared.statistics import cohens_kappa, fleiss_kappa, percent_agreement
from ..utils import copy_sqlite_database, ensure_dir

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


@dataclass(frozen=True)
class AgreementSample:
    unit_id: str
    reviewer_ids: tuple[str, ...]
    values: tuple[str, ...]


class ProjectContext(QtCore.QObject):
    project_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.project_root: Optional[Path] = None
        self.project_db: Optional[Database] = None
        self.project_row: Optional[Dict[str, object]] = None
        self._corpus_cache: Dict[str, Database] = {}

    def open_project(self, directory: Path) -> None:
        directory = directory.resolve()
        project_db = Database(directory / "project.db")
        with project_db.transaction() as conn:
            ensure_schema(conn, PROJECT_MODELS)
        self.project_root = directory
        self.project_db = project_db
        self._corpus_cache.clear()
        self.project_row = self._load_project_row()
        self.project_changed.emit()

    def _load_project_row(self) -> Optional[Dict[str, object]]:
        try:
            db = self.require_db()
        except RuntimeError:
            return None
        with db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def require_project(self) -> Path:
        if not self.project_root:
            raise RuntimeError("No project opened")
        return self.project_root

    def require_db(self) -> Database:
        if not self.project_db:
            raise RuntimeError("Project database not initialized")
        return self.project_db

    def reload(self) -> None:
        self.project_row = self._load_project_row()
        self._corpus_cache.clear()
        self.project_changed.emit()

    def current_project_id(self) -> Optional[str]:
        if not self.project_row:
            self.project_row = self._load_project_row()
        if not self.project_row:
            return None
        return str(self.project_row.get("project_id"))

    def list_phenotypes(self) -> List[sqlite3.Row]:
        db = self.require_db()
        params: List[object] = []
        sql = "SELECT * FROM phenotypes"
        project_id = self.current_project_id()
        if project_id:
            sql += " WHERE project_id=?"
            params.append(project_id)
        sql += " ORDER BY name"
        with db.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def get_phenotype(self, pheno_id: str) -> Optional[sqlite3.Row]:
        db = self.require_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
        return row

    def list_rounds(self, pheno_id: str) -> List[sqlite3.Row]:
        db = self.require_db()
        with db.connect() as conn:
            return conn.execute(
                "SELECT * FROM rounds WHERE pheno_id=? ORDER BY round_number",
                (pheno_id,),
            ).fetchall()

    def get_round(self, round_id: str) -> Optional[sqlite3.Row]:
        db = self.require_db()
        with db.connect() as conn:
            return conn.execute(
                "SELECT * FROM rounds WHERE round_id=?",
                (round_id,),
            ).fetchone()

    def get_round_config(self, round_id: str) -> Optional[Dict[str, object]]:
        db = self.require_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT config_json FROM round_configs WHERE round_id=?",
                (round_id,),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["config_json"])
        except json.JSONDecodeError:
            return None

    def resolve_project_path(self, relative: str) -> Path:
        root = self.require_project()
        return (root / relative).resolve()

    def resolve_corpus_path(self, pheno_id: str) -> Path:
        pheno = self.get_phenotype(pheno_id)
        if not pheno:
            raise RuntimeError(f"Phenotype {pheno_id} not found")
        corpus_path = pheno["corpus_path"]
        if not corpus_path:
            raise RuntimeError("Phenotype does not define a corpus")
        root = self.require_project()
        return (root / corpus_path).resolve()

    def get_corpus_db(self, pheno_id: str) -> Database:
        if pheno_id in self._corpus_cache:
            return self._corpus_cache[pheno_id]
        path = self.resolve_corpus_path(pheno_id)
        db = Database(path)
        with db.transaction() as conn:
            ensure_schema(conn, [models.Patient, models.Document])
        self._corpus_cache[pheno_id] = db
        return db

    def create_phenotype(
        self,
        *,
        name: str,
        level: str,
        description: str,
        corpus_source: Path,
    ) -> models.Phenotype:
        project_id = self.current_project_id()
        if not project_id:
            raise RuntimeError("Project metadata missing; ensure a project record exists")
        pheno_id = str(uuid.uuid4())
        project_root = self.require_project()
        phenotype_dir = ensure_dir(project_root / "phenotypes" / pheno_id)
        rounds_dir = ensure_dir(phenotype_dir / "rounds")
        _ = rounds_dir  # make mypy happy about unused variable
        corpus_dir = ensure_dir(phenotype_dir / "corpus")
        target_corpus = corpus_dir / "corpus.db"
        copy_sqlite_database(corpus_source, target_corpus)
        relative_corpus = target_corpus.relative_to(project_root)
        record = models.Phenotype(
            pheno_id=pheno_id,
            project_id=project_id,
            name=name,
            level=level,
            description=description,
            corpus_path=str(relative_corpus.as_posix()),
        )
        db = self.require_db()
        with db.transaction() as conn:
            record.save(conn)
        self.project_changed.emit()
        return record

    def update_cache_after_round(self, pheno_id: str) -> None:
        # Keep API parity with previous refresh pattern
        self._corpus_cache.pop(pheno_id, None)
        self.project_changed.emit()


class PhenotypeDialog(QtWidgets.QDialog):
    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.corpus_path: Optional[Path] = None
        self.setWindowTitle("Add phenotype")
        self.resize(400, 300)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.level_combo = QtWidgets.QComboBox()
        self.level_combo.addItems(["single_doc", "multi_doc"])
        corpus_layout = QtWidgets.QHBoxLayout()
        self.corpus_edit = QtWidgets.QLineEdit()
        self.corpus_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_corpus)
        corpus_layout.addWidget(self.corpus_edit)
        corpus_layout.addWidget(browse_btn)
        self.description_edit = QtWidgets.QPlainTextEdit()
        form.addRow("Name", self.name_edit)
        form.addRow("Level", self.level_combo)
        form.addRow("Corpus", corpus_layout)
        form.addRow("Description", self.description_edit)
        layout.addLayout(form)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _browse_corpus(self) -> None:
        start_dir = str(self.ctx.project_root or Path.home())
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select corpus database",
            start_dir,
            "SQLite databases (*.db);;All files (*)",
        )
        if not path_str:
            return
        self.corpus_path = Path(path_str)
        self.corpus_edit.setText(str(self.corpus_path))

    def accept(self) -> None:  # noqa: D401 - Qt override
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Validation", "Phenotype name is required.")
            return
        if not self.corpus_path or not self.corpus_path.exists():
            QtWidgets.QMessageBox.warning(self, "Validation", "Select a valid corpus database file.")
            return
        super().accept()

    def values(self) -> Dict[str, object]:
        return {
            "name": self.name_edit.text().strip(),
            "level": self.level_combo.currentText(),
            "description": self.description_edit.toPlainText().strip(),
            "corpus_path": self.corpus_path,
        }


class RoundBuilderDialog(QtWidgets.QDialog):
    def __init__(
        self,
        ctx: ProjectContext,
        pheno_row: sqlite3.Row,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.pheno_row = pheno_row
        self.created_round_id: Optional[str] = None
        self.created_round_number: Optional[int] = None
        self.setWindowTitle(f"New round • {pheno_row['name']}")
        self.resize(720, 760)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(container)
        scroll_layout.setContentsMargins(8, 8, 8, 8)

        setup_group = QtWidgets.QGroupBox("Round setup")
        setup_form = QtWidgets.QFormLayout(setup_group)
        self.labelset_edit = QtWidgets.QLineEdit()
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setMaximum(2**31 - 1)
        self.overlap_spin = QtWidgets.QSpinBox()
        self.overlap_spin.setRange(0, 1000)
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setRange(1, 10000)
        self.status_combo = QtWidgets.QComboBox()
        self.status_combo.addItems(["draft", "active", "closed", "adjudicating", "finalized"])
        setup_form.addRow("Label set ID", self.labelset_edit)
        setup_form.addRow("Seed", self.seed_spin)
        setup_form.addRow("Overlap N", self.overlap_spin)
        setup_form.addRow("Sample per reviewer", self.sample_spin)
        setup_form.addRow("Status", self.status_combo)
        scroll_layout.addWidget(setup_group)

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
        scroll_layout.addWidget(filter_group)

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
        scroll_layout.addWidget(strat_group)

        scroll_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

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

    def _prompt_reviewers(self) -> Optional[List[Dict[str, str]]]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Reviewers")
        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget(0, 1)
        table.setHorizontalHeaderLabels(["Reviewer name"])
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        def add_row() -> None:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))

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
        names: List[str] = []
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            name = item.text().strip() if item else ""
            if name:
                names.append(name)
        if not names:
            QtWidgets.QMessageBox.warning(self, "Reviewers", "Add at least one reviewer name.")
            return []
        seen: Set[str] = set()
        reviewers: List[Dict[str, str]] = []
        for name in names:
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "reviewer"
            candidate = slug
            counter = 2
            while candidate in seen:
                candidate = f"{slug}_{counter}"
                counter += 1
            seen.add(candidate)
            reviewers.append({"id": candidate, "name": name})
        return reviewers

    def _next_round_number(self) -> int:
        pheno_id = self.pheno_row["pheno_id"]
        db = self.ctx.require_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT MAX(round_number) FROM rounds WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
        return (row[0] or 0) + 1

    def _build_label_schema(self, labelset_id: str, db: Database) -> Dict[str, object]:
        with db.connect() as conn:
            labels = conn.execute(
                "SELECT * FROM labels WHERE labelset_id=? ORDER BY order_index",
                (labelset_id,),
            ).fetchall()
            options = conn.execute(
                "SELECT * FROM label_options WHERE label_id IN (SELECT label_id FROM labels WHERE labelset_id=?)",
                (labelset_id,),
            ).fetchall()
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
        return {"labelset_id": labelset_id, "labels": schema_labels}

    def accept(self) -> None:  # noqa: D401 - Qt override
        if not self._create_round():
            return
        super().accept()

    def _create_round(self) -> bool:
        pheno_id = self.pheno_row["pheno_id"]
        pheno_level = self.pheno_row["level"]
        ctx = self.ctx
        db = ctx.require_db()
        seed = self.seed_spin.value()
        overlap = self.overlap_spin.value()
        reviewers = self._prompt_reviewers()
        if not reviewers:
            return False
        labelset_id = self.labelset_edit.text().strip() or f"auto_{pheno_id}"
        created_at = QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)
        default_labels: List[Dict[str, object]] = []
        with db.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM label_sets WHERE labelset_id=?",
                (labelset_id,),
            ).fetchone()
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
        try:
            corpus_rows = candidate_documents(ctx.get_corpus_db(pheno_id), pheno_level, filters)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Round", f"Failed to query corpus: {exc}")
            return False
        if not corpus_rows:
            QtWidgets.QMessageBox.warning(self, "Round", "The selected corpus returned no candidate documents.")
            return False
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
                "Round",
                "Not enough candidate documents were found to satisfy the desired sample size for: "
                + ", ".join(missing),
            )
        round_id = str(uuid.uuid4())
        round_number = self._next_round_number()
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
        project_root = ctx.require_project()
        round_dir = ensure_dir(project_root / "phenotypes" / pheno_id / "rounds" / f"round_{round_number}")
        write_manifest(round_dir / "manifest.csv", assignments)
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
            assignment_dir = ensure_dir(round_dir / "assignments" / reviewer["id"])
            db_path = assignment_dir / "assignment.db"
            assignment_db = initialize_assignment_db(db_path)
            populate_assignment_db(assignment_db, reviewer["id"], assignments[reviewer["id"]].units)
            label_schema = self._build_label_schema(labelset_id, db)
            schema_path = assignment_dir / "label_schema.json"
            schema_path.write_text(json.dumps(label_schema, indent=2), encoding="utf-8")
        self.created_round_id = round_id
        self.created_round_number = round_number
        return True


class ProjectTreeWidget(QtWidgets.QTreeWidget):
    node_selected = QtCore.Signal(dict)

    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setHeaderHidden(True)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.currentItemChanged.connect(self._on_current_item_changed)
        self.ctx.project_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.clear()
        project = self.ctx.project_row or self.ctx._load_project_row()
        if not project:
            placeholder = QtWidgets.QTreeWidgetItem(["No project loaded"])
            placeholder.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.addTopLevelItem(placeholder)
            return
        display_name = project.get("name") or project.get("project_id") or "Project"
        project_item = QtWidgets.QTreeWidgetItem([str(display_name)])
        project_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "project", "project": dict(project)})
        self.addTopLevelItem(project_item)
        project_item.setExpanded(True)
        for pheno in self.ctx.list_phenotypes():
            pheno_item = self._build_phenotype_item(pheno)
            project_item.addChild(pheno_item)
            pheno_item.setExpanded(True)
        self.expandItem(project_item)
        if project_item.childCount():
            self.setCurrentItem(project_item.child(0))
        else:
            self.setCurrentItem(project_item)

    def _build_phenotype_item(self, pheno: sqlite3.Row) -> QtWidgets.QTreeWidgetItem:
        pheno_item = QtWidgets.QTreeWidgetItem([f"{pheno['name']} ({pheno['level']})"])
        pheno_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "phenotype", "pheno": dict(pheno)})
        rounds = self.ctx.list_rounds(pheno["pheno_id"])
        for round_row in rounds:
            label = f"Round {round_row['round_number']} ({round_row['status']})"
            child = QtWidgets.QTreeWidgetItem([label])
            child.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "round", "round": dict(round_row)})
            pheno_item.addChild(child)
        corpus_item = QtWidgets.QTreeWidgetItem(["Corpus"])
        corpus_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "corpus", "pheno": dict(pheno)})
        pheno_item.addChild(corpus_item)
        iaa_item = QtWidgets.QTreeWidgetItem(["IAA"])
        iaa_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "iaa", "pheno": dict(pheno)})
        pheno_item.addChild(iaa_item)
        return pheno_item

    def _on_current_item_changed(
        self,
        current: Optional[QtWidgets.QTreeWidgetItem],
        previous: Optional[QtWidgets.QTreeWidgetItem],
    ) -> None:
        del previous
        if not current:
            self.node_selected.emit({})
            return
        data = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(data, dict):
            self.node_selected.emit(data)

    def _show_context_menu(self, point: QtCore.QPoint) -> None:
        item = self.itemAt(point)
        if not item:
            return
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        menu = QtWidgets.QMenu(self)
        node_type = data.get("type")
        if node_type == "project":
            action = menu.addAction("Add phenotype…")
            action.triggered.connect(lambda: self._add_phenotype(item))
        elif node_type == "phenotype":
            action = menu.addAction("Add round…")
            action.triggered.connect(lambda: self._add_round(item))
        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(point))

    def _add_phenotype(self, item: QtWidgets.QTreeWidgetItem) -> None:
        del item
        dialog = PhenotypeDialog(self.ctx, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        corpus_path = values.get("corpus_path")
        if not isinstance(corpus_path, Path):
            QtWidgets.QMessageBox.warning(self, "Phenotype", "Invalid corpus selection.")
            return
        try:
            record = self.ctx.create_phenotype(
                name=str(values.get("name", "")),
                level=str(values.get("level", "single_doc")),
                description=str(values.get("description", "")),
                corpus_source=corpus_path,
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Phenotype", f"Failed to create phenotype: {exc}")
            return
        QtCore.QTimer.singleShot(0, lambda: self._select_phenotype(record.pheno_id))

    def _add_round(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
        pheno = data.get("pheno")
        if not isinstance(pheno, dict):
            return
        pheno_row = self.ctx.get_phenotype(pheno["pheno_id"])
        if not pheno_row:
            QtWidgets.QMessageBox.warning(self, "Round", "Phenotype record not found.")
            return
        dialog = RoundBuilderDialog(self.ctx, pheno_row, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        round_id = dialog.created_round_id
        if round_id:
            self.ctx.project_changed.emit()
            QtCore.QTimer.singleShot(0, lambda: self._select_round(pheno_row["pheno_id"], round_id))

    def _iter_items(self, root: Optional[QtWidgets.QTreeWidgetItem]) -> Iterable[QtWidgets.QTreeWidgetItem]:
        if not root:
            return
        yield root
        for index in range(root.childCount()):
            yield from self._iter_items(root.child(index))

    def _select_phenotype(self, pheno_id: str) -> None:
        for item in self._iter_items(self.topLevelItem(0)):
            data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and data.get("type") == "phenotype" and data.get("pheno", {}).get("pheno_id") == pheno_id:
                self.setCurrentItem(item)
                self.expandItem(item)
                return
        self.refresh()

    def _select_round(self, pheno_id: str, round_id: str) -> None:
        self.refresh()
        project_item = self.topLevelItem(0)
        target_round: Optional[QtWidgets.QTreeWidgetItem] = None
        for pheno_item in self._iter_items(project_item):
            data = pheno_item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and data.get("type") == "round" and data.get("round", {}).get("round_id") == round_id:
                target_round = pheno_item
                break
        if target_round:
            self.setCurrentItem(target_round)
            parent = target_round.parent()
            if parent:
                self.expandItem(parent)



class ProjectOverviewWidget(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QTextBrowser()
        layout.addWidget(self.text)
        layout.addStretch()

    def set_project(self, project: Optional[Dict[str, object]]) -> None:
        if not project:
            self.text.setPlainText("Select a project to view metadata.")
            return
        lines = [
            f"Project: {project.get('name') or '—'}",
            f"Project ID: {project.get('project_id') or '—'}",
            f"Created by: {project.get('created_by') or '—'}",
            f"Created at: {project.get('created_at') or '—'}",
        ]
        phenotypes = self.ctx.list_phenotypes()
        if phenotypes:
            lines.append("")
            lines.append("Phenotypes:")
            for pheno in phenotypes:
                lines.append(f"  • {pheno['name']} ({pheno['level']})")
        else:
            lines.append("")
            lines.append("No phenotypes defined. Right-click the project to add one.")
        self.text.setPlainText("\n".join(lines))


class PhenotypeDetailWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QFormLayout(self)
        self.name_label = QtWidgets.QLabel()
        self.level_label = QtWidgets.QLabel()
        self.description_label = QtWidgets.QTextEdit()
        self.description_label.setReadOnly(True)
        self.corpus_label = QtWidgets.QLabel()
        self.description_label.setFixedHeight(120)
        layout.addRow("Name", self.name_label)
        layout.addRow("Level", self.level_label)
        layout.addRow("Corpus", self.corpus_label)
        layout.addRow("Description", self.description_label)

    def set_phenotype(self, pheno: Optional[Dict[str, object]]) -> None:
        if not pheno:
            self.name_label.clear()
            self.level_label.clear()
            self.description_label.clear()
            self.corpus_label.clear()
            return
        self.name_label.setText(str(pheno.get("name", "")))
        self.level_label.setText(str(pheno.get("level", "")))
        self.corpus_label.setText(str(pheno.get("corpus_path", "")))
        self.description_label.setPlainText(str(pheno.get("description", "")))


class RoundDetailWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.meta_form = QtWidgets.QFormLayout()
        self.round_label = QtWidgets.QLabel()
        self.status_label = QtWidgets.QLabel()
        self.labelset_label = QtWidgets.QLabel()
        self.seed_label = QtWidgets.QLabel()
        self.overlap_label = QtWidgets.QLabel()
        self.meta_form.addRow("Round", self.round_label)
        self.meta_form.addRow("Status", self.status_label)
        self.meta_form.addRow("Label set", self.labelset_label)
        self.meta_form.addRow("Seed", self.seed_label)
        self.meta_form.addRow("Overlap", self.overlap_label)
        layout.addLayout(self.meta_form)
        self.config_view = QtWidgets.QTextEdit()
        self.config_view.setReadOnly(True)
        self.config_view.setPlaceholderText("Select a round to view configuration")
        layout.addWidget(self.config_view)

    def set_round(self, round_row: Optional[Dict[str, object]], config: Optional[Dict[str, object]]) -> None:
        if not round_row:
            self.round_label.clear()
            self.status_label.clear()
            self.labelset_label.clear()
            self.seed_label.clear()
            self.overlap_label.clear()
            self.config_view.clear()
            return
        self.round_label.setText(f"Round {round_row.get('round_number')} ({round_row.get('round_id')})")
        self.status_label.setText(str(round_row.get("status", "")))
        self.labelset_label.setText(str(round_row.get("labelset_id", "")))
        self.seed_label.setText(str(round_row.get("rng_seed", "")))
        self.overlap_label.setText(str(round_row.get("overlap_n", "")))
        if config:
            self.config_view.setPlainText(self._summarize_config(config))
        else:
            self.config_view.setPlainText("Configuration not available.")
        self.config_view.moveCursor(QtGui.QTextCursor.MoveOperation.Start)

    def _summarize_config(self, config: Dict[str, object]) -> str:
        sections: List[str] = []
        setup_items: List[str] = []
        setup_items.append(f"Round ID: {config.get('round_id', '—')}")
        if config.get("round_number") is not None:
            setup_items.append(f"Round number: {config['round_number']}")
        setup_items.append(f"Label set: {config.get('labelset_id', '—')}")
        setup_items.append(f"Status: {config.get('status', 'draft')}")
        if config.get("sample_per_reviewer"):
            setup_items.append(f"Sample per reviewer: {config['sample_per_reviewer']}")
        setup_items.append(f"Overlap units: {config.get('overlap_n', 0)}")
        setup_items.append(f"RNG seed: {config.get('rng_seed', 0)}")
        reviewers = config.get("reviewers") or []
        if reviewers:
            reviewer_names = [str(reviewer.get("name") or reviewer.get("id")) for reviewer in reviewers]
            setup_items.append(f"Reviewers: {', '.join(reviewer_names)}")
        sections.append(self._format_section("Round setup", setup_items))

        filters = config.get("filters") or {}
        filter_items: List[str] = []
        patient_filters = filters.get("patient") or {}
        for key, value in patient_filters.items():
            label = {
                "sta3n_in": "Patient STA3N",
                "year_range": "Patient year range",
                "softlabel_gte": "Softlabel ≥",
            }.get(key, key)
            filter_items.append(f"Patient – {label}: {self._format_filter_value(key, value)}")
        note_filters = filters.get("note") or {}
        for key, value in note_filters.items():
            label = {
                "notetype_in": "Note types",
                "note_year_range": "Note year range",
                "regex": "Regex",
            }.get(key, key)
            filter_items.append(f"Note – {label}: {self._format_filter_value(key, value)}")
        if filter_items:
            sections.append(self._format_section("Sampling filters", filter_items))

        stratification = config.get("stratification") or {}
        strat_items: List[str] = []
        keys = stratification.get("keys")
        if keys:
            if isinstance(keys, list):
                strat_items.append(f"Stratify by: {', '.join(keys)}")
            else:
                strat_items.append(f"Stratify by: {keys}")
        if stratification.get("sample_per_stratum"):
            strat_items.append(f"Sample per stratum: {stratification['sample_per_stratum']}")
        if strat_items:
            sections.append(self._format_section("Stratification", strat_items))

        summary = "\n\n".join(section for section in sections if section)
        return summary or "Configuration not available."

    @staticmethod
    def _format_section(title: str, entries: List[str]) -> str:
        filtered = [entry for entry in entries if entry]
        if not filtered:
            return ""
        lines = [title]
        lines.extend(f"  • {entry}" for entry in filtered)
        return "\n".join(lines)

    @staticmethod
    def _format_filter_value(key: str, value: object) -> str:
        if isinstance(value, list):
            if len(value) == 2 and all(isinstance(v, (int, float, str)) for v in value):
                return f"{value[0]} – {value[1]}"
            return ", ".join(str(v) for v in value)
        if key == "softlabel_gte":
            return str(value)
        return str(value)


class CorpusWidget(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        layout = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel("Select a phenotype to view corpus contents.")
        layout.addWidget(self.summary_label)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Doc ID", "Patient", "Note type", "Date", "Preview"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def set_phenotype(self, pheno: Optional[Dict[str, object]]) -> None:
        if not pheno:
            self.summary_label.setText("Select a phenotype to view corpus contents.")
            self.table.setRowCount(0)
            return
        pheno_id = pheno.get("pheno_id")
        if not pheno_id:
            self.summary_label.setText("Phenotype metadata incomplete.")
            self.table.setRowCount(0)
            return
        try:
            db = self.ctx.get_corpus_db(pheno_id)
        except Exception as exc:  # noqa: BLE001
            self.summary_label.setText(f"Corpus unavailable: {exc}")
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
                (row["preview"] or "").replace("\n", " ")
                + ("…" if row["preview"] and len(row["preview"]) == 200 else ""),
            ]
            for col_index, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.table.setItem(row_index, col_index, item)
        self.table.resizeColumnsToContents()



class IaaWidget(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.current_pheno: Optional[Dict[str, object]] = None
        self.current_round: Optional[Dict[str, object]] = None
        self.current_reviewer_names: Dict[str, str] = {}
        self.assignment_paths: Dict[str, Path] = {}
        self.unit_rows: List[Dict[str, object]] = []
        self.round_manifest: Dict[str, Dict[str, bool]] = {}
        self.label_lookup: Dict[str, str] = {}
        self.label_order: List[str] = []
        self.reviewer_column_order: List[str] = []
        self._setup_ui()
        self.ctx.project_changed.connect(self.reset)

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        content_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(content_splitter)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.round_table = QtWidgets.QTableWidget()
        self.round_table.setColumnCount(3)
        self.round_table.setHorizontalHeaderLabels(["Round", "Status", "Reviewers"])
        self.round_table.verticalHeader().setVisible(False)
        self.round_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.round_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.round_table.itemSelectionChanged.connect(self._on_round_selected)
        self.round_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.round_table)

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
        left_layout.addLayout(controls)

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
        left_layout.addLayout(import_layout)

        self.import_status_label = QtWidgets.QLabel()
        self.import_status_label.setWordWrap(True)
        left_layout.addWidget(self.import_status_label)
        self._import_summary: str = ""
        self._waiting_summary: str = ""

        self.round_summary = QtWidgets.QLabel("Select a round to review agreement metrics")
        self.round_summary.setWordWrap(True)
        left_layout.addWidget(self.round_summary)

        units_label = QtWidgets.QLabel("Imported units (overlapping assignments shown first)")
        units_label.setWordWrap(True)
        left_layout.addWidget(units_label)

        self.unit_table = QtWidgets.QTableWidget()
        self._update_unit_table_headers()
        self.unit_table.verticalHeader().setVisible(False)
        self.unit_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.unit_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.unit_table.setWordWrap(True)
        self.unit_table.itemSelectionChanged.connect(self._on_unit_selected)
        self.unit_table.horizontalHeader().setStretchLastSection(True)
        self.unit_table.cellDoubleClicked.connect(self._show_annotation_dialog)
        left_layout.addWidget(self.unit_table, 1)
        self._unit_metadata_column_count = 5

        doc_panel = QtWidgets.QWidget()
        doc_panel_layout = QtWidgets.QVBoxLayout(doc_panel)
        doc_panel_layout.setContentsMargins(0, 0, 0, 0)
        doc_panel_layout.setSpacing(8)
        doc_panel_layout.addWidget(QtWidgets.QLabel("Documents"))

        doc_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        doc_table_container = QtWidgets.QWidget()
        doc_layout = QtWidgets.QVBoxLayout(doc_table_container)
        doc_layout.setContentsMargins(0, 0, 0, 0)
        self.document_table = QtWidgets.QTableWidget()
        self.document_table.setColumnCount(4)
        self.document_table.setHorizontalHeaderLabels(["Document", "Type", "Date", "Facility"])
        self.document_table.verticalHeader().setVisible(False)
        self.document_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.document_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.document_table.itemSelectionChanged.connect(self._on_document_selected)
        self.document_table.horizontalHeader().setStretchLastSection(True)
        doc_layout.addWidget(self.document_table)
        doc_splitter.addWidget(doc_table_container)

        self.document_preview = QtWidgets.QTextEdit()
        self.document_preview.setReadOnly(True)
        self.document_preview.setPlaceholderText("Select a document to preview text")
        doc_splitter.addWidget(self.document_preview)
        doc_splitter.setStretchFactor(0, 1)
        doc_splitter.setStretchFactor(1, 2)
        doc_panel_layout.addWidget(doc_splitter)

        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(doc_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

    def reset(self) -> None:
        self.current_pheno = None
        self.current_round = None
        self.current_reviewer_names = {}
        self.assignment_paths = {}
        self.unit_rows = []
        self.round_manifest = {}
        self.label_lookup = {}
        self.label_order = []
        self.reviewer_column_order = []
        self.round_table.setRowCount(0)
        self.label_selector.clear()
        self.unit_table.setRowCount(0)
        self.document_table.setRowCount(0)
        self.document_preview.clear()
        self.manual_reviewer_combo.clear()
        self.manual_import_btn.setEnabled(False)
        self.auto_import_btn.setEnabled(False)
        self._set_import_summary("")
        self._set_waiting_summary("")
        self.round_summary.setText("Select a round to review agreement metrics")

    def set_phenotype(self, pheno: Optional[Dict[str, object]]) -> None:
        self.reset()
        if not pheno:
            return
        self.current_pheno = pheno
        self._load_rounds()

    def _load_rounds(self) -> None:
        if not self.current_pheno:
            return
        pheno_id = self.current_pheno.get("pheno_id")
        if not pheno_id:
            return
        db = self.ctx.require_db()
        with db.connect() as conn:
            rounds = conn.execute(
                "SELECT round_id, round_number, status, labelset_id FROM rounds WHERE pheno_id=? ORDER BY round_number DESC",
                (pheno_id,),
            ).fetchall()
        self.round_table.setRowCount(len(rounds))
        for row_idx, round_row in enumerate(rounds):
            item = QtWidgets.QTableWidgetItem(f"Round {round_row['round_number']}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(round_row))
            self.round_table.setItem(row_idx, 0, item)
            self.round_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(round_row["status"]))
            self.round_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(round_row["labelset_id"]))
        if rounds:
            self.round_table.selectRow(0)
        else:
            self._on_round_selected()

    def _on_round_selected(self) -> None:
        items = self.round_table.selectedItems()
        if not items:
            self.current_round = None
            self.label_selector.clear()
            self.unit_table.setRowCount(0)
            self.document_table.setRowCount(0)
            self.document_preview.clear()
            self.manual_reviewer_combo.clear()
            self.manual_import_btn.setEnabled(False)
            self.auto_import_btn.setEnabled(False)
            self.import_status_label.clear()
            self.round_summary.setText("Select a round to review agreement metrics")
            return
        round_meta = items[0].data(QtCore.Qt.ItemDataRole.UserRole) or {}
        self.current_round = round_meta
        pheno_id = round_meta.get("pheno_id") or (self.current_pheno or {}).get("pheno_id")
        if pheno_id:
            self.current_round["pheno_id"] = pheno_id
        self._load_round_details()

    def _load_round_details(self) -> None:
        if not self.current_round:
            return
        round_id = self.current_round.get("round_id")
        if not round_id:
            return
        db = self.ctx.require_db()
        with db.connect() as conn:
            reviewers = conn.execute(
                "SELECT reviewer_id, name FROM reviewers WHERE reviewer_id IN (SELECT reviewer_id FROM assignments WHERE round_id=?)",
                (round_id,),
            ).fetchall()
            labels = conn.execute(
                "SELECT labels.label_id, labels.name FROM labels JOIN label_sets ON label_sets.labelset_id = labels.labelset_id "
                "WHERE label_sets.labelset_id=? ORDER BY labels.order_index",
                (self.current_round.get("labelset_id"),),
            ).fetchall()
        self.current_reviewer_names = {row["reviewer_id"]: row["name"] for row in reviewers}
        self.current_round["reviewers"] = [
            {"reviewer_id": row["reviewer_id"], "name": row["name"]} for row in reviewers
        ]
        self.manual_reviewer_combo.clear()
        for reviewer in reviewers:
            self.manual_reviewer_combo.addItem(reviewer["name"], reviewer["reviewer_id"])
        self.manual_import_btn.setEnabled(bool(reviewers))
        self.auto_import_btn.setEnabled(False)
        self.label_selector.clear()
        self.label_lookup = {row["label_id"]: row["name"] for row in labels}
        self.label_order = [row["label_id"] for row in labels]
        for row in labels:
            self.label_selector.addItem(row["name"], row["label_id"])
        self.assignment_paths = {}
        self.unit_rows = []
        self.round_manifest = {}
        self.reviewer_column_order = []
        self._set_import_summary("")
        self._set_waiting_summary("")
        self.round_summary.setText("Assignments not yet imported for this round.")
        self._auto_discover_imports()
        self._refresh_units_table()
        self._update_auto_import_state()

    def _auto_discover_imports(self) -> None:
        round_dir = self._resolve_round_dir()
        if not round_dir:
            return
        self._discover_existing_imports(round_dir)
        if self.assignment_paths:
            self._set_import_summary("Detected existing assignment imports.")

    def _on_auto_import_clicked(self) -> None:
        if not self.current_round:
            QtWidgets.QMessageBox.information(self, "Assignment import", "Select a round before importing.")
            return
        round_dir = self._resolve_round_dir()
        if not round_dir:
            QtWidgets.QMessageBox.warning(self, "Assignment import", "Round directory unavailable.")
            return
        sources, problems = self._collect_submission_sources(round_dir)
        if problems:
            issues = [
                f"- {self.current_reviewer_names.get(rid, rid)}: {reason}"
                for rid, reason in sorted(
                    problems.items(), key=lambda item: self.current_reviewer_names.get(item[0], item[0])
                )
            ]
            QtWidgets.QMessageBox.warning(
                self,
                "Assignment import",
                "Cannot import submissions until all reviewers have submitted receipts:\n" + "\n".join(issues),
            )
            self._update_auto_import_state()
            return
        if not sources:
            QtWidgets.QMessageBox.information(
                self,
                "Assignment import",
                "No reviewer submissions were detected.",
            )
            self._update_auto_import_state()
            return
        self._import_round_assignments(round_dir, silent=False, sources=sources)

    def _import_round_assignments(
        self,
        round_dir: Path,
        silent: bool = False,
        sources: Optional[Dict[str, Path]] = None,
    ) -> None:
        statuses: List[str] = []
        errors = 0
        for reviewer in self.current_round.get("reviewers", []):
            reviewer_id = reviewer.get("reviewer_id")
            if not reviewer_id:
                continue
            display_name = reviewer.get("name", reviewer_id)
            if sources is None:
                src = round_dir / "imports" / f"{reviewer_id}_assignment.db"
            else:
                src = sources.get(reviewer_id)
            if not src or not src.exists():
                statuses.append(f"{display_name}: no submission found")
                continue
            try:
                target_path = self._copy_assignment_to_imports(reviewer_id, src)
            except Exception as exc:  # noqa: BLE001
                statuses.append(f"{display_name}: import failed ({exc})")
                errors += 1
            else:
                self.assignment_paths[reviewer_id] = target_path
                statuses.append(f"{display_name}: imported")
        summary = "\n".join(statuses) if statuses else "No reviewers for this round."
        self._set_import_summary(summary)
        if not silent:
            if errors:
                QtWidgets.QMessageBox.warning(self, "Assignment import", summary)
            else:
                QtWidgets.QMessageBox.information(self, "Assignment import", summary)
        self._refresh_units_table()
        self._update_auto_import_state()

    def _manual_import_assignment(self) -> None:
        if not self.current_round:
            QtWidgets.QMessageBox.information(self, "Assignment import", "Select a round before importing.")
            return
        reviewer_id = self.manual_reviewer_combo.currentData()
        if not reviewer_id:
            QtWidgets.QMessageBox.information(self, "Assignment import", "Select a reviewer to import.")
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
            target_path = self._copy_assignment_to_imports(reviewer_id, Path(path_str))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self,
                "Assignment import",
                f"Failed to import assignment for {display_name}: {exc}",
            )
            return
        self.assignment_paths[reviewer_id] = target_path
        self._set_import_summary(f"{display_name}: imported from manual selection")
        QtWidgets.QMessageBox.information(
            self,
            "Assignment import",
            f"Imported assignment for {display_name}.",
        )
        self._refresh_units_table()
        self._update_auto_import_state()

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

    def _resolve_round_dir(self) -> Optional[Path]:
        if not self.current_round:
            return None
        try:
            project_root = self.ctx.require_project()
        except RuntimeError:
            return None
        pheno_id = self.current_round.get("pheno_id")
        round_number = self.current_round.get("round_number")
        if not pheno_id or round_number is None:
            return None
        return project_root / "phenotypes" / pheno_id / f"rounds/round_{round_number}"

    def _load_manifest(self, round_dir: Path) -> Dict[str, Dict[str, bool]]:
        manifest: Dict[str, Dict[str, bool]] = {}
        manifest_path = round_dir / "manifest.csv"
        if not manifest_path.exists():
            return manifest
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                unit_id = row.get("unit_id")
                reviewer_id = row.get("assigned_to")
                if not unit_id or not reviewer_id:
                    continue
                flag = str(row.get("is_overlap", "")).strip().lower() in {"1", "true", "yes"}
                manifest.setdefault(unit_id, {})[reviewer_id] = flag
        return manifest

    def _discover_existing_imports(self, round_dir: Path) -> None:
        imports_dir = round_dir / "imports"
        if not imports_dir.exists():
            return
        for reviewer in self.current_round.get("reviewers", []):
            reviewer_id = reviewer.get("reviewer_id")
            if not reviewer_id:
                continue
            candidate = imports_dir / f"{reviewer_id}_assignment.db"
            if candidate.exists():
                self.assignment_paths[reviewer_id] = candidate

    def _collect_submission_sources(self, round_dir: Path) -> tuple[Dict[str, Path], Dict[str, str]]:
        sources: Dict[str, Path] = {}
        problems: Dict[str, str] = {}
        if not self.current_round:
            return sources, problems
        for reviewer in self.current_round.get("reviewers", []):
            reviewer_id = reviewer.get("reviewer_id")
            if not reviewer_id:
                continue
            assignment_dir = round_dir / "assignments" / reviewer_id
            receipt = assignment_dir / "submitted.json"
            assignment_db = assignment_dir / "assignment.db"
            if not receipt.exists():
                problems[reviewer_id] = "submission receipt not found"
                continue
            if not assignment_db.exists():
                problems[reviewer_id] = "assignment.db not found"
                continue
            sources[reviewer_id] = assignment_db
        return sources, problems

    def _update_auto_import_state(self) -> None:
        round_dir = self._resolve_round_dir()
        if not round_dir or not self.current_round:
            self.auto_import_btn.setEnabled(False)
            self._set_waiting_summary("")
            return
        sources, problems = self._collect_submission_sources(round_dir)
        self.auto_import_btn.setEnabled(bool(sources) and not problems)
        if problems:
            waiting_parts = [
                f"{self.current_reviewer_names.get(rid, rid)} ({reason})"
                for rid, reason in sorted(
                    problems.items(), key=lambda item: self.current_reviewer_names.get(item[0], item[0])
                )
            ]
            self._set_waiting_summary("Waiting for submissions from: " + ", ".join(waiting_parts))
        else:
            self._set_waiting_summary("")

    def _set_import_summary(self, summary: str) -> None:
        self._import_summary = summary.strip()
        self._update_import_status_label()

    def _set_waiting_summary(self, waiting: str) -> None:
        self._waiting_summary = waiting.strip()
        self._update_import_status_label()

    def _update_import_status_label(self) -> None:
        lines = []
        if self._import_summary:
            lines.append(self._import_summary)
        if self._waiting_summary:
            lines.append(self._waiting_summary)
        self.import_status_label.setText("\n".join(lines))

    def _update_unit_table_headers(self, reviewer_ids: Optional[List[str]] = None) -> None:
        if reviewer_ids is None:
            reviewer_ids = self.reviewer_column_order
        headers = ["Unit", "Patient", "Document", "Overlap", "Status"]
        for reviewer_id in reviewer_ids:
            headers.append(self.current_reviewer_names.get(reviewer_id, reviewer_id))
        self.unit_table.setColumnCount(len(headers))
        self.unit_table.setHorizontalHeaderLabels(headers)

    def _format_annotation_summary(self, annotations: Dict[str, object]) -> str:
        if not annotations:
            return ""
        lines: List[str] = []
        seen: Set[str] = set()
        for label_id in self.label_order:
            if label_id not in annotations:
                continue
            seen.add(label_id)
            lines.append(self._format_annotation_line(label_id, annotations[label_id]))
        remaining = set(annotations.keys()) - seen
        for label_id in sorted(remaining):
            lines.append(self._format_annotation_line(label_id, annotations[label_id]))
        return "\n".join(line for line in lines if line)

    def _format_annotation_line(self, label_id: str, entry: object) -> str:
        label_name = self.label_lookup.get(label_id, label_id)
        display_value = ""
        notes_value = ""
        if isinstance(entry, dict):
            display_value = str(entry.get("display") or "")
            raw_notes = entry.get("notes")
            notes_value = str(raw_notes).strip() if raw_notes else ""
        else:
            display_value = str(entry) if entry is not None else ""
        parts: List[str] = []
        if display_value:
            parts.append(display_value)
        if notes_value:
            parts.append(f"Notes: {notes_value}")
        if not parts:
            parts.append("—")
        return f"{label_name}: {'; '.join(parts)}"

    def _selected_unit_index(self) -> Optional[int]:
        current_row = self.unit_table.currentRow()
        if current_row < 0:
            return None
        item = self.unit_table.item(current_row, 0)
        if not item:
            return None
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(data, int):
            return data
        try:
            return int(data)
        except (TypeError, ValueError):
            return None

    def _selected_unit_id(self) -> Optional[str]:
        index = self._selected_unit_index()
        if index is None or index >= len(self.unit_rows):
            return None
        row = self.unit_rows[index]
        return row.get("unit_id")

    def _clear_document_panel(self) -> None:
        self.document_table.clearContents()
        self.document_table.setRowCount(0)
        self.document_preview.clear()

    def _refresh_units_table(self) -> None:
        selected_unit = self._selected_unit_id()
        existing_discord = {row["unit_id"] for row in self.unit_rows if row.get("discord")}
        self.unit_rows = []
        self.unit_table.clearContents()
        self.unit_table.setRowCount(0)
        self._clear_document_panel()
        if not self.assignment_paths:
            self._update_unit_table_headers()
            return
        for reviewer_id in sorted(self.assignment_paths.keys()):
            if reviewer_id not in self.reviewer_column_order:
                self.reviewer_column_order.append(reviewer_id)
        self.reviewer_column_order.sort(key=lambda rid: self.current_reviewer_names.get(rid, rid).lower())
        unit_map: Dict[str, Dict[str, object]] = {}
        for reviewer_id in self.reviewer_column_order:
            path = self.assignment_paths.get(reviewer_id)
            if not path or not path.exists():
                continue
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                units = conn.execute(
                    "SELECT unit_id, patient_icn, doc_id FROM units ORDER BY display_rank"
                ).fetchall()
                annotations = conn.execute(
                    "SELECT unit_id, label_id, value, value_num, value_date, na, notes FROM annotations"
                ).fetchall()
            ann_map: Dict[str, Dict[str, Dict[str, object]]] = {}
            for ann_row in annotations:
                unit_id = ann_row["unit_id"]
                formatted = self._format_value(ann_row)
                ann_map.setdefault(unit_id, {})[ann_row["label_id"]] = {
                    "display": formatted,
                    "notes": ann_row["notes"] or "",
                    "value": ann_row["value"],
                    "value_num": ann_row["value_num"],
                    "value_date": ann_row["value_date"],
                    "na": ann_row["na"],
                }
            for unit_row in units:
                unit_id = unit_row["unit_id"]
                entry = unit_map.setdefault(
                    unit_id,
                    {
                        "unit_id": unit_id,
                        "patient_icn": unit_row["patient_icn"],
                        "doc_id": unit_row["doc_id"],
                        "reviewer_annotations": {},
                        "reviewer_ids": set(),
                    },
                )
                entry["patient_icn"] = entry.get("patient_icn") or unit_row["patient_icn"]
                entry["doc_id"] = entry.get("doc_id") or unit_row["doc_id"]
                entry["reviewer_ids"].add(reviewer_id)
                entry["reviewer_annotations"][reviewer_id] = ann_map.get(unit_id, {})
        for unit_id, entry in unit_map.items():
            reviewer_ids = sorted(entry["reviewer_ids"], key=lambda rid: self.current_reviewer_names.get(rid, rid))
            entry["reviewer_ids"] = reviewer_ids
            entry["is_overlap"] = self._is_overlap_unit(unit_id, reviewer_ids)
            entry["discord"] = unit_id in existing_discord
            summaries: Dict[str, str] = {}
            for reviewer_id in self.reviewer_column_order:
                annotations = entry["reviewer_annotations"].get(reviewer_id, {})
                summaries[reviewer_id] = self._format_annotation_summary(annotations)
            entry["reviewer_summaries"] = summaries
            self.unit_rows.append(entry)
        for index, row in enumerate(self.unit_rows):
            row["index"] = index
        self._display_unit_rows(selected_unit)

    def _display_unit_rows(self, selected_unit: Optional[str] = None) -> None:
        self.unit_table.clearContents()
        self._update_unit_table_headers()
        if not self.unit_rows:
            self.unit_table.setRowCount(0)
            return
        sorted_rows = sorted(
            self.unit_rows,
            key=lambda row: (not row["is_overlap"], row["unit_id"]),
        )
        self.unit_table.setRowCount(len(sorted_rows))
        highlight = QtGui.QColor("#ffebee")
        for row_index, row in enumerate(sorted_rows):
            items: List[QtWidgets.QTableWidgetItem] = []
            unit_item = QtWidgets.QTableWidgetItem(row["unit_id"])
            unit_item.setData(QtCore.Qt.ItemDataRole.UserRole, row["index"])
            items.append(unit_item)
            patient_item = QtWidgets.QTableWidgetItem(row.get("patient_icn") or "")
            items.append(patient_item)
            doc_item = QtWidgets.QTableWidgetItem(row.get("doc_id") or "—")
            items.append(doc_item)
            overlap_item = QtWidgets.QTableWidgetItem("Yes" if row["is_overlap"] else "No")
            items.append(overlap_item)
            status_item = QtWidgets.QTableWidgetItem("Discordant" if row.get("discord") else "")
            items.append(status_item)
            for reviewer_id in self.reviewer_column_order:
                summary = row["reviewer_summaries"].get(reviewer_id, "")
                summary_item = QtWidgets.QTableWidgetItem(summary)
                summary_item.setToolTip(summary)
                items.append(summary_item)
            for column, item in enumerate(items):
                self.unit_table.setItem(row_index, column, item)
            if row["is_overlap"]:
                for item in items:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
            if row.get("discord"):
                for item in items:
                    item.setBackground(highlight)
            if selected_unit and row["unit_id"] == selected_unit:
                self.unit_table.selectRow(row_index)
            self.unit_table.resizeRowToContents(row_index)
        self.unit_table.resizeColumnsToContents()

    def _on_unit_selected(self) -> None:
        index = self._selected_unit_index()
        if index is None or index >= len(self.unit_rows):
            self._clear_document_panel()
            return
        row = self.unit_rows[index]
        self._populate_document_table(row)

    def _show_annotation_dialog(self, row: int, column: int) -> None:
        if column < self._unit_metadata_column_count:
            return
        item = self.unit_table.item(row, 0)
        if not item:
            return
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        try:
            index = int(data)
        except (TypeError, ValueError):
            return
        if index < 0 or index >= len(self.unit_rows):
            return
        reviewer_offset = column - self._unit_metadata_column_count
        if reviewer_offset < 0 or reviewer_offset >= len(self.reviewer_column_order):
            return
        reviewer_id = self.reviewer_column_order[reviewer_offset]
        row_data = self.unit_rows[index]
        annotations = row_data.get("reviewer_annotations", {}).get(reviewer_id, {})
        detail = self._format_annotation_summary(annotations)
        if not detail:
            detail = "No annotations submitted."
        reviewer_name = self.current_reviewer_names.get(reviewer_id, reviewer_id)
        unit_id = row_data.get("unit_id", "")
        QtWidgets.QMessageBox.information(
            self,
            "Annotation details",
            f"Reviewer: {reviewer_name}\nUnit: {unit_id}\n\n{detail}",
        )

    def _populate_document_table(self, unit_row: Dict[str, object]) -> None:
        self.document_table.clearContents()
        self.document_table.setRowCount(0)
        self.document_preview.clear()
        unit_id = unit_row.get("unit_id")
        if not unit_id:
            return
        reviewer_ids = unit_row.get("reviewer_ids") or []
        assignment_path: Optional[Path] = None
        for reviewer_id in reviewer_ids:
            candidate = self.assignment_paths.get(reviewer_id)
            if candidate and candidate.exists():
                assignment_path = candidate
                break
        if not assignment_path:
            return
        with sqlite3.connect(assignment_path) as conn:
            conn.row_factory = sqlite3.Row
            doc_rows = conn.execute(
                """
                SELECT unit_notes.doc_id, unit_notes.order_index, documents.text
                FROM unit_notes
                LEFT JOIN documents ON documents.doc_id = unit_notes.doc_id
                WHERE unit_notes.unit_id=?
                ORDER BY unit_notes.order_index
                """,
                (unit_id,),
            ).fetchall()
        if not doc_rows:
            return
        metadata: Dict[str, sqlite3.Row] = {}
        corpus_db: Optional[Database] = None
        pheno_id = (self.current_pheno or {}).get("pheno_id")
        if pheno_id:
            try:
                corpus_db = self.ctx.get_corpus_db(pheno_id)
            except Exception:
                corpus_db = None
        if corpus_db:
            with corpus_db.connect() as conn:
                placeholders = ",".join(["?"] * len(doc_rows))
                rows = conn.execute(
                    f"SELECT doc_id, notetype, date_note, sta3n FROM documents WHERE doc_id IN ({placeholders})",
                    [row["doc_id"] for row in doc_rows],
                ).fetchall()
            metadata = {row["doc_id"]: row for row in rows}
        self.document_table.setRowCount(len(doc_rows))
        for idx, doc_row in enumerate(doc_rows):
            doc_id = doc_row["doc_id"]
            meta = metadata.get(doc_id)
            values = [
                doc_id,
                meta["notetype"] if meta else "—",
                meta["date_note"] if meta else "—",
                meta["sta3n"] if meta else "—",
            ]
            for col, value in enumerate(values):
                self.document_table.setItem(idx, col, QtWidgets.QTableWidgetItem(str(value)))
        self.document_table.resizeColumnsToContents()
        if doc_rows:
            self.document_table.selectRow(0)
            self.document_preview.setPlainText(doc_rows[0]["text"] or "")

    def _on_document_selected(self) -> None:
        current_row = self.document_table.currentRow()
        if current_row < 0:
            self.document_preview.clear()
            return
        unit_index = self._selected_unit_index()
        if unit_index is None or unit_index >= len(self.unit_rows):
            self.document_preview.clear()
            return
        reviewer_ids = self.unit_rows[unit_index].get("reviewer_ids") or []
        assignment_path: Optional[Path] = None
        for reviewer_id in reviewer_ids:
            candidate = self.assignment_paths.get(reviewer_id)
            if candidate and candidate.exists():
                assignment_path = candidate
                break
        if not assignment_path:
            self.document_preview.clear()
            return
        unit_id = self.unit_rows[unit_index].get("unit_id")
        if not unit_id:
            self.document_preview.clear()
            return
        with sqlite3.connect(assignment_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT documents.text FROM unit_notes JOIN documents ON documents.doc_id = unit_notes.doc_id "
                "WHERE unit_notes.unit_id=? ORDER BY unit_notes.order_index LIMIT 1 OFFSET ?",
                (unit_id, current_row),
            ).fetchone()
        if row:
            self.document_preview.setPlainText(row["text"] or "")
        else:
            self.document_preview.clear()

    def _is_overlap_unit(self, unit_id: str, reviewer_ids: Iterable[str]) -> bool:
        reviewer_list = list(reviewer_ids)
        manifest_entry = self.round_manifest.get(unit_id)
        if manifest_entry:
            flagged = [rid for rid in reviewer_list if manifest_entry.get(rid)]
            if len(flagged) >= 2:
                return True
            if flagged:
                return False
            return sum(1 for flag in manifest_entry.values() if flag) >= 2
        return len(set(reviewer_list)) > 1

    def _apply_discord_flags(self, discordant_ids: Set[str]) -> None:
        for row in self.unit_rows:
            row["discord"] = row["unit_id"] in discordant_ids
        self._display_unit_rows()

    def _scroll_to_first_discordant(self, discordant_ids: Set[str]) -> None:
        if not discordant_ids:
            return
        for row_index in range(self.unit_table.rowCount()):
            item = self.unit_table.item(row_index, 0)
            if item and item.text() in discordant_ids:
                self.unit_table.scrollToItem(
                    item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter
                )
                break

    def _prepare_agreement_samples(
        self, values_by_unit: Dict[str, Dict[str, str]]
    ) -> tuple[List[AgreementSample], Set[str], List[str]]:
        samples: List[AgreementSample] = []
        discordant_ids: Set[str] = set()
        included_reviewers: Set[str] = set()
        for unit_id, ratings in values_by_unit.items():
            reviewer_ids = tuple(sorted(ratings.keys()))
            if not reviewer_ids:
                continue
            if not self._is_overlap_unit(unit_id, reviewer_ids):
                continue
            if len(reviewer_ids) < 2:
                continue
            included_reviewers.update(reviewer_ids)
            values = tuple(ratings[reviewer_id] for reviewer_id in reviewer_ids)
            samples.append(AgreementSample(unit_id, reviewer_ids, values))
            if len(set(values)) > 1:
                discordant_ids.add(unit_id)
        return samples, discordant_ids, sorted(included_reviewers)

    def _compute_agreement(self) -> None:
        if not self.current_round:
            QtWidgets.QMessageBox.warning(self, "IAA", "Select a round first.")
            return
        label_id = self.label_selector.currentData()
        if not label_id:
            QtWidgets.QMessageBox.information(self, "IAA", "Select a label to evaluate")
            return
        label_id = str(label_id)
        round_dir = self._resolve_round_dir()
        if not round_dir:
            QtWidgets.QMessageBox.warning(self, "IAA", "Round directory is unavailable.")
            return
        aggregate_path = round_dir / "round_aggregate.db"
        if not aggregate_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "IAA",
                "Round aggregate not found. Import assignments and build the aggregate before calculating agreement.",
            )
            return
        self.round_manifest = self._load_manifest(round_dir)
        round_id = self.current_round["round_id"]
        with sqlite3.connect(aggregate_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT unit_id, reviewer_id, value, value_num, value_date, na
                FROM unit_annotations
                WHERE round_id=? AND label_id=?
                ORDER BY unit_id, reviewer_id
                """,
                (round_id, label_id),
            ).fetchall()
        if not rows:
            QtWidgets.QMessageBox.information(
                self,
                "IAA",
                "No annotations found for the selected label.",
            )
            self._apply_discord_flags(set())
            return
        values_by_unit: Dict[str, Dict[str, str]] = {}
        for row in rows:
            unit_id = row["unit_id"]
            reviewer_id = row["reviewer_id"]
            value = self._format_value(row)
            values_by_unit.setdefault(unit_id, {})[reviewer_id] = value
        samples, discordant_ids, reviewer_ids = self._prepare_agreement_samples(values_by_unit)
        if not samples:
            QtWidgets.QMessageBox.information(
                self,
                "IAA",
                "No overlapping units with complete annotations were found.",
            )
            self._apply_discord_flags(set())
            return
        metric = self.metric_selector.currentText()
        result_lines: List[str] = []
        label_name = self.label_lookup.get(label_id, self.label_selector.currentText() or label_id)
        if metric == "Percent agreement":
            value = percent_agreement([list(sample.values) for sample in samples])
            result_lines.append(
                f"Percent agreement: {value:.3%} across {len(samples)} overlapped units"
            )
        elif metric == "Cohen's kappa":
            if len(reviewer_ids) != 2:
                QtWidgets.QMessageBox.warning(self, "IAA", "Cohen's kappa requires exactly two reviewers")
                return
            expected_order = tuple(reviewer_ids)
            rater_a: List[str] = []
            rater_b: List[str] = []
            for sample in samples:
                if sample.reviewer_ids != expected_order:
                    continue
                rater_a.append(sample.values[0])
                rater_b.append(sample.values[1])
            if not rater_a or not rater_b:
                QtWidgets.QMessageBox.warning(
                    self,
                    "IAA",
                    "Insufficient overlapping annotations for Cohen's kappa.",
                )
                return
            value = cohens_kappa(rater_a, rater_b)
            result_lines.append(
                f"Cohen's kappa: {value:.3f} across {len(rater_a)} overlapped units"
            )
        else:
            rater_counts = {len(sample.values) for sample in samples}
            if len(rater_counts) != 1:
                QtWidgets.QMessageBox.warning(
                    self,
                    "IAA",
                    "Fleiss' kappa requires a consistent number of ratings per unit.",
                )
                return
            categories = sorted({value for sample in samples for value in sample.values})
            matrix: List[List[int]] = []
            for sample in samples:
                counts = [sample.values.count(category) for category in categories]
                matrix.append(counts)
            value = fleiss_kappa(matrix)
            result_lines.append(
                f"Fleiss' kappa: {value:.3f} across {len(matrix)} overlapped units"
            )
        known_units = {row.get("unit_id") for row in self.unit_rows}
        filtered_discordant = {unit_id for unit_id in discordant_ids if unit_id in known_units}
        result_lines.append(f"Discordant units: {len(filtered_discordant)}")
        heading = f"Agreement for {label_name}"
        result_text = "\n".join([heading, *result_lines])
        QtWidgets.QMessageBox.information(self, "IAA results", result_text)
        self.round_summary.setText(result_text)
        self._apply_discord_flags(filtered_discordant)
        self._scroll_to_first_discordant(filtered_discordant)

    @staticmethod
    def _format_value(row: sqlite3.Row) -> str:
        if row["na"]:
            return "N/A"
        if row["value_num"] is not None:
            return format(row["value_num"], "g")
        if row["value"] is not None and row["value"] != "":
            return str(row["value"])
        if row["value_date"]:
            return row["value_date"]
        return ""



class AdminMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ctx = ProjectContext()
        self.setWindowTitle("VAAnnotate Admin")
        self.resize(1280, 860)
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
        self.tree = ProjectTreeWidget(self.ctx)
        splitter.addWidget(self.tree)

        self.stack = QtWidgets.QStackedWidget()
        self.project_view = ProjectOverviewWidget(self.ctx)
        self.pheno_view = PhenotypeDetailWidget()
        self.round_view = RoundDetailWidget()
        self.corpus_view = CorpusWidget(self.ctx)
        self.iaa_view = IaaWidget(self.ctx)
        self.stack.addWidget(self.project_view)
        self.stack.addWidget(self.pheno_view)
        self.stack.addWidget(self.round_view)
        self.stack.addWidget(self.corpus_view)
        self.stack.addWidget(self.iaa_view)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)

        self.view_index = {
            "project": self.stack.indexOf(self.project_view),
            "phenotype": self.stack.indexOf(self.pheno_view),
            "round": self.stack.indexOf(self.round_view),
            "corpus": self.stack.indexOf(self.corpus_view),
            "iaa": self.stack.indexOf(self.iaa_view),
        }
        self.tree.node_selected.connect(self._on_node_selected)

    def _show_view(self, key: str) -> None:
        index = self.view_index.get(key, self.view_index["project"])
        self.stack.setCurrentIndex(index)

    def _on_node_selected(self, data: Dict[str, object]) -> None:
        node_type = data.get("type")
        if node_type == "project":
            self.project_view.set_project(data.get("project"))
            self._show_view("project")
        elif node_type == "phenotype":
            pheno = data.get("pheno")
            self.pheno_view.set_phenotype(pheno if isinstance(pheno, dict) else None)
            self._show_view("phenotype")
        elif node_type == "round":
            round_row = data.get("round")
            if isinstance(round_row, dict):
                config = self.ctx.get_round_config(round_row.get("round_id", ""))
                self.round_view.set_round(round_row, config)
            else:
                self.round_view.set_round(None, None)
            self._show_view("round")
        elif node_type == "corpus":
            pheno = data.get("pheno")
            self.corpus_view.set_phenotype(pheno if isinstance(pheno, dict) else None)
            self._show_view("corpus")
        elif node_type == "iaa":
            pheno = data.get("pheno")
            self.iaa_view.set_phenotype(pheno if isinstance(pheno, dict) else None)
            self._show_view("iaa")
        else:
            self.project_view.set_project(self.ctx.project_row)
            self._show_view("project")

    def _open_project(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select project folder")
        if not directory:
            return
        self.ctx.open_project(Path(directory))
        self.project_view.set_project(self.ctx.project_row)
        self.tree.refresh()


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = AdminMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
