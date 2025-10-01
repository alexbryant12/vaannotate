"""PySide6 based Admin application for VAAnnotate."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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
        splitter.addWidget(self.round_list)

        right = QtWidgets.QWidget()
        splitter.addWidget(right)
        form = QtWidgets.QFormLayout(right)

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

        form.addRow("Phenotype", self.pheno_combo)
        form.addRow("Label set ID", self.labelset_edit)
        form.addRow("Seed", self.seed_spin)
        form.addRow("Overlap N", self.overlap_spin)
        form.addRow("Sample per reviewer", self.sample_spin)
        form.addRow("Status", self.status_combo)
        form.addRow(self.create_btn)

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
        filters = SamplingFilters(patient_filters={}, note_filters={})
        corpus_rows = candidate_documents(ctx.require_corpus_db(), "single_doc", filters)
        if not corpus_rows:
            QtWidgets.QMessageBox.warning(self, "No corpus", "The corpus database has no documents to sample")
            return
        assignments = allocate_units(corpus_rows[: self.sample_spin.value() * len(reviewers)], reviewers, overlap, seed)
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
            config = models.RoundConfig(round_id=round_id, config_json=json.dumps({"seed": seed}))
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
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.metric_selector = QtWidgets.QComboBox()
        self.metric_selector.addItems(["Percent agreement", "Cohen's kappa", "Fleiss' kappa"])
        splitter.addWidget(self.metric_selector)

        self.results_view = QtWidgets.QTextEdit()
        splitter.addWidget(self.results_view)
        splitter.setStretchFactor(1, 3)

        run_btn = QtWidgets.QPushButton("Compute from sample data")
        run_btn.clicked.connect(self._run_example)
        layout.addWidget(run_btn)

    def _run_example(self) -> None:
        sample = [["yes", "yes", "no"], ["no", "no", "no"], ["yes", "yes", "yes"]]
        metric = self.metric_selector.currentText()
        if metric == "Percent agreement":
            value = percent_agreement(sample)
        elif metric == "Cohen's kappa":
            value = cohens_kappa([row[0] for row in sample], [row[1] for row in sample])
        else:
            matrix = []
            categories = sorted({item for row in sample for item in row})
            for row in sample:
                counts = [row.count(cat) for cat in categories]
                matrix.append(counts)
            value = fleiss_kappa(matrix)
        self.results_view.setPlainText(f"{metric}: {value:.3f}")

    def refresh(self) -> None:
        # No dynamic content yet; placeholder to satisfy the navigation stack
        pass


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
        nav_widget.addItems(["Projects", "Phenotypes", "Rounds", "IAA"])
        nav_widget.currentRowChanged.connect(self._switch_page)
        splitter.addWidget(nav_widget)

        self.stack = QtWidgets.QStackedWidget()
        self.pages = [
            ProjectsPage(self.ctx),
            PhenotypePage(self.ctx),
            RoundPage(self.ctx),
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
