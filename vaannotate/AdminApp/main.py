"""PySide6 based Admin application for VAAnnotate."""
from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import threading
import re
import sqlite3
import sys
import uuid
import tempfile
from collections.abc import Mapping as ABCMapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Mapping, Type, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from vaannotate.schema import ROUND_AGG_SCHEMA
from vaannotate.shared import models
from vaannotate.shared.database import Database, ensure_schema
from vaannotate.shared.metadata import (
    MetadataField,
    MetadataFilterCondition,
    discover_corpus_metadata,
)
from vaannotate.shared.sampling import (
    ReviewerAssignment,
    SamplingFilters,
    allocate_units,
    candidate_documents,
    populate_assignment_db,
    write_manifest,
)
from vaannotate.shared.statistics import cohens_kappa, fleiss_kappa, percent_agreement
from vaannotate.shared.theme import apply_dark_palette
from vaannotate.AdminApp.prompt_builder import (
    PromptBuilderConfig,
    PromptExperimentConfig,
    PromptExperimentSweep,
    PromptInferenceJob,
)
from vaannotate.corpus import TABULAR_EXTENSIONS, import_tabular_corpus
from vaannotate.project import (
    build_label_config,
    fetch_labelset,
    init_project,
    resolve_label_config_path,
)
from vaannotate.utils import copy_sqlite_database, ensure_dir
from vaannotate.rounds import AssignmentUnit as RoundAssignmentUnit, RoundBuilder
from vaannotate.vaannotate_ai_backend import (
    CancelledError,
    BackendResult,
    engine,
    run_ai_backend_and_collect,
)

PROJECT_MODELS = [
    models.Project,
    models.Phenotype,
    models.ProjectCorpus,
    models.LabelSet,
    models.Label,
    models.LabelOption,
    models.Round,
    models.RoundConfig,
    models.Reviewer,
    models.Assignment,
]

ASSIGNMENT_MODELS = [
    models.AssignmentUnit,
    models.AssignmentUnitNote,
    models.AssignmentDocument,
    models.Annotation,
    models.Rationale,
    models.Event,
]


def build_round_assignment_units(
    assignments: Mapping[str, ReviewerAssignment],
) -> Dict[str, List[RoundAssignmentUnit]]:
    """Normalize sampling assignments into round assignment units."""

    reviewer_assignments: Dict[str, List[RoundAssignmentUnit]] = {}
    for reviewer_id, assignment in assignments.items():
        units: List[RoundAssignmentUnit] = []
        for unit in assignment.units:
            payload = dict(unit)
            payload.setdefault("documents", payload.get("documents") or [])
            payload.setdefault(
                "strata_key",
                payload.get("strata_key")
                or payload.get("strata")
                or "random_sampling",
            )
            unit_identifier = (
                payload.get("unit_id")
                or payload.get("doc_id")
                or payload.get("patient_icn")
            )
            unit_id = str(unit_identifier or "")
            if not unit_id:
                continue
            doc_id_value = payload.get("doc_id")
            doc_id = None if doc_id_value is None else str(doc_id_value)
            patient_icn = str(payload.get("patient_icn") or "")
            units.append(
                RoundAssignmentUnit(
                    unit_id,
                    patient_icn,
                    doc_id,
                    payload,
                )
            )
        if units:
            reviewer_assignments[reviewer_id] = units
    return reviewer_assignments


@dataclass(frozen=True)
class AgreementSample:
    unit_id: str
    reviewer_ids: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True)
class LabelDefinition:
    label_id: str
    name: str
    type: str
    na_allowed: bool
    unit: Optional[str]
    min_value: Optional[float]
    max_value: Optional[float]
    options: List[Dict[str, object]]
    rules: str = ""


@dataclass
class RoundCreationContext:
    pheno_id: str
    pheno_level: str
    project_id: str
    phenotype_storage_path: Optional[str]
    seed: int
    overlap: int
    total_n: int
    status: str
    labelset_id: str
    labelset_missing: bool
    default_labels: List[Dict[str, object]]
    reviewers: List[Dict[str, str]]
    corpus_id: str
    corpus_record: Optional[sqlite3.Row]
    created_at: str
    created_by: str
    db: Database
    assisted_review_enabled: bool = False
    assisted_review_top_n: int = 0
    ai_backend_overrides: Dict[str, object] = field(default_factory=dict)


@dataclass
class AIRoundJobConfig:
    context: RoundCreationContext
    round_number: int
    round_id: str
    round_dir: Path
    prior_rounds: List[int]
    timestamp: str


class AIRoundLogDialog(QtWidgets.QDialog):
    cancel_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI backend progress")
        self.resize(640, 480)
        self._allow_close = False
        layout = QtWidgets.QVBoxLayout(self)
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.button_box)
        self._close_button = self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self._cancel_button = self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        if self._close_button:
            self._close_button.setEnabled(False)
        if self._cancel_button:
            self._cancel_button.clicked.connect(self.cancel_requested.emit)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if not self._allow_close:
            event.ignore()
            return
        super().closeEvent(event)

    def reset_for_run(self) -> None:
        self.log_output.clear()
        self._allow_close = False
        if self._close_button:
            self._close_button.setEnabled(False)
        if self._cancel_button:
            self._cancel_button.setEnabled(True)

    def mark_complete(self) -> None:
        self._allow_close = True
        if self._close_button:
            self._close_button.setEnabled(True)
        if self._cancel_button:
            self._cancel_button.setEnabled(False)


def _deep_update_dict(target: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into target and return the target dict."""

    for key, value in updates.items():
        if isinstance(value, Mapping):
            current = target.get(key)
            if not isinstance(current, Mapping):
                current = {}
            target[key] = _deep_update_dict(dict(current), value)
        else:
            target[key] = copy.deepcopy(value)
    return target


AI_CONFIG_TOOLTIPS: Dict[str, Dict[str, str]] = {
    "index": {
        "type": "FAISS index type: flat (brute force), hnsw (graph), or ivf (quantized inverted lists).",
        "nlist": "Number of clusters/lists for IVF indexes.",
        "nprobe": "How many IVF lists to search at query time (higher = better recall).",
        "hnsw_M": "HNSW graph degree (edges per node).",
        "hnsw_efSearch": "HNSW search breadth; larger trades memory for recall.",
        "persist": "Persist built FAISS indexes to disk for reuse.",
    },
    "rag": {
        "chunk_size": "Maximum characters per text chunk before embedding.",
        "chunk_overlap": "Overlap between consecutive chunks to preserve context.",
        "normalize_embeddings": "L2-normalize embeddings before indexing/searching.",
        "per_label_topk": "Top-k chunks to keep per label during retrieval.",
        "use_mmr": "Use maximal marginal relevance to diversify retrieved chunks.",
        "mmr_lambda": "MMR trade-off between relevance (1.0) and diversity (0.0).",
        "mmr_candidates": "Candidate pool size for MMR selection.",
        "use_keywords": "Blend keyword search results into retrieval.",
        "keyword_topk": "How many keyword hits to include when enabled.",
        "keywords": "Comma-separated keywords to seed lexical retrieval across all labels.",
        "label_keywords": "Optional JSON mapping of label_id → keywords (list or comma-separated string) for BM25 search.",
        "min_context_chunks": "Minimum chunks of context to pass to the LLM.",
        "mmr_multiplier": "Scale the number of chunks considered for MMR diversification.",
        "neighbor_hops": "How many hops to explore around selected chunks for neighbors.",
    },
    "llm": {
        "model_name": "LLM deployment/model identifier.",
        "backend": "LLM backend to use: azure or exllamav2 (local).",
        "temperature": "Sampling temperature for LLM calls.",
        "n_consistency": "Self-consistency samples for JSON calls.",
        "logprobs": "Request log-probabilities from the LLM when available.",
        "top_logprobs": "Top-N token log-probs to request.",
        "prediction_field": "JSON field containing the model prediction.",
        "timeout": "Per-request timeout (seconds).",
        "retry_max": "Maximum LLM retry attempts.",
        "retry_backoff": "Exponential backoff base (seconds) between retries.",
        "max_context_chars": "Guardrail on total characters passed to the LLM.",
        "rpm_limit": "Requests-per-minute throttle; leave 0 to disable.",
        "include_reasoning": "Ask the LLM to return reasoning/rationale text.",
        "few_shot_examples": "Optional JSON mapping of label_id → examples, each with 'context' and JSON 'answer'.",
        "azure_api_version": "Azure OpenAI API version to target.",
        "azure_endpoint": "Custom Azure OpenAI endpoint URL.",
        "local_model_dir": "Path to the local ExLlamaV2 model directory.",
        "local_max_seq_len": "Override maximum sequence length for local models (0=default).",
        "local_max_new_tokens": "Override maximum new tokens for local generation (0=default).",
    },
    "select": {
        "batch_size": "Number of candidate units to select for the next round.",
        "pct_disagreement": "Fraction pulled from reviewer disagreement bucket.",
        "pct_uncertain": "Fraction pulled from LLM-uncertain bucket.",
        "pct_easy_qc": "Fraction pulled from LLM-certain/easy QC bucket.",
        "pct_diversity": "Fraction pulled from diversity bucket.",
    },
    "llmfirst": {
        "n_probe_units": "How many units to probe when estimating uncertainty.",
        "topk": "Top-K relevant chunks to feed into LLM-first probes.",
        "json_trace_policy": "Fallback policy for JSON traces (e.g., 'fallback').",
        "progress_min_interval_s": "Minimum seconds between progress updates.",
        "exemplar_K": "Number of exemplar documents to retrieve per label.",
        "exemplar_generate": "Generate synthetic exemplars when none exist.",
        "exemplar_temperature": "Sampling temperature for exemplar generation.",
        "fc_enable": "Enable forced-choice micro-probe scoring.",
        "enrich": "Enrich probes with additional context/details.",
        "probe_enrichment_mix": "Blend between enriched vs. uniform probe selection (0-1).",
        "probe_enrichment_equalize": "Equalize enrichment per parent label instead of proportional.",
        "probe_ce_unit_sample": "How many units to sample for cross-encoder scoring.",
        "probe_ce_search_topk_per_unit": "Top-K search hits per unit for CE reranking.",
        "probe_ce_rerank_m": "Aggregate top-M cross-encoder scores per unit.",
        "probe_ce_unit_agg": "Aggregation mode for CE scores (max/mean).",
        "single_doc_context": "Context strategy for single-document phenotypes (rag/full).",
        "single_doc_full_context_max_chars": "Max characters when using full-document context.",
    },
    "disagree": {
        "round_policy": "How to weight prior rounds when measuring disagreement (last/all/decay).",
        "decay_half_life": "Half-life (rounds) when round_policy=decay.",
        "high_entropy_threshold": "Entropy threshold for treating disagreements as high entropy.",
        "seeds_per_label": "Seed examples per label for disagreement expansion.",
        "snippets_per_seed": "Relevant snippets to collect per seed case.",
        "similar_chunks_per_seed": "Neighbor chunks to pull per seed during expansion.",
        "expanded_per_label": "Expanded cases to include per label.",
        "date_disagree_days": "Days apart to treat date labels as conflicting.",
        "numeric_disagree_abs": "Absolute numeric delta that counts as disagreement.",
        "numeric_disagree_rel": "Relative numeric delta that counts as disagreement (0-1).",
    },
    "diversity": {
        "rag_k": "Chunks to retrieve per label for diversity sampling.",
        "min_rel_quantile": "Minimum relevance quantile to consider for diversity sampling.",
        "mmr_lambda": "Diversity MMR weighting (1=relevance, 0=diversity).",
        "sample_cap": "Maximum candidates to sample from diversity bucket.",
        "adaptive_relax": "Relax diversity thresholds when few candidates are found.",
        "use_proto": "Use prototype-based representations when building diversity candidates.",
    },
    "scjitter": {
        "enable": "Apply stochastic jittering when sampling easy/uncertain buckets.",
        "rag_topk_range": "Range of RAG top-k values to sample from (min,max).",
        "rag_dropout_p": "Probability of dropping a retrieved chunk.",
        "temperature_range": "Range of LLM temperatures sampled for jittering.",
        "shuffle_context": "Shuffle retrieved context before prompting the LLM.",
    },
    "orchestrator": {
        "final_llm_labeling": "Run a final LLM labeling pass after selection.",
        "final_llm_labeling_n_consistency": "Self-consistency passes for final labeling (≥1).",
    },
}


class AIAdvancedConfigDialog(QtWidgets.QDialog):
    """Dialog that surfaces all engine.py configuration options."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], config: Mapping[str, Any]) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI backend advanced settings")
        self.resize(760, 820)
        self._config: Dict[str, Any] = dict(config)
        self.result_config: Dict[str, Any] = {}
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(6, 6, 6, 6)

        self._section_widgets: Dict[str, Dict[str, QtWidgets.QWidget]] = {}

        sections: list[tuple[str, str]] = [
            ("Indexing", "index"),
            ("Retrieval (RAG)", "rag"),
            ("LLM", "llm"),
            ("Selection buckets", "select"),
            ("LLM-first probing", "llmfirst"),
            ("Disagreement", "disagree"),
            ("Diversity", "diversity"),
            ("Self-consistency jitter", "scjitter"),
        ]

        for title, key in sections:
            section_values = self._config.get(key, {}) if isinstance(self._config.get(key), Mapping) else {}
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            widgets: Dict[str, QtWidgets.QWidget] = {}
            for field_name, value in section_values.items():
                widget = self._build_field_widget(key, field_name, value)
                widgets[field_name] = widget
                label_text = field_name.replace("_", " ").capitalize()
                form.addRow(label_text, widget)
            self._section_widgets[key] = widgets
            container_layout.addWidget(group)

        orch_group = QtWidgets.QGroupBox("Orchestrator")
        orch_form = QtWidgets.QFormLayout(orch_group)
        orch_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._orch_widgets: Dict[str, QtWidgets.QWidget] = {}
        for field_name in ("final_llm_labeling", "final_llm_labeling_n_consistency"):
            value = self._config.get(field_name)
            widget = self._build_field_widget("orchestrator", field_name, value)
            self._orch_widgets[field_name] = widget
            label_text = field_name.replace("_", " ").capitalize()
            orch_form.addRow(label_text, widget)
        container_layout.addWidget(orch_group)

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _build_field_widget(self, section: str, name: str, value: Any) -> QtWidgets.QWidget:
        tooltip = AI_CONFIG_TOOLTIPS.get(section, {}).get(name, "")
        widget: QtWidgets.QWidget
        if name == "type" and section == "index":
            combo = QtWidgets.QComboBox()
            for label, data in (("Flat", "flat"), ("HNSW", "hnsw"), ("IVF", "ivf")):
                combo.addItem(label, data)
            idx = combo.findData(str(value))
            combo.setCurrentIndex(max(0, idx))
            widget = combo
        elif name == "backend" and section == "llm":
            combo = QtWidgets.QComboBox()
            combo.addItem("Azure OpenAI", "azure")
            combo.addItem("Local ExLlamaV2", "exllamav2")
            idx = combo.findData(str(value))
            combo.setCurrentIndex(max(0, idx))
            widget = combo
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            edit = QtWidgets.QLineEdit()
            edit.setText(", ".join(str(v) for v in value))
            edit.setPlaceholderText("min,max")
            edit.setProperty("value_type", "tuple")
            edit.setProperty(
                "tuple_cast",
                float if any(isinstance(v, float) for v in value) else int,
            )
            edit.setProperty("tuple_factory", tuple if isinstance(value, tuple) else list)
            widget = edit
        elif name in {"few_shot_examples", "label_keywords"}:
            edit = QtWidgets.QPlainTextEdit()
            try:
                edit.setPlainText(json.dumps(value, indent=2, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                edit.setPlainText("" if value is None else str(value))
            edit.setPlaceholderText(
                "JSON mapping of label_id to [{\"context\": \"...\", \"answer\": \"{\\\"prediction\\\": ...}\"}]"
                if name == "few_shot_examples"
                else "JSON mapping of label_id to keyword lists (or comma-separated strings)."
            )
            edit.setProperty("value_type", "json_object")
            widget = edit
        elif isinstance(value, list):
            edit = QtWidgets.QLineEdit()
            edit.setText(", ".join(str(v) for v in value))
            edit.setPlaceholderText("comma-separated values")
            edit.setProperty("value_type", "list")
            widget = edit
        elif isinstance(value, bool):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(value))
            widget = checkbox
        elif isinstance(value, float):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.05)
            spin.setValue(float(value))
            widget = spin
        elif isinstance(value, int):
            spin = QtWidgets.QSpinBox()
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setValue(int(value))
            widget = spin
        else:
            edit = QtWidgets.QLineEdit()
            edit.setText("" if value is None else str(value))
            widget = edit
        if tooltip:
            widget.setToolTip(tooltip)
        return widget

    def _collect_section_values(self, widgets: Mapping[str, QtWidgets.QWidget]) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for key, widget in widgets.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                values[key] = bool(widget.isChecked())
            elif isinstance(widget, QtWidgets.QComboBox):
                data = widget.currentData()
                values[key] = data if data is not None else widget.currentText()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                values[key] = float(widget.value())
            elif isinstance(widget, QtWidgets.QSpinBox):
                values[key] = int(widget.value())
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                text = widget.toPlainText().strip()
                if widget.property("value_type") == "json_object":
                    if not text:
                        values[key] = {}
                    else:
                        try:
                            values[key] = json.loads(text)
                        except Exception:  # noqa: BLE001
                            values[key] = text
                else:
                    values[key] = text
            elif isinstance(widget, QtWidgets.QLineEdit):
                if widget.property("value_type") == "tuple":
                    text = widget.text().strip()
                    parts = [p.strip() for p in text.split(",") if p.strip()]
                    cast = widget.property("tuple_cast") or str
                    factory = widget.property("tuple_factory") or tuple
                    try:
                        values[key] = factory(cast(p) for p in parts)
                    except Exception:  # noqa: BLE001
                        values[key] = text
                elif widget.property("value_type") == "list":
                    text = widget.text().strip()
                    parts = [p.strip() for p in re.split(r"[,\n]", text) if p.strip()]
                    values[key] = parts
                else:
                    values[key] = widget.text().strip()
        return values

    def _on_accept(self) -> None:
        result: Dict[str, Any] = {}
        for section, widgets in self._section_widgets.items():
            result[section] = self._collect_section_values(widgets)
        result.update(self._collect_section_values(self._orch_widgets))
        self.result_config = result
        self.accept()

class AIRoundWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    log_message = QtCore.Signal(str)

    def __init__(
        self,
        project_root: Path,
        job: AIRoundJobConfig,
        finalize: bool,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        cleanup_dir: bool = False,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.job = job
        self.finalize = finalize
        self.cfg_overrides = dict(cfg_overrides or {})
        self.cleanup_dir = cleanup_dir
        self.env_overrides = {str(key): str(value)
                              for key, value in (env_overrides or {}).items()
                              if str(value)}
        self._cancel_event = threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    @QtCore.Slot()
    def cancel(self) -> None:  # noqa: D401 - Qt slot
        if not self._cancel_event.is_set():
            self._cancel_event.set()
            self.log_message.emit("Cancellation requested…")

    @QtCore.Slot()
    def run(self) -> None:  # noqa: D401 - Qt slot
        try:
            ensure_dir(self.job.round_dir)
            label_config_payload, label_config_path = self._prepare_label_config()
            if self.finalize and label_config_payload and label_config_path:
                try:
                    self._write_label_config(label_config_path, label_config_payload)
                except Exception as exc:  # noqa: BLE001
                    self.log_message.emit(f"Warning: failed to write label_config.json ({exc})")
            original_env: Dict[str, Optional[str]] = {}
            try:
                for key, value in self.env_overrides.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value
                result = run_ai_backend_and_collect(
                    self.project_root,
                    self.job.context.pheno_id,
                    self.job.context.labelset_id,
                    self.job.prior_rounds,
                    self.job.round_dir,
                    self.job.context.pheno_level,
                    self.job.context.created_by,
                    timestamp=self.job.timestamp,
                    cfg_overrides=self.cfg_overrides,
                    label_config=label_config_payload,
                    log_callback=self.log_message.emit,
                    cancel_callback=self._cancel_event.is_set,
                    corpus_record=self.job.context.corpus_record,
                    corpus_id=self.job.context.corpus_id,
                )
            finally:
                for key, prior in original_env.items():
                    if prior is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = prior
            if self._cancel_event.is_set():
                raise CancelledError("AI backend run cancelled")
            payload: Dict[str, object] = {"backend_result": result}
            if self.finalize:
                if self._cancel_event.is_set():
                    raise CancelledError("AI backend run cancelled")
                self.log_message.emit("Finalizing round artifacts…")
                self._prepare_labelset_and_reviewers()
                if self._cancel_event.is_set():
                    raise CancelledError("AI backend run cancelled")
                try:
                    self.job.context.db.flush_to_disk()
                except Exception as exc:  # noqa: BLE001
                    self.log_message.emit(
                        f"Warning: unable to flush project database before build ({exc})"
                    )
                config_path = self._write_round_config(result)
                try:
                    if self._cancel_event.is_set():
                        raise CancelledError("AI backend run cancelled")
                    builder = RoundBuilder(self.project_root)
                    build_result = builder.generate_round(
                        self.job.context.pheno_id,
                        config_path,
                        created_by=self.job.context.created_by,
                        preselected_units_csv=result.csv_path,
                        env_overrides=self.env_overrides,
                    )
                    try:
                        self.job.context.db.refresh_from_disk()
                    except Exception as exc:  # noqa: BLE001
                        self.log_message.emit(
                            f"Warning: unable to refresh project database after build ({exc})"
                        )
                finally:
                    try:
                        config_path.unlink()
                    except OSError:
                        pass
                payload["build_result"] = build_result
            self.finished.emit(payload, None)
        except CancelledError:
            self.log_message.emit("AI backend run cancelled.")
            self.finished.emit({"cancelled": True}, None)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(None, exc)
        finally:
            if self.cleanup_dir:
                shutil.rmtree(self.job.round_dir, ignore_errors=True)

    def _prepare_label_config(self) -> Tuple[Optional[Dict[str, object]], Optional[Path]]:
        context = self.job.context
        project_db = self.project_root / "project.db"
        try:
            with sqlite3.connect(project_db) as conn:
                conn.row_factory = sqlite3.Row
                labelset = fetch_labelset(conn, context.labelset_id)
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"Warning: unable to build label_config ({exc})")
            return None, None
        generated = build_label_config(labelset)
        config_path = resolve_label_config_path(self.project_root, context.labelset_id)
        existing_payload: Dict[str, object] = {}
        for candidate in self._label_config_candidates(context, config_path):
            if not candidate.exists():
                continue
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing_payload = loaded
                    if candidate != config_path:
                        self.log_message.emit(
                            f"Info: loaded legacy label_config.json from {candidate}"
                        )
                    break
            except Exception as exc:  # noqa: BLE001
                self.log_message.emit(
                    f"Warning: failed to parse existing label_config.json ({exc})"
                )
        merged = self._merge_label_config(existing_payload, generated)
        return merged, config_path

    def _label_config_candidates(
        self, context: RoundCreationContext, new_path: Path
    ) -> Iterable[Path]:
        yield new_path
        legacy = self._resolve_phenotype_dir(context) / "ai" / "label_config.json"
        if legacy != new_path:
            yield legacy

    def _resolve_phenotype_dir(self, context: RoundCreationContext) -> Path:
        storage = context.phenotype_storage_path
        if storage:
            candidate = Path(storage)
            if not candidate.is_absolute():
                candidate = (self.project_root / candidate).resolve()
        else:
            candidate = self.project_root / "phenotypes" / context.pheno_id
        return ensure_dir(candidate)

    @staticmethod
    def _merge_label_config(
        existing: Dict[str, object], generated: Dict[str, object]
    ) -> Dict[str, object]:
        merged: Dict[str, object] = dict(existing)
        for key, value in generated.items():
            if key == "_meta":
                meta: Dict[str, object] = {}
                existing_meta = merged.get("_meta")
                if isinstance(existing_meta, dict):
                    meta.update(existing_meta)
                if isinstance(value, dict):
                    meta.update(value)
                merged["_meta"] = meta
                continue
            if isinstance(value, dict):
                entry: Dict[str, object] = {}
                existing_entry = merged.get(key)
                if isinstance(existing_entry, dict):
                    entry.update(existing_entry)
                entry.update(value)
                merged[key] = entry
            else:
                merged[key] = value
        return merged

    def _write_label_config(self, path: Path, payload: Dict[str, object]) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log_message.emit(f"Updated label_config.json at {path}")

    def _prepare_labelset_and_reviewers(self) -> None:
        context = self.job.context
        with context.db.transaction() as conn:
            if context.labelset_missing:
                labelset = models.LabelSet(
                    labelset_id=context.labelset_id,
                    project_id=context.project_id,
                    pheno_id=context.pheno_id,
                    version=1,
                    created_at=context.created_at,
                    created_by=context.created_by,
                    notes="Auto-generated",
                )
                labelset.save(conn)
                for label in context.default_labels:
                    label_record = models.Label(
                        label_id=label["label_id"],
                        labelset_id=context.labelset_id,
                        name=label["name"],
                        type=label["type"],
                        required=label.get("required", False),
                        order_index=0,
                        rules=label.get("rules", ""),
                        gating_expr=None,
                        na_allowed=int(label.get("na_allowed", False)),
                        unit=None,
                        min=None,
                        max=None,
                    )
                    label_record.save(conn)
                    for idx, option in enumerate(label.get("options", [])):
                        option_record = models.LabelOption(
                            option_id=str(uuid.uuid4()),
                            labelset_id=context.labelset_id,
                            label_id=label_record.label_id,
                            value=str(option.get("value", "")),
                            display=str(option.get("display", option.get("value", ""))),
                            order_index=idx,
                            weight=option.get("weight"),
                        )
                        option_record.save(conn)
            for reviewer in context.reviewers:
                reviewer_record = models.Reviewer(
                    reviewer_id=reviewer["id"],
                    name=reviewer.get("name", reviewer["id"]),
                    email=reviewer.get("email", ""),
                    windows_account=None,
                )
                reviewer_record.save(conn)

    def _write_round_config(self, backend_result: BackendResult) -> Path:
        context = self.job.context
        csv_path = backend_result.csv_path
        payload: Dict[str, Any] = {
            "pheno_id": context.pheno_id,
            "labelset_id": context.labelset_id,
            "corpus_id": context.corpus_id,
            "round_number": self.job.round_number,
            "round_id": self.job.round_id,
            "rng_seed": context.seed,
            "overlap_n": context.overlap,
            "total_n": context.total_n,
            "status": context.status,
            "reviewers": context.reviewers,
            "preselected_units_csv": str(csv_path),
            "prior_rounds": list(self.job.prior_rounds),
        }
        if context.corpus_record:
            try:
                payload["corpus_name"] = context.corpus_record["name"]
            except Exception:  # noqa: BLE001
                payload["corpus_name"] = context.corpus_record.get("name")  # type: ignore[call-arg]
            try:
                payload["corpus_path"] = context.corpus_record["relative_path"]
            except Exception:  # noqa: BLE001
                payload["corpus_path"] = context.corpus_record.get("relative_path")  # type: ignore[call-arg]
        ai_backend_payload: Dict[str, object] = {
            "prior_rounds": list(self.job.prior_rounds),
            "invoked_at": self.job.timestamp,
        }
        if context.ai_backend_overrides:
            ai_backend_payload.update(context.ai_backend_overrides)
        payload["ai_backend"] = ai_backend_payload
        if context.assisted_review_enabled:
            payload["assisted_review"] = {
                "enabled": True,
                "top_snippets": context.assisted_review_top_n,
            }
        artifacts = backend_result.artifacts or {}
        final_labels_path = artifacts.get("final_labels")
        if final_labels_path:
            payload["ai_backend"]["final_llm_labels"] = str(final_labels_path)
        final_labels_json = artifacts.get("final_labels_json")
        if final_labels_json:
            payload["ai_backend"]["final_llm_labels_json"] = str(final_labels_json)
        final_probe = artifacts.get("final_family_probe")
        if final_probe:
            payload["ai_backend"]["final_llm_family_probe"] = str(final_probe)
        final_probe_json = artifacts.get("final_family_probe_json")
        if final_probe_json:
            payload["ai_backend"]["final_llm_family_probe_json"] = str(final_probe_json)
        if "final_llm_labeling" in self.cfg_overrides:
            payload["final_llm_labeling"] = bool(self.cfg_overrides.get("final_llm_labeling"))
        if "final_llm_include_reasoning" in self.cfg_overrides:
            payload["final_llm_include_reasoning"] = bool(
                self.cfg_overrides.get("final_llm_include_reasoning")
            )
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with handle:
            json.dump(payload, handle, indent=2)
        return Path(handle.name)
class ProjectContext(QtCore.QObject):
    project_changed = QtCore.Signal()
    dirty_changed = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.project_root: Optional[Path] = None
        self.project_db: Optional[Database] = None
        self.project_row: Optional[Dict[str, object]] = None
        self._corpus_cache: Dict[str, Database] = {}
        self._external_db_cache: Dict[Path, Database] = {}
        self._managed_dbs: List[Database] = []
        self._corpus_metadata_cache: Dict[str, List[MetadataField]] = {}
        self._pending_manifests: Dict[Path, Dict[str, ReviewerAssignment]] = {}
        self._pending_text_writes: Dict[Path, str] = {}
        self._pending_deletions: Set[Tuple[str, Path]] = set()
        self._dirty_flag = False
        self._last_dirty_state = False

    def _register_db(self, db: Database) -> Database:
        if db not in self._managed_dbs:
            self._managed_dbs.append(db)
        return db

    def _cache_database(self, db: Database) -> Database:
        db.enable_memory_cache()
        return self._register_db(db)

    def _shutdown_caches(self) -> None:
        for cached_db in self._managed_dbs:
            cached_db.close()
        self._corpus_cache.clear()
        self._external_db_cache.clear()
        self._managed_dbs = []
        self._corpus_metadata_cache.clear()
        self._pending_manifests.clear()
        self._pending_text_writes.clear()
        self._pending_deletions.clear()
        self._dirty_flag = False
        self._emit_dirty_state()

    def _emit_dirty_state(self) -> None:
        dirty = self.has_unsaved_changes()
        if dirty != self._last_dirty_state:
            self._last_dirty_state = dirty
            self.dirty_changed.emit(dirty)

    def _mark_dirty(self) -> None:
        self._dirty_flag = True
        self._emit_dirty_state()

    def mark_dirty(self) -> None:
        self._mark_dirty()

    @staticmethod
    def _path_is_within(candidate: Path, parent: Path) -> bool:
        try:
            candidate.relative_to(parent)
            return True
        except ValueError:
            return False

    def _clear_pending_artifacts(self, target: Path) -> None:
        resolved_target = target.resolve()
        for mapping in (self._pending_text_writes, self._pending_manifests):
            for path in list(mapping.keys()):
                resolved_candidate = path.resolve()
                if resolved_candidate == resolved_target or self._path_is_within(
                    resolved_candidate, resolved_target
                ):
                    mapping.pop(path, None)

    def _clear_pending_deletions(self, target: Path) -> None:
        resolved_target = target.resolve()
        for action, candidate in list(self._pending_deletions):
            resolved_candidate = candidate.resolve()
            if resolved_candidate == resolved_target or self._path_is_within(
                resolved_target, resolved_candidate
            ):
                self._pending_deletions.discard((action, candidate))

    def _schedule_deletion(self, path: Path, *, mode: str = "auto") -> None:
        resolved = path.resolve()
        action = mode
        if mode == "auto":
            action = "tree" if resolved.exists() and resolved.is_dir() else "file"
        task = (action, resolved)
        if task in self._pending_deletions:
            return
        self._pending_deletions.add(task)
        self._clear_pending_artifacts(resolved)

    def has_unsaved_changes(self) -> bool:
        if (
            self._dirty_flag
            or self._pending_manifests
            or self._pending_text_writes
            or self._pending_deletions
        ):
            return True
        return any(db.is_dirty for db in self._managed_dbs)

    def save_all(self) -> None:
        errors: List[str] = []
        for db in list(self._managed_dbs):
            try:
                db.flush_to_disk()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{db.path}: {exc}")
        for path, content in list(self._pending_text_writes.items()):
            try:
                ensure_dir(path.parent)
                path.write_text(content, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")
        for path, assignments in list(self._pending_manifests.items()):
            try:
                ensure_dir(path.parent)
                write_manifest(path, assignments)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")
        for action, target in sorted(
            self._pending_deletions, key=lambda item: len(item[1].parts), reverse=True
        ):
            try:
                if action == "tree":
                    if target.exists():
                        shutil.rmtree(target, ignore_errors=True)
                elif action == "rmdir":
                    try:
                        target.rmdir()
                    except OSError:
                        pass
                else:
                    if target.exists():
                        target.unlink()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{target}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
        self._pending_text_writes.clear()
        self._pending_manifests.clear()
        self._pending_deletions.clear()
        self._dirty_flag = False
        self._emit_dirty_state()

    def _get_cached_db(
        self,
        path: Path,
        *,
        models_schema: Sequence[Type[models.Record]] | None = None,
        statements: Sequence[str] | None = None,
    ) -> Database:
        resolved = path.resolve()
        cached = self._external_db_cache.get(resolved)
        if cached:
            return cached
        db = self._cache_database(Database(resolved))
        if models_schema:
            with db.transaction() as conn:
                ensure_schema(conn, models_schema)
        if statements:
            with db.transaction() as conn:
                for statement in statements:
                    try:
                        conn.executescript(statement)
                    except sqlite3.OperationalError as exc:
                        message = str(exc).lower()
                        if "duplicate column" in message:
                            continue
                        raise
        self._external_db_cache[resolved] = db
        return db

    def get_assignment_db(self, path: Path, *, create: bool = False) -> Optional[Database]:
        resolved = path.resolve()
        cached = self._external_db_cache.get(resolved)
        if cached:
            return cached
        if not path.exists() and not create:
            return None
        db = self._get_cached_db(resolved, models_schema=ASSIGNMENT_MODELS)
        self._ensure_assignment_metadata_columns(db)
        return db

    def _ensure_assignment_metadata_columns(self, db: Database) -> None:
        try:
            with db.connect() as conn:
                info_rows = conn.execute("PRAGMA table_info(documents)").fetchall()
                if not any(row["name"] == "metadata_json" for row in info_rows):
                    conn.execute("ALTER TABLE documents ADD COLUMN metadata_json TEXT")
                    conn.commit()
        except sqlite3.DatabaseError:
            return

    def prepare_assignment_db(self, path: Path) -> Database:
        resolved = path.resolve()
        existing = self._external_db_cache.pop(resolved, None)
        if existing:
            existing.close()
            if existing in self._managed_dbs:
                self._managed_dbs.remove(existing)
        db = self._cache_database(Database(resolved))
        with db.transaction() as conn:
            ensure_schema(conn, ASSIGNMENT_MODELS)
            for table in [
                "units",
                "unit_notes",
                "documents",
                "annotations",
                "rationales",
                "events",
            ]:
                conn.execute(f"DELETE FROM {table}")
        self._external_db_cache[resolved] = db
        self._mark_dirty()
        return db

    def refresh_assignment_cache(self, path: Path) -> Optional[Database]:
        resolved = path.resolve()
        existing = self._external_db_cache.pop(resolved, None)
        if existing:
            existing.close()
            if existing in self._managed_dbs:
                self._managed_dbs.remove(existing)
        if not path.exists():
            return None
        return self.get_assignment_db(path)

    def get_round_aggregate_db(self, round_dir: Path, *, create: bool = False) -> Optional[Database]:
        path = (round_dir / "round_aggregate.db").resolve()
        cached = self._external_db_cache.get(path)
        if cached:
            return cached
        if not path.exists() and not create:
            return None
        return self._get_cached_db(path, statements=ROUND_AGG_SCHEMA)

    @staticmethod
    def _manifest_from_assignments(
        assignments: Dict[str, ReviewerAssignment]
    ) -> Dict[str, Dict[str, bool]]:
        manifest: Dict[str, Dict[str, bool]] = {}
        for reviewer_id, assignment in assignments.items():
            for unit in assignment.units:
                unit_id = str(unit.get("unit_id"))
                if not unit_id:
                    continue
                manifest.setdefault(unit_id, {})[reviewer_id] = bool(unit.get("is_overlap"))
        return manifest

    def register_manifest(self, path: Path, assignments: Dict[str, ReviewerAssignment]) -> None:
        resolved = path.resolve()
        self._clear_pending_deletions(resolved)
        self._pending_manifests[resolved] = copy.deepcopy(assignments)
        self._mark_dirty()

    def get_manifest_flags(self, path: Path) -> Dict[str, Dict[str, bool]]:
        resolved = path.resolve()
        pending = self._pending_manifests.get(resolved)
        if pending is not None:
            return self._manifest_from_assignments(pending)
        manifest: Dict[str, Dict[str, bool]] = {}
        if not path.exists():
            return manifest
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                unit_id = row.get("unit_id")
                reviewer_id = row.get("assigned_to")
                if not unit_id or not reviewer_id:
                    continue
                flag = str(row.get("is_overlap", "")).strip().lower() in {"1", "true", "yes"}
                manifest.setdefault(unit_id, {})[reviewer_id] = flag
        return manifest

    def register_text_file(self, path: Path, content: str) -> None:
        resolved = path.resolve()
        self._clear_pending_deletions(resolved)
        self._pending_text_writes[resolved] = content
        self._mark_dirty()

    def _preload_round_assets(self) -> None:
        if not self.project_root:
            return
        try:
            corpora = list(self.list_corpora())
            phenotypes = list(self.list_phenotypes())
        except Exception:  # noqa: BLE001
            return
        for corpus in corpora:
            corpus_id: Optional[object]
            if isinstance(corpus, sqlite3.Row):
                corpus_id = corpus["corpus_id"] if "corpus_id" in corpus.keys() else None
            elif isinstance(corpus, Mapping):
                corpus_id = corpus.get("corpus_id")
            else:
                try:
                    corpus_id = corpus["corpus_id"]  # type: ignore[index]
                except Exception:  # noqa: BLE001
                    corpus_id = None
            if not corpus_id:
                continue
            try:
                self.get_corpus_db(str(corpus_id))
            except Exception:  # noqa: BLE001
                continue
        for pheno in phenotypes:
            pheno_id: Optional[object]
            if isinstance(pheno, sqlite3.Row):
                pheno_id = pheno["pheno_id"] if "pheno_id" in pheno.keys() else None
            elif isinstance(pheno, Mapping):
                pheno_id = pheno.get("pheno_id")
            else:
                try:
                    pheno_id = pheno["pheno_id"]  # type: ignore[index]
                except Exception:  # noqa: BLE001
                    pheno_id = None
            if not pheno_id:
                continue
            try:
                pheno_dir = self.resolve_phenotype_dir(str(pheno_id))
            except Exception:  # noqa: BLE001
                continue
            rounds_dir = pheno_dir / "rounds"
            if not rounds_dir.exists():
                continue
            for round_dir in sorted(rounds_dir.glob("round_*")):
                if round_dir.is_dir():
                    agg_db = self.get_round_aggregate_db(round_dir)
                    _ = agg_db  # force cache load
                    imports_dir = round_dir / "imports"
                    if imports_dir.exists():
                        for assignment_path in imports_dir.glob("*_assignment.db"):
                            self.get_assignment_db(assignment_path)

    def open_project(self, directory: Path) -> None:
        self._shutdown_caches()
        directory = directory.resolve()
        project_db = self._cache_database(Database(directory / "project.db"))
        with project_db.transaction() as conn:
            ensure_schema(conn, PROJECT_MODELS)
        self.project_root = directory
        self.project_db = project_db
        self.project_row = self._load_project_row()
        self._preload_round_assets()
        self._emit_dirty_state()
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

    def list_label_sets(self) -> List[sqlite3.Row]:
        db = self.require_db()
        project_id = self.current_project_id()
        if not project_id:
            return []
        with db.connect() as conn:
            return conn.execute(
                "SELECT * FROM label_sets WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()

    def get_labelset(self, labelset_id: str) -> Optional[sqlite3.Row]:
        db = self.require_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM label_sets WHERE labelset_id=?",
                (labelset_id,),
            ).fetchone()
        return row

    def load_labelset_details(self, labelset_id: str) -> Optional[Dict[str, object]]:
        db = self.require_db()
        with db.connect() as conn:
            try:
                return fetch_labelset(conn, labelset_id)
            except ValueError:
                return None

    def list_all_label_ids(self) -> Set[str]:
        db = self.require_db()
        with db.connect() as conn:
            rows = conn.execute("SELECT label_id FROM labels").fetchall()
        return {str(row["label_id"]) for row in rows if row["label_id"]}

    def list_label_sets_for_pheno(self, pheno_id: str) -> List[Dict[str, object]]:
        db = self.require_db()
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT ls.*, COUNT(lbl.label_id) AS label_count
                FROM label_sets ls
                LEFT JOIN labels lbl ON lbl.labelset_id = ls.labelset_id
                WHERE ls.pheno_id = ? OR ls.labelset_id IN (
                    SELECT labelset_id FROM rounds WHERE pheno_id = ?
                )
                GROUP BY ls.labelset_id
                ORDER BY ls.created_at DESC
                """,
                (pheno_id, pheno_id),
            ).fetchall()
            round_counts = conn.execute(
                """
                SELECT labelset_id, COUNT(*) AS round_count
                FROM rounds
                WHERE pheno_id=?
                GROUP BY labelset_id
                """,
                (pheno_id,),
            ).fetchall()
        round_map = {str(row["labelset_id"]): int(row["round_count"]) for row in round_counts}
        results: List[Dict[str, object]] = []
        for row in rows:
            record = dict(row)
            labelset_id = str(record.get("labelset_id") or "")
            label_count = int(record.get("label_count") or 0)
            assigned = str(record.get("pheno_id") or "") == pheno_id
            results.append(
                {
                    "labelset": record,
                    "labelset_id": labelset_id,
                    "label_count": label_count,
                    "round_count": round_map.get(labelset_id, 0),
                    "assigned_to_pheno": assigned,
                }
            )
        return results

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

    def list_corpora(self) -> List[sqlite3.Row]:
        project_id = self.current_project_id()
        if not project_id:
            return []
        db = self.require_db()
        with db.connect() as conn:
            return conn.execute(
                "SELECT * FROM project_corpora WHERE project_id=? ORDER BY name",
                (project_id,),
            ).fetchall()

    def get_corpus(self, corpus_id: str) -> Optional[sqlite3.Row]:
        db = self.require_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM project_corpora WHERE corpus_id=?",
                (corpus_id,),
            ).fetchone()
        return row

    def resolve_corpus_path(self, corpus_id: str) -> Path:
        row = self.get_corpus(corpus_id)
        if not row:
            raise RuntimeError(f"Corpus {corpus_id} not found")
        relative = str(row["relative_path"])
        if not relative:
            raise RuntimeError("Corpus record is missing a storage path")
        root = self.require_project()
        return (root / relative).resolve()

    def get_corpus_db(self, corpus_id: str) -> Database:
        if corpus_id in self._corpus_cache:
            return self._corpus_cache[corpus_id]
        path = self.resolve_corpus_path(corpus_id)
        db = self._cache_database(Database(path))
        with db.transaction() as conn:
            ensure_schema(conn, [models.Patient, models.Document])
        self._corpus_cache[corpus_id] = db
        return db

    def get_corpus_metadata_fields(self, corpus_id: str) -> List[MetadataField]:
        cached = self._corpus_metadata_cache.get(corpus_id)
        if cached is not None:
            return cached
        db = self.get_corpus_db(corpus_id)
        try:
            fields = discover_corpus_metadata(db)
        except Exception:
            fields = []
        self._corpus_metadata_cache[corpus_id] = fields
        return fields

    def import_corpus(self, source_path: Path, *, name: Optional[str] = None) -> models.ProjectCorpus:
        project_id = self.current_project_id()
        if not project_id:
            raise RuntimeError("Project metadata missing; ensure a project record exists")
        project_root = self.require_project()
        corpora_root = ensure_dir(project_root / "corpora")
        base_name = name or source_path.stem or "corpus"
        slug = re.sub(r"[^a-z0-9]+", "_", base_name.lower()).strip("_") or "corpus"
        candidate = slug
        counter = 2
        while (corpora_root / candidate).exists():
            candidate = f"{slug}_{counter}"
            counter += 1
        corpus_dir = ensure_dir(corpora_root / candidate)
        target_corpus = corpus_dir / "corpus.db"
        if source_path.suffix.lower() in TABULAR_EXTENSIONS:
            import_tabular_corpus(source_path, target_corpus)
        else:
            copy_sqlite_database(source_path, target_corpus)
        relative_corpus = target_corpus.relative_to(project_root)
        record = models.ProjectCorpus(
            corpus_id=str(uuid.uuid4()),
            project_id=project_id,
            name=base_name,
            relative_path=str(relative_corpus.as_posix()),
            created_at=datetime.utcnow().isoformat(),
        )
        db = self.require_db()
        with db.transaction() as conn:
            record.save(conn)
        self._mark_dirty()
        self.project_changed.emit()
        return record

    def create_phenotype(
        self,
        *,
        name: str,
        level: str,
        description: str,
    ) -> models.Phenotype:
        project_id = self.current_project_id()
        if not project_id:
            raise RuntimeError("Project metadata missing; ensure a project record exists")
        pheno_id = str(uuid.uuid4())
        project_root = self.require_project()
        phenotype_dir = self._ensure_phenotype_dir(name)
        rounds_dir = ensure_dir(phenotype_dir / "rounds")
        _ = rounds_dir  # make mypy happy about unused variable
        relative_storage = phenotype_dir.relative_to(project_root)
        record = models.Phenotype(
            pheno_id=pheno_id,
            project_id=project_id,
            name=name,
            level=level,
            description=description,
            storage_path=str(relative_storage.as_posix()),
        )
        db = self.require_db()
        with db.transaction() as conn:
            record.save(conn)
        self._mark_dirty()
        self.project_changed.emit()
        return record

    def create_labelset(
        self,
        *,
        labelset_id: str,
        created_by: str,
        notes: str,
        labels: List[Dict[str, object]],
        pheno_id: Optional[str] = None,
    ) -> models.LabelSet:
        project_id = self.current_project_id()
        if not project_id:
            raise RuntimeError("Project metadata missing; ensure a project record exists")
        created_at = QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)
        record = models.LabelSet(
            labelset_id=labelset_id,
            project_id=project_id,
            pheno_id=pheno_id,
            version=1,
            created_at=created_at,
            created_by=created_by,
            notes=notes,
        )
        db = self.require_db()
        with db.transaction() as conn:
            record.save(conn)
            for order_index, label in enumerate(labels):
                label_record = models.Label(
                    label_id=label["label_id"],
                    labelset_id=labelset_id,
                    name=label["name"],
                    type=label["type"],
                    required=1 if label.get("required") else 0,
                    order_index=order_index,
                    rules=label.get("rules", ""),
                    gating_expr=label.get("gating_expr"),
                    na_allowed=1 if label.get("na_allowed") else 0,
                    unit=label.get("unit"),
                    min=label.get("min"),
                    max=label.get("max"),
                )
                label_record.save(conn)
                for opt_index, option in enumerate(label.get("options", [])):
                    option_record = models.LabelOption(
                        option_id=option.get("option_id") or str(uuid.uuid4()),
                        labelset_id=labelset_id,
                        label_id=label_record.label_id,
                        value=str(option.get("value", "")),
                        display=str(option.get("display", option.get("value", ""))),
                        order_index=opt_index,
                        weight=option.get("weight"),
                    )
                    option_record.save(conn)
        self._mark_dirty()
        self.project_changed.emit()
        return record

    def _ensure_phenotype_dir(self, name: str) -> Path:
        project_root = self.require_project()
        phenotypes_root = ensure_dir(project_root / "phenotypes")
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "phenotype"
        candidate = slug
        counter = 2
        while (phenotypes_root / candidate).exists():
            candidate = f"{slug}_{counter}"
            counter += 1
        return ensure_dir(phenotypes_root / candidate)

    def resolve_phenotype_dir(self, pheno_id: str) -> Path:
        pheno = self.get_phenotype(pheno_id)
        if not pheno:
            raise RuntimeError(f"Phenotype {pheno_id} not found")
        storage_path = Path(str(pheno["storage_path"]))
        if storage_path.is_absolute():
            phenotype_dir = storage_path
        else:
            project_root = self.require_project()
            phenotype_dir = (project_root / storage_path).resolve()
        return phenotype_dir

    def resolve_round_dir(self, pheno_id: str, round_number: int) -> Path:
        phenotype_dir = self.resolve_phenotype_dir(pheno_id)
        return phenotype_dir / "rounds" / f"round_{round_number}"

    def update_cache_after_round(self, corpus_id: str) -> None:
        # Keep API parity with previous refresh pattern
        self._corpus_cache.pop(corpus_id, None)
        self.project_changed.emit()

    def update_round_status(self, round_id: str, status: str) -> None:
        valid_statuses = {"draft", "active", "closed", "adjudicating", "finalized"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid round status: {status}")
        db = self.require_db()
        with db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE rounds SET status=? WHERE round_id=?",
                (status, round_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Round {round_id} not found")
            config_row = conn.execute(
                "SELECT config_json FROM round_configs WHERE round_id=?",
                (round_id,),
            ).fetchone()
            if config_row:
                try:
                    config_payload = json.loads(config_row["config_json"])
                except json.JSONDecodeError:
                    config_payload = None
                if isinstance(config_payload, dict):
                    config_payload["status"] = status
                    conn.execute(
                        "UPDATE round_configs SET config_json=? WHERE round_id=?",
                        (json.dumps(config_payload, indent=2), round_id),
                    )
        self._mark_dirty()
        self.project_changed.emit()

    def delete_corpus(self, corpus_id: str, *, delete_files: bool = True) -> None:
        record = self.get_corpus(corpus_id)
        if not record:
            raise RuntimeError(f"Corpus {corpus_id} not found")
        target_path: Optional[Path] = None
        parent_dir: Optional[Path] = None
        if delete_files:
            relative = str(record.get("relative_path") or "")
            if relative:
                try:
                    target_path = self.resolve_project_path(relative)
                    parent_dir = target_path.parent
                except Exception:
                    target_path = None
        db = self.require_db()
        with db.transaction() as conn:
            conn.execute("DELETE FROM project_corpora WHERE corpus_id=?", (corpus_id,))
        self._corpus_cache.pop(corpus_id, None)
        self._corpus_metadata_cache.pop(corpus_id, None)
        self._mark_dirty()
        if delete_files and target_path:
            mode = "tree" if target_path.exists() and target_path.is_dir() else "file"
            self._schedule_deletion(target_path, mode=mode)
            if parent_dir:
                self._schedule_deletion(parent_dir, mode="rmdir")
        self.project_changed.emit()

    def delete_round(
        self,
        round_id: str,
        *,
        delete_files: bool = True,
        emit_signal: bool = True,
    ) -> None:
        round_row = self.get_round(round_id)
        if not round_row:
            raise RuntimeError(f"Round {round_id} not found")
        pheno_id = str(round_row["pheno_id"])
        round_number_raw = round_row["round_number"]
        round_dir: Optional[Path] = None
        if delete_files:
            try:
                round_number = int(round_number_raw)
            except (TypeError, ValueError):
                round_number = None
            if round_number is not None:
                try:
                    round_dir = self.resolve_round_dir(pheno_id, round_number)
                except Exception:
                    round_dir = None
        db = self.require_db()
        with db.transaction() as conn:
            conn.execute("DELETE FROM assignments WHERE round_id=?", (round_id,))
            conn.execute("DELETE FROM round_configs WHERE round_id=?", (round_id,))
            conn.execute("DELETE FROM rounds WHERE round_id=?", (round_id,))
        self._mark_dirty()
        if delete_files and round_dir:
            self._schedule_deletion(round_dir, mode="tree")
        if emit_signal:
            self.project_changed.emit()

    def delete_phenotype(self, pheno_id: str, *, delete_files: bool = True) -> None:
        pheno = self.get_phenotype(pheno_id)
        if not pheno:
            raise RuntimeError(f"Phenotype {pheno_id} not found")
        phenotype_dir: Optional[Path] = None
        if delete_files:
            try:
                phenotype_dir = self.resolve_phenotype_dir(pheno_id)
            except Exception:
                phenotype_dir = None
        for round_row in self.list_rounds(pheno_id):
            round_id = str(round_row["round_id"])
            self.delete_round(round_id, delete_files=delete_files, emit_signal=False)
        db = self.require_db()
        with db.transaction() as conn:
            conn.execute("DELETE FROM phenotypes WHERE pheno_id=?", (pheno_id,))
        self._mark_dirty()
        if delete_files and phenotype_dir:
            self._schedule_deletion(phenotype_dir, mode="tree")
        self.project_changed.emit()


class PhenotypeDialog(QtWidgets.QDialog):
    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Add phenotype")
        self.resize(400, 260)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.level_combo = QtWidgets.QComboBox()
        self.level_combo.addItems(["single_doc", "multi_doc"])
        self.description_edit = QtWidgets.QPlainTextEdit()
        form.addRow("Name", self.name_edit)
        form.addRow("Level", self.level_combo)
        form.addRow("Description", self.description_edit)
        layout.addLayout(form)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def accept(self) -> None:  # noqa: D401 - Qt override
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Validation", "Phenotype name is required.")
            return
        super().accept()

    def values(self) -> Dict[str, object]:
        return {
            "name": self.name_edit.text().strip(),
            "level": self.level_combo.currentText(),
            "description": self.description_edit.toPlainText().strip(),
        }


class LabelEditorDialog(QtWidgets.QDialog):
    TYPE_CHOICES: List[tuple[str, str]] = [
        ("categorical_single", "Categorical (single choice)"),
        ("categorical_multi", "Categorical (multi choice)"),
        ("ordinal", "Ordinal"),
        ("boolean", "Boolean"),
        ("integer", "Integer"),
        ("float", "Float"),
        ("date", "Date"),
        ("text", "Free text"),
    ]

    def __init__(
        self,
        *,
        existing_ids: Optional[Set[str]] = None,
        data: Optional[Dict[str, object]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Label details")
        self.resize(480, 520)
        self._existing_ids = existing_ids or set()
        self._initial_id = str(data.get("label_id")) if data and data.get("label_id") else None
        self._setup_ui()
        if data:
            self._load_data(data)
        self._update_field_visibility()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.label_id_edit = QtWidgets.QLineEdit()
        form.addRow("Label ID", self.label_id_edit)
        self.name_edit = QtWidgets.QLineEdit()
        form.addRow("Display name", self.name_edit)
        self.type_combo = QtWidgets.QComboBox()
        for value, label in self.TYPE_CHOICES:
            self.type_combo.addItem(label, value)
        self.type_combo.currentIndexChanged.connect(self._update_field_visibility)
        form.addRow("Type", self.type_combo)
        self.required_check = QtWidgets.QCheckBox("Required")
        form.addRow("Required", self.required_check)
        self.na_check = QtWidgets.QCheckBox("Allow N/A")
        form.addRow("N/A", self.na_check)
        self.gating_edit = QtWidgets.QLineEdit()
        form.addRow("Gating expression", self.gating_edit)
        self.rules_edit = QtWidgets.QPlainTextEdit()
        form.addRow("Annotation rules", self.rules_edit)
        self.unit_edit = QtWidgets.QLineEdit()
        form.addRow("Unit", self.unit_edit)
        self.min_edit = QtWidgets.QLineEdit()
        self.max_edit = QtWidgets.QLineEdit()
        range_layout = QtWidgets.QHBoxLayout()
        range_layout.addWidget(self.min_edit)
        range_layout.addWidget(QtWidgets.QLabel("to"))
        range_layout.addWidget(self.max_edit)
        form.addRow("Range", range_layout)
        layout.addLayout(form)

        self.options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout(self.options_group)
        self.options_table = QtWidgets.QTableWidget(0, 3)
        self.options_table.setHorizontalHeaderLabels(["Value", "Display", "Weight"])
        self.options_table.horizontalHeader().setStretchLastSection(True)
        self.options_table.verticalHeader().setVisible(False)
        self.options_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.options_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        options_layout.addWidget(self.options_table)
        options_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add option")
        add_btn.clicked.connect(self._add_option)
        remove_btn = QtWidgets.QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_option)
        options_buttons.addWidget(add_btn)
        options_buttons.addWidget(remove_btn)
        options_buttons.addStretch(1)
        options_layout.addLayout(options_buttons)
        layout.addWidget(self.options_group)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _load_data(self, data: Dict[str, object]) -> None:
        self.label_id_edit.setText(str(data.get("label_id", "")))
        self.name_edit.setText(str(data.get("name", "")))
        type_value = str(data.get("type", ""))
        index = self.type_combo.findData(type_value)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)
        self.required_check.setChecked(bool(data.get("required")))
        self.na_check.setChecked(bool(data.get("na_allowed")))
        self.gating_edit.setText(str(data.get("gating_expr", "")))
        self.rules_edit.setPlainText(str(data.get("rules", "")))
        self.unit_edit.setText(str(data.get("unit") or ""))
        self.min_edit.setText("" if data.get("min") is None else str(data.get("min")))
        self.max_edit.setText("" if data.get("max") is None else str(data.get("max")))
        for option in data.get("options", []):
            row = self.options_table.rowCount()
            self.options_table.insertRow(row)
            self.options_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(option.get("value", ""))))
            self.options_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(option.get("display", ""))))
            weight = option.get("weight")
            self.options_table.setItem(row, 2, QtWidgets.QTableWidgetItem("" if weight is None else str(weight)))

    def _update_field_visibility(self) -> None:
        type_value = self.type_combo.currentData()
        requires_options = type_value in {"categorical_single", "categorical_multi", "ordinal", "boolean"}
        self.options_group.setVisible(requires_options)
        is_numeric = type_value in {"integer", "float"}
        self.unit_edit.setEnabled(is_numeric)
        self.min_edit.setEnabled(is_numeric)
        self.max_edit.setEnabled(is_numeric)

    def _add_option(self) -> None:
        value, ok = QtWidgets.QInputDialog.getText(self, "Add option", "Value")
        if not ok or not value.strip():
            return
        display, ok = QtWidgets.QInputDialog.getText(self, "Add option", "Display", text=value)
        if not ok:
            return
        weight_text, ok = QtWidgets.QInputDialog.getText(self, "Add option", "Weight (optional)")
        if not ok:
            return
        row = self.options_table.rowCount()
        self.options_table.insertRow(row)
        self.options_table.setItem(row, 0, QtWidgets.QTableWidgetItem(value.strip()))
        self.options_table.setItem(row, 1, QtWidgets.QTableWidgetItem(display.strip()))
        self.options_table.setItem(row, 2, QtWidgets.QTableWidgetItem(weight_text.strip()))

    def _remove_option(self) -> None:
        row = self.options_table.currentRow()
        if row >= 0:
            self.options_table.removeRow(row)

    def _collect_options(self) -> List[Dict[str, object]]:
        options: List[Dict[str, object]] = []
        for row in range(self.options_table.rowCount()):
            value_item = self.options_table.item(row, 0)
            display_item = self.options_table.item(row, 1)
            weight_item = self.options_table.item(row, 2)
            value = value_item.text().strip() if value_item else ""
            if not value:
                continue
            display = display_item.text().strip() if display_item else value
            weight_text = weight_item.text().strip() if weight_item else ""
            weight: Optional[float]
            if weight_text:
                try:
                    weight = float(weight_text)
                except ValueError:
                    weight = None
            else:
                weight = None
            options.append(
                {
                    "value": value,
                    "display": display or value,
                    "weight": weight,
                }
            )
        return options

    def _parse_float(self, text: str) -> Optional[float]:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError("Enter numeric values for range bounds")

    def accept(self) -> None:  # noqa: D401 - Qt override
        label_id = self.label_id_edit.text().strip()
        if not label_id:
            QtWidgets.QMessageBox.warning(self, "Label", "Label ID is required.")
            return
        existing = self._existing_ids - ({self._initial_id} if self._initial_id else set())
        if label_id in existing:
            QtWidgets.QMessageBox.warning(self, "Label", "Another label already uses this ID.")
            return
        if not self.name_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "Label", "Display name is required.")
            return
        type_value = self.type_combo.currentData()
        if not type_value:
            QtWidgets.QMessageBox.warning(self, "Label", "Select a label type.")
            return
        if type_value in {"categorical_single", "categorical_multi", "ordinal", "boolean"}:
            options = self._collect_options()
            if len(options) < 1:
                QtWidgets.QMessageBox.warning(self, "Label", "Add at least one option for the selected type.")
                return
        try:
            _ = self._parse_float(self.min_edit.text())
            _ = self._parse_float(self.max_edit.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Label", str(exc))
            return
        super().accept()

    def values(self) -> Dict[str, object]:
        type_value = str(self.type_combo.currentData())
        options = self._collect_options() if type_value in {"categorical_single", "categorical_multi", "ordinal", "boolean"} else []
        min_value: Optional[float]
        max_value: Optional[float]
        try:
            min_value = self._parse_float(self.min_edit.text())
            max_value = self._parse_float(self.max_edit.text())
        except ValueError:
            min_value = max_value = None
        return {
            "label_id": self.label_id_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "type": type_value,
            "required": self.required_check.isChecked(),
            "na_allowed": self.na_check.isChecked(),
            "gating_expr": self.gating_edit.text().strip() or None,
            "rules": self.rules_edit.toPlainText().strip(),
            "unit": self.unit_edit.text().strip() or None,
            "min": min_value,
            "max": max_value,
            "options": options,
        }


class LabelSetWizardDialog(QtWidgets.QDialog):
    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.labels: List[Dict[str, object]] = []
        self.setWindowTitle("Create label set")
        self.resize(520, 640)
        self._setup_ui()
        self._populate_copy_sources()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.id_edit = QtWidgets.QLineEdit()
        self.id_edit.setPlaceholderText("Unique label set ID")
        form.addRow("Label set ID", self.id_edit)
        self.copy_combo = QtWidgets.QComboBox()
        self.copy_combo.addItem("Start from blank", None)
        self.copy_combo.currentIndexChanged.connect(self._on_copy_source_changed)
        form.addRow("Copy from", self.copy_combo)
        self.creator_edit = QtWidgets.QLineEdit()
        creator_default = "admin"
        if self.ctx.project_row and self.ctx.project_row.get("created_by"):
            creator_default = str(self.ctx.project_row["created_by"])
        self.creator_edit.setText(creator_default)
        form.addRow("Created by", self.creator_edit)
        self.notes_edit = QtWidgets.QPlainTextEdit()
        form.addRow("Notes", self.notes_edit)
        layout.addLayout(form)

        self.label_list = QtWidgets.QListWidget()
        self.label_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.label_list)

        button_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add label")
        add_btn.clicked.connect(self._add_label)
        edit_btn = QtWidgets.QPushButton("Edit label")
        edit_btn.clicked.connect(self._edit_label)
        remove_btn = QtWidgets.QPushButton("Remove label")
        remove_btn.clicked.connect(self._remove_label)
        up_btn = QtWidgets.QPushButton("Move up")
        up_btn.clicked.connect(lambda: self._move_label(-1))
        down_btn = QtWidgets.QPushButton("Move down")
        down_btn.clicked.connect(lambda: self._move_label(1))
        button_row.addWidget(add_btn)
        button_row.addWidget(edit_btn)
        button_row.addWidget(remove_btn)
        button_row.addWidget(up_btn)
        button_row.addWidget(down_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _populate_copy_sources(self) -> None:
        if not hasattr(self, "copy_combo"):
            return
        self.copy_combo.blockSignals(True)
        self.copy_combo.clear()
        self.copy_combo.addItem("Start from blank", None)
        for row in self.ctx.list_label_sets():
            labelset_id = str(row["labelset_id"])
            created_at = ""
            if "created_at" in row.keys() and row["created_at"]:
                created_at = str(row["created_at"])
            notes = ""
            if "notes" in row.keys() and row["notes"]:
                notes = str(row["notes"]).strip()
            display = labelset_id
            if created_at:
                display += f" ({created_at})"
            if notes:
                display += f" — {notes}"
            self.copy_combo.addItem(display, labelset_id)
        self.copy_combo.blockSignals(False)
        self.copy_combo.setCurrentIndex(0)

    def _on_copy_source_changed(self, index: int) -> None:
        if index <= 0:
            return
        labelset_id = self.copy_combo.itemData(index)
        if not isinstance(labelset_id, str):
            return
        payload = self.ctx.load_labelset_details(labelset_id)
        if not payload:
            return
        self._apply_copied_labelset(payload)

    def _generate_unique_label_id(self, base_id: str, used_ids: Set[str]) -> str:
        sanitized = re.sub(r"\s+", "_", base_id.strip()) if base_id.strip() else "label"
        candidate = sanitized
        if candidate in used_ids:
            suffix = 2
            candidate = f"{sanitized}_copy"
            while candidate in used_ids:
                candidate = f"{sanitized}_{suffix}"
                suffix += 1
        used_ids.add(candidate)
        return candidate

    def _apply_copied_labelset(self, payload: Dict[str, object]) -> None:
        used_ids = set(self.ctx.list_all_label_ids())
        new_labels: List[Dict[str, object]] = []
        id_map: Dict[str, str] = {}
        raw_labels = payload.get("labels") or []
        if isinstance(raw_labels, list):
            for label in raw_labels:
                if not isinstance(label, (dict, ABCMapping)):
                    continue
                label_data = dict(label)
                old_id = str(label_data.get("label_id") or "")
                new_id = self._generate_unique_label_id(old_id, used_ids)
                id_map[old_id] = new_id
                label_data["label_id"] = new_id
                options_payload: List[Dict[str, object]] = []
                options = label_data.get("options")
                if isinstance(options, list):
                    for option in options:
                        if not isinstance(option, (dict, ABCMapping)):
                            continue
                        option_data = dict(option)
                        option_data.pop("option_id", None)
                        options_payload.append(option_data)
                label_data["options"] = options_payload
                new_labels.append(label_data)
        for label in new_labels:
            expr = label.get("gating_expr")
            if not isinstance(expr, str) or not expr.strip():
                continue
            updated = expr
            for old_id, new_id in id_map.items():
                if not old_id:
                    continue
                updated = re.sub(rf"\b{re.escape(old_id)}\b", new_id, updated)
            label["gating_expr"] = updated
        self.labels = new_labels
        self._refresh_label_list()
        if self.labels:
            self.label_list.setCurrentRow(0)
        notes_text = str(payload.get("notes") or "").strip()
        self.notes_edit.setPlainText(notes_text)
        creator_text = str(payload.get("created_by") or "").strip()
        if creator_text:
            self.creator_edit.setText(creator_text)
        base_labelset_id = str(payload.get("labelset_id") or "").strip()
        if base_labelset_id:
            self.id_edit.setPlaceholderText(f"{base_labelset_id}_copy")

    def _refresh_label_list(self) -> None:
        self.label_list.clear()
        for label in self.labels:
            summary = f"{label['name']} ({label['type']})"
            item = QtWidgets.QListWidgetItem(summary)
            self.label_list.addItem(item)

    def _add_label(self) -> None:
        existing_ids = {
            str(label_id)
            for label_id in self.ctx.list_all_label_ids()
            if isinstance(label_id, str) and label_id
        }
        existing_ids.update(
            str(label.get("label_id"))
            for label in self.labels
            if isinstance(label.get("label_id"), str) and label.get("label_id")
        )
        dialog = LabelEditorDialog(existing_ids=existing_ids, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.labels.append(dialog.values())
        self._refresh_label_list()
        self.label_list.setCurrentRow(self.label_list.count() - 1)

    def _edit_label(self) -> None:
        row = self.label_list.currentRow()
        if row < 0 or row >= len(self.labels):
            return
        existing_ids = {
            str(label_id)
            for label_id in self.ctx.list_all_label_ids()
            if isinstance(label_id, str) and label_id
        }
        existing_ids.update(
            str(label.get("label_id"))
            for index, label in enumerate(self.labels)
            if index != row and isinstance(label.get("label_id"), str) and label.get("label_id")
        )
        dialog = LabelEditorDialog(existing_ids=existing_ids, data=self.labels[row], parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.labels[row] = dialog.values()
        self._refresh_label_list()
        self.label_list.setCurrentRow(row)

    def _remove_label(self) -> None:
        row = self.label_list.currentRow()
        if row < 0 or row >= len(self.labels):
            return
        del self.labels[row]
        self._refresh_label_list()
        if self.labels:
            self.label_list.setCurrentRow(min(row, len(self.labels) - 1))

    def _move_label(self, delta: int) -> None:
        row = self.label_list.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= len(self.labels):
            return
        self.labels[row], self.labels[target] = self.labels[target], self.labels[row]
        self._refresh_label_list()
        self.label_list.setCurrentRow(target)

    def accept(self) -> None:  # noqa: D401 - Qt override
        labelset_id = self.id_edit.text().strip()
        if not labelset_id:
            QtWidgets.QMessageBox.warning(self, "Label set", "Enter a label set ID.")
            return
        existing = self.ctx.get_labelset(labelset_id)
        if existing:
            QtWidgets.QMessageBox.warning(self, "Label set", "A label set with this ID already exists.")
            return
        if not self.labels:
            QtWidgets.QMessageBox.warning(self, "Label set", "Add at least one label.")
            return
        super().accept()

    def values(self) -> Dict[str, object]:
        return {
            "labelset_id": self.id_edit.text().strip(),
            "created_by": self.creator_edit.text().strip() or "admin",
            "notes": self.notes_edit.toPlainText().strip() or "",
            "labels": self.labels,
        }


class PhenotypeLabelSetsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        ctx: ProjectContext,
        pheno: Dict[str, object],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.pheno = dict(pheno)
        self.pheno_id = str(self.pheno.get("pheno_id") or "")
        self.pheno_name = str(self.pheno.get("name") or self.pheno_id)
        self._entries: List[Dict[str, object]] = []
        self.setWindowTitle(f"Label sets • {self.pheno_name}")
        self.resize(720, 520)
        self._setup_ui()
        self._load_labelsets()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "View label sets associated with this phenotype, including gating dependencies."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(
            ["Label set", "Labels", "Created", "Created by", "Association"]
        )
        layout.addWidget(self.table)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Label", "Type", "Gate"])
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree)

        self.notes_view = QtWidgets.QPlainTextEdit()
        self.notes_view.setPlaceholderText("Notes")
        self.notes_view.setReadOnly(True)
        layout.addWidget(self.notes_view)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def _load_labelsets(self) -> None:
        self._entries = self.ctx.list_label_sets_for_pheno(self.pheno_id)
        self.table.setRowCount(len(self._entries))
        for row_index, entry in enumerate(self._entries):
            record = entry.get("labelset", {})
            labelset_id = entry.get("labelset_id", "")
            label_count = entry.get("label_count", 0)
            created = str(record.get("created_at") or "")
            created_by = str(record.get("created_by") or "")
            association_parts: List[str] = []
            if entry.get("assigned_to_pheno"):
                association_parts.append("Assigned")
            rounds_used = entry.get("round_count", 0)
            if rounds_used:
                association_parts.append(f"Used in {rounds_used} round(s)")
            association = ", ".join(association_parts) if association_parts else "Available"
            item = QtWidgets.QTableWidgetItem(labelset_id)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, labelset_id)
            self.table.setItem(row_index, 0, item)
            self.table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(str(label_count)))
            self.table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(created))
            self.table.setItem(row_index, 3, QtWidgets.QTableWidgetItem(created_by))
            self.table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(association))
        self.table.resizeColumnsToContents()
        if self._entries:
            self.table.selectRow(0)
            self._show_labelset_details(0)
        else:
            self._show_labelset_details(None)

    def _on_selection_changed(self) -> None:
        selection = self.table.selectionModel().selectedRows()
        if selection:
            self._show_labelset_details(selection[0].row())
        else:
            self._show_labelset_details(None)

    def _show_labelset_details(self, row: Optional[int]) -> None:
        self.tree.clear()
        self.notes_view.clear()
        if row is None or row < 0 or row >= len(self._entries):
            return
        entry = self._entries[row]
        labelset_id = entry.get("labelset_id")
        if not isinstance(labelset_id, str):
            return
        details = self.ctx.load_labelset_details(labelset_id)
        if not isinstance(details, dict):
            return
        notes = str(details.get("notes") or "").strip()
        if notes:
            self.notes_view.setPlainText(notes)
        config = build_label_config(details)
        meta = config.get("_meta") if isinstance(config, dict) else {}
        tree_payload = meta.get("dependency_tree") if isinstance(meta, dict) else None
        if isinstance(tree_payload, list) and tree_payload:
            for node in tree_payload:
                if isinstance(node, dict):
                    item = self._build_tree_item(node)
                    self.tree.addTopLevelItem(item)
            self.tree.expandAll()
        else:
            labels = details.get("labels") if isinstance(details.get("labels"), list) else []
            for label in labels:  # type: ignore[assignment]
                if not isinstance(label, dict):
                    continue
                label_id = str(label.get("label_id") or "")
                name = str(label.get("name") or "")
                display = f"{label_id} — {name}" if name else label_id
                gate = str(label.get("gating_expr") or "")
                item = QtWidgets.QTreeWidgetItem([display, str(label.get("type") or ""), gate])
                self.tree.addTopLevelItem(item)
        header = self.tree.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)

    def _build_tree_item(self, node: Dict[str, object]) -> QtWidgets.QTreeWidgetItem:
        label_id = str(node.get("label_id") or "")
        name = str(node.get("name") or "")
        display = f"{label_id} — {name}" if name else label_id
        gate = str(node.get("condition") or "")
        item = QtWidgets.QTreeWidgetItem([display, str(node.get("type") or ""), gate])
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    child_item = self._build_tree_item(child)
                    item.addChild(child_item)
        return item


class PromptInferenceWorker(QtCore.QObject):
    """Background runner for prompt experiment sweeps."""

    log_message = QtCore.Signal(str)
    finished = QtCore.Signal(list)
    errored = QtCore.Signal(str)

    def __init__(
        self,
        job: PromptInferenceJob,
        variants: list[PromptExperimentConfig],
        user: str,
    ) -> None:
        super().__init__()
        self.job = job
        self.variants = variants
        self.user = user
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _log(self, message: str) -> None:
        self.log_message.emit(message)

    def _cancelled_cb(self) -> bool:
        return self._cancelled

    @QtCore.Slot()
    def run(self) -> None:
        try:
            results = self.job.run(
                self.variants,
                user=self.user,
                log_callback=self._log,
                cancel_callback=self._cancelled_cb,
            )
            self.finished.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.errored.emit(str(exc))


class PromptInferenceDialog(QtWidgets.QDialog):
    """UI for building prompt sweeps and running inference jobs."""

    def __init__(
        self,
        ctx: ProjectContext,
        pheno: Dict[str, object],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.pheno = dict(pheno)
        self.pheno_id = str(self.pheno.get("pheno_id") or "")
        self.pheno_name = str(self.pheno.get("name") or self.pheno_id)
        self.pheno_level = str(self.pheno.get("level") or "single_doc")
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[PromptInferenceWorker] = None
        self.setWindowTitle(f"Inference • {self.pheno_name}")
        self.resize(900, 720)
        self._build_ui()
        self._load_options()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        description = QtWidgets.QLabel(
            "Configure prompt experiments using adjudicated rounds, then run and review results."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QtWidgets.QFormLayout()

        self.labelset_combo = QtWidgets.QComboBox()
        form.addRow("Label set", self.labelset_combo)

        self.rounds_list = QtWidgets.QListWidget()
        self.rounds_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        form.addRow("Adjudicated rounds", self.rounds_list)

        self.corpus_combo = QtWidgets.QComboBox()
        form.addRow("Gold standard corpus", self.corpus_combo)

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["default", "openai", "local"])
        form.addRow("Backend", self.backend_combo)

        self.system_prompt = QtWidgets.QPlainTextEdit()
        self.system_prompt.setPlaceholderText("Optional system prompt override")
        form.addRow("System prompt", self.system_prompt)

        self.rule_overrides = QtWidgets.QPlainTextEdit()
        self.rule_overrides.setPlaceholderText("Optional JSON mapping of label_id → rule text")
        form.addRow("Rule overrides", self.rule_overrides)

        self.few_shot_examples = QtWidgets.QPlainTextEdit()
        self.few_shot_examples.setPlaceholderText(
            "Optional JSON object containing few-shot examples keyed by label_id"
        )
        form.addRow("Few-shot examples", self.few_shot_examples)

        layout.addLayout(form)

        sweep_group = QtWidgets.QGroupBox("Sweep options")
        sweep_layout = QtWidgets.QGridLayout(sweep_group)

        self.enable_zero_shot = QtWidgets.QCheckBox("Include zero-shot")
        self.enable_zero_shot.setChecked(True)
        self.enable_few_shot = QtWidgets.QCheckBox("Include few-shot")
        self.enable_few_shot.setChecked(True)
        self.include_family_tree = QtWidgets.QCheckBox("Family-tree mode")
        self.include_family_tree.setChecked(True)
        self.include_single_shot = QtWidgets.QCheckBox("Single-shot mode")
        self.include_single_shot.setChecked(True)
        sweep_layout.addWidget(self.enable_zero_shot, 0, 0)
        sweep_layout.addWidget(self.enable_few_shot, 0, 1)
        sweep_layout.addWidget(self.include_family_tree, 1, 0)
        sweep_layout.addWidget(self.include_single_shot, 1, 1)

        self.mmr_values = QtWidgets.QLineEdit("0.7")
        self.chunk_sizes = QtWidgets.QLineEdit("1500")
        self.num_chunks = QtWidgets.QLineEdit("6")
        sweep_layout.addWidget(QtWidgets.QLabel("MMR λ (comma-separated)"), 2, 0)
        sweep_layout.addWidget(self.mmr_values, 2, 1)
        sweep_layout.addWidget(QtWidgets.QLabel("Chunk sizes"), 3, 0)
        sweep_layout.addWidget(self.chunk_sizes, 3, 1)
        sweep_layout.addWidget(QtWidgets.QLabel("Chunks per label"), 4, 0)
        sweep_layout.addWidget(self.num_chunks, 4, 1)

        layout.addWidget(sweep_group)

        self.checkpoint_label = QtWidgets.QLabel()
        layout.addWidget(self.checkpoint_label)

        self.results_table = QtWidgets.QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Variant", "Agreement", "Metrics"])
        self.results_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.results_table)

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Logs, checkpoints, and backend output will appear here.")
        layout.addWidget(self.log_output)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.run_button = self.button_box.addButton(
            "Run experiments", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self.cancel_button = self.button_box.addButton(
            "Cancel run", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole
        )
        self.cancel_button.setEnabled(False)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.run_button.clicked.connect(self._start_run)
        self.cancel_button.clicked.connect(self._cancel_run)

    def _load_options(self) -> None:
        self.labelset_combo.clear()
        for entry in self.ctx.list_label_sets_for_pheno(self.pheno_id):
            labelset = entry.get("labelset") or {}
            labelset_id = str(labelset.get("labelset_id") or "")
            display = labelset_id
            if entry.get("assigned_to_pheno"):
                display = f"{display} (assigned)"
            self.labelset_combo.addItem(display, labelset_id)
        self.rounds_list.clear()
        for round_row in self.ctx.list_rounds(self.pheno_id):
            round_number = int(round_row.get("round_number") or 0)
            status = str(round_row.get("status") or "")
            label = f"Round {round_number} — {status}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, round_number)
            if status.lower() in {"finalized", "adjudicating", "closed"}:
                item.setSelected(True)
            self.rounds_list.addItem(item)
        self.corpus_combo.clear()
        self.corpus_combo.addItem("<None>", "")
        for corpus in self.ctx.list_corpora():
            corpus_id = str(corpus.get("corpus_id") or "")
            name = str(corpus.get("name") or corpus_id)
            self.corpus_combo.addItem(f"{name} ({corpus_id})", corpus_id)
        checkpoint_path = (
            self.ctx.require_project()
            / "prompt_runs"
            / f"{self.pheno_id}_prompt_checkpoint.json"
        )
        self.checkpoint_label.setText(f"Checkpoint: {checkpoint_path}")

    @staticmethod
    def _parse_float_list(value: str, default: Sequence[float]) -> list[float]:
        parts = [v.strip() for v in value.split(",") if v.strip()]
        if not parts:
            return list(default)
        parsed: list[float] = []
        for part in parts:
            try:
                parsed.append(float(part))
            except ValueError:
                continue
        return parsed or list(default)

    @staticmethod
    def _parse_int_list(value: str, default: Sequence[int]) -> list[int]:
        parts = [v.strip() for v in value.split(",") if v.strip()]
        if not parts:
            return list(default)
        parsed: list[int] = []
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError:
                continue
        return parsed or list(default)

    def _selected_rounds(self) -> list[int]:
        return [
            int(item.data(QtCore.Qt.ItemDataRole.UserRole) or 0)
            for item in self.rounds_list.selectedItems()
        ]

    def _start_run(self) -> None:
        labelset_id = str(self.labelset_combo.currentData() or "")
        if not labelset_id:
            QtWidgets.QMessageBox.warning(self, "Inference", "Select a label set first.")
            return
        rounds = self._selected_rounds()
        if not rounds:
            QtWidgets.QMessageBox.warning(
                self,
                "Inference",
                "Choose at least one adjudicated round to evaluate against.",
            )
            return
        base_config = PromptBuilderConfig(
            labelset_id=labelset_id,
            system_prompt=self.system_prompt.toPlainText(),
            use_few_shot=False,
            few_shot_examples=self._load_json_field(self.few_shot_examples),
            label_rule_overrides=self._load_json_field(self.rule_overrides),
            backend=str(self.backend_combo.currentText()),
        )
        sweep = PromptExperimentSweep(base=base_config)
        sweep.zero_shot = self.enable_zero_shot.isChecked()
        sweep.few_shot = self.enable_few_shot.isChecked()
        sweep.mmr_lambdas = self._parse_float_list(self.mmr_values.text(), sweep.mmr_lambdas)
        sweep.chunk_sizes = self._parse_int_list(self.chunk_sizes.text(), sweep.chunk_sizes)
        sweep.num_chunks = self._parse_int_list(self.num_chunks.text(), sweep.num_chunks)
        variants = sweep.variants()
        if not self.include_family_tree.isChecked():
            variants = [v for v in variants if v.config.inference_mode != "family_tree"]
        if not self.include_single_shot.isChecked():
            variants = [v for v in variants if v.config.inference_mode != "single_shot"]
        if not variants:
            QtWidgets.QMessageBox.warning(self, "Inference", "No experiment variants to run.")
            return

        corpus_id = str(self.corpus_combo.currentData() or "") or None
        corpus_path = None
        if corpus_id:
            try:
                corpus_path = self.ctx.resolve_corpus_path(corpus_id)
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Inference", f"Corpus error: {exc}")
                return

        job = PromptInferenceJob(
            project_root=self.ctx.require_project(),
            pheno_id=self.pheno_id,
            labelset_id=labelset_id,
            phenotype_level=self.pheno_level,
            adjudicated_rounds=rounds,
            corpus_id=corpus_id,
            corpus_path=corpus_path,
        )

        self.log_output.appendPlainText(
            f"Starting {len(variants)} variant(s). Checkpoint → {job.checkpoint_path}"
        )
        self._set_running(True)
        self._worker_thread = QtCore.QThread(self)
        self._worker = PromptInferenceWorker(job, variants, user="admin")
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.log_message.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.errored.connect(self._on_error)
        self._worker_thread.start()

    def _load_json_field(self, widget: QtWidgets.QPlainTextEdit) -> Dict[str, object]:
        raw = widget.toPlainText().strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            QtWidgets.QMessageBox.warning(self, "Inference", "Invalid JSON; ignoring field.")
        return {}

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setEnabled(
            not running
        )

    def _cancel_run(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._on_log("Cancellation requested…")

    @QtCore.Slot(str)
    def _on_log(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")

    @QtCore.Slot(list)
    def _on_finished(self, results: list) -> None:  # type: ignore[override]
        self._populate_results(results)
        self._teardown_worker()
        self._set_running(False)
        QtWidgets.QMessageBox.information(self, "Inference", "Experiments completed.")

    @QtCore.Slot(str)
    def _on_error(self, message: str) -> None:
        self._on_log(f"Error: {message}")
        self._teardown_worker()
        self._set_running(False)
        QtWidgets.QMessageBox.critical(self, "Inference", message)

    def _teardown_worker(self) -> None:
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        self._worker_thread = None
        self._worker = None

    def _populate_results(self, results: list) -> None:
        self.results_table.setRowCount(0)
        for idx, result in enumerate(results):
            self.results_table.insertRow(idx)
            name_item = QtWidgets.QTableWidgetItem(str(result.name))
            agreement = result.agreement
            agreement_item = QtWidgets.QTableWidgetItem(
                "" if agreement is None else f"{float(agreement):.3f}"
            )
            metrics_json = json.dumps(result.metrics, indent=2, sort_keys=True)
            metrics_item = QtWidgets.QTableWidgetItem(metrics_json)
            self.results_table.setItem(idx, 0, name_item)
            self.results_table.setItem(idx, 1, agreement_item)
            self.results_table.setItem(idx, 2, metrics_item)
        self.results_table.resizeColumnsToContents()


    def _add_label(self) -> None:
        existing_ids = {label["label_id"] for label in self.labels}
        dialog = LabelEditorDialog(existing_ids=existing_ids, parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.labels.append(dialog.values())
        self._refresh_label_list()

    def _edit_label(self) -> None:
        row = self.label_list.currentRow()
        if row < 0 or row >= len(self.labels):
            return
        existing_ids = {label["label_id"] for label in self.labels}
        dialog = LabelEditorDialog(existing_ids=existing_ids, data=self.labels[row], parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.labels[row] = dialog.values()
        self._refresh_label_list()
        self.label_list.setCurrentRow(row)

    def _remove_label(self) -> None:
        row = self.label_list.currentRow()
        if row < 0 or row >= len(self.labels):
            return
        del self.labels[row]
        self._refresh_label_list()

    def _move_label(self, delta: int) -> None:
        row = self.label_list.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= len(self.labels):
            return
        self.labels[row], self.labels[target] = self.labels[target], self.labels[row]
        self._refresh_label_list()
        self.label_list.setCurrentRow(target)

    def accept(self) -> None:  # noqa: D401 - Qt override
        labelset_id = self.id_edit.text().strip()
        if not labelset_id:
            QtWidgets.QMessageBox.warning(self, "Label set", "Enter a label set ID.")
            return
        existing = self.ctx.get_labelset(labelset_id)
        if existing:
            QtWidgets.QMessageBox.warning(self, "Label set", "A label set with this ID already exists.")
            return
        if not self.labels:
            QtWidgets.QMessageBox.warning(self, "Label set", "Add at least one label.")
            return
        super().accept()

    def values(self) -> Dict[str, object]:
        return {
            "labelset_id": self.id_edit.text().strip(),
            "created_by": self.creator_edit.text().strip() or "admin",
            "notes": self.notes_edit.toPlainText().strip() or "",
            "labels": self.labels,
        }


class MetadataFilterDialog(QtWidgets.QDialog):
    def __init__(
        self,
        field: MetadataField,
        parent: Optional[QtWidgets.QWidget] = None,
        existing: Optional[MetadataFilterCondition] = None,
    ) -> None:
        super().__init__(parent)
        self.field = field
        self._condition: Optional[MetadataFilterCondition] = None
        self._existing = existing
        self.setWindowTitle(f"Filter • {field.label}")
        self._build_ui()
        if existing:
            self._apply_existing(existing)

    def _build_ui(self) -> None:
        layout = QtWidgets.QFormLayout(self)
        description = QtWidgets.QLabel(f"{self.field.label} ({self.field.data_type})")
        description.setWordWrap(True)
        layout.addRow(description)

        if self.field.data_type == "number":
            self.min_edit = QtWidgets.QLineEdit()
            self.min_edit.setPlaceholderText("Optional minimum")
            self.min_edit.setValidator(QtGui.QDoubleValidator(self))
            self.max_edit = QtWidgets.QLineEdit()
            self.max_edit.setPlaceholderText("Optional maximum")
            self.max_edit.setValidator(QtGui.QDoubleValidator(self))
            self.value_edit = QtWidgets.QLineEdit()
            self.value_edit.setPlaceholderText("Exact values (comma separated)")
            layout.addRow("Minimum", self.min_edit)
            layout.addRow("Maximum", self.max_edit)
            layout.addRow("Values", self.value_edit)
        elif self.field.data_type == "date":
            self.start_check = QtWidgets.QCheckBox("Enable")
            self.start_date = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
            self.start_date.setDisplayFormat("yyyy-MM-dd")
            self.start_date.setCalendarPopup(True)
            self.start_date.setEnabled(False)
            self.start_check.toggled.connect(self.start_date.setEnabled)

            self.end_check = QtWidgets.QCheckBox("Enable")
            self.end_date = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
            self.end_date.setDisplayFormat("yyyy-MM-dd")
            self.end_date.setCalendarPopup(True)
            self.end_date.setEnabled(False)
            self.end_check.toggled.connect(self.end_date.setEnabled)

            self.date_values_edit = QtWidgets.QLineEdit()
            self.date_values_edit.setPlaceholderText("Specific dates (comma separated)")

            layout.addRow("Start date", self._wrap_controls(self.start_check, self.start_date))
            layout.addRow("End date", self._wrap_controls(self.end_check, self.end_date))
            layout.addRow("Values", self.date_values_edit)
        else:
            self.value_edit = QtWidgets.QLineEdit()
            self.value_edit.setPlaceholderText("Values (comma separated)")
            layout.addRow("Values", self.value_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @staticmethod
    def _wrap_controls(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        inner = QtWidgets.QHBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(6)
        for widget in widgets:
            inner.addWidget(widget)
        inner.addStretch()
        return container

    def _apply_existing(self, existing: MetadataFilterCondition) -> None:
        if self.field.data_type == "number":
            if existing.min_value is not None:
                self.min_edit.setText(str(existing.min_value))
            if existing.max_value is not None:
                self.max_edit.setText(str(existing.max_value))
            if existing.values:
                self.value_edit.setText(", ".join(existing.values))
        elif self.field.data_type == "date":
            if existing.min_value:
                self._set_date_value(self.start_date, self.start_check, existing.min_value)
            if existing.max_value:
                self._set_date_value(self.end_date, self.end_check, existing.max_value)
            if existing.values:
                self.date_values_edit.setText(", ".join(existing.values))
        else:
            if existing.values:
                self.value_edit.setText(", ".join(existing.values))

    @staticmethod
    def _set_date_value(
        widget: QtWidgets.QDateEdit,
        checkbox: QtWidgets.QCheckBox,
        value: str,
    ) -> None:
        date = QtCore.QDate.fromString(value, QtCore.Qt.DateFormat.ISODate)
        if date.isValid():
            widget.setDate(date)
            checkbox.setChecked(True)

    @staticmethod
    def _date_to_string(widget: QtWidgets.QDateEdit) -> Optional[str]:
        date = widget.date()
        if not date.isValid():
            return None
        return date.toString(QtCore.Qt.DateFormat.ISODate)

    def _on_accept(self) -> None:
        if self.field.data_type == "number":
            min_value = self.min_edit.text().strip()
            max_value = self.max_edit.text().strip()
            values = [value.strip() for value in self.value_edit.text().split(",") if value.strip()]
            if not min_value and not max_value and not values:
                QtWidgets.QMessageBox.warning(self, "Filter", "Specify at least one constraint.")
                return
            condition = MetadataFilterCondition(
                field=self.field.key,
                label=self.field.label,
                scope=self.field.scope,
                data_type=self.field.data_type,
                min_value=min_value or None,
                max_value=max_value or None,
                values=values or None,
            )
        elif self.field.data_type == "date":
            min_value = self._date_to_string(self.start_date) if self.start_check.isChecked() else None
            max_value = self._date_to_string(self.end_date) if self.end_check.isChecked() else None
            values = [value.strip() for value in self.date_values_edit.text().split(",") if value.strip()]
            if not min_value and not max_value and not values:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Filter",
                    "Specify a start date, end date, or specific values.",
                )
                return
            condition = MetadataFilterCondition(
                field=self.field.key,
                label=self.field.label,
                scope=self.field.scope,
                data_type=self.field.data_type,
                min_value=min_value,
                max_value=max_value,
                values=values or None,
            )
        else:
            values = [value.strip() for value in self.value_edit.text().split(",") if value.strip()]
            if not values:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Filter",
                    "Enter one or more values to filter by.",
                )
                return
            condition = MetadataFilterCondition(
                field=self.field.key,
                label=self.field.label,
                scope=self.field.scope,
                data_type=self.field.data_type,
                values=values,
            )
        self._condition = condition
        super().accept()

    def condition(self) -> Optional[MetadataFilterCondition]:
        return self._condition


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
        self._available_reviewers = self._load_existing_reviewers()
        self._selected_reviewer_ids: Set[str] = set()
        self._labelset_options = self._load_labelset_ids()
        self._corpus_options = self._load_corpus_options()
        self._selected_corpus_id: Optional[str] = None
        self._metadata_fields: List[MetadataField] = []
        self._metadata_lookup: Dict[str, MetadataField] = {}
        self._ai_thread: Optional[QtCore.QThread] = None
        self._ai_worker: Optional[AIRoundWorker] = None
        self._ai_pending_job: Optional[AIRoundJobConfig] = None
        self._ai_job_running = False
        self._ai_progress_active = False
        self._ai_progress_stamp: str = ""
        self._ai_progress_text: str = ""
        self._ai_progress_block_number: Optional[int] = None
        self._ai_log_dialog: Optional[AIRoundLogDialog] = None
        self.ai_log_output: Optional[QtWidgets.QPlainTextEdit] = None
        self._ai_engine_overrides: Dict[str, Any] = {}
        self._assisted_review_reminder_shown = False
        self._llm_prompt_shown = False
        self.ctx.project_changed.connect(self._refresh_labelset_options)
        self.ctx.project_changed.connect(self._refresh_corpus_options)
        self.ctx.project_changed.connect(self._refresh_ai_round_options)
        self._setup_ui()

    @staticmethod
    def _safe_mapping_get(
        mapping: Optional[sqlite3.Row | Mapping[str, object]],
        key: str,
        default: object = None,
    ) -> object:
        if mapping is None:
            return default
        if isinstance(mapping, sqlite3.Row):
            if key in mapping.keys():
                return mapping[key]
            return default
        if isinstance(mapping, Mapping):
            return mapping.get(key, default)
        try:
            return mapping[key]  # type: ignore[index]
        except Exception:
            return default

    def _update_ai_batch_size_label(self) -> None:
        if not hasattr(self, "ai_batch_size_label"):
            return
        value = self.total_n_spin.value() if hasattr(self, "total_n_spin") else 0
        self.ai_batch_size_label.setText(f"{value} (matches Total N)")

    def _build_ai_config_snapshot(self) -> Dict[str, Any]:
        try:
            base_cfg: Dict[str, Any] = asdict(engine.OrchestratorConfig())
        except Exception:  # noqa: BLE001
            base_cfg = {}
        _deep_update_dict(base_cfg, self._ai_engine_overrides or {})
        select_cfg = base_cfg.get("select", {}) if isinstance(base_cfg.get("select"), Mapping) else {}
        if hasattr(self, "total_n_spin"):
            select_cfg["batch_size"] = int(self.total_n_spin.value())
        if hasattr(self, "ai_disagreement_pct"):
            select_cfg["pct_disagreement"] = float(self.ai_disagreement_pct.value())
        if hasattr(self, "ai_uncertain_pct"):
            select_cfg["pct_uncertain"] = float(self.ai_uncertain_pct.value())
        if hasattr(self, "ai_easy_pct"):
            select_cfg["pct_easy_qc"] = float(self.ai_easy_pct.value())
        if hasattr(self, "ai_diversity_pct"):
            select_cfg["pct_diversity"] = float(self.ai_diversity_pct.value())
        base_cfg["select"] = select_cfg
        llm_cfg = base_cfg.get("llm", {}) if isinstance(base_cfg.get("llm"), Mapping) else {}
        backend_choice = self._current_ai_backend()
        if backend_choice:
            llm_cfg["backend"] = backend_choice
        include_checkbox = getattr(self, "ai_include_reasoning_checkbox", None)
        if isinstance(include_checkbox, QtWidgets.QCheckBox):
            llm_cfg["include_reasoning"] = bool(include_checkbox.isChecked())
        if backend_choice == "azure":
            if hasattr(self, "ai_azure_key_edit"):
                azure_key = self.ai_azure_key_edit.text().strip()
                if azure_key:
                    llm_cfg["azure_api_key"] = azure_key
            if hasattr(self, "ai_azure_version_edit"):
                version = self.ai_azure_version_edit.text().strip()
                if version:
                    llm_cfg["azure_api_version"] = version
            if hasattr(self, "ai_azure_endpoint_edit"):
                endpoint = self.ai_azure_endpoint_edit.text().strip()
                if endpoint:
                    llm_cfg["azure_endpoint"] = endpoint
        elif backend_choice:
            if hasattr(self, "ai_local_model_path_edit"):
                model_dir = self.ai_local_model_path_edit.text().strip()
                if model_dir:
                    llm_cfg["local_model_dir"] = model_dir
            if hasattr(self, "ai_local_max_seq_spin"):
                llm_cfg["local_max_seq_len"] = int(self.ai_local_max_seq_spin.value())
            if hasattr(self, "ai_local_max_new_tokens_spin"):
                llm_cfg["local_max_new_tokens"] = int(self.ai_local_max_new_tokens_spin.value())
        base_cfg["llm"] = llm_cfg
        if hasattr(self, "ai_final_llm_checkbox"):
            base_cfg["final_llm_labeling"] = bool(self.ai_final_llm_checkbox.isChecked())
        if "final_llm_labeling_n_consistency" not in base_cfg:
            try:
                base_cfg["final_llm_labeling_n_consistency"] = int(
                    engine.OrchestratorConfig().final_llm_labeling_n_consistency
                )
            except Exception:  # noqa: BLE001
                base_cfg["final_llm_labeling_n_consistency"] = 1
        if (
            str(self.pheno_row["level"] or "single_doc") == "single_doc"
            and isinstance(getattr(self, "ai_single_doc_context_combo", None), QtWidgets.QComboBox)
        ):
            llmfirst_cfg = base_cfg.get("llmfirst", {}) if isinstance(base_cfg.get("llmfirst"), Mapping) else {}
            mode_value = self.ai_single_doc_context_combo.currentData()
            if not isinstance(mode_value, str) or not mode_value:
                mode_value = "rag"
            llmfirst_cfg["single_doc_context"] = mode_value
            base_cfg["llmfirst"] = llmfirst_cfg
        return base_cfg

    def _apply_ai_config_to_controls(self, config: Mapping[str, Any]) -> None:
        select_cfg = config.get("select", {}) if isinstance(config.get("select"), Mapping) else {}
        batch_size = select_cfg.get("batch_size")
        if hasattr(self, "total_n_spin") and isinstance(batch_size, (int, float)):
            try:
                self.total_n_spin.setValue(int(batch_size))
            except Exception:  # noqa: BLE001
                pass
        for attr_name, key in (
            ("ai_disagreement_pct", "pct_disagreement"),
            ("ai_uncertain_pct", "pct_uncertain"),
            ("ai_easy_pct", "pct_easy_qc"),
            ("ai_diversity_pct", "pct_diversity"),
        ):
            widget = getattr(self, attr_name, None)
            value = select_cfg.get(key)
            if isinstance(widget, QtWidgets.QDoubleSpinBox) and isinstance(value, (int, float)):
                widget.setValue(float(value))
        llm_cfg = config.get("llm", {}) if isinstance(config.get("llm"), Mapping) else {}
        backend_choice = llm_cfg.get("backend")
        if backend_choice and hasattr(self, "ai_backend_combo"):
            idx = self.ai_backend_combo.findData(str(backend_choice))
            if idx >= 0:
                self.ai_backend_combo.setCurrentIndex(idx)
                self._update_ai_backend_fields()
        include_reasoning = llm_cfg.get("include_reasoning")
        checkbox = getattr(self, "ai_include_reasoning_checkbox", None)
        if isinstance(checkbox, QtWidgets.QCheckBox) and isinstance(include_reasoning, bool):
            checkbox.setChecked(include_reasoning)
        if backend_choice == "azure":
            if hasattr(self, "ai_azure_key_edit") and llm_cfg.get("azure_api_key"):
                self.ai_azure_key_edit.setText(str(llm_cfg.get("azure_api_key")))
            if hasattr(self, "ai_azure_version_edit") and llm_cfg.get("azure_api_version"):
                self.ai_azure_version_edit.setText(str(llm_cfg.get("azure_api_version")))
            if hasattr(self, "ai_azure_endpoint_edit") and llm_cfg.get("azure_endpoint"):
                self.ai_azure_endpoint_edit.setText(str(llm_cfg.get("azure_endpoint")))
        elif backend_choice:
            if hasattr(self, "ai_local_model_path_edit") and llm_cfg.get("local_model_dir"):
                self.ai_local_model_path_edit.setText(str(llm_cfg.get("local_model_dir")))
            if hasattr(self, "ai_local_max_seq_spin") and llm_cfg.get("local_max_seq_len") is not None:
                try:
                    self.ai_local_max_seq_spin.setValue(int(llm_cfg.get("local_max_seq_len")))
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(self, "ai_local_max_new_tokens_spin") and llm_cfg.get("local_max_new_tokens") is not None:
                try:
                    self.ai_local_max_new_tokens_spin.setValue(int(llm_cfg.get("local_max_new_tokens")))
                except Exception:  # noqa: BLE001
                    pass
        if hasattr(self, "ai_final_llm_checkbox") and "final_llm_labeling" in config:
            try:
                self.ai_final_llm_checkbox.setChecked(bool(config.get("final_llm_labeling")))
            except Exception:  # noqa: BLE001
                pass
        if (
            str(self.pheno_row["level"] or "single_doc") == "single_doc"
            and isinstance(getattr(self, "ai_single_doc_context_combo", None), QtWidgets.QComboBox)
        ):
            value = None
            llmfirst_cfg = config.get("llmfirst", {}) if isinstance(config.get("llmfirst"), Mapping) else {}
            if llmfirst_cfg:
                value = llmfirst_cfg.get("single_doc_context")
            if value:
                idx = self.ai_single_doc_context_combo.findData(str(value))
                if idx >= 0:
                    self.ai_single_doc_context_combo.setCurrentIndex(idx)

    def _open_ai_advanced_settings(self) -> None:
        config = self._build_ai_config_snapshot()
        dialog = AIAdvancedConfigDialog(self, config)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._ai_engine_overrides = dialog.result_config or {}
        self._apply_ai_config_to_controls(self._ai_engine_overrides)


    def _on_random_final_llm_toggled(self, checked: bool) -> None:
        if hasattr(self, "random_final_llm_group"):
            self.random_final_llm_group.setEnabled(bool(checked))

    def _on_ai_final_llm_toggled(self, checked: bool) -> None:
        checkbox = getattr(self, "ai_include_reasoning_checkbox", None)
        if isinstance(checkbox, QtWidgets.QCheckBox):
            checkbox.setEnabled(bool(checked))

    def _on_assisted_review_toggled(self, checked: bool) -> None:
        if hasattr(self, "assisted_review_spin"):
            self.assisted_review_spin.setEnabled(bool(checked))
        if checked and not self._assisted_review_reminder_shown:
            backend = self._current_ai_backend()
            if backend == "azure":
                azure_present = False
                for attr in ("ai_azure_key_edit", "random_azure_key_edit"):
                    widget = getattr(self, attr, None)
                    if isinstance(widget, QtWidgets.QLineEdit) and widget.text().strip():
                        azure_present = True
                        break
                if not azure_present:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Assisted chart review",
                        (
                            "Assisted chart review uses label exemplars generated by the AI backend. "
                            "Enter Azure OpenAI credentials so exemplars can be generated. "
                            "Without credentials the system will fall back to label rules."
                        ),
                    )
            else:
                local_present = False
                for attr in ("ai_local_model_path_edit", "random_local_model_path_edit"):
                    widget = getattr(self, attr, None)
                    if isinstance(widget, QtWidgets.QLineEdit) and widget.text().strip():
                        local_present = True
                        break
                if not local_present:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Assisted chart review",
                        (
                            "Assisted chart review uses label exemplars generated by the AI backend. "
                            "Select a local LLM model directory so exemplars can be generated. "
                            "Without a model path the system will fall back to label rules."
                        ),
                    )
            self._assisted_review_reminder_shown = True

    def _browse_for_directory(self, title: str, current: str = "") -> Optional[str]:
        initial = current or ""
        if not initial and getattr(self.ctx, "project_root", None):
            initial = str(self.ctx.project_root)
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, title, initial)
        if directory:
            return directory
        return None

    def _on_select_embedding_model(self) -> None:
        sender = self.sender()
        edit: Optional[QtWidgets.QLineEdit] = None
        if sender is getattr(self, "random_embedding_browse_btn", None):
            edit = getattr(self, "random_embedding_path_edit", None)
        elif sender is getattr(self, "ai_embedding_browse_btn", None):
            edit = getattr(self, "ai_embedding_path_edit", None)
        else:
            edit = getattr(self, "ai_embedding_path_edit", None)
        current = edit.text().strip() if isinstance(edit, QtWidgets.QLineEdit) else ""
        directory = self._browse_for_directory("Select embedding model directory", current)
        if directory and isinstance(edit, QtWidgets.QLineEdit):
            edit.setText(directory)

    def _on_select_reranker_model(self) -> None:
        sender = self.sender()
        edit: Optional[QtWidgets.QLineEdit] = None
        if sender is getattr(self, "random_reranker_browse_btn", None):
            edit = getattr(self, "random_reranker_path_edit", None)
        elif sender is getattr(self, "ai_reranker_browse_btn", None):
            edit = getattr(self, "ai_reranker_path_edit", None)
        else:
            edit = getattr(self, "ai_reranker_path_edit", None)
        current = edit.text().strip() if isinstance(edit, QtWidgets.QLineEdit) else ""
        directory = self._browse_for_directory("Select re-ranker model directory", current)
        if directory and isinstance(edit, QtWidgets.QLineEdit):
            edit.setText(directory)

    def _on_select_local_model_dir(self) -> None:
        sender = self.sender()
        edit: Optional[QtWidgets.QLineEdit] = None
        if sender is getattr(self, "random_local_model_browse_btn", None):
            edit = getattr(self, "random_local_model_path_edit", None)
        elif sender is getattr(self, "ai_local_model_browse_btn", None):
            edit = getattr(self, "ai_local_model_path_edit", None)
        else:
            edit = getattr(self, "ai_local_model_path_edit", None)
        current = edit.text().strip() if isinstance(edit, QtWidgets.QLineEdit) else ""
        directory = self._browse_for_directory("Select local LLM model directory", current)
        if directory and isinstance(edit, QtWidgets.QLineEdit):
            edit.setText(directory)

    def _current_ai_backend(self) -> str:
        combo = getattr(self, "ai_backend_combo", None)
        if isinstance(combo, QtWidgets.QComboBox):
            data = combo.currentData()
            if isinstance(data, str) and data:
                return data
        return "azure"

    def _current_random_backend(self) -> str:
        combo = getattr(self, "random_backend_combo", None)
        if isinstance(combo, QtWidgets.QComboBox):
            data = combo.currentData()
            if isinstance(data, str) and data:
                return data
        return "azure"

    def _update_ai_backend_fields(self) -> None:
        backend = self._current_ai_backend()
        show_azure = backend == "azure"
        show_local = backend != "azure"
        for widget in getattr(self, "_ai_azure_widgets", []):
            if isinstance(widget, QtWidgets.QWidget):
                widget.setVisible(show_azure)
        for widget in getattr(self, "_ai_local_widgets", []):
            if isinstance(widget, QtWidgets.QWidget):
                widget.setVisible(show_local)

    def _update_random_backend_fields(self) -> None:
        backend = self._current_random_backend()
        show_azure = backend == "azure"
        show_local = backend != "azure"
        for widget in getattr(self, "_random_azure_widgets", []):
            if isinstance(widget, QtWidgets.QWidget):
                widget.setVisible(show_azure)
        for widget in getattr(self, "_random_local_widgets", []):
            if isinstance(widget, QtWidgets.QWidget):
                widget.setVisible(show_local)

    def _setup_ui(self) -> None:
        self.ai_single_doc_context_combo: Optional[QtWidgets.QComboBox] = None
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(container)
        scroll_layout.setContentsMargins(8, 8, 8, 8)

        setup_group = QtWidgets.QGroupBox("Round setup")
        setup_form = QtWidgets.QFormLayout(setup_group)
        corpus_layout = QtWidgets.QHBoxLayout()
        self.corpus_combo = QtWidgets.QComboBox()
        self.corpus_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.corpus_combo.currentIndexChanged.connect(self._on_corpus_changed)
        corpus_layout.addWidget(self.corpus_combo)
        import_btn = QtWidgets.QPushButton("Import…")
        import_btn.clicked.connect(self._on_import_corpus)
        corpus_layout.addWidget(import_btn)
        setup_form.addRow("Corpus", corpus_layout)
        self.labelset_combo = QtWidgets.QComboBox()
        self.labelset_combo.setEditable(True)
        self.labelset_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        line_edit = self.labelset_combo.lineEdit()
        if line_edit:
            line_edit.setPlaceholderText("Select or enter label set ID")
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setMaximum(2**31 - 1)
        self.overlap_spin = QtWidgets.QSpinBox()
        self.overlap_spin.setRange(0, 1000)
        self.total_n_spin = QtWidgets.QSpinBox()
        self.total_n_spin.setRange(1, 1000000)
        self.status_combo = QtWidgets.QComboBox()
        self.status_combo.addItems(["draft", "active", "closed", "adjudicating", "finalized"])
        unit_label = "patients" if self.pheno_row["level"] == "multi_doc" else "documents"
        self.independent_checkbox = QtWidgets.QCheckBox(
            f"Exclude previously reviewed {unit_label}"
        )
        self.independent_checkbox.setToolTip(
            "When enabled, sampling will skip any units that were included in previous rounds for this phenotype."
        )
        setup_form.addRow("Label set", self.labelset_combo)
        setup_form.addRow("Seed", self.seed_spin)
        setup_form.addRow("Overlap N", self.overlap_spin)
        setup_form.addRow("Total N", self.total_n_spin)
        setup_form.addRow("Status", self.status_combo)
        setup_form.addRow("Independent sampling", self.independent_checkbox)
        assisted_widget = QtWidgets.QWidget()
        assisted_layout = QtWidgets.QHBoxLayout(assisted_widget)
        assisted_layout.setContentsMargins(0, 0, 0, 0)
        assisted_layout.setSpacing(6)
        self.assisted_review_checkbox = QtWidgets.QCheckBox("Enable assisted chart review")
        self.assisted_review_checkbox.toggled.connect(self._on_assisted_review_toggled)
        assisted_layout.addWidget(self.assisted_review_checkbox)
        assisted_layout.addStretch()
        assisted_layout.addWidget(QtWidgets.QLabel("Top N"))
        self.assisted_review_spin = QtWidgets.QSpinBox()
        self.assisted_review_spin.setRange(1, 20)
        self.assisted_review_spin.setValue(5)
        self.assisted_review_spin.setEnabled(False)
        assisted_layout.addWidget(self.assisted_review_spin)
        setup_form.addRow("Assisted review", assisted_widget)
        scroll_layout.addWidget(setup_group)

        reviewer_group = QtWidgets.QGroupBox("Reviewers")
        reviewer_layout = QtWidgets.QVBoxLayout(reviewer_group)
        self.reviewer_list = QtWidgets.QListWidget()
        self.reviewer_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        reviewer_layout.addWidget(self.reviewer_list)

        existing_layout = QtWidgets.QHBoxLayout()
        self.reviewer_combo = QtWidgets.QComboBox()
        self.reviewer_combo.addItem("Select existing reviewer…", None)
        existing_layout.addWidget(self.reviewer_combo)
        add_existing_btn = QtWidgets.QPushButton("Add selected reviewer")
        add_existing_btn.clicked.connect(self._on_add_existing_reviewer)
        existing_layout.addWidget(add_existing_btn)
        reviewer_layout.addLayout(existing_layout)

        new_layout = QtWidgets.QHBoxLayout()
        self.new_reviewer_edit = QtWidgets.QLineEdit()
        self.new_reviewer_edit.setPlaceholderText("Enter new reviewer name")
        new_layout.addWidget(self.new_reviewer_edit)
        add_new_btn = QtWidgets.QPushButton("Add new reviewer")
        add_new_btn.clicked.connect(self._on_add_new_reviewer)
        new_layout.addWidget(add_new_btn)
        reviewer_layout.addLayout(new_layout)

        remove_btn = QtWidgets.QPushButton("Remove selected reviewer")
        remove_btn.clicked.connect(self._remove_selected_reviewer)
        reviewer_layout.addWidget(remove_btn)
        scroll_layout.addWidget(reviewer_group)

        method_group = QtWidgets.QGroupBox("Generation method")
        method_layout = QtWidgets.QHBoxLayout(method_group)
        self.random_sampling_radio = QtWidgets.QRadioButton("Random sampling")
        self.active_learning_radio = QtWidgets.QRadioButton("Active learning backend")
        method_layout.addWidget(self.random_sampling_radio)
        method_layout.addWidget(self.active_learning_radio)
        method_layout.addStretch()
        self.random_sampling_radio.setChecked(True)
        self.random_sampling_radio.toggled.connect(self._on_generation_mode_changed)
        self.active_learning_radio.toggled.connect(self._on_generation_mode_changed)
        scroll_layout.addWidget(method_group)

        self.random_config_container = QtWidgets.QWidget()
        random_layout = QtWidgets.QVBoxLayout(self.random_config_container)
        random_layout.setContentsMargins(0, 0, 0, 0)

        filter_group = QtWidgets.QGroupBox("Sampling filters")
        filter_layout = QtWidgets.QVBoxLayout(filter_group)
        self.filter_list = QtWidgets.QListWidget()
        self.filter_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.filter_list.itemDoubleClicked.connect(self._on_edit_filter)
        self.filter_list.currentRowChanged.connect(self._update_filter_buttons)
        filter_layout.addWidget(self.filter_list)
        logic_row = QtWidgets.QHBoxLayout()
        logic_label = QtWidgets.QLabel("Match mode:")
        logic_row.addWidget(logic_label)
        self.filter_logic_combo = QtWidgets.QComboBox()
        self.filter_logic_combo.addItem("Match all conditions", False)
        self.filter_logic_combo.addItem("Match any condition", True)
        self.filter_logic_combo.setToolTip(
            "Choose whether documents must satisfy every filter or any single filter."
        )
        logic_row.addWidget(self.filter_logic_combo)
        logic_row.addStretch()
        filter_layout.addLayout(logic_row)
        filter_controls = QtWidgets.QHBoxLayout()
        self.filter_field_combo = QtWidgets.QComboBox()
        self.filter_field_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        filter_controls.addWidget(self.filter_field_combo)
        self.add_filter_btn = QtWidgets.QPushButton("Add filter")
        self.add_filter_btn.clicked.connect(self._on_add_filter)
        filter_controls.addWidget(self.add_filter_btn)
        self.remove_filter_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_filter_btn.clicked.connect(self._on_remove_filter)
        filter_controls.addWidget(self.remove_filter_btn)
        filter_layout.addLayout(filter_controls)
        random_layout.addWidget(filter_group)

        strat_group = QtWidgets.QGroupBox("Stratification")
        strat_layout = QtWidgets.QVBoxLayout(strat_group)
        self.strat_list = QtWidgets.QListWidget()
        self.strat_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.strat_list.currentRowChanged.connect(self._update_strat_buttons)
        strat_layout.addWidget(self.strat_list)
        strat_controls = QtWidgets.QHBoxLayout()
        self.strat_field_combo = QtWidgets.QComboBox()
        self.strat_field_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        strat_controls.addWidget(self.strat_field_combo)
        self.add_strat_btn = QtWidgets.QPushButton("Add field")
        self.add_strat_btn.clicked.connect(self._on_add_strat_field)
        strat_controls.addWidget(self.add_strat_btn)
        self.remove_strat_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_strat_btn.clicked.connect(self._on_remove_strat_field)
        strat_controls.addWidget(self.remove_strat_btn)
        strat_layout.addLayout(strat_controls)
        random_layout.addWidget(strat_group)

        final_llm_row = QtWidgets.QHBoxLayout()
        self.random_final_llm_checkbox = QtWidgets.QCheckBox("Run final LLM labeling")
        self.random_final_llm_checkbox.setChecked(True)
        self.random_final_llm_checkbox.toggled.connect(self._on_random_final_llm_toggled)
        final_llm_row.addWidget(self.random_final_llm_checkbox)
        final_llm_row.addStretch()
        random_layout.addLayout(final_llm_row)

        self.random_final_llm_group = QtWidgets.QGroupBox("Final LLM configuration")
        random_llm_layout = QtWidgets.QFormLayout(self.random_final_llm_group)
        random_llm_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.random_backend_combo = QtWidgets.QComboBox()
        self.random_backend_combo.addItem("Azure OpenAI", "azure")
        self.random_backend_combo.addItem("Local ExLlamaV2", "exllamav2")
        self.random_backend_combo.currentIndexChanged.connect(self._update_random_backend_fields)
        random_llm_layout.addRow("Backend", self.random_backend_combo)

        self.random_azure_key_edit = QtWidgets.QLineEdit()
        self.random_azure_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.random_azure_key_edit.setPlaceholderText("Enter Azure OpenAI API key")
        random_llm_layout.addRow("Azure API key", self.random_azure_key_edit)
        random_azure_key_label = random_llm_layout.labelForField(self.random_azure_key_edit)

        self.random_azure_version_edit = QtWidgets.QLineEdit()
        self.random_azure_version_edit.setPlaceholderText(
            "Enter Azure OpenAI API version (e.g. 2024-06-01)"
        )
        random_llm_layout.addRow("Azure API version", self.random_azure_version_edit)
        random_azure_version_label = random_llm_layout.labelForField(self.random_azure_version_edit)

        self.random_azure_endpoint_edit = QtWidgets.QLineEdit()
        self.random_azure_endpoint_edit.setPlaceholderText("Enter Azure OpenAI endpoint URL")
        random_llm_layout.addRow("Azure endpoint", self.random_azure_endpoint_edit)
        random_azure_endpoint_label = random_llm_layout.labelForField(self.random_azure_endpoint_edit)

        random_local_model_widget = QtWidgets.QWidget()
        random_local_model_layout = QtWidgets.QHBoxLayout(random_local_model_widget)
        random_local_model_layout.setContentsMargins(0, 0, 0, 0)
        self.random_local_model_path_edit = QtWidgets.QLineEdit()
        self.random_local_model_path_edit.setPlaceholderText("Select local model directory")
        random_local_model_layout.addWidget(self.random_local_model_path_edit)
        self.random_local_model_browse_btn = QtWidgets.QPushButton("Choose…")
        self.random_local_model_browse_btn.clicked.connect(self._on_select_local_model_dir)
        random_local_model_layout.addWidget(self.random_local_model_browse_btn)
        random_local_model_layout.setStretch(0, 1)
        random_local_model_layout.setStretch(1, 0)
        random_llm_layout.addRow("Local model dir", random_local_model_widget)
        random_local_model_label = random_llm_layout.labelForField(random_local_model_widget)

        self.random_local_max_seq_spin = QtWidgets.QSpinBox()
        self.random_local_max_seq_spin.setRange(0, 262144)
        self.random_local_max_seq_spin.setSpecialValueText("Model default")
        self.random_local_max_seq_spin.setSuffix(" tokens")
        random_llm_layout.addRow("Max sequence length", self.random_local_max_seq_spin)
        random_local_max_seq_label = random_llm_layout.labelForField(self.random_local_max_seq_spin)

        self.random_local_max_new_tokens_spin = QtWidgets.QSpinBox()
        self.random_local_max_new_tokens_spin.setRange(0, 8192)
        self.random_local_max_new_tokens_spin.setSpecialValueText("Model default")
        self.random_local_max_new_tokens_spin.setSuffix(" tokens")
        random_llm_layout.addRow("Max new tokens", self.random_local_max_new_tokens_spin)
        random_local_max_new_tokens_label = random_llm_layout.labelForField(
            self.random_local_max_new_tokens_spin
        )

        random_embed_row = QtWidgets.QHBoxLayout()
        self.random_embedding_path_edit = QtWidgets.QLineEdit()
        self.random_embedding_path_edit.setPlaceholderText("Select embedding model directory")
        random_embed_row.addWidget(self.random_embedding_path_edit)
        self.random_embedding_browse_btn = QtWidgets.QPushButton("Choose…")
        self.random_embedding_browse_btn.clicked.connect(self._on_select_embedding_model)
        random_embed_row.addWidget(self.random_embedding_browse_btn)
        random_embed_row.setStretch(0, 1)
        random_embed_row.setStretch(1, 0)
        random_llm_layout.addRow("Embedding model", random_embed_row)

        random_rerank_row = QtWidgets.QHBoxLayout()
        self.random_reranker_path_edit = QtWidgets.QLineEdit()
        self.random_reranker_path_edit.setPlaceholderText("Select re-ranker model directory")
        random_rerank_row.addWidget(self.random_reranker_path_edit)
        self.random_reranker_browse_btn = QtWidgets.QPushButton("Choose…")
        self.random_reranker_browse_btn.clicked.connect(self._on_select_reranker_model)
        random_rerank_row.addWidget(self.random_reranker_browse_btn)
        random_rerank_row.setStretch(0, 1)
        random_rerank_row.setStretch(1, 0)
        random_llm_layout.addRow("Re-ranker model", random_rerank_row)

        self.random_include_reasoning_checkbox = QtWidgets.QCheckBox("Include reasoning")
        self.random_include_reasoning_checkbox.setChecked(False)
        random_llm_layout.addRow("Include reasoning", self.random_include_reasoning_checkbox)

        self._random_azure_widgets = [
            w
            for w in (
                random_azure_key_label,
                self.random_azure_key_edit,
                random_azure_version_label,
                self.random_azure_version_edit,
                random_azure_endpoint_label,
                self.random_azure_endpoint_edit,
            )
            if isinstance(w, QtWidgets.QWidget)
        ]
        self._random_local_widgets = [
            w
            for w in (
                random_local_model_label,
                random_local_model_widget,
                self.random_local_max_seq_spin,
                random_local_max_seq_label,
                self.random_local_max_new_tokens_spin,
                random_local_max_new_tokens_label,
            )
            if isinstance(w, QtWidgets.QWidget)
        ]

        self._update_random_backend_fields()

        random_layout.addWidget(self.random_final_llm_group)
        scroll_layout.addWidget(self.random_config_container)

        self.ai_group = QtWidgets.QGroupBox("Active learning configuration")
        ai_layout = QtWidgets.QVBoxLayout(self.ai_group)
        self.ai_controls_container = QtWidgets.QWidget()
        ai_controls_layout = QtWidgets.QVBoxLayout(self.ai_controls_container)
        ai_controls_layout.setContentsMargins(0, 0, 0, 0)
        ai_config_group = QtWidgets.QGroupBox("Backend configuration")
        ai_config_layout = QtWidgets.QFormLayout(ai_config_group)
        ai_config_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.ai_batch_size_label = QtWidgets.QLabel()
        self.ai_batch_size_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        ai_config_layout.addRow("Batch size", self.ai_batch_size_label)
        self.ai_backend_combo = QtWidgets.QComboBox()
        self.ai_backend_combo.addItem("Azure OpenAI", "azure")
        self.ai_backend_combo.addItem("Local ExLlamaV2", "exllamav2")
        self.ai_backend_combo.currentIndexChanged.connect(self._update_ai_backend_fields)
        ai_config_layout.addRow("LLM backend", self.ai_backend_combo)
        self.ai_azure_key_edit = QtWidgets.QLineEdit()
        self.ai_azure_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.ai_azure_key_edit.setPlaceholderText("Enter Azure OpenAI API key")
        ai_config_layout.addRow("Azure API key", self.ai_azure_key_edit)
        ai_azure_key_label = ai_config_layout.labelForField(self.ai_azure_key_edit)
        self.ai_azure_version_edit = QtWidgets.QLineEdit()
        self.ai_azure_version_edit.setPlaceholderText(
            "Enter Azure OpenAI API version (e.g. 2024-06-01)"
        )
        ai_config_layout.addRow("Azure API version", self.ai_azure_version_edit)
        ai_azure_version_label = ai_config_layout.labelForField(self.ai_azure_version_edit)
        self.ai_azure_endpoint_edit = QtWidgets.QLineEdit()
        self.ai_azure_endpoint_edit.setPlaceholderText("Enter Azure OpenAI endpoint URL")
        ai_config_layout.addRow("Azure endpoint", self.ai_azure_endpoint_edit)
        ai_azure_endpoint_label = ai_config_layout.labelForField(self.ai_azure_endpoint_edit)
        ai_local_model_widget = QtWidgets.QWidget()
        ai_local_model_layout = QtWidgets.QHBoxLayout(ai_local_model_widget)
        ai_local_model_layout.setContentsMargins(0, 0, 0, 0)
        self.ai_local_model_path_edit = QtWidgets.QLineEdit()
        self.ai_local_model_path_edit.setPlaceholderText("Select local model directory")
        ai_local_model_layout.addWidget(self.ai_local_model_path_edit)
        self.ai_local_model_browse_btn = QtWidgets.QPushButton("Choose…")
        self.ai_local_model_browse_btn.clicked.connect(self._on_select_local_model_dir)
        ai_local_model_layout.addWidget(self.ai_local_model_browse_btn)
        ai_local_model_layout.setStretch(0, 1)
        ai_local_model_layout.setStretch(1, 0)
        ai_config_layout.addRow("Local model dir", ai_local_model_widget)
        ai_local_model_label = ai_config_layout.labelForField(ai_local_model_widget)
        self.ai_local_max_seq_spin = QtWidgets.QSpinBox()
        self.ai_local_max_seq_spin.setRange(0, 262144)
        self.ai_local_max_seq_spin.setSpecialValueText("Model default")
        self.ai_local_max_seq_spin.setSuffix(" tokens")
        ai_config_layout.addRow("Max sequence length", self.ai_local_max_seq_spin)
        ai_local_max_seq_label = ai_config_layout.labelForField(self.ai_local_max_seq_spin)
        self.ai_local_max_new_tokens_spin = QtWidgets.QSpinBox()
        self.ai_local_max_new_tokens_spin.setRange(0, 8192)
        self.ai_local_max_new_tokens_spin.setSpecialValueText("Model default")
        self.ai_local_max_new_tokens_spin.setSuffix(" tokens")
        ai_config_layout.addRow("Max new tokens", self.ai_local_max_new_tokens_spin)
        ai_local_max_new_tokens_label = ai_config_layout.labelForField(
            self.ai_local_max_new_tokens_spin
        )
        embed_row = QtWidgets.QHBoxLayout()
        self.ai_embedding_path_edit = QtWidgets.QLineEdit()
        self.ai_embedding_path_edit.setPlaceholderText("Select embedding model directory")
        embed_row.addWidget(self.ai_embedding_path_edit)
        self.ai_embedding_browse_btn = QtWidgets.QPushButton("Choose…")
        self.ai_embedding_browse_btn.clicked.connect(self._on_select_embedding_model)
        embed_row.addWidget(self.ai_embedding_browse_btn)
        embed_row.setStretch(0, 1)
        embed_row.setStretch(1, 0)
        ai_config_layout.addRow("Embedding model", embed_row)
        rerank_row = QtWidgets.QHBoxLayout()
        self.ai_reranker_path_edit = QtWidgets.QLineEdit()
        self.ai_reranker_path_edit.setPlaceholderText("Select re-ranker model directory")
        rerank_row.addWidget(self.ai_reranker_path_edit)
        self.ai_reranker_browse_btn = QtWidgets.QPushButton("Choose…")
        self.ai_reranker_browse_btn.clicked.connect(self._on_select_reranker_model)
        rerank_row.addWidget(self.ai_reranker_browse_btn)
        rerank_row.setStretch(0, 1)
        rerank_row.setStretch(1, 0)
        ai_config_layout.addRow("Re-ranker model", rerank_row)
        self._ai_azure_widgets = [
            w
            for w in (
                ai_azure_key_label,
                self.ai_azure_key_edit,
                ai_azure_version_label,
                self.ai_azure_version_edit,
                ai_azure_endpoint_label,
                self.ai_azure_endpoint_edit,
            )
            if isinstance(w, QtWidgets.QWidget)
        ]
        self._ai_local_widgets = [
            w
            for w in (
                ai_local_model_label,
                ai_local_model_widget,
                self.ai_local_max_seq_spin,
                ai_local_max_seq_label,
                self.ai_local_max_new_tokens_spin,
                ai_local_max_new_tokens_label,
            )
            if isinstance(w, QtWidgets.QWidget)
        ]
        self._update_ai_backend_fields()
        self.ai_disagreement_pct = QtWidgets.QDoubleSpinBox()
        self.ai_disagreement_pct.setRange(0.0, 1.0)
        self.ai_disagreement_pct.setDecimals(2)
        self.ai_disagreement_pct.setSingleStep(0.05)
        self.ai_disagreement_pct.setValue(0.30)
        ai_config_layout.addRow("Disagreement pct", self.ai_disagreement_pct)
        self.ai_uncertain_pct = QtWidgets.QDoubleSpinBox()
        self.ai_uncertain_pct.setRange(0.0, 1.0)
        self.ai_uncertain_pct.setDecimals(2)
        self.ai_uncertain_pct.setSingleStep(0.05)
        self.ai_uncertain_pct.setValue(0.30)
        ai_config_layout.addRow("LLM uncertain pct", self.ai_uncertain_pct)
        self.ai_easy_pct = QtWidgets.QDoubleSpinBox()
        self.ai_easy_pct.setRange(0.0, 1.0)
        self.ai_easy_pct.setDecimals(2)
        self.ai_easy_pct.setSingleStep(0.05)
        self.ai_easy_pct.setValue(0.10)
        ai_config_layout.addRow("LLM certain pct", self.ai_easy_pct)
        self.ai_diversity_pct = QtWidgets.QDoubleSpinBox()
        self.ai_diversity_pct.setRange(0.0, 1.0)
        self.ai_diversity_pct.setDecimals(2)
        self.ai_diversity_pct.setSingleStep(0.05)
        self.ai_diversity_pct.setValue(0.30)
        ai_config_layout.addRow("Diversity pct", self.ai_diversity_pct)
        self.ai_final_llm_checkbox = QtWidgets.QCheckBox("Run final LLM labeling")
        self.ai_final_llm_checkbox.setChecked(True)
        self.ai_final_llm_checkbox.toggled.connect(self._on_ai_final_llm_toggled)
        ai_config_layout.addRow("Final LLM labeling", self.ai_final_llm_checkbox)

        self.ai_include_reasoning_checkbox = QtWidgets.QCheckBox("Include reasoning")
        self.ai_include_reasoning_checkbox.setChecked(False)
        ai_config_layout.addRow("Include reasoning", self.ai_include_reasoning_checkbox)
        if str(self.pheno_row["level"] or "single_doc") == "single_doc":
            self.ai_single_doc_context_combo = QtWidgets.QComboBox()
            self.ai_single_doc_context_combo.addItem("RAG snippets", "rag")
            self.ai_single_doc_context_combo.addItem("Full document", "full")
            self.ai_single_doc_context_combo.setToolTip(
                "Choose how the LLM context is built for single-document phenotypes."
            )
            ai_config_layout.addRow("Single-doc context", self.ai_single_doc_context_combo)
        self.ai_advanced_settings_btn = QtWidgets.QPushButton("Advanced settings…")
        self.ai_advanced_settings_btn.setToolTip(
            "Open the full AI engine configuration (indexing, RAG, LLM, buckets, and more)."
        )
        self.ai_advanced_settings_btn.clicked.connect(self._open_ai_advanced_settings)
        ai_config_layout.addRow("Advanced", self.ai_advanced_settings_btn)
        pct_hint = QtWidgets.QLabel("Fractions should sum to ≤ 1.0; remaining slots are auto-filled.")
        pct_hint.setWordWrap(True)
        ai_config_layout.addRow(pct_hint)
        ai_controls_layout.addWidget(ai_config_group)
        prior_label = QtWidgets.QLabel("Prior rounds to include")
        ai_controls_layout.addWidget(prior_label)
        self.ai_rounds_list = QtWidgets.QListWidget()
        self.ai_rounds_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        ai_controls_layout.addWidget(self.ai_rounds_list)
        ai_button_row = QtWidgets.QHBoxLayout()
        self.ai_generate_btn = QtWidgets.QPushButton("Generate AI round")
        self.ai_generate_btn.clicked.connect(self._on_ai_generate)
        ai_button_row.addWidget(self.ai_generate_btn)
        self.ai_cancel_btn = QtWidgets.QPushButton("Cancel run")
        self.ai_cancel_btn.clicked.connect(self._on_cancel_ai_job)
        self.ai_cancel_btn.setEnabled(False)
        ai_button_row.addWidget(self.ai_cancel_btn)
        ai_button_row.addStretch()
        ai_controls_layout.addLayout(ai_button_row)
        ai_layout.addWidget(self.ai_controls_container)
        scroll_layout.addWidget(self.ai_group)

        scroll_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._refresh_corpus_options()
        self._refresh_labelset_options()
        self._refresh_reviewer_options()
        self._refresh_ai_round_options()
        self._update_ai_batch_size_label()
        if hasattr(self, "ai_embedding_path_edit"):
            self.ai_embedding_path_edit.setText(os.getenv("MED_EMBED_MODEL_NAME", "E:/medembed_small"))
        if hasattr(self, "ai_reranker_path_edit"):
            self.ai_reranker_path_edit.setText(os.getenv("RERANKER_MODEL_NAME", "E:/ms_marco_MiniLM_L6_v2"))
        backend_env = (os.getenv("LLM_BACKEND") or "azure").lower()
        if backend_env in {"exllama", "exllamav2", "local"}:
            backend_choice_key = "exllamav2"
        else:
            backend_choice_key = "azure"
        if hasattr(self, "ai_backend_combo"):
            idx = self.ai_backend_combo.findData(backend_choice_key)
            if idx < 0:
                idx = 0
            self.ai_backend_combo.setCurrentIndex(idx)
            self._update_ai_backend_fields()
        if hasattr(self, "ai_azure_key_edit"):
            self.ai_azure_key_edit.setText(os.getenv("AZURE_OPENAI_API_KEY", ""))
        if hasattr(self, "ai_azure_version_edit"):
            self.ai_azure_version_edit.setText(
                os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-28")
            )
        if hasattr(self, "ai_azure_endpoint_edit"):
            self.ai_azure_endpoint_edit.setText(os.getenv("AZURE_OPENAI_ENDPOINT", "https://spd-prod-openai-va-apim.azure-api.us/api"))
        if hasattr(self, "ai_local_model_path_edit"):
            self.ai_local_model_path_edit.setText(os.getenv("LOCAL_LLM_MODEL_DIR", ""))
        if hasattr(self, "ai_local_max_seq_spin"):
            try:
                max_seq_env = int(os.getenv("LOCAL_LLM_MAX_SEQ_LEN", "0") or 0)
            except ValueError:
                max_seq_env = 0
            self.ai_local_max_seq_spin.setValue(max_seq_env)
        if hasattr(self, "ai_local_max_new_tokens_spin"):
            try:
                max_new_env = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "0") or 0)
            except ValueError:
                max_new_env = 0
            self.ai_local_max_new_tokens_spin.setValue(max_new_env)
        if hasattr(self, "random_embedding_path_edit"):
            self.random_embedding_path_edit.setText(os.getenv("MED_EMBED_MODEL_NAME", "E:/medembed_small"))
        if hasattr(self, "random_reranker_path_edit"):
            self.random_reranker_path_edit.setText(os.getenv("RERANKER_MODEL_NAME", "E:/ms_marco_MiniLM_L6_v2"))
        if hasattr(self, "random_backend_combo"):
            idx = self.random_backend_combo.findData(backend_choice_key)
            if idx < 0:
                idx = 0
            self.random_backend_combo.setCurrentIndex(idx)
            self._update_random_backend_fields()
        if hasattr(self, "random_azure_key_edit"):
            self.random_azure_key_edit.setText(os.getenv("AZURE_OPENAI_API_KEY", ""))
        if hasattr(self, "random_azure_version_edit"):
            self.random_azure_version_edit.setText(
                os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-28")
            )
        if hasattr(self, "random_azure_endpoint_edit"):
            self.random_azure_endpoint_edit.setText(os.getenv("AZURE_OPENAI_ENDPOINT", "https://spd-prod-openai-va-apim.azure-api.us/api"))
        if hasattr(self, "random_local_model_path_edit"):
            self.random_local_model_path_edit.setText(os.getenv("LOCAL_LLM_MODEL_DIR", ""))
        if hasattr(self, "random_local_max_seq_spin"):
            try:
                random_max_seq_env = int(os.getenv("LOCAL_LLM_MAX_SEQ_LEN", "0") or 0)
            except ValueError:
                random_max_seq_env = 0
            self.random_local_max_seq_spin.setValue(random_max_seq_env)
        if hasattr(self, "random_local_max_new_tokens_spin"):
            try:
                random_max_new_env = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "0") or 0)
            except ValueError:
                random_max_new_env = 0
            self.random_local_max_new_tokens_spin.setValue(random_max_new_env)
        if hasattr(self, "random_final_llm_checkbox"):
            self._on_random_final_llm_toggled(self.random_final_llm_checkbox.isChecked())
        if hasattr(self, "ai_final_llm_checkbox"):
            self._on_ai_final_llm_toggled(self.ai_final_llm_checkbox.isChecked())
        if hasattr(self, "assisted_review_checkbox"):
            self._on_assisted_review_toggled(self.assisted_review_checkbox.isChecked())
        self.total_n_spin.valueChanged.connect(self._update_ai_batch_size_label)
        self._on_generation_mode_changed()

    def _collect_filters(self) -> SamplingFilters:
        conditions: List[MetadataFilterCondition] = []
        if hasattr(self, "filter_list"):
            for row in range(self.filter_list.count()):
                item = self.filter_list.item(row)
                data = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, MetadataFilterCondition):
                    conditions.append(data)
                elif isinstance(data, Mapping):
                    try:
                        condition = MetadataFilterCondition.from_payload(data)
                    except Exception:  # noqa: BLE001
                        continue
                    conditions.append(condition)
        match_any = False
        if hasattr(self, "filter_logic_combo"):
            data = self.filter_logic_combo.currentData()
            match_any = bool(data)
        return SamplingFilters(metadata_filters=conditions, match_any=match_any)

    def _load_reviewed_unit_ids(self, corpus_id: Optional[str]) -> Set[str]:
        pheno_id = self.pheno_row["pheno_id"]
        level_value = self._safe_mapping_get(self.pheno_row, "level", "single_doc")
        level = str(level_value or "single_doc")
        reviewed: Set[str] = set()
        if not corpus_id:
            return reviewed
        try:
            rounds = self.ctx.list_rounds(pheno_id)
        except Exception:
            return reviewed
        for round_row in rounds:
            round_number = self._safe_mapping_get(round_row, "round_number")
            if round_number is None:
                continue
            round_id = self._safe_mapping_get(round_row, "round_id")
            if round_id:
                config = self.ctx.get_round_config(str(round_id))
                if config and config.get("corpus_id") and config.get("corpus_id") != corpus_id:
                    continue
            try:
                round_dir = self.ctx.resolve_round_dir(pheno_id, int(round_number))
            except Exception:
                continue
            manifest_path = round_dir / "manifest.csv"
            if not manifest_path.exists():
                continue
            try:
                with manifest_path.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        if level == "multi_doc":
                            key = row.get("patient_icn")
                        else:
                            key = row.get("doc_id") or row.get("unit_id")
                            if not key:
                                key = row.get("patient_icn")
                        if key:
                            reviewed.add(str(key))
            except Exception:
                continue
        return reviewed

    def _row_unit_identifier(self, row: sqlite3.Row | Dict[str, object]) -> Optional[str]:
        keys = ["unit_id"]
        if self.pheno_row["level"] == "multi_doc":
            keys.append("patient_icn")
        else:
            keys.extend(["doc_id", "patient_icn"])
        for key in keys:
            value: Optional[object] = None
            if isinstance(row, dict):
                value = row.get(key)
            else:
                try:
                    value = row[key]  # type: ignore[index]
                except (KeyError, IndexError, TypeError):
                    value = None
            if value not in (None, ""):
                return str(value)
        return None

    def _prompt_reviewers(self) -> Optional[List[Dict[str, str]]]:
        reviewers: List[Dict[str, str]] = []
        for row in range(self.reviewer_list.count()):
            item = self.reviewer_list.item(row)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                reviewers.append(data)
        if not reviewers:
            QtWidgets.QMessageBox.warning(self, "Reviewers", "Add at least one reviewer.")
            return []
        return reviewers

    def _refresh_labelset_options(self) -> None:
        self._labelset_options = self._load_labelset_ids()
        self.labelset_combo.blockSignals(True)
        self.labelset_combo.clear()
        for labelset_id in self._labelset_options:
            self.labelset_combo.addItem(labelset_id)
        if self._labelset_options:
            self.labelset_combo.setCurrentIndex(0)
        else:
            default_id = f"auto_{self.pheno_row['pheno_id']}"
            self.labelset_combo.setEditText(default_id)
        self.labelset_combo.blockSignals(False)

    def _refresh_reviewer_options(self) -> None:
        self._available_reviewers = self._load_existing_reviewers()
        self.reviewer_combo.blockSignals(True)
        self.reviewer_combo.clear()
        self.reviewer_combo.addItem("Select existing reviewer…", None)
        for reviewer in self._available_reviewers:
            display = f"{reviewer['name']} ({reviewer['id']})"
            self.reviewer_combo.addItem(display, reviewer)
        self.reviewer_combo.blockSignals(False)

    def _refresh_ai_round_options(self) -> None:
        if not hasattr(self, "ai_rounds_list"):
            return
        self.ai_rounds_list.clear()
        try:
            rounds = self.ctx.list_rounds(self.pheno_row["pheno_id"])
        except Exception:  # noqa: BLE001
            rounds = []
        for round_row in rounds:
            round_number = self._safe_mapping_get(round_row, "round_number")
            if round_number is None:
                continue
            item = QtWidgets.QListWidgetItem(f"Round {round_number}")
            try:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, int(round_number))
            except Exception:  # noqa: BLE001
                continue
            self.ai_rounds_list.addItem(item)

    def _using_ai_backend(self) -> bool:
        radio = getattr(self, "active_learning_radio", None)
        return bool(radio and radio.isChecked())

    def _on_generation_mode_changed(self) -> None:
        using_ai = self._using_ai_backend()
        if hasattr(self, "random_config_container"):
            self.random_config_container.setVisible(not using_ai)
        if hasattr(self, "ai_group"):
            show_ai = using_ai or self._ai_job_running
            self.ai_group.setVisible(show_ai)
        if hasattr(self, "ai_controls_container"):
            should_enable = using_ai or self._ai_job_running
            self.ai_controls_container.setEnabled(should_enable)
        self._update_ai_buttons()

    def _update_ai_buttons(self) -> None:
        enabled = self._using_ai_backend()
        active = bool(enabled and not self._ai_job_running)
        if hasattr(self, "ai_generate_btn"):
            self.ai_generate_btn.setEnabled(active)
        if hasattr(self, "ai_cancel_btn"):
            self.ai_cancel_btn.setEnabled(self._ai_job_running)
        if hasattr(self, "ai_rounds_list"):
            self.ai_rounds_list.setEnabled(active)
        if hasattr(self, "ai_controls_container"):
            container_enabled = enabled or self._ai_job_running
            self.ai_controls_container.setEnabled(container_enabled)

    def _on_ai_generate(self) -> None:
        if self._ai_job_running:
            return
        self._start_ai_round()

    def _on_cancel_ai_job(self) -> None:
        if not self._ai_job_running or not self._ai_worker:
            return
        cancel_event = getattr(self._ai_worker, "cancel_event", None)
        already_set = False
        if isinstance(cancel_event, threading.Event):
            already_set = cancel_event.is_set()
            cancel_event.set()
        if already_set:
            return
        self._append_ai_log("Cancellation requested by user…")
        QtCore.QMetaObject.invokeMethod(
            self._ai_worker,
            "cancel",
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

    def _append_ai_log(self, message: str) -> None:
        if not self.ai_log_output:
            return
        if message.startswith("\r"):
            text = message[1:].strip()
            if not text:
                return
            if not self._ai_progress_active:
                stamp = datetime.utcnow().strftime("%H:%M:%S")
                self._ai_progress_stamp = stamp
                block_number = self._write_ai_log_line(text, stamp, append_newline=True)
                self._ai_progress_active = True
                self._ai_progress_block_number = block_number
            else:
                stamp = self._ai_progress_stamp or datetime.utcnow().strftime("%H:%M:%S")
                self._ai_progress_block_number = self._replace_last_ai_log_line(
                    text,
                    stamp,
                    self._ai_progress_block_number,
                )
            self._ai_progress_text = text
            return

        clean_message = message.strip()
        if self._ai_progress_active:
            if clean_message == self._ai_progress_text:
                self._ai_progress_active = False
                self._ai_progress_text = clean_message
                self._ai_progress_block_number = None
                return
            self._ai_progress_active = False
            self._ai_progress_block_number = None

        stamp = datetime.utcnow().strftime("%H:%M:%S")
        self._ai_progress_active = False
        self._ai_progress_stamp = stamp
        self._ai_progress_text = ""
        self._ai_progress_block_number = None
        self._write_ai_log_line(message, stamp, append_newline=True)

    def _write_ai_log_line(self, text: str, stamp: str, append_newline: bool = False) -> int:
        if not self.ai_log_output:
            return -1
        doc = self.ai_log_output.document()
        cursor = self.ai_log_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        if append_newline and not doc.isEmpty():
            cursor.insertBlock()
        cursor.insertText(f"[{stamp}] {text}")
        self.ai_log_output.setTextCursor(cursor)
        self.ai_log_output.ensureCursorVisible()
        return doc.blockCount() - 1

    def _replace_last_ai_log_line(
        self,
        text: str,
        stamp: str,
        block_number: Optional[int],
    ) -> Optional[int]:
        if not self.ai_log_output:
            return None
        doc = self.ai_log_output.document()
        if block_number is None:
            target_block = doc.lastBlock()
        else:
            target_block = doc.findBlockByNumber(block_number)
            if not target_block.isValid():
                target_block = doc.lastBlock()
        target_block_number = target_block.blockNumber()
        cursor = QtGui.QTextCursor(target_block)
        cursor.select(QtGui.QTextCursor.BlockUnderCursor)
        cursor.insertText(f"[{stamp}] {text}")
        self.ai_log_output.setTextCursor(cursor)
        self.ai_log_output.ensureCursorVisible()
        return target_block_number

    def reject(self) -> None:  # type: ignore[override]
        if self._ai_job_running:
            confirm = QtWidgets.QMessageBox.question(
                self,
                "Cancel AI run",
                "The AI backend is still running. Cancel the run before closing?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
                self._on_cancel_ai_job()
            return
        super().reject()

    def _selected_prior_round_numbers(self) -> List[int]:
        rounds: List[int] = []
        if not hasattr(self, "ai_rounds_list"):
            return rounds
        for index in range(self.ai_rounds_list.count()):
            item = self.ai_rounds_list.item(index)
            if item and item.isSelected():
                value = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(value, int):
                    rounds.append(value)
        return rounds

    def _build_ai_context(self) -> Optional[RoundCreationContext]:
        pheno_id = self.pheno_row["pheno_id"]
        pheno_level = str(self.pheno_row["level"] or "single_doc")
        ctx = self.ctx
        project_id = ctx.current_project_id()
        if not project_id:
            QtWidgets.QMessageBox.critical(
                self,
                "Round",
                "Project metadata is missing; reload the project and try again.",
            )
            return None
        db = ctx.require_db()
        reviewers = self._prompt_reviewers()
        if not reviewers:
            return None
        labelset_id = self.labelset_combo.currentText().strip() or f"auto_{pheno_id}"
        created_at = QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)
        with db.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM label_sets WHERE labelset_id=?",
                (labelset_id,),
            ).fetchone()
        labelset_missing = not bool(exists)
        default_labels: List[Dict[str, object]] = []
        if labelset_missing:
            default_labels.append(
                {
                    "label_id": str(uuid.uuid4()),
                    "name": "Has_phenotype",
                    "type": "boolean",
                    "required": 1,
                    "na_allowed": 0,
                    "options": [
                        {"value": "yes", "display": "Yes"},
                        {"value": "no", "display": "No"},
                        {"value": "unknown", "display": "Unknown"},
                    ],
                }
            )
        seed = self.seed_spin.value()
        overlap = self.overlap_spin.value()
        total_n = self.total_n_spin.value()
        if total_n < overlap:
            QtWidgets.QMessageBox.warning(
                self,
                "Round",
                "Total N must be greater than or equal to the overlap count.",
            )
            return None
        corpus_id = self._selected_corpus_id
        if not corpus_id:
            QtWidgets.QMessageBox.warning(self, "Round", "Select a corpus for this round.")
            return None
        try:
            corpus_record = self.ctx.get_corpus(corpus_id)
        except Exception:  # noqa: BLE001
            corpus_record = None
        created_by = "admin"
        project_row = ctx.project_row or ctx._load_project_row()
        if project_row and project_row.get("created_by"):
            created_by = str(project_row["created_by"])
        status = self.status_combo.currentText()
        storage_path: Optional[str] = None
        try:
            if "storage_path" in self.pheno_row.keys():
                raw_storage = self.pheno_row["storage_path"]
                if raw_storage:
                    storage_path = str(raw_storage)
        except Exception:  # noqa: BLE001
            storage_path = None

        assisted_enabled = (
            bool(getattr(self, "assisted_review_checkbox", None))
            and bool(self.assisted_review_checkbox.isChecked())
        )
        assisted_top_n = (
            int(self.assisted_review_spin.value())
            if assisted_enabled and hasattr(self, "assisted_review_spin")
            else 0
        )

        backend_cfg: Dict[str, object] = {}
        backend_choice = self._current_ai_backend()
        if backend_choice:
            backend_cfg["backend"] = backend_choice
        if hasattr(self, "ai_embedding_path_edit"):
            embed_path = self.ai_embedding_path_edit.text().strip()
            if embed_path:
                backend_cfg["embedding_model_dir"] = embed_path
        if hasattr(self, "ai_reranker_path_edit"):
            rerank_path = self.ai_reranker_path_edit.text().strip()
            if rerank_path:
                backend_cfg["reranker_model_dir"] = rerank_path
        if backend_choice == "azure":
            if hasattr(self, "ai_azure_version_edit"):
                azure_version = self.ai_azure_version_edit.text().strip()
                if azure_version:
                    backend_cfg["azure_api_version"] = azure_version
            if hasattr(self, "ai_azure_endpoint_edit"):
                azure_endpoint = self.ai_azure_endpoint_edit.text().strip()
                if azure_endpoint:
                    backend_cfg["azure_endpoint"] = azure_endpoint
        else:
            if hasattr(self, "ai_local_model_path_edit"):
                model_dir = self.ai_local_model_path_edit.text().strip()
                if model_dir:
                    backend_cfg["local_model_dir"] = model_dir
            if hasattr(self, "ai_local_max_seq_spin"):
                max_seq = int(self.ai_local_max_seq_spin.value())
                if max_seq > 0:
                    backend_cfg["local_max_seq_len"] = max_seq
            if hasattr(self, "ai_local_max_new_tokens_spin"):
                max_new = int(self.ai_local_max_new_tokens_spin.value())
                if max_new > 0:
                    backend_cfg["local_max_new_tokens"] = max_new
        if (
            pheno_level == "single_doc"
            and isinstance(self.ai_single_doc_context_combo, QtWidgets.QComboBox)
        ):
            mode_value = self.ai_single_doc_context_combo.currentData()
            if not isinstance(mode_value, str) or not mode_value:
                mode_value = "rag"
            existing_llmfirst = backend_cfg.get("llmfirst")
            if isinstance(existing_llmfirst, dict):
                llmfirst_cfg = existing_llmfirst
            else:
                llmfirst_cfg = {}
                backend_cfg["llmfirst"] = llmfirst_cfg
            llmfirst_cfg["single_doc_context"] = mode_value

        backend_cfg["config_overrides"] = copy.deepcopy(self._collect_ai_overrides())

        return RoundCreationContext(
            pheno_id=pheno_id,
            pheno_level=pheno_level,
            project_id=project_id,
            phenotype_storage_path=storage_path,
            seed=seed,
            overlap=overlap,
            total_n=total_n,
            status=status,
            labelset_id=labelset_id,
            labelset_missing=labelset_missing,
            default_labels=default_labels,
            reviewers=reviewers,
            corpus_id=corpus_id,
            corpus_record=corpus_record,
            created_at=created_at,
            created_by=created_by,
            db=db,
            assisted_review_enabled=assisted_enabled,
            assisted_review_top_n=assisted_top_n,
            ai_backend_overrides=backend_cfg,
        )

    def _start_ai_round(self) -> bool:
        context = self._build_ai_context()
        if not context:
            return False
        prior_rounds = self._selected_prior_round_numbers()
        timestamp = datetime.utcnow().isoformat()
        round_number = self._next_round_number()
        try:
            round_dir = ensure_dir(self.ctx.resolve_round_dir(context.pheno_id, round_number))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Round", str(exc))
            return False
        round_id = f"{context.pheno_id}_r{round_number}"
        job = AIRoundJobConfig(
            context=context,
            round_number=round_number,
            round_id=round_id,
            round_dir=round_dir,
            prior_rounds=prior_rounds,
            timestamp=timestamp,
        )
        project_root = self.ctx.require_project()
        cfg_overrides = self._collect_ai_overrides()
        env_overrides = self._collect_ai_environment()
        if env_overrides is None:
            return False
        self._open_ai_log_dialog()
        if not prior_rounds:
            self._append_ai_log("No prior rounds selected; running cold-start configuration.")
        worker = AIRoundWorker(
            project_root,
            job,
            finalize=True,
            cfg_overrides=cfg_overrides,
            cleanup_dir=False,
            env_overrides=env_overrides,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        worker.log_message.connect(self._append_ai_log)
        worker.finished.connect(self._handle_ai_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_ai_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        self._ai_thread = thread
        self._ai_worker = worker
        self._ai_pending_job = job
        self._ai_job_running = True
        if self.ai_log_output:
            self.ai_log_output.clear()
        self._ai_progress_active = False
        self._ai_progress_stamp = ""
        self._ai_progress_text = ""
        self._append_ai_log("Starting AI backend…")
        self._update_ai_buttons()
        ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_button:
            ok_button.setEnabled(False)
        thread.start()
        return True

    def _open_ai_log_dialog(self) -> None:
        dialog = self._ai_log_dialog
        if dialog is None:
            dialog = AIRoundLogDialog(self)
            dialog.cancel_requested.connect(self._on_cancel_ai_job)
            dialog.finished.connect(self._on_ai_log_dialog_closed)
            self._ai_log_dialog = dialog
        dialog.reset_for_run()
        self.ai_log_output = dialog.log_output
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _mark_ai_log_complete(self) -> None:
        if self._ai_log_dialog is None:
            return
        self._ai_log_dialog.mark_complete()

    def _close_ai_log_dialog(self) -> None:
        if self._ai_log_dialog is None:
            return
        dialog = self._ai_log_dialog
        self._ai_log_dialog = None
        self.ai_log_output = None
        dialog.close()
        dialog.deleteLater()

    def _on_ai_log_dialog_closed(self) -> None:
        self._ai_log_dialog = None
        self.ai_log_output = None

    def _collect_ai_environment(self) -> Optional[Dict[str, str]]:
        if not self._using_ai_backend():
            return {}
        env: Dict[str, str] = {}
        missing: List[str] = []
        embed_path = ""
        if hasattr(self, "ai_embedding_path_edit"):
            embed_path = self.ai_embedding_path_edit.text().strip()
        rerank_path = ""
        if hasattr(self, "ai_reranker_path_edit"):
            rerank_path = self.ai_reranker_path_edit.text().strip()
        if not embed_path:
            missing.append("embedding model directory")
        else:
            embed_dir = Path(embed_path).expanduser()
            if not embed_dir.exists() or not embed_dir.is_dir():
                QtWidgets.QMessageBox.warning(
                    self,
                    "AI backend",
                    "The selected embedding model directory does not exist or is not a directory.",
                )
                return None
            env["MED_EMBED_MODEL_NAME"] = str(embed_dir)
        if not rerank_path:
            missing.append("re-ranker model directory")
        else:
            rerank_dir = Path(rerank_path).expanduser()
            if not rerank_dir.exists() or not rerank_dir.is_dir():
                QtWidgets.QMessageBox.warning(
                    self,
                    "AI backend",
                    "The selected re-ranker model directory does not exist or is not a directory.",
                )
                return None
            env["RERANKER_MODEL_NAME"] = str(rerank_dir)
        backend_choice = self._current_ai_backend()
        if backend_choice == "azure":
            azure_key = ""
            if hasattr(self, "ai_azure_key_edit"):
                azure_key = self.ai_azure_key_edit.text().strip()
            azure_version = ""
            if hasattr(self, "ai_azure_version_edit"):
                azure_version = self.ai_azure_version_edit.text().strip()
            azure_endpoint = ""
            if hasattr(self, "ai_azure_endpoint_edit"):
                azure_endpoint = self.ai_azure_endpoint_edit.text().strip()
            if not azure_key:
                missing.append("Azure API key")
            else:
                env["AZURE_OPENAI_API_KEY"] = azure_key
            if not azure_version:
                missing.append("Azure API version")
            else:
                env["AZURE_OPENAI_API_VERSION"] = azure_version
            if not azure_endpoint:
                missing.append("Azure endpoint")
            else:
                env["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
            env["LLM_BACKEND"] = "azure"
        else:
            model_path = ""
            if hasattr(self, "ai_local_model_path_edit"):
                model_path = self.ai_local_model_path_edit.text().strip()
            if not model_path:
                missing.append("local model directory")
            else:
                model_dir = Path(model_path).expanduser()
                if not model_dir.exists() or not model_dir.is_dir():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "AI backend",
                        "The selected local LLM model directory does not exist or is not a directory.",
                    )
                    return None
                env["LOCAL_LLM_MODEL_DIR"] = str(model_dir)
            if hasattr(self, "ai_local_max_seq_spin"):
                max_seq = int(self.ai_local_max_seq_spin.value())
                if max_seq > 0:
                    env["LOCAL_LLM_MAX_SEQ_LEN"] = str(max_seq)
            if hasattr(self, "ai_local_max_new_tokens_spin"):
                max_new = int(self.ai_local_max_new_tokens_spin.value())
                if max_new > 0:
                    env["LOCAL_LLM_MAX_NEW_TOKENS"] = str(max_new)
            env["LLM_BACKEND"] = "exllamav2"
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "AI backend",
                f"Provide the {' and '.join(missing)} before running the AI backend.",
            )
            return None
        return env

    def _collect_random_llm_environment(
        self,
        *,
        context_label: str,
        require_checkbox: bool,
    ) -> Optional[Dict[str, str]]:
        if require_checkbox:
            checkbox = getattr(self, "random_final_llm_checkbox", None)
            if not isinstance(checkbox, QtWidgets.QCheckBox) or not checkbox.isChecked():
                return {}
        env: Dict[str, str] = {}
        missing: List[str] = []
        embed_path = (
            self.random_embedding_path_edit.text().strip()
            if hasattr(self, "random_embedding_path_edit")
            else ""
        )
        rerank_path = (
            self.random_reranker_path_edit.text().strip()
            if hasattr(self, "random_reranker_path_edit")
            else ""
        )
        if not embed_path:
            missing.append("embedding model directory")
        else:
            embed_dir = Path(embed_path).expanduser()
            if not embed_dir.exists() or not embed_dir.is_dir():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Final LLM labeling",
                    "The selected embedding model directory does not exist or is not a directory.",
                )
                return None
            env["MED_EMBED_MODEL_NAME"] = str(embed_dir)
        if not rerank_path:
            missing.append("re-ranker model directory")
        else:
            rerank_dir = Path(rerank_path).expanduser()
            if not rerank_dir.exists() or not rerank_dir.is_dir():
                QtWidgets.QMessageBox.warning(
                    self,
                    context_label,
                    "The selected re-ranker model directory does not exist or is not a directory.",
                )
                return None
            env["RERANKER_MODEL_NAME"] = str(rerank_dir)
        backend_choice = RoundBuilderDialog._current_random_backend(self)
        if backend_choice == "azure":
            azure_key = (
                self.random_azure_key_edit.text().strip()
                if hasattr(self, "random_azure_key_edit")
                else ""
            )
            azure_version = (
                self.random_azure_version_edit.text().strip()
                if hasattr(self, "random_azure_version_edit")
                else ""
            )
            azure_endpoint = (
                self.random_azure_endpoint_edit.text().strip()
                if hasattr(self, "random_azure_endpoint_edit")
                else ""
            )
            if not azure_key:
                missing.append("Azure API key")
            else:
                env["AZURE_OPENAI_API_KEY"] = azure_key
            if not azure_version:
                missing.append("Azure API version")
            else:
                env["AZURE_OPENAI_API_VERSION"] = azure_version
            if not azure_endpoint:
                missing.append("Azure endpoint")
            else:
                env["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
            env["LLM_BACKEND"] = "azure"
        else:
            model_path = (
                self.random_local_model_path_edit.text().strip()
                if hasattr(self, "random_local_model_path_edit")
                else ""
            )
            if not model_path:
                missing.append("local model directory")
            else:
                model_dir = Path(model_path).expanduser()
                if not model_dir.exists() or not model_dir.is_dir():
                    QtWidgets.QMessageBox.warning(
                        self,
                        context_label,
                        "The selected local LLM model directory does not exist or is not a directory.",
                    )
                    return None
                env["LOCAL_LLM_MODEL_DIR"] = str(model_dir)
            if hasattr(self, "random_local_max_seq_spin"):
                max_seq = int(self.random_local_max_seq_spin.value())
                if max_seq > 0:
                    env["LOCAL_LLM_MAX_SEQ_LEN"] = str(max_seq)
            if hasattr(self, "random_local_max_new_tokens_spin"):
                max_new = int(self.random_local_max_new_tokens_spin.value())
                if max_new > 0:
                    env["LOCAL_LLM_MAX_NEW_TOKENS"] = str(max_new)
            env["LLM_BACKEND"] = "exllamav2"
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                context_label,
                f"Provide the {' and '.join(missing)} before continuing.",
            )
            return None
        return env

    def _collect_random_final_llm_environment(self) -> Optional[Dict[str, str]]:
        return self._collect_random_llm_environment(
            context_label="Final LLM labeling",
            require_checkbox=True,
        )

    def _collect_ai_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = copy.deepcopy(self._ai_engine_overrides) if self._ai_engine_overrides else {}
        if not self._using_ai_backend():
            return overrides
        select: Dict[str, Any] = overrides.get("select", {}) if isinstance(overrides.get("select"), Mapping) else {}
        if hasattr(self, "total_n_spin"):
            select["batch_size"] = int(self.total_n_spin.value())
        if hasattr(self, "ai_disagreement_pct"):
            select["pct_disagreement"] = float(self.ai_disagreement_pct.value())
        if hasattr(self, "ai_uncertain_pct"):
            select["pct_uncertain"] = float(self.ai_uncertain_pct.value())
        if hasattr(self, "ai_easy_pct"):
            select["pct_easy_qc"] = float(self.ai_easy_pct.value())
        if hasattr(self, "ai_diversity_pct"):
            select["pct_diversity"] = float(self.ai_diversity_pct.value())
        if select:
            overrides["select"] = select
        if hasattr(self, "ai_final_llm_checkbox"):
            overrides["final_llm_labeling"] = bool(self.ai_final_llm_checkbox.isChecked())
        if "final_llm_labeling_n_consistency" not in overrides:
            try:
                overrides["final_llm_labeling_n_consistency"] = int(
                    engine.OrchestratorConfig().final_llm_labeling_n_consistency
                )
            except Exception:  # noqa: BLE001
                overrides["final_llm_labeling_n_consistency"] = 1
        include_reasoning_value: Optional[bool] = None
        include_checkbox = getattr(self, "ai_include_reasoning_checkbox", None)
        if isinstance(include_checkbox, QtWidgets.QCheckBox):
            include_reasoning_value = bool(include_checkbox.isChecked())
            overrides["final_llm_include_reasoning"] = include_reasoning_value
        llmfirst_overrides: Dict[str, Any] = {}
        if (
            str(self.pheno_row["level"] or "single_doc") == "single_doc"
            and isinstance(self.ai_single_doc_context_combo, QtWidgets.QComboBox)
        ):
            mode_value = self.ai_single_doc_context_combo.currentData()
            if not isinstance(mode_value, str) or not mode_value:
                mode_value = "rag"
            llmfirst_overrides["single_doc_context"] = mode_value
        if llmfirst_overrides:
            overrides["llmfirst"] = llmfirst_overrides
        llm_overrides: Dict[str, Any] = {}
        backend_choice = self._current_ai_backend()
        if backend_choice:
            llm_overrides["backend"] = backend_choice
        if include_reasoning_value is not None:
            llm_overrides["include_reasoning"] = include_reasoning_value
        if backend_choice == "azure":
            if hasattr(self, "ai_azure_version_edit"):
                version = self.ai_azure_version_edit.text().strip()
                if version:
                    llm_overrides["azure_api_version"] = version
            if hasattr(self, "ai_azure_endpoint_edit"):
                endpoint = self.ai_azure_endpoint_edit.text().strip()
                if endpoint:
                    llm_overrides["azure_endpoint"] = endpoint
        else:
            if hasattr(self, "ai_local_model_path_edit"):
                model_dir = self.ai_local_model_path_edit.text().strip()
                if model_dir:
                    llm_overrides["local_model_dir"] = model_dir
            if hasattr(self, "ai_local_max_seq_spin"):
                max_seq = int(self.ai_local_max_seq_spin.value())
                if max_seq > 0:
                    llm_overrides["local_max_seq_len"] = max_seq
            if hasattr(self, "ai_local_max_new_tokens_spin"):
                max_new = int(self.ai_local_max_new_tokens_spin.value())
                if max_new > 0:
                    llm_overrides["local_max_new_tokens"] = max_new
        if llm_overrides:
            overrides["llm"] = llm_overrides
        return overrides

    def _on_ai_thread_finished(self) -> None:
        self._ai_thread = None
        self._ai_worker = None

    def _handle_ai_finished(self, payload: object, error: object) -> None:
        self._ai_job_running = False
        self._ai_progress_active = False
        self._ai_progress_stamp = ""
        self._ai_progress_text = ""
        self._mark_ai_log_complete()
        ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_button:
            ok_button.setEnabled(True)
        self._update_ai_buttons()
        if error:
            QtWidgets.QMessageBox.critical(self, "AI backend", str(error))
            self._ai_pending_job = None
            return
        if not isinstance(payload, dict):
            self._ai_pending_job = None
            return
        if payload.get("cancelled"):
            self._ai_pending_job = None
            self._close_ai_log_dialog()
            return
        backend_result = payload.get("backend_result")
        build_result = payload.get("build_result")
        job = self._ai_pending_job
        self._ai_pending_job = None
        if not isinstance(backend_result, BackendResult) or job is None or not isinstance(build_result, dict):
            return
        self.created_round_id = build_result.get("round_id") or job.round_id
        self.created_round_number = job.round_number
        try:
            self.ctx.mark_dirty()
            self.ctx.update_cache_after_round(job.context.corpus_id)
        except Exception:  # noqa: BLE001
            pass
        self._close_ai_log_dialog()
        super().accept()

    def _load_labelset_ids(self) -> List[str]:
        rows = self.ctx.list_label_sets()
        return [str(row["labelset_id"]) for row in rows]

    def _load_corpus_options(self) -> List[sqlite3.Row]:
        return list(self.ctx.list_corpora())

    def _refresh_corpus_options(self) -> None:
        if not hasattr(self, "corpus_combo"):
            return
        self._corpus_options = self._load_corpus_options()
        self.corpus_combo.blockSignals(True)
        self.corpus_combo.clear()
        if not self._corpus_options:
            self.corpus_combo.addItem("Select corpus…", None)
            self.corpus_combo.setEnabled(False)
            self._selected_corpus_id = None
        else:
            self.corpus_combo.setEnabled(True)
            for row in self._corpus_options:
                corpus_id = str(row["corpus_id"])
                display = f"{row['name']} ({corpus_id})"
                self.corpus_combo.addItem(display, corpus_id)
                if self._selected_corpus_id == corpus_id:
                    self.corpus_combo.setCurrentIndex(self.corpus_combo.count() - 1)
        self.corpus_combo.blockSignals(False)
        if self._corpus_options and not self._selected_corpus_id:
            self.corpus_combo.setCurrentIndex(0)
            self._on_corpus_changed(0)
        elif not self._corpus_options:
            self.corpus_combo.setCurrentIndex(0)
            self._on_corpus_changed(0)

    def _on_corpus_changed(self, index: int) -> None:
        del index
        corpus_id = self.corpus_combo.currentData()
        self._selected_corpus_id = corpus_id if isinstance(corpus_id, str) else None
        self._load_metadata_fields(self._selected_corpus_id)

    def _on_import_corpus(self) -> None:
        start_dir = str(self.ctx.project_root or Path.home())
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select corpus file",
            start_dir,
            "Corpus files (*.db *.sqlite *.sqlite3 *.csv *.parquet *.pq);;All files (*)",
        )
        if not path_str:
            return
        try:
            record = self.ctx.import_corpus(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Corpus", f"Failed to import corpus: {exc}")
            return
        self._selected_corpus_id = record.corpus_id
        self._refresh_corpus_options()

    def _load_metadata_fields(self, corpus_id: Optional[str]) -> None:
        if corpus_id:
            try:
                fields = self.ctx.get_corpus_metadata_fields(corpus_id)
            except Exception:  # noqa: BLE001
                fields = []
        else:
            fields = []
        self._metadata_fields = list(fields)
        self._metadata_lookup = {field.key: field for field in self._metadata_fields}
        if hasattr(self, "filter_list"):
            self.filter_list.clear()
        if hasattr(self, "filter_logic_combo"):
            self.filter_logic_combo.setCurrentIndex(0)
        if hasattr(self, "strat_list"):
            self.strat_list.clear()
        self._refresh_filter_field_options()
        self._refresh_strat_field_options()
        self._update_filter_buttons()
        self._update_strat_buttons()

    def _available_metadata_fields(self) -> List[MetadataField]:
        return list(self._metadata_fields)

    def _current_filter_field_keys(self) -> Set[str]:
        keys: Set[str] = set()
        if not hasattr(self, "filter_list"):
            return keys
        for row in range(self.filter_list.count()):
            item = self.filter_list.item(row)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, MetadataFilterCondition):
                keys.add(data.field)
        return keys

    def _selected_stratification_keys(self) -> List[str]:
        keys: List[str] = []
        if not hasattr(self, "strat_list"):
            return keys
        for row in range(self.strat_list.count()):
            item = self.strat_list.item(row)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, str):
                keys.append(data)
        return keys

    def _refresh_filter_field_options(self) -> None:
        if not hasattr(self, "filter_field_combo"):
            return
        available = [
            field
            for field in self._available_metadata_fields()
            if field.key not in self._current_filter_field_keys()
        ]
        self.filter_field_combo.blockSignals(True)
        self.filter_field_combo.clear()
        self.filter_field_combo.addItem("Select metadata field…", None)
        for field in available:
            self.filter_field_combo.addItem(field.label, field.key)
        self.filter_field_combo.blockSignals(False)
        if hasattr(self, "add_filter_btn"):
            self.add_filter_btn.setEnabled(bool(available))

    def _refresh_strat_field_options(self) -> None:
        if not hasattr(self, "strat_field_combo"):
            return
        used = set(self._selected_stratification_keys())
        available = [field for field in self._available_metadata_fields() if field.key not in used]
        self.strat_field_combo.blockSignals(True)
        self.strat_field_combo.clear()
        self.strat_field_combo.addItem("Select metadata field…", None)
        for field in available:
            self.strat_field_combo.addItem(field.label, field.key)
        self.strat_field_combo.blockSignals(False)
        if hasattr(self, "add_strat_btn"):
            self.add_strat_btn.setEnabled(bool(available))

    def _update_filter_buttons(self) -> None:
        if not hasattr(self, "remove_filter_btn"):
            return
        has_selection = hasattr(self, "filter_list") and self.filter_list.currentRow() >= 0
        self.remove_filter_btn.setEnabled(has_selection)
        if hasattr(self, "filter_field_combo") and hasattr(self, "add_filter_btn"):
            self.add_filter_btn.setEnabled(self.filter_field_combo.count() > 1)

    def _update_strat_buttons(self) -> None:
        if not hasattr(self, "remove_strat_btn"):
            return
        has_selection = hasattr(self, "strat_list") and self.strat_list.currentRow() >= 0
        self.remove_strat_btn.setEnabled(has_selection)
        if hasattr(self, "strat_field_combo") and hasattr(self, "add_strat_btn"):
            self.add_strat_btn.setEnabled(self.strat_field_combo.count() > 1)

    def _format_filter_summary(self, condition: MetadataFilterCondition) -> str:
        label = condition.label or condition.field
        parts: List[str] = []
        if condition.min_value is not None and condition.max_value is not None:
            parts.append(f"{condition.min_value} ≤ value ≤ {condition.max_value}")
        elif condition.min_value is not None:
            parts.append(f"≥ {condition.min_value}")
        elif condition.max_value is not None:
            parts.append(f"≤ {condition.max_value}")
        if condition.values:
            if len(condition.values) == 1:
                parts.append(f"= {condition.values[0]}")
            else:
                parts.append(", ".join(condition.values))
        if not parts:
            parts.append("No constraints")
        return f"{label}: {'; '.join(parts)}"

    def _add_filter_condition(self, condition: MetadataFilterCondition) -> None:
        item = QtWidgets.QListWidgetItem(self._format_filter_summary(condition))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, condition)
        self.filter_list.addItem(item)
        self.filter_list.setCurrentItem(item)

    def _on_add_filter(self) -> None:
        data = self.filter_field_combo.currentData() if hasattr(self, "filter_field_combo") else None
        if not isinstance(data, str):
            return
        field = self._metadata_lookup.get(data)
        if not field:
            return
        dialog = MetadataFilterDialog(field, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        condition = dialog.condition()
        if not condition:
            return
        self._add_filter_condition(condition)
        self._refresh_filter_field_options()
        self._update_filter_buttons()

    def _on_edit_filter(self, item: QtWidgets.QListWidgetItem) -> None:
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, MetadataFilterCondition):
            return
        field = self._metadata_lookup.get(data.field)
        if not field:
            return
        dialog = MetadataFilterDialog(field, self, existing=data)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        condition = dialog.condition()
        if not condition:
            return
        item.setData(QtCore.Qt.ItemDataRole.UserRole, condition)
        item.setText(self._format_filter_summary(condition))
        self._refresh_filter_field_options()
        self._update_filter_buttons()

    def _on_remove_filter(self) -> None:
        if not hasattr(self, "filter_list"):
            return
        row = self.filter_list.currentRow()
        if row < 0:
            return
        self.filter_list.takeItem(row)
        self._refresh_filter_field_options()
        self._update_filter_buttons()

    def _on_add_strat_field(self) -> None:
        data = self.strat_field_combo.currentData() if hasattr(self, "strat_field_combo") else None
        if not isinstance(data, str):
            return
        if data in self._selected_stratification_keys():
            return
        field = self._metadata_lookup.get(data)
        if not field:
            return
        item = QtWidgets.QListWidgetItem(field.label)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, data)
        self.strat_list.addItem(item)
        self.strat_list.setCurrentItem(item)
        self._refresh_strat_field_options()
        self._update_strat_buttons()

    def _on_remove_strat_field(self) -> None:
        if not hasattr(self, "strat_list"):
            return
        row = self.strat_list.currentRow()
        if row < 0:
            return
        self.strat_list.takeItem(row)
        self._refresh_strat_field_options()
        self._update_strat_buttons()

    def _load_existing_reviewers(self) -> List[Dict[str, str]]:
        db = self.ctx.require_db()
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT reviewer_id, name, email FROM reviewers ORDER BY name",
            ).fetchall()
        reviewers: List[Dict[str, str]] = []
        for row in rows:
            reviewers.append(
                {
                    "id": str(row["reviewer_id"]),
                    "name": str(row["name"]),
                    "email": str(row["email"] or ""),
                }
            )
        llm_id = RoundBuilder.LLM_REVIEWER_ID
        if not any(str(reviewer.get("id", "")).lower() == llm_id for reviewer in reviewers):
            reviewers.append({"id": llm_id, "name": "LLM", "email": ""})
        return reviewers

    def _handle_llm_reviewer_added(self) -> None:
        checkbox = getattr(self, "random_final_llm_checkbox", None)
        if isinstance(checkbox, QtWidgets.QCheckBox):
            if not checkbox.isChecked():
                checkbox.setChecked(True)
        if not self._llm_prompt_shown:
            QtWidgets.QMessageBox.information(
                self,
                "LLM reviewer",
                (
                    "Final LLM labeling is required when the LLM reviewer is included. "
                    "Provide Azure OpenAI credentials in the Final LLM configuration "
                    "so the assignment can be completed automatically."
                ),
            )
            self._llm_prompt_shown = True
        azure_edit = getattr(self, "random_azure_key_edit", None)
        if isinstance(azure_edit, QtWidgets.QLineEdit):
            azure_edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _generate_reviewer_id(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "reviewer"
        candidate = slug
        counter = 2
        existing_ids = {reviewer["id"] for reviewer in self._available_reviewers}
        used_ids = existing_ids | self._selected_reviewer_ids
        while candidate in used_ids:
            candidate = f"{slug}_{counter}"
            counter += 1
        return candidate

    def _add_reviewer_entry(self, reviewer: Dict[str, str]) -> None:
        reviewer_id = reviewer.get("id")
        if not reviewer_id:
            return
        if reviewer_id in self._selected_reviewer_ids:
            QtWidgets.QMessageBox.information(
                self,
                "Reviewers",
                f"Reviewer {reviewer.get('name', reviewer_id)} is already selected.",
            )
            return
        display = f"{reviewer.get('name', reviewer_id)} ({reviewer_id})"
        item = QtWidgets.QListWidgetItem(display)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, reviewer)
        self.reviewer_list.addItem(item)
        self._selected_reviewer_ids.add(reviewer_id)
        if reviewer_id.lower() == RoundBuilder.LLM_REVIEWER_ID:
            self._handle_llm_reviewer_added()

    def _on_add_existing_reviewer(self) -> None:
        data = self.reviewer_combo.currentData()
        if not isinstance(data, dict):
            return
        self._add_reviewer_entry(dict(data))

    def _on_add_new_reviewer(self) -> None:
        name = self.new_reviewer_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Reviewers", "Enter a reviewer name to add.")
            return
        reviewer_id = self._generate_reviewer_id(name)
        reviewer = {"id": reviewer_id, "name": name, "email": ""}
        self._add_reviewer_entry(reviewer)
        self.new_reviewer_edit.clear()

    def _remove_selected_reviewer(self) -> None:
        row = self.reviewer_list.currentRow()
        if row < 0:
            return
        item = self.reviewer_list.takeItem(row)
        if not item:
            return
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(data, dict):
            reviewer_id = data.get("id")
            if reviewer_id:
                self._selected_reviewer_ids.discard(reviewer_id)

    def _next_round_number(self) -> int:
        pheno_id = self.pheno_row["pheno_id"]
        db = self.ctx.require_db()
        with db.connect() as conn:
            row = conn.execute(
                "SELECT MAX(round_number) FROM rounds WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
        return (row[0] or 0) + 1

    def _build_label_schema(
        self,
        labelset_id: str,
        db: Database,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Dict[str, object]:
        def fetch(connection: sqlite3.Connection) -> Dict[str, object]:
            labelset_row = connection.execute(
                "SELECT * FROM label_sets WHERE labelset_id=?",
                (labelset_id,),
            ).fetchone()
            labels = connection.execute(
                "SELECT * FROM labels WHERE labelset_id=? ORDER BY order_index",
                (labelset_id,),
            ).fetchall()
            options = connection.execute(
                "SELECT * FROM label_options WHERE labelset_id=?",
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
            payload: Dict[str, object] = {"labelset_id": labelset_id, "labels": schema_labels}
            if labelset_row:
                payload["labelset_name"] = labelset_row["labelset_id"]
                payload["created_by"] = labelset_row["created_by"]
                payload["created_at"] = labelset_row["created_at"]
                payload["notes"] = labelset_row["notes"]
            return payload

        if conn is not None:
            return fetch(conn)
        with db.connect() as connection:
            return fetch(connection)

    def accept(self) -> None:  # noqa: D401 - Qt override
        if self._using_ai_backend():
            if self._ai_job_running:
                return
            if not self._start_ai_round():
                return
            return
        if not self._create_round():
            return
        super().accept()

    def _create_round(self) -> bool:
        pheno_id = self.pheno_row["pheno_id"]
        pheno_level = self.pheno_row["level"]
        ctx = self.ctx
        project_id = ctx.current_project_id()
        if not project_id:
            QtWidgets.QMessageBox.critical(self, "Round", "Project metadata is missing; reload the project and try again.")
            return False
        db = ctx.require_db()
        seed = self.seed_spin.value()
        overlap = self.overlap_spin.value()
        independent = self.independent_checkbox.isChecked()
        sampling_metadata: Dict[str, object] = {"independent": bool(independent)}
        reviewers = self._prompt_reviewers()
        if not reviewers:
            return False
        llm_selected = any(
            str(reviewer.get("id", "")).lower() == RoundBuilder.LLM_REVIEWER_ID
            for reviewer in reviewers
        )
        if llm_selected and (
            not hasattr(self, "random_final_llm_checkbox")
            or not self.random_final_llm_checkbox.isChecked()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Final LLM labeling",
                "Enable final LLM labeling when the LLM reviewer is included.",
            )
            return False
        labelset_id = self.labelset_combo.currentText().strip() or f"auto_{pheno_id}"
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
        total_n = self.total_n_spin.value()
        if total_n < overlap:
            QtWidgets.QMessageBox.warning(
                self,
                "Round",
                "Total N must be greater than or equal to the overlap count.",
            )
            return False
        filters = self._collect_filters()
        strat_field_keys = self._selected_stratification_keys()
        assisted_enabled = (
            bool(getattr(self, "assisted_review_checkbox", None))
            and bool(self.assisted_review_checkbox.isChecked())
        )
        assisted_top_n = (
            int(self.assisted_review_spin.value())
            if assisted_enabled and hasattr(self, "assisted_review_spin")
            else 0
        )
        corpus_id = self._selected_corpus_id
        if not corpus_id:
            QtWidgets.QMessageBox.warning(self, "Round", "Select a corpus for this round.")
            return False
        corpus_record = self.ctx.get_corpus(corpus_id)
        try:
            corpus_rows = candidate_documents(
                ctx.get_corpus_db(corpus_id),
                pheno_level,
                filters,
                metadata_fields=self._metadata_fields,
                stratify_keys=strat_field_keys,
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Round", f"Failed to query corpus: {exc}")
            return False
        if not corpus_rows:
            QtWidgets.QMessageBox.warning(self, "Round", "The selected corpus returned no candidate documents.")
            return False
        shortage_warned = False
        if independent:
            reviewed_units = self._load_reviewed_unit_ids(corpus_id)
            sampling_metadata["previously_reviewed_units"] = len(reviewed_units)
            filtered_rows: List[sqlite3.Row | Dict[str, object]] = []
            for row in corpus_rows:
                identifier = self._row_unit_identifier(row)
                if identifier and identifier in reviewed_units:
                    continue
                filtered_rows.append(row)
            excluded_count = len(corpus_rows) - len(filtered_rows)
            sampling_metadata["excluded_prior_units"] = excluded_count
            corpus_rows = filtered_rows
            sampling_metadata["available_unreviewed"] = len(corpus_rows)
            if not corpus_rows:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Round",
                    "No unreviewed units remain for this phenotype. Reduce the independent sampling requirements or add new data.",
                )
                return False
            if len(corpus_rows) < total_n:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Round",
                    (
                        "After excluding previously reviewed units, fewer candidates remain than requested. "
                        "All available unreviewed units will be used."
                    ),
                )
                shortage_warned = True
        if len(corpus_rows) < total_n and not shortage_warned:
            QtWidgets.QMessageBox.warning(
                self,
                "Round",
                "Fewer candidate units were found than the requested total. All available units will be used.",
            )
        strat_aliases = [
            self._metadata_lookup[key].alias
            for key in strat_field_keys
            if key in self._metadata_lookup
        ]
        try:
            assignments = allocate_units(
                corpus_rows,
                reviewers,
                overlap,
                seed,
                total_n=total_n,
                strat_keys=strat_aliases or None,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Round", str(exc))
            return False
        unique_units = {
            unit["unit_id"]
            for assignment in assignments.values()
            for unit in assignment.units
        }
        if len(unique_units) < total_n:
            QtWidgets.QMessageBox.warning(
                self,
                "Round",
                (
                    f"Only {len(unique_units)} unique units could be allocated out of the requested {total_n}. "
                    "Reviewers will receive as even a distribution as possible."
                ),
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
        round_dir = ensure_dir(ctx.resolve_round_dir(pheno_id, round_number))
        ctx.register_manifest(round_dir / "manifest.csv", assignments)
        label_schema: Optional[Dict[str, object]] = None
        with db.transaction() as conn:
            if default_labels:
                labelset = models.LabelSet(
                    labelset_id=labelset_id,
                    project_id=project_id,
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
                            labelset_id=labelset_id,
                            label_id=label_record.label_id,
                            value=option["value"],
                            display=option["display"],
                            order_index=idx,
                            weight=None,
                        )
                        option_record.save(conn)
            round_record.save(conn)
            label_schema = self._build_label_schema(labelset_id, db, conn)
            config_payload: Dict[str, object] = {
                "pheno_id": pheno_id,
                "labelset_id": labelset_id,
                "corpus_id": corpus_id,
                "round_number": round_number,
                "round_id": round_id,
                "rng_seed": seed,
                "overlap_n": overlap,
                "total_n": total_n,
                "status": self.status_combo.currentText(),
                "reviewers": reviewers,
            }
            backend_cfg: Dict[str, object] = {}
            final_llm_enabled = False
            if hasattr(self, "random_final_llm_checkbox"):
                final_llm_enabled = bool(self.random_final_llm_checkbox.isChecked())
                config_payload["final_llm_labeling"] = final_llm_enabled
            if final_llm_enabled or assisted_enabled:
                include_reasoning = False
                if isinstance(getattr(self, "random_include_reasoning_checkbox", None), QtWidgets.QCheckBox):
                    include_reasoning = bool(self.random_include_reasoning_checkbox.isChecked())
                config_payload["final_llm_include_reasoning"] = include_reasoning
                backend_choice = self._current_random_backend()
                if backend_choice:
                    backend_cfg["backend"] = backend_choice
                embed_path = (
                    self.random_embedding_path_edit.text().strip()
                    if hasattr(self, "random_embedding_path_edit")
                    else ""
                )
                rerank_path = (
                    self.random_reranker_path_edit.text().strip()
                    if hasattr(self, "random_reranker_path_edit")
                    else ""
                )
                if embed_path:
                    backend_cfg["embedding_model_dir"] = embed_path
                if rerank_path:
                    backend_cfg["reranker_model_dir"] = rerank_path
                if backend_choice == "azure":
                    azure_version = (
                        self.random_azure_version_edit.text().strip()
                        if hasattr(self, "random_azure_version_edit")
                        else ""
                    )
                    azure_endpoint = (
                        self.random_azure_endpoint_edit.text().strip()
                        if hasattr(self, "random_azure_endpoint_edit")
                        else ""
                    )
                    if azure_version:
                        backend_cfg["azure_api_version"] = azure_version
                    if azure_endpoint:
                        backend_cfg["azure_endpoint"] = azure_endpoint
                else:
                    model_dir = (
                        self.random_local_model_path_edit.text().strip()
                        if hasattr(self, "random_local_model_path_edit")
                        else ""
                    )
                    if model_dir:
                        backend_cfg["local_model_dir"] = model_dir
                    if hasattr(self, "random_local_max_seq_spin"):
                        max_seq = int(self.random_local_max_seq_spin.value())
                        if max_seq > 0:
                            backend_cfg["local_max_seq_len"] = max_seq
                    if hasattr(self, "random_local_max_new_tokens_spin"):
                        max_new = int(self.random_local_max_new_tokens_spin.value())
                        if max_new > 0:
                            backend_cfg["local_max_new_tokens"] = max_new
                if backend_cfg:
                    config_payload.setdefault("ai_backend", {}).update(backend_cfg)
            if assisted_enabled:
                config_payload["assisted_review"] = {
                    "enabled": True,
                    "top_snippets": assisted_top_n,
                }
                if corpus_record:
                    config_payload["corpus_name"] = corpus_record["name"]
                    config_payload["corpus_path"] = corpus_record["relative_path"]
                if sampling_metadata:
                    config_payload["sampling"] = sampling_metadata
                if filters.metadata_filters:
                    metadata_payload: Dict[str, object] = {
                        "conditions": [
                            condition.to_payload() for condition in filters.metadata_filters
                        ]
                    }
                    metadata_payload["logic"] = "any" if filters.match_any else "all"
                    config_payload["filters"] = {"metadata": metadata_payload}
                if strat_field_keys:
                    config_payload["stratification"] = {
                        "fields": strat_field_keys,
                        "labels": [
                            self._metadata_lookup[key].label
                            for key in strat_field_keys
                            if key in self._metadata_lookup
                        ],
                        "aliases": strat_aliases,
                    }
                if label_schema:
                    config_payload["label_schema"] = label_schema
                config = models.RoundConfig(
                    round_id=round_id,
                    config_json=json.dumps(config_payload, indent=2),
                )
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
        ctx.register_text_file(round_dir / "round_config.json", json.dumps(config_payload, indent=2))
        if label_schema is None:
            label_schema = self._build_label_schema(labelset_id, db)
        for reviewer in reviewers:
            assignment_dir = ensure_dir(round_dir / "assignments" / reviewer["id"])
            db_path = assignment_dir / "assignment.db"
            assignment_db = ctx.prepare_assignment_db(db_path)
            populate_assignment_db(assignment_db, reviewer["id"], assignments[reviewer["id"]].units)
            schema_path = assignment_dir / "label_schema.json"
            ctx.register_text_file(schema_path, json.dumps(label_schema, indent=2))
        assisted_result: Dict[str, object] = {}
        if assisted_enabled and assisted_top_n > 0:
            try:
                assisted_result = self._generate_random_assisted_review(
                    config_payload=config_payload,
                    assignments=assignments,
                    round_dir=round_dir,
                    top_snippets=assisted_top_n,
                )
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(
                    self,
                    "Assisted chart review",
                    f"Failed to generate assisted review snippets: {exc}",
                )
            else:
                if assisted_result:
                    updated_config = json.dumps(config_payload, indent=2)
                    ctx.register_text_file(round_dir / "round_config.json", updated_config)
                    with db.transaction() as conn:
                        conn.execute(
                            "UPDATE round_configs SET config_json=? WHERE round_id=?",
                            (updated_config, round_id),
                        )
        final_llm_outputs: Dict[str, str] = {}
        if hasattr(self, "random_final_llm_checkbox") and self.random_final_llm_checkbox.isChecked():
            try:
                final_llm_outputs = self._run_random_final_llm_labeling(
                    round_id=round_id,
                    round_dir=round_dir,
                    config_payload=config_payload,
                    assignments=assignments,
                )
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    self,
                    "Final LLM labeling",
                    f"Failed to run final LLM labeling: {exc}",
                )
                return False
            if final_llm_outputs:
                config_payload.setdefault("final_llm_outputs", {}).update(final_llm_outputs)
                updated_config = json.dumps(config_payload, indent=2)
                ctx.register_text_file(round_dir / "round_config.json", updated_config)
                with db.transaction() as conn:
                    conn.execute(
                        "UPDATE round_configs SET config_json=? WHERE round_id=?",
                        (updated_config, round_id),
                    )
        self.created_round_id = round_id
        self.created_round_number = round_number
        ctx.mark_dirty()
        ctx.update_cache_after_round(corpus_id)
        return True

    def _run_random_final_llm_labeling(
        self,
        *,
        round_id: str,
        round_dir: Path,
        config_payload: Dict[str, object],
        assignments: Dict[str, ReviewerAssignment],
    ) -> Dict[str, str]:
        project_root = getattr(self.ctx, "project_root", None)
        if not project_root:
            raise RuntimeError("Project root is not available")
        builder = RoundBuilder(project_root)
        db = self.ctx.require_db()
        pheno_id = str(config_payload.get("pheno_id") or self.pheno_row["pheno_id"])
        labelset_id = str(config_payload.get("labelset_id") or "")
        if not labelset_id:
            raise RuntimeError("Label set ID is required for final LLM labeling")
        with db.connect() as conn:
            conn.row_factory = sqlite3.Row
            pheno_row = conn.execute(
                "SELECT * FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
            if not pheno_row:
                raise RuntimeError(f"Phenotype {pheno_id} not found")
            labelset = fetch_labelset(conn, labelset_id)
        label_schema_payload = builder._build_label_schema_payload(labelset)

        reviewer_assignments = build_round_assignment_units(assignments)
        if not reviewer_assignments:
            raise RuntimeError("No assignments available for final LLM labeling")

        config_payload = dict(config_payload)
        config_payload["final_llm_labeling"] = True

        env_overrides = self._collect_random_final_llm_environment()
        if env_overrides is None:
            raise RuntimeError("Final LLM labeling configuration is incomplete")

        self._open_ai_log_dialog()
        self._ai_progress_active = False
        self._ai_progress_stamp = ""
        self._ai_progress_text = ""
        self._ai_progress_block_number = None
        self._append_ai_log(f"Running final LLM labeling for round {round_id}…")
        try:
            outputs = builder.run_final_llm_labeling(
                pheno_row=pheno_row,
                labelset=labelset,
                round_dir=round_dir,
                reviewer_assignments=reviewer_assignments,
                config=config_payload,
                config_base=round_dir,
                log_callback=self._append_ai_log,
                env_overrides=env_overrides,
                auto_submit_llm=False,
            )
            if outputs:
                self._apply_llm_reviewer_submission_with_context(
                    builder=builder,
                    config_payload=config_payload,
                    round_dir=round_dir,
                    reviewer_assignments=reviewer_assignments,
                    label_schema=label_schema_payload,
                    final_llm_outputs=outputs,
                )
        except Exception as exc:  # noqa: BLE001
            self._append_ai_log(f"Final LLM labeling failed: {exc}")
            raise
        else:
            if outputs:
                self._append_ai_log("Final LLM labeling complete.")
            else:
                self._append_ai_log("Final LLM labeling finished with no outputs.")
            return outputs
        finally:
            self._mark_ai_log_complete()

    def _generate_random_assisted_review(
        self,
        *,
        config_payload: Dict[str, object],
        assignments: Mapping[str, ReviewerAssignment],
        round_dir: Path,
        top_snippets: int,
    ) -> Dict[str, object]:
        if top_snippets <= 0:
            return {}
        project_root = getattr(self.ctx, "project_root", None)
        if not project_root:
            raise RuntimeError("Project root is not available")
        reviewer_assignments = build_round_assignment_units(assignments)
        if not reviewer_assignments:
            return {}
        builder = RoundBuilder(project_root)
        db = self.ctx.require_db()
        pheno_id = str(config_payload.get("pheno_id") or self.pheno_row["pheno_id"])
        labelset_id = str(config_payload.get("labelset_id") or "")
        if not labelset_id:
            raise RuntimeError("Label set ID is required for assisted review")
        with db.connect() as conn:
            conn.row_factory = sqlite3.Row
            pheno_row = conn.execute(
                "SELECT * FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
            if not pheno_row:
                raise RuntimeError(f"Phenotype {pheno_id} not found")
            labelset = fetch_labelset(conn, labelset_id)
        config_copy = dict(config_payload)
        assist_cfg = config_copy.setdefault("assisted_review", {})
        assist_cfg["enabled"] = True
        assist_cfg["top_snippets"] = int(top_snippets)

        env_overrides = RoundBuilderDialog._collect_random_llm_environment(
            self,
            context_label="Assisted chart review",
            require_checkbox=False,
        )
        if env_overrides is None:
            raise RuntimeError("Assisted chart review configuration is incomplete")

        previous_env: Dict[str, Optional[str]] = {}
        for key, value in env_overrides.items():
            previous_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            assisted_data = builder._generate_assisted_review_snippets(
                pheno_row=pheno_row,
                labelset=labelset,
                round_dir=round_dir,
                reviewer_assignments=reviewer_assignments,
                config=config_copy,
                config_base=round_dir,
                top_n=int(top_snippets),
            )
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        if not assisted_data:
            return {}

        assist_dir = ensure_dir(round_dir / "reports" / "assisted_review")
        assist_path = assist_dir / "snippets.json"
        serialized = builder._json_dumps(assisted_data)
        self.ctx.register_text_file(assist_path, serialized)

        try:
            relative_path = assist_path.relative_to(round_dir)
        except ValueError:
            try:
                relative_path = assist_path.resolve().relative_to(round_dir.resolve())
            except ValueError:
                relative_path = assist_path

        assist_cfg = config_payload.setdefault("assisted_review", {})
        assist_cfg["enabled"] = True
        assist_cfg["top_snippets"] = int(top_snippets)
        assist_cfg["generated_at"] = assisted_data.get("generated_at")
        assist_cfg["snippets_json"] = str(relative_path)

        return {"snippets_json": str(relative_path)}

    def _apply_llm_reviewer_submission_with_context(
        self,
        *,
        builder: RoundBuilder,
        config_payload: Mapping[str, object],
        round_dir: Path,
        reviewer_assignments: Mapping[str, Sequence[RoundAssignmentUnit]],
        label_schema: Mapping[str, Any],
        final_llm_outputs: Mapping[str, str],
    ) -> None:
        llm_ids = builder._llm_reviewer_ids(config_payload)
        if not llm_ids:
            return
        predictions = builder._extract_llm_predictions(final_llm_outputs, round_dir)
        if not predictions:
            raise RuntimeError("Final LLM labeling outputs are missing LLM predictions")
        labels_obj = label_schema.get("labels") if isinstance(label_schema, Mapping) else None
        if not isinstance(labels_obj, Sequence):
            raise RuntimeError("Label schema is required to populate LLM assignments")
        label_lookup = {
            str(label.get("label_id")): label
            for label in labels_obj
            if isinstance(label, Mapping) and label.get("label_id")
        }
        if not label_lookup:
            raise RuntimeError("Label schema did not include label identifiers")
        db = self.ctx.require_db()
        round_id = str(config_payload.get("round_id") or "")
        timestamp = datetime.utcnow().isoformat()
        for reviewer_id in llm_ids:
            assignments = reviewer_assignments.get(reviewer_id)
            if not assignments:
                continue
            unit_ids = [
                str(unit.unit_id)
                for unit in assignments
                if getattr(unit, "unit_id", None)
            ]
            if not unit_ids:
                continue
            assignment_dir = round_dir / "assignments" / reviewer_id
            assignment_db = self.ctx.get_assignment_db(assignment_dir / "assignment.db")
            if assignment_db is None:
                raise RuntimeError(f"Assignment database not found for reviewer {reviewer_id}")
            annotation_records: list[models.Annotation] = []
            for unit_id in unit_ids:
                label_values = predictions.get(unit_id, {})
                for label_id, label_info in label_lookup.items():
                    raw_value = label_values.get(label_id)
                    value, value_num, value_date, na_flag = builder._normalize_annotation_value(
                        label_info,
                        raw_value,
                    )
                    annotation_records.append(
                        models.Annotation(
                            unit_id=unit_id,
                            label_id=label_id,
                            value=value,
                            value_num=value_num,
                            value_date=value_date,
                            na=na_flag,
                            notes=None,
                        )
                    )
            with assignment_db.transaction() as conn:
                conn.executemany(
                    "DELETE FROM annotations WHERE unit_id=?",
                    [(unit_id,) for unit_id in unit_ids],
                )
                if annotation_records:
                    models.Annotation.insert_many(conn, annotation_records)
                for unit_id in unit_ids:
                    conn.execute(
                        "UPDATE units SET complete=1, completed_at=? WHERE unit_id=?",
                        (timestamp, unit_id),
                    )
            receipt = {
                "unit_count": len(unit_ids),
                "completed": len(unit_ids),
                "submitted_at": timestamp,
            }
            self.ctx.register_text_file(
                assignment_dir / "submitted.json",
                json.dumps(receipt, indent=2),
            )
            if round_id:
                with db.transaction() as conn:
                    conn.execute(
                        "UPDATE assignments SET status='submitted' WHERE round_id=? AND reviewer_id=?",
                        (round_id, reviewer_id),
                    )


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
        project_item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {"type": "project", "project": dict(project)},
        )
        self.addTopLevelItem(project_item)
        project_item.setExpanded(True)
        corpora_section = QtWidgets.QTreeWidgetItem(["Corpora"])
        corpora_section.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "corpora_section"})
        project_item.addChild(corpora_section)
        for corpus in self.ctx.list_corpora():
            corpus_item = self._build_corpus_item(corpus)
            corpora_section.addChild(corpus_item)
        corpora_section.setExpanded(True)

        phenotypes_section = QtWidgets.QTreeWidgetItem(["Phenotypes"])
        phenotypes_section.setData(
            0, QtCore.Qt.ItemDataRole.UserRole, {"type": "phenotypes_section"}
        )
        project_item.addChild(phenotypes_section)
        for pheno in self.ctx.list_phenotypes():
            pheno_item = self._build_phenotype_item(pheno)
            phenotypes_section.addChild(pheno_item)
            pheno_item.setExpanded(True)
        phenotypes_section.setExpanded(True)

        self.expandItem(project_item)
        if corpora_section.childCount():
            self.setCurrentItem(corpora_section.child(0))
        elif phenotypes_section.childCount():
            self.setCurrentItem(phenotypes_section.child(0))
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
        iaa_item = QtWidgets.QTreeWidgetItem(["IAA"])
        iaa_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "iaa", "pheno": dict(pheno)})
        pheno_item.addChild(iaa_item)
        return pheno_item

    def _build_corpus_item(self, corpus: sqlite3.Row) -> QtWidgets.QTreeWidgetItem:
        corpus_dict = dict(corpus)
        name = corpus_dict.get("name")
        corpus_id = corpus_dict.get("corpus_id")
        label = f"{name} ({corpus_id})" if name else str(corpus_id or "Corpus")
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            {"type": "corpus", "corpus": corpus_dict},
        )
        return item

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
            label_action = menu.addAction("Add label set…")
            label_action.triggered.connect(lambda: self._add_labelset(item))
            corpus_action = menu.addAction("Create corpus…")
            corpus_action.triggered.connect(lambda: self._create_corpus(item))
        elif node_type == "corpus":
            delete_action = menu.addAction("Delete corpus…")
            delete_action.triggered.connect(lambda: self._delete_corpus(item))
        elif node_type == "phenotype":
            view_labelsets_action = menu.addAction("View label sets…")
            view_labelsets_action.triggered.connect(lambda: self._view_labelsets(item))
            action = menu.addAction("Add round…")
            action.triggered.connect(lambda: self._add_round(item))
            inference_action = menu.addAction("Inference…")
            inference_action.triggered.connect(lambda: self._open_inference(item))
            delete_action = menu.addAction("Delete phenotype…")
            delete_action.triggered.connect(lambda: self._delete_phenotype(item))
        elif node_type == "round":
            round_info = data.get("round") or {}
            current_status = str(round_info.get("status", ""))
            status_menu = menu.addMenu("Set status")
            for status in ["draft", "active", "closed", "adjudicating", "finalized"]:
                action = status_menu.addAction(status.capitalize())
                action.setCheckable(True)
                action.setChecked(status == current_status)
                action.triggered.connect(lambda _, s=status: self._change_round_status(item, s))
            menu.addSeparator()
            delete_action = menu.addAction("Delete round…")
            delete_action.triggered.connect(lambda: self._delete_round(item))
        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(point))

    def _add_phenotype(self, item: QtWidgets.QTreeWidgetItem) -> None:
        del item
        dialog = PhenotypeDialog(self.ctx, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            record = self.ctx.create_phenotype(
                name=str(values.get("name", "")),
                level=str(values.get("level", "single_doc")),
                description=str(values.get("description", "")),
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Phenotype", f"Failed to create phenotype: {exc}")
            return
        QtCore.QTimer.singleShot(0, lambda: self._select_phenotype(record.pheno_id))

    def _create_corpus(self, item: QtWidgets.QTreeWidgetItem) -> None:
        del item
        start_dir = str(self.ctx.project_root or Path.home())
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select corpus file",
            start_dir,
            "Corpus files (*.db *.sqlite *.sqlite3 *.csv *.parquet *.pq);;All files (*)",
        )
        if not path_str:
            return
        try:
            record = self.ctx.import_corpus(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Corpus", f"Failed to import corpus: {exc}")
            return
        QtCore.QTimer.singleShot(0, lambda: self._select_corpus(record.corpus_id))

    def _add_labelset(self, item: QtWidgets.QTreeWidgetItem) -> None:
        del item
        dialog = LabelSetWizardDialog(self.ctx, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        try:
            self.ctx.create_labelset(
                labelset_id=str(values.get("labelset_id", "")),
                created_by=str(values.get("created_by", "admin")),
                notes=str(values.get("notes", "")),
                labels=[dict(label) for label in values.get("labels", [])],
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Label set", f"Failed to create label set: {exc}")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Label set",
            f"Label set '{values.get('labelset_id')}' created.",
        )

    def _view_labelsets(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
        pheno = data.get("pheno")
        if isinstance(pheno, dict):
            pheno_data = pheno
        elif pheno is not None:
            try:
                pheno_data = dict(pheno)
            except Exception:  # noqa: BLE001
                pheno_data = {}
        else:
            pheno_data = {}
        pheno_id = str(pheno_data.get("pheno_id") or "")
        if not pheno_id:
            return
        dialog = PhenotypeLabelSetsDialog(self.ctx, pheno_data, self)
        dialog.exec()

    def _open_inference(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
        pheno = data.get("pheno")
        if not isinstance(pheno, dict):
            return
        dialog = PromptInferenceDialog(self.ctx, pheno, self)
        dialog.exec()

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

    def _delete_corpus(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
        corpus = data.get("corpus")
        if not isinstance(corpus, dict):
            return
        corpus_id = str(corpus.get("corpus_id") or "")
        if not corpus_id:
            return
        display_name = str(corpus.get("name") or corpus_id)
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete corpus",
            (
                f"Delete corpus '{display_name}'?\n\n"
                "This will remove the corpus record from the project. Files will also be deleted."
            ),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.ctx.delete_corpus(corpus_id)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Delete corpus", f"Failed to delete corpus: {exc}")

    def _delete_phenotype(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
        pheno = data.get("pheno")
        if not isinstance(pheno, dict):
            return
        pheno_id = str(pheno.get("pheno_id") or "")
        if not pheno_id:
            return
        display_name = str(pheno.get("name") or pheno_id)
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete phenotype",
            (
                f"Delete phenotype '{display_name}'?\n\n"
                "All rounds and associated files for this phenotype will be permanently removed."
            ),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.ctx.delete_phenotype(pheno_id)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Delete phenotype", f"Failed to delete phenotype: {exc}")
            return
        QtCore.QTimer.singleShot(0, self.refresh)

    def _delete_round(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
        round_info = data.get("round")
        if not isinstance(round_info, dict):
            return
        round_id = str(round_info.get("round_id") or "")
        if not round_id:
            return
        pheno_id = str(round_info.get("pheno_id") or "")
        round_label = item.text(0) or round_id
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete round",
            (
                f"Delete {round_label}?\n\n"
                "Assignments, configuration, and round files will be permanently removed."
            ),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.ctx.delete_round(round_id)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Delete round", f"Failed to delete round: {exc}")
            return
        if pheno_id:
            QtCore.QTimer.singleShot(0, lambda: self._select_phenotype(pheno_id))

    def _change_round_status(self, item: QtWidgets.QTreeWidgetItem, status: str) -> None:
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        round_info = data.get("round")
        if not isinstance(round_info, dict):
            return
        round_id = round_info.get("round_id")
        pheno_id = round_info.get("pheno_id")
        if not round_id or not pheno_id:
            return
        try:
            self.ctx.update_round_status(round_id, status)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Round status", f"Failed to update status: {exc}")
            return
        QtCore.QTimer.singleShot(0, lambda: self._select_round(pheno_id, round_id))

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

    def _select_corpus(self, corpus_id: str) -> None:
        project_item = self.topLevelItem(0)
        if not project_item:
            return
        for item in self._iter_items(project_item):
            data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if (
                isinstance(data, dict)
                and data.get("type") == "corpus"
                and data.get("corpus", {}).get("corpus_id") == corpus_id
            ):
                self.setCurrentItem(item)
                parent = item.parent()
                while parent:
                    self.expandItem(parent)
                    parent = parent.parent()
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
        self.storage_label = QtWidgets.QLabel()
        self.description_label.setFixedHeight(120)
        layout.addRow("Name", self.name_label)
        layout.addRow("Level", self.level_label)
        layout.addRow("Storage", self.storage_label)
        layout.addRow("Description", self.description_label)

    def set_phenotype(self, pheno: Optional[Dict[str, object]]) -> None:
        if not pheno:
            self.name_label.clear()
            self.level_label.clear()
            self.description_label.clear()
            self.storage_label.clear()
            return
        self.name_label.setText(str(pheno.get("name", "")))
        self.level_label.setText(str(pheno.get("level", "")))
        self.storage_label.setText(str(pheno.get("storage_path", "")))
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
        labelset_text = str(round_row.get("labelset_id", ""))
        if config:
            schema = config.get("label_schema")
            if isinstance(schema, dict):
                schema_name = schema.get("labelset_name")
                if schema_name and schema_name != labelset_text:
                    labelset_text = f"{labelset_text} — {schema_name}"
        self.labelset_label.setText(labelset_text)
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
        corpus_id = config.get("corpus_id")
        corpus_name = config.get("corpus_name")
        if corpus_id or corpus_name:
            if corpus_id and corpus_name:
                corpus_display = f"{corpus_name} ({corpus_id})"
            else:
                corpus_display = corpus_id or corpus_name
            setup_items.append(f"Corpus: {corpus_display}")
            if config.get("corpus_path"):
                setup_items.append(f"Corpus path: {config['corpus_path']}")
        if config.get("total_n"):
            setup_items.append(f"Total units: {config['total_n']}")
        setup_items.append(f"Overlap units: {config.get('overlap_n', 0)}")
        setup_items.append(f"RNG seed: {config.get('rng_seed', 0)}")
        reviewers = config.get("reviewers") or []
        if reviewers:
            reviewer_names = [str(reviewer.get("name") or reviewer.get("id")) for reviewer in reviewers]
            setup_items.append(f"Reviewers: {', '.join(reviewer_names)}")
        sections.append(self._format_section("Round setup", setup_items))

        sampling_config = config.get("sampling") or {}
        if isinstance(sampling_config, dict):
            sampling_items: List[str] = []
            if "independent" in sampling_config:
                independent = sampling_config.get("independent")
                sampling_items.append(
                    "Independent sampling: " + ("Yes" if independent else "No")
                )
            excluded = sampling_config.get("excluded_prior_units")
            if excluded is not None:
                sampling_items.append(f"Excluded previously reviewed units: {excluded}")
            available = sampling_config.get("available_unreviewed")
            if available is not None:
                sampling_items.append(f"Available unreviewed units: {available}")
            previous_total = sampling_config.get("previously_reviewed_units")
            if previous_total is not None:
                sampling_items.append(f"Previously reviewed units considered: {previous_total}")
            if sampling_items:
                sections.append(self._format_section("Sampling", sampling_items))

        filters = config.get("filters") or {}
        filter_items: List[str] = []
        metadata_filters = filters.get("metadata")
        metadata_items: List[str] = []
        logic_value = "all"
        raw_conditions: Sequence[Mapping[str, object]] | None
        if isinstance(metadata_filters, dict):
            logic_value = str(metadata_filters.get("logic") or "all").lower()
            raw = metadata_filters.get("conditions")
            raw_conditions = raw if isinstance(raw, (list, tuple)) else None
        elif isinstance(metadata_filters, list):
            raw_conditions = metadata_filters
        else:
            raw_conditions = None
        if raw_conditions:
            for entry in raw_conditions:
                try:
                    condition = MetadataFilterCondition.from_payload(entry)
                except Exception:  # noqa: BLE001
                    continue
                metadata_items.append(self._describe_metadata_filter(condition))
            if metadata_items:
                if logic_value == "any":
                    metadata_items.insert(0, "Match any condition (OR logic)")
                else:
                    metadata_items.insert(0, "Match all conditions (AND logic)")
                filter_items.extend(metadata_items)
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
        labels = stratification.get("labels")
        fields = stratification.get("fields")
        if isinstance(labels, list) and labels:
            strat_items.append(f"Stratify by: {', '.join(str(label) for label in labels)}")
        elif isinstance(fields, list) and fields:
            strat_items.append(f"Stratify by: {', '.join(str(field) for field in fields)}")
        else:
            keys = stratification.get("keys")
            if keys:
                if isinstance(keys, list):
                    strat_items.append(f"Stratify by: {', '.join(keys)}")
                else:
                    strat_items.append(f"Stratify by: {keys}")
        if strat_items:
            sections.append(self._format_section("Stratification", strat_items))

        label_schema = config.get("label_schema") or {}
        if isinstance(label_schema, dict):
            label_items: List[str] = []
            labelset_name = str(label_schema.get("labelset_name") or "")
            labelset_id = str(label_schema.get("labelset_id") or "")
            if labelset_name:
                label_items.append(f"Name: {labelset_name}")
            elif labelset_id:
                label_items.append(f"Name: {labelset_id}")
            if labelset_id and labelset_id != labelset_name:
                label_items.append(f"Identifier: {labelset_id}")
            notes = str(label_schema.get("notes") or "").strip()
            if notes:
                label_items.append(f"Notes: {notes}")
            created_by = label_schema.get("created_by")
            created_at = label_schema.get("created_at")
            if created_by and created_at:
                label_items.append(f"Created by {created_by} on {created_at}")
            elif created_by:
                label_items.append(f"Created by {created_by}")
            for label in label_schema.get("labels", []):
                if not isinstance(label, dict):
                    continue
                name = str(label.get("name") or label.get("label_id") or "Label")
                label_type = str(label.get("type", "unknown"))
                required = "required" if label.get("required") else "optional"
                entry = f"{name} — {label_type} ({required})"
                if label.get("gating_expr"):
                    entry += f" [Gate: {label['gating_expr']}]"
                options = label.get("options") or []
                if options:
                    option_names = ", ".join(
                        str(option.get("display") or option.get("value")) for option in options if isinstance(option, dict)
                    )
                    if option_names:
                        entry += f" — Options: {option_names}"
                rules = str(label.get("rules") or "").strip()
                if rules:
                    condensed = " ".join(line.strip() for line in rules.splitlines() if line.strip())
                    if condensed:
                        entry += f"\n    Rules: {condensed}"
                label_items.append(entry)
            if label_items:
                sections.append(self._format_section("Label set", label_items))

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

    @staticmethod
    def _describe_metadata_filter(condition: MetadataFilterCondition) -> str:
        label = condition.label or condition.field
        parts: List[str] = []
        if condition.min_value is not None and condition.max_value is not None:
            parts.append(f"{condition.min_value} ≤ value ≤ {condition.max_value}")
        elif condition.min_value is not None:
            parts.append(f"≥ {condition.min_value}")
        elif condition.max_value is not None:
            parts.append(f"≤ {condition.max_value}")
        if condition.values:
            if len(condition.values) == 1:
                parts.append(f"= {condition.values[0]}")
            else:
                parts.append(", ".join(condition.values))
        if not parts:
            parts.append("No constraints")
        return f"{label}: {'; '.join(parts)}"


class CorpusWidget(QtWidgets.QWidget):
    def __init__(self, ctx: ProjectContext, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.current_corpus_id: Optional[str] = None
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Corpus:"))
        self.corpus_combo = QtWidgets.QComboBox()
        self.corpus_combo.currentIndexChanged.connect(self._on_corpus_selected)
        controls.addWidget(self.corpus_combo, 1)
        import_btn = QtWidgets.QPushButton("Import corpus…")
        import_btn.clicked.connect(self._import_corpus)
        controls.addWidget(import_btn)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_corpora)
        controls.addWidget(refresh_btn)
        layout.addLayout(controls)

        self.summary_label = QtWidgets.QLabel("Select a corpus to view its contents.")
        layout.addWidget(self.summary_label)
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self.ctx.project_changed.connect(self._refresh_corpora)
        self._refresh_corpora()

    def set_phenotype(self, pheno: Optional[Dict[str, object]]) -> None:
        del pheno
        self.set_corpus(None)

    def set_corpus(self, corpus: Optional[Dict[str, object]]) -> None:
        if corpus and corpus.get("corpus_id"):
            self.current_corpus_id = str(corpus["corpus_id"])
        else:
            self.current_corpus_id = None
        self._refresh_corpora()

    def _refresh_corpora(self) -> None:
        corpora = self.ctx.list_corpora()
        current_id = self.current_corpus_id
        self.corpus_combo.blockSignals(True)
        self.corpus_combo.clear()
        if not corpora:
            self.corpus_combo.addItem("No corpora available", None)
            self.corpus_combo.setEnabled(False)
            self.summary_label.setText("Import a corpus to view its contents.")
            self.table.setRowCount(0)
            self.current_corpus_id = None
        else:
            self.corpus_combo.setEnabled(True)
            for corpus in corpora:
                corpus_id = str(corpus["corpus_id"])
                display = f"{corpus['name']} ({corpus_id})"
                self.corpus_combo.addItem(display, corpus_id)
                if current_id and corpus_id == current_id:
                    self.corpus_combo.setCurrentIndex(self.corpus_combo.count() - 1)
        self.corpus_combo.blockSignals(False)
        if corpora:
            if current_id and any(str(row["corpus_id"]) == current_id for row in corpora):
                self._load_corpus(current_id)
            else:
                self.corpus_combo.setCurrentIndex(0)
                self._on_corpus_selected(0)

    def _import_corpus(self) -> None:
        start_dir = str(self.ctx.project_root or Path.home())
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select corpus file",
            start_dir,
            "Corpus files (*.db *.sqlite *.sqlite3 *.csv *.parquet *.pq);;All files (*)",
        )
        if not path_str:
            return
        try:
            record = self.ctx.import_corpus(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Corpus", f"Failed to import corpus: {exc}")
            return
        self.current_corpus_id = record.corpus_id
        self._refresh_corpora()

    def _on_corpus_selected(self, index: int) -> None:
        del index
        corpus_id = self.corpus_combo.currentData()
        if not isinstance(corpus_id, str):
            self.current_corpus_id = None
            self.summary_label.setText("Select a corpus to view its contents.")
            self.table.setRowCount(0)
            return
        self.current_corpus_id = corpus_id
        self._load_corpus(corpus_id)

    def _load_corpus(self, corpus_id: str) -> None:
        try:
            db = self.ctx.get_corpus_db(corpus_id)
        except Exception as exc:  # noqa: BLE001
            self.summary_label.setText(f"Corpus unavailable: {exc}")
            self.table.setRowCount(0)
            return
        with db.connect() as conn:
            patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            columns = [row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()]
            if columns:
                truncated_columns = {"text"}
                select_parts = []
                for column in columns:
                    identifier = f'"{column}"'
                    if column in truncated_columns:
                        select_parts.append(
                            (
                                "CASE WHEN length({identifier}) > 200 "
                                "THEN substr({identifier}, 1, 200) || '…' "
                                "ELSE {identifier} END AS {identifier}"
                            ).format(identifier=identifier)
                        )
                    else:
                        select_parts.append(identifier)
                order_column: Optional[str]
                if "date_note" in columns:
                    order_column = '"date_note" DESC'
                elif "doc_id" in columns:
                    order_column = '"doc_id" DESC'
                else:
                    order_column = "rowid DESC"
                query = "SELECT {columns} FROM documents ORDER BY {order} LIMIT 50".format(
                    columns=", ".join(select_parts),
                    order=order_column,
                )
                rows = conn.execute(query).fetchall()
            else:
                rows = []

        metadata_column_order: List[str] = []
        row_metadata: List[Dict[str, object]] = []
        if rows and "metadata_json" in columns:
            seen_metadata_keys: Set[str] = set()
            for row in rows:
                parsed_metadata: Dict[str, object] = {}
                raw_metadata = row["metadata_json"]
                if isinstance(raw_metadata, str) and raw_metadata.strip():
                    try:
                        metadata_payload = json.loads(raw_metadata)
                    except json.JSONDecodeError:
                        metadata_payload = None
                    if isinstance(metadata_payload, Mapping):
                        for key, value in metadata_payload.items():
                            if isinstance(value, (dict, list)):
                                display_value = json.dumps(value, ensure_ascii=False)
                            else:
                                display_value = value
                            parsed_metadata[key] = display_value
                            if key not in seen_metadata_keys:
                                seen_metadata_keys.add(key)
                                metadata_column_order.append(key)
                row_metadata.append(parsed_metadata)
        else:
            row_metadata = [{} for _ in rows]

        display_columns = [column for column in columns if column != "metadata_json"]
        if metadata_column_order:
            display_columns.extend(metadata_column_order)

        self.summary_label.setText(
            f"Patients: {patient_count:,} • Documents: {document_count:,} • Showing {len(rows)} most recent notes"
        )
        if display_columns:
            self.table.setColumnCount(len(display_columns))
            self.table.setHorizontalHeaderLabels(display_columns)
        else:
            self.table.setColumnCount(0)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            metadata_for_row = row_metadata[row_index] if row_metadata else {}
            for col_index, column in enumerate(display_columns):
                if column in metadata_column_order:
                    value = metadata_for_row.get(column)
                else:
                    value = row[column]
                text = "" if value is None else str(value).replace("\n", " ")
                if len(text) > 200:
                    text = f"{text[:200]}…"
                item = QtWidgets.QTableWidgetItem(text)
                self.table.setItem(row_index, col_index, item)
        self.table.resizeColumnsToContents()



class IaaWidget(QtWidgets.QWidget):
    METADATA_PRIORITY = [
        "patient_icn",
        "doc_id",
        "display_rank",
        "note_count",
        "complete",
        "opened_at",
        "completed_at",
    ]
    DOCUMENT_METADATA_PRIORITY = [
        "patient_icn",
        "date_note",
        "note_year",
        "notetype",
        "sta3n",
        "cptname",
        "softlabel",
    ]

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
        self._unit_table_column_map: List[Dict[str, object]] = []
        self._unit_table_cache: Dict[str, Dict[str, object]] = {}
        self.unit_metadata_keys: List[str] = []
        self._discordant_units_by_label: Dict[str, Set[str]] = {}
        self._last_evaluated_label_id: Optional[str] = None
        self._active_discord_ids: Set[str] = set()
        self.label_definitions: Dict[str, LabelDefinition] = {}
        self._document_metadata_columns_cache: Dict[str, List[str]] = {}
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

        units_header = QtWidgets.QHBoxLayout()
        units_label = QtWidgets.QLabel("Imported units (overlapping assignments shown first)")
        units_label.setWordWrap(True)
        units_header.addWidget(units_label)
        units_header.addStretch()
        self.export_btn = QtWidgets.QPushButton("Export round data…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_round_data)
        units_header.addWidget(self.export_btn)
        left_layout.addLayout(units_header)

        self.unit_table = QtWidgets.QTableWidget()
        self._update_unit_table_headers()
        self.unit_table.verticalHeader().setVisible(False)
        self.unit_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.unit_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.unit_table.setWordWrap(True)
        self.unit_table.itemSelectionChanged.connect(self._on_unit_selected)
        self.unit_table.setSortingEnabled(True)
        self.unit_table.horizontalHeader().setStretchLastSection(True)
        self.unit_table.cellDoubleClicked.connect(self._show_annotation_dialog)
        self.unit_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.unit_table.customContextMenuRequested.connect(self._on_unit_table_context_menu)
        left_layout.addWidget(self.unit_table, 1)

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
        self.document_table.setColumnCount(0)
        self.document_table.verticalHeader().setVisible(False)
        self.document_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.document_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.document_table.itemSelectionChanged.connect(self._on_document_selected)
        self.document_table.setSortingEnabled(True)
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
        self._unit_table_column_map = []
        self.round_table.setRowCount(0)
        self.label_selector.clear()
        self.unit_table.setRowCount(0)
        self.document_table.setRowCount(0)
        self.document_preview.clear()
        self.manual_reviewer_combo.clear()
        self.manual_import_btn.setEnabled(False)
        self.auto_import_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._set_import_summary("")
        self._set_waiting_summary("")
        self.unit_metadata_keys = []
        self._discordant_units_by_label = {}
        self._last_evaluated_label_id = None
        self._active_discord_ids = set()
        self.label_definitions = {}
        self.round_summary.setText("Select a round to review agreement metrics")
        self._update_unit_table_headers([])

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
        config = self.ctx.get_round_config(str(round_id))
        if isinstance(config, dict):
            self.current_round.setdefault("config", config)
            corpus_id = config.get("corpus_id")
            if corpus_id:
                self.current_round["corpus_id"] = corpus_id
        db = self.ctx.require_db()
        with db.connect() as conn:
            reviewers = conn.execute(
                "SELECT reviewer_id, name FROM reviewers WHERE reviewer_id IN (SELECT reviewer_id FROM assignments WHERE round_id=?)",
                (round_id,),
            ).fetchall()
            labels = conn.execute(
                "SELECT labels.label_id, labels.name, labels.type, labels.na_allowed, labels.unit, labels.min, labels.max, labels.rules "
                "FROM labels JOIN label_sets ON label_sets.labelset_id = labels.labelset_id "
                "WHERE label_sets.labelset_id=? ORDER BY labels.order_index",
                (self.current_round.get("labelset_id"),),
            ).fetchall()
            label_ids = [row["label_id"] for row in labels]
            options_map: Dict[str, List[Dict[str, object]]] = {}
            if label_ids:
                placeholders = ",".join(["?"] * len(label_ids))
                option_rows = conn.execute(
                    f"SELECT label_id, value, display, order_index FROM label_options WHERE labelset_id=? AND label_id IN ({placeholders}) ORDER BY label_id, order_index",
                    (self.current_round.get("labelset_id"), *label_ids),
                ).fetchall()
                for opt in option_rows:
                    options_map.setdefault(opt["label_id"], []).append(
                        {"value": opt["value"], "display": opt["display"], "order_index": opt["order_index"]}
                    )
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
        self.label_definitions = {
            row["label_id"]: LabelDefinition(
                label_id=row["label_id"],
                name=row["name"],
                type=row["type"],
                na_allowed=bool(row["na_allowed"]),
                unit=row["unit"],
                min_value=row["min"],
                max_value=row["max"],
                options=options_map.get(row["label_id"], []),
                rules=str(row["rules"] or ""),
            )
            for row in labels
        }
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
        self._load_persisted_agreement_state()
        if self._last_evaluated_label_id:
            idx = self.label_selector.findData(self._last_evaluated_label_id)
            if idx >= 0:
                self.label_selector.setCurrentIndex(idx)
        self._refresh_units_table()
        self._update_auto_import_state()

    def _auto_discover_imports(self) -> None:
        round_dir = self._resolve_round_dir()
        if not round_dir:
            return
        self._discover_existing_imports(round_dir)
        if self.assignment_paths:
            summary = "Detected existing assignment imports."
            aggregate_path = round_dir / "round_aggregate.db"
            if not aggregate_path.exists():
                try:
                    self._rebuild_round_aggregate(round_dir)
                except Exception as exc:  # noqa: BLE001
                    summary = f"{summary}\nAggregate build failed: {exc}"
                else:
                    summary = f"{summary}\nRound aggregate rebuilt."
            self._set_import_summary(summary)

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
        imported_any = False
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
                self.ctx.refresh_assignment_cache(target_path)
                statuses.append(f"{display_name}: imported")
                imported_any = True
        aggregate_message = ""
        if imported_any:
            try:
                self._rebuild_round_aggregate(round_dir)
            except Exception as exc:  # noqa: BLE001
                aggregate_message = f"Aggregate build failed: {exc}"
                errors += 1
            else:
                aggregate_message = "Round aggregate rebuilt."
        summary_lines = statuses if statuses else ["No reviewers for this round."]
        if aggregate_message:
            summary_lines.append(aggregate_message)
        summary = "\n".join(summary_lines)
        self._set_import_summary(summary)
        if not silent:
            if errors:
                QtWidgets.QMessageBox.warning(self, "Assignment import", summary)
            else:
                QtWidgets.QMessageBox.information(self, "Assignment import", summary)
        self._load_persisted_agreement_state()
        self._refresh_units_table(force=True)
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
        self.ctx.refresh_assignment_cache(target_path)
        aggregate_message = ""
        aggregate_failed = False
        round_dir = self._resolve_round_dir()
        if round_dir:
            try:
                self._rebuild_round_aggregate(round_dir)
            except Exception as exc:  # noqa: BLE001
                aggregate_message = f"Aggregate build failed: {exc}"
                aggregate_failed = True
            else:
                aggregate_message = "Round aggregate rebuilt."
        summary = f"{display_name}: imported from manual selection"
        if aggregate_message:
            summary = f"{summary}\n{aggregate_message}"
        self._set_import_summary(summary)
        if aggregate_failed:
            QtWidgets.QMessageBox.warning(self, "Assignment import", summary)
        else:
            QtWidgets.QMessageBox.information(self, "Assignment import", summary)
        self._load_persisted_agreement_state()
        self._refresh_units_table(force=True)
        self._update_auto_import_state()

    def _copy_assignment_to_imports(self, reviewer_id: str, source: Path) -> Path:
        if not self.current_round:
            raise RuntimeError("Round context missing")
        pheno_id = self.current_round.get("pheno_id")
        round_number = self.current_round.get("round_number")
        if not pheno_id or round_number is None:
            raise RuntimeError("Round metadata incomplete")
        round_dir = self.ctx.resolve_round_dir(pheno_id, int(round_number))
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
        self.ctx.refresh_assignment_cache(target_path)
        self.ctx.mark_dirty()
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
        return self.ctx.resolve_round_dir(pheno_id, int(round_number))

    def _load_manifest(self, round_dir: Path) -> Dict[str, Dict[str, bool]]:
        manifest_path = round_dir / "manifest.csv"
        return self.ctx.get_manifest_flags(manifest_path)

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
                self.ctx.get_assignment_db(candidate)

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

    def _rebuild_round_aggregate(self, round_dir: Path) -> Path:
        if not self.current_round:
            raise RuntimeError("Round context missing")
        round_id = self.current_round.get("round_id")
        if not round_id:
            raise RuntimeError("Round metadata incomplete")
        imports_dir = round_dir / "imports"
        if not imports_dir.exists():
            raise RuntimeError("No imported assignments found")
        aggregate_db = self.ctx.get_round_aggregate_db(round_dir, create=True)
        if not aggregate_db:
            raise RuntimeError("Failed to initialize round aggregate database")
        with aggregate_db.transaction() as agg_conn:
            agg_conn.execute("DELETE FROM unit_annotations")
            agg_conn.execute("DELETE FROM unit_summary")
            for assignment_path in sorted(imports_dir.glob("*_assignment.db")):
                reviewer_id = assignment_path.stem
                if reviewer_id.endswith("_assignment"):
                    reviewer_id = reviewer_id[: -len("_assignment")]
                assignment_db = self.ctx.get_assignment_db(assignment_path)
                if not assignment_db:
                    continue
                with assignment_db.connect() as assign_conn:
                    for unit_row in assign_conn.execute(
                        "SELECT unit_id, patient_icn, doc_id FROM units"
                    ):
                        agg_conn.execute(
                            """
                            INSERT OR IGNORE INTO unit_summary(round_id, unit_id, patient_icn, doc_id)
                            VALUES (?,?,?,?)
                            """,
                            (
                                round_id,
                                unit_row["unit_id"],
                                unit_row["patient_icn"],
                                unit_row["doc_id"],
                            ),
                        )
                    for ann_row in assign_conn.execute(
                        "SELECT unit_id, label_id, value, value_num, value_date, na, notes FROM annotations"
                    ):
                        agg_conn.execute(
                            """
                            INSERT INTO unit_annotations(round_id, unit_id, reviewer_id, label_id, value, value_num, value_date, na, notes)
                            VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                round_id,
                                ann_row["unit_id"],
                                reviewer_id,
                                ann_row["label_id"],
                                ann_row["value"],
                                ann_row["value_num"],
                                ann_row["value_date"],
                                ann_row["na"],
                                ann_row["notes"],
                            ),
                        )
        self.ctx.mark_dirty()
        return (round_dir / "round_aggregate.db").resolve()

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
        headers = ["Unit", "Overlap", "Status"]
        self._unit_table_column_map = [
            {"type": "meta", "key": "unit"},
            {"type": "meta", "key": "overlap"},
            {"type": "meta", "key": "status"},
        ]
        for label_id in self.label_order:
            label_name = self.label_lookup.get(label_id, label_id)
            for reviewer_id in reviewer_ids:
                reviewer_name = self.current_reviewer_names.get(reviewer_id, reviewer_id)
                headers.append(f"{label_name} ({reviewer_name})")
                self._unit_table_column_map.append(
                    {
                        "type": "label",
                        "label_id": label_id,
                        "reviewer_id": reviewer_id,
                    }
                )
                headers.append("Change history")
                self._unit_table_column_map.append(
                    {
                        "type": "history",
                        "label_id": label_id,
                        "reviewer_id": reviewer_id,
                    }
                )
        for key in self.unit_metadata_keys:
            headers.append(self._metadata_header_for(key))
            self._unit_table_column_map.append({"type": "metadata", "key": key})
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

    def _extract_annotation_parts(self, entry: object) -> tuple[str, str]:
        display_value = ""
        notes_value = ""
        if isinstance(entry, dict):
            display_value = str(entry.get("display") or "")
            raw_notes = entry.get("notes")
            notes_value = str(raw_notes).strip() if raw_notes else ""
        elif entry is not None:
            display_value = str(entry)
        return display_value, notes_value

    @staticmethod
    def _normalize_rationale_text(snippet: object) -> str:
        text = str(snippet or "")
        return text.replace("\u2029", "\n")

    def _format_rationale_snippet(self, snippet: object) -> str:
        text = self._normalize_rationale_text(snippet)
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        if len(text) > 80:
            return text[:77] + "…"
        return text

    def _format_rationale_summary(self, rationales: Iterable[Dict[str, object]]) -> str:
        items: List[str] = []
        for rationale in rationales or []:
            doc_id = str(rationale.get("doc_id") or "—")
            start = rationale.get("start_offset")
            end = rationale.get("end_offset")
            if isinstance(start, int) and isinstance(end, int):
                range_part = f"{start}-{end}"
            else:
                range_part = ""
            snippet = self._format_rationale_snippet(rationale.get("snippet"))
            details = doc_id
            if range_part:
                details = f"{details} [{range_part}]"
            if snippet:
                details = f"{details} {snippet}"
            items.append(details.strip())
        return f"Highlights: {'; '.join(items)}" if items else ""

    def _format_annotation_line(self, label_id: str, entry: object) -> str:
        label_name = self.label_lookup.get(label_id, label_id)
        display_value, notes_value = self._extract_annotation_parts(entry)
        parts: List[str] = []
        if display_value:
            parts.append(display_value)
        if notes_value:
            parts.append(f"Notes: {notes_value}")
        rationales = []
        if isinstance(entry, dict):
            rationales = entry.get("rationales") or []
        highlight_text = self._format_rationale_summary(rationales)
        if highlight_text:
            parts.append(highlight_text)
        if not parts:
            parts.append("—")
        return f"{label_name}: {'; '.join(parts)}"

    def _format_annotation_cell(self, entry: object) -> str:
        display_value, _notes_value = self._extract_annotation_parts(entry)
        return display_value

    def _annotation_context_for_cell(
        self, table_row: int, column_info: Dict[str, object]
    ) -> Optional[Dict[str, object]]:
        if column_info.get("type") != "label":
            return None
        index_item = self.unit_table.item(table_row, 0)
        if not index_item:
            return None
        data = index_item.data(QtCore.Qt.ItemDataRole.UserRole)
        try:
            index = int(data)
        except (TypeError, ValueError):
            return None
        if index < 0 or index >= len(self.unit_rows):
            return None
        row_data = self.unit_rows[index]
        reviewer_id = column_info.get("reviewer_id")
        label_id = column_info.get("label_id")
        if not reviewer_id or not label_id:
            return None
        annotations = row_data.get("reviewer_annotations", {}).get(reviewer_id, {})
        entry = annotations.get(label_id)
        label_name = self.label_lookup.get(label_id, label_id)
        reviewer_name = self.current_reviewer_names.get(reviewer_id, reviewer_id)
        unit_id = str(row_data.get("unit_id") or "")
        return {
            "row": row_data,
            "entry": entry,
            "reviewer_id": reviewer_id,
            "label_id": label_id,
            "label_name": label_name,
            "reviewer_name": reviewer_name,
            "unit_id": unit_id,
        }

    def _show_annotation_notes_dialog(
        self, context: Dict[str, object], notes_text: str
    ) -> None:
        if not notes_text:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Reviewer notes")
        layout = QtWidgets.QVBoxLayout(dialog)
        header = QtWidgets.QLabel(
            "\n".join(
                [
                    f"Label: {context.get('label_name', '')}",
                    f"Reviewer: {context.get('reviewer_name', '')}",
                    f"Unit: {context.get('unit_id') or '—'}",
                ]
            )
        )
        header.setWordWrap(True)
        layout.addWidget(header)
        notes_view = QtWidgets.QPlainTextEdit()
        notes_view.setReadOnly(True)
        notes_view.setPlainText(notes_text)
        notes_view.setMinimumSize(400, 200)
        layout.addWidget(notes_view)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        dialog.exec()

    def _show_annotation_highlights_dialog(
        self, context: Dict[str, object], rationales: Iterable[Dict[str, object]]
    ) -> None:
        items = list(rationales or [])
        if not items:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Reviewer highlights")
        layout = QtWidgets.QVBoxLayout(dialog)
        header = QtWidgets.QLabel(
            "\n".join(
                [
                    f"Label: {context.get('label_name', '')}",
                    f"Reviewer: {context.get('reviewer_name', '')}",
                    f"Unit: {context.get('unit_id') or '—'}",
                ]
            )
        )
        header.setWordWrap(True)
        layout.addWidget(header)
        tree = QtWidgets.QTreeWidget()
        tree.setColumnCount(3)
        tree.setHeaderLabels(["Document", "Range", "Text"])
        tree.setRootIsDecorated(False)
        tree.setUniformRowHeights(False)
        tree.setAlternatingRowColors(True)
        tree.setWordWrap(True)
        tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        for rationale in items:
            doc_id = str(rationale.get("doc_id") or "—")
            start = rationale.get("start_offset")
            end = rationale.get("end_offset")
            if isinstance(start, int) and isinstance(end, int):
                range_text = f"{start}-{end}" if end >= start else str(start)
            elif isinstance(start, int):
                range_text = str(start)
            else:
                range_text = ""
            full_text = self._normalize_rationale_text(rationale.get("snippet"))
            item = QtWidgets.QTreeWidgetItem([doc_id, range_text, full_text])
            if full_text:
                item.setToolTip(2, full_text)
            tree.addTopLevelItem(item)
        header_view = tree.header()
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        tree.setMinimumSize(500, 220)
        layout.addWidget(tree)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        dialog.resize(600, 320)
        dialog.exec()

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

    def _current_assignment_state(self) -> Dict[str, tuple[str, float, int]]:
        state: Dict[str, tuple[str, float, int]] = {}
        for reviewer_id, path in self.assignment_paths.items():
            if not path:
                continue
            try:
                stat_result = path.stat()
            except OSError:
                continue
            state[reviewer_id] = (str(path), stat_result.st_mtime, stat_result.st_size)
        return state

    def _aggregate_state_token(self, round_dir: Optional[Path]) -> Optional[tuple[str, float, int]]:
        if not round_dir:
            return None
        path = (round_dir / "round_aggregate.db").resolve()
        try:
            stat_result = path.stat()
        except OSError:
            return None
        return (str(path), stat_result.st_mtime, stat_result.st_size)

    def _register_metadata_key(self, key: Optional[str]) -> None:
        if not key:
            return
        if key not in self.unit_metadata_keys:
            self.unit_metadata_keys.append(key)

    def _finalize_metadata_keys(self) -> None:
        unique: List[str] = []
        for key in self.unit_metadata_keys:
            if key not in unique:
                unique.append(key)
        priority = [key for key in self.METADATA_PRIORITY if key in unique]
        others = [key for key in unique if key not in self.METADATA_PRIORITY]
        self.unit_metadata_keys = priority + others

    def _metadata_header_for(self, key: str) -> str:
        mapping = {
            "patient_icn": "Patient",
            "doc_id": "Document",
            "display_rank": "Display rank",
            "note_count": "Note count",
            "complete": "Complete",
            "opened_at": "Opened at",
            "completed_at": "Completed at",
        }
        return mapping.get(key, key.replace("_", " ").title())

    def _format_metadata_value(self, key: str, value: object) -> str:
        if key == "doc_id" and not value:
            return "—"
        if value is None:
            return ""
        if key == "complete" or isinstance(value, bool):
            return "Yes" if bool(value) else "No"
        return str(value)

    def _collect_unit_metadata(self, summary: sqlite3.Row) -> Dict[str, object]:
        metadata: Dict[str, object] = {}
        metadata["patient_icn"] = summary["patient_icn"]
        metadata["doc_id"] = summary["doc_id"]
        self._register_metadata_key("patient_icn")
        self._register_metadata_key("doc_id")
        raw = summary.get("metadata_json") if hasattr(summary, "get") else summary["metadata_json"]
        if raw:
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                for key, value in sorted(payload.items()):
                    if key in {"unit_id"}:
                        continue
                    if key in metadata and metadata[key]:
                        continue
                    metadata[key] = value
                    self._register_metadata_key(key)
        return metadata

    def _metadata_from_assignment_row(self, unit_row: sqlite3.Row) -> Dict[str, object]:
        data = dict(unit_row)
        metadata: Dict[str, object] = {}
        metadata["patient_icn"] = data.get("patient_icn")
        metadata["doc_id"] = data.get("doc_id")
        self._register_metadata_key("patient_icn")
        self._register_metadata_key("doc_id")
        for key, value in sorted(data.items()):
            if key in {"unit_id", "patient_icn", "doc_id"}:
                continue
            metadata[key] = value
            self._register_metadata_key(key)
        return metadata

    @staticmethod
    def _normalize_rationales(rows: Iterable[sqlite3.Row]) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
        mapping: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        for row in rows:
            unit_id = str(row["unit_id"])
            label_id = str(row["label_id"])
            entry = {
                "doc_id": str(row["doc_id"] or ""),
                "start_offset": int(row["start_offset"]),
                "end_offset": int(row["end_offset"]),
                "snippet": str(row["snippet"] or ""),
            }
            mapping.setdefault(unit_id, {}).setdefault(label_id, []).append(entry)
        for label_map in mapping.values():
            for entries in label_map.values():
                entries.sort(
                    key=lambda value: (
                        value.get("doc_id", ""),
                        int(value.get("start_offset", 0)),
                        int(value.get("end_offset", 0)),
                    )
                )
        return mapping

    def _collect_reviewer_rationales(self) -> Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]]:
        cache: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]] = {}
        for reviewer_id, path in self.assignment_paths.items():
            if not path:
                continue
            db = self.ctx.get_assignment_db(path)
            if not db:
                continue
            try:
                with db.connect() as conn:
                    rows = conn.execute(
                        "SELECT unit_id, label_id, doc_id, start_offset, end_offset, snippet FROM rationales"
                    ).fetchall()
            except Exception:
                continue
            normalized = self._normalize_rationales(rows)
            reviewer_map: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
            for unit_id, labels in normalized.items():
                reviewer_map[unit_id] = {
                    label_id: [dict(entry) for entry in entries] for label_id, entries in labels.items()
                }
            cache[reviewer_id] = reviewer_map
        return cache

    def _load_unit_rows_from_aggregate(
        self,
        aggregate_db: Database,
        round_id: str,
        existing_discord: Set[str],
    ) -> bool:
        try:
            with aggregate_db.connect() as conn:
                summary_rows = conn.execute(
                    "SELECT unit_id, patient_icn, doc_id, metadata_json FROM unit_summary WHERE round_id=?",
                    (round_id,),
                ).fetchall()
                annotation_rows = conn.execute(
                    "SELECT unit_id, reviewer_id, label_id, value, value_num, value_date, na, notes FROM unit_annotations WHERE round_id=?",
                    (round_id,),
                ).fetchall()
                history_rows = conn.execute(
                    "SELECT unit_id, reviewer_id, label_id, history FROM annotation_change_history WHERE round_id=?",
                    (round_id,),
                ).fetchall()
        except Exception:
            return False
        reviewer_rationales = self._collect_reviewer_rationales()
        unit_map: Dict[str, Dict[str, object]] = {}
        history_map: Dict[str, Dict[tuple[str, str], str]] = {}
        for history in history_rows:
            unit_history = history_map.setdefault(history["unit_id"], {})
            key = (history["reviewer_id"], history["label_id"])
            unit_history[key] = history["history"] or ""
        for summary in summary_rows:
            unit_id = summary["unit_id"]
            metadata = self._collect_unit_metadata(summary)
            entry = {
                "unit_id": unit_id,
                "patient_icn": metadata.get("patient_icn"),
                "doc_id": metadata.get("doc_id"),
                "metadata": metadata,
                "reviewer_annotations": {},
                "reviewer_ids": set(),
                "change_history": history_map.get(unit_id, {}),
                "is_overlap": False,
                "discord": unit_id in existing_discord,
            }
            unit_map[unit_id] = entry
        for ann_row in annotation_rows:
            unit_id = ann_row["unit_id"]
            entry = unit_map.setdefault(
                unit_id,
                {
                    "unit_id": unit_id,
                    "patient_icn": None,
                    "doc_id": None,
                    "metadata": {},
                    "reviewer_annotations": {},
                    "reviewer_ids": set(),
                    "change_history": history_map.get(unit_id, {}),
                    "is_overlap": False,
                    "discord": unit_id in existing_discord,
                },
            )
            reviewer_id = ann_row["reviewer_id"]
            label_id = ann_row["label_id"]
            entry["reviewer_ids"].add(reviewer_id)
            annotations = entry["reviewer_annotations"].setdefault(reviewer_id, {})
            annotations[label_id] = {
                "display": self._compute_annotation_display(
                    ann_row["value"], ann_row["value_num"], ann_row["value_date"], ann_row["na"]
                ),
                "notes": ann_row["notes"] or "",
                "value": ann_row["value"],
                "value_num": ann_row["value_num"],
                "value_date": ann_row["value_date"],
                "na": ann_row["na"],
                "rationales": [
                    dict(entry)
                    for entry in reviewer_rationales.get(reviewer_id, {}).get(unit_id, {}).get(label_id, [])
                ],
            }
        self.unit_rows = []
        for unit_id, entry in unit_map.items():
            metadata = entry.get("metadata") or {}
            if not metadata:
                metadata.update({"patient_icn": entry.get("patient_icn"), "doc_id": entry.get("doc_id")})
                self._register_metadata_key("patient_icn")
                self._register_metadata_key("doc_id")
            reviewers = sorted(
                entry.get("reviewer_ids", []),
                key=lambda rid: self.current_reviewer_names.get(rid, rid),
            )
            entry["reviewer_ids"] = reviewers
            entry["is_overlap"] = self._is_overlap_unit(unit_id, reviewers)
            entry["patient_icn"] = metadata.get("patient_icn")
            entry["doc_id"] = metadata.get("doc_id")
            entry["discord"] = unit_id in existing_discord
            self.unit_rows.append(entry)
        return bool(self.unit_rows)

    def _load_unit_rows_from_assignments(self, existing_discord: Set[str]) -> None:
        unit_map: Dict[str, Dict[str, object]] = {}
        for reviewer_id, path in self.assignment_paths.items():
            if not path:
                continue
            db = self.ctx.get_assignment_db(path)
            if not db:
                continue
            with db.connect() as conn:
                units = conn.execute("SELECT * FROM units ORDER BY display_rank").fetchall()
                annotations = conn.execute(
                    "SELECT unit_id, label_id, value, value_num, value_date, na, notes FROM annotations",
                ).fetchall()
                rationale_rows = conn.execute(
                    "SELECT unit_id, label_id, doc_id, start_offset, end_offset, snippet FROM rationales"
                ).fetchall()
            rationale_map = self._normalize_rationales(rationale_rows)
            ann_map: Dict[str, Dict[str, Dict[str, object]]] = {}
            for ann_row in annotations:
                unit_id = ann_row["unit_id"]
                ann_map.setdefault(unit_id, {})[ann_row["label_id"]] = {
                    "display": self._format_value(ann_row),
                    "notes": ann_row["notes"] or "",
                    "value": ann_row["value"],
                    "value_num": ann_row["value_num"],
                    "value_date": ann_row["value_date"],
                    "na": ann_row["na"],
                    "rationales": [
                        dict(entry)
                        for entry in rationale_map.get(unit_id, {}).get(ann_row["label_id"], [])
                    ],
                }
            for unit_row in units:
                unit_id = unit_row["unit_id"]
                entry = unit_map.setdefault(
                    unit_id,
                    {
                        "unit_id": unit_id,
                        "patient_icn": unit_row["patient_icn"],
                        "doc_id": unit_row["doc_id"],
                        "metadata": {},
                        "reviewer_annotations": {},
                        "reviewer_ids": set(),
                        "change_history": {},
                        "is_overlap": False,
                        "discord": unit_id in existing_discord,
                    },
                )
                metadata = entry.setdefault("metadata", {})
                new_metadata = self._metadata_from_assignment_row(unit_row)
                for key, value in new_metadata.items():
                    if key not in metadata or metadata[key] in (None, "", "—"):
                        metadata[key] = value
                entry["patient_icn"] = metadata.get("patient_icn")
                entry["doc_id"] = metadata.get("doc_id")
                entry["reviewer_ids"].add(reviewer_id)
                entry["reviewer_annotations"][reviewer_id] = ann_map.get(unit_id, {})
        self.unit_rows = []
        for unit_id, entry in unit_map.items():
            reviewers = sorted(
                entry.get("reviewer_ids", []),
                key=lambda rid: self.current_reviewer_names.get(rid, rid),
            )
            entry["reviewer_ids"] = reviewers
            entry["is_overlap"] = self._is_overlap_unit(unit_id, reviewers)
            entry["discord"] = unit_id in existing_discord
            self.unit_rows.append(entry)

    def _determine_reviewer_order(self) -> List[str]:
        reviewer_ids: Set[str] = set()
        for reviewer in self.current_round.get("reviewers", []):
            rid = reviewer.get("reviewer_id")
            if rid:
                reviewer_ids.add(rid)
        reviewer_ids.update(self.assignment_paths.keys())
        for row in self.unit_rows:
            for reviewer_id in row.get("reviewer_annotations", {}).keys():
                if reviewer_id:
                    reviewer_ids.add(reviewer_id)
        return sorted(
            reviewer_ids,
            key=lambda rid: self.current_reviewer_names.get(rid, rid).lower(),
        )

    def _load_persisted_agreement_state(self) -> None:
        if not self.current_round:
            self._discordant_units_by_label = {}
            self._active_discord_ids = set()
            self._last_evaluated_label_id = None
            return
        round_id = self.current_round.get("round_id")
        if not round_id:
            self._discordant_units_by_label = {}
            self._active_discord_ids = set()
            self._last_evaluated_label_id = None
            return
        round_dir = self._resolve_round_dir()
        if not round_dir:
            self._discordant_units_by_label = {}
            self._active_discord_ids = set()
            return
        aggregate_db = self.ctx.get_round_aggregate_db(round_dir, create=False)
        if not aggregate_db:
            self._discordant_units_by_label = {}
            self._active_discord_ids = set()
            self._last_evaluated_label_id = None
            return
        try:
            with aggregate_db.connect() as conn:
                state_row = conn.execute(
                    "SELECT last_label_id FROM round_state WHERE round_id=?",
                    (round_id,),
                ).fetchone()
                status_rows = conn.execute(
                    "SELECT label_id, unit_id, status FROM unit_status WHERE round_id=?",
                    (round_id,),
                ).fetchall()
        except Exception:
            self._discordant_units_by_label = {}
            self._active_discord_ids = set()
            return
        if state_row and state_row["last_label_id"]:
            self._last_evaluated_label_id = state_row["last_label_id"]
        else:
            self._last_evaluated_label_id = None
        status_map: Dict[str, Set[str]] = {}
        for row in status_rows:
            status_text = (row["status"] or "").lower()
            if status_text == "discordant":
                status_map.setdefault(row["label_id"], set()).add(row["unit_id"])
        self._discordant_units_by_label = status_map
        if self._last_evaluated_label_id:
            self._active_discord_ids = set(
                status_map.get(self._last_evaluated_label_id, set())
            )
        else:
            self._active_discord_ids = set()
        try:
            self.round_manifest = self._load_manifest(round_dir)
        except Exception:
            pass

    def _update_cached_unit_rows(self) -> None:
        if not self.current_round:
            return
        round_id = self.current_round.get("round_id")
        if not round_id:
            return
        cache_entry = self._unit_table_cache.get(str(round_id))
        if cache_entry is not None:
            cache_entry["unit_rows"] = copy.deepcopy(self.unit_rows)
            cache_entry["metadata_keys"] = list(self.unit_metadata_keys)

    def _persist_discord_state(self, label_id: str, discordant_ids: Set[str]) -> None:
        round_dir = self._resolve_round_dir()
        if not round_dir or not self.current_round:
            return
        round_id = self.current_round.get("round_id")
        if not round_id:
            return
        aggregate_db = self.ctx.get_round_aggregate_db(round_dir, create=True)
        if not aggregate_db:
            return
        with aggregate_db.transaction() as conn:
            conn.execute(
                "DELETE FROM unit_status WHERE round_id=? AND label_id=?",
                (round_id, label_id),
            )
            if discordant_ids:
                conn.executemany(
                    "INSERT INTO unit_status(round_id, label_id, unit_id, status) VALUES (?,?,?,?)",
                    [
                        (round_id, label_id, unit_id, "discordant")
                        for unit_id in sorted(discordant_ids)
                    ],
                )
            conn.execute(
                "INSERT INTO round_state(round_id, last_label_id) VALUES (?, ?) "
                "ON CONFLICT(round_id) DO UPDATE SET last_label_id=excluded.last_label_id",
                (round_id, label_id),
            )
        self.ctx.mark_dirty()

    def _update_discord_state(self, label_id: str, discordant_ids: Set[str]) -> None:
        self._discordant_units_by_label[label_id] = set(discordant_ids)
        self._last_evaluated_label_id = label_id
        self._apply_discord_flags(discordant_ids)
        self._persist_discord_state(label_id, discordant_ids)

    def _persist_annotation_edit(
        self,
        unit_id: str,
        reviewer_id: str,
        label_id: str,
        annotation: Dict[str, object],
        history_entry: str,
    ) -> None:
        round_dir = self._resolve_round_dir()
        if not round_dir or not self.current_round:
            return
        round_id = self.current_round.get("round_id")
        if not round_id:
            return
        aggregate_db = self.ctx.get_round_aggregate_db(round_dir, create=True)
        if not aggregate_db:
            return
        with aggregate_db.transaction() as conn:
            existing = conn.execute(
                "SELECT notes FROM unit_annotations WHERE round_id=? AND unit_id=? AND reviewer_id=? AND label_id=?",
                (round_id, unit_id, reviewer_id, label_id),
            ).fetchone()
            notes_value = annotation.get("notes", "") if annotation else ""
            if existing:
                if not notes_value:
                    notes_value = existing["notes"] or ""
                conn.execute(
                    "UPDATE unit_annotations SET value=?, value_num=?, value_date=?, na=?, notes=? WHERE round_id=? AND unit_id=? AND reviewer_id=? AND label_id=?",
                    (
                        annotation.get("value"),
                        annotation.get("value_num"),
                        annotation.get("value_date"),
                        annotation.get("na"),
                        notes_value,
                        round_id,
                        unit_id,
                        reviewer_id,
                        label_id,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO unit_annotations(round_id, unit_id, reviewer_id, label_id, value, value_num, value_date, na, notes) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        round_id,
                        unit_id,
                        reviewer_id,
                        label_id,
                        annotation.get("value"),
                        annotation.get("value_num"),
                        annotation.get("value_date"),
                        annotation.get("na"),
                        notes_value,
                    ),
                )
            history_row = conn.execute(
                "SELECT history FROM annotation_change_history WHERE round_id=? AND unit_id=? AND reviewer_id=? AND label_id=?",
                (round_id, unit_id, reviewer_id, label_id),
            ).fetchone()
            if history_row:
                existing_history = history_row["history"] or ""
                combined = f"{existing_history}\n{history_entry}" if existing_history else history_entry
                conn.execute(
                    "UPDATE annotation_change_history SET history=? WHERE round_id=? AND unit_id=? AND reviewer_id=? AND label_id=?",
                    (combined, round_id, unit_id, reviewer_id, label_id),
                )
            else:
                conn.execute(
                    "INSERT INTO annotation_change_history(round_id, unit_id, reviewer_id, label_id, history) VALUES (?,?,?,?,?)",
                    (round_id, unit_id, reviewer_id, label_id, history_entry),
                )
        self.ctx.mark_dirty()

    def _refresh_units_table(self, force: bool = False) -> None:
        selected_unit = self._selected_unit_id()
        self._load_persisted_agreement_state()
        existing_discord = set(self._active_discord_ids)
        self.unit_rows = []
        self.unit_metadata_keys = []
        self.unit_table.clearContents()
        self.unit_table.setRowCount(0)
        self._clear_document_panel()
        if not self.current_round:
            self.reviewer_column_order = []
            self._update_unit_table_headers()
            self._display_unit_rows(selected_unit)
            return
        round_id = self.current_round.get('round_id')
        project_id = self.ctx.current_project_id()
        assignment_state = self._current_assignment_state()
        round_dir = self._resolve_round_dir()
        aggregate_state = self._aggregate_state_token(round_dir)
        if round_id and not force:
            cached = self._unit_table_cache.get(str(round_id))
            if (
                cached
                and cached.get('assignment_state') == assignment_state
                and cached.get('project_id') == project_id
                and cached.get('aggregate_state') == aggregate_state
            ):
                self.reviewer_column_order = list(cached.get('reviewer_order', []))
                self.unit_rows = copy.deepcopy(cached.get('unit_rows', []))
                self.unit_metadata_keys = list(cached.get('metadata_keys', []))
                for row in self.unit_rows:
                    row['discord'] = row.get('unit_id') in existing_discord
                self._update_unit_table_headers()
                self._display_unit_rows(selected_unit)
                return
        aggregate_db = None
        if round_dir:
            aggregate_db = self.ctx.get_round_aggregate_db(round_dir, create=False)
        loaded = False
        if aggregate_db and round_id:
            loaded = self._load_unit_rows_from_aggregate(aggregate_db, str(round_id), existing_discord)
        if not loaded:
            self._load_unit_rows_from_assignments(existing_discord)
        self._finalize_metadata_keys()
        self.reviewer_column_order = self._determine_reviewer_order()
        for index, row in enumerate(self.unit_rows):
            row['index'] = index
        self._update_unit_table_headers()
        self._display_unit_rows(selected_unit)
        if round_id:
            self._unit_table_cache[str(round_id)] = {
                'unit_rows': copy.deepcopy(self.unit_rows),
                'reviewer_order': list(self.reviewer_column_order),
                'assignment_state': assignment_state,
                'project_id': project_id,
                'metadata_keys': list(self.unit_metadata_keys),
                'aggregate_state': aggregate_state,
            }

    def _export_round_data(self) -> None:
        if not self.unit_rows:
            QtWidgets.QMessageBox.information(
                self,
                "Export round data",
                "No data available to export.",
            )
            return
        if not self.current_round:
            QtWidgets.QMessageBox.information(
                self,
                "Export round data",
                "Select a round before exporting.",
            )
            return
        has_annotations = any(
            annotations
            for row in self.unit_rows
            for annotations in (row.get("reviewer_annotations") or {}).values()
        )
        if not has_annotations:
            QtWidgets.QMessageBox.information(
                self,
                "Export round data",
                "No reviewer annotations available to export.",
            )
            return
        round_id = str(self.current_round.get("round_id") or "")
        if not round_id:
            QtWidgets.QMessageBox.information(
                self,
                "Export round data",
                "Round metadata is incomplete; refresh and try again.",
            )
            return
        pheno_id = str(self.current_round.get("pheno_id") or "")
        unit_ids: List[str] = []
        doc_ids_by_unit: Dict[str, Set[str]] = {}
        for unit_row in self.unit_rows:
            unit_id = str(unit_row.get("unit_id") or "")
            if not unit_id:
                continue
            unit_ids.append(unit_id)
            doc_ids = doc_ids_by_unit.setdefault(unit_id, set())
            metadata = unit_row.get("metadata") or {}
            candidate_doc = metadata.get("doc_id") or unit_row.get("doc_id")
            if candidate_doc:
                doc_ids.add(str(candidate_doc))
            reviewer_annotations = unit_row.get("reviewer_annotations") or {}
            for annotation_map in reviewer_annotations.values():
                for annotation in annotation_map.values():
                    for rationale in annotation.get("rationales") or []:
                        doc_ref = rationale.get("doc_id")
                        if doc_ref:
                            doc_ids.add(str(doc_ref))
        documents_by_unit = self._gather_unit_documents(unit_ids, doc_ids_by_unit)
        project_root = self.ctx.project_root or Path.home()
        default_name = f"{round_id}_round_data.csv"
        default_path = Path(project_root) / default_name
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export round data",
            str(default_path),
            "CSV files (*.csv);;All files (*)",
        )
        if not path_str:
            return
        target_path = Path(path_str)
        if target_path.suffix.lower() != ".csv":
            target_path = target_path.with_suffix(".csv")
        headers = [
            "round_id",
            "phenotype_id",
            "unit_id",
            "doc_id",
            "patient_icn",
            "reviewer_id",
            "reviewer_name",
            "label_id",
            "label_name",
            "label_value",
            "label_value_num",
            "label_value_date",
            "label_na",
            "reviewer_notes",
            "rationales_json",
            "document_text",
            "document_metadata_json",
            "label_rules",
            "label_change_history",
        ]
        try:
            rows_written = 0
            with target_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                label_rules = {label_id: definition.rules for label_id, definition in self.label_definitions.items()}
                for unit_row in self.unit_rows:
                    unit_id = str(unit_row.get("unit_id") or "")
                    if not unit_id:
                        continue
                    metadata = unit_row.get("metadata") or {}
                    patient_icn = str(metadata.get("patient_icn") or unit_row.get("patient_icn") or "")
                    reviewer_annotations = unit_row.get("reviewer_annotations") or {}
                    change_history = unit_row.get("change_history") or {}
                    doc_entries = documents_by_unit.get(unit_id, {})
                    sorted_docs = sorted(
                        doc_entries.items(),
                        key=lambda item: item[1].get("order_index", 0),
                    )
                    ordered_doc_ids: List[str] = [doc_id for doc_id, _ in sorted_docs]
                    known_doc_ids = set(ordered_doc_ids)
                    for doc_id in sorted(doc_ids_by_unit.get(unit_id, set())):
                        if doc_id and doc_id not in known_doc_ids:
                            ordered_doc_ids.append(doc_id)
                            known_doc_ids.add(doc_id)
                    if not ordered_doc_ids:
                        ordered_doc_ids = [""]
                    for reviewer_id, annotation_map in sorted(
                        reviewer_annotations.items(), key=lambda item: self.current_reviewer_names.get(item[0], item[0])
                    ):
                        reviewer_name = self.current_reviewer_names.get(reviewer_id, reviewer_id)
                        ordered_labels = [label_id for label_id in self.label_order if label_id in annotation_map]
                        extra_labels = [label_id for label_id in annotation_map.keys() if label_id not in self.label_order]
                        for label_id in ordered_labels + sorted(extra_labels):
                            annotation = annotation_map.get(label_id)
                            if not annotation:
                                continue
                            label_name = self.label_lookup.get(label_id, label_id)
                            rules_text = str(label_rules.get(label_id, "") or "")
                            history_text = str(change_history.get((reviewer_id, label_id), "") or "")
                            value = annotation.get("value")
                            value_str = "" if value is None else str(value)
                            value_num = annotation.get("value_num")
                            value_num_str = ""
                            if value_num is not None:
                                try:
                                    value_num_str = format(value_num, "g")
                                except Exception:  # noqa: BLE001
                                    value_num_str = str(value_num)
                            value_date = str(annotation.get("value_date") or "")
                            notes = str(annotation.get("notes") or "")
                            na_flag = 1 if annotation.get("na") else 0
                            rationales = annotation.get("rationales") or []
                            normalized_rationales = [
                                {
                                    "doc_id": str(rationale.get("doc_id") or ""),
                                    "start_offset": rationale.get("start_offset"),
                                    "end_offset": rationale.get("end_offset"),
                                    "snippet": rationale.get("snippet"),
                                }
                                for rationale in rationales
                            ]
                            for doc_id in ordered_doc_ids:
                                doc_id_str = str(doc_id or "")
                                doc_entry = doc_entries.get(doc_id_str, {}) if doc_id_str else {}
                                doc_text = str(doc_entry.get("text") or "")
                                metadata_json = str(doc_entry.get("metadata_json") or "")
                                if doc_id_str:
                                    doc_specific_rationales = [
                                        entry
                                        for entry in normalized_rationales
                                        if entry.get("doc_id") == doc_id_str
                                    ]
                                else:
                                    doc_specific_rationales = normalized_rationales
                                rationale_json = (
                                    json.dumps(doc_specific_rationales, ensure_ascii=False)
                                    if doc_specific_rationales
                                    else ""
                                )
                                writer.writerow(
                                    [
                                        round_id,
                                        pheno_id,
                                        unit_id,
                                        doc_id_str,
                                        patient_icn,
                                        str(reviewer_id),
                                        str(reviewer_name),
                                        str(label_id),
                                        str(label_name),
                                        value_str,
                                        value_num_str,
                                        value_date,
                                        na_flag,
                                        notes,
                                        rationale_json,
                                        doc_text,
                                        metadata_json,
                                        rules_text,
                                        history_text,
                                    ]
                                )
                                rows_written += 1
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self,
                "Export round data",
                f"Failed to export round data: {exc}",
            )
            return
        if rows_written == 0:
            try:
                if target_path.exists():
                    target_path.unlink()
            except Exception:  # noqa: BLE001
                pass
            QtWidgets.QMessageBox.information(
                self,
                "Export round data",
                "No annotation rows were available to export.",
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            "Export round data",
            f"Round data exported to {target_path} ({rows_written} rows).",
        )

    def _display_unit_rows(self, selected_unit: Optional[str] = None) -> None:
        header = self.unit_table.horizontalHeader()
        sort_section = header.sortIndicatorSection()
        sort_order = header.sortIndicatorOrder()
        was_sorting = self.unit_table.isSortingEnabled()
        self.unit_table.setSortingEnabled(False)
        self.unit_table.clearContents()
        if not self.unit_rows:
            self.unit_table.setRowCount(0)
            self.export_btn.setEnabled(False)
            self.unit_table.setSortingEnabled(was_sorting)
            return
        sorted_rows = sorted(
            self.unit_rows,
            key=lambda row: (not row["is_overlap"], row["unit_id"]),
        )
        self.unit_table.setRowCount(len(sorted_rows))
        highlight = QtGui.QColor("#ffebee")
        for row_index, row in enumerate(sorted_rows):
            row_items: List[QtWidgets.QTableWidgetItem] = []
            for column_info in self._unit_table_column_map:
                column_type = column_info.get("type")
                if column_type == "meta":
                    key = column_info.get("key")
                    if key == "unit":
                        text = row.get("unit_id", "")
                        item = QtWidgets.QTableWidgetItem(text)
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, row.get("index"))
                    elif key == "overlap":
                        item = QtWidgets.QTableWidgetItem("Yes" if row.get("is_overlap") else "No")
                    elif key == "status":
                        item = QtWidgets.QTableWidgetItem("Discordant" if row.get("discord") else "")
                    else:
                        item = QtWidgets.QTableWidgetItem("")
                elif column_type == "label":
                    reviewer_id = column_info.get("reviewer_id")
                    label_id = column_info.get("label_id")
                    annotations = row.get("reviewer_annotations", {}).get(reviewer_id, {})
                    entry = annotations.get(label_id)
                    text = self._format_annotation_cell(entry)
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setToolTip(self._format_annotation_line(label_id, entry))
                elif column_type == "history":
                    reviewer_id = column_info.get("reviewer_id")
                    label_id = column_info.get("label_id")
                    history_map = row.get("change_history", {})
                    history_text = history_map.get((reviewer_id, label_id), "")
                    item = QtWidgets.QTableWidgetItem(history_text)
                    if history_text:
                        item.setToolTip(history_text)
                elif column_type == "metadata":
                    key = column_info.get("key")
                    metadata = row.get("metadata", {})
                    value = metadata.get(key)
                    item = QtWidgets.QTableWidgetItem(self._format_metadata_value(key, value))
                else:
                    item = QtWidgets.QTableWidgetItem("")
                row_items.append(item)
            for column, item in enumerate(row_items):
                self.unit_table.setItem(row_index, column, item)
            if row.get("is_overlap"):
                for item in row_items:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
            if row.get("discord"):
                for item in row_items:
                    item.setBackground(highlight)
            self.unit_table.resizeRowToContents(row_index)
        self.unit_table.resizeColumnsToContents()
        self.export_btn.setEnabled(True)
        self.unit_table.setSortingEnabled(was_sorting)
        if was_sorting and header.isSortIndicatorShown():
            self.unit_table.sortItems(sort_section, sort_order)
        if selected_unit:
            self._select_unit_in_table(selected_unit)

    def _on_unit_table_context_menu(self, point: QtCore.QPoint) -> None:
        item = self.unit_table.itemAt(point)
        if not item:
            return
        column = item.column()
        if column < 0 or column >= len(self._unit_table_column_map):
            return
        column_info = self._unit_table_column_map[column]
        if column_info.get("type") != "label":
            return
        menu = QtWidgets.QMenu(self.unit_table)
        edit_action = menu.addAction("Edit reviewer label…")
        context = self._annotation_context_for_cell(item.row(), column_info)
        notes_action: Optional[QtGui.QAction] = None
        highlights_action: Optional[QtGui.QAction] = None
        notes_value = ""
        rationale_items: List[Dict[str, object]] = []
        if context:
            entry = context.get("entry")
            _display_value, notes_value = self._extract_annotation_parts(entry)
            if isinstance(entry, dict):
                rationale_items = list(entry.get("rationales") or [])
            if notes_value or rationale_items:
                menu.addSeparator()
            if notes_value:
                notes_action = menu.addAction("View notes…")
            if rationale_items:
                highlights_action = menu.addAction("View highlights…")
        global_pos = self.unit_table.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)
        if chosen == edit_action:
            self._edit_reviewer_label(item.row(), column)
        elif notes_action and chosen == notes_action and context:
            self._show_annotation_notes_dialog(context, notes_value)
        elif highlights_action and chosen == highlights_action and context:
            self._show_annotation_highlights_dialog(context, rationale_items)

    def _edit_reviewer_label(self, row: int, column: int) -> None:
        if column < 0 or column >= len(self._unit_table_column_map):
            return
        column_info = self._unit_table_column_map[column]
        if column_info.get("type") != "label":
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
        label_id = column_info.get("label_id")
        reviewer_id = column_info.get("reviewer_id")
        if not label_id or not reviewer_id:
            return
        definition = self.label_definitions.get(label_id)
        if not definition:
            QtWidgets.QMessageBox.warning(
                self,
                "Edit reviewer label",
                "Label definition is unavailable; reload the project and try again.",
            )
            return
        unit_row = self.unit_rows[index]
        reviewer_annotations = unit_row.setdefault("reviewer_annotations", {})
        annotations = reviewer_annotations.setdefault(reviewer_id, {})
        current_annotation = annotations.get(label_id, {})
        current_display = self._format_annotation_cell(current_annotation)
        dialog = ReviewerLabelEditDialog(
            definition,
            reviewer_name=self.current_reviewer_names.get(reviewer_id, reviewer_id),
            unit_id=str(unit_row.get("unit_id") or ""),
            current_annotation=current_annotation,
            current_display=current_display,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        result = dialog.result_data
        if not result:
            return
        annotation_payload = result["annotation"]
        rationale = result["rationale"]
        old_annotation = current_annotation or {}
        if (
            old_annotation.get("value") == annotation_payload["value"]
            and old_annotation.get("value_num") == annotation_payload["value_num"]
            and old_annotation.get("value_date") == annotation_payload["value_date"]
            and bool(old_annotation.get("na")) == bool(annotation_payload["na"])
        ):
            QtWidgets.QMessageBox.information(
                self,
                "Edit reviewer label",
                "No changes detected for the selected label.",
            )
            return
        notes = old_annotation.get("notes", "") if isinstance(old_annotation, dict) else ""
        new_entry = {
            "display": self._compute_annotation_display(
                annotation_payload["value"],
                annotation_payload["value_num"],
                annotation_payload["value_date"],
                annotation_payload["na"],
            ),
            "notes": notes,
            "value": annotation_payload["value"],
            "value_num": annotation_payload["value_num"],
            "value_date": annotation_payload["value_date"],
            "na": annotation_payload["na"],
        }
        annotations[label_id] = new_entry
        history_map = unit_row.setdefault("change_history", {})
        old_display = self._format_annotation_cell(old_annotation)
        new_display = self._format_annotation_cell(new_entry)
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        history_entry = f"{timestamp}: {old_display or '—'} -> {new_display or '—'} ({rationale})"
        key = (reviewer_id, label_id)
        if key in history_map and history_map[key]:
            history_map[key] = history_map[key] + "\n" + history_entry
        else:
            history_map[key] = history_entry
        unit_id = str(unit_row.get("unit_id") or "")
        self._persist_annotation_edit(unit_id, reviewer_id, label_id, {**annotation_payload, "notes": notes}, history_entry)
        current_label_id = self.label_selector.currentData()
        self._load_persisted_agreement_state()
        self._refresh_units_table(force=True)
        if current_label_id is not None:
            idx = self.label_selector.findData(current_label_id)
            if idx >= 0:
                self.label_selector.setCurrentIndex(idx)
        if unit_id:
            self._select_unit_in_table(unit_id)

    def _select_unit_in_table(self, unit_id: str) -> None:
        if not unit_id:
            return
        for row_index in range(self.unit_table.rowCount()):
            item = self.unit_table.item(row_index, 0)
            if item and item.text() == unit_id:
                self.unit_table.selectRow(row_index)
                break

    def _on_unit_selected(self) -> None:
        index = self._selected_unit_index()
        if index is None or index >= len(self.unit_rows):
            self._clear_document_panel()
            return
        row = self.unit_rows[index]
        self._populate_document_table(row)

    def _show_annotation_dialog(self, row: int, column: int) -> None:
        if column < 0 or column >= len(self._unit_table_column_map):
            return
        column_info = self._unit_table_column_map[column]
        if column_info.get("type") != "label":
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
        reviewer_id = column_info.get("reviewer_id")
        label_id = column_info.get("label_id")
        row_data = self.unit_rows[index]
        annotations = row_data.get("reviewer_annotations", {}).get(reviewer_id, {})
        detail = self._format_annotation_summary(annotations)
        label_detail = ""
        if label_id is not None:
            label_detail = self._format_annotation_line(label_id, annotations.get(label_id))
        if detail:
            message_body = f"{label_detail}\n\nAll annotations:\n{detail}" if label_detail else detail
        else:
            message_body = label_detail or "No annotations submitted."
        reviewer_name = self.current_reviewer_names.get(reviewer_id, reviewer_id)
        unit_id = row_data.get("unit_id", "")
        QtWidgets.QMessageBox.information(
            self,
            "Annotation details",
            f"Reviewer: {reviewer_name}\nUnit: {unit_id}\n\n{message_body}",
        )

    @staticmethod
    def _chunked_list(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
        if chunk_size <= 0:
            yield list(items)
            return
        for index in range(0, len(items), chunk_size):
            yield list(items[index : index + chunk_size])

    def _gather_unit_documents(
        self,
        unit_ids: Iterable[str],
        doc_ids_by_unit: Dict[str, Set[str]],
    ) -> Dict[str, Dict[str, Dict[str, object]]]:
        unit_id_set = {str(uid) for uid in unit_ids if uid}
        unit_id_set.update(str(uid) for uid in doc_ids_by_unit.keys() if uid)
        if not unit_id_set:
            return {}
        result: Dict[str, Dict[str, Dict[str, object]]] = {unit_id: {} for unit_id in unit_id_set}
        missing_units = set(unit_id_set)
        assignment_dbs: List[Database] = []
        for path in self.assignment_paths.values():
            if not path:
                continue
            db = self.ctx.get_assignment_db(path)
            if db:
                assignment_dbs.append(db)
        for db in assignment_dbs:
            if not missing_units:
                break
            try:
                with db.connect() as conn:
                    units_to_query = list(missing_units)
                    for chunk in self._chunked_list(units_to_query, 200):
                        if not chunk:
                            continue
                        placeholders = ",".join(["?"] * len(chunk))
                        query = (
                            "SELECT unit_notes.unit_id, unit_notes.doc_id, unit_notes.order_index, "
                            "documents.text, documents.metadata_json "
                            "FROM unit_notes "
                            "LEFT JOIN documents ON documents.doc_id = unit_notes.doc_id "
                            f"WHERE unit_notes.unit_id IN ({placeholders}) "
                            "ORDER BY unit_notes.unit_id, unit_notes.order_index"
                        )
                        rows = conn.execute(query, chunk).fetchall()
                        for row in rows:
                            unit_value = str(row["unit_id"])
                            doc_value = str(row["doc_id"] or "")
                            if not doc_value:
                                continue
                            payload = {
                                "text": row["text"] or "",
                                "metadata_json": row["metadata_json"] or "",
                                "order_index": int(row["order_index"] or 0),
                            }
                            result.setdefault(unit_value, {})[doc_value] = payload
                        for candidate in chunk:
                            candidate_str = str(candidate)
                            if result.get(candidate_str):
                                missing_units.discard(candidate_str)
            except Exception:  # noqa: BLE001
                continue
        for unit_id in doc_ids_by_unit.keys():
            result.setdefault(str(unit_id), {})
        required_doc_ids: Set[str] = set()
        for unit_id, doc_ids in doc_ids_by_unit.items():
            unit_key = str(unit_id)
            known = set(result.get(unit_key, {}).keys())
            for doc_id in doc_ids:
                doc_key = str(doc_id)
                if doc_key and doc_key not in known:
                    required_doc_ids.add(doc_key)
        corpus_db: Optional[Database] = None
        corpus_id = (self.current_round or {}).get("corpus_id")
        if corpus_id:
            try:
                corpus_db = self.ctx.get_corpus_db(corpus_id)
            except Exception:  # noqa: BLE001
                corpus_db = None
        corpus_docs: Dict[str, Dict[str, object]] = {}
        if corpus_db and required_doc_ids:
            try:
                with corpus_db.connect() as conn:
                    doc_list = list(required_doc_ids)
                    for chunk in self._chunked_list(doc_list, 200):
                        if not chunk:
                            continue
                        placeholders = ",".join(["?"] * len(chunk))
                        query = (
                            "SELECT doc_id, text, metadata_json FROM documents "
                            f"WHERE doc_id IN ({placeholders})"
                        )
                        rows = conn.execute(query, chunk).fetchall()
                        for row in rows:
                            doc_key = str(row["doc_id"] or "")
                            corpus_docs[doc_key] = {
                                "text": row["text"] or "",
                                "metadata_json": row["metadata_json"] or "",
                                "order_index": 0,
                            }
            except Exception:  # noqa: BLE001
                corpus_docs = {}
        for unit_id, doc_ids in doc_ids_by_unit.items():
            unit_key = str(unit_id)
            storage = result.setdefault(unit_key, {})
            for doc_id in doc_ids:
                doc_key = str(doc_id)
                if not doc_key or doc_key in storage:
                    continue
                if doc_key in corpus_docs:
                    storage[doc_key] = dict(corpus_docs[doc_key])
        return result

    def _populate_document_table(self, unit_row: Dict[str, object]) -> None:
        self.document_table.clearContents()
        self.document_table.setRowCount(0)
        self.document_preview.clear()
        unit_id = unit_row.get("unit_id")
        if not unit_id:
            return
        reviewer_ids = unit_row.get("reviewer_ids") or []
        assignment_db: Optional[Database] = None
        for reviewer_id in reviewer_ids:
            candidate = self.assignment_paths.get(reviewer_id)
            if not candidate:
                continue
            db = self.ctx.get_assignment_db(candidate)
            if db:
                assignment_db = db
                break
        if not assignment_db:
            return
        with assignment_db.connect() as conn:
            doc_rows = conn.execute(
                """
                SELECT unit_notes.doc_id, unit_notes.order_index, documents.text, documents.metadata_json
                FROM unit_notes
                LEFT JOIN documents ON documents.doc_id = unit_notes.doc_id
                WHERE unit_notes.unit_id=?
                ORDER BY unit_notes.order_index
                """,
                (unit_id,),
            ).fetchall()
        if not doc_rows:
            return
        metadata_by_doc: Dict[str, Dict[str, object]] = {}
        for row in doc_rows:
            doc_id = row["doc_id"]
            payload: Dict[str, object] = {}
            metadata_json = row.get("metadata_json") if hasattr(row, "get") else row["metadata_json"]
            if metadata_json:
                try:
                    parsed = json.loads(metadata_json)
                except Exception:  # noqa: BLE001
                    parsed = {}
                if isinstance(parsed, dict):
                    payload.update(parsed)
            metadata_by_doc[doc_id] = payload

        corpus_db: Optional[Database] = None
        corpus_id = (self.current_round or {}).get("corpus_id")
        if corpus_id:
            try:
                corpus_db = self.ctx.get_corpus_db(corpus_id)
            except Exception:
                corpus_db = None
        doc_ids = [row["doc_id"] for row in doc_rows]
        if corpus_db and doc_ids:
            columns = self._document_metadata_columns_cache.get(str(corpus_db.path))
            if columns is None:
                try:
                    with corpus_db.connect() as conn:
                        info_rows = conn.execute("PRAGMA table_info(documents)").fetchall()
                    columns = [
                        row["name"]
                        for row in info_rows
                        if row["name"].lower() not in {"doc_id", "text", "hash"}
                    ]
                except Exception:
                    columns = []
                self._document_metadata_columns_cache[str(corpus_db.path)] = columns
            if columns:
                with corpus_db.connect() as conn:
                    placeholders = ",".join(["?"] * len(doc_ids))
                    select_clause = ", ".join(columns)
                    rows = conn.execute(
                        f"SELECT doc_id, {select_clause} FROM documents WHERE doc_id IN ({placeholders})",
                        doc_ids,
                    ).fetchall()
                for row in rows:
                    payload = metadata_by_doc.setdefault(row["doc_id"], {})
                    for column in columns:
                        payload[column] = row[column]

        discovered_keys: List[str] = []
        for payload in metadata_by_doc.values():
            for key in payload.keys():
                if key not in discovered_keys:
                    discovered_keys.append(key)
        priority = [key for key in self.DOCUMENT_METADATA_PRIORITY if key in discovered_keys]
        remaining = sorted(
            [key for key in discovered_keys if key not in self.DOCUMENT_METADATA_PRIORITY],
            key=str.lower,
        )
        metadata_keys = priority + remaining

        headers = ["#", "Document ID"] + [self._metadata_header_for(key) for key in metadata_keys]
        column_labels = headers + ["Preview"]
        self.document_table.blockSignals(True)
        self.document_table.setSortingEnabled(False)
        self.document_table.clear()
        self.document_table.setColumnCount(len(column_labels))
        self.document_table.setHorizontalHeaderLabels(column_labels)
        self.document_table.setRowCount(len(doc_rows))
        preview_column = len(column_labels) - 1
        for idx, doc_row in enumerate(doc_rows):
            doc_id = doc_row["doc_id"]
            order_value = doc_row.get("order_index") if hasattr(doc_row, "get") else doc_row["order_index"]
            order_item = QtWidgets.QTableWidgetItem(str(order_value))
            order_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc_id)
            order_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.document_table.setItem(idx, 0, order_item)

            doc_item = QtWidgets.QTableWidgetItem(str(doc_id))
            doc_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc_id)
            doc_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.document_table.setItem(idx, 1, doc_item)

            payload = metadata_by_doc.get(doc_id, {})
            for offset, key in enumerate(metadata_keys, start=2):
                value = payload.get(key)
                display_value = self._format_metadata_value(key, value)
                item = QtWidgets.QTableWidgetItem(display_value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, doc_id)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.document_table.setItem(idx, offset, item)

            preview_raw = doc_row["text"] or ""
            preview_str = str(preview_raw)
            preview_text = preview_str[:200].replace("\n", " ")
            if len(preview_str) > 200:
                preview_text += "…"
            preview_item = QtWidgets.QTableWidgetItem(preview_text)
            preview_item.setData(QtCore.Qt.ItemDataRole.UserRole, doc_id)
            preview_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.document_table.setItem(idx, preview_column, preview_item)
        self.document_table.blockSignals(False)
        self.document_table.setSortingEnabled(True)
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
        assignment_db: Optional[Database] = None
        for reviewer_id in reviewer_ids:
            candidate = self.assignment_paths.get(reviewer_id)
            if not candidate:
                continue
            db = self.ctx.get_assignment_db(candidate)
            if db:
                assignment_db = db
                break
        if not assignment_db:
            self.document_preview.clear()
            return
        unit_id = self.unit_rows[unit_index].get("unit_id")
        if not unit_id:
            self.document_preview.clear()
            return
        doc_item = self.document_table.item(current_row, 1)
        doc_id = doc_item.data(QtCore.Qt.ItemDataRole.UserRole) if doc_item else None
        if not doc_id and doc_item:
            doc_id = doc_item.text()
        if not doc_id:
            self.document_preview.clear()
            return
        with assignment_db.connect() as conn:
            row = conn.execute(
                "SELECT text FROM documents WHERE doc_id=?",
                (doc_id,),
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
        self._active_discord_ids = set(discordant_ids)
        for row in self.unit_rows:
            row["discord"] = row["unit_id"] in discordant_ids
        self._display_unit_rows()
        self._update_cached_unit_rows()

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
        aggregate_db = self.ctx.get_round_aggregate_db(round_dir, create=False)
        if not aggregate_db and not aggregate_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "IAA",
                "Round aggregate not found. Import assignments and build the aggregate before calculating agreement.",
            )
            return
        if not aggregate_db:
            aggregate_db = self.ctx.get_round_aggregate_db(round_dir, create=True)
        if not aggregate_db:
            QtWidgets.QMessageBox.warning(
                self,
                "IAA",
                "Round aggregate could not be loaded.",
            )
            return
        self.round_manifest = self._load_manifest(round_dir)
        round_id = self.current_round["round_id"]
        with aggregate_db.connect() as conn:
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
            self._update_discord_state(label_id, set())
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
            self._update_discord_state(label_id, set())
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
        self._update_discord_state(label_id, filtered_discordant)
        self._scroll_to_first_discordant(filtered_discordant)

    @staticmethod
    def _compute_annotation_display(
        value: Optional[str],
        value_num: Optional[float],
        value_date: Optional[str],
        na: object,
    ) -> str:
        if na:
            return "N/A"
        if value_num is not None:
            return format(value_num, "g")
        if value is not None and value != "":
            return str(value)
        if value_date:
            return value_date
        return ""

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




class ReviewerLabelEditDialog(QtWidgets.QDialog):
    def __init__(
        self,
        definition: LabelDefinition,
        reviewer_name: str,
        unit_id: str,
        current_annotation: Dict[str, object],
        current_display: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.definition = definition
        self.setWindowTitle("Edit reviewer label")
        self.result_data: Optional[Dict[str, object]] = None
        self.value_combo: Optional[QtWidgets.QComboBox] = None
        self.value_line_edit: Optional[QtWidgets.QLineEdit] = None
        self.value_text_edit: Optional[QtWidgets.QPlainTextEdit] = None
        self.value_date_edit: Optional[QtWidgets.QDateEdit] = None
        self.multi_checkboxes: List[QtWidgets.QCheckBox] = []
        self.na_checkbox: Optional[QtWidgets.QCheckBox] = None

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            f"Reviewer: {reviewer_name}\nUnit: {unit_id or '—'}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)
        previous = QtWidgets.QLabel(f"Previous value: {current_display or '—'}")
        previous.setWordWrap(True)
        layout.addWidget(previous)

        self._build_value_controls(layout, current_annotation)

        rationale_label = QtWidgets.QLabel("Rationale for change:")
        layout.addWidget(rationale_label)
        self.rationale_edit = QtWidgets.QPlainTextEdit()
        self.rationale_edit.setPlaceholderText("Explain why this value should change")
        self.rationale_edit.setMinimumHeight(80)
        layout.addWidget(self.rationale_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_value_controls(
        self, layout: QtWidgets.QVBoxLayout, current_annotation: Dict[str, object]
    ) -> None:
        type_lower = self.definition.type.lower()
        current_value = (current_annotation.get("value") or "") if current_annotation else ""
        current_values = set(str(current_value).split(",")) if current_value else set()
        current_value_num = current_annotation.get("value_num")
        current_value_date = current_annotation.get("value_date")
        is_na = bool(current_annotation.get("na"))

        if type_lower in {"boolean", "categorical_single", "ordinal"}:
            combo = QtWidgets.QComboBox()
            combo.addItem("Select…", "")
            for option in sorted(self.definition.options, key=lambda opt: opt.get("order_index", 0)):
                combo.addItem(str(option.get("display")), option.get("value"))
            if current_value:
                index = combo.findData(current_value)
                if index >= 0:
                    combo.setCurrentIndex(index)
            self.value_combo = combo
            layout.addWidget(combo)
        elif type_lower == "categorical_multi":
            box = QtWidgets.QGroupBox("Select all that apply")
            box_layout = QtWidgets.QVBoxLayout(box)
            for option in sorted(self.definition.options, key=lambda opt: opt.get("order_index", 0)):
                checkbox = QtWidgets.QCheckBox(str(option.get("display")))
                checkbox.setProperty("option_value", option.get("value"))
                if option.get("value") in current_values:
                    checkbox.setChecked(True)
                self.multi_checkboxes.append(checkbox)
                box_layout.addWidget(checkbox)
            layout.addWidget(box)
        elif type_lower in {"integer", "float"}:
            line = QtWidgets.QLineEdit()
            if type_lower == "integer":
                validator = QtGui.QIntValidator()
            else:
                validator = QtGui.QDoubleValidator()
                validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
            line.setValidator(validator)
            if current_value:
                line.setText(str(current_value))
            elif current_value_num is not None:
                line.setText(str(current_value_num))
            self.value_line_edit = line
            layout.addWidget(line)
        elif type_lower == "date":
            date_edit = QtWidgets.QDateEdit()
            date_edit.setCalendarPopup(True)
            if current_value_date:
                parsed = QtCore.QDate.fromString(str(current_value_date), QtCore.Qt.DateFormat.ISODate)
                if parsed.isValid():
                    date_edit.setDate(parsed)
            self.value_date_edit = date_edit
            layout.addWidget(date_edit)
        else:
            text_edit = QtWidgets.QPlainTextEdit()
            text_edit.setPlainText(str(current_value))
            text_edit.setMinimumHeight(80)
            self.value_text_edit = text_edit
            layout.addWidget(text_edit)

        if self.definition.na_allowed:
            na_box = QtWidgets.QCheckBox("Mark as N/A")
            na_box.setChecked(is_na)
            na_box.stateChanged.connect(
                lambda state: self._set_value_widgets_enabled(state != QtCore.Qt.CheckState.Checked)
            )
            self.na_checkbox = na_box
            layout.addWidget(na_box)
            self._set_value_widgets_enabled(not is_na)

    def _set_value_widgets_enabled(self, enabled: bool) -> None:
        if self.value_combo is not None:
            self.value_combo.setEnabled(enabled)
        if self.value_line_edit is not None:
            self.value_line_edit.setEnabled(enabled)
        if self.value_text_edit is not None:
            self.value_text_edit.setEnabled(enabled)
        if self.value_date_edit is not None:
            self.value_date_edit.setEnabled(enabled)
        for checkbox in self.multi_checkboxes:
            checkbox.setEnabled(enabled)

    def accept(self) -> None:  # noqa: D401 - Qt override
        rationale = self.rationale_edit.toPlainText().strip()
        if not rationale:
            QtWidgets.QMessageBox.warning(
                self,
                "Edit reviewer label",
                "Please provide a rationale for the change.",
            )
            return
        is_na = bool(self.na_checkbox.isChecked()) if self.na_checkbox else False
        value: Optional[str]
        value_num: Optional[float]
        value_date: Optional[str]
        if is_na:
            value = None
            value_num = None
            value_date = None
        else:
            type_lower = self.definition.type.lower()
            if type_lower in {"boolean", "categorical_single", "ordinal"}:
                assert self.value_combo is not None
                selected = self.value_combo.currentData()
                if not selected:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Edit reviewer label",
                        "Select a value before saving.",
                    )
                    return
                value = str(selected)
                value_num = None
                value_date = None
            elif type_lower == "categorical_multi":
                selections = [
                    str(cb.property("option_value"))
                    for cb in self.multi_checkboxes
                    if cb.isChecked()
                ]
                if not selections:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Edit reviewer label",
                        "Select at least one option before saving.",
                    )
                    return
                value = ",".join(selections)
                value_num = None
                value_date = None
            elif type_lower in {"integer", "float"}:
                assert self.value_line_edit is not None
                text = self.value_line_edit.text().strip()
                if not text:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Edit reviewer label",
                        "Enter a numeric value before saving.",
                    )
                    return
                try:
                    if type_lower == "integer":
                        value_num = int(text)
                    else:
                        value_num = float(text)
                except ValueError:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Edit reviewer label",
                        "Enter a valid numeric value.",
                    )
                    return
                value = text
                value_date = None
            elif type_lower == "date":
                assert self.value_date_edit is not None
                value_date = self.value_date_edit.date().toString(QtCore.Qt.DateFormat.ISODate)
                value = None
                value_num = None
            else:
                assert self.value_text_edit is not None
                value = self.value_text_edit.toPlainText().strip()
                value_num = None
                value_date = None

        annotation_payload = {
            "value": value,
            "value_num": value_num,
            "value_date": value_date,
            "na": 1 if is_na else 0,
        }
        self.result_data = {
            "annotation": annotation_payload,
            "rationale": rationale,
        }
        super().accept()


class AdminMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ctx = ProjectContext()
        self.setWindowTitle("VAAnnotate Admin")
        self.resize(1280, 860)
        self._setup_menu()
        self._setup_central()
        self.ctx.dirty_changed.connect(self._on_dirty_changed)
        self._on_dirty_changed(False)

    def _setup_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        new_action = file_menu.addAction("Create new project…")
        new_action.triggered.connect(self._create_project)
        self.save_action = file_menu.addAction("Save project")
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._save_project)
        self.save_action.setEnabled(False)
        file_menu.addSeparator()
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

    def _on_dirty_changed(self, dirty: bool) -> None:
        if hasattr(self, "save_action"):
            self.save_action.setEnabled(dirty)
        title = "VAAnnotate Admin"
        if dirty:
            title += " • Unsaved changes"
        self.setWindowTitle(title)

    def _save_project(self) -> None:
        try:
            self.ctx.save_all()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save project", f"Failed to save project: {exc}")
            return
        QtWidgets.QMessageBox.information(self, "Save project", "Project saved.")

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
            corpus = data.get("corpus")
            self.corpus_view.set_corpus(corpus if isinstance(corpus, dict) else None)
            self._show_view("corpus")
        elif node_type == "iaa":
            pheno = data.get("pheno")
            self.iaa_view.set_phenotype(pheno if isinstance(pheno, dict) else None)
            self._show_view("iaa")
        else:
            self.project_view.set_project(self.ctx.project_row)
            self._show_view("project")

    def _create_project(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Create project", "Project name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Create project", "Project name is required.")
            return
        start_dir = str(self.ctx.project_root or Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select project folder",
            start_dir,
        )
        if not directory:
            return
        selected_dir = Path(directory).resolve()
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "project"
        project_dir = selected_dir
        if selected_dir.exists() and any(selected_dir.iterdir()):
            response = QtWidgets.QMessageBox.question(
                self,
                "Create project",
                (
                    "The selected directory is not empty. "
                    "Do you want to create the project inside a new subdirectory here?"
                ),
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if response != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            candidate = selected_dir / slug
            counter = 2
            while candidate.exists():
                candidate = selected_dir / f"{slug}_{counter}"
                counter += 1
            project_dir = candidate
        if project_dir.exists() and any(project_dir.iterdir()):
            QtWidgets.QMessageBox.warning(
                self,
                "Create project",
                f"The directory '{project_dir}' already exists and is not empty. Select a different location.",
            )
            return
        try:
            ensure_dir(project_dir)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Create project", f"Failed to prepare project directory: {exc}")
            return
        project_id = slug or f"project_{uuid.uuid4().hex[:8]}"
        try:
            paths = init_project(project_dir, project_id, name, "admin_app")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Create project", f"Failed to create project: {exc}")
            return
        self.ctx.open_project(paths.root)
        self.project_view.set_project(self.ctx.project_row)
        self.tree.refresh()
        QtWidgets.QMessageBox.information(
            self,
            "Create project",
            f"Project '{name}' created at {paths.root}.",
        )

    def _open_project(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select project folder")
        if not directory:
            return
        self.ctx.open_project(Path(directory))
        self.project_view.set_project(self.ctx.project_row)
        self.tree.refresh()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: D401 - Qt override
        if self.ctx.has_unsaved_changes():
            response = QtWidgets.QMessageBox.question(
                self,
                "Exit",
                "Save changes before exiting?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Save,
            )
            if response == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if response == QtWidgets.QMessageBox.StandardButton.Save:
                try:
                    self.ctx.save_all()
                except Exception as exc:  # noqa: BLE001
                    QtWidgets.QMessageBox.critical(self, "Save project", f"Failed to save project: {exc}")
                    event.ignore()
                    return
        super().closeEvent(event)


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)
    window = AdminMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
