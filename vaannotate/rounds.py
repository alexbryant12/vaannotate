"""Round creation, sampling, and aggregation."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Set

from .project import build_label_config, fetch_labelset
from .schema import initialize_assignment_db, initialize_round_aggregate_db
from .shared.metadata import (
    MetadataFilterCondition,
    MetadataField,
    discover_corpus_metadata,
    extract_document_metadata,
    normalize_date_value,
)
from .shared.sampling import SamplingFilters, candidate_documents
from .utils import (
    canonical_json,
    copy_sqlite_database,
    deterministic_choice,
    ensure_dir,
    stable_hash,
)


def _date_sort_value(value: object) -> tuple:
    if value is None:
        return ("", "")
    text = str(value)
    normalized = normalize_date_value(text)
    if normalized:
        return (normalized, text)
    return (text, text)


@dataclass
class CandidateUnit:
    unit_id: str
    patient_icn: str
    doc_id: str | None
    strata_key: str
    payload: dict


@dataclass
class AssignmentUnit:
    unit_id: str
    patient_icn: str
    doc_id: str | None
    payload: dict


class RoundBuilder:
    LLM_REVIEWER_ID = "llm"

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.project_db = project_root / "project.db"

    @staticmethod
    def _llm_reviewer_ids(config: Mapping[str, Any]) -> list[str]:
        reviewers = config.get("reviewers")
        if not isinstance(reviewers, Sequence):
            return []
        llm_ids: list[str] = []
        for reviewer in reviewers:
            if not isinstance(reviewer, Mapping):
                continue
            raw_id = reviewer.get("id") or reviewer.get("reviewer_id")
            if raw_id is None:
                continue
            reviewer_id = str(raw_id)
            if reviewer_id and reviewer_id.lower() == RoundBuilder.LLM_REVIEWER_ID:
                llm_ids.append(reviewer_id)
        return llm_ids

    def _extract_llm_predictions(
        self,
        final_llm_outputs: Mapping[str, str],
        base_dir: Path,
    ) -> dict[str, dict[str, object]]:
        labels_path_value = final_llm_outputs.get("final_llm_labels_json")
        if not labels_path_value:
            return {}
        try:
            labels_path = Path(str(labels_path_value))
        except Exception:  # noqa: BLE001
            return {}
        if not labels_path.is_absolute():
            labels_path = (base_dir / labels_path).resolve()
        if not labels_path.exists():
            return {}
        try:
            payload = json.loads(labels_path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, list):
            return {}
        predictions: dict[str, dict[str, object]] = {}
        for row in payload:
            if not isinstance(row, Mapping):
                continue
            unit_value = row.get("unit_id")
            if unit_value is None:
                continue
            unit_id = str(unit_value)
            if not unit_id:
                continue
            unit_map = predictions.setdefault(unit_id, {})
            for key, value in row.items():
                if not isinstance(key, str) or not key.endswith("_llm"):
                    continue
                label_id = key[: -len("_llm")]
                if not label_id:
                    continue
                unit_map[label_id] = value
        return predictions

    @staticmethod
    def _normalize_annotation_value(
        label_info: Mapping[str, Any],
        raw_value: object,
    ) -> tuple[Optional[str], Optional[float], Optional[str], int]:
        label_type = str(label_info.get("type") or "").lower()
        na_allowed = bool(label_info.get("na_allowed"))
        if raw_value is None:
            return (None, None, None, 1 if na_allowed else 0)
        if label_type in {"integer", "float"}:
            text = str(raw_value).strip()
            if not text:
                return (None, None, None, 1 if na_allowed else 0)
            try:
                numeric = float(text)
            except ValueError:
                numeric = None
            return (text, numeric, None, 0)
        if label_type == "date":
            text = str(raw_value).strip()
            if not text:
                return (None, None, None, 1 if na_allowed else 0)
            return (None, None, text, 0)
        if label_type == "categorical_multi":
            entries: list[str]
            if isinstance(raw_value, (list, tuple, set)):
                entries = [
                    str(item).strip()
                    for item in raw_value
                    if item not in (None, "")
                ]
            else:
                text = str(raw_value)
                entries = [part.strip() for part in text.split(",") if part.strip()]
            if not entries:
                return (None, None, None, 1 if na_allowed else 0)
            joined = ",".join(entries)
            return (joined, None, None, 0)
        text_value = str(raw_value).strip()
        if not text_value:
            if na_allowed:
                return (None, None, None, 1)
            return ("", None, None, 0)
        return (text_value, None, None, 0)

    def _auto_submit_llm_assignments(
        self,
        *,
        round_id: str,
        round_dir: Path,
        reviewer_assignments: Mapping[str, Sequence[AssignmentUnit]],
        label_schema: Mapping[str, Any] | None,
        config: Mapping[str, Any],
        final_llm_outputs: Mapping[str, str],
        project_conn: sqlite3.Connection | None = None,
    ) -> None:
        llm_ids = self._llm_reviewer_ids(config)
        if not llm_ids:
            return
        predictions = self._extract_llm_predictions(final_llm_outputs, round_dir)
        if not predictions:
            raise RuntimeError("Final LLM labeling outputs are missing LLM predictions")
        labels = []
        if isinstance(label_schema, Mapping):
            raw_labels = label_schema.get("labels")
            if isinstance(raw_labels, Sequence):
                labels = [label for label in raw_labels if isinstance(label, Mapping)]
        if not labels:
            raise RuntimeError("Label schema is required to populate LLM assignments")
        label_lookup = {str(label.get("label_id")): label for label in labels if label.get("label_id") is not None}
        if not label_lookup:
            raise RuntimeError("Label schema did not contain label identifiers")
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
            missing_predictions = [unit_id for unit_id in unit_ids if unit_id not in predictions]
            if missing_predictions:
                logging.getLogger(__name__).warning(
                    "Skipping %d unit(s) missing LLM predictions for reviewer %s: %s",
                    len(missing_predictions),
                    reviewer_id,
                    ", ".join(sorted(missing_predictions)),
                )
            unit_ids = [unit_id for unit_id in unit_ids if unit_id in predictions]
            if not unit_ids:
                continue
            assign_dir = round_dir / "assignments" / reviewer_id
            db_path = assign_dir / "assignment.db"
            if not db_path.exists():
                raise RuntimeError(f"Assignment database not found for reviewer {reviewer_id}")
            annotation_rows: list[tuple[str, str, Optional[str], Optional[float], Optional[str], int, Optional[str]]] = []
            for unit_id in unit_ids:
                label_values = predictions.get(unit_id, {})
                for label_id, label_info in label_lookup.items():
                    raw_value = label_values.get(label_id)
                    value, value_num, value_date, na_flag = self._normalize_annotation_value(label_info, raw_value)
                    annotation_rows.append(
                        (
                            unit_id,
                            label_id,
                            value,
                            value_num,
                            value_date,
                            na_flag,
                            None,
                        )
                    )
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.executemany(
                    "DELETE FROM annotations WHERE unit_id=?",
                    [(unit_id,) for unit_id in unit_ids],
                )
                if annotation_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO annotations(unit_id, label_id, value, value_num, value_date, na, notes)
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        annotation_rows,
                    )
                conn.executemany(
                    "UPDATE units SET complete=1, completed_at=? WHERE unit_id=?",
                    [(timestamp, unit_id) for unit_id in unit_ids],
                )
                conn.commit()
            receipt = {
                "unit_count": len(unit_ids),
                "completed": len(unit_ids),
                "submitted_at": timestamp,
            }
            receipt_path = assign_dir / "submitted.json"
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        round_key = str(round_id or "")
        if not round_key:
            return
        updater: sqlite3.Connection | None = project_conn
        if updater is None:
            updater = self._connect_project()
        try:
            for reviewer_id in llm_ids:
                updater.execute(
                    "UPDATE assignments SET status='submitted' WHERE round_id=? AND reviewer_id=?",
                    (round_key, reviewer_id),
                )
        finally:
            if project_conn is None and updater is not None:
                updater.commit()
                updater.close()

    def _connect_project(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.project_db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _connect_corpus(self, corpus_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(corpus_path)
        conn.row_factory = sqlite3.Row
        return conn

    def generate_round(
        self,
        pheno_id: str,
        config_path: Path,
        created_by: str,
        preselected_units_csv: Path | str | None = None,
        env_overrides: Mapping[str, str] | None = None,
    ) -> dict:
        config_path = Path(config_path)
        config = json.loads(config_path.read_text("utf-8"))
        config_base = config_path.parent
        ai_backend_config = config.get("ai_backend") if isinstance(config.get("ai_backend"), Mapping) else {}
        final_llm_flag = config.get("final_llm_labeling")
        if final_llm_flag is None:
            final_llm_enabled = bool(ai_backend_config.get("final_llm_labels")) or bool(
                ai_backend_config.get("final_llm_labels_json")
            )
        else:
            final_llm_enabled = bool(final_llm_flag)
        config["final_llm_labeling"] = final_llm_enabled
        include_reasoning = self._final_llm_include_reasoning(config)
        config["final_llm_include_reasoning"] = include_reasoning
        final_llm_outputs: Dict[str, str] = {}
        overrides = {
            str(key): str(value)
            for key, value in (env_overrides or {}).items()
            if value is not None and str(value)
        }
        previous_env: Dict[str, Optional[str]] = {}
        for key, value in overrides.items():
            previous_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            with self._connect_project() as project_conn:
                pheno = project_conn.execute(
                    "SELECT * FROM phenotypes WHERE pheno_id=?",
                    (pheno_id,),
                ).fetchone()
                if not pheno:
                    raise ValueError(f"Phenotype {pheno_id} not found")
                storage_path = pheno["storage_path"]
                if not storage_path:
                    raise ValueError("Phenotype storage path is not defined")
                storage_dir = Path(storage_path)
                if not storage_dir.is_absolute():
                    storage_dir = (self.project_root / storage_dir).resolve()
                corpus_id = config.get("corpus_id")
                corpus_row = None
                if corpus_id:
                    corpus_row = project_conn.execute(
                        "SELECT * FROM project_corpora WHERE corpus_id=?",
                        (corpus_id,),
                    ).fetchone()
                if corpus_row:
                    corpus_path = self.project_root / corpus_row["relative_path"]
                else:
                    legacy_path = config.get("corpus_path")
                    if not legacy_path:
                        raise ValueError("Round configuration does not specify a corpus")
                    corpus_path = Path(legacy_path)
                    if not corpus_path.is_absolute():
                        corpus_path = (self.project_root / corpus_path).resolve()
                with self._connect_corpus(corpus_path) as corpus_conn:
                    labelset = fetch_labelset(project_conn, config["labelset_id"])
                    round_number = config.get("round_number")
                    csv_override: Path | None = None
                    if not preselected_units_csv:
                        csv_value = config.get("preselected_units_csv")
                        if csv_value:
                            csv_override = Path(csv_value)
                            if not csv_override.is_absolute():
                                csv_override = (config_base / csv_override).resolve()
                    else:
                        csv_override = Path(preselected_units_csv)
                    round_id = config.get("round_id") or f"{pheno_id}_r{round_number}"
                    status = str(config.get("status") or "active")
                    if csv_override and not csv_override.exists():
                        raise FileNotFoundError(csv_override)
                rng_seed = config.get("rng_seed", 0)
                config_hash = stable_hash(canonical_json(config))
                phenotype_dir = storage_dir
                round_dir = phenotype_dir / "rounds" / f"round_{round_number}"
                ensure_dir(round_dir)
                ensure_dir(round_dir / "assignments")
                ensure_dir(round_dir / "imports")
                ensure_dir(round_dir / "reports" / "confusion_matrices")
                ensure_dir(round_dir / "reports" / "exports")
    
                if csv_override:
                    candidates = list(
                        self._build_preselected_candidates(corpus_conn, pheno, csv_override)
                    )
                else:
                    candidates = list(self._build_candidates(corpus_conn, pheno, config))
                if not candidates:
                    raise ValueError("No candidates matched filters")
    
                reviewer_ids = [reviewer["id"] for reviewer in config["reviewers"]]
                reviewer_assignments = self._allocate_units(
                    candidates,
                    reviewer_ids,
                    rng_seed,
                    config.get("overlap_n", 0),
                    config.get("total_n"),
                    config.get("stratification"),
                    preserve_input_order=bool(csv_override),
                )
    
                if csv_override:
                    config["preselected_units_csv"] = str(csv_override)
                config["status"] = status
    
                manifest_path = round_dir / "manifest.csv"
                with manifest_path.open("w", encoding="utf-8", newline="") as fh:
                    fieldnames = ["unit_id", "patient_icn", "doc_id", "strata_key", "assigned_to", "is_overlap"]
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    for reviewer_id, assignments in reviewer_assignments.items():
                        for unit in assignments:
                            writer.writerow(
                                {
                                    "unit_id": unit.unit_id,
                                    "patient_icn": unit.patient_icn,
                                    "doc_id": unit.doc_id,
                                    "strata_key": unit.payload["strata_key"],
                                    "assigned_to": reviewer_id,
                                    "is_overlap": 1 if unit.payload.get("is_overlap") else 0,
                                }
                            )
    
                (round_dir / "round_config.json").write_text(canonical_json(config), encoding="utf-8")
    
                project_conn.execute(
                    """
                    INSERT OR REPLACE INTO rounds(round_id, pheno_id, round_number, labelset_id, config_hash, rng_seed, status, created_at)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        round_id,
                        pheno_id,
                        round_number,
                        labelset["labelset_id"],
                        config_hash,
                        rng_seed,
                        status,
                        datetime.utcnow().isoformat(),
                    ),
                )
                project_conn.execute(
                    "INSERT OR REPLACE INTO round_configs(round_id, config_json) VALUES (?, ?)",
                    (round_id, canonical_json(config)),
                )
    
                label_schema_payload = self._build_label_schema_payload(labelset)
                label_schema_text = json.dumps(label_schema_payload, indent=2)
    
                for reviewer in config["reviewers"]:
                    assign_dir = ensure_dir(round_dir / "assignments" / reviewer["id"])
                    assignment_db = assign_dir / "assignment.db"
                    with initialize_assignment_db(assignment_db) as assign_conn:
                        units = reviewer_assignments[reviewer["id"]]
                        for display_rank, unit in enumerate(units):
                            assign_conn.execute(
                                "INSERT OR REPLACE INTO units(unit_id, display_rank, patient_icn, doc_id, note_count) VALUES (?,?,?,?,?)",
                                (
                                    unit.unit_id,
                                    display_rank,
                                    unit.patient_icn,
                                    unit.doc_id,
                                    unit.payload.get("note_count"),
                                ),
                            )
                            documents = unit.payload.get("documents", [])
                            assign_conn.execute(
                                "DELETE FROM unit_notes WHERE unit_id=?",
                                (unit.unit_id,),
                            )
                            for order_index, doc in enumerate(documents):
                                doc_id = doc.get("doc_id")
                                if not doc_id:
                                    continue
                                metadata = extract_document_metadata(doc)
                                metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None
                                assign_conn.execute(
                                    "INSERT OR REPLACE INTO documents(doc_id, hash, text, metadata_json) VALUES (?,?,?,?)",
                                    (
                                        doc_id,
                                        doc.get("hash", ""),
                                        doc.get("text", ""),
                                        metadata_json,
                                    ),
                                )
                                assign_conn.execute(
                                    "INSERT OR REPLACE INTO unit_notes(unit_id, doc_id, order_index) VALUES (?,?,?)",
                                    (unit.unit_id, doc_id, order_index),
                                )
                        for label in labelset["labels"]:
                            for unit in units:
                                assign_conn.execute(
                                    "INSERT OR IGNORE INTO annotations(unit_id, label_id, na) VALUES (?, ?, 0)",
                                    (unit.unit_id, label["label_id"]),
                                )
                    (assign_dir / "client.exe").write_text("Annotator client placeholder", encoding="utf-8")
                    (assign_dir / "assignment.lock").write_text("", encoding="utf-8")
                    ensure_dir(assign_dir / "logs")
                    (assign_dir / "label_schema.json").write_text(
                        label_schema_text,
                        encoding="utf-8",
                    )
                    project_conn.execute(
                        """
                        INSERT OR REPLACE INTO assignments(assign_id, round_id, reviewer_id, sample_size, overlap_n, created_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"{round_id}_{reviewer['id']}",
                            round_id,
                            reviewer["id"],
                            len(reviewer_assignments[reviewer["id"]]),
                            config.get("overlap_n", 0),
                            datetime.utcnow().isoformat(),
                            "open",
                        ),
                    )
    
                if final_llm_enabled:
                    try:
                        outputs = self._apply_final_llm_labeling(
                            pheno_row=pheno,
                            labelset=labelset,
                            round_dir=round_dir,
                            reviewer_assignments=reviewer_assignments,
                            config=config,
                            config_base=config_base,
                            include_reasoning=include_reasoning,
                        )
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(f"Final LLM labeling failed: {exc}") from exc
                    else:
                        final_llm_outputs.update(outputs)
                        config.setdefault("final_llm_outputs", {}).update(outputs)
                        (round_dir / "round_config.json").write_text(
                            canonical_json(config),
                            encoding="utf-8",
                        )
                        try:
                            self._auto_submit_llm_assignments(
                                round_id=round_id,
                                round_dir=round_dir,
                                reviewer_assignments=reviewer_assignments,
                                label_schema=label_schema_payload,
                                config=config,
                                final_llm_outputs=outputs,
                                project_conn=project_conn,
                            )
                        except Exception as autop_exc:  # noqa: BLE001
                            raise RuntimeError(
                                f"Failed to populate LLM reviewer assignment: {autop_exc}"
                            ) from autop_exc
    
                assisted_result: Dict[str, Any] = {}
                assisted_cfg = config.get("assisted_review") if isinstance(config.get("assisted_review"), Mapping) else None
                if isinstance(assisted_cfg, Mapping) and assisted_cfg.get("enabled"):
                    raw_top = assisted_cfg.get("top_snippets")
                    top_snippets = 0
                    try:
                        top_snippets = int(raw_top)
                    except (TypeError, ValueError):
                        top_snippets = 0
                    if top_snippets > 0:
                        assisted_data = self._generate_assisted_review_snippets(
                            pheno_row=pheno,
                            labelset=labelset,
                            round_dir=round_dir,
                            reviewer_assignments=reviewer_assignments,
                            config=config,
                            config_base=config_base,
                            top_n=top_snippets,
                        )
                        if assisted_data:
                            assist_dir = ensure_dir(round_dir / "reports" / "assisted_review")
                            assist_path = assist_dir / "snippets.json"
                            assist_path.write_text(
                                self._json_dumps(assisted_data),
                                encoding="utf-8",
                            )
                            try:
                                relative_assist_path = assist_path.relative_to(round_dir)
                            except ValueError:
                                try:
                                    relative_assist_path = assist_path.resolve().relative_to(
                                        round_dir.resolve()
                                    )
                                except ValueError:
                                    relative_assist_path = assist_path
                            assist_path_value = str(relative_assist_path)
                            updated_cfg = config.setdefault("assisted_review", {})
                            updated_cfg["enabled"] = True
                            updated_cfg["top_snippets"] = top_snippets
                            updated_cfg["generated_at"] = assisted_data.get("generated_at")
                            updated_cfg["snippets_json"] = assist_path_value
                            (round_dir / "round_config.json").write_text(
                                canonical_json(config),
                                encoding="utf-8",
                            )
                            project_conn.execute(
                                "INSERT OR REPLACE INTO round_configs(round_id, config_json) VALUES (?, ?)",
                                (round_id, canonical_json(config)),
                            )
                            assisted_result = {"snippets_json": assist_path_value}
    
                project_conn.commit()
                result_payload: Dict[str, Any] = {
                    "round_id": round_id,
                    "round_dir": str(round_dir),
                    "assignment_counts": {rid: len(units) for rid, units in reviewer_assignments.items()},
                }
                if final_llm_outputs:
                    result_payload["final_llm_outputs"] = dict(final_llm_outputs)
                if assisted_result:
                    result_payload.setdefault("assisted_review", {}).update(assisted_result)
                return result_payload
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _apply_final_llm_labeling(
        self,
        *,
        pheno_row: sqlite3.Row,
        labelset: Mapping[str, object],
        round_dir: Path,
        reviewer_assignments: Mapping[str, Sequence[AssignmentUnit]],
        config: Mapping[str, Any],
        config_base: Path,
        include_reasoning: bool,
    ) -> Dict[str, str]:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("pandas is required for final LLM labeling") from exc

        ai_backend_config = config.get("ai_backend") if isinstance(config.get("ai_backend"), Mapping) else {}

        labels_df = None
        probe_df = None

        labels_path = self._resolve_optional_path(ai_backend_config.get("final_llm_labels"), config_base)
        if labels_path and labels_path.exists():
            labels_df = pd.read_parquet(labels_path)
        else:
            labels_json_path = self._resolve_optional_path(
                ai_backend_config.get("final_llm_labels_json"), config_base
            )
            if labels_json_path and labels_json_path.exists():
                labels_df = pd.read_json(labels_json_path)

        probe_path = self._resolve_optional_path(ai_backend_config.get("final_llm_family_probe"), config_base)
        if probe_path and probe_path.exists():
            probe_df = pd.read_parquet(probe_path)
        else:
            probe_json_path = self._resolve_optional_path(
                ai_backend_config.get("final_llm_family_probe_json"), config_base
            )
            if probe_json_path and probe_json_path.exists():
                probe_df = pd.read_json(probe_json_path)

        if labels_df is None or probe_df is None:
            labels_df, probe_df = self._run_final_llm_labeling_inference(
                pheno_row=pheno_row,
                labelset=labelset,
                round_dir=round_dir,
                reviewer_assignments=reviewer_assignments,
                config=config,
                include_reasoning=include_reasoning,
            )

        if not include_reasoning:
            if labels_df is not None:
                drop_cols = [col for col in labels_df.columns if str(col).endswith("_llm_reason")]
                if drop_cols:
                    labels_df = labels_df.drop(columns=drop_cols)
            if probe_df is not None and "llm_reasoning" in probe_df.columns:
                probe_df = probe_df.drop(columns=["llm_reasoning"])

        exports_dir = ensure_dir(Path(round_dir) / "reports" / "exports")
        return self._write_final_llm_outputs(labels_df=labels_df, probe_df=probe_df, exports_dir=exports_dir)

    def _generate_assisted_review_snippets(
        self,
        *,
        pheno_row: sqlite3.Row,
        labelset: Mapping[str, object],
        round_dir: Path,
        reviewer_assignments: Mapping[str, Sequence[AssignmentUnit]],
        config: Mapping[str, Any],
        config_base: Path,
        top_n: int,
    ) -> dict[str, object]:
        if top_n <= 0:
            return {}
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("pandas is required for assisted chart review") from exc
        try:
            from vaannotate.vaannotate_ai_backend.engine import (
                ActiveLearningLLMFirst,
                FamilyLabeler,
                OrchestratorConfig,
                Paths,
                _contexts_for_unit_label,
            )
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("AI backend components are required for assisted chart review") from exc

        units_by_id, notes_rows = self._collect_round_units_and_notes(reviewer_assignments)
        if not units_by_id:
            return {}
        if not notes_rows:
            return {}

        work_dir = ensure_dir(Path(round_dir) / "imports" / "assisted_review")
        notes_path = work_dir / "notes.parquet"
        ann_path = work_dir / "annotations.parquet"
        notes_df = pd.DataFrame(notes_rows)
        notes_df["patient_icn"] = notes_df["patient_icn"].astype(str)
        notes_df["doc_id"] = notes_df["doc_id"].astype(str)
        notes_df["text"] = notes_df["text"].astype(str)
        notes_df["unit_id"] = notes_df["unit_id"].astype(str)
        notes_df.to_parquet(notes_path, index=False)
        ann_df = pd.DataFrame(
            {
                "round_id": pd.Series(dtype="object"),
                "unit_id": pd.Series(dtype="object"),
                "doc_id": pd.Series(dtype="object"),
                "label_id": pd.Series(dtype="object"),
                "reviewer_id": pd.Series(dtype="object"),
                "label_value": pd.Series(dtype="object"),
                "label_value_num": pd.Series(dtype="float64"),
                "label_value_date": pd.Series(dtype="datetime64[ns]"),
            }
        )
        ann_df.to_parquet(ann_path, index=False)

        cfg = OrchestratorConfig()
        cfg.final_llm_labeling = False
        phenotype_level = str(pheno_row["level"] or "multi_doc")
        label_config_payload = build_label_config(labelset)
        label_keywords = self._extract_label_keywords(config)
        if label_keywords:
            label_config_payload = self._apply_label_keywords(label_config_payload, label_keywords)
        paths = Paths(str(notes_path), str(ann_path), str(work_dir / "engine_outputs"))
        few_shot_examples = self._extract_few_shot_examples(config, labelset)
        if few_shot_examples:
            try:
                setattr(cfg.llm, "few_shot_examples", few_shot_examples)
            except Exception:  # noqa: BLE001
                pass

        ai_backend_config = config.get("ai_backend") if isinstance(config.get("ai_backend"), Mapping) else {}
        llmfirst_overrides = (
            ai_backend_config.get("llmfirst")
            if isinstance(ai_backend_config.get("llmfirst"), Mapping)
            else {}
        )
        mode_override = llmfirst_overrides.get("single_doc_context")
        if mode_override:
            setattr(cfg.llmfirst, "single_doc_context", str(mode_override))
        limit_override = llmfirst_overrides.get("single_doc_full_context_max_chars")
        if limit_override is not None:
            try:
                limit_value = int(limit_override)
            except (TypeError, ValueError):
                limit_value = None
            if limit_value and limit_value > 0:
                setattr(cfg.llmfirst, "single_doc_full_context_max_chars", limit_value)
        env_overrides: Dict[str, str] = {}
        backend_choice = str(ai_backend_config.get("backend") or "").strip().lower()
        if not backend_choice:
            if self._resolve_optional_path(ai_backend_config.get("local_model_dir"), config_base):
                backend_choice = "exllamav2"
            elif any(
                str(ai_backend_config.get(key) or "").strip()
                for key in ("azure_endpoint", "azure_api_version")
            ):
                backend_choice = "azure"
        backend_env_value: str | None = None
        if backend_choice == "azure":
            backend_env_value = "azure"
        elif backend_choice in {"exllamav2", "exllama", "local"}:
            backend_env_value = "exllamav2" if backend_choice == "local" else backend_choice
            local_model_path = self._resolve_optional_path(
                ai_backend_config.get("local_model_dir"), config_base
            )
            if local_model_path:
                env_overrides["LOCAL_LLM_MODEL_DIR"] = str(local_model_path)
            local_max_seq = ai_backend_config.get("local_max_seq_len")
            if local_max_seq is not None:
                try:
                    local_max_seq_value = int(local_max_seq)
                except (TypeError, ValueError):
                    local_max_seq_value = None
                if local_max_seq_value and local_max_seq_value > 0:
                    env_overrides["LOCAL_LLM_MAX_SEQ_LEN"] = str(local_max_seq_value)
            local_max_new = ai_backend_config.get("local_max_new_tokens")
            if local_max_new is not None:
                try:
                    local_max_new_value = int(local_max_new)
                except (TypeError, ValueError):
                    local_max_new_value = None
                if local_max_new_value and local_max_new_value > 0:
                    env_overrides["LOCAL_LLM_MAX_NEW_TOKENS"] = str(local_max_new_value)
        if backend_env_value:
            env_overrides["LLM_BACKEND"] = backend_env_value
        embed_path = self._resolve_optional_path(ai_backend_config.get("embedding_model_dir"), config_base)
        if embed_path:
            env_overrides["MED_EMBED_MODEL_NAME"] = str(embed_path)
        rerank_path = self._resolve_optional_path(ai_backend_config.get("reranker_model_dir"), config_base)
        if rerank_path:
            env_overrides["RERANKER_MODEL_NAME"] = str(rerank_path)
        if backend_env_value == "azure":
            azure_version = ai_backend_config.get("azure_api_version")
            if azure_version:
                env_overrides["AZURE_OPENAI_API_VERSION"] = str(azure_version)
            azure_endpoint = ai_backend_config.get("azure_endpoint")
            if azure_endpoint:
                env_overrides["AZURE_OPENAI_ENDPOINT"] = str(azure_endpoint)

        previous_env: Dict[str, Optional[str]] = {}
        for key, value in env_overrides.items():
            previous_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            orchestrator = ActiveLearningLLMFirst(
                paths=paths,
                cfg=cfg,
                label_config=label_config_payload,
                phenotype_level=phenotype_level,
            )
            orchestrator.store.build_chunk_index(
                orchestrator.repo.notes,
                orchestrator.cfg.rag,
                orchestrator.cfg.index,
            )
            orchestrator.ensure_llm_backend()
            _, _, current_rules_map, current_label_types = orchestrator._label_maps()
            family_labeler = FamilyLabeler(
                orchestrator.llm,
                orchestrator.rag,
                orchestrator.repo,
                orchestrator.label_config,
                orchestrator.cfg.scjitter,
                orchestrator.cfg.llmfirst,
            )
            try:
                family_labeler.ensure_label_exemplars(current_rules_map, K=max(1, top_n))
            except Exception:  # noqa: BLE001 - exemplar generation best effort
                logging.getLogger(__name__).info("Assisted review falling back to label rules for exemplars")

            unit_snippets: Dict[str, Dict[str, list[dict[str, Any]]]] = {}
            for unit_id in sorted(units_by_id.keys()):
                unit_map: Dict[str, list[dict[str, Any]]] = {}
                for label_id in sorted(current_label_types.keys()):
                    rules_text = current_rules_map.get(label_id, "")
                    contexts = _contexts_for_unit_label(
                        orchestrator.rag,
                        orchestrator.repo,
                        unit_id,
                        label_id,
                        rules_text,
                        topk_override=top_n,
                        single_doc_context_mode=getattr(orchestrator.cfg.llmfirst, "single_doc_context", "rag"),
                        full_doc_char_limit=getattr(orchestrator.cfg.llmfirst, "single_doc_full_context_max_chars", None),
                    )
                    if not contexts:
                        continue
                    entries: list[dict[str, Any]] = []
                    for ctx_entry in contexts[:top_n]:
                        metadata_value = ctx_entry.get("metadata") if isinstance(ctx_entry, Mapping) else {}
                        metadata: Dict[str, Any]
                        if isinstance(metadata_value, Mapping):
                            metadata = {
                                str(key): self._normalize_for_json(value)
                                for key, value in metadata_value.items()
                                if value is not None
                            }
                        else:
                            metadata = {}
                        doc_id = ctx_entry.get("doc_id")
                        chunk_id = ctx_entry.get("chunk_id")
                        try:
                            chunk_numeric = int(chunk_id) if chunk_id is not None else None
                        except (TypeError, ValueError):
                            chunk_numeric = None
                        try:
                            score_value = float(ctx_entry.get("score"))
                        except (TypeError, ValueError):
                            score_value = None
                        entry: Dict[str, Any] = {
                            "doc_id": str(doc_id) if doc_id is not None else "",
                            "chunk_id": chunk_numeric,
                            "score": score_value,
                            "source": str(ctx_entry.get("source") or ""),
                            "text": str(ctx_entry.get("text") or ""),
                            "metadata": metadata,
                        }
                        patient_icn = units_by_id[unit_id].patient_icn
                        if patient_icn:
                            entry["patient_icn"] = str(patient_icn)
                        entries.append(entry)
                    if entries:
                        unit_map[str(label_id)] = entries
                if unit_map:
                    unit_snippets[str(unit_id)] = unit_map
        finally:
            for key, previous in previous_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

        if not unit_snippets:
            return {}
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "top_snippets": int(top_n),
            "unit_snippets": unit_snippets,
        }

    def run_final_llm_labeling(
        self,
        *,
        pheno_row: sqlite3.Row,
        labelset: Mapping[str, object],
        round_dir: Path,
        reviewer_assignments: Mapping[str, Sequence[AssignmentUnit]],
        config: Mapping[str, Any],
        config_base: Path,
        log_callback: callable | None = None,
        env_overrides: Mapping[str, str] | None = None,
        auto_submit_llm: bool = True,
    ) -> Dict[str, str]:
        """Execute final LLM labeling while forwarding progress logs."""

        handler: logging.Handler | None = None
        logger = None
        original_level: int | None = None
        if log_callback:
            try:
                from vaannotate.vaannotate_ai_backend import engine
            except ImportError:  # pragma: no cover - runtime guard
                handler = None
            else:
                class _ForwardHandler(logging.Handler):
                    def __init__(self, callback: callable) -> None:
                        super().__init__(level=logging.INFO)
                        self._callback = callback

                    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
                        try:
                            message = record.getMessage()
                        except Exception:  # noqa: BLE001
                            message = str(record.msg)
                        if not message:
                            return
                        try:
                            self._callback(message)
                        except Exception:  # noqa: BLE001
                            pass

                logger = engine.LOGGER
                original_level = logger.level
                if original_level is None or original_level > logging.INFO:
                    logger.setLevel(logging.INFO)
                handler = _ForwardHandler(log_callback)
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

        overrides = {
            str(key): str(value)
            for key, value in (env_overrides or {}).items()
            if value is not None and str(value)
        }
        previous_env: Dict[str, Optional[str]] = {}
        for key, value in overrides.items():
            previous_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            try:
                outputs = self._apply_final_llm_labeling(
                    pheno_row=pheno_row,
                    labelset=labelset,
                    round_dir=round_dir,
                    reviewer_assignments=reviewer_assignments,
                    config=config,
                    config_base=config_base,
                    include_reasoning=self._final_llm_include_reasoning(config),
                )
                if auto_submit_llm and outputs:
                    try:
                        label_schema = self._build_label_schema_payload(labelset)
                        self._auto_submit_llm_assignments(
                            round_id=str(config.get("round_id") or ""),
                            round_dir=round_dir,
                            reviewer_assignments=reviewer_assignments,
                            label_schema=label_schema,
                            config=config,
                            final_llm_outputs=outputs,
                        )
                    except Exception as autop_exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"Failed to populate LLM reviewer assignment: {autop_exc}"
                        ) from autop_exc
                return outputs
            finally:
                if handler and logger:
                    logger.removeHandler(handler)
                    if original_level is not None:
                        logger.setLevel(original_level)
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _collect_round_units_and_notes(
        self, reviewer_assignments: Mapping[str, Sequence[AssignmentUnit]]
    ) -> tuple[Dict[str, AssignmentUnit], list[dict[str, Any]]]:
        units_by_id: Dict[str, AssignmentUnit] = {}
        for assignments in reviewer_assignments.values():
            for unit in assignments:
                units_by_id.setdefault(unit.unit_id, unit)

        notes_rows: list[dict[str, Any]] = []
        for unit in units_by_id.values():
            documents = unit.payload.get("documents") if isinstance(unit.payload, Mapping) else None
            docs_list = list(documents) if isinstance(documents, (list, tuple)) else []
            if not docs_list:
                doc_identifier = unit.doc_id or f"{unit.unit_id}_doc"
                notes_rows.append(
                    {
                        "patient_icn": str(unit.patient_icn or ""),
                        "doc_id": str(doc_identifier or unit.unit_id),
                        "text": str(unit.payload.get("text", "") if isinstance(unit.payload, Mapping) else ""),
                        "unit_id": str(unit.unit_id),
                    }
                )
                continue
            for index, doc in enumerate(docs_list):
                if not isinstance(doc, Mapping):
                    continue
                doc_id = doc.get("doc_id") or f"{unit.unit_id}_{index}"
                notes_rows.append(
                    {
                        "patient_icn": str(unit.patient_icn or doc.get("patient_icn") or ""),
                        "doc_id": str(doc_id),
                        "text": str(doc.get("text", "")),
                        "unit_id": str(unit.unit_id),
                        "order_index": index,
                        "metadata_json": doc.get("metadata_json"),
                        "date_note": doc.get("date_note"),
                    }
                )
        return units_by_id, notes_rows

    def _resolve_optional_path(self, raw: Any, base_dir: Path) -> Path | None:
        if raw is None:
            return None
        try:
            candidate = Path(str(raw))
        except Exception:  # noqa: BLE001
            return None
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return candidate

    @staticmethod
    def _coerce_optional_bool(value: object) -> Optional[bool]:
        if value is None:
            return None
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
        return None

    def _final_llm_include_reasoning(self, config: Mapping[str, Any]) -> bool:
        candidates: list[object] = [config.get("final_llm_include_reasoning")]
        llm_cfg = config.get("llm_labeling")
        if isinstance(llm_cfg, Mapping):
            candidates.append(llm_cfg.get("include_reasoning"))
            candidates.append(llm_cfg.get("final_llm_include_reasoning"))
        ai_backend_cfg = config.get("ai_backend")
        if isinstance(ai_backend_cfg, Mapping):
            candidates.append(ai_backend_cfg.get("final_llm_include_reasoning"))
            overrides = ai_backend_cfg.get("config_overrides")
            if isinstance(overrides, Mapping):
                candidates.append(overrides.get("final_llm_include_reasoning"))
                llm_overrides = overrides.get("llm")
                if isinstance(llm_overrides, Mapping):
                    candidates.append(llm_overrides.get("include_reasoning"))
        for candidate in candidates:
            parsed = self._coerce_optional_bool(candidate)
            if parsed is not None:
                return parsed
        return True

    @staticmethod
    def _extract_few_shot_examples(
        config: Mapping[str, Any], labelset: Mapping[str, Any] | None = None
    ) -> Dict[str, list[dict[str, str]]]:
        ai_backend_cfg = config.get("ai_backend") if isinstance(config.get("ai_backend"), Mapping) else {}
        llm_cfg = ai_backend_cfg.get("llm") if isinstance(ai_backend_cfg, Mapping) and isinstance(ai_backend_cfg.get("llm"), Mapping) else {}
        top_llm_cfg = config.get("llm") if isinstance(config.get("llm"), Mapping) else {}

        candidates: list[object] = []
        if isinstance(ai_backend_cfg, Mapping):
            candidates.append(ai_backend_cfg.get("few_shot_examples"))
        if isinstance(llm_cfg, Mapping):
            candidates.append(llm_cfg.get("few_shot_examples"))
        if isinstance(top_llm_cfg, Mapping):
            candidates.append(top_llm_cfg.get("few_shot_examples"))

        few_shot_raw: Mapping[str, object] | None = None
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                few_shot_raw = candidate
                break

        cleaned: Dict[str, list[dict[str, str]]] = {}
        if few_shot_raw:
            for label_id, examples in few_shot_raw.items():
                if not isinstance(examples, Sequence):
                    continue
                parsed_examples: list[dict[str, str]] = []
                for entry in examples:
                    if not isinstance(entry, Mapping):
                        continue
                    example: dict[str, str] = {}
                    if entry.get("context") is not None:
                        example["context"] = str(entry.get("context"))
                    if entry.get("answer") is not None:
                        example["answer"] = str(entry.get("answer"))
                    if example:
                        parsed_examples.append(example)
                if parsed_examples:
                    cleaned[str(label_id)] = parsed_examples

        if labelset:
            labels = labelset.get("labels") if isinstance(labelset, Mapping) else None
            if isinstance(labels, Sequence):
                for label in labels:
                    if not isinstance(label, Mapping):
                        continue
                    label_id = str(label.get("label_id") or "")
                    if not label_id:
                        continue
                    raw_examples = label.get("few_shot_examples")
                    if not isinstance(raw_examples, Sequence):
                        continue
                    parsed_examples: list[dict[str, str]] = []
                    for entry in raw_examples:
                        if not isinstance(entry, Mapping):
                            continue
                        example: dict[str, str] = {}
                        if entry.get("context") is not None:
                            example["context"] = str(entry.get("context"))
                        if entry.get("answer") is not None:
                            example["answer"] = str(entry.get("answer"))
                        if example:
                            parsed_examples.append(example)
                    if parsed_examples and label_id not in cleaned:
                        cleaned[label_id] = parsed_examples
        return cleaned

    @staticmethod
    def _extract_label_keywords(config: Mapping[str, Any]) -> Dict[str, list[str]]:
        ai_backend_cfg = config.get("ai_backend") if isinstance(config.get("ai_backend"), Mapping) else {}
        rag_cfg = ai_backend_cfg.get("rag") if isinstance(ai_backend_cfg, Mapping) and isinstance(ai_backend_cfg.get("rag"), Mapping) else {}
        top_rag_cfg = config.get("rag") if isinstance(config.get("rag"), Mapping) else {}
        raw_candidates: list[object] = []
        if isinstance(rag_cfg, Mapping):
            raw_candidates.append(rag_cfg.get("label_keywords"))
        if isinstance(top_rag_cfg, Mapping):
            raw_candidates.append(top_rag_cfg.get("label_keywords"))

        raw: Mapping[str, object] | None = None
        for candidate in raw_candidates:
            if isinstance(candidate, Mapping):
                raw = candidate
                break

        if not raw:
            return {}

        parsed: Dict[str, list[str]] = {}
        for label_id, keywords in raw.items():
            values: list[str] = []
            if isinstance(keywords, str):
                values.extend([kw.strip() for kw in re.split(r"[,\n]", keywords) if kw.strip()])
            elif isinstance(keywords, Sequence):
                values.extend([str(kw).strip() for kw in keywords if isinstance(kw, str) and kw.strip()])
            if values:
                parsed[str(label_id)] = values
        return parsed

    @staticmethod
    def _apply_label_keywords(
        label_config: Mapping[str, object], label_keywords: Mapping[str, Sequence[str]]
    ) -> Dict[str, object]:
        merged = {key: copy.deepcopy(value) for key, value in label_config.items()}
        for label_id, keywords in label_keywords.items():
            if not keywords:
                continue
            entry = merged.get(label_id)
            entry_payload = copy.deepcopy(entry) if isinstance(entry, Mapping) else {}
            entry_payload["keywords"] = list(keywords)
            merged[label_id] = entry_payload
        return merged

    def _run_final_llm_labeling_inference(
        self,
        *,
        pheno_row: sqlite3.Row,
        labelset: Mapping[str, object],
        round_dir: Path,
        reviewer_assignments: Mapping[str, Sequence[AssignmentUnit]],
        config: Mapping[str, Any],
        include_reasoning: bool,
    ) -> tuple["pd.DataFrame", "pd.DataFrame"]:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("pandas is required for final LLM labeling") from exc
        try:
            from vaannotate.vaannotate_ai_backend.engine import (
                ActiveLearningLLMFirst,
                FamilyLabeler,
                OrchestratorConfig,
                Paths,
                _jsonify_cols,
            )
            from vaannotate.vaannotate_ai_backend import orchestrator as orchestrator_module
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("AI backend components are required for final LLM labeling") from exc

        units_by_id, notes_rows = self._collect_round_units_and_notes(reviewer_assignments)
        if not units_by_id:
            raise RuntimeError("No assignment units available for final LLM labeling")

        if not notes_rows:
            raise RuntimeError("Unable to assemble corpus notes for final LLM labeling")

        notes_df = pd.DataFrame(notes_rows)
        notes_df["patient_icn"] = notes_df["patient_icn"].astype(str)
        notes_df["doc_id"] = notes_df["doc_id"].astype(str)
        notes_df["text"] = notes_df["text"].astype(str)
        notes_df["unit_id"] = notes_df["unit_id"].astype(str)

        ann_df = pd.DataFrame(
            {
                "round_id": pd.Series(dtype="object"),
                "unit_id": pd.Series(dtype="object"),
                "doc_id": pd.Series(dtype="object"),
                "label_id": pd.Series(dtype="object"),
                "reviewer_id": pd.Series(dtype="object"),
                "label_value": pd.Series(dtype="object"),
                "label_value_num": pd.Series(dtype="float64"),
                "label_value_date": pd.Series(dtype="datetime64[ns]"),
            }
        )

        work_dir = ensure_dir(Path(round_dir) / "imports" / "final_llm_labeling")
        notes_path = work_dir / "notes.parquet"
        ann_path = work_dir / "annotations.parquet"
        notes_df.to_parquet(notes_path, index=False)
        ann_df.to_parquet(ann_path, index=False)

        ai_backend_config = config.get("ai_backend") if isinstance(config.get("ai_backend"), Mapping) else {}
        config_overrides = ai_backend_config.get("config_overrides") if isinstance(ai_backend_config, Mapping) else None

        consistency = 1
        candidates: list[Any] = [config.get("final_llm_labeling_n_consistency")]
        llm_cfg = config.get("llm_labeling")
        if isinstance(llm_cfg, Mapping):
            for key in ("consistency_runs", "n_consistency", "final_llm_label_consistency"):
                candidates.append(llm_cfg.get(key))
        if isinstance(config_overrides, Mapping):
            candidates.append(config_overrides.get("final_llm_labeling_n_consistency"))
            override_llm = config_overrides.get("llm_labeling")
            if isinstance(override_llm, Mapping):
                for key in ("consistency_runs", "n_consistency", "final_llm_label_consistency"):
                    candidates.append(override_llm.get(key))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                consistency = value
                break

        cfg = OrchestratorConfig()
        overrides: Dict[str, Any] = {}
        if isinstance(config_overrides, Mapping):
            overrides.update(config_overrides)

        llmfirst_overrides = (
            ai_backend_config.get("llmfirst")
            if isinstance(ai_backend_config.get("llmfirst"), Mapping)
            else {}
        )
        if llmfirst_overrides:
            overrides.setdefault("llmfirst", {}).update(llmfirst_overrides)

        if overrides:
            try:
                orchestrator_module._apply_overrides(cfg, overrides)
            except Exception:  # noqa: BLE001
                pass

        cfg.final_llm_labeling = True
        cfg.final_llm_labeling_n_consistency = max(1, consistency)
        setattr(cfg.llmfirst, "final_llm_label_consistency", cfg.final_llm_labeling_n_consistency)
        setattr(cfg.llm, "include_reasoning", bool(include_reasoning))
        few_shot_examples = self._extract_few_shot_examples(config, labelset)
        if few_shot_examples:
            try:
                setattr(cfg.llm, "few_shot_examples", few_shot_examples)
            except Exception:  # noqa: BLE001
                pass

        phenotype_level = str(pheno_row["level"] or "multi_doc")
        label_config_payload = build_label_config(labelset)
        label_keywords = self._extract_label_keywords(config)
        if label_keywords:
            label_config_payload = self._apply_label_keywords(label_config_payload, label_keywords)
        paths = Paths(str(notes_path), str(ann_path), str(work_dir / "engine_outputs"))
        orchestrator = ActiveLearningLLMFirst(
            paths=paths,
            cfg=cfg,
            label_config=label_config_payload,
            phenotype_level=phenotype_level,
        )

        # Ensure the RAG retriever has access to corpus embeddings. The AI backend
        # pathway builds the chunk index prior to any LLM labeling so that each
        # call has patient-level context with reranking. The random sampling
        # pathway skips the orchestrator run loop, so we need to explicitly build
        # the embeddings/index here as well.
        orchestrator.store.build_chunk_index(
            orchestrator.repo.notes,
            orchestrator.cfg.rag,
            orchestrator.cfg.index,
        )

        _, _, current_rules_map, current_label_types = orchestrator._label_maps()
        family_labeler = FamilyLabeler(
            orchestrator.llm,
            orchestrator.rag,
            orchestrator.repo,
            orchestrator.label_config,
            orchestrator.cfg.scjitter,
            orchestrator.cfg.llmfirst,
        )

        fam_rows: list[dict[str, Any]] = []
        for unit_id in units_by_id.keys():
            fam_rows.extend(
                family_labeler.label_family_for_unit(
                    unit_id,
                    current_label_types,
                    current_rules_map,
                    json_only=True,
                    json_n_consistency=cfg.final_llm_labeling_n_consistency,
                    json_jitter=False,
                )
            )

        fam_df = pd.DataFrame(fam_rows)
        if not fam_df.empty:
            if "runs" in fam_df.columns:
                fam_df.rename(columns={"runs": "llm_runs"}, inplace=True)
            if "consistency" in fam_df.columns:
                fam_df.rename(columns={"consistency": "llm_consistency"}, inplace=True)
            if "prediction" in fam_df.columns:
                fam_df.rename(columns={"prediction": "llm_prediction"}, inplace=True)
            if include_reasoning and "llm_runs" in fam_df.columns:
                fam_df["llm_reasoning"] = fam_df["llm_runs"].map(
                    lambda runs: (runs[0].get("raw", {}).get("reasoning") if isinstance(runs, list) and runs else None)
                )
            fam_df = _jsonify_cols(
                fam_df,
                [col for col in ("rag_context", "llm_runs", "fc_probs") if col in fam_df.columns],
            )

        if fam_df.empty:
            labels_df = pd.DataFrame(columns=["unit_id"])
            return labels_df, fam_df

        pivot = fam_df[["unit_id", "label_id", "llm_prediction"]].copy()
        pivot["col"] = pivot["label_id"].astype(str) + "_llm"
        fam_wide = (
            pivot.pivot_table(index="unit_id", columns="col", values="llm_prediction", aggfunc="first")
            .reset_index()
        )
        if include_reasoning and "llm_reasoning" in fam_df.columns:
            reasoning = fam_df[["unit_id", "label_id", "llm_reasoning"]].copy()
            reasoning["colr"] = reasoning["label_id"].astype(str) + "_llm_reason"
            fam_reason = (
                reasoning.pivot_table(index="unit_id", columns="colr", values="llm_reasoning", aggfunc="first")
                .reset_index()
            )
            fam_wide = fam_wide.merge(fam_reason, on="unit_id", how="left")

        return fam_wide, fam_df

    def _write_final_llm_outputs(
        self,
        *,
        labels_df: "pd.DataFrame" | None,
        probe_df: "pd.DataFrame" | None,
        exports_dir: Path,
    ) -> Dict[str, str]:
        import pandas as pd

        outputs: Dict[str, str] = {}

        if labels_df is not None:
            labels_df = labels_df.copy()
            labels_df = labels_df.replace({pd.NA: None}).where(pd.notnull(labels_df), None)
            labels_path = exports_dir / "final_llm_labels.parquet"
            labels_df.to_parquet(labels_path, index=False)
            outputs["final_llm_labels"] = str(labels_path)
            labels_json_path = labels_path.with_suffix(".json")
            labels_json_path.write_text(
                self._json_dumps(labels_df.to_dict(orient="records")),
                encoding="utf-8",
            )
            outputs["final_llm_labels_json"] = str(labels_json_path)

        if probe_df is not None:
            probe_df = probe_df.copy()
            probe_df = probe_df.replace({pd.NA: None}).where(pd.notnull(probe_df), None)
            probe_path = exports_dir / "final_llm_family_probe.parquet"
            probe_df.to_parquet(probe_path, index=False)
            outputs["final_llm_family_probe"] = str(probe_path)
            probe_json_path = probe_path.with_suffix(".json")
            probe_json_path.write_text(
                self._json_dumps(probe_df.to_dict(orient="records")),
                encoding="utf-8",
            )
            outputs["final_llm_family_probe_json"] = str(probe_json_path)
            nested = self._family_probe_to_nested(probe_df)
            nested_path = exports_dir / "final_llm_labels_by_unit.json"
            nested_path.write_text(self._json_dumps(nested), encoding="utf-8")
            outputs["final_llm_labels_by_unit"] = str(nested_path)

        return outputs

    def _family_probe_to_nested(self, probe_df: "pd.DataFrame") -> Dict[str, list[dict[str, Any]]]:
        import pandas as pd

        result: Dict[str, list[dict[str, Any]]] = {}
        if probe_df is None or probe_df.empty:
            return result
        normalized = probe_df.replace({pd.NA: None}).where(pd.notnull(probe_df), None)
        for row in normalized.to_dict(orient="records"):
            unit_id = str(row.get("unit_id") or "")
            if not unit_id:
                continue
            entry = {
                key: self._normalize_for_json(value)
                for key, value in row.items()
                if key != "unit_id" and value is not None
            }
            if not entry:
                continue
            result.setdefault(unit_id, []).append(entry)
        return result

    def _json_dumps(self, payload: object) -> str:
        normalized = self._normalize_for_json(payload)
        return json.dumps(normalized, indent=2, ensure_ascii=False)

    @staticmethod
    def _normalize_for_json(value: object) -> object:
        import numpy as np
        import pandas as pd

        if value is None:
            return None
        if isinstance(value, (str, bool)):
            return value
        if isinstance(value, (int,)):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                return None
            return value
        if isinstance(value, np.generic):
            try:
                return RoundBuilder._normalize_for_json(value.item())
            except Exception:  # noqa: BLE001
                return float(value)
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): RoundBuilder._normalize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [RoundBuilder._normalize_for_json(v) for v in value]
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                return RoundBuilder._normalize_for_json(value.tolist())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return RoundBuilder._normalize_for_json(value.item())
            except Exception:  # noqa: BLE001
                pass
        return value

    @staticmethod
    def _build_label_schema_payload(labelset: Mapping[str, object]) -> Dict[str, object]:
        schema_labels = []
        raw_labels = labelset.get("labels", []) if isinstance(labelset, Mapping) else []
        if isinstance(raw_labels, list):
            iterable_labels = raw_labels
        else:
            iterable_labels = []
        for entry in iterable_labels:
            if not isinstance(entry, Mapping):
                continue
            options = []
            raw_options = entry.get("options", [])
            if isinstance(raw_options, list):
                for option in raw_options:
                    if not isinstance(option, Mapping):
                        continue
                    options.append(
                        {
                            "value": option.get("value"),
                            "display": option.get("display"),
                            "order_index": option.get("order_index"),
                            "weight": option.get("weight"),
                        }
                    )
            options.sort(key=lambda opt: (opt.get("order_index") is None, opt.get("order_index", 0)))
            schema_labels.append(
                {
                    "label_id": entry.get("label_id"),
                    "name": entry.get("name"),
                    "type": entry.get("type"),
                    "required": bool(entry.get("required")),
                    "na_allowed": bool(entry.get("na_allowed")),
                    "rules": entry.get("rules"),
                    "unit": entry.get("unit"),
                    "range": {"min": entry.get("min"), "max": entry.get("max")},
                    "gating_expr": entry.get("gating_expr"),
                    "options": options,
                    "keywords": entry.get("keywords", []),
                    "few_shot_examples": entry.get("few_shot_examples", []),
                }
            )

        labelset_id = labelset.get("labelset_id") if isinstance(labelset, Mapping) else None
        payload: Dict[str, object] = {
            "labelset_id": labelset_id,
            "labels": schema_labels,
        }
        if labelset_id is not None:
            labelset_name = None
            if isinstance(labelset, Mapping):
                labelset_name = labelset.get("labelset_name")
            payload["labelset_name"] = labelset_name or labelset_id
        if isinstance(labelset, Mapping):
            for key in ("created_by", "created_at", "notes"):
                value = labelset.get(key)
                if value is not None:
                    payload[key] = value
        return payload

    def _build_candidates(self, corpus_conn: sqlite3.Connection, pheno_row: sqlite3.Row, config: dict) -> Iterator[CandidateUnit]:
        level = pheno_row["level"]
        filters_config = config.get("filters", {})
        strat_config = config.get("stratification") or {}
        metadata_filters_config = filters_config.get("metadata")
        strat_field_keys: list[str] = []
        use_metadata_sampling = isinstance(metadata_filters_config, (list, dict))
        raw_metadata_filters: list[Mapping[str, object]] = []
        metadata_logic = "all"
        if isinstance(metadata_filters_config, dict):
            metadata_logic = str(metadata_filters_config.get("logic") or "all").lower()
            raw_conditions = metadata_filters_config.get("conditions")
            if isinstance(raw_conditions, list):
                raw_metadata_filters = [
                    entry for entry in raw_conditions if isinstance(entry, Mapping)
                ]
        elif isinstance(metadata_filters_config, list):
            raw_metadata_filters = [
                entry for entry in metadata_filters_config if isinstance(entry, Mapping)
            ]
        if isinstance(strat_config, dict):
            raw_fields = strat_config.get("fields")
            if isinstance(raw_fields, list):
                strat_field_keys = [str(field) for field in raw_fields if str(field)]
                if strat_field_keys:
                    use_metadata_sampling = True
            else:
                legacy_keys = strat_config.get("keys")
                if isinstance(legacy_keys, list):
                    strat_field_keys = [str(key) for key in legacy_keys]
                elif legacy_keys:
                    strat_field_keys = [str(legacy_keys)]

        if use_metadata_sampling:
            metadata_fields: Sequence[MetadataField] = discover_corpus_metadata(corpus_conn)
            field_lookup = {field.key: field for field in metadata_fields}

            conditions: list[MetadataFilterCondition] = []
            missing_filter_fields: list[str] = []
            for entry in raw_metadata_filters:
                try:
                    condition = MetadataFilterCondition.from_payload(entry)
                except Exception:  # noqa: BLE001
                    continue
                field = field_lookup.get(condition.field)
                if not field:
                    missing_filter_fields.append(condition.field or condition.label)
                    continue
                conditions.append(condition)
            if missing_filter_fields:
                raise ValueError(
                    "Unknown metadata fields in filters: "
                    + ", ".join(sorted({field for field in missing_filter_fields if field})),
                )

            sampling_filters = SamplingFilters(
                metadata_filters=conditions,
                match_any=metadata_logic == "any",
            )

            filtered_strat_keys: list[str] = []
            strat_aliases: list[str] = []
            missing_strat_fields: list[str] = []
            for key in strat_field_keys:
                field = field_lookup.get(key)
                if not field:
                    missing_strat_fields.append(key)
                    continue
                filtered_strat_keys.append(field.key)
                strat_aliases.append(field.alias)
            if missing_strat_fields:
                raise ValueError(
                    "Unknown metadata fields requested for stratification: "
                    + ", ".join(sorted(set(missing_strat_fields))),
                )

            rows = candidate_documents(
                corpus_conn,
                level,
                sampling_filters,
                metadata_fields=metadata_fields,
                stratify_keys=filtered_strat_keys,
            )
            if level == "single_doc":
                for row in rows:
                    row_dict = dict(row)
                    strata_key = self._compute_strata_key(row_dict, strat_aliases)
                    document_metadata = extract_document_metadata(row_dict)
                    documents = [
                        {
                            "doc_id": row_dict["doc_id"],
                            "hash": row_dict.get("hash"),
                            "text": row_dict.get("text"),
                            "order_index": 0,
                            "metadata_json": row_dict.get("metadata_json"),
                            "metadata": document_metadata,
                            "date_note": row_dict.get("date_note"),
                        }
                    ]
                    payload = {
                        "date_note": row_dict.get("date_note"),
                        "hash": row_dict.get("hash"),
                        "metadata_json": row_dict.get("metadata_json"),
                        "metadata": document_metadata,
                        "strata_key": strata_key,
                        "note_count": 1,
                        "documents": documents,
                        "display_label": row_dict["doc_id"],
                    }
                    yield CandidateUnit(row_dict["doc_id"], row_dict["patient_icn"], row_dict["doc_id"], strata_key, payload)
            else:
                for entry in rows:
                    strata_key = self._compute_strata_key(entry, strat_aliases)
                    payload = dict(entry)
                    payload["strata_key"] = strata_key
                    payload.setdefault("display_label", entry.get("unit_id") or entry.get("patient_icn"))
                    yield CandidateUnit(
                        str(entry.get("unit_id") or entry.get("patient_icn")),
                        str(entry.get("patient_icn")),
                        entry.get("doc_id"),
                        strata_key,
                        payload,
                    )
            return

        params: list = []
        where_clauses = []
        regex_expr = None
        regex_flags = ""
        if "note" in filters_config:
            nf = filters_config["note"]
            regex_expr = nf.get("regex")
            regex_flags = nf.get("regex_flags", "")
            if regex_expr:
                where_clauses.append("documents.text REGEXP ?")
                params.append(regex_expr)
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        if regex_expr:
            flags = 0
            if "i" in regex_flags.lower():
                flags |= re.IGNORECASE
            pattern = re.compile(regex_expr, flags)
            corpus_conn.create_function("REGEXP", 2, lambda expr, item, compiled=pattern: 1 if compiled.search(item or "") else 0)
        else:
            corpus_conn.create_function("REGEXP", 2, lambda expr, item: 1)
        strat_keys_config = strat_config.get("keys") or []
        if isinstance(strat_keys_config, list):
            legacy_aliases = [str(key) for key in strat_keys_config]
        elif strat_keys_config:
            legacy_aliases = [str(strat_keys_config)]
        else:
            legacy_aliases = []
        cursor = corpus_conn.execute(
            f"""
            SELECT documents.doc_id, documents.patient_icn, documents.date_note,
                   documents.hash, documents.text, documents.metadata_json
            FROM documents
            WHERE {where_sql}
            """,
            params,
        )
        if level == "single_doc":
            for row in cursor:
                row_dict = dict(row)
                strata_key = self._compute_strata_key(row, legacy_aliases)
                document_metadata = extract_document_metadata(row_dict)
                documents = [
                    {
                        "doc_id": row["doc_id"],
                        "hash": row["hash"],
                        "text": row["text"],
                        "order_index": 0,
                        "metadata_json": row_dict.get("metadata_json"),
                        "metadata": document_metadata,
                        "date_note": row_dict.get("date_note"),
                    }
                ]
                payload = {
                    "date_note": row["date_note"],
                    "hash": row["hash"],
                    "metadata_json": row_dict.get("metadata_json"),
                    "metadata": document_metadata,
                    "strata_key": strata_key,
                    "note_count": 1,
                    "documents": documents,
                    "display_label": row["doc_id"],
                }
                yield CandidateUnit(row["doc_id"], row["patient_icn"], row["doc_id"], strata_key, payload)
        else:
            patient_docs: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for row in cursor:
                patient_docs[row["patient_icn"]].append(row)
            for patient_icn, docs in patient_docs.items():
                primary_row = docs[0]
                strata_key = self._compute_strata_key(primary_row, legacy_aliases)
                ordered_docs = sorted(
                    docs,
                    key=lambda item: (_date_sort_value(item["date_note"]), item["doc_id"]),
                )
                doc_payloads = []
                for idx, doc in enumerate(ordered_docs):
                    doc_dict = dict(doc)
                    metadata = extract_document_metadata(doc_dict)
                    doc_payloads.append(
                        {
                            "doc_id": doc_dict["doc_id"],
                            "hash": doc_dict.get("hash"),
                            "text": doc_dict.get("text"),
                            "order_index": idx,
                            "metadata_json": doc_dict.get("metadata_json"),
                            "metadata": metadata,
                            "date_note": doc_dict.get("date_note"),
                        }
                    )
                primary_dict = dict(primary_row)
                primary_metadata = extract_document_metadata(primary_dict)
                payload = {
                    "date_note": primary_dict.get("date_note"),
                    "metadata_json": primary_dict.get("metadata_json"),
                    "metadata": primary_metadata,
                    "strata_key": strata_key,
                    "note_count": len(docs),
                    "documents": doc_payloads,
                    "display_label": patient_icn,
                }
                yield CandidateUnit(patient_icn, patient_icn, None, strata_key, payload)

    def _build_preselected_candidates(
        self,
        corpus_conn: sqlite3.Connection,
        pheno_row: sqlite3.Row,
        csv_path: Path,
    ) -> Iterator[CandidateUnit]:
        level = str(pheno_row["level"] or "single_doc")
        rows_by_unit: dict[str, dict] = {}
        unit_order: list[str] = []
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for position, row in enumerate(reader):
                if not isinstance(row, Mapping):
                    continue
                normalized: dict[str, str] = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    normalized[key.strip().lower()] = (value or "").strip()
                raw_unit = (
                    normalized.get("unit_id")
                    or normalized.get("doc_id")
                    or normalized.get("patient_icn")
                    or normalized.get("patienticn")
                )
                if not raw_unit:
                    continue
                unit_id = str(raw_unit)
                entry = rows_by_unit.get(unit_id)
                if not entry:
                    entry = {
                        "unit_id": unit_id,
                        "patient_icn": normalized.get("patient_icn")
                        or normalized.get("patienticn"),
                        "doc_ids": [],
                        "strata_key": normalized.get("strata_key") or "ai_import",
                        "selection_reasons": [],
                        "doc_order": [],
                        "first_index": position,
                    }
                    rows_by_unit[unit_id] = entry
                    unit_order.append(unit_id)
                if not entry.get("patient_icn"):
                    patient_value = normalized.get("patient_icn") or normalized.get("patienticn")
                    if patient_value:
                        entry["patient_icn"] = patient_value
                doc_id = normalized.get("doc_id") or normalized.get("document_id")
                if doc_id:
                    doc_id = str(doc_id)
                    entry["doc_ids"].append(doc_id)
                    entry["doc_order"].append(doc_id)
                reason = normalized.get("selection_reason") or normalized.get("reason")
                if reason:
                    entry["selection_reasons"].append(reason)
                strata_value = normalized.get("strata_key")
                if strata_value:
                    entry["strata_key"] = str(strata_value)

        if not rows_by_unit:
            return

        doc_ids: list[str] = []
        for entry in rows_by_unit.values():
            doc_ids.extend(entry.get("doc_ids", []))
        doc_map = self._fetch_documents_by_ids(corpus_conn, doc_ids)

        for unit_id in unit_order:
            entry = rows_by_unit[unit_id]
            strata_key = str(entry.get("strata_key") or "ai_import")
            selection_reason = next(
                (reason for reason in entry.get("selection_reasons", []) if reason),
                None,
            )
            if level == "single_doc":
                doc_id = entry["doc_ids"][0] if entry["doc_ids"] else unit_id
                doc_row = doc_map.get(doc_id)
                if not doc_row:
                    raise ValueError(
                        f"Document {doc_id} referenced in {csv_path} was not found in the corpus"
                    )
                row_dict = dict(doc_row)
                patient_icn = str(
                    entry.get("patient_icn")
                    or row_dict.get("patient_icn")
                    or row_dict.get("patienticn")
                    or ""
                )
                metadata = extract_document_metadata(row_dict)
                documents = [
                    {
                        "doc_id": row_dict.get("doc_id"),
                        "hash": row_dict.get("hash"),
                        "text": row_dict.get("text"),
                        "order_index": 0,
                        "metadata_json": row_dict.get("metadata_json"),
                        "metadata": metadata,
                        "date_note": row_dict.get("date_note"),
                    }
                ]
                payload = {
                    "date_note": row_dict.get("date_note"),
                    "hash": row_dict.get("hash"),
                    "metadata_json": row_dict.get("metadata_json"),
                    "metadata": metadata,
                    "strata_key": strata_key,
                    "note_count": 1,
                    "documents": documents,
                    "display_label": row_dict.get("doc_id") or unit_id,
                }
                if selection_reason:
                    payload["selection_reason"] = selection_reason
                yield CandidateUnit(str(unit_id), patient_icn, row_dict.get("doc_id"), strata_key, payload)
                continue

            patient_icn = entry.get("patient_icn")
            doc_rows: list[sqlite3.Row] = []
            if entry["doc_ids"]:
                for doc_id in entry["doc_ids"]:
                    doc_row = doc_map.get(doc_id)
                    if not doc_row:
                        raise ValueError(
                            f"Document {doc_id} referenced in {csv_path} was not found in the corpus"
                        )
                    doc_rows.append(doc_row)
            else:
                if not patient_icn:
                    raise ValueError(
                        f"Preselected unit {unit_id} is missing patient identifier and document IDs"
                    )
                doc_rows = self._fetch_patient_documents(corpus_conn, str(patient_icn))
                if not doc_rows:
                    raise ValueError(
                        f"No documents found for patient {patient_icn} referenced by {unit_id}"
                    )

            ordered = []
            for index, doc_row in enumerate(doc_rows):
                doc_dict = dict(doc_row)
                metadata = extract_document_metadata(doc_dict)
                ordered.append(
                    {
                        "doc_id": doc_dict.get("doc_id"),
                        "hash": doc_dict.get("hash"),
                        "text": doc_dict.get("text"),
                        "order_index": index,
                        "metadata_json": doc_dict.get("metadata_json"),
                        "metadata": metadata,
                        "date_note": doc_dict.get("date_note"),
                    }
                )
                if not patient_icn and doc_dict.get("patient_icn"):
                    patient_icn = doc_dict.get("patient_icn")
            if not patient_icn:
                raise ValueError(
                    f"Could not determine patient identifier for preselected unit {unit_id}"
                )
            payload = {
                "metadata_json": ordered[0].get("metadata_json") if ordered else None,
                "metadata": ordered[0].get("metadata") if ordered else None,
                "strata_key": strata_key,
                "note_count": len(ordered),
                "documents": ordered,
                "display_label": unit_id,
            }
            if selection_reason:
                payload["selection_reason"] = selection_reason
            yield CandidateUnit(str(unit_id), str(patient_icn), None, strata_key, payload)

    def _fetch_documents_by_ids(
        self, corpus_conn: sqlite3.Connection, doc_ids: Sequence[str]
    ) -> dict[str, sqlite3.Row]:
        unique_ids = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id]
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        cursor = corpus_conn.execute(
            f"""
            SELECT doc_id, patient_icn, date_note, hash, text, metadata_json
            FROM documents
            WHERE doc_id IN ({placeholders})
            """,
            unique_ids,
        )
        return {str(row["doc_id"]): row for row in cursor.fetchall()}

    def _fetch_patient_documents(
        self, corpus_conn: sqlite3.Connection, patient_icn: str
    ) -> list[sqlite3.Row]:
        cursor = corpus_conn.execute(
            """
            SELECT doc_id, patient_icn, date_note, hash, text, metadata_json
            FROM documents
            WHERE patient_icn=?
            """,
            (patient_icn,),
        )
        rows = cursor.fetchall()
        return sorted(
            rows,
            key=lambda item: (_date_sort_value(item["date_note"]), item["doc_id"]),
        )

    def _compute_strata_key(self, row: sqlite3.Row | dict, aliases: Sequence[str]) -> str:
        if not aliases:
            return "default"
        keys: list[str] = []
        for alias in aliases:
            value: object | None
            if isinstance(row, dict):
                value = row.get(alias)
                if value is None and alias in row:
                    value = row[alias]
            else:
                try:
                    value = row[alias]
                except Exception:
                    value = None
            keys.append("" if value is None else str(value))
        return "|".join(keys) if keys else "default"

    def _allocate_units(
        self,
        candidates: list[CandidateUnit],
        reviewers: list[str],
        rng_seed: int,
        overlap_n: int,
        total_n: int | None,
        strat_config: dict | None,
        preserve_input_order: bool = False,
    ) -> dict[str, list[AssignmentUnit]]:
        by_strata: dict[str, list[CandidateUnit]] = defaultdict(list)
        strata_order: list[str] = []
        for candidate in candidates:
            by_strata[candidate.strata_key].append(candidate)
            if preserve_input_order and candidate.strata_key not in strata_order:
                strata_order.append(candidate.strata_key)
        assignments: dict[str, list[AssignmentUnit]] = {rid: [] for rid in reviewers}
        stratified = self._stratification_requested(strat_config)
        if stratified and total_n is not None and len(by_strata) > total_n:
            raise ValueError(
                "The requested sample size is smaller than the number of strata. "
                "Increase the total units or adjust the stratification keys."
            )
        total_available = sum(len(items) for items in by_strata.values())
        target_total = total_available if total_n is None else min(total_n, total_available)
        if total_n is None:
            allocations = {key: len(items) for key, items in by_strata.items()}
        else:
            allocations = {key: 0 for key in by_strata}
            remaining = target_total
            active = [key for key, items in by_strata.items() if items]
            while remaining > 0 and active:
                share = max(1, remaining // len(active))
                next_active: list[str] = []
                for strata_key in list(active):
                    capacity = len(by_strata[strata_key]) - allocations[strata_key]
                    if capacity <= 0:
                        continue
                    take = min(share, capacity, remaining)
                    allocations[strata_key] += take
                    remaining -= take
                    if allocations[strata_key] < len(by_strata[strata_key]):
                        next_active.append(strata_key)
                    if remaining == 0:
                        break
                active = next_active if remaining > 0 else []
        assigned_total = 0
        selected_by_strata: dict[str, list[CandidateUnit]] = {}
        strata_keys = strata_order if preserve_input_order and strata_order else sorted(by_strata.keys())
        for strata_key in strata_keys:
            strata_candidates = by_strata[strata_key]
            if preserve_input_order:
                shuffled = list(strata_candidates)
            else:
                shuffled = deterministic_choice(strata_candidates, stable_hash(rng_seed, strata_key))
            take = allocations.get(strata_key, len(strata_candidates))
            if total_n is not None and assigned_total + take > target_total:
                take = max(0, target_total - assigned_total)
            selected = list(shuffled[:take])
            assigned_total += len(selected)
            selected_by_strata[strata_key] = selected

        overlap_allocations: dict[str, int] = {key: 0 for key in selected_by_strata}
        if overlap_n:
            remaining_overlap = min(
                overlap_n, sum(len(items) for items in selected_by_strata.values())
            )
            active = [key for key, items in selected_by_strata.items() if items]
            while remaining_overlap > 0 and active:
                share = max(1, remaining_overlap // len(active))
                next_active: list[str] = []
                for strata_key in list(active):
                    capacity = len(selected_by_strata[strata_key]) - overlap_allocations[strata_key]
                    if capacity <= 0:
                        continue
                    take = min(share, capacity, remaining_overlap)
                    overlap_allocations[strata_key] += take
                    remaining_overlap -= take
                    if (
                        overlap_allocations[strata_key] < len(selected_by_strata[strata_key])
                        and remaining_overlap > 0
                    ):
                        next_active.append(strata_key)
                    if remaining_overlap == 0:
                        break
                active = next_active if remaining_overlap > 0 else []

        for strata_key in sorted(selected_by_strata.keys()):
            selected = selected_by_strata.get(strata_key, [])
            overlap_count = overlap_allocations.get(strata_key, 0)
            overlap_units = selected[:overlap_count]
            remainder = selected[overlap_count:]
            per_reviewer = self._split_among_reviewers(remainder, reviewers, rng_seed, strata_key)
            for reviewer_id in reviewers:
                combined = []
                for unit in overlap_units:
                    combined.append(
                        AssignmentUnit(
                            unit.unit_id,
                            unit.patient_icn,
                            unit.doc_id,
                            {**unit.payload, "is_overlap": True},
                        )
                    )
                combined.extend(per_reviewer.get(reviewer_id, []))
                assignments[reviewer_id].extend(combined)
        for reviewer_id, units in assignments.items():
            seed = stable_hash(rng_seed, reviewer_id)
            shuffled = deterministic_choice(units, stable_hash(seed, "display"))
            assignments[reviewer_id] = [
                AssignmentUnit(unit.unit_id, unit.patient_icn, unit.doc_id, unit.payload)
                for unit in shuffled
            ]
        return assignments

    def _split_among_reviewers(
        self,
        units: list[CandidateUnit],
        reviewers: list[str],
        rng_seed: int,
        strata_key: str,
    ) -> dict[str, list[AssignmentUnit]]:
        shuffled = deterministic_choice(units, stable_hash(rng_seed, strata_key, "split"))
        buckets: dict[str, list[AssignmentUnit]] = {rid: [] for rid in reviewers}
        for idx, unit in enumerate(shuffled):
            reviewer_id = reviewers[idx % len(reviewers)]
            buckets[reviewer_id].append(
                AssignmentUnit(
                    unit.unit_id,
                    unit.patient_icn,
                    unit.doc_id,
                    {**unit.payload, "is_overlap": False},
                )
            )
        return buckets

    def _stratification_requested(self, strat_config: dict | None) -> bool:
        if not strat_config or not isinstance(strat_config, dict):
            return False
        fields = strat_config.get("fields")
        if isinstance(fields, list) and fields:
            return True
        keys = strat_config.get("keys")
        if isinstance(keys, list):
            return bool(keys)
        return bool(keys)

    # Aggregation & import helpers
    def import_assignment(self, pheno_id: str, round_number: int, reviewer_id: str) -> Path:
        with self._connect_project() as conn:
            pheno = conn.execute(
                "SELECT storage_path FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
            if not pheno:
                raise ValueError(f"Phenotype {pheno_id} not found")
            storage_path = Path(pheno["storage_path"])
            if not storage_path.is_absolute():
                storage_path = (self.project_root / storage_path).resolve()
            phenotype_dir = storage_path
            round_dir = phenotype_dir / "rounds" / f"round_{round_number}"
            assignment_dir = round_dir / "assignments" / reviewer_id
            assignment_db = assignment_dir / "assignment.db"
            imports_dir = ensure_dir(round_dir / "imports")
            target_path = imports_dir / f"{reviewer_id}_assignment.db"
            copy_sqlite_database(assignment_db, target_path)
            round_id = conn.execute(
                "SELECT round_id FROM rounds WHERE pheno_id=? AND round_number=?",
                (pheno_id, round_number),
            ).fetchone()
            if not round_id:
                raise ValueError("Round metadata missing")
            conn.execute(
                "UPDATE assignments SET status='imported' WHERE round_id=? AND reviewer_id=?",
                (round_id["round_id"], reviewer_id),
            )
            conn.commit()
        return target_path

    def build_round_aggregate(self, pheno_id: str, round_number: int) -> Path:
        with self._connect_project() as conn:
            pheno = conn.execute(
                "SELECT storage_path FROM phenotypes WHERE pheno_id=?",
                (pheno_id,),
            ).fetchone()
            if not pheno:
                raise ValueError(f"Phenotype {pheno_id} not found")
            storage_path = Path(pheno["storage_path"])
            if not storage_path.is_absolute():
                storage_path = (self.project_root / storage_path).resolve()
            phenotype_dir = storage_path
        round_dir = phenotype_dir / "rounds" / f"round_{round_number}"
        imports_dir = ensure_dir(round_dir / "imports")
        aggregate_db = round_dir / "round_aggregate.db"
        with initialize_round_aggregate_db(aggregate_db) as agg_conn:
            agg_conn.execute("DELETE FROM unit_annotations")
            agg_conn.execute("DELETE FROM unit_summary")
            for assignment_path in imports_dir.glob("*_assignment.db"):
                reviewer_id = assignment_path.stem
                if reviewer_id.endswith("_assignment"):
                    reviewer_id = reviewer_id[: -len("_assignment")]
                with sqlite3.connect(assignment_path) as assign_conn:
                    assign_conn.row_factory = sqlite3.Row
                    for unit_row in assign_conn.execute("SELECT * FROM units"):
                        agg_conn.execute(
                            "INSERT OR IGNORE INTO unit_summary(round_id, unit_id, patient_icn, doc_id) VALUES (?,?,?,?)",
                            (
                                f"{pheno_id}_r{round_number}",
                                unit_row["unit_id"],
                                unit_row["patient_icn"],
                                unit_row["doc_id"],
                            ),
                        )
                    for ann_row in assign_conn.execute("SELECT * FROM annotations"):
                        agg_conn.execute(
                            """
                            INSERT INTO unit_annotations(round_id, unit_id, reviewer_id, label_id, value, value_num, value_date, na, notes)
                            VALUES (?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                f"{pheno_id}_r{round_number}",
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
            agg_conn.commit()
        return aggregate_db
