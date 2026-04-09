from __future__ import annotations

import logging
import os
import time
import hashlib
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from ..adapters import _load_label_config_bundle, export_inputs_from_repo
from ..config import OrchestratorConfig, Paths
from ..experiments import _normalize_local_model_overrides
from ..label_configs import LabelConfigBundle
from ..llm_backends import build_llm_backend
from ..orchestration import _build_shared_components, BackendSession
from ..orchestrator import _apply_overrides
from ..services import LLMLabeler
from ..services.label_dependencies import build_label_dependencies, evaluate_gating
from ..utils.io import read_table
from ..utils.job_manifest import read_manifest, write_manifest_atomic
from ..utils.jsonish import _jsonify_cols
from .prompt_tasks import (
    FamilyPromptTask,
    SinglePromptTask,
    df_to_family_prompt_tasks,
    df_to_single_prompt_tasks,
    family_prompt_tasks_to_df,
    single_prompt_tasks_to_df,
)
from ..project_experiments import _infer_unit_id_column


@dataclass
class PromptPrecomputeJob:
    job_id: str
    project_root: Path
    pheno_id: str
    labelset_id: str
    phenotype_level: str  # e.g. "single_doc" | "multi_doc"
    labeling_mode: str  # "family" | "single_prompt"
    cfg_overrides: dict[str, Any]
    llm_overrides: dict[str, Any] | None = None
    corpus_id: str | None = None
    corpus_path: Path | None = None
    notes_path: Path | None = None
    annotations_path: Path | None = None
    dataset_path: Path | None = None
    dataset_column_map: dict[str, str] | None = None
    job_dir: Path | None = None
    batch_size: int = 128
    env_overrides: dict[str, str] | None = None
    status_callback: Callable[[str], None] | None = None


@dataclass
class PromptInferenceJob:
    job_id: str
    prompt_job_id: str
    project_root: Path
    prompt_job_dir: Path | None
    phenotype_level: str
    labeling_mode: str  # "family" | "single_prompt"
    cfg_overrides: dict[str, Any]
    llm_overrides: dict[str, Any] | None
    job_dir: Path | None = None
    batch_limit: int | None = None
    off_hours_only: bool = False
    status_callback: Callable[[str], None] | None = None


def _inference_explicitly_configures_logprobs(
    cfg_overrides: dict[str, Any] | None,
    llm_overrides: dict[str, Any] | None,
) -> bool:
    cfg_llm = cfg_overrides.get("llm") if isinstance(cfg_overrides, dict) else None
    if isinstance(cfg_llm, dict) and (
        "logprobs" in cfg_llm or "top_logprobs" in cfg_llm
    ):
        return True
    if isinstance(llm_overrides, dict) and (
        "logprobs" in llm_overrides or "top_logprobs" in llm_overrides
    ):
        return True
    return False


def _empty_annotations_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "round_id",
            "unit_id",
            "doc_id",
            "label_id",
            "reviewer_id",
            "label_value",
            "label_value_num",
            "label_value_date",
            "labelset_id",
            "document_text",
            "rationales_json",
            "document_metadata_json",
            "label_rules",
            "reviewer_notes",
        ]
    )


def _prepare_notes_from_external_dataset(
    dataset_path: Path,
    phenotype_level: str,
    dataset_column_map: dict[str, str] | None,
) -> pd.DataFrame:
    df = read_table(str(dataset_path)).copy()
    colmap = {str(k): str(v) for k, v in (dataset_column_map or {}).items() if str(k) and str(v)}
    rename_map = {src: dest for dest, src in colmap.items() if src in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    if "text" not in df.columns:
        raise ValueError("External inference dataset must provide a 'text' column or map one via dataset_column_map.")

    if "doc_id" not in df.columns:
        if phenotype_level == "single_doc":
            raise ValueError("single_doc prompt precompute requires a 'doc_id' column in the external dataset.")
        df["doc_id"] = df["unit_id"].astype(str) if "unit_id" in df.columns else df.index.astype(str)

    if "patient_icn" not in df.columns:
        fallback = "unit_id" if "unit_id" in df.columns else "doc_id"
        df["patient_icn"] = df[fallback].astype(str)

    if "unit_id" not in df.columns:
        df["unit_id"] = _infer_unit_id_column(df, phenotype_level)
    else:
        df["unit_id"] = df["unit_id"].astype(str)

    df["doc_id"] = df["doc_id"].astype(str)
    df["patient_icn"] = df["patient_icn"].astype(str)
    df["text"] = df["text"].astype(str)
    if "notetype" not in df.columns:
        df["notetype"] = ""
    return df


def _ordered_label_ids(label_config: dict[str, Any], label_types: dict[str, str]) -> list[str]:
    _parent_to_children, child_to_parents, roots = build_label_dependencies(label_config or {})
    ordered = list(roots)
    for lid in sorted(label_types.keys()):
        if lid not in ordered:
            ordered.append(lid)
    for lid in sorted(child_to_parents.keys()):
        if lid not in ordered:
            ordered.append(lid)
    return ordered


def _emit_precompute_status(job: PromptPrecomputeJob, message: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(message)
    if job.status_callback is not None:
        try:
            job.status_callback(message)
        except Exception:
            logger.debug("Prompt precompute status callback failed", exc_info=True)


def _emit_inference_status(job: PromptInferenceJob, message: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(message)
    if job.status_callback is not None:
        try:
            job.status_callback(message)
        except Exception:
            logger.debug("Prompt inference status callback failed", exc_info=True)


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "ETA unknown"
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"ETA {hours}h {minutes}m {secs}s"
    if minutes:
        return f"ETA {minutes}m {secs}s"
    return f"ETA {secs}s"


def _prompt_precompute_requires_retrieval_index(
    cfg: OrchestratorConfig,
    phenotype_level: str,
) -> bool:
    level = str(phenotype_level or "").strip().lower()
    context_mode = str(getattr(cfg.llmfirst, "single_doc_context", "rag") or "rag").strip().lower()
    return not (level == "single_doc" and context_mode == "full")


def _single_doc_precompute_defaults_to_full_context(job: PromptPrecomputeJob) -> bool:
    if str(job.phenotype_level or "").strip().lower() != "single_doc":
        return False
    if str(job.labeling_mode or "").strip().lower() != "single_prompt":
        return False

    cfg_llmfirst = job.cfg_overrides.get("llmfirst") if isinstance(job.cfg_overrides, dict) else None
    if isinstance(cfg_llmfirst, dict) and cfg_llmfirst.get("single_doc_context") is not None:
        return False

    llm_overrides = job.llm_overrides if isinstance(job.llm_overrides, dict) else {}
    llmfirst_overrides = llm_overrides.get("llmfirst") if isinstance(llm_overrides, dict) else None
    if isinstance(llmfirst_overrides, dict) and llmfirst_overrides.get("single_doc_context") is not None:
        return False

    return True


class _NoIndexStore:
    """Minimal store for full-document single-doc prompt precompute jobs."""

    chunk_meta: list[dict[str, Any]]

    def __init__(self) -> None:
        self.chunk_meta = []

    def get_patient_chunk_indices(self, _doc_id: str) -> list[int]:
        return []

    def _compute_corpus_fingerprint(self, notes_df: pd.DataFrame, rag_cfg: Any) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(
            f"no_index:chunk_size={getattr(rag_cfg, 'chunk_size', None)},overlap={getattr(rag_cfg, 'chunk_overlap', None)}".encode(
                "utf-8"
            )
        )
        if isinstance(notes_df, pd.DataFrame):
            if "hash" in notes_df.columns and "doc_id" in notes_df.columns:
                pairs = (
                    notes_df[["doc_id", "hash"]]
                    .fillna("")
                    .astype(str)
                    .sort_values(["doc_id", "hash"])
                )
                for row in pairs.itertuples(index=False):
                    h.update(f"{row.doc_id}:{row.hash}|".encode("utf-8"))
            elif "doc_id" in notes_df.columns and "text" in notes_df.columns:
                pairs = (
                    notes_df[["doc_id", "text"]]
                    .fillna("")
                    .astype(str)
                    .sort_values("doc_id")
                )
                for row in pairs.itertuples(index=False):
                    h.update(f"{row.doc_id}:{row.text}|".encode("utf-8"))
        return h.hexdigest()


class _NoIndexEmbedder:
    name_or_path = "no-index"


class _NoIndexModels:
    embedder = _NoIndexEmbedder()
    reranker = object()


def run_prompt_precompute_job(job: PromptPrecomputeJob) -> None:
    """
    Build RAG contexts and prompt tasks for a large unlabeled corpus.

    - Resolves job_dir, notes/annotations paths.
    - Builds OrchestratorConfig from cfg_overrides.
    - Builds a BackendSession with a shared EmbeddingStore cache under job_dir.
    - Initializes a manifest (or loads if existing).
    - Enumerates unit_ids, shards them into batches, and, for each pending batch,
      delegates to helper functions to build SinglePromptTask or FamilyPromptTask
      rows and writes them to a batch parquet file.
    - Updates the manifest using write_manifest_atomic.
    """

    _emit_precompute_status(job, f"Starting prompt precompute job {job.job_id}")

    applied_env: dict[str, str] = {
        str(key): str(value)
        for key, value in (job.env_overrides or {}).items()
        if str(value)
    }
    original_env: dict[str, str | None] = {}
    for key, value in applied_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        job_dir = job.job_dir or job.project_root / "admin_tools" / "prompt_jobs" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "prompts_single").mkdir(parents=True, exist_ok=True)
        (job_dir / "prompts_family").mkdir(parents=True, exist_ok=True)
        (job_dir / "cache").mkdir(parents=True, exist_ok=True)

        manifest_path = job_dir / "job_manifest.json"
        manifest = read_manifest(manifest_path)
        if isinstance(manifest, dict):
            manifest_cfg = manifest.get("cfg_overrides")
            if not job.cfg_overrides and isinstance(manifest_cfg, dict):
                job.cfg_overrides = manifest_cfg

            manifest_llm = manifest.get("llm_overrides")
            if not job.llm_overrides and isinstance(manifest_llm, dict):
                job.llm_overrides = manifest_llm

        if job.notes_path is not None:
            notes_path = job.notes_path
            ann_path = job.annotations_path or (job_dir / "annotations.parquet")
            if job.annotations_path is None:
                _empty_annotations_frame().to_parquet(ann_path, index=False)
        elif job.dataset_path is not None:
            notes_df = _prepare_notes_from_external_dataset(
                job.dataset_path,
                job.phenotype_level,
                job.dataset_column_map,
            )
            notes_path = job_dir / "notes.parquet"
            ann_path = job_dir / "annotations.parquet"
            notes_df.to_parquet(notes_path, index=False)
            _empty_annotations_frame().to_parquet(ann_path, index=False)
        else:
            notes_df, ann_df = export_inputs_from_repo(
                job.project_root,
                job.pheno_id,
                [],
                corpus_id=job.corpus_id,
                corpus_path=str(job.corpus_path) if job.corpus_path else None,
            )
            notes_df = notes_df.copy()
            notes_df["unit_id"] = _infer_unit_id_column(notes_df, job.phenotype_level)

            notes_path = job_dir / "notes.parquet"
            ann_path = job_dir / "annotations.parquet"

            notes_df.to_parquet(notes_path)
            ann_df.to_parquet(ann_path)

        cfg = OrchestratorConfig()
        overrides = _normalize_local_model_overrides(job.cfg_overrides or {})
        if overrides:
            _apply_overrides(cfg, dict(overrides))

        llm_overrides = job.llm_overrides or {}
        if llm_overrides:
            _apply_overrides(cfg, {"llm": llm_overrides})
            if "llmfirst" in llm_overrides:
                _apply_overrides(cfg, {"llmfirst": llm_overrides.get("llmfirst")})

        if _single_doc_precompute_defaults_to_full_context(job):
            setattr(cfg.llmfirst, "single_doc_context", "full")
            if not isinstance(job.cfg_overrides, dict):
                job.cfg_overrides = {}
            llmfirst_cfg = job.cfg_overrides.get("llmfirst")
            if not isinstance(llmfirst_cfg, dict):
                llmfirst_cfg = {}
                job.cfg_overrides["llmfirst"] = llmfirst_cfg
            llmfirst_cfg.setdefault("single_doc_context", "full")

        session_paths = Paths(
            notes_path=str(notes_path),
            annotations_path=str(ann_path),
            outdir=str(job_dir / "_session"),
            cache_dir_override=str(job_dir / "cache"),
        )

        needs_retrieval_index = _prompt_precompute_requires_retrieval_index(
            cfg,
            job.phenotype_level,
        )
        session: BackendSession | None = None
        if needs_retrieval_index:
            session = BackendSession.from_env(session_paths, cfg)

        label_config_bundle = _load_label_config_bundle(
            job.project_root,
            job.pheno_id,
            job.labelset_id,
            [],
            overrides=overrides.get("label_config") if isinstance(overrides, dict) else None,
        )

        if not manifest:
            manifest = {
                "job_id": job.job_id,
                "pheno_id": job.pheno_id,
                "labelset_id": job.labelset_id,
                "phenotype_level": job.phenotype_level,
                "labeling_mode": job.labeling_mode,
                "cfg_overrides": job.cfg_overrides,
                "llm_overrides": job.llm_overrides or {},
                "corpus_id": job.corpus_id,
                "corpus_path": str(job.corpus_path) if job.corpus_path else None,
                "notes_path": str(job.notes_path) if job.notes_path else None,
                "annotations_path": str(job.annotations_path) if job.annotations_path else None,
                "batch_size": job.batch_size,
                "dataset_path": str(job.dataset_path) if job.dataset_path else None,
                "dataset_column_map": job.dataset_column_map or {},
                "batches": [],
            }
        elif isinstance(manifest, dict):
            manifest["cfg_overrides"] = job.cfg_overrides
            manifest["llm_overrides"] = job.llm_overrides or {}
            manifest["corpus_id"] = job.corpus_id if job.corpus_id else manifest.get("corpus_id")
            manifest["corpus_path"] = (
                str(job.corpus_path)
                if job.corpus_path
                else manifest.get("corpus_path")
            )
            manifest["notes_path"] = (
                str(job.notes_path)
                if job.notes_path
                else manifest.get("notes_path")
            )
            manifest["annotations_path"] = (
                str(job.annotations_path)
                if job.annotations_path
                else manifest.get("annotations_path")
            )
            manifest["dataset_path"] = str(job.dataset_path) if job.dataset_path else manifest.get("dataset_path")
            manifest["dataset_column_map"] = job.dataset_column_map or manifest.get("dataset_column_map") or {}

        repo_notes = notes_df if (job.notes_path is None or job.dataset_path is not None) else read_table(str(notes_path))
        manifest = _initialize_and_update_batches_for_prompt_precompute(manifest, job, repo_notes)
        write_manifest_atomic(manifest_path, manifest)

        completed_batches = sum(
            1 for batch in manifest.get("batches", []) if batch.get("status") == "completed"
        )
        total_batches = len(manifest.get("batches", []))
        total_units = sum(len(batch.get("unit_ids", []) or []) for batch in manifest.get("batches", []))
        prep_message = (
            f"Prepared prompt precompute for {total_units} units across {total_batches} batches "
            f"({completed_batches} already completed). "
        )
        if needs_retrieval_index:
            prep_message += "Building retrieval index…"
        else:
            prep_message += "Using full-document single-doc context; skipping retrieval index build."
        _emit_precompute_status(job, prep_message)

        manifest = _run_prompt_precompute_batches(
            manifest,
            job,
            cfg,
            session,
            label_config_bundle,
            needs_retrieval_index=needs_retrieval_index,
            manifest_path=manifest_path,
        )

        write_manifest_atomic(manifest_path, manifest)
    finally:
        for key, value in applied_env.items():
            if key in original_env and original_env[key] is not None:
                os.environ[key] = original_env[key] or ""
            elif key in os.environ:
                os.environ.pop(key, None)


def _initialize_and_update_batches_for_prompt_precompute(
    manifest: dict,
    job: PromptPrecomputeJob,
    repo_notes: pd.DataFrame,
) -> dict:
    log = logging.getLogger(__name__)

    if "unit_id" not in repo_notes.columns:
        log.info("Inferring unit_id column for prompt precompute job %s", job.job_id)
        repo_notes = repo_notes.copy()
        repo_notes["unit_id"] = _infer_unit_id_column(repo_notes, job.phenotype_level)

    batches = manifest.get("batches") or []
    if batches:
        manifest["batches"] = batches
        return manifest

    unit_ids = sorted({str(u) for u in repo_notes["unit_id"].astype(str)})
    batch_size = int(job.batch_size or 0) or 1

    manifest_batches: list[dict] = []
    for idx in range(0, len(unit_ids), batch_size):
        chunk = unit_ids[idx : idx + batch_size]
        manifest_batches.append(
            {
                "batch_id": len(manifest_batches),
                "unit_ids": chunk,
                "status": "pending",
                "n_tasks": 0,
                "path": None,
            }
        )

    manifest["batches"] = manifest_batches
    return manifest


def _run_prompt_precompute_batches(
    manifest: dict,
    job: PromptPrecomputeJob,
    cfg: OrchestratorConfig,
    session: BackendSession | None,
    label_config_bundle: LabelConfigBundle,
    *,
    needs_retrieval_index: bool,
    manifest_path: Path,
) -> dict:
    job_dir = job.job_dir or job.project_root / "admin_tools" / "prompt_jobs" / job.job_id

    paths = Paths(
        notes_path=str(job_dir / "notes.parquet"),
        annotations_path=str(job_dir / "annotations.parquet"),
        outdir=str(job_dir / "_session"),
        cache_dir_override=str(job_dir / "cache"),
    )

    if needs_retrieval_index:
        if session is None:
            session = BackendSession.from_env(paths, cfg)
        components = _build_shared_components(
            paths,
            cfg,
            label_config_bundle,
            phenotype_level=job.phenotype_level,
            include_llm=False,
            models=session.models,
            store=session.store,
        )
        repo = components["repo"]
        store = components["store"]
        context_builder = components["context_builder"]
    else:
        components = _build_shared_components(
            paths,
            cfg,
            label_config_bundle,
            phenotype_level=job.phenotype_level,
            include_llm=False,
            models=_NoIndexModels(),
            store=_NoIndexStore(),
        )
        repo = components["repo"]
        store = components["store"]
        context_builder = components["context_builder"]
    llm_labeler = LLMLabeler(
        object(),
        label_config_bundle,
        cfg.llm,
        sc_cfg=cfg.scjitter,
        cache_dir=str(job_dir / "cache"),
    )

    if needs_retrieval_index:
        store.build_chunk_index(repo.notes, cfg.rag, cfg.index)

    try:
        rules_map = label_config_bundle.current_rules_map(components.get("label_config"))
    except TypeError:
        rules_map = label_config_bundle.current_rules_map()

    try:
        label_types = label_config_bundle.current_label_types(components.get("label_config"))
    except TypeError:
        label_types = label_config_bundle.current_label_types()

    rules_map = rules_map or {}
    label_types = label_types or {}
    label_config = (label_config_bundle.current or {}).copy()
    label_ids = _ordered_label_ids(label_config, label_types)

    rag_fingerprint = store._compute_corpus_fingerprint(repo.notes, cfg.rag)
    prompts_per_unit = max(1, len(label_ids)) if job.labeling_mode == "family" else 1
    total_batches = len(manifest.get("batches", []))
    total_prompts = sum(
        len(batch.get("unit_ids", []) or []) * prompts_per_unit
        for batch in manifest.get("batches", [])
    )
    completed_prompts = sum(
        int(batch.get("n_tasks") or 0)
        for batch in manifest.get("batches", [])
        if batch.get("status") == "completed"
    )
    started_at = time.monotonic()
    processed_this_run = 0

    for batch in manifest.get("batches", []):
        batch_id = batch.get("batch_id")
        batch_path = batch.get("path")
        if (
            batch.get("status") == "completed"
            and batch_path
            and (job_dir / str(batch_path)).exists()
            and os.access(job_dir / str(batch_path), os.R_OK)
        ):
            continue

        unit_ids = [str(u) for u in batch.get("unit_ids", [])]
        tasks: list[SinglePromptTask | FamilyPromptTask] = []
        remaining_before = max(total_prompts - completed_prompts, 0)
        elapsed = max(time.monotonic() - started_at, 0.0)
        rate = (processed_this_run / elapsed) if elapsed > 0 and processed_this_run > 0 else None
        eta_before = (remaining_before / rate) if rate and rate > 0 else None
        _emit_precompute_status(
            job,
            f"Precompute batch {int(batch_id) + 1}/{total_batches}: "
            f"{completed_prompts:,} prompts done, {remaining_before:,} prompts to go, "
            f"{_format_eta(eta_before)}.",
        )

        if job.labeling_mode == "single_prompt":
            for unit_id in unit_ids:
                ctx_snippets = context_builder.build_context_for_family(
                    unit_id,
                    label_ids=label_ids,
                    rules_map=rules_map,
                    topk_per_label=cfg.rag.top_k_final,
                    max_snippets=None,
                    max_chars=getattr(cfg.llmfirst, "single_prompt_max_chars", None),
                    single_doc_context_mode=getattr(cfg.llmfirst, "single_doc_context", "rag"),
                    full_doc_char_limit=getattr(cfg.llmfirst, "single_doc_full_context_max_chars", None),
                )

                rules_subset = {lid: rules_map.get(lid, "") for lid in label_ids}
                label_type_subset = {lid: label_types.get(lid, "") for lid in label_ids}

                tasks.append(
                    SinglePromptTask(
                        job_id=job.job_id,
                        prompt_id=f"{job.job_id}:unit:{unit_id}",
                        unit_id=unit_id,
                        label_ids=label_ids,
                        ctx_snippets=ctx_snippets,
                        rules_map=rules_subset,
                        label_types=label_type_subset,
                        rag_fingerprint=rag_fingerprint,
                        prompt_payload=llm_labeler.build_multi_label_prompt_payload(
                            label_ids=label_ids,
                            label_types=label_type_subset,
                            rules_map=rules_subset,
                            ctx_snippets=ctx_snippets,
                        ),
                        meta={
                            "pheno_id": job.pheno_id,
                            "labelset_id": job.labelset_id,
                            "phenotype_level": job.phenotype_level,
                            "label_order": label_ids,
                        },
                    )
                )

            df = single_prompt_tasks_to_df(tasks)
            out_path = job_dir / "prompts_single" / f"prompts_batch_{int(batch_id):05d}.parquet"
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, out_path)
            batch["path"] = str(out_path.relative_to(job_dir))

        elif job.labeling_mode == "family":
            ordered_labels = label_ids

            for unit_id in unit_ids:
                for label_id in ordered_labels:
                    ctx_snippets = context_builder.build_context_for_label(
                        unit_id,
                        label_id,
                        rules_map.get(label_id, ""),
                    )

                    tasks.append(
                        FamilyPromptTask(
                            job_id=job.job_id,
                            prompt_id=f"{job.job_id}:unit:{unit_id}:label:{label_id}",
                            unit_id=unit_id,
                            label_id=label_id,
                            label_type=label_types.get(label_id, ""),
                            label_rules=rules_map.get(label_id, ""),
                            ctx_snippets=ctx_snippets,
                            rag_fingerprint=rag_fingerprint,
                            prompt_payload=llm_labeler.build_single_label_prompt_payload(
                                label_id=label_id,
                                label_type=label_types.get(label_id, ""),
                                label_rules=rules_map.get(label_id, ""),
                                snippets=ctx_snippets,
                            ),
                            meta={
                                "pheno_id": job.pheno_id,
                                "labelset_id": job.labelset_id,
                                "phenotype_level": job.phenotype_level,
                                "label_order": ordered_labels,
                                "gated_by": list((label_config.get(label_id, {}) or {}).get("gated_by", []))
                                if isinstance((label_config.get(label_id, {}) or {}).get("gated_by", []), list)
                                else ([str((label_config.get(label_id, {}) or {}).get("gated_by"))]
                                      if (label_config.get(label_id, {}) or {}).get("gated_by")
                                      else []),
                            },
                        )
                    )

            df = family_prompt_tasks_to_df(tasks)
            out_path = job_dir / "prompts_family" / f"prompts_batch_{int(batch_id):05d}.parquet"
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, out_path)
            batch["path"] = str(out_path.relative_to(job_dir))

        else:
            logging.getLogger(__name__).warning(
                "Unknown labeling mode %s; skipping batch %s", job.labeling_mode, batch_id
            )
            continue

        batch["status"] = "completed"
        batch["n_tasks"] = len(tasks)
        completed_prompts += len(tasks)
        processed_this_run += len(tasks)

        write_manifest_atomic(manifest_path, manifest)
        remaining_after = max(total_prompts - completed_prompts, 0)
        elapsed = max(time.monotonic() - started_at, 0.0)
        rate = (processed_this_run / elapsed) if elapsed > 0 and processed_this_run > 0 else None
        eta_after = (remaining_after / rate) if rate and rate > 0 else None
        _emit_precompute_status(
            job,
            f"Completed batch {int(batch_id) + 1}/{total_batches}: "
            f"{completed_prompts:,}/{total_prompts:,} prompts done, "
            f"{remaining_after:,} prompts to go, {_format_eta(eta_after)}.",
        )

    _emit_precompute_status(
        job,
        f"Prompt precompute finished: {completed_prompts:,}/{total_prompts:,} prompts completed.",
    )

    return manifest


def run_prompt_inference_job(job: PromptInferenceJob) -> None:
    """
    Run LLM inference over precomputed prompt tasks (SinglePromptTask or FamilyPromptTask).

    - Locates the corresponding prompt job and its manifest.
    - Builds an OrchestratorConfig for inference (cfg_overrides + optional llm_overrides).
    - Builds an LLM backend (LLMLabeler, label_config_bundle, label dependency graph).
    - Iterates over prompt batch parquet files and for each pending batch:
      - Delegates to mode-specific helpers to obtain a tall predictions DataFrame.
      - Writes outputs_batch_{id:05d}.parquet atomically under this job_dir.
    - Updates this job's manifest with per-batch status and output paths.
    """

    log = logging.getLogger(__name__)
    _emit_inference_status(job, f"Starting prompt inference job {job.job_id}")

    job_dir = job.job_dir or job.project_root / "admin_tools" / "prompt_inference" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "outputs").mkdir(parents=True, exist_ok=True)

    prompt_job_dir = job.prompt_job_dir or job.project_root / "admin_tools" / "prompt_jobs" / job.prompt_job_id
    prompt_manifest = read_manifest(prompt_job_dir / "job_manifest.json")

    prompt_labeling_mode = prompt_manifest.get("labeling_mode") if prompt_manifest else None
    prompt_level = prompt_manifest.get("phenotype_level") if prompt_manifest else None
    if prompt_labeling_mode and prompt_labeling_mode != job.labeling_mode:
        log.warning(
            "Labeling mode mismatch between inference (%s) and prompt job (%s)",
            job.labeling_mode,
            prompt_labeling_mode,
        )
    if prompt_level and prompt_level != job.phenotype_level:
        log.warning(
            "Phenotype level mismatch between inference (%s) and prompt job (%s)",
            job.phenotype_level,
            prompt_level,
        )

    manifest_path = job_dir / "job_manifest.json"
    manifest = read_manifest(manifest_path)
    if isinstance(manifest, dict):
        manifest_cfg = manifest.get("cfg_overrides")
        if not job.cfg_overrides and isinstance(manifest_cfg, dict):
            job.cfg_overrides = manifest_cfg

        manifest_llm = manifest.get("llm_overrides")
        if not job.llm_overrides and isinstance(manifest_llm, dict):
            job.llm_overrides = manifest_llm

        manifest_off_hours_only = manifest.get("off_hours_only")
        if not job.off_hours_only and isinstance(manifest_off_hours_only, bool):
            job.off_hours_only = manifest_off_hours_only

    cfg = OrchestratorConfig()
    overrides = _normalize_local_model_overrides(job.cfg_overrides or {})
    if overrides:
        _apply_overrides(cfg, dict(overrides))

    llm_overrides = job.llm_overrides or {}
    if llm_overrides:
        _apply_overrides(cfg, {"llm": llm_overrides})
        if "llmfirst" in llm_overrides:
            _apply_overrides(cfg, {"llmfirst": llm_overrides.get("llmfirst")})

    if not _inference_explicitly_configures_logprobs(job.cfg_overrides, job.llm_overrides):
        cfg.llm.logprobs = False
        cfg.llm.top_logprobs = 0

    pheno_id = prompt_manifest.get("pheno_id") if prompt_manifest else None
    labelset_id = prompt_manifest.get("labelset_id") if prompt_manifest else None
    label_overrides = overrides.get("label_config") if isinstance(overrides, dict) else None
    label_config_bundle = _load_label_config_bundle(
        job.project_root,
        pheno_id,
        labelset_id,
        [],
        overrides=label_overrides,
    )

    llm_labeler = LLMLabeler(
        build_llm_backend(cfg.llm),
        label_config_bundle,
        cfg.llm,
        sc_cfg=cfg.scjitter,
        cache_dir=str(job_dir / "cache"),
    )
    llm_labeler.label_config = (
        getattr(label_config_bundle, "current", None)
        or getattr(label_config_bundle, "current_config", None)
        or {}
    )

    if not manifest:
        manifest = {
            "job_id": job.job_id,
            "prompt_job_id": job.prompt_job_id,
            "labeling_mode": job.labeling_mode,
            "phenotype_level": job.phenotype_level,
            "cfg_overrides": job.cfg_overrides,
            "llm_overrides": job.llm_overrides or {},
            "off_hours_only": bool(job.off_hours_only),
            "batches": [],
        }
    elif isinstance(manifest, dict):
        manifest["cfg_overrides"] = job.cfg_overrides
        manifest["llm_overrides"] = job.llm_overrides or {}
        manifest["off_hours_only"] = bool(job.off_hours_only)

    manifest = _initialize_and_update_batches_for_prompt_inference(manifest, job, prompt_manifest)
    write_manifest_atomic(manifest_path, manifest)

    total_batches = len(manifest.get("batches", []))
    completed_batches = sum(
        1 for batch in manifest.get("batches", []) if batch.get("status") == "completed"
    )
    _emit_inference_status(
        job,
        f"Prepared prompt inference for {total_batches} batches "
        f"({completed_batches} already completed).",
    )

    manifest = _run_prompt_inference_batches(
        manifest,
        job,
        cfg,
        llm_labeler,
        label_config_bundle,
        prompt_job_dir,
        job_dir,
        manifest_path,
    )

    write_manifest_atomic(manifest_path, manifest)
    completed_after = sum(
        1 for batch in manifest.get("batches", []) if batch.get("status") == "completed"
    )
    _emit_inference_status(
        job,
        f"Prompt inference finished: {completed_after}/{len(manifest.get('batches', []))} batches completed.",
    )


def _initialize_and_update_batches_for_prompt_inference(
    manifest: dict,
    job: PromptInferenceJob,
    prompt_manifest: dict,
) -> dict:
    prompt_batches = prompt_manifest.get("batches", []) if prompt_manifest else []

    batches = manifest.get("batches") or []
    if not batches:
        manifest["batches"] = [
            {
                "batch_id": b.get("batch_id"),
                "prompt_batch_path": b.get("path"),
                "status": "pending",
                "n_rows": 0,
                "output_path": None,
            }
            for b in prompt_batches
        ]
        return manifest

    existing_ids = {b.get("batch_id") for b in batches}
    for pb in prompt_batches:
        pb_id = pb.get("batch_id")
        if pb_id not in existing_ids:
            batches.append(
                {
                    "batch_id": pb_id,
                    "prompt_batch_path": pb.get("path"),
                    "status": "pending",
                    "n_rows": 0,
                    "output_path": None,
                }
            )

    manifest["batches"] = batches
    return manifest


def _run_prompt_inference_batches(
    manifest: dict,
    job: PromptInferenceJob,
    cfg: OrchestratorConfig,
    llm_labeler: LLMLabeler,
    label_config_bundle: LabelConfigBundle,
    prompt_job_dir: Path,
    inference_job_dir: Path,
    manifest_path: Path,
) -> dict:
    log = logging.getLogger(__name__)
    max_batches = int(job.batch_limit) if job.batch_limit and int(job.batch_limit) > 0 else None
    processed_batches = 0

    try:
        rules_map = label_config_bundle.current_rules_map()
    except TypeError:
        rules_map = label_config_bundle.current_rules_map(None)

    try:
        label_types = label_config_bundle.current_label_types()
    except TypeError:
        label_types = label_config_bundle.current_label_types(None)

    rules_map = rules_map or {}
    label_types = label_types or {}
    total_batches = len(manifest.get("batches", []))
    completed_batches = sum(
        1 for batch in manifest.get("batches", []) if batch.get("status") == "completed"
    )
    started_at = time.time()

    for batch in manifest.get("batches", []):
        if job.off_hours_only:
            _wait_for_off_hours_inference_window(log)
        if max_batches is not None and processed_batches >= max_batches:
            break
        batch_id = batch.get("batch_id")
        out_path_str = batch.get("output_path")
        out_path = inference_job_dir / str(out_path_str) if out_path_str else None
        if (
            batch.get("status") == "completed"
            and out_path
            and out_path.exists()
            and os.access(out_path, os.R_OK)
        ):
            continue

        prompt_path = batch.get("prompt_batch_path")
        if not prompt_path:
            log.warning("Missing prompt batch path for batch %s; skipping", batch_id)
            continue

        prompt_batch_path = prompt_job_dir / str(prompt_path)
        if not prompt_batch_path.exists():
            log.warning(
                "Prompt batch file %s not found for batch %s", prompt_batch_path, batch_id
            )
            continue

        df_prompts = pd.read_parquet(prompt_batch_path)

        if job.labeling_mode == "single_prompt":
            df_out = _run_single_prompt_batch(
                df_prompts,
                llm_labeler,
                label_config_bundle,
                rules_map,
                label_types,
            )
        elif job.labeling_mode == "family":
            df_out = _run_family_prompt_batch(
                df_prompts,
                llm_labeler,
                label_config_bundle,
                rules_map,
                label_types,
            )
        else:
            log.warning("Unknown labeling mode %s; skipping", job.labeling_mode)
            continue

        out_path = inference_job_dir / "outputs" / f"outputs_batch_{int(batch_id):05d}.parquet"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        df_out.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)

        batch["status"] = "completed"
        batch["n_rows"] = len(df_out)
        batch["output_path"] = str(Path("outputs") / out_path.name)
        processed_batches += 1
        completed_batches += 1

        write_manifest_atomic(manifest_path, manifest)
        elapsed = max(time.time() - started_at, 1e-6)
        avg_seconds_per_batch = elapsed / max(completed_batches, 1)
        remaining_batches = max(total_batches - completed_batches, 0)
        eta_seconds = avg_seconds_per_batch * remaining_batches
        _emit_inference_status(
            job,
            f"Completed inference batch {int(batch_id) + 1}/{total_batches}: "
            f"{completed_batches}/{total_batches} batches done, "
            f"{remaining_batches} remaining, {_format_eta(eta_seconds)}.",
        )

    return manifest


def _in_off_hours_inference_window(now_utc: datetime | None = None) -> bool:
    """Return True when now is in the allowed large-corpus inference window.

    Allowed schedule:
    - Monday-Friday: 10:00 PM to 6:00 AM America/New_York
    - Saturday-Sunday: all day
    """

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    now_est = now_utc.astimezone(ZoneInfo("America/New_York"))
    weekday = now_est.weekday()  # 0=Mon ... 6=Sun
    if weekday >= 5:
        return True
    return now_est.hour >= 22 or now_est.hour < 6


def _seconds_until_next_off_hours_window(now_utc: datetime | None = None) -> float:
    """Return seconds until the next allowed inference window in America/New_York."""

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    eastern = ZoneInfo("America/New_York")
    now_est = now_utc.astimezone(eastern)
    if _in_off_hours_inference_window(now_utc):
        return 0.0

    weekday = now_est.weekday()
    current_day_at_22 = now_est.replace(hour=22, minute=0, second=0, microsecond=0)
    if weekday < 4:
        next_start = current_day_at_22
    elif weekday == 4:
        # Friday daytime -> weekend opens at midnight between Friday/Saturday.
        next_start = (now_est + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        next_start = now_est

    return max((next_start - now_est).total_seconds(), 0.0)


def _wait_for_off_hours_inference_window(log: logging.Logger) -> None:
    if _in_off_hours_inference_window():
        return

    seconds = _seconds_until_next_off_hours_window()
    sleep_seconds = max(seconds, 60.0)
    log.info(
        "Off-hours-only inference enabled; pausing for %.0f seconds until next allowed window.",
        sleep_seconds,
    )
    time.sleep(sleep_seconds)


def _run_single_prompt_batch(
    df_prompts: pd.DataFrame,
    llm_labeler: "LLMLabeler",
    label_config_bundle: "LabelConfigBundle",
    rules_map: dict[str, str],
    label_types: dict[str, str],
) -> pd.DataFrame:
    tasks = df_to_single_prompt_tasks(df_prompts)
    current_label_config = (
        getattr(label_config_bundle, "current", None)
        or getattr(label_config_bundle, "current_config", None)
        or {}
    )

    parent_to_children, child_to_parents, roots = build_label_dependencies(
        current_label_config
    )

    rows: list[dict] = []
    malformed_streak = 0
    streak_limit = int(
        getattr(
            getattr(getattr(llm_labeler, "cfg", None), "llmfirst", None),
            "malformed_json_unit_streak_limit",
            10,
        )
        or 10
    )
    for task in tasks:
        unit_id = str(task.unit_id)
        label_ids = [str(lid) for lid in task.label_ids]

        result = llm_labeler.annotate_multi(
            unit_id=unit_id,
            label_ids=label_ids,
            label_types=task.label_types,
            rules_map=task.rules_map,
            ctx_snippets=task.ctx_snippets,
        )

        preds = result.get("predictions") or {}
        runs = result.get("runs", [])
        json_valid = bool(result.get("json_valid", bool(preds)))
        if not json_valid:
            malformed_streak += 1
            if malformed_streak >= streak_limit:
                raise RuntimeError(
                    "Stopping prompt inference after "
                    f"{malformed_streak} consecutive malformed JSON unit responses. "
                    "This usually indicates an LLM API outage or network connectivity issue."
                )
        else:
            malformed_streak = 0

        parent_preds = {
            (unit_id, str(lid)): info.get("prediction") if isinstance(info, dict) else None
            for lid, info in preds.items()
        }

        for lid in label_ids:
            info = preds.get(str(lid)) or {}
            value = info.get("prediction") if isinstance(info, dict) else None

            gated_ok = evaluate_gating(
                child_id=str(lid),
                unit_id=unit_id,
                parent_preds=parent_preds,
                label_types=task.label_types,
                label_config=current_label_config,
            )
            if not gated_ok:
                value = None

            rows.append(
                {
                    "unit_id": unit_id,
                    "label_id": str(lid),
                    "label_type": task.label_types.get(str(lid), label_types.get(str(lid))),
                    "llm_prediction": value,
                    "llm_consistency": info.get("consistency") if isinstance(info, dict) else None,
                    "llm_reasoning": info.get("reasoning") if isinstance(info, dict) else None,
                    "rag_context": task.ctx_snippets,
                    "llm_runs": runs,
                }
            )

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    return _jsonify_cols(df_out, [c for c in ["rag_context", "llm_runs"] if c in df_out.columns])


def _run_family_prompt_batch(
    df_prompts: pd.DataFrame,
    llm_labeler: "LLMLabeler",
    label_config_bundle: "LabelConfigBundle",
    rules_map: dict[str, str],
    label_types: dict[str, str],
) -> pd.DataFrame:
    tasks = df_to_family_prompt_tasks(df_prompts)
    current_label_config = (
        getattr(label_config_bundle, "current", None)
        or getattr(label_config_bundle, "current_config", None)
        or {}
    )

    ctx_by_pair = {(t.unit_id, t.label_id): t for t in tasks}

    parent_to_children, child_to_parents, roots = build_label_dependencies(
        current_label_config
    )

    all_label_ids = sorted(label_types.keys())
    ordered_labels = list(roots) + [lid for lid in all_label_ids if lid not in roots]

    unit_ids = sorted({t.unit_id for t in tasks})
    rows = []

    n_consistency = int(getattr(getattr(llm_labeler, "cfg", None), "n_consistency", 1) or 1)
    jitter_params = bool(getattr(getattr(llm_labeler, "scCfg", None), "enable_jitter", True))

    for unit_id in unit_ids:
        parent_preds: dict[tuple[str, str], Any] = {}

        for label_id in ordered_labels:
            label_id_str = str(label_id)
            allowed = evaluate_gating(
                child_id=label_id_str,
                unit_id=unit_id,
                parent_preds=parent_preds,
                label_types=label_types,
                label_config=current_label_config,
            )
            if not allowed:
                continue

            task = ctx_by_pair.get((unit_id, label_id_str))
            if task is None:
                ctx_snippets = []
                label_rules = rules_map.get(label_id_str, "")
                label_type = label_types.get(label_id_str, "categorical")
            else:
                ctx_snippets = task.ctx_snippets
                label_rules = task.label_rules
                label_type = task.label_type

            result = llm_labeler.annotate(
                unit_id=unit_id,
                label_id=label_id_str,
                label_type=label_type,
                label_rules=label_rules,
                snippets=ctx_snippets,
                n_consistency=n_consistency,
                jitter_params=jitter_params,
                rag_diagnostics=None,
            )

            prediction = result.get("prediction")
            consistency = result.get("consistency_agreement")
            runs = result.get("runs", [])
            reasoning = None
            if isinstance(runs, list) and runs:
                first_run = runs[0] if isinstance(runs[0], dict) else {}
                raw = first_run.get("raw") if isinstance(first_run, dict) else {}
                if isinstance(raw, dict):
                    reasoning = raw.get("reasoning")

            parent_preds[(unit_id, label_id_str)] = prediction

            rows.append(
                {
                    "unit_id": unit_id,
                    "label_id": label_id_str,
                    "label_type": label_type,
                    "llm_prediction": prediction,
                    "llm_consistency": consistency,
                    "llm_reasoning": reasoning,
                    "rag_context": ctx_snippets,
                    "llm_runs": runs,
                }
            )

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    return _jsonify_cols(df_out, [c for c in ["rag_context", "llm_runs"] if c in df_out.columns])
