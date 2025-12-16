from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..adapters import _load_label_config_bundle, export_inputs_from_repo
from ..config import OrchestratorConfig, Paths
from ..experiments import _normalize_local_model_overrides
from ..label_configs import LabelConfigBundle
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
    notes_path: Path | None = None
    annotations_path: Path | None = None
    job_dir: Path | None = None
    batch_size: int = 128
    env_overrides: dict[str, str] | None = None


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

    log = logging.getLogger(__name__)
    log.info("Starting prompt precompute job %s", job.job_id)

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
        if not job.llm_overrides and isinstance(manifest, dict):
            manifest_llm = manifest.get("llm_overrides")
            if isinstance(manifest_llm, dict):
                job.llm_overrides = manifest_llm

        if job.notes_path is not None and job.annotations_path is not None:
            notes_path = job.notes_path
            ann_path = job.annotations_path
        else:
            notes_df, ann_df = export_inputs_from_repo(job.project_root, job.pheno_id, [])
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

        session_paths = Paths(
            notes_path=str(notes_path),
            annotations_path=str(ann_path),
            outdir=str(job_dir / "_session"),
            cache_dir_override=str(job_dir / "cache"),
        )
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
                "batch_size": job.batch_size,
                "batches": [],
            }

        repo_notes = notes_df if job.notes_path is None else read_table(str(notes_path))
        manifest = _initialize_and_update_batches_for_prompt_precompute(manifest, job, repo_notes)
        manifest = _run_prompt_precompute_batches(
            manifest,
            job,
            cfg,
            session,
            label_config_bundle,
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
    session: BackendSession,
    label_config_bundle: LabelConfigBundle,
) -> dict:
    log = logging.getLogger(__name__)

    job_dir = job.job_dir or job.project_root / "admin_tools" / "prompt_jobs" / job.job_id

    paths = Paths(
        notes_path=str(job_dir / "notes.parquet"),
        annotations_path=str(job_dir / "annotations.parquet"),
        outdir=str(job_dir / "_session"),
        cache_dir_override=str(job_dir / "cache"),
    )

    components = _build_shared_components(
        paths,
        cfg,
        label_config_bundle,
        phenotype_level=job.phenotype_level,
        models=session.models,
        store=session.store,
    )

    repo = components["repo"]
    store = components["store"]
    context_builder = components["context_builder"]

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
    label_ids = sorted(label_types.keys())

    rag_fingerprint = store._compute_corpus_fingerprint(repo.notes, cfg.rag)

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

        if job.labeling_mode == "single_prompt":
            log.info("Building single-prompt batch %s with %d units", batch_id, len(unit_ids))
            for unit_id in unit_ids:
                ctx_snippets = context_builder.build_context_for_family(
                    unit_id,
                    label_ids=label_ids,
                    rules_map=rules_map,
                    topk_per_label=cfg.rag.top_k_final,
                    max_snippets=None,
                    max_chars=getattr(cfg.llmfirst, "single_prompt_max_chars", None),
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
                        meta={
                            "pheno_id": job.pheno_id,
                            "labelset_id": job.labelset_id,
                            "phenotype_level": job.phenotype_level,
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
            log.info("Building family-prompt batch %s with %d units", batch_id, len(unit_ids))
            label_config = label_config_bundle.current or {}
            _parent_to_children, _child_to_parents, roots = build_label_dependencies(label_config)
            ordered_labels = list(roots)
            for lid in sorted(label_types.keys()):
                if lid not in ordered_labels:
                    ordered_labels.append(lid)

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
                            meta={
                                "pheno_id": job.pheno_id,
                                "labelset_id": job.labelset_id,
                                "phenotype_level": job.phenotype_level,
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
            log.warning("Unknown labeling mode %s; skipping batch %s", job.labeling_mode, batch_id)
            continue

        batch["status"] = "completed"
        batch["n_tasks"] = len(tasks)

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
    log.info("Starting prompt inference job %s", job.job_id)

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

    cfg = OrchestratorConfig()
    overrides = _normalize_local_model_overrides(job.cfg_overrides or {})
    if overrides:
        _apply_overrides(cfg, dict(overrides))

    llm_overrides = job.llm_overrides or {}
    if llm_overrides:
        _apply_overrides(cfg, {"llm": llm_overrides})
        if "llmfirst" in llm_overrides:
            _apply_overrides(cfg, {"llmfirst": llm_overrides.get("llmfirst")})

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

    session_paths = Paths(
        notes_path=str(prompt_job_dir / "notes.parquet"),
        annotations_path=str(prompt_job_dir / "annotations.parquet"),
        outdir=str(job_dir / "_session"),
        cache_dir_override=str(job_dir / "cache"),
    )
    session = BackendSession.from_env(session_paths, cfg)

    components = _build_shared_components(
        session_paths,
        cfg,
        label_config_bundle,
        phenotype_level=job.phenotype_level,
        models=session.models,
        store=session.store,
    )
    llm_labeler: LLMLabeler = components["llm_labeler"]

    manifest_path = job_dir / "job_manifest.json"
    manifest = read_manifest(manifest_path)
    if not manifest:
        manifest = {
            "job_id": job.job_id,
            "prompt_job_id": job.prompt_job_id,
            "labeling_mode": job.labeling_mode,
            "phenotype_level": job.phenotype_level,
            "cfg_overrides": job.cfg_overrides,
            "llm_overrides": job.llm_overrides or {},
            "batches": [],
        }

    manifest = _initialize_and_update_batches_for_prompt_inference(manifest, job, prompt_manifest)
    manifest = _run_prompt_inference_batches(
        manifest,
        job,
        cfg,
        llm_labeler,
        label_config_bundle,
        prompt_job_dir,
        job_dir,
    )

    write_manifest_atomic(manifest_path, manifest)


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
) -> dict:
    log = logging.getLogger(__name__)

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

    for batch in manifest.get("batches", []):
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

    return manifest


def _run_single_prompt_batch(
    df_prompts: pd.DataFrame,
    llm_labeler: "LLMLabeler",
    label_config_bundle: "LabelConfigBundle",
    rules_map: dict[str, str],
    label_types: dict[str, str],
) -> pd.DataFrame:
    tasks = df_to_single_prompt_tasks(df_prompts)

    parent_to_children, child_to_parents, roots = build_label_dependencies(
        label_config_bundle.current_config
    )

    rows: list[dict] = []
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
                label_config=label_config_bundle.current_config,
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

    ctx_by_pair = {(t.unit_id, t.label_id): t for t in tasks}

    parent_to_children, child_to_parents, roots = build_label_dependencies(
        label_config_bundle.current_config
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
                label_config=label_config_bundle.current_config,
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
