from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ..adapters import _load_label_config_bundle, export_inputs_from_repo
from ..config import OrchestratorConfig, Paths
from ..experiments import _normalize_local_model_overrides
from ..orchestrator import _apply_overrides
from ..orchestration import BackendSession
from ..pipelines.inference import InferencePipeline
from ..pipelines.large_corpus_jobs import (
    PromptInferenceJob,
    PromptPrecomputeJob,
    run_prompt_inference_job,
    run_prompt_precompute_job,
)
from ..project_experiments import _infer_unit_id_column

LOGGER = logging.getLogger(__name__)


def validate_large_corpus_parity(
    project_root: str | Path,
    pheno_id: str,
    labelset_id: str,
    phenotype_level: str,
    labeling_mode: str,
    *,
    cfg_overrides: dict[str, Any] | None = None,
    llm_overrides: dict[str, Any] | None = None,
    workdir: str | Path | None = None,
) -> pd.DataFrame:
    """Run single-pass inference and two-stage inference, asserting parity.

    The helper writes temporary parquet inputs under ``workdir`` (defaulting to
    ``project_root/admin_tools/large_corpus_validation/<uuid>``), runs the
    existing :class:`InferencePipeline` in the requested ``labeling_mode``, then
    executes :func:`run_prompt_precompute_job` followed by
    :func:`run_prompt_inference_job` over the same inputs. Predictions are
    compared on ``[unit_id, label_id, llm_prediction]``; an
    :class:`AssertionError` is raised if they diverge.
    """

    project_root = Path(project_root)
    cfg_overrides = cfg_overrides or {}
    llm_overrides = llm_overrides or {}

    run_id = uuid.uuid4().hex[:8]
    base_dir = Path(workdir) if workdir else project_root / "admin_tools" / "large_corpus_validation" / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    notes_df, ann_df = export_inputs_from_repo(project_root, pheno_id, [])
    notes_df = notes_df.copy()
    notes_df["unit_id"] = _infer_unit_id_column(notes_df, phenotype_level)

    notes_path = base_dir / "notes.parquet"
    ann_path = base_dir / "annotations.parquet"
    notes_df.to_parquet(notes_path)
    ann_df.to_parquet(ann_path)

    normalized_overrides = _normalize_local_model_overrides(cfg_overrides)
    label_config_override = None
    if isinstance(normalized_overrides, dict) and isinstance(
        normalized_overrides.get("label_config"), dict
    ):
        label_config_override = normalized_overrides["label_config"]

    label_config_bundle = _load_label_config_bundle(
        project_root, pheno_id, labelset_id, [], overrides=label_config_override
    )

    cfg = OrchestratorConfig()
    if normalized_overrides:
        _apply_overrides(cfg, dict(normalized_overrides))

    if llm_overrides:
        _apply_overrides(cfg, dict(llm_overrides))

    _apply_overrides(cfg, {"llmfirst": {"inference_labeling_mode": labeling_mode}})

    one_pass_paths = Paths(
        notes_path=str(notes_path),
        annotations_path=str(ann_path),
        outdir=str(base_dir / "one_pass"),
        cache_dir_override=str(base_dir / "cache"),
    )

    session = BackendSession.from_env(one_pass_paths, cfg)
    pipeline: InferencePipeline = session.build_inference_pipeline(
        one_pass_paths, cfg, label_config_bundle, phenotype_level
    )
    df_single = pipeline.run()

    prompt_job = PromptPrecomputeJob(
        job_id=f"validate-precompute-{run_id}",
        project_root=project_root,
        pheno_id=pheno_id,
        labelset_id=labelset_id,
        phenotype_level=phenotype_level,
        labeling_mode=labeling_mode,
        cfg_overrides=cfg_overrides,
        notes_path=notes_path,
        annotations_path=ann_path,
        job_dir=base_dir / "prompt_job",
        batch_size=len(notes_df),
    )
    run_prompt_precompute_job(prompt_job)

    inference_job = PromptInferenceJob(
        job_id=f"validate-infer-{run_id}",
        prompt_job_id=prompt_job.job_id,
        project_root=project_root,
        prompt_job_dir=prompt_job.job_dir,
        phenotype_level=phenotype_level,
        labeling_mode=labeling_mode,
        cfg_overrides=cfg_overrides,
        llm_overrides=llm_overrides,
        job_dir=base_dir / "prompt_inference",
    )
    run_prompt_inference_job(inference_job)

    outputs_dir = Path(
        inference_job.job_dir
        or project_root / "admin_tools" / "prompt_inference" / inference_job.job_id
    )
    outputs_path = outputs_dir / "outputs"
    batches = sorted(outputs_path.glob("outputs_batch_*.parquet"))
    if batches:
        df_two = pd.concat([pd.read_parquet(p) for p in batches], ignore_index=True)
    else:
        df_two = pd.DataFrame(columns=["unit_id", "label_id", "llm_prediction"])

    df_single_cmp = df_single[["unit_id", "label_id", "llm_prediction"]].copy()
    df_single_cmp["label_id"] = df_single_cmp["label_id"].astype(str)

    df_two_cmp = df_two[["unit_id", "label_id", "llm_prediction"]].copy()
    df_two_cmp["label_id"] = df_two_cmp["label_id"].astype(str)

    df_single_cmp = df_single_cmp.sort_values(["unit_id", "label_id"]).reset_index(drop=True)
    df_two_cmp = df_two_cmp.sort_values(["unit_id", "label_id"]).reset_index(drop=True)

    LOGGER.info(
        "Validating large-corpus parity: %s rows single-pass vs %s two-stage",
        len(df_single_cmp),
        len(df_two_cmp),
    )

    pd.testing.assert_frame_equal(df_single_cmp, df_two_cmp, check_dtype=False)
    return df_two
