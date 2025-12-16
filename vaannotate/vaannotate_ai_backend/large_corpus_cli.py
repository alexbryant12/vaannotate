from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from .pipelines.large_corpus_jobs import (
    PromptInferenceJob,
    PromptPrecomputeJob,
    run_prompt_inference_job,
    run_prompt_precompute_job,
)

LOG = logging.getLogger(__name__)


def _load_cfg_from_experiments_manifest(
    project_root: Path, experiment_name: str | None, experiments_dir: Path | None = None
) -> dict[str, Any]:
    """Return cfg_overrides for an experiment from experiments.json if available."""

    if not experiment_name:
        return {}

    manifest_dir = experiments_dir or project_root / "admin_tools" / "experiments"
    manifest_path = manifest_dir / "experiments.json"
    if not manifest_path.exists():
        LOG.warning(
            "Experiments manifest not found at %s; ignoring experiment name %s",
            manifest_path,
            experiment_name,
        )
        return {}

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        LOG.warning("Failed to parse experiments manifest at %s", manifest_path)
        return {}

    entry = payload.get(str(experiment_name)) if isinstance(payload, dict) else None
    if isinstance(entry, dict) and isinstance(entry.get("cfg_overrides"), dict):
        return dict(entry["cfg_overrides"])

    LOG.warning("Experiment %s not found in manifest; using default cfg", experiment_name)
    return {}


def _default_job_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def create_prompt_precompute_job(
    project_root: str | Path,
    pheno_id: str,
    labelset_id: str,
    phenotype_level: str,
    labeling_mode: str,
    *,
    batch_size: int = 128,
    cfg_overrides: dict[str, Any] | None = None,
    experiment_name: str | None = None,
    experiments_dir: str | Path | None = None,
    job_id: str | None = None,
    notes_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    job_dir: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> PromptPrecomputeJob:
    """Create and run a prompt precompute job."""

    project_root = Path(project_root)
    overrides = cfg_overrides or _load_cfg_from_experiments_manifest(
        project_root, experiment_name, Path(experiments_dir) if experiments_dir else None
    )

    job = PromptPrecomputeJob(
        job_id=job_id or _default_job_id("prompt-precompute"),
        project_root=project_root,
        pheno_id=pheno_id,
        labelset_id=labelset_id,
        phenotype_level=phenotype_level,
        labeling_mode=labeling_mode,
        cfg_overrides=overrides,
        notes_path=Path(notes_path) if notes_path else None,
        annotations_path=Path(annotations_path) if annotations_path else None,
        job_dir=Path(job_dir) if job_dir else None,
        batch_size=batch_size,
        env_overrides={str(k): str(v) for k, v in (env_overrides or {}).items() if str(v)},
    )

    run_prompt_precompute_job(job)
    return job


def create_prompt_inference_job(
    project_root: str | Path,
    prompt_job_id: str,
    phenotype_level: str,
    labeling_mode: str,
    *,
    cfg_overrides: dict[str, Any] | None = None,
    llm_overrides: dict[str, Any] | None = None,
    experiment_name: str | None = None,
    experiments_dir: str | Path | None = None,
    job_id: str | None = None,
    prompt_job_dir: str | Path | None = None,
    job_dir: str | Path | None = None,
    batch_limit: int | None = None,
) -> PromptInferenceJob:
    """Create and run a prompt inference job."""

    project_root = Path(project_root)
    overrides = cfg_overrides or _load_cfg_from_experiments_manifest(
        project_root, experiment_name, Path(experiments_dir) if experiments_dir else None
    )

    job = PromptInferenceJob(
        job_id=job_id or _default_job_id("prompt-infer"),
        prompt_job_id=prompt_job_id,
        project_root=project_root,
        prompt_job_dir=Path(prompt_job_dir) if prompt_job_dir else None,
        phenotype_level=phenotype_level,
        labeling_mode=labeling_mode,
        cfg_overrides=overrides,
        llm_overrides=llm_overrides,
        job_dir=Path(job_dir) if job_dir else None,
        batch_limit=batch_limit,
    )

    run_prompt_inference_job(job)
    return job


def _parse_json_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Large-corpus prompt pipeline entrypoints")
    sub = parser.add_subparsers(dest="command", required=True)

    precompute = sub.add_parser("precompute", help="Precompute prompts for a corpus")
    precompute.add_argument("project_root", type=Path)
    precompute.add_argument("pheno_id")
    precompute.add_argument("labelset_id")
    precompute.add_argument("phenotype_level")
    precompute.add_argument("labeling_mode", choices=["family", "single_prompt"])
    precompute.add_argument("--batch-size", type=int, default=128)
    precompute.add_argument("--job-id")
    precompute.add_argument("--job-dir", type=Path)
    precompute.add_argument("--notes-path", type=Path)
    precompute.add_argument("--annotations-path", type=Path)
    precompute.add_argument("--cfg", help="JSON overrides or path to JSON file")
    precompute.add_argument("--experiment-name", help="Name from experiments manifest")
    precompute.add_argument("--experiments-dir", type=Path, help="Override experiments manifest dir")

    infer = sub.add_parser("infer", help="Run LLM inference over precomputed prompts")
    infer.add_argument("project_root", type=Path)
    infer.add_argument("prompt_job_id")
    infer.add_argument("phenotype_level")
    infer.add_argument("labeling_mode", choices=["family", "single_prompt"])
    infer.add_argument("--job-id")
    infer.add_argument("--prompt-job-dir", type=Path)
    infer.add_argument("--job-dir", type=Path)
    infer.add_argument("--batch-limit", type=int)
    infer.add_argument("--cfg", help="JSON overrides or path to JSON file")
    infer.add_argument("--llm-cfg", help="LLM-only overrides or path to JSON file")
    infer.add_argument("--experiment-name", help="Name from experiments manifest")
    infer.add_argument("--experiments-dir", type=Path, help="Override experiments manifest dir")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.command == "precompute":
        overrides = _parse_json_arg(args.cfg)
        create_prompt_precompute_job(
            project_root=args.project_root,
            pheno_id=args.pheno_id,
            labelset_id=args.labelset_id,
            phenotype_level=args.phenotype_level,
            labeling_mode=args.labeling_mode,
            batch_size=args.batch_size,
            cfg_overrides=overrides,
            experiment_name=args.experiment_name,
            experiments_dir=args.experiments_dir,
            job_id=args.job_id,
            notes_path=args.notes_path,
            annotations_path=args.annotations_path,
            job_dir=args.job_dir,
        )
    elif args.command == "infer":
        overrides = _parse_json_arg(args.cfg)
        llm_overrides = _parse_json_arg(args.llm_cfg)
        create_prompt_inference_job(
            project_root=args.project_root,
            prompt_job_id=args.prompt_job_id,
            phenotype_level=args.phenotype_level,
            labeling_mode=args.labeling_mode,
            cfg_overrides=overrides,
            llm_overrides=llm_overrides,
            experiment_name=args.experiment_name,
            experiments_dir=args.experiments_dir,
            job_id=args.job_id,
            prompt_job_dir=args.prompt_job_dir,
            job_dir=args.job_dir,
            batch_limit=args.batch_limit,
        )


if __name__ == "__main__":
    main()
