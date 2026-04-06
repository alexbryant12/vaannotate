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
from .utils.job_manifest import read_manifest

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
    llm_overrides: dict[str, Any] | None = None,
    experiment_name: str | None = None,
    experiments_dir: str | Path | None = None,
    job_id: str | None = None,
    corpus_id: str | None = None,
    corpus_path: str | Path | None = None,
    notes_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    dataset_column_map: dict[str, str] | None = None,
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
        llm_overrides=llm_overrides,
        corpus_id=corpus_id,
        corpus_path=Path(corpus_path) if corpus_path else None,
        notes_path=Path(notes_path) if notes_path else None,
        annotations_path=Path(annotations_path) if annotations_path else None,
        dataset_path=Path(dataset_path) if dataset_path else None,
        dataset_column_map=dataset_column_map,
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
    off_hours_only: bool = False,
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
        off_hours_only=off_hours_only,
    )

    run_prompt_inference_job(job)
    return job


def _infer_project_root_from_inference_job_dir(job_dir: Path) -> Path | None:
    parts = tuple(job_dir.parts)
    if len(parts) >= 3 and parts[-3:] == ("admin_tools", "prompt_inference", job_dir.name):
        return job_dir.parents[2]
    return None


def resume_prompt_inference_job(
    job_dir: str | Path | None = None,
    *,
    batch_limit: int | None = None,
    off_hours_only: bool | None = None,
    project_root: str | Path | None = None,
    prompt_job_dir: str | Path | None = None,
) -> PromptInferenceJob:
    """Resume a prompt inference job from an existing inference job directory."""

    resolved_job_dir = Path(job_dir) if job_dir else Path.cwd()
    manifest_path = resolved_job_dir / "job_manifest.json"
    manifest = read_manifest(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected inference manifest at {manifest_path}")

    prompt_job_id = str(manifest.get("prompt_job_id") or "").strip()
    if not prompt_job_id:
        raise ValueError(f"Missing prompt_job_id in {manifest_path}")

    phenotype_level = str(manifest.get("phenotype_level") or "").strip()
    if not phenotype_level:
        raise ValueError(f"Missing phenotype_level in {manifest_path}")

    labeling_mode = str(manifest.get("labeling_mode") or "").strip()
    if labeling_mode not in {"family", "single_prompt"}:
        raise ValueError(f"Invalid labeling_mode {labeling_mode!r} in {manifest_path}")

    cfg_overrides = manifest.get("cfg_overrides")
    llm_overrides = manifest.get("llm_overrides")
    default_off_hours_only = bool(manifest.get("off_hours_only"))

    resolved_project_root = (
        Path(project_root)
        if project_root
        else _infer_project_root_from_inference_job_dir(resolved_job_dir)
    )
    if resolved_project_root is None:
        raise ValueError(
            "Unable to infer project_root from inference job directory. "
            "Pass --project-root explicitly."
        )

    resolved_prompt_job_dir = (
        Path(prompt_job_dir)
        if prompt_job_dir
        else resolved_project_root / "admin_tools" / "prompt_jobs" / prompt_job_id
    )

    job = PromptInferenceJob(
        job_id=str(manifest.get("job_id") or resolved_job_dir.name),
        prompt_job_id=prompt_job_id,
        project_root=resolved_project_root,
        prompt_job_dir=resolved_prompt_job_dir,
        phenotype_level=phenotype_level,
        labeling_mode=labeling_mode,
        cfg_overrides=cfg_overrides if isinstance(cfg_overrides, dict) else {},
        llm_overrides=llm_overrides if isinstance(llm_overrides, dict) else None,
        job_dir=resolved_job_dir,
        batch_limit=batch_limit,
        off_hours_only=default_off_hours_only if off_hours_only is None else bool(off_hours_only),
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
    precompute.add_argument("--corpus-id")
    precompute.add_argument("--corpus-path", type=Path)
    precompute.add_argument("--notes-path", type=Path)
    precompute.add_argument("--annotations-path", type=Path)
    precompute.add_argument("--dataset-path", type=Path, help="External corpus table for prompt generation.")
    precompute.add_argument("--dataset-columns", help="JSON mapping from canonical columns to source columns.")
    precompute.add_argument("--cfg", help="JSON overrides or path to JSON file")
    precompute.add_argument("--llm-cfg", help="LLM-only overrides or path to JSON file")
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
    infer.add_argument(
        "--off-hours-only",
        action="store_true",
        help="Restrict large-corpus inference to 10pm-6am ET on weekdays and all weekend.",
    )
    infer.add_argument("--cfg", help="JSON overrides or path to JSON file")
    infer.add_argument("--llm-cfg", help="LLM-only overrides or path to JSON file")
    infer.add_argument("--experiment-name", help="Name from experiments manifest")
    infer.add_argument("--experiments-dir", type=Path, help="Override experiments manifest dir")

    resume_infer = sub.add_parser(
        "resume-infer",
        help="Resume an existing prompt inference job from its job directory (defaults to CWD).",
    )
    resume_infer.add_argument("job_dir", type=Path, nargs="?", default=Path.cwd())
    resume_infer.add_argument("--project-root", type=Path, help="Override project root inference.")
    resume_infer.add_argument("--prompt-job-dir", type=Path, help="Override prompt job directory.")
    resume_infer.add_argument("--batch-limit", type=int, help="Process at most this many pending batches.")
    off_hours_group = resume_infer.add_mutually_exclusive_group()
    off_hours_group.add_argument(
        "--off-hours-only",
        dest="off_hours_only",
        action="store_const",
        const=True,
        help="Force off-hours-only inference scheduling for this resume run.",
    )
    off_hours_group.add_argument(
        "--disable-off-hours-only",
        dest="off_hours_only",
        action="store_const",
        const=False,
        help="Disable off-hours-only inference scheduling for this resume run.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.command == "precompute":
        overrides = _parse_json_arg(args.cfg)
        llm_overrides = _parse_json_arg(args.llm_cfg)
        create_prompt_precompute_job(
            project_root=args.project_root,
            pheno_id=args.pheno_id,
            labelset_id=args.labelset_id,
            phenotype_level=args.phenotype_level,
            labeling_mode=args.labeling_mode,
            batch_size=args.batch_size,
            cfg_overrides=overrides,
            llm_overrides=llm_overrides,
            experiment_name=args.experiment_name,
            experiments_dir=args.experiments_dir,
            job_id=args.job_id,
            corpus_id=args.corpus_id,
            corpus_path=args.corpus_path,
            notes_path=args.notes_path,
            annotations_path=args.annotations_path,
            dataset_path=args.dataset_path,
            dataset_column_map=_parse_json_arg(args.dataset_columns),
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
            off_hours_only=bool(args.off_hours_only),
        )
    elif args.command == "resume-infer":
        resume_prompt_inference_job(
            job_dir=args.job_dir,
            batch_limit=args.batch_limit,
            off_hours_only=args.off_hours_only,
            project_root=args.project_root,
            prompt_job_dir=args.prompt_job_dir,
        )


if __name__ == "__main__":
    main()
