from __future__ import annotations

import json
from pathlib import Path

from vaannotate.vaannotate_ai_backend import large_corpus_cli


def test_resume_prompt_inference_job_uses_existing_manifest(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    job_dir = project_root / "admin_tools" / "prompt_inference" / "infer-1"
    job_dir.mkdir(parents=True)
    (job_dir / "job_manifest.json").write_text(
        json.dumps(
            {
                "job_id": "infer-1",
                "prompt_job_id": "prompt-123",
                "phenotype_level": "single_doc",
                "labeling_mode": "single_prompt",
                "cfg_overrides": {"llmfirst": {"single_doc_context": "full"}},
                "llm_overrides": {"backend": "hf"},
                "off_hours_only": True,
            }
        ),
        encoding="utf-8",
    )

    observed: list[object] = []

    def fake_run(job):  # type: ignore[no-untyped-def]
        observed.append(job)

    monkeypatch.setattr(large_corpus_cli, "run_prompt_inference_job", fake_run)

    resumed = large_corpus_cli.resume_prompt_inference_job(job_dir)

    assert observed
    assert resumed.job_id == "infer-1"
    assert resumed.prompt_job_id == "prompt-123"
    assert resumed.job_dir == job_dir
    assert resumed.project_root == project_root
    assert resumed.prompt_job_dir == project_root / "admin_tools" / "prompt_jobs" / "prompt-123"
    assert resumed.off_hours_only is True
    assert resumed.cfg_overrides.get("llmfirst", {}).get("single_doc_context") == "full"


def test_resume_prompt_inference_job_can_override_off_hours(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    job_dir = project_root / "admin_tools" / "prompt_inference" / "infer-2"
    job_dir.mkdir(parents=True)
    (job_dir / "job_manifest.json").write_text(
        json.dumps(
            {
                "job_id": "infer-2",
                "prompt_job_id": "prompt-222",
                "phenotype_level": "single_doc",
                "labeling_mode": "family",
                "off_hours_only": True,
            }
        ),
        encoding="utf-8",
    )

    observed: list[object] = []

    def fake_run(job):  # type: ignore[no-untyped-def]
        observed.append(job)

    monkeypatch.setattr(large_corpus_cli, "run_prompt_inference_job", fake_run)

    resumed = large_corpus_cli.resume_prompt_inference_job(
        job_dir,
        off_hours_only=False,
        batch_limit=3,
    )

    assert observed
    assert resumed.off_hours_only is False
    assert resumed.batch_limit == 3
