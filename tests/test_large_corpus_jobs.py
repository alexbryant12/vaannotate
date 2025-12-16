from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend.pipelines import large_corpus_jobs as jobs


def test_prompt_precompute_applies_env_overrides(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)

    notes_df = pd.DataFrame(
        {"unit_id": ["u1"], "text": ["example"], "patient_icn": ["p1"]}
    )
    ann_df = pd.DataFrame({"unit_id": ["u1"], "label": ["y"]})

    manifest_environments: list[dict[str, str | None]] = []

    class DummyLabelBundle:
        current: dict = {}

        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

    class DummySession:
        models = object()
        store = object()

    def fake_export_inputs_from_repo(_project_root, _pheno_id, _rounds):  # type: ignore[no-untyped-def]
        return notes_df, ann_df

    def fake_backend_from_env(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        manifest_environments.append(
            {
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            }
        )
        return DummySession()

    def fake_build_shared_components(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return {"repo": type("R", (), {"notes": notes_df})(), "store": object(), "context_builder": object()}

    def fake_load_label_config_bundle(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return DummyLabelBundle()

    def fake_run_batches(manifest, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        manifest_environments.append(
            {
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            }
        )
        return manifest

    monkeypatch.setattr(jobs, "export_inputs_from_repo", fake_export_inputs_from_repo)
    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(fake_backend_from_env))
    monkeypatch.setattr(jobs, "_build_shared_components", fake_build_shared_components)
    monkeypatch.setattr(jobs, "_load_label_config_bundle", fake_load_label_config_bundle)
    monkeypatch.setattr(jobs, "_run_prompt_precompute_batches", fake_run_batches)

    env_overrides = {
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",
        "AZURE_OPENAI_ENDPOINT": "https://example.azure.com/",
    }

    job = jobs.PromptPrecomputeJob(
        job_id="job-1",
        project_root=project_root,
        pheno_id="p1",
        labelset_id="ls",
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        notes_path=None,
        annotations_path=None,
        job_dir=None,
        batch_size=1,
        env_overrides=env_overrides,
    )

    assert os.getenv("AZURE_OPENAI_API_KEY") is None

    jobs.run_prompt_precompute_job(job)

    assert manifest_environments
    for snapshot in manifest_environments:
        assert snapshot["AZURE_OPENAI_API_KEY"] == "test-key"
        assert snapshot["AZURE_OPENAI_API_VERSION"] == "2024-06-01"
        assert snapshot["AZURE_OPENAI_ENDPOINT"] == "https://example.azure.com/"

    assert os.getenv("AZURE_OPENAI_API_KEY") is None
    assert os.getenv("AZURE_OPENAI_API_VERSION") is None
    assert os.getenv("AZURE_OPENAI_ENDPOINT") is None

