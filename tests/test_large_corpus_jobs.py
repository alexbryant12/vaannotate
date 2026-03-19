from __future__ import annotations

import json
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
    configs: list[dict[str, object]] = []

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
        cfg = _args[1] if len(_args) >= 2 else _kwargs.get("config")
        if cfg is not None:
            configs.append({"llm_backend": getattr(cfg.llm, "backend", None)})
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
        llm_overrides={"backend": "azure_openai"},
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

    assert configs and configs[0]["llm_backend"] == "azure_openai"

    assert os.getenv("AZURE_OPENAI_API_KEY") is None
    assert os.getenv("AZURE_OPENAI_API_VERSION") is None
    assert os.getenv("AZURE_OPENAI_ENDPOINT") is None


def test_prompt_precompute_resumes_with_manifest_overrides(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    job_dir = project_root / "admin_tools" / "prompt_jobs" / "job-2"
    job_dir.mkdir(parents=True)

    manifest_overrides = {"label_config": {"foo": "bar"}}
    manifest_llm_overrides = {"backend": "hf"}
    (job_dir / "job_manifest.json").write_text(
        json.dumps({"cfg_overrides": manifest_overrides, "llm_overrides": manifest_llm_overrides})
    )

    notes_df = pd.DataFrame({"unit_id": ["u1"], "text": ["example"]})
    ann_df = pd.DataFrame({"unit_id": ["u1"], "label": ["y"]})

    applied_overrides: list[dict] = []

    def fake_export_inputs_from_repo(_project_root, _pheno_id, _rounds):  # type: ignore[no-untyped-def]
        return notes_df, ann_df

    def fake_backend_from_env(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return object()

    def fake_load_label_config_bundle(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return type(
            "Bundle",
            (),
            {
                "current_rules_map": staticmethod(lambda *_args, **_kwargs: {}),
                "current_label_types": staticmethod(lambda *_args, **_kwargs: {}),
                "current": {},
            },
        )()

    def fake_run_batches(manifest, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return manifest

    monkeypatch.setattr(jobs, "export_inputs_from_repo", fake_export_inputs_from_repo)
    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(fake_backend_from_env))
    monkeypatch.setattr(jobs, "_load_label_config_bundle", fake_load_label_config_bundle)
    monkeypatch.setattr(jobs, "_run_prompt_precompute_batches", fake_run_batches)
    monkeypatch.setattr(jobs, "_normalize_local_model_overrides", lambda overrides: overrides)
    monkeypatch.setattr(jobs, "_apply_overrides", lambda _cfg, overrides: applied_overrides.append(overrides))

    job = jobs.PromptPrecomputeJob(
        job_id="job-2",
        project_root=project_root,
        pheno_id="p1",
        labelset_id="ls",
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        llm_overrides=None,
        notes_path=None,
        annotations_path=None,
        job_dir=job_dir,
        batch_size=1,
        env_overrides=None,
    )

    jobs.run_prompt_precompute_job(job)

    assert manifest_overrides in applied_overrides
    assert {"llm": manifest_llm_overrides} in applied_overrides

    manifest = jobs.read_manifest(job_dir / "job_manifest.json")
    assert manifest and manifest.get("cfg_overrides") == manifest_overrides
    assert manifest.get("llm_overrides") == manifest_llm_overrides


def test_prompt_inference_resumes_with_manifest_overrides(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    prompt_job_dir = project_root / "admin_tools" / "prompt_jobs" / "prompt-1"
    prompt_job_dir.mkdir(parents=True)
    prompt_manifest = {
        "pheno_id": "p1",
        "labelset_id": "ls",
        "phenotype_level": "single_doc",
        "labeling_mode": "single_prompt",
        "batches": [],
    }
    (prompt_job_dir / "job_manifest.json").write_text(json.dumps(prompt_manifest))

    job_dir = project_root / "admin_tools" / "prompt_inference" / "job-3"
    job_dir.mkdir(parents=True)
    inference_manifest = {
        "cfg_overrides": {"label_config": {"baz": 2}},
        "llm_overrides": {"backend": "bedrock"},
        "batches": [],
    }
    (job_dir / "job_manifest.json").write_text(json.dumps(inference_manifest))

    applied_overrides: list[dict] = []

    class DummySession:
        models = object()
        store = object()

    def fake_backend_from_env(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return DummySession()

    def fake_build_shared_components(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return {"llm_labeler": object()}

    def fake_load_label_config_bundle(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return type(
            "Bundle",
            (),
            {
                "current_rules_map": staticmethod(lambda *_args, **_kwargs: {}),
                "current_label_types": staticmethod(lambda *_args, **_kwargs: {}),
                "current_config": {},
            },
        )()

    def fake_run_batches(manifest, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return manifest

    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(fake_backend_from_env))
    monkeypatch.setattr(jobs, "_build_shared_components", fake_build_shared_components)
    monkeypatch.setattr(jobs, "_load_label_config_bundle", fake_load_label_config_bundle)
    monkeypatch.setattr(jobs, "_run_prompt_inference_batches", fake_run_batches)
    monkeypatch.setattr(jobs, "_normalize_local_model_overrides", lambda overrides: overrides)
    monkeypatch.setattr(jobs, "_apply_overrides", lambda _cfg, overrides: applied_overrides.append(overrides))

    job = jobs.PromptInferenceJob(
        job_id="job-3",
        prompt_job_id="prompt-1",
        project_root=project_root,
        prompt_job_dir=prompt_job_dir,
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        llm_overrides=None,
        job_dir=job_dir,
        batch_limit=None,
    )

    jobs.run_prompt_inference_job(job)

    assert inference_manifest["cfg_overrides"] in applied_overrides
    assert {"llm": inference_manifest["llm_overrides"]} in applied_overrides

    manifest = jobs.read_manifest(job_dir / "job_manifest.json")
    assert manifest and manifest.get("cfg_overrides") == inference_manifest["cfg_overrides"]
    assert manifest.get("llm_overrides") == inference_manifest["llm_overrides"]


def test_prompt_precompute_accepts_external_dataset_and_stages_annotations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)

    notes_source = tmp_path / "notes.feather"
    pd.DataFrame(
        {
            "person_id": ["p1", "p1", "p2"],
            "note_id": ["d1", "d2", "d3"],
            "note_text": ["alpha", "beta", "gamma"],
            "note_type": ["A", "B", "C"],
        }
    ).to_feather(notes_source)

    seen_paths: list[tuple[str, str]] = []

    class DummySession:
        models = object()
        store = object()

    def fake_backend_from_env(paths, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        seen_paths.append((paths.notes_path, paths.annotations_path))
        return DummySession()

    def fake_load_label_config_bundle(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return type(
            "Bundle",
            (),
            {
                "current_rules_map": staticmethod(lambda *_args, **_kwargs: {"L1": "rule"}),
                "current_label_types": staticmethod(lambda *_args, **_kwargs: {"L1": "binary"}),
                "current": {"L1": {"gated_by": None}},
            },
        )()

    def fake_run_batches(manifest, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return manifest

    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(fake_backend_from_env))
    monkeypatch.setattr(jobs, "_load_label_config_bundle", fake_load_label_config_bundle)
    monkeypatch.setattr(jobs, "_run_prompt_precompute_batches", fake_run_batches)

    job = jobs.PromptPrecomputeJob(
        job_id="job-external",
        project_root=project_root,
        pheno_id="ph",
        labelset_id="ls",
        phenotype_level="multi_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        notes_path=notes_source,
        annotations_path=None,
        notes_column_map={
            "patient_icn": "person_id",
            "doc_id": "note_id",
            "text": "note_text",
            "notetype": "note_type",
        },
        batch_size=2,
    )

    jobs.run_prompt_precompute_job(job)

    prompt_job_dir = project_root / "admin_tools" / "prompt_jobs" / "job-external"
    staged_notes = pd.read_parquet(prompt_job_dir / "notes.parquet")
    staged_ann = pd.read_parquet(prompt_job_dir / "annotations.parquet")

    assert list(staged_notes["patient_icn"]) == ["p1", "p1", "p2"]
    assert list(staged_notes["doc_id"]) == ["d1", "d2", "d3"]
    assert list(staged_notes["unit_id"]) == ["p1", "p1", "p2"]
    assert list(staged_notes["notetype"]) == ["A", "B", "C"]
    assert staged_ann.empty
    assert seen_paths == [
        (
            str(prompt_job_dir / "notes.parquet"),
            str(prompt_job_dir / "annotations.parquet"),
        )
    ]

    manifest = jobs.read_manifest(prompt_job_dir / "job_manifest.json")
    assert manifest["input_source"]["source_kind"] == "external_dataset"
    assert manifest["input_source"]["notes_column_map"]["text"] == "note_text"
    assert [batch["unit_ids"] for batch in manifest["batches"]] == [["p1", "p2"]]


def test_family_prompt_precompute_orders_parents_before_descendants(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    job_dir = project_root / "admin_tools" / "prompt_jobs" / "job-family"
    (job_dir / "cache").mkdir(parents=True, exist_ok=True)
    (job_dir / "prompts_family").mkdir(parents=True, exist_ok=True)
    (job_dir / "prompts_single").mkdir(parents=True, exist_ok=True)

    notes_df = pd.DataFrame(
        {
            "patient_icn": ["p1"],
            "doc_id": ["d1"],
            "text": ["family note"],
            "unit_id": ["p1"],
        }
    )
    notes_df.to_parquet(job_dir / "notes.parquet", index=False)
    jobs._empty_annotations_df().to_parquet(job_dir / "annotations.parquet", index=False)

    class DummyStore:
        def build_chunk_index(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        @staticmethod
        def _compute_corpus_fingerprint(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "fp"

    class DummyContextBuilder:
        @staticmethod
        def build_context_for_label(unit_id, label_id, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return [{"doc_id": unit_id, "chunk_id": label_id, "text": f"{unit_id}:{label_id}", "score": 1.0}]

    session = type("Session", (), {"models": object(), "store": DummyStore()})()
    label_config = {
        "Root": {"type": "binary"},
        "Child": {"type": "binary", "gated_by": "Root"},
        "Grandchild": {"type": "binary", "gated_by": "Child"},
    }
    bundle = type(
        "Bundle",
        (),
        {
            "current": label_config,
            "current_config": label_config,
            "current_rules_map": staticmethod(lambda *_args, **_kwargs: {k: "" for k in label_config}),
            "current_label_types": staticmethod(lambda *_args, **_kwargs: {k: "binary" for k in label_config}),
        },
    )()

    monkeypatch.setattr(
        jobs,
        "_build_shared_components",
        lambda *_args, **_kwargs: {  # type: ignore[no-untyped-def]
            "repo": type("Repo", (), {"notes": notes_df})(),
            "store": session.store,
            "context_builder": DummyContextBuilder(),
        },
    )

    manifest = {
        "job_id": "job-family",
        "batches": [{"batch_id": 0, "unit_ids": ["p1"], "status": "pending", "n_tasks": 0, "path": None}],
    }
    job = jobs.PromptPrecomputeJob(
        job_id="job-family",
        project_root=project_root,
        pheno_id="ph",
        labelset_id="ls",
        phenotype_level="multi_doc",
        labeling_mode="family",
        cfg_overrides={},
        job_dir=job_dir,
        batch_size=1,
    )

    jobs._run_prompt_precompute_batches(
        manifest,
        job,
        jobs.OrchestratorConfig(),
        session,
        bundle,
        job_dir / "job_manifest.json",
    )

    prompts = pd.read_parquet(job_dir / "prompts_family" / "prompts_batch_00000.parquet")
    assert list(prompts["label_id"]) == ["Root", "Child", "Grandchild"]
