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


def test_prompt_precompute_external_dataset_family_generates_all_labels(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    dataset_path = tmp_path / "notes.feather"
    pd.DataFrame(
        {
            "person_id": ["p1"],
            "note_id": ["d1"],
            "note_text": ["Patient has diabetes documented in the chart."],
        }
    ).to_feather(dataset_path)

    class DummyLabelBundle:
        current = {
            "root": {"type": "binary", "rule": "Root rule"},
            "child": {"type": "binary", "rule": "Child rule", "gated_by": "root"},
        }

        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"root": "Root rule", "child": "Child rule"}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"root": "binary", "child": "binary"}

    class DummyStore:
        def build_chunk_index(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def _compute_corpus_fingerprint(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "fp-1"

    class DummySession:
        models = object()
        store = DummyStore()

    class DummyContextBuilder:
        def build_context_for_label(self, unit_id, label_id, label_rules):  # type: ignore[no-untyped-def]
            return [{"doc_id": unit_id, "chunk_id": label_id, "text": label_rules, "score": 1.0, "metadata": {}}]

        def build_context_for_family(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return [{"doc_id": "d1", "chunk_id": "all", "text": "context", "score": 1.0, "metadata": {}}]

    class DummyLLMLabeler:
        def build_single_label_prompt_payload(self, **kwargs):  # type: ignore[no-untyped-def]
            return {"prompt": {"system": kwargs["label_id"], "user": "ctx"}, "messages": [], "response_format": {}}

        def build_multi_label_prompt_payload(self, **kwargs):  # type: ignore[no-untyped-def]
            return {"prompt": {"system": ",".join(kwargs["label_ids"]), "user": "ctx"}, "messages": [], "response_format": {}}

    def fake_backend_from_env(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return DummySession()

    def fake_load_label_config_bundle(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return DummyLabelBundle()

    def fake_build_shared_components(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        notes_df = pd.read_parquet(project_root / "admin_tools" / "prompt_jobs" / "job-ext" / "notes.parquet")
        return {
            "repo": type("R", (), {"notes": notes_df})(),
            "store": DummyStore(),
            "context_builder": DummyContextBuilder(),
            "llm_labeler": DummyLLMLabeler(),
        }

    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(fake_backend_from_env))
    monkeypatch.setattr(jobs, "_load_label_config_bundle", fake_load_label_config_bundle)
    monkeypatch.setattr(jobs, "_build_shared_components", fake_build_shared_components)

    job = jobs.PromptPrecomputeJob(
        job_id="job-ext",
        project_root=project_root,
        pheno_id="p1",
        labelset_id="ls",
        phenotype_level="multi_doc",
        labeling_mode="family",
        cfg_overrides={},
        dataset_path=dataset_path,
        dataset_column_map={"patient_icn": "person_id", "doc_id": "note_id", "text": "note_text"},
        batch_size=10,
    )

    jobs.run_prompt_precompute_job(job)

    out_path = project_root / "admin_tools" / "prompt_jobs" / "job-ext" / "prompts_family" / "prompts_batch_00000.parquet"
    df = pd.read_parquet(out_path)
    assert set(df["label_id"]) == {"root", "child"}
    assert "prompt_payload" in df.columns
    assert (project_root / "admin_tools" / "prompt_jobs" / "job-ext" / "notes.parquet").exists()


def test_prompt_precompute_single_prompt_stores_prompt_payload(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    notes_df = pd.DataFrame({"unit_id": ["u1"], "patient_icn": ["p1"], "doc_id": ["d1"], "text": ["example"]})
    ann_df = pd.DataFrame({"unit_id": ["u1"], "label": ["y"]})

    class DummyStore:
        def build_chunk_index(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def _compute_corpus_fingerprint(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "fp-2"

    class DummySession:
        models = object()
        store = DummyStore()

    class DummyBundle:
        current = {"a": {"type": "binary", "rule": "Rule A"}, "b": {"type": "binary", "rule": "Rule B"}}

        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"a": "Rule A", "b": "Rule B"}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"a": "binary", "b": "binary"}

    class DummyContextBuilder:
        def build_context_for_family(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return [{"doc_id": "d1", "chunk_id": "all", "text": "family ctx", "score": 1.0, "metadata": {}}]

    class DummyLLMLabeler:
        def build_multi_label_prompt_payload(self, **kwargs):  # type: ignore[no-untyped-def]
            return {"prompt": {"system": "sys", "user": "ctx"}, "messages": [], "response_format": {}, "label_ids": kwargs["label_ids"]}

    monkeypatch.setattr(jobs, "export_inputs_from_repo", lambda *_args, **_kwargs: (notes_df, ann_df))
    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(lambda *_args, **_kwargs: DummySession()))
    monkeypatch.setattr(jobs, "_load_label_config_bundle", lambda *_args, **_kwargs: DummyBundle())
    monkeypatch.setattr(
        jobs,
        "_build_shared_components",
        lambda *_args, **_kwargs: {"repo": type("R", (), {"notes": notes_df})(), "store": DummyStore(), "context_builder": DummyContextBuilder(), "llm_labeler": DummyLLMLabeler()},
    )

    job = jobs.PromptPrecomputeJob(
        job_id="job-single",
        project_root=project_root,
        pheno_id="p1",
        labelset_id="ls",
        phenotype_level="multi_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        batch_size=10,
    )

    jobs.run_prompt_precompute_job(job)

    out_path = project_root / "admin_tools" / "prompt_jobs" / "job-single" / "prompts_single" / "prompts_batch_00000.parquet"
    df = pd.read_parquet(out_path)
    assert "prompt_payload" in df.columns
    assert "a" in str(df.iloc[0]["prompt_payload"])
