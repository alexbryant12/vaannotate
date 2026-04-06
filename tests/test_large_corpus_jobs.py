from __future__ import annotations

import json
import os
from datetime import datetime, timezone
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

    def fake_export_inputs_from_repo(_project_root, _pheno_id, _rounds, **_kwargs):  # type: ignore[no-untyped-def]
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

    if configs:
        assert configs[0]["llm_backend"] == "azure_openai"

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

    def fake_export_inputs_from_repo(_project_root, _pheno_id, _rounds, **_kwargs):  # type: ignore[no-untyped-def]
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
    assert manifest
    cfg_overrides = manifest.get("cfg_overrides") if isinstance(manifest, dict) else {}
    assert isinstance(cfg_overrides, dict)
    for key, value in manifest_overrides.items():
        assert cfg_overrides.get(key) == value
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

    class DummyLLMLabeler:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.label_config = {}

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

    monkeypatch.setattr(jobs, "build_llm_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(jobs, "LLMLabeler", DummyLLMLabeler)
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


def test_prompt_inference_defaults_logprobs_off(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    prompt_job_dir = project_root / "admin_tools" / "prompt_jobs" / "prompt-1"
    prompt_job_dir.mkdir(parents=True)
    (prompt_job_dir / "job_manifest.json").write_text(
        json.dumps(
            {
                "pheno_id": "p1",
                "labelset_id": "ls",
                "phenotype_level": "single_doc",
                "labeling_mode": "single_prompt",
                "batches": [],
            }
        )
    )

    captured: dict[str, object] = {}

    class DummyLLMLabeler:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.label_config = {}

    def fake_build_llm_backend(llm_cfg):  # type: ignore[no-untyped-def]
        captured["logprobs"] = llm_cfg.logprobs
        captured["top_logprobs"] = llm_cfg.top_logprobs
        return object()

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

    monkeypatch.setattr(jobs, "build_llm_backend", fake_build_llm_backend)
    monkeypatch.setattr(jobs, "LLMLabeler", DummyLLMLabeler)
    monkeypatch.setattr(jobs, "_load_label_config_bundle", fake_load_label_config_bundle)
    monkeypatch.setattr(jobs, "_run_prompt_inference_batches", lambda manifest, *_args, **_kwargs: manifest)

    job = jobs.PromptInferenceJob(
        job_id="job-default-logprobs-off",
        prompt_job_id="prompt-1",
        project_root=project_root,
        prompt_job_dir=prompt_job_dir,
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        llm_overrides=None,
        job_dir=project_root / "admin_tools" / "prompt_inference" / "job-default-logprobs-off",
    )

    jobs.run_prompt_inference_job(job)

    assert captured["logprobs"] is False
    assert captured["top_logprobs"] == 0


def test_prompt_inference_respects_explicit_logprobs_override(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    prompt_job_dir = project_root / "admin_tools" / "prompt_jobs" / "prompt-1"
    prompt_job_dir.mkdir(parents=True)
    (prompt_job_dir / "job_manifest.json").write_text(
        json.dumps(
            {
                "pheno_id": "p1",
                "labelset_id": "ls",
                "phenotype_level": "single_doc",
                "labeling_mode": "single_prompt",
                "batches": [],
            }
        )
    )

    captured: dict[str, object] = {}

    class DummyLLMLabeler:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.label_config = {}

    def fake_build_llm_backend(llm_cfg):  # type: ignore[no-untyped-def]
        captured["logprobs"] = llm_cfg.logprobs
        captured["top_logprobs"] = llm_cfg.top_logprobs
        return object()

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

    monkeypatch.setattr(jobs, "build_llm_backend", fake_build_llm_backend)
    monkeypatch.setattr(jobs, "LLMLabeler", DummyLLMLabeler)
    monkeypatch.setattr(jobs, "_load_label_config_bundle", fake_load_label_config_bundle)
    monkeypatch.setattr(jobs, "_run_prompt_inference_batches", lambda manifest, *_args, **_kwargs: manifest)

    job = jobs.PromptInferenceJob(
        job_id="job-explicit-logprobs-on",
        prompt_job_id="prompt-1",
        project_root=project_root,
        prompt_job_dir=prompt_job_dir,
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        llm_overrides={"logprobs": True, "top_logprobs": 3},
        job_dir=project_root / "admin_tools" / "prompt_inference" / "job-explicit-logprobs-on",
    )

    jobs.run_prompt_inference_job(job)

    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 3


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


def test_prompt_precompute_single_doc_full_context_skips_retrieval_index(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    notes_df = pd.DataFrame({"unit_id": ["u1"], "patient_icn": ["p1"], "doc_id": ["d1"], "text": ["example"]})
    ann_df = pd.DataFrame({"unit_id": ["u1"], "label": ["y"]})

    build_calls: list[str] = []
    family_kwargs: list[dict[str, object]] = []
    statuses: list[str] = []

    class DummyStore:
        def build_chunk_index(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            build_calls.append("called")

        def _compute_corpus_fingerprint(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "fp-full"

    class DummySession:
        models = object()
        store = DummyStore()

    class DummyBundle:
        current = {"a": {"type": "binary", "rule": "Rule A"}}

        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"a": "Rule A"}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"a": "binary"}

    class DummyContextBuilder:
        def build_context_for_family(self, *_args, **kwargs):  # type: ignore[no-untyped-def]
            family_kwargs.append(kwargs)
            return [{"doc_id": "d1", "chunk_id": "__full__", "text": "full ctx", "score": 1.0, "metadata": {}}]

    monkeypatch.setattr(jobs, "export_inputs_from_repo", lambda *_args, **_kwargs: (notes_df, ann_df))
    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(lambda *_args, **_kwargs: DummySession()))
    monkeypatch.setattr(jobs, "_load_label_config_bundle", lambda *_args, **_kwargs: DummyBundle())
    monkeypatch.setattr(
        jobs,
        "_build_shared_components",
        lambda *_args, **_kwargs: {"repo": type("R", (), {"notes": notes_df})(), "store": DummyStore(), "context_builder": DummyContextBuilder()},
    )

    job = jobs.PromptPrecomputeJob(
        job_id="job-full",
        project_root=project_root,
        pheno_id="p1",
        labelset_id="ls",
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={"llmfirst": {"single_doc_context": "full", "single_doc_full_context_max_chars": 321}},
        batch_size=10,
        status_callback=statuses.append,
    )

    jobs.run_prompt_precompute_job(job)

    assert not build_calls
    assert family_kwargs and family_kwargs[0]["single_doc_context_mode"] == "full"
    assert family_kwargs[0]["full_doc_char_limit"] == 321
    assert any("skipping retrieval index build" in status.lower() for status in statuses)


def test_prompt_precompute_single_doc_single_prompt_defaults_to_full_and_skips_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True)
    notes_df = pd.DataFrame({"unit_id": ["u1"], "patient_icn": ["p1"], "doc_id": ["d1"], "text": ["example"]})
    ann_df = pd.DataFrame({"unit_id": ["u1"], "label": ["y"]})

    build_calls: list[str] = []
    family_kwargs: list[dict[str, object]] = []
    statuses: list[str] = []

    class DummyBundle:
        current = {"a": {"type": "binary", "rule": "Rule A"}}

        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"a": "Rule A"}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {"a": "binary"}

    def _fail_from_env(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("BackendSession.from_env should not be called for default full-context single-doc precompute")

    monkeypatch.setattr(jobs, "export_inputs_from_repo", lambda *_args, **_kwargs: (notes_df, ann_df))
    monkeypatch.setattr(jobs.BackendSession, "from_env", staticmethod(_fail_from_env))
    monkeypatch.setattr(jobs, "_load_label_config_bundle", lambda *_args, **_kwargs: DummyBundle())

    class DummyStore:
        def _compute_corpus_fingerprint(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return "fp-default-full"

    class DummyContextBuilder:
        def build_context_for_family(self, *_args, **kwargs):  # type: ignore[no-untyped-def]
            family_kwargs.append(dict(kwargs))
            return [{"doc_id": "d1", "chunk_id": "__full__", "text": "full ctx", "score": 1.0, "metadata": {}}]

    monkeypatch.setattr(
        jobs,
        "_build_shared_components",
        lambda *_args, **_kwargs: {
            "repo": type("R", (), {"notes": notes_df})(),
            "store": DummyStore(),
            "context_builder": DummyContextBuilder(),
            "label_config": DummyBundle.current,
        },
    )

    job = jobs.PromptPrecomputeJob(
        job_id="job-default-full",
        project_root=project_root,
        pheno_id="p1",
        labelset_id="ls",
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        batch_size=10,
        status_callback=statuses.append,
    )

    jobs.run_prompt_precompute_job(job)

    assert job.cfg_overrides.get("llmfirst", {}).get("single_doc_context") == "full"
    assert family_kwargs and family_kwargs[0]["single_doc_context_mode"] == "full"
    assert any("skipping retrieval index build" in status.lower() for status in statuses)


def test_prompt_inference_batch_limit_caps_processed_batches(monkeypatch, tmp_path: Path) -> None:
    prompt_job_dir = tmp_path / "prompt_job"
    inference_job_dir = tmp_path / "inference_job"
    prompt_job_dir.mkdir(parents=True)
    inference_job_dir.mkdir(parents=True)
    (inference_job_dir / "outputs").mkdir(parents=True)

    for batch_id in range(3):
        pd.DataFrame({"x": [batch_id]}).to_parquet(prompt_job_dir / f"prompts_{batch_id}.parquet", index=False)

    manifest = {
        "batches": [
            {
                "batch_id": i,
                "prompt_batch_path": f"prompts_{i}.parquet",
                "status": "pending",
                "n_rows": 0,
                "output_path": None,
            }
            for i in range(3)
        ]
    }

    job = jobs.PromptInferenceJob(
        job_id="inf-1",
        prompt_job_id="prompt-1",
        project_root=tmp_path,
        prompt_job_dir=prompt_job_dir,
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        llm_overrides=None,
        job_dir=inference_job_dir,
        batch_limit=2,
    )

    monkeypatch.setattr(
        jobs,
        "_run_single_prompt_batch",
        lambda *_args, **_kwargs: pd.DataFrame([{"unit_id": "u1", "label_id": "l1"}]),
    )

    class DummyBundle:
        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

    out = jobs._run_prompt_inference_batches(
        manifest,
        job,
        jobs.OrchestratorConfig(),
        llm_labeler=object(),
        label_config_bundle=DummyBundle(),
        prompt_job_dir=prompt_job_dir,
        inference_job_dir=inference_job_dir,
        manifest_path=inference_job_dir / "job_manifest.json",
    )

    completed_ids = [b["batch_id"] for b in out["batches"] if b["status"] == "completed"]
    pending_ids = [b["batch_id"] for b in out["batches"] if b["status"] != "completed"]
    assert completed_ids == [0, 1]
    assert pending_ids == [2]
    assert (inference_job_dir / "outputs" / "outputs_batch_00000.parquet").exists()
    assert (inference_job_dir / "outputs" / "outputs_batch_00001.parquet").exists()
    assert not (inference_job_dir / "outputs" / "outputs_batch_00002.parquet").exists()


def test_inference_off_hours_window_logic_weekday_and_weekend() -> None:
    # Monday daytime ET -> blocked
    monday_day_utc = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)  # 12:00 ET
    assert jobs._in_off_hours_inference_window(monday_day_utc) is False

    # Monday late ET -> allowed
    monday_night_utc = datetime(2026, 1, 6, 4, 0, tzinfo=timezone.utc)  # 23:00 ET Monday
    assert jobs._in_off_hours_inference_window(monday_night_utc) is True

    # Saturday daytime ET -> allowed
    saturday_utc = datetime(2026, 1, 10, 16, 0, tzinfo=timezone.utc)  # 11:00 ET Saturday
    assert jobs._in_off_hours_inference_window(saturday_utc) is True


def test_inference_off_hours_waits_between_batches(monkeypatch, tmp_path: Path) -> None:
    prompt_job_dir = tmp_path / "prompt_job"
    inference_job_dir = tmp_path / "inference_job"
    prompt_job_dir.mkdir(parents=True)
    inference_job_dir.mkdir(parents=True)
    (inference_job_dir / "outputs").mkdir(parents=True)

    pd.DataFrame({"x": [1]}).to_parquet(prompt_job_dir / "prompts_0.parquet", index=False)

    manifest = {
        "batches": [
            {
                "batch_id": 0,
                "prompt_batch_path": "prompts_0.parquet",
                "status": "pending",
                "n_rows": 0,
                "output_path": None,
            }
        ]
    }

    job = jobs.PromptInferenceJob(
        job_id="inf-off-hours",
        prompt_job_id="prompt-1",
        project_root=tmp_path,
        prompt_job_dir=prompt_job_dir,
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        llm_overrides=None,
        job_dir=inference_job_dir,
        off_hours_only=True,
    )

    waited: list[bool] = []
    monkeypatch.setattr(jobs, "_wait_for_off_hours_inference_window", lambda *_args, **_kwargs: waited.append(True))
    monkeypatch.setattr(
        jobs,
        "_run_single_prompt_batch",
        lambda *_args, **_kwargs: pd.DataFrame([{"unit_id": "u1", "label_id": "l1"}]),
    )

    class DummyBundle:
        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

    jobs._run_prompt_inference_batches(
        manifest,
        job,
        jobs.OrchestratorConfig(),
        llm_labeler=object(),
        label_config_bundle=DummyBundle(),
        prompt_job_dir=prompt_job_dir,
        inference_job_dir=inference_job_dir,
        manifest_path=inference_job_dir / "job_manifest.json",
    )

    assert waited == [True]


def test_prompt_inference_emits_status_after_each_completed_batch(monkeypatch, tmp_path: Path) -> None:
    prompt_job_dir = tmp_path / "prompt_job"
    inference_job_dir = tmp_path / "inference_job"
    prompt_job_dir.mkdir(parents=True)
    inference_job_dir.mkdir(parents=True)
    (inference_job_dir / "outputs").mkdir(parents=True)

    for batch_id in range(2):
        pd.DataFrame({"x": [batch_id]}).to_parquet(prompt_job_dir / f"prompts_{batch_id}.parquet", index=False)

    manifest = {
        "batches": [
            {
                "batch_id": i,
                "prompt_batch_path": f"prompts_{i}.parquet",
                "status": "pending",
                "n_rows": 0,
                "output_path": None,
            }
            for i in range(2)
        ]
    }

    statuses: list[str] = []
    job = jobs.PromptInferenceJob(
        job_id="inf-status",
        prompt_job_id="prompt-1",
        project_root=tmp_path,
        prompt_job_dir=prompt_job_dir,
        phenotype_level="single_doc",
        labeling_mode="single_prompt",
        cfg_overrides={},
        llm_overrides=None,
        job_dir=inference_job_dir,
        status_callback=statuses.append,
    )

    monkeypatch.setattr(
        jobs,
        "_run_single_prompt_batch",
        lambda *_args, **_kwargs: pd.DataFrame([{"unit_id": "u1", "label_id": "l1"}]),
    )

    class DummyBundle:
        @staticmethod
        def current_rules_map(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

        @staticmethod
        def current_label_types(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            return {}

    jobs._run_prompt_inference_batches(
        manifest,
        job,
        jobs.OrchestratorConfig(),
        llm_labeler=object(),
        label_config_bundle=DummyBundle(),
        prompt_job_dir=prompt_job_dir,
        inference_job_dir=inference_job_dir,
        manifest_path=inference_job_dir / "job_manifest.json",
    )

    per_batch_statuses = [msg for msg in statuses if msg.startswith("Completed inference batch ")]
    assert len(per_batch_statuses) == 2
