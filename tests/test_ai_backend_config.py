"""Tests for AI backend configuration helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


def _install_stub_modules() -> None:
    class _DummyModel:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple stub
            pass

    stubs = {
        "numpy": {},
        "pandas": {},
        "sentence_transformers": {
            "SentenceTransformer": _DummyModel,
            "CrossEncoder": _DummyModel,
        },
        "langchain_text_splitters": {
            "RecursiveCharacterTextSplitter": type("RecursiveCharacterTextSplitter", (), {}),
        },
        "langchain": {},
        "langchain.text_splitter": {
            "RecursiveCharacterTextSplitter": type("RecursiveCharacterTextSplitter", (), {}),
        },
    }
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module


_install_stub_modules()


def _load_engine_module():
    module_path = ROOT / "vaannotate" / "vaannotate_ai_backend" / "engine.py"
    spec = importlib.util.spec_from_file_location(
        "vaannotate.vaannotate_ai_backend.engine", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load engine module for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE_MODULE = _load_engine_module()


def _load_config_module():
    module_path = ROOT / "vaannotate" / "vaannotate_ai_backend" / "config.py"
    spec = importlib.util.spec_from_file_location(
        "vaannotate.vaannotate_ai_backend.config", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load config module for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG_MODULE = _load_config_module()


def test_llm_config_respects_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """LLMConfig should read environment variables at instantiation time."""

    LLMConfig = CONFIG_MODULE.LLMConfig

    local_dir = tmp_path / "LocalModel"
    local_dir.mkdir()

    monkeypatch.setenv("LLM_BACKEND", "exllamav2")
    monkeypatch.setenv("LOCAL_LLM_MODEL_DIR", str(local_dir))
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    cfg_local = LLMConfig()
    assert cfg_local.backend == "exllamav2"
    assert cfg_local.local_model_dir == str(local_dir)

    monkeypatch.setenv("LLM_BACKEND", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")

    cfg_azure = LLMConfig()
    assert cfg_azure.backend == "azure"
    assert cfg_azure.azure_endpoint == "https://example.azure.com"
    assert cfg_azure.azure_api_key == "key"


def test_cross_encoder_default_max_length_applied() -> None:
    reranker = type("_StubRerankerNoMax", (), {})()

    ENGINE_MODULE._ensure_default_ce_max_length(reranker, default=777)

    assert hasattr(reranker, "max_length")
    assert reranker.max_length == 777


def test_cross_encoder_existing_max_length_preserved() -> None:
    reranker = type("_StubRerankerWithMax", (), {"max_length": 1024})()

    ENGINE_MODULE._ensure_default_ce_max_length(reranker, default=777)

    assert reranker.max_length == 1024
