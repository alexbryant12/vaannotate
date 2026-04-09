from __future__ import annotations

from types import SimpleNamespace

from vaannotate.vaannotate_ai_backend.config import LLMConfig
from vaannotate.vaannotate_ai_backend.llm_backends import AzureOpenAIBackend, ExLlamaV2Backend


def _fake_response(content: str):
    message = SimpleNamespace(content=content, parsed=None)
    choice = SimpleNamespace(message=message, content=content, logprobs=None)
    return SimpleNamespace(choices=[choice])


def test_azure_json_call_retries_once_on_malformed_json():
    backend = object.__new__(AzureOpenAIBackend)
    backend.cfg = LLMConfig(model_name="fake-model")
    backend._last_call_ts = 0.0

    calls = {"count": 0}

    def _create(**kwargs):  # noqa: ANN003
        del kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return _fake_response('{"prediction": "yes"')
        return _fake_response('{"prediction": "yes"}')

    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    result = backend.json_call(
        [{"role": "user", "content": "test"}],
        temperature=0.0,
        logprobs=False,
        top_logprobs=None,
        response_format={"type": "json_object", "json_schema": {"type": "object"}},
    )

    assert calls["count"] == 2
    assert result.data["prediction"] == "yes"


def test_azure_json_call_fails_after_two_invalid_object_attempts():
    backend = object.__new__(AzureOpenAIBackend)
    backend.cfg = LLMConfig(model_name="fake-model")
    backend._last_call_ts = 0.0

    calls = {"count": 0}

    def _create(**kwargs):  # noqa: ANN003
        del kwargs
        calls["count"] += 1
        return _fake_response('["not-an-object"]')

    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    try:
        backend.json_call(
            [{"role": "user", "content": "test"}],
            temperature=0.0,
            logprobs=False,
            top_logprobs=None,
            response_format={"type": "json_object", "json_schema": {"type": "object"}},
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Failed to parse JSON response" in str(exc) or "Expected JSON object" in str(exc)
        assert calls["count"] == 2


def test_exllama_json_call_retries_once_on_non_object_json():
    backend = object.__new__(ExLlamaV2Backend)
    backend.cfg = LLMConfig(model_name="fake-local")
    backend._last_call_ts = 0.0
    backend._json_stop_conditions = ()
    backend._format_messages = lambda messages: "prompt"  # noqa: ARG005

    calls = {"count": 0}

    def _generate(prompt, **kwargs):  # noqa: ANN003
        del prompt, kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return '["bad"]', [], []
        return '{"prediction":"no"}', [], []

    backend._generate = _generate

    result = backend.json_call(
        [{"role": "user", "content": "test"}],
        temperature=0.0,
        logprobs=False,
        top_logprobs=None,
        response_format={"type": "json_object", "json_schema": {"type": "object"}},
    )

    assert calls["count"] == 2
    assert result.data["prediction"] == "no"
