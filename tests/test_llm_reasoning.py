from __future__ import annotations

import json
import types

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend import config as ai_config
from vaannotate.vaannotate_ai_backend.label_configs import LabelConfigBundle
from vaannotate.vaannotate_ai_backend.llm_backends import JSONCallResult, _first_choice_or_raise
from vaannotate.vaannotate_ai_backend.services.llm_labeler import LLMLabeler


def test_llm_annotator_omits_reasoning_when_disabled(tmp_path):
    calls: dict[str, object] = {}

    class DummyBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def json_call(self, messages, **kwargs):  # noqa: D401
            calls["messages"] = list(messages)
            payload = {"prediction": "yes", "reasoning": "explanation"}
            return JSONCallResult(
                data=payload,
                content=json.dumps(payload),
                raw_response=None,
                latency_s=0.01,
                logprobs=None,
            )

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.n_consistency = 1
    llm_cfg.n_consistency = 1
    llm_cfg.n_consistency = 1
    llm_cfg.include_reasoning = False
    annotator = LLMLabeler(
        DummyBackend(llm_cfg),
        LabelConfigBundle(),
        llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )

    result = annotator.annotate(
        unit_id="unit-1",
        label_id="Flag",
        label_type="categorical",
        label_rules="",
        snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
        n_consistency=1,
        jitter_params=False,
    )

    system_message = next(msg["content"] for msg in calls["messages"] if msg["role"] == "system")
    assert "prediction" in system_message
    assert "reasoning" not in system_message

    run = result["runs"][0]
    assert run.get("reasoning") in (None, "")


def test_llm_annotator_multicategorical_inline_keys(tmp_path):
    calls: dict[str, object] = {}

    class DummyBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def json_call(self, messages, **kwargs):  # noqa: D401
            calls["messages"] = list(messages)
            calls["response_format"] = kwargs.get("response_format")
            payload = {
                "reasoning": "evidence",
                "Option A": "Yes",
                "Option B": "No",
            }
            return JSONCallResult(
                data=payload,
                content=json.dumps(payload),
                raw_response=None,
                latency_s=0.01,
                logprobs=None,
            )

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.include_reasoning = True
    annotator = LLMLabeler(
        DummyBackend(llm_cfg),
        LabelConfigBundle(),
        llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )
    annotator.label_config = {"Flag": {"options": ["Option A", "Option B"]}}

    result = annotator.annotate(
        unit_id="unit-2",
        label_id="Flag",
        label_type="categorical_multi",
        label_rules="",
        snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
        n_consistency=1,
        jitter_params=False,
    )

    run = result["runs"][0]
    assert run["prediction"] == "Option A"
    assert run.get("raw_prediction") == "Option A"

    response_format = calls.get("response_format") or {}
    schema = (response_format.get("json_schema") if isinstance(response_format, dict) else {}) or {}
    pred_schema = (schema.get("properties") or {}).get("prediction", {})
    assert "anyOf" in pred_schema


def test_multicategorical_probe_uses_json_consistency(tmp_path):
    calls: dict[str, int] = {"json": 0, "fc": 0}

    class DummyBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def json_call(self, messages, **kwargs):  # noqa: D401
            calls["json"] += 1
            payload = {"prediction": ["Option A"]}
            return JSONCallResult(
                data=payload,
                content=json.dumps(payload),
                raw_response=None,
                latency_s=0.01,
                logprobs=None,
            )

        def forced_choice(self, *_, **__):  # noqa: D401
            calls["fc"] += 1
            raise AssertionError("forced_choice should not run for multi-categorical labels")

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.n_consistency = 1
    annotator = LLMLabeler(
        DummyBackend(llm_cfg),
        LabelConfigBundle(),
        llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )
    annotator.label_config = {"Flag": {"options": ["Option A", "Option B"]}}

    class DummyContextBuilder:
        def build_context_for_label(self, *_, **__):  # noqa: D401
            return [{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}]

    class DummyRetriever:
        def get_last_diagnostics(self, *_, **__):  # noqa: D401
            return {}

    cfg = types.SimpleNamespace(
        topk=1,
        single_doc_context="rag",
        single_doc_full_context_max_chars=None,
        fc_enable=True,
    )

    rows = annotator.label_unit(
        unit_id="unit-3",
        label_ids=["Flag"],
        label_types={"Flag": "categorical_multi"},
        per_label_rules={"Flag": ""},
        context_builder=DummyContextBuilder(),
        retriever=DummyRetriever(),
        llmfirst_cfg=cfg,
    )

    assert calls["fc"] == 0
    assert calls["json"] == 1
    assert rows[0]["prediction"] == "Option A"


def test_label_bundle_preserves_multiselect_routing(tmp_path):
    calls: dict[str, int] = {"json": 0, "fc": 0}

    class DummyBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def json_call(self, messages, **kwargs):  # noqa: D401
            calls["json"] += 1
            payload = {"prediction": {"Option A": "Yes"}}
            return JSONCallResult(
                data=payload,
                content=json.dumps(payload),
                raw_response=None,
                latency_s=0.01,
                logprobs=None,
            )

        def forced_choice(self, *_, **__):  # noqa: D401
            calls["fc"] += 1
            raise AssertionError("forced_choice should not run for multi-categorical labels")

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.n_consistency = 1
    label_config = {"Flag": {"type": "categorical_multi", "options": ["Option A", "Option B"]}}
    bundle = LabelConfigBundle(current=label_config)
    annotator = LLMLabeler(
        DummyBackend(llm_cfg),
        bundle,
        llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )

    class DummyContextBuilder:
        def build_context_for_label(self, *_, **__):  # noqa: D401
            return [{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}]

    class DummyRetriever:
        def get_last_diagnostics(self, *_, **__):  # noqa: D401
            return {}
        label_configs = label_config

    label_types = bundle.current_label_types(label_config)
    assert label_types["Flag"] == "categorical_multi"

    cfg = types.SimpleNamespace(
        topk=1,
        single_doc_context="rag",
        single_doc_full_context_max_chars=None,
        fc_enable=True,
    )

    rows = annotator.label_unit(
        unit_id="unit-3",
        label_ids=["Flag"],
        label_types=label_types,
        per_label_rules={"Flag": ""},
        context_builder=DummyContextBuilder(),
        retriever=DummyRetriever(),
        llmfirst_cfg=cfg,
    )

    assert calls["fc"] == 0
    assert calls["json"] == 1
    assert rows[0]["prediction"] == "Option A"


def test_first_choice_or_raise_returns_first_choice() -> None:
    resp = types.SimpleNamespace(choices=["first", "second"])
    assert _first_choice_or_raise(resp, operation="json_call") == "first"


def test_first_choice_or_raise_raises_for_empty_choices() -> None:
    resp = types.SimpleNamespace(choices=[])
    try:
        _first_choice_or_raise(resp, operation="forced_choice")
    except RuntimeError as exc:
        assert "forced_choice returned no choices" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError for empty choices")


def test_multilabel_prompt_switches_to_multi_select(tmp_path):
    calls: dict[str, object] = {}

    class DummyBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def json_call(self, messages, **kwargs):  # noqa: D401
            calls["messages"] = list(messages)
            calls["response_format"] = kwargs.get("response_format")
            payload = {
                "Multi": {"prediction": {"Option A": "Yes", "Option B": "No"}},
                "Single": {"prediction": "Option B"},
            }
            return JSONCallResult(
                data=payload,
                content=json.dumps(payload),
                raw_response=None,
                latency_s=0.01,
                logprobs=None,
            )

    llm_cfg = ai_config.LLMConfig()
    annotator = LLMLabeler(
        DummyBackend(llm_cfg),
        LabelConfigBundle(),
        llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )
    annotator.label_config = {
        "Multi": {"options": ["Option A", "Option B"]},
        "Single": {"options": ["Option A", "Option B"]},
    }

    result = annotator.annotate_multi(
        unit_id="unit-4",
        label_ids=["Multi", "Single"],
        label_types={"Multi": "categorical_multi", "Single": "categorical"},
        rules_map={"Multi": "", "Single": ""},
        ctx_snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
    )

    system_message = next(msg["content"] for msg in calls["messages"] if msg["role"] == "system")
    assert "Select all supported options" in system_message
    assert "selected option key" in system_message

    response_format = calls.get("response_format") or {}
    schema = (response_format.get("json_schema") if isinstance(response_format, dict) else {}) or {}
    multi_schema = (((schema.get("properties") or {}).get("Multi", {}) or {}).get("properties") or {}).get("prediction", {})
    assert "anyOf" in multi_schema

    object_schema = next((alt for alt in multi_schema.get("anyOf", []) if alt.get("type") == "object"), {})
    option_schema = (object_schema.get("properties") or {}).get("Option A", {})
    enums = option_schema.get("enum") or []
    assert {"Yes", "No", "Unknown"}.issubset(set(enums))

    assert result["predictions"]["Multi"]["prediction"] == "Option A"
    assert result["predictions"]["Single"]["prediction"] == "Option B"


def test_multilabel_prompt_omits_reasoning_when_disabled(tmp_path):
    calls: dict[str, object] = {}

    class DummyBackend:
        def __init__(self, cfg):
            self.cfg = cfg

        def json_call(self, messages, **kwargs):  # noqa: D401
            calls["messages"] = list(messages)
            calls["response_format"] = kwargs.get("response_format")
            payload = {"Flag": {"prediction": "yes", "reasoning": "ignored"}}
            return JSONCallResult(
                data=payload,
                content=json.dumps(payload),
                raw_response=None,
                latency_s=0.01,
                logprobs=None,
            )

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.include_reasoning = False
    annotator = LLMLabeler(
        DummyBackend(llm_cfg),
        LabelConfigBundle(),
        llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )

    annotator.annotate_multi(
        unit_id="unit-reasoning-off",
        label_ids=["Flag"],
        label_types={"Flag": "categorical"},
        rules_map={"Flag": ""},
        ctx_snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
    )

    system_message = next(msg["content"] for msg in calls["messages"] if msg["role"] == "system")
    assert "reasoning (optional)" not in system_message
    assert "prediction (required)" in system_message
    response_format = calls.get("response_format") or {}
    schema = (response_format.get("json_schema") if isinstance(response_format, dict) else {}) or {}
    label_schema = (schema.get("properties") or {}).get("Flag", {}) or {}
    assert "reasoning" not in ((label_schema.get("properties") or {}))


def test_single_label_few_shot_answer_is_normalized_to_json(tmp_path):
    class DummyBackend:
        def json_call(self, *_, **__):  # pragma: no cover - not invoked in this test
            raise AssertionError("json_call should not be reached")

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.few_shot_examples = {"Flag": [{"context": "c1", "answer": "yes"}]}
    annotator = LLMLabeler(
        llm_backend=DummyBackend(),
        label_config_bundle=LabelConfigBundle(),
        llm_config=llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )

    payload = annotator.build_single_label_prompt_payload(
        label_id="Flag",
        label_type="categorical",
        label_rules="",
        snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
    )
    messages = payload["messages"]
    assistant_example = next(msg["content"] for msg in messages if msg["role"] == "assistant")
    assert json.loads(assistant_example) == {"prediction": "yes"}


def test_multi_label_few_shot_answer_is_wrapped_by_label(tmp_path):
    class DummyBackend:
        def json_call(self, *_, **__):  # pragma: no cover - not invoked in this test
            raise AssertionError("json_call should not be reached")

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.include_reasoning = False
    llm_cfg.few_shot_examples = {
        "Flag": [{"context": "c1", "answer": '{"prediction":"yes","reasoning":"because"}'}]
    }
    annotator = LLMLabeler(
        llm_backend=DummyBackend(),
        label_config_bundle=LabelConfigBundle(),
        llm_config=llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )

    payload = annotator.build_multi_label_prompt_payload(
        label_ids=["Flag"],
        label_types={"Flag": "categorical"},
        rules_map={"Flag": ""},
        ctx_snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
    )
    assistant_example = next(
        msg["content"] for msg in payload["messages"] if msg["role"] == "assistant"
    )
    parsed = json.loads(assistant_example)
    assert parsed == {"Flag": {"prediction": "yes"}}


def test_few_shot_reasoning_field_is_included_when_enabled(tmp_path):
    class DummyBackend:
        def json_call(self, *_, **__):  # pragma: no cover - not invoked in this test
            raise AssertionError("json_call should not be reached")

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.include_reasoning = True
    llm_cfg.few_shot_examples = {
        "Flag": [{"context": "c1", "answer": "yes", "reasoning": "because evidence supports it"}]
    }
    annotator = LLMLabeler(
        llm_backend=DummyBackend(),
        label_config_bundle=LabelConfigBundle(),
        llm_config=llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )

    payload = annotator.build_single_label_prompt_payload(
        label_id="Flag",
        label_type="categorical",
        label_rules="",
        snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
    )
    assistant_example = next(
        msg["content"] for msg in payload["messages"] if msg["role"] == "assistant"
    )
    assert json.loads(assistant_example) == {
        "prediction": "yes",
        "reasoning": "because evidence supports it",
    }


def test_multilabel_prompt_supports_per_label_reasoning_flags(tmp_path):
    class DummyBackend:
        def json_call(self, *_, **__):  # pragma: no cover - not invoked in this test
            raise AssertionError("json_call should not be reached")

    llm_cfg = ai_config.LLMConfig()
    llm_cfg.include_reasoning = False
    llm_cfg.include_reasoning_by_label = {"A": True, "B": False}
    annotator = LLMLabeler(
        llm_backend=DummyBackend(),
        label_config_bundle=LabelConfigBundle(),
        llm_config=llm_cfg,
        sc_cfg=ai_config.SCJitterConfig(),
        cache_dir=str(tmp_path),
    )

    payload = annotator.build_multi_label_prompt_payload(
        label_ids=["A", "B"],
        label_types={"A": "categorical", "B": "categorical"},
        rules_map={"A": "", "B": ""},
        ctx_snippets=[{"doc_id": "doc-1", "chunk_id": 1, "text": "note", "metadata": {}}],
    )
    schema = payload["response_format"]["json_schema"]
    a_props = ((schema.get("properties") or {}).get("A", {}) or {}).get("properties") or {}
    b_props = ((schema.get("properties") or {}).get("B", {}) or {}).get("properties") or {}
    assert "reasoning" in a_props
    assert "reasoning" not in b_props
