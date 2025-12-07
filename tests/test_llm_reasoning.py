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
from vaannotate.vaannotate_ai_backend.llm_backends import JSONCallResult
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
