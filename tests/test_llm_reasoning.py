from __future__ import annotations

import json

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
