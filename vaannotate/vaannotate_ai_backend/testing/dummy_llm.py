"""Deterministic test double for the LLM labeler.

This avoids external model calls for end-to-end AI backend tests.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..label_configs import LabelConfigBundle


class DummyLLMLabeler:
    """Keyword-based, deterministic labeler for tests."""

    def __init__(self, label_config: Mapping[str, object] | LabelConfigBundle | None = None):
        self.calls: list[str] = []
        if isinstance(label_config, LabelConfigBundle):
            self.label_config = label_config.current or {}
        else:
            self.label_config = label_config or {}

    def _label_options(self, label_id: str) -> list[str]:
        cfg = self.label_config.get(str(label_id), {}) if isinstance(self.label_config, Mapping) else {}
        opts = cfg.get("options") if isinstance(cfg, Mapping) else None
        return [str(o) for o in opts] if isinstance(opts, (list, tuple)) else []

    @staticmethod
    def summarize_label_rule_for_rerank(_label_id: str, label_rules: Optional[str]) -> str:
        """Return a compact deterministic summary of a label's rules."""

        text = (label_rules or "").strip()
        if len(text) <= 200:
            return text
        return text[:197].rstrip() + "..."

    def _resolve_label_value(self, intent: str, label_id: str) -> str:
        opts = self._label_options(label_id)
        lookup = {"yes": None, "no": None, "uncertain": None}
        for opt in opts:
            lower = opt.lower()
            if "yes" in lower and lookup["yes"] is None:
                lookup["yes"] = opt
            if "no" in lower and lookup["no"] is None:
                lookup["no"] = opt
            if ("uncertain" in lower or "maybe" in lower) and lookup["uncertain"] is None:
                lookup["uncertain"] = opt
        default_map = {"yes": "yes", "no": "no", "uncertain": "uncertain"}
        return lookup.get(intent) or default_map[intent]

    @staticmethod
    def _unit_text(unit_id: str, context_builder, retriever) -> str:
        repo = getattr(context_builder, "repo", None) or getattr(retriever, "_repo", None)
        if repo is None:
            return ""
        try:
            notes = repo.notes
            matches = notes[notes["unit_id"].astype(str) == str(unit_id)]
            if matches.empty:
                return ""
            return str(matches.iloc[0].get("text", ""))
        except Exception:
            return ""

    def label_unit(
        self,
        unit_id: str,
        label_ids: list[str],
        *,
        label_types: Mapping[str, str],
        per_label_rules: Mapping[str, str],
        context_builder,
        retriever,
        llmfirst_cfg,
        json_only: bool = False,
        json_n_consistency: int = 1,
        json_jitter: bool = False,
    ) -> list[dict]:
        """Return hard-coded predictions for each label_id."""

        del json_only, json_n_consistency, json_jitter, llmfirst_cfg, per_label_rules  # unused
        text = self._unit_text(unit_id, context_builder, retriever).lower()
        self.calls.append(str(unit_id))

        rows: list[dict] = []
        for label_id in label_ids:
            label_type = label_types.get(label_id, "categorical") if isinstance(label_types, Mapping) else "categorical"
            lower = text
            intent = "no"
            negative_tokens = ["no pneumonitis", "without pneumonitis", "no evidence of pneumonitis", "no signs of pneumonitis"]
            if any(tok in lower for tok in negative_tokens):
                intent = "no"
            elif "steroid" in lower or "pneumonitis" in lower:
                intent = "yes"
            prediction = self._resolve_label_value(intent, label_id)

            rows.append(
                {
                    "unit_id": str(unit_id),
                    "label_id": str(label_id),
                    "label_type": label_type,
                    "prediction": prediction,
                    "rag_context": [],
                    "consistency": 1.0,
                    "runs": [
                        {
                            "prediction": prediction,
                            "raw_prediction": prediction,
                            "raw": {"reasoning": "dummy"},
                        }
                    ],
                }
            )
        return rows


def make_dummy_llm_labeler(label_config: Mapping[str, object] | LabelConfigBundle | None = None) -> DummyLLMLabeler:
    """Factory helper to build a deterministic DummyLLMLabeler."""

    return DummyLLMLabeler(label_config=label_config)


__all__ = ["DummyLLMLabeler", "make_dummy_llm_labeler"]
