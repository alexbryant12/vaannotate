"""Service-layer utilities for active learning and inference pipelines."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .context_builder import ContextBuilder
    from .diversity import DiversitySelector
    from .disagreement import DisagreementScorer
    from .llm_labeler import LLMLabeler, LLM_RECORDER
    from .selection import ActiveLearningSelector
    from .uncertainty import LLMUncertaintyScorer


__all__ = [
    "ContextBuilder",
    "DiversitySelector",
    "DisagreementScorer",
    "LLMLabeler",
    "LLM_RECORDER",
    "ActiveLearningSelector",
    "LLMUncertaintyScorer",
]


def __getattr__(name: str) -> Any:
    if name in {"ContextBuilder"}:
        from .context_builder import ContextBuilder

        return ContextBuilder
    if name in {"DiversitySelector"}:
        from .diversity import DiversitySelector

        return DiversitySelector
    if name in {"DisagreementScorer"}:
        from .disagreement import DisagreementScorer

        return DisagreementScorer
    if name in {"LLMLabeler", "LLM_RECORDER"}:
        from .llm_labeler import LLMLabeler, LLM_RECORDER

        return {
            "LLMLabeler": LLMLabeler,
            "LLM_RECORDER": LLM_RECORDER,
        }[name]
    if name in {"ActiveLearningSelector"}:
        from .selection import ActiveLearningSelector

        return ActiveLearningSelector
    if name in {"LLMUncertaintyScorer"}:
        from .uncertainty import LLMUncertaintyScorer

        return LLMUncertaintyScorer
    raise AttributeError(name)
