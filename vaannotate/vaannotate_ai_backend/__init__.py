"""Lightweight facade for the AI backend integration.

Expose orchestration helpers for both active-learning and inference-only
workflows via :func:`build_next_batch` and :func:`run_inference`.
"""

__version__ = "0.1.0"

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters import BackendResult
    from .experiments import InferenceExperimentResult
    from .orchestration import BackendSession

# Layering overview:
# - core.*: primitives such as DataRepository, EmbeddingStore, and RetrievalCoordinator
# - config: configuration dataclasses for pipelines and orchestration
# - services.*: retrieval, context building, LLM labelers, disagreement/diversity, and family/gating helpers
# - pipelines.*: active learning and inference workflows composed from services
# - orchestrator.py: external entrypoints used by the app and round builder

__all__ = [
    "__version__",
    "BackendResult",
    "build_next_batch",
    "run_inference",
    "export_inputs_from_repo",
    "run_ai_backend_and_collect",
    "CancelledError",
    "BackendSession",
    "InferenceExperimentResult",
    "run_inference_experiments",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports to avoid importing heavy deps at package import."""
    if name in {"build_next_batch", "run_inference"}:
        from .orchestrator import build_next_batch, run_inference

        return {
            "build_next_batch": build_next_batch,
            "run_inference": run_inference,
        }[name]
    if name in {"BackendResult", "export_inputs_from_repo", "run_ai_backend_and_collect"}:
        from .adapters import BackendResult, export_inputs_from_repo, run_ai_backend_and_collect

        return {
            "BackendResult": BackendResult,
            "export_inputs_from_repo": export_inputs_from_repo,
            "run_ai_backend_and_collect": run_ai_backend_and_collect,
        }[name]
    if name == "CancelledError":
        from .utils.runtime import CancelledError

        return CancelledError
    if name == "BackendSession":
        from .orchestration import BackendSession

        return BackendSession
    if name in {"InferenceExperimentResult", "run_inference_experiments"}:
        from .experiments import InferenceExperimentResult, run_inference_experiments

        return {
            "InferenceExperimentResult": InferenceExperimentResult,
            "run_inference_experiments": run_inference_experiments,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
