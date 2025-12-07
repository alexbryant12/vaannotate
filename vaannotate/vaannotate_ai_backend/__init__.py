"""Lightweight facade for the AI backend integration.

Expose orchestration helpers for both active-learning and inference-only
workflows via :func:`build_next_batch` and :func:`run_inference`.
"""

__version__ = "0.1.0"

from .orchestrator import build_next_batch, run_inference
from .orchestration import BackendSession
from .adapters import BackendResult, export_inputs_from_repo, run_ai_backend_and_collect
from .experiments import InferenceExperimentResult, run_inference_experiments
from .utils.runtime import CancelledError

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
