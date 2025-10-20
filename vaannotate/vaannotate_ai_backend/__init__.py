"""Lightweight facade for the AI backend integration."""

__version__ = "0.1.0"

from .orchestrator import build_next_batch
from .adapters import BackendResult, export_inputs_from_repo, run_ai_backend_and_collect

__all__ = [
    "__version__",
    "BackendResult",
    "build_next_batch",
    "export_inputs_from_repo",
    "run_ai_backend_and_collect",
]
