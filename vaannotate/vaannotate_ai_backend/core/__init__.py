"""Core primitives for the VAAnnotate AI backend.

Exports are loaded lazily so importing ``vaannotate_ai_backend.core`` does not
eagerly import heavyweight ML dependencies (for example torch/sentence-transformers)
on CLI code paths that only need lightweight config types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .data import DataRepository
    from .embeddings import EmbeddingStore, Models, build_models_from_env
    from .index_config import IndexConfig
    from .retrieval import RetrievalCoordinator, SemanticQuery

__all__ = [
    "DataRepository",
    "EmbeddingStore",
    "IndexConfig",
    "Models",
    "build_models_from_env",
    "RetrievalCoordinator",
    "SemanticQuery",
]


def __getattr__(name: str):
    if name == "DataRepository":
        from .data import DataRepository

        return DataRepository
    if name in {"EmbeddingStore", "Models", "build_models_from_env"}:
        from .embeddings import EmbeddingStore, Models, build_models_from_env

        return {
            "EmbeddingStore": EmbeddingStore,
            "Models": Models,
            "build_models_from_env": build_models_from_env,
        }[name]
    if name == "IndexConfig":
        from .index_config import IndexConfig

        return IndexConfig
    if name in {"RetrievalCoordinator", "SemanticQuery"}:
        from .retrieval import RetrievalCoordinator, SemanticQuery

        return {"RetrievalCoordinator": RetrievalCoordinator, "SemanticQuery": SemanticQuery}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
