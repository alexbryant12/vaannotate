from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vaannotate.vaannotate_ai_backend import config as ai_config
from vaannotate.vaannotate_ai_backend.core import embeddings as embeddings_mod
from vaannotate.vaannotate_ai_backend.core.data import DataRepository


class _StubEmbedder:
    def __init__(self, name: str) -> None:
        self.name_or_path = name
        self.calls = 0

    def encode(
        self,
        texts,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self.calls += 1
        n = len(texts)
        # simple deterministic embeddings
        data = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
        if normalize_embeddings and data.size:
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data = data / norms
        return data


class _StubCrossEncoder:
    pass


class _DummyIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ntotal = 0

    def add(self, matrix: np.ndarray) -> None:
        self._ntotal = int(matrix.shape[0])

    @property
    def ntotal(self) -> int:
        return self._ntotal


_FAISS_STUB = types.SimpleNamespace(
    IndexFlatIP=lambda dim: _DummyIndex(dim),
    IndexFlatL2=lambda dim: _DummyIndex(dim),
    METRIC_INNER_PRODUCT=0,
    METRIC_L2=1,
    write_index=lambda idx, path: None,
    read_index=lambda path: (_DummyIndex(0)),
)


@pytest.fixture(autouse=True)
def _patch_faiss(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(embeddings_mod, "faiss", _FAISS_STUB)
    yield


def _build_repo_notes(level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    notes = pd.DataFrame(
        [
            {"patient_icn": "p1", "doc_id": "d1", "text": "alpha beta"},
            {"patient_icn": "p1", "doc_id": "d2", "text": "gamma delta"},
        ]
    )
    ann = pd.DataFrame(
        [
            {
                "round_id": "r1",
                "unit_id": "p1" if level == "multi_doc" else "d1",
                "doc_id": "d1",
                "label_id": "lab",
                "reviewer_id": "rev",
                "label_value": "yes",
            }
        ]
    )
    repo = DataRepository(notes, ann, phenotype_level=level)
    return repo.notes.copy(), repo.ann.copy()


def _make_store(cache_dir: Path, embedder_name: str) -> tuple[embeddings_mod.EmbeddingStore, _StubEmbedder]:
    embedder = _StubEmbedder(embedder_name)
    models = embeddings_mod.Models(embedder=embedder, reranker=_StubCrossEncoder())
    store = embeddings_mod.EmbeddingStore(models, cache_dir=str(cache_dir))
    return store, embedder


def _rag_index_cfg() -> tuple[ai_config.RAGConfig, ai_config.IndexConfig]:
    rag_cfg = ai_config.RAGConfig(chunk_size=32, chunk_overlap=0)
    index_cfg = ai_config.IndexConfig(type="flat")
    return rag_cfg, index_cfg


def test_cached_embeddings_reuse_for_single_doc(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    rag_cfg, index_cfg = _rag_index_cfg()

    notes_multi, _ = _build_repo_notes("multi_doc")
    store_multi, embedder_multi = _make_store(cache_dir, "stub-embed")
    store_multi.build_chunk_index(notes_multi, rag_cfg, index_cfg)
    assert embedder_multi.calls == 1
    multi_units = {meta["unit_id"] for meta in store_multi.chunk_meta}
    assert multi_units == {"p1"}

    notes_single, _ = _build_repo_notes("single_doc")
    store_single, embedder_single = _make_store(cache_dir, "stub-embed")
    store_single.build_chunk_index(notes_single, rag_cfg, index_cfg)

    assert embedder_single.calls == 0  # reused cached embeddings
    single_units = {meta["unit_id"] for meta in store_single.chunk_meta}
    assert single_units == {"d1", "d2"}
    assert set(store_single.unit_to_chunk_idxs.keys()) == {"d1", "d2"}


def test_cached_embeddings_reuse_for_multi_doc(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    rag_cfg, index_cfg = _rag_index_cfg()

    notes_single, _ = _build_repo_notes("single_doc")
    store_single, embedder_single = _make_store(cache_dir, "stub-embed")
    store_single.build_chunk_index(notes_single, rag_cfg, index_cfg)
    assert embedder_single.calls == 1
    single_units = {meta["unit_id"] for meta in store_single.chunk_meta}
    assert single_units == {"d1", "d2"}

    notes_multi, _ = _build_repo_notes("multi_doc")
    store_multi, embedder_multi = _make_store(cache_dir, "stub-embed")
    store_multi.build_chunk_index(notes_multi, rag_cfg, index_cfg)

    assert embedder_multi.calls == 0
    multi_units = {meta["unit_id"] for meta in store_multi.chunk_meta}
    assert multi_units == {"p1"}
    assert set(store_multi.unit_to_chunk_idxs.keys()) == {"p1"}
