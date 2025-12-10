from types import SimpleNamespace
from typing import Any

import numpy as np

from vaannotate.vaannotate_ai_backend.services.rag_retriever import RAGRetriever
from vaannotate.vaannotate_ai_backend.services.context_builder import ContextBuilder


def test_manual_and_exemplar_queries_used_for_semantic_and_ce() -> None:
    engine = RAGRetriever.__new__(RAGRetriever)

    engine.cfg = SimpleNamespace(
        per_label_topk=2,
        min_context_chunks=1,
        mmr_multiplier=1,
        neighbor_hops=0,
        use_keywords=False,
        exemplar_K=2,
    )

    class DummyStore:
        def __init__(self) -> None:
            self.chunk_meta = [
                {
                    "doc_id": "doc1",
                    "chunk_id": 0,
                    "text": "chunk text",
                    "note_type": "demo",
                }
            ]
            self.X = np.array([[1.0, 0.0]], dtype=np.float32)

        def get_patient_chunk_indices(self, unit_id: str) -> list[int]:
            return [0]

        def _embed(self, texts: list[str]) -> np.ndarray:
            # Encode each query as a simple 2-d vector to keep shapes consistent
            return np.array([[float(len(t)), 0.0] for t in texts], dtype=np.float32)

    engine.store = DummyStore()

    engine.label_configs = {"lab": {"search_query": "manual override"}}
    engine._get_label_query_embs = lambda *_args, **_kwargs: None
    engine._get_label_query_texts = lambda *_args, **_kwargs: ["exemplar text"]
    engine._reciprocal_rank_fusion = lambda runs: [hit for run in runs for hit in run]
    engine._bm25_hits_for_patient = lambda *_args, **_kwargs: []
    engine._mmr_select = lambda *_args, **_kwargs: []
    engine._extract_meta = lambda _meta: {}

    captured_diags: dict[str, Any] = {}

    cross_queries: list[str] = []

    def _cross_scores_cached(query: str, cand_texts: list[str]) -> list[float]:
        cross_queries.append(query)
        return [1.0 for _ in cand_texts]

    def _set_last_diagnostics(_unit: str, _label: str, diagnostics: dict, **_kwargs: Any) -> None:
        captured_diags.update(diagnostics)

    engine._cross_scores_cached = _cross_scores_cached  # type: ignore[attr-defined]
    engine.set_last_diagnostics = _set_last_diagnostics  # type: ignore[assignment]

    result = engine.retrieve_for_patient_label("patient1", "lab", "label rule text")

    assert result, "RAG pipeline should return ranked snippets"
    assert captured_diags.get("queries") == ["manual override"]
    assert captured_diags.get("query_sources") == ["manual"]
    # CE re-ranking should have been invoked for the manual query
    assert cross_queries == ["manual override"]


def test_context_builder_rag_is_consistent_with_retriever() -> None:
    """Ensure ContextBuilder returns a subset of retriever chunks in RAG mode."""

    class DummyRepo:
        phenotype_level = "rag"

    class DummyStore:
        pass

    class DummyRetriever:
        def __init__(self) -> None:
            self.store = DummyStore()

        def retrieve_for_patient_label(self, *_, **__) -> list[dict]:
            return [
                {"chunk_id": "c1", "doc_id": "d1", "text": "snippet 1", "score": 0.9},
                {"chunk_id": "c2", "doc_id": "d2", "text": "snippet 2", "score": 0.8},
            ]

    retriever = DummyRetriever()
    builder = ContextBuilder(DummyRepo(), DummyStore(), retriever, SimpleNamespace(), SimpleNamespace())

    unit_id = "unit-123"
    label_id = "label-abc"
    rules = "demo rules"

    snips_retriever = retriever.retrieve_for_patient_label(unit_id, label_id, rules)
    snips_builder = builder.build_context_for_label(
        unit_id,
        label_id,
        rules,
        single_doc_context_mode=False,
    )

    ids_retriever = {s["chunk_id"] for s in snips_retriever}
    ids_builder = {s["chunk_id"] for s in snips_builder}

    assert ids_builder.issubset(ids_retriever)
