from types import SimpleNamespace
from typing import Any

import numpy as np

from vaannotate.vaannotate_ai_backend.engine import RAGRetriever


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
    assert captured_diags.get("queries") == ["manual override", "exemplar text"]
    assert captured_diags.get("query_sources") == ["manual", "exemplar"]
    # CE re-ranking should have been invoked for each query text
    assert cross_queries == ["manual override", "exemplar text"]
