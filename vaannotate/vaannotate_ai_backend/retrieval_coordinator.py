from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple


@dataclass
class SemanticQuery:
    """A semantic retrieval request containing text and a precomputed embedding."""
    text: str
    embedding: object
    source: str = "semantic"


class RetrievalCoordinator:
    """Coordinate semantic + keyword retrieval into a single candidate pool."""

    def __init__(
        self,
        semantic_searcher: Callable[[SemanticQuery, int], List[dict]],
        keyword_searcher: Callable[[List[str], int], List[dict]],
    ):
        self.semantic_searcher = semantic_searcher
        self.keyword_searcher = keyword_searcher

    def build_candidate_pool(
        self,
        *,
        semantic_queries: Iterable[SemanticQuery],
        keywords: Iterable[str],
        top_k_final: int,
        keyword_fraction: float,
        pool_factor: int = 3,
        oversample: float = 1.5,
    ) -> tuple[list[dict], dict]:
        keyword_fraction = max(0.0, min(1.0, float(keyword_fraction)))
        pool_target = max(1, int(math.ceil(top_k_final * pool_factor * oversample)))
        keyword_fraction = 0.0 if not keywords else keyword_fraction
        semantic_only = keyword_fraction <= 0.0
        keyword_only = keyword_fraction >= 1.0

        semantic_target = pool_target if semantic_only else int(
            math.ceil(pool_target * (1.0 - keyword_fraction))
        )
        keyword_target = pool_target if keyword_only else int(
            math.ceil(pool_target * keyword_fraction)
        )

        def _need(target: int) -> int:
            return max(1, int(math.ceil(max(target, top_k_final) * oversample)))

        semantic_hits = self._run_semantic_search(
            semantic_queries, need=_need(semantic_target)
        )
        keyword_hits = self._run_keyword_search(list(keywords), need=_need(keyword_target))

        semantic_best = self._dedup_channel(semantic_hits, channel="semantic")
        keyword_best = self._dedup_channel(keyword_hits, channel="keyword")

        combined: Dict[Tuple[str, int], dict] = {}
        selected = self._select_from_channel(combined, semantic_best, semantic_target)
        selected += self._select_from_channel(combined, keyword_best, keyword_target)

        remaining = self._remaining_candidates(
            semantic_best, keyword_best, semantic_target, keyword_target
        )
        for hit in remaining:
            if len(selected) >= pool_target:
                break
            key = (str(hit.get("doc_id")), int(hit.get("chunk_id", -1)))
            if key in combined:
                continue
            combined[key] = hit
            selected.append(hit)

        selected.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)

        diagnostics = {
            "pool_target": pool_target,
            "semantic_target": semantic_target,
            "keyword_target": keyword_target,
            "semantic_hits": len(semantic_hits),
            "keyword_hits": len(keyword_hits),
            "semantic_kept": len(semantic_best),
            "keyword_kept": len(keyword_best),
            "final_pool": len(selected),
        }

        return selected, diagnostics

    def _run_semantic_search(
        self, semantic_queries: Iterable[SemanticQuery], need: int
    ) -> list[dict]:
        hits: list[dict] = []
        for query in semantic_queries:
            run_hits = self.semantic_searcher(query, need) or []
            for hit in run_hits:
                hit = dict(hit)
                hit.setdefault("source", f"patient_{query.source}")
                hits.append(hit)
        return hits

    def _run_keyword_search(self, keywords: List[str], need: int) -> list[dict]:
        if not keywords:
            return []
        hits = self.keyword_searcher(keywords, need) or []
        for hit in hits:
            hit.setdefault("source", "keyword")
        return hits

    def _dedup_channel(self, hits: list[dict], channel: str) -> list[dict]:
        best: Dict[Tuple[str, int], dict] = {}
        for hit in hits:
            key = (str(hit.get("doc_id")), int(hit.get("chunk_id", -1)))
            score = float(hit.get("score", 0.0))
            prev = best.get(key)
            if prev is None or score > float(prev.get("score", 0.0)):
                updated = dict(hit)
                updated[channel + "_score"] = score
                updated["score"] = score
                best[key] = updated
        return sorted(best.values(), key=lambda h: float(h.get("score", 0.0)), reverse=True)

    def _select_from_channel(
        self, combined: Dict[Tuple[str, int], dict], hits: list[dict], target: int
    ) -> list[dict]:
        selected: list[dict] = []
        for hit in hits[:target]:
            key = (str(hit.get("doc_id")), int(hit.get("chunk_id", -1)))
            if key in combined:
                existing = combined[key]
                existing["score"] = max(
                    float(existing.get("score", 0.0)), float(hit.get("score", 0.0))
                )
                existing.update({k: v for k, v in hit.items() if k.endswith("_score")})
                continue
            combined[key] = hit
            selected.append(hit)
        return selected

    def _remaining_candidates(
        self,
        semantic_hits: list[dict],
        keyword_hits: list[dict],
        semantic_target: int,
        keyword_target: int,
    ) -> list[dict]:
        leftovers = semantic_hits[semantic_target:] + keyword_hits[keyword_target:]
        leftovers.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)
        return leftovers

