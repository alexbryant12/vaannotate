from __future__ import annotations

"""Utilities for constructing RAG contexts for unit/label pairs."""

from typing import Dict, List, Mapping, Optional

from collections import Counter
import numpy as np

from ..core.data import DataRepository
from ..core.embeddings import EmbeddingStore
from ..core.retrieval import RetrievalCoordinator, SemanticQuery
from ..label_configs import LabelConfigBundle


_FULL_DOC_CONTEXT_FALLBACK_CHARS = 12000


class ContextBuilder:
    """Canonical builder for RAG contexts spanning notes, annotations, and label metadata."""

    def __init__(
        self,
        data_repo: DataRepository,
        emb_store: EmbeddingStore,
        retriever,
        rag_config,
        label_config_bundle: LabelConfigBundle,
    ):
        self.repo = data_repo
        self.store = emb_store
        self.retriever = retriever
        self.cfg = rag_config
        self.label_config_bundle = label_config_bundle

    def build_context_for_label(
        self,
        unit_id: str,
        label_id: str,
        label_rules: str,
        *,
        topk_override: int | None = None,
        min_k_override: int | None = None,
        mmr_lambda_override: float | None = None,
        single_doc_context_mode: str = "rag",
        full_doc_char_limit: int | None = None,
    ) -> list[dict]:
        mode_raw = single_doc_context_mode if isinstance(single_doc_context_mode, str) else "rag"
        mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else "rag"
        resolved_unit_id = str(unit_id)
        if getattr(self.repo, "phenotype_level", "").strip().lower() == "single_doc":
            resolver = getattr(self.repo, "doc_id_for_unit", None)
            if callable(resolver):
                resolved = resolver(unit_id)
                if resolved:
                    resolved_unit_id = str(resolved)
        if self.repo.phenotype_level == "single_doc" and mode == "full":
            doc_id = resolved_unit_id
            text = self.repo.notes_by_doc().get(doc_id)
            if not isinstance(text, str) or not text:
                return []

            limit = None
            if full_doc_char_limit is not None:
                try:
                    limit = int(full_doc_char_limit)
                except (TypeError, ValueError):
                    limit = None
            if limit is None or limit <= 0:
                limit = _FULL_DOC_CONTEXT_FALLBACK_CHARS

            snippet_text = text[:limit]

            metadata: Dict[str, object] = {}
            try:
                idxs = self.retriever.store.get_patient_chunk_indices(doc_id)
            except Exception:
                idxs = []
            if idxs:
                try:
                    chunk_meta = self.retriever.store.chunk_meta[idxs[0]]
                    metadata = self.retriever._extract_meta(chunk_meta) or {}
                except Exception:
                    metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("other_meta", "")

            try:
                self.retriever.set_last_diagnostics(
                    resolved_unit_id,
                    label_id,
                    {
                        "rag_mode": "full_doc",
                        "unit_id": resolved_unit_id,
                        "label_id": str(label_id),
                        "stage": "full_doc",
                        "final_selection": {"count": 1, "score_stats": {"min": 1.0, "max": 1.0, "mean": 1.0}},
                    },
                    original_unit_id=str(unit_id),
                )
            except Exception:
                pass

            return [
                {
                    "doc_id": doc_id,
                    "chunk_id": "__full__",
                    "text": snippet_text,
                    "score": 1.0,
                    "source": "full_doc",
                    "metadata": metadata,
                }
            ]

        return self._retrieve_rag_context(
            resolved_unit_id,
            label_id,
            label_rules,
            topk_override=topk_override,
            min_k_override=min_k_override,
            mmr_lambda_override=mmr_lambda_override,
            original_unit_id=str(unit_id),
        )

    def build_context_for_family(
        self,
        unit_id: str,
        label_ids: list[str],
        rules_map: Mapping[str, str],
        *,
        topk_per_label: int | None = None,
        max_snippets: int | None = None,
        max_chars: int | None = None,
    ) -> list[dict]:
        """
        Build a single merged context for a unit across many labels.

        - For each label_id in label_ids:
            - Look up rules_text = rules_map.get(label_id, "")
            - Call build_context_for_label(unit_id, label_id, rules_text, ...)
            - Take up to topk_per_label snippets (if provided), otherwise keep all.
        - Deduplicate snippets across labels (e.g. by (doc_id, chunk_id) or another stable key in the snippet dict).
        - Sort snippets in a stable way (e.g. by descending score, then by chronological metadata if available).
        - Apply max_snippets to truncate the list if provided.
        - Apply a character budget if max_chars is not None, by walking the sorted snippets and stopping when the cumulative length of snippet["text"] (or their rendered form) exceeds max_chars.
        - Return the final list of snippet dicts, retaining the same shape as build_context_for_label.
        """

        collected: list[dict] = []
        topk: int | None = None
        if topk_per_label is not None:
            try:
                topk = max(0, int(topk_per_label))
            except Exception:
                topk = None

        for label_id in label_ids:
            snippets = self.build_context_for_label(unit_id, label_id, rules_map.get(label_id, ""))
            if topk is not None:
                snippets = snippets[:topk]
            collected.extend(snippets)

        seen: dict[tuple[object, object], dict] = {}
        ordered: list[dict] = []
        for snip in collected:
            key = (snip.get("doc_id"), snip.get("chunk_id"))
            if key in seen:
                continue
            seen[key] = snip
            ordered.append(snip)

        def _score_key(item: dict, idx: int) -> tuple:
            raw_score = item.get("score")
            try:
                score = float(raw_score)
            except Exception:
                score = 0.0
            return (-score, idx)

        scored = sorted(enumerate(ordered), key=lambda t: _score_key(t[1], t[0]))
        sorted_snippets = [ordered[idx] for idx, _ in scored]

        if max_snippets is not None:
            try:
                limit = int(max_snippets)
            except Exception:
                limit = None
            if limit is not None and limit >= 0:
                sorted_snippets = sorted_snippets[:limit]

        if max_chars is not None:
            try:
                char_budget = int(max_chars)
            except Exception:
                char_budget = None
            if char_budget is not None and char_budget > 0:
                total = 0
                budgeted: list[dict] = []
                for snip in sorted_snippets:
                    text = snip.get("text", "")
                    length = len(text) if isinstance(text, str) else len(str(text))
                    if total + length > char_budget:
                        break
                    budgeted.append(snip)
                    total += length
                sorted_snippets = budgeted

        return sorted_snippets

    def _retrieve_rag_context(
        self,
        unit_id: str,
        label_id: str,
        label_rules: str,
        *,
        topk_override: int | None = None,
        min_k_override: int | None = None,
        mmr_lambda_override: float | None = None,
        original_unit_id: str | None = None,
    ) -> list[dict]:
        import numpy as _np

        diagnostics: dict[str, object] = {
            "unit_id": str(unit_id),
            "label_id": str(label_id),
            "stage": "init",
        }

        def _score_stats(items: list[dict]) -> dict:
            scores = [float(it.get("score", 0.0)) for it in items if isinstance(it.get("score"), (int, float))]
            if not scores:
                return {}
            return {
                "min": float(min(scores)),
                "max": float(max(scores)),
                "mean": float(_np.mean(scores)),
            }

        def _patient_local_rank(_unit: str, _q_emb: _np.ndarray, need: int) -> list[dict]:
            idxs = self.store.get_patient_chunk_indices(_unit)
            if not idxs:
                return []
            X = self.store.X[idxs]
            Xn = X / (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            qn = _q_emb / (_np.linalg.norm(_q_emb) + 1e-12)
            sims = (Xn @ qn)
            order = _np.argsort(-sims)[: max(need, 50)]
            out = []
            for j in order:
                m = self.store.chunk_meta[idxs[j]]
                out.append(
                    {
                        "doc_id": m["doc_id"],
                        "chunk_id": m["chunk_id"],
                        "metadata": self.retriever._extract_meta(m),
                        "text": m["text"],
                        "score": float(sims[j]),
                        "source": "patient_local",
                    }
                )
            return out

        def _patient_local_rank_multi(_unit: str, Q: np.ndarray, need: int) -> list[dict]:
            idxs = self.store.get_patient_chunk_indices(str(_unit))
            if not idxs:
                return []
            X = self.store.X[idxs].astype("float32", copy=False)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            sims = Xn @ Qn.T
            best = sims.max(axis=1)
            order = np.argsort(-best)[: max(need, 50)]
            out = []
            for j in order:
                m = self.store.chunk_meta[idxs[j]]
                out.append(
                    {
                        "doc_id": m["doc_id"],
                        "chunk_id": m["chunk_id"],
                        "text": m["text"],
                        "score": float(best[j]),
                        "source": "patient_local_multi",
                    }
                )
            return out

        def _neighbors(items: list[dict], hops: int = 1) -> list[dict]:
            if hops <= 0:
                return []
            by_doc: dict[str, dict[int, int]] = {}
            for ix, m in enumerate(self.store.chunk_meta):
                by_doc.setdefault(m["doc_id"], {})[int(m["chunk_id"])] = ix

            out, seen = [], set((it["doc_id"], int(it["chunk_id"])) for it in items)
            for it in items:
                did, cid = it["doc_id"], int(it["chunk_id"])
                table = by_doc.get(did, {})
                for d in range(1, hops + 1):
                    for cid2 in (cid - d, cid + d):
                        ix = table.get(cid2)
                        if ix is None:
                            continue
                        key = (did, cid2)
                        if key in seen:
                            continue
                        seen.add(key)
                        m = self.store.chunk_meta[ix]
                        out.append(
                            {
                                "doc_id": m["doc_id"],
                                "chunk_id": m["chunk_id"],
                                "metadata": self.retriever._extract_meta(m),
                                "text": m["text"],
                                "score": 0.0,
                                "source": "neighbor",
                            }
                        )
            return out

        def _dedup_only(items: list[dict]) -> list[dict]:
            if not items:
                return []
            seen, dedup = set(), []
            for it in items:
                key = (it.get("doc_id"), int(it.get("chunk_id")))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(it)
            return dedup

        cfg_rag = getattr(self, "cfg", None) or self.cfg
        cfg_final_k = getattr(cfg_rag, "top_k_final", None)
        final_k_raw = topk_override or cfg_final_k
        try:
            final_k = max(1, int(final_k_raw))
        except Exception:
            final_k = 1
        min_k = min_k_override or max(1, getattr(cfg_rag, "min_context_chunks", 3))
        mmr_mult = max(1, getattr(cfg_rag, "mmr_multiplier", 3))
        hops = int(getattr(cfg_rag, "neighbor_hops", 1))
        keyword_fraction = float(getattr(cfg_rag, "keyword_fraction", 0.0))
        keyword_fraction = max(0.0, min(1.0, keyword_fraction))
        use_kw = keyword_fraction > 0.0 and bool(getattr(cfg_rag, "use_keywords", True))
        mmr_select_k = final_k * mmr_mult

        lam = mmr_lambda_override
        if lam is None:
            lam = getattr(cfg_rag, "mmr_lambda", None)
        lam = None if lam is None else float(lam)
        if lam is not None:
            lam = max(0.0, min(1.0, lam))
        diagnostics.update(
            {
                "stage": "config",
                "rag_mode": "patient_local",
                "final_k": final_k,
                "min_k": min_k,
                "mmr": {"lambda": lam, "multiplier": mmr_mult, "select_k": mmr_select_k},
            }
        )

        try:
            label_types = self.repo.label_types()
        except Exception:
            label_types = {}
        base_k = getattr(self.cfg, "exemplar_K", None)
        K_use = int(base_k if base_k is not None else 6)
        if K_use <= 0:
            K_use = 6
        opts = _options_for_label(
            label_id,
            label_types.get(str(label_id), "categorical"),
            getattr(self, "label_configs", {}),
        ) or []
        if opts:
            K_use = max(K_use, len(opts))
        cached_exemplar_embs = self.retriever._get_label_query_embs(label_id, label_rules, K=K_use)
        exemplar_texts = self.retriever._get_label_query_texts(label_id, label_rules, K=K_use) or []

        lblcfg = self.retriever.label_configs.get(label_id, {}) if isinstance(self.retriever.label_configs, dict) else {}
        manual_query = None
        if isinstance(lblcfg, Mapping):
            raw_manual = lblcfg.get("search_query") or lblcfg.get("rag_query")
            if isinstance(raw_manual, str) and raw_manual.strip():
                manual_query = raw_manual.strip()

        mmr_query_embs: list[np.ndarray] = []
        semantic_queries_struct: list[SemanticQuery] = []

        valid_exemplars = [t for t in exemplar_texts if isinstance(t, str) and t.strip()]

        if manual_query:
            query_texts = [manual_query]
            query_embs = [None]
            query_sources = ["manual"]
        elif valid_exemplars:
            query_texts = list(valid_exemplars)
            query_sources = ["exemplar"] * len(valid_exemplars)
            if cached_exemplar_embs is not None and getattr(cached_exemplar_embs, "ndim", 1) == 2:
                query_embs = [
                    cached_exemplar_embs[i]
                    for i in range(min(len(valid_exemplars), cached_exemplar_embs.shape[0]))
                ]
                if len(query_embs) < len(query_texts):
                    query_embs.extend([None] * (len(query_texts) - len(query_embs)))
            else:
                query_embs = [None] * len(valid_exemplars)
        else:
            fallback_rules = (label_rules or "").strip()
            query_texts = [fallback_rules]
            query_embs = [None]
            query_sources = ["rules"]

        missing_idxs = [i for i, emb in enumerate(query_embs) if emb is None]
        if missing_idxs:
            missing_embs = list(self.store._embed([query_texts[i] for i in missing_idxs]))
            for idx, emb in zip(missing_idxs, missing_embs):
                query_embs[idx] = emb

        query_source = query_sources[0] if query_sources else "rules"
        rerank_query_texts = list(query_texts)

        diagnostics.update(
            {
                "stage": "queries",
                "manual_query": manual_query,
                "queries": query_texts,
                "exemplar_queries": exemplar_texts,
                "query_source": query_source,
                "query_sources": query_sources,
                "rerank_queries": rerank_query_texts,
            }
        )

        for q_text, q_emb, q_src in zip(query_texts, query_embs, query_sources):
            semantic_queries_struct.append(SemanticQuery(text=q_text, embedding=q_emb, source=q_src))
            if q_emb is not None:
                mmr_query_embs.append(q_emb)

        q_emb = np.mean(np.vstack(mmr_query_embs), axis=0) if mmr_query_embs else None
        query = query_texts[0] if query_texts else ""

        keywords: list[str] = []
        if use_kw:
            lblcfg = self.retriever.label_configs.get(label_id, {}) if isinstance(self.retriever.label_configs, dict) else {}
            cfg_keywords = getattr(cfg_rag, "keywords", [])
            if isinstance(cfg_keywords, (list, tuple)):
                keywords.extend(str(k) for k in cfg_keywords if isinstance(k, str) and k.strip())
            per_label_kw_cfg = getattr(cfg_rag, "label_keywords", {})
            if isinstance(per_label_kw_cfg, Mapping):
                label_kw = per_label_kw_cfg.get(label_id) or per_label_kw_cfg.get(str(label_id))
                if isinstance(label_kw, (list, tuple)):
                    keywords.extend(str(k) for k in label_kw if isinstance(k, str) and k.strip())
            lbl_keywords = lblcfg.get("keywords", []) if isinstance(lblcfg, Mapping) else []
            if isinstance(lbl_keywords, (list, tuple)):
                keywords.extend(str(k) for k in lbl_keywords if isinstance(k, str) and k.strip())

        uniq_keywords: list[str] = []
        if keywords:
            seen_kw = set()
            for kw in keywords:
                if kw in seen_kw:
                    continue
                seen_kw.add(kw)
                uniq_keywords.append(kw)

        pool_factor = int(getattr(cfg_rag, "pool_factor", 3))
        pool_oversample = float(getattr(cfg_rag, "pool_oversample", 1.5))

        coordinator = RetrievalCoordinator(
            semantic_searcher=lambda sq, need: _patient_local_rank(str(unit_id), sq.embedding, need),
            keyword_searcher=lambda kws, need: (self.retriever._bm25_hits_for_patient(str(unit_id), kws) or [])[:need],
        )

        pool, pool_diag = coordinator.build_candidate_pool(
            semantic_queries=semantic_queries_struct,
            keywords=uniq_keywords,
            top_k_final=final_k,
            keyword_fraction=keyword_fraction,
            pool_factor=pool_factor,
            oversample=pool_oversample,
        )

        diagnostics["retrieval"] = pool_diag
        diagnostics["keyword_search"] = {
            "keywords": uniq_keywords,
            "hit_count": pool_diag.get("keyword_hits", 0),
        }
        diagnostics["semantic_search"] = {
            "runs": len(semantic_queries_struct),
            "score_stats": _score_stats(pool),
        }
        diagnostics["pool"] = {
            "semantic": pool_diag.get("semantic_hits", 0),
            "bm25": pool_diag.get("keyword_hits", 0),
            "deduped": pool_diag.get("final_pool", 0),
            "targets": {
                "semantic": pool_diag.get("semantic_target", 0),
                "keyword": pool_diag.get("keyword_target", 0),
                "pool": pool_diag.get("pool_target", 0),
            },
        }

        if len(pool) < max(min_k, final_k):
            extra = _neighbors(pool, hops=hops)
            pool = _dedup_only(pool + extra)
            diagnostics["pool"]["neighbors_added"] = len(pool) - diagnostics["pool"].get("deduped", 0)
        if not pool:
            diagnostics["stage"] = "empty_pool"
            self.retriever.set_last_diagnostics(unit_id, label_id, diagnostics, original_unit_id=original_unit_id)
            return []

        by_doc: dict[str, dict[int, int]] = {}
        for ix, m in enumerate(self.store.chunk_meta):
            by_doc.setdefault(str(m["doc_id"]), {})[int(m["chunk_id"])] = ix

        cand_idxs, cand_items = [], []
        for it in pool:
            did = str(it.get("doc_id"))
            cid = int(it.get("chunk_id"))
            ix = by_doc.get(did, {}).get(cid)
            if ix is not None:
                cand_idxs.append(ix)
                cand_items.append(it)

        diagnostics.setdefault("pool", {})["source_counts"] = dict(
            Counter(str(it.get("source")) for it in pool)
        )

        if q_emb is None:
            embed_text = query if query_texts else ""
            q_emb = self.store._embed([embed_text])[0]

        def _cross_scores_for_queries(q_texts: list[str], cand_texts: list[str]) -> list[float]:
            if not q_texts:
                return [float(s) for s in self.retriever._cross_scores_cached(query, cand_texts)]
            per_query = [self.retriever._cross_scores_cached(qt, cand_texts) for qt in q_texts]
            return list(np.max(np.vstack(per_query), axis=0)) if per_query else [0.0] * len(cand_texts)

        if not cand_idxs:
            texts = [it["text"] for it in pool]
            rr = _cross_scores_for_queries(rerank_query_texts, texts)
            for it, s in zip(pool, rr):
                it["score"] = float(s)
            pool.sort(key=lambda d: d["score"], reverse=True)
            return pool[:final_k]

        k_pre = min(len(cand_items), max(final_k, min_k, mmr_select_k))
        diagnostics.setdefault("mmr", {}).update({"candidate_pool": len(cand_items), "select_size": k_pre})
        pre: list[dict] = []
        pre_idxs: list[int] = []
        for ix, it in zip(cand_idxs[:k_pre], cand_items[:k_pre]):
            pre.append(it)
            pre_idxs.append(ix)

        texts = [it["text"] for it in pre]
        rr = _cross_scores_for_queries(rerank_query_texts, texts)
        for it, s in zip(pre, rr):
            it["score"] = float(s)

        scored = [
            {"item": it, "store_idx": ix}
            for it, ix in sorted(zip(pre, pre_idxs), key=lambda t: t[0].get("score", 0.0), reverse=True)
        ]

        if lam is not None:
            sel = self.retriever._mmr_select_ranked(
                [d["store_idx"] for d in scored],
                [float(d["item"].get("score", 0.0)) for d in scored],
                k=final_k,
                lam=lam,
            )
            idx_to_item = {d["store_idx"]: d["item"] for d in scored}
            out = [idx_to_item[i] for i in sel if i in idx_to_item]
        else:
            out = [d["item"] for d in scored[:final_k]]

        diagnostics.setdefault("mmr", {}).update(
            {
                "used_lambda": lam,
                "pre_ce_count": len(pre),
                "final_before_topoff": len(out),
                "pre_score_stats": _score_stats(pre),
            }
        )

        if len(out) < min_k:
            picked = set((o["doc_id"], int(o["chunk_id"])) for o in out)
            rest = [it for it in pool if (it["doc_id"], int(it["chunk_id"])) not in picked]
            if rest:
                rr2 = _cross_scores_for_queries(rerank_query_texts, [it["text"] for it in rest])
                for it, s in zip(rest, rr2):
                    it["score"] = float(s)
                rest.sort(key=lambda d: d["score"], reverse=True)
                need = min_k - len(out)
                out.extend(rest[:need])

        diagnostics["final_selection"] = {
            "count": len(out),
            "score_stats": _score_stats(out),
        }

        diagnostics["stage"] = "complete"
        self.retriever.set_last_diagnostics(unit_id, label_id, diagnostics, original_unit_id=original_unit_id)
        return out[:final_k]


def _options_for_label(label_id: str, label_type: str, label_config: dict) -> list[str]:
    cfg = label_config.get(label_id, {}) if isinstance(label_config, dict) else {}
    if label_type in {"categorical", "categorical_multi"}:
        return cfg.get("options", []) or []
    elif label_type == "binary":
        return ["yes", "no"]
    return []
