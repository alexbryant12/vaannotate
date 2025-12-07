#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RAG retrieval helpers and utilities."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..config import RAGConfig
from ..core.data import DataRepository
from ..core.embeddings import EmbeddingStore, Models
from ..core.retrieval import RetrievalCoordinator, SemanticQuery
from ..utils.hashing import _stable_rules_hash, stable_hash_pair
from ..utils.jsonish import _maybe_parse_jsonish

try:
    from cachetools import LRUCache
except Exception:
    from collections import OrderedDict

    class LRUCache(dict):
        def __init__(self, maxsize=10000):
            super().__init__()
            self._order = OrderedDict()
            self._max = maxsize

        def __setitem__(self, k, v):
            if k in self._order:
                self._order.move_to_end(k)
            self._order[k] = None
            super().__setitem__(k, v)
            if len(self._order) > self._max:
                old, _ = self._order.popitem(last=False)
                super().pop(old, None)

        def get(self, k, default=None):
            if k in self._order:
                self._order.move_to_end(k)
            return super().get(k, default)


def _options_for_label(label_id: str, label_type: str, label_config: dict) -> list[str]:
    cfg = label_config.get(label_id, {}) if isinstance(label_config, dict) else {}
    if label_type == "categorical":
        return cfg.get("options", []) or []
    if label_type == "binary":
        return ["yes", "no"]
    return []

class RAGRetriever:
    _RR_CACHE_MAX = 200000
    def __init__(self, store: EmbeddingStore, models: Models, cfg: RAGConfig, label_configs: Optional[dict]=None, notes_by_doc: Optional[Dict[str,str]]=None, repo: Optional[DataRepository]=None):
        self.store = store; self.models = models; self.cfg = cfg
        self.label_configs = label_configs or {}
        self._notes_by_doc = notes_by_doc or {}
        self._repo = repo
        self._rr_cache = LRUCache(maxsize=self._RR_CACHE_MAX)
        self._bm25_cache: Dict[str, dict] = {}
        self._label_query_texts = {}   # (label_id, rules_hash, K) -> List[str]
        self._label_query_embs  = {}   # (label_id, rules_hash, K) -> np.ndarray[K,d]
        # Optional concise rule text to keep re-ranker context compact
        self.rerank_rule_overrides: dict[str, str] = {}
        self._last_diagnostics: dict[tuple[str, str], dict] = {}

    def set_last_diagnostics(
        self, unit_id: str, label_id: str, diagnostics: dict, original_unit_id: str | None = None
    ):
        """Cache the most recent retrieval diagnostics for a (unit, label) pair."""

        key = (str(unit_id), str(label_id))
        self._last_diagnostics[key] = diagnostics or {}
        if original_unit_id is not None:
            self._last_diagnostics[(str(original_unit_id), str(label_id))] = diagnostics or {}

    def get_last_diagnostics(self, unit_id: str, label_id: str) -> dict:
        return self._last_diagnostics.get((str(unit_id), str(label_id)), {})

    def set_label_exemplars(self, label_id: str, rules: str, K: int, texts: list[str]):
        key = (str(label_id), _stable_rules_hash(label_id, rules, K, getattr(self.models.embedder, "name_or_path", "")), int(K))
        self._label_query_texts[key] = [t for t in texts if isinstance(t, str) and t.strip()]
        if self._label_query_texts[key]:
            E = self.store._embed(self._label_query_texts[key])   # (K, d)
            self._label_query_embs[key] = E

    def _get_label_query_embs(self, label_id: str, rules: str, K: int):
        key = (str(label_id), _stable_rules_hash(label_id, rules, K, getattr(self.models.embedder, "name_or_path", "")), int(K))
        return self._label_query_embs.get(key)

    def _get_label_query_texts(self, label_id: str, rules: str, K: int):
        key = (str(label_id), _stable_rules_hash(label_id, rules, K, getattr(self.models.embedder, "name_or_path", "")), int(K))
        return self._label_query_texts.get(key)

    def _rerank_rules_text(self, label_id: str, label_rules: str | None) -> str:
        """Return a compact rule string when available for re-ranking only."""

        if self.rerank_rule_overrides and label_id in self.rerank_rule_overrides:
            return self.rerank_rule_overrides[label_id]
        key = str(label_id)
        if self.rerank_rule_overrides and key in self.rerank_rule_overrides:
            return self.rerank_rule_overrides[key]
        return label_rules or ""
    
    def _extract_meta(self, m:dict) -> dict:
        """
        Normalize chunk-level metadata into a compact dict that is JSON-safe and
        useful for the LLM context header: {date, note_type, modality, title}.
        Accepts raw chunk_meta dict.
        """
        out = {}
        # Common fields
        if "date_note" in m and m["date_note"]:
            out["date"] = str(m["date_note"])

        note_type = ""
        if "notetype" in m and m["notetype"]:
            note_type = str(m["notetype"])

        # Try to parse richer metadata JSON if present (may include notetype)
        meta_raw = m.get("document_metadata_json") or m.get("metadata_json")
        meta_obj = _maybe_parse_jsonish(meta_raw)
        if not note_type and isinstance(meta_obj, dict):
            for key in ("note_type", "notetype", "noteType"):
                val = meta_obj.get(key)
                if val:
                    note_type = str(val)
                    break

        if note_type:
            out["note_type"] = note_type

        out['other_meta'] = str(meta_raw)
        return out

    def _rr_key(self, q: str, t: str) -> str:
        return stable_hash_pair(q or '', t or '')
    

    def _cross_scores_cached(self, query: str, texts: list) -> list:
        missing_idx, miss_pairs, scores = [], [], [None] * len(texts)
        for i, t in enumerate(texts):
            k = self._rr_key(query, t)
            sc = self._rr_cache.get(k)
            if sc is None:
                missing_idx.append(i); miss_pairs.append((query, t))
            else:
                scores[i] = sc
        if missing_idx:
            rr = self.models.reranker.predict(miss_pairs, batch_size=getattr(self.models, 'rerank_batch', 64), show_progress_bar=False)
            for i, sc in zip(missing_idx, rr):
                scores[i] = float(sc)
                k = self._rr_key(query, texts[i])
                # eviction handled by LRUCache automatically
                self._rr_cache[k] = float(sc)
        return scores

    def _build_query(self, label_id: str, label_rules: Optional[str]) -> str:
        if label_rules and isinstance(label_rules, str):
            return label_rules.strip()
        return ""

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b\w+\b", str(text).lower())

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        if hasattr(self.store, "_tokenize_for_bm25"):
            try:
                return self.store._tokenize_for_bm25(text)
            except Exception:
                pass
        return self._tokenize(text)

    def _bm25_index_for_patient(self, unit_id: str) -> Optional[dict]:
        uid = str(unit_id)
        cached = self._bm25_cache.get(uid)
        if cached is not None:
            return cached

        idx: Optional[dict] = None
        if hasattr(self.store, "bm25_index_for_unit"):
            try:
                idx = self.store.bm25_index_for_unit(uid)
            except Exception:
                idx = None

        if idx is None:
            return None

        self._bm25_cache[uid] = idx
        return idx

    def _bm25_hits_for_patient(self, unit_id: str, keywords: List[str]) -> List[dict]:
        if not keywords:
            return []

        index = self._bm25_index_for_patient(unit_id)
        if not index:
            return []

        query_tokens: List[str] = []
        for kw in keywords:
            if isinstance(kw, str):
                query_tokens.extend(self._tokenize_for_bm25(kw))
        if not query_tokens:
            return []

        docs = index.get("docs") or []
        metas = index.get("metas") or []
        if not docs or not metas:
            return []

        # Global IDF statistics are maintained by the EmbeddingStore.
        idf = getattr(self.store, "idf_global", {}) or {}
        if not idf and isinstance(getattr(self.store, "bm25_indices", None), dict):
            indices_dict = getattr(self.store, "bm25_indices", {})
            if isinstance(indices_dict, dict):
                idf = indices_dict.get("idf_global", {}) or {}

        if not idf:
            return []

        avgdl = float(index.get("avgdl") or 1.0)
        k1, b = 1.5, 0.75
        scores: List[tuple[float, dict]] = []

        for toks, meta in zip(docs, metas):
            tf = Counter(toks)
            dl = len(toks) or 1
            score = 0.0
            for term in query_tokens:
                if term not in tf or term not in idf:
                    continue
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * dl / avgdl)
                score += idf[term] * (freq * (k1 + 1)) / (denom + 1e-12)
            if score <= 0.0:
                continue
            scores.append((score, meta))

        scores.sort(key=lambda pair: pair[0], reverse=True)

        out: List[dict] = []
        for score, meta in scores[: self.cfg.keyword_topk]:
            out.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "metadata": self._extract_meta(meta),
                    "text": meta.get("text", ""),
                    "score": float(score),
                    "source": "bm25",
                }
            )
        return out

    def _reciprocal_rank_fusion(self, runs: List[List[dict]], constant: int = 60) -> List[dict]:
        fused: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for run in runs:
            if not run:
                continue
            for rank, item in enumerate(run):
                doc_id = str(item.get("doc_id"))
                try:
                    chunk_id = int(item.get("chunk_id", -1))
                except Exception:
                    continue
                if chunk_id < 0:
                    continue
                key = (doc_id, chunk_id)
                entry = fused.setdefault(key, {"score": 0.0, "item": item})
                entry["score"] += 1.0 / (constant + rank + 1)
                if entry["item"] is not item:
                    merged = dict(entry["item"])
                    for k, v in item.items():
                        if k not in merged or merged[k] in (None, ""):
                            merged[k] = v
                    entry["item"] = merged
        fused_items = []
        for data in fused.values():
            merged_item = dict(data.get("item", {}))
            merged_item["score"] = float(data.get("score", 0.0))
            fused_items.append(merged_item)
        fused_items.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return fused_items

    def _mmr_select(self, q_emb: np.ndarray, cand_idxs: List[int], k: int, lam: float) -> List[int]:
        if k<=0 or not cand_idxs: return []
        X = self.store.X[cand_idxs]
        Xn = X / (np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
        q = q_emb / (np.linalg.norm(q_emb)+1e-12)
        simq = Xn @ q
        sel = []; used = np.zeros(len(cand_idxs), dtype=bool)
        for _ in range(min(k,len(cand_idxs))):
            if not sel:
                i = int(np.argmax(simq)); sel.append(i); used[i]=True; continue
            M = Xn[[i for i,u in enumerate(used) if u]]
            maxsim = (Xn @ M.T).max(axis=1)
            scores = lam*simq - (1-lam)*maxsim
            scores[used] = -1e9
            i = int(np.argmax(scores)); sel.append(i); used[i]=True
        return [cand_idxs[i] for i in sel]

    def _mmr_select_ranked(
        self,
        cand_idxs: list[int],
        rel_scores: list[float],
        k: int,
        lam: float,
    ) -> list[int]:
        """
        Greedy MMR using CE relevance scores and embedding cosine similarity.
        Expects cand_idxs to align 1:1 with rel_scores.
        """
        if k <= 0 or not cand_idxs:
            return []
        import numpy as _np

        rel = _np.asarray(rel_scores, dtype=float)
        X = self.store.X[cand_idxs]
        Xn = X / (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        used = _np.zeros(len(cand_idxs), dtype=bool)
        sel: list[int] = []
        for _ in range(min(k, len(cand_idxs))):
            if not sel:
                i = int(_np.argmax(rel))
            else:
                M = Xn[used]
                maxsim = (Xn @ M.T).max(axis=1)
                scores = lam * rel - (1 - lam) * maxsim
                scores[used] = -1e9
                i = int(_np.argmax(scores))
            sel.append(i)
            used[i] = True
        return [cand_idxs[i] for i in sel]

    def _dedup_rerank(self, query: str, items: List[dict], final_topk: int) -> List[dict]:
        if not items: return []
        seen = set(); dedup = []
        for it in items:
            key = (it.get("doc_id"), it.get("chunk_id"))
            if key in seen: continue
            seen.add(key); dedup.append(it)
        texts = [it["text"] for it in dedup]
        rr = self._cross_scores_cached(query, texts)
        for it, s in zip(dedup, rr):
            it["score"] = float(s) + float(it.get("score",0.0))
        dedup.sort(key=lambda d: d["score"], reverse=True)
        return dedup[:final_topk]

    def retrieve_for_patient_label(
        self,
        unit_id: str | None,
        label_id: str,
        label_rules: str | None,
        topk_override: int | None = None,
        min_k_override: int | None = None,
        mmr_lambda_override: float | None = None,
        *,
        original_unit_id: str | None = None,
    ) -> list[dict]:
        """
        Patient-only RAG for (unit_id, label_id).
        Returns a list of snippets: [{"doc_id","chunk_id","text","score","source"}, ...]
        No global/corpus search lives here; use expand_from_snippets for discovery across patients.

        If mmr_lambda_override (or cfg.mmr_lambda) is set in [0,1], apply MMR on the CE-ranked pool
        to pick a diverse final top-K.
        """
        import numpy as _np
        
        # ---- helpers ----
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
                out.append({
                    "doc_id": m["doc_id"],
                    "chunk_id": m["chunk_id"],
                    "metadata": self._extract_meta(m),
                    "text": m["text"],
                    "score": float(sims[j]),
                    "source": "patient_local",
                })
            return out
        
        def _patient_local_rank_multi(self, unit_id: str, Q: np.ndarray, need: int) -> list[dict]:
            idxs = self.store.get_patient_chunk_indices(str(unit_id))
            if not idxs:
                return []
            X  = self.store.X[idxs].astype("float32", copy=False)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)        # (K,d)
            sims = Xn @ Qn.T                                                   # (n_chunks, K)
            best = sims.max(axis=1)                                            # (n_chunks,)
            order = np.argsort(-best)[: max(need, 50)]
            out = []
            for j in order:
                m = self.store.chunk_meta[idxs[j]]
                out.append({
                    "doc_id": m["doc_id"],
                    "chunk_id": m["chunk_id"],
                    "text": m["text"],
                    "score": float(best[j]),
                    "source": "patient_local_multi"
                })
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
                        out.append({
                            "doc_id": m["doc_id"],
                            "chunk_id": m["chunk_id"],
                            "metadata": self._extract_meta(m),
                            "text": m["text"],
                            "score": 0.0,
                            "source": "neighbor",
                        })
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
        
        # ---- config ----
        cfg_rag = getattr(self, "cfg", getattr(self, "rag", None)) or self.cfg
        cfg_final_k = getattr(cfg_rag, "top_k_final", None)
        final_k_raw = topk_override or cfg_final_k or getattr(cfg_rag, "per_label_topk", 6)
        try:
            final_k = max(1, int(final_k_raw))
        except Exception:
            final_k = 1
        min_k     = min_k_override or max(1, getattr(cfg_rag, "min_context_chunks", 3))
        mmr_mult  = max(1, getattr(cfg_rag, "mmr_multiplier", 3))  # pool size before CE = final_k * mmr_mult
        hops      = int(getattr(cfg_rag, "neighbor_hops", 1))
        keyword_fraction = float(getattr(cfg_rag, "keyword_fraction", 0.0))
        keyword_fraction = max(0.0, min(1.0, keyword_fraction))
        use_kw    = keyword_fraction > 0.0 and bool(getattr(cfg_rag, "use_keywords", True))
        mmr_select_k = final_k * mmr_mult

        # λ (0..1)
        lam = mmr_lambda_override
        if lam is None: lam = getattr(cfg_rag, "mmr_lambda", None)
        lam = None if lam is None else float(lam)
        if lam is not None: lam = max(0.0, min(1.0, lam))
        diagnostics.update(
            {
                "stage": "config",
                "rag_mode": "patient_local",
                "final_k": final_k,
                "min_k": min_k,
                "mmr": {"lambda": lam, "multiplier": mmr_mult, "select_k": mmr_select_k},
            }
        )

        # ---- label-aware query + embedding ----
        try:
            label_types = repo.label_types()
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
        cached_exemplar_embs = self._get_label_query_embs(label_id, label_rules, K=K_use)
        exemplar_texts = self._get_label_query_texts(label_id, label_rules, K=K_use) or []

        lblcfg = self.label_configs.get(label_id, {}) if isinstance(self.label_configs, dict) else {}
        manual_query = None
        if isinstance(lblcfg, Mapping):
            raw_manual = lblcfg.get("search_query") or lblcfg.get("rag_query")
            if isinstance(raw_manual, str) and raw_manual.strip():
                manual_query = raw_manual.strip()

        mmr_query_embs: list[np.ndarray] = []
        semantic_queries_struct: list[SemanticQuery] = []

        valid_exemplars = [t for t in exemplar_texts if isinstance(t, str) and t.strip()]

        # Build prioritized query list: manual override > exemplars > label rules
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
                    # pad for later embedding
                    query_embs.extend([None] * (len(query_texts) - len(query_embs)))
            else:
                query_embs = [None] * len(valid_exemplars)
        else:
            fallback_rules = (label_rules or "").strip()
            query_texts = [fallback_rules]
            query_embs = [None]
            query_sources = ["rules"]

        # Fill any missing embeddings
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
            semantic_queries_struct.append(
                SemanticQuery(text=q_text, embedding=q_emb, source=q_src)
            )
            if q_emb is not None:
                mmr_query_embs.append(q_emb)

        q_emb = np.mean(np.vstack(mmr_query_embs), axis=0) if mmr_query_embs else None
        query = query_texts[0] if query_texts else ""

        keywords: list[str] = []
        if use_kw:
            lblcfg = self.label_configs.get(label_id, {}) if isinstance(self.label_configs, dict) else {}
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
            keyword_searcher=lambda kws, need: (self._bm25_hits_for_patient(str(unit_id), kws) or [])[:need],
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

        # Neighbor padding for short pools
        if len(pool) < max(min_k, final_k):
            extra = _neighbors(pool, hops=hops)
            pool = _dedup_only(pool + extra)
            diagnostics["pool"]["neighbors_added"] = len(pool) - diagnostics["pool"].get("deduped", 0)
        if not pool:
            diagnostics["stage"] = "empty_pool"
            self.set_last_diagnostics(unit_id, label_id, diagnostics, original_unit_id=original_unit_id)
            return []
    
        # Build (doc,chunk) -> ix map with STRING doc_id, INT chunk_id
        by_doc: dict[str, dict[int, int]] = {}
        for ix, m in enumerate(self.store.chunk_meta):
            by_doc.setdefault(str(m["doc_id"]), {})[int(m["chunk_id"])] = ix
    
        cand_idxs, cand_items = [], []
        for it in pool:
            did = str(it.get("doc_id")); cid = int(it.get("chunk_id"))
            ix = by_doc.get(did, {}).get(cid)
            if ix is not None:
                cand_idxs.append(ix); cand_items.append(it)

        diagnostics.setdefault("pool", {})["source_counts"] = dict(
            Counter(str(it.get("source")) for it in pool)
        )

        if q_emb is None:
            embed_text = query if query_texts else ""
            q_emb = self.store._embed([embed_text])[0]

        def _cross_scores_for_queries(q_texts: list[str], cand_texts: list[str]) -> list[float]:
            if not q_texts:
                return [float(s) for s in self._cross_scores_cached(query, cand_texts)]
            per_query = [self._cross_scores_cached(qt, cand_texts) for qt in q_texts]
            return list(np.max(np.vstack(per_query), axis=0)) if per_query else [0.0] * len(cand_texts)

        # CE fallback if mapping failed
        if not cand_idxs:
            texts = [it["text"] for it in pool]
            rr = _cross_scores_for_queries(rerank_query_texts, texts)
            for it, s in zip(pool, rr):
                it["score"] = float(s)
            pool.sort(key=lambda d: d["score"], reverse=True)
            return pool[:final_k]
    
        # Preselect for CE scoring (head)
        k_pre = min(len(cand_items), max(final_k, min_k, mmr_select_k))
        diagnostics.setdefault("mmr", {}).update(
            {"candidate_pool": len(cand_items), "select_size": k_pre}
        )
        pre: list[dict] = []
        pre_idxs: list[int] = []
        for ix, it in zip(cand_idxs[:k_pre], cand_items[:k_pre]):
            pre.append(it)
            pre_idxs.append(ix)

        # CE last, CE-only score
        texts = [it["text"] for it in pre]
        rr = _cross_scores_for_queries(rerank_query_texts, texts)
        for it, s in zip(pre, rr):
            it["score"] = float(s)

        # Sort by CE to feed MMR or take head
        scored = [
            {"item": it, "store_idx": ix}
            for it, ix in sorted(
                zip(pre, pre_idxs), key=lambda t: t[0].get("score", 0.0), reverse=True
            )
        ]

        if lam is not None:
            sel = self._mmr_select_ranked(
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

        # Top-off to min_k using CE on the rest
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
        self.set_last_diagnostics(unit_id, label_id, diagnostics, original_unit_id=original_unit_id)
        return out[:final_k]

    def expand_from_snippets(self, label_id: str, snippets: List[str], seen_pairs: set, per_seed_k: int=100) -> Dict[str,float]:
        out: Dict[str,float] = {}
        for sn in snippets:
            if not sn: continue
            sims, idxs = self.store.search([sn], topk=max(per_seed_k, 50))
            idxs = idxs[0]; sims = sims[0]
            pre = []
            for sc, ix in zip(sims, idxs):
                if ix < 0: continue
                m = self.store.chunk_meta[ix]
                pre.append((ix, m["text"], m["unit_id"], float(sc)))
            texts = [txt for _, txt, _, _ in pre]
            rr = self._cross_scores_cached(sn, texts)
            for (ix, _txt, uid, _sc), s in zip(pre, rr):
                if (uid, label_id) in seen_pairs: continue
                out[uid] = max(out.get(uid, 0.0), float(s))
        return out



