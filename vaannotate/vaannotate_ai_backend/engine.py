#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
active_learning_next_batch_llmfirst-6.py  — LLM-first pipeline with Forced-Choice Micro-Probe

What’s in this file (high level):
  • Complete, modular active-learning pipeline for multi-document/patient labeling
  • Buckets:
      (1) Expanded Disagreement (new, unseen patients similar to prior high-entropy cases)
      (2) LLM-uncertain (forced-choice uncertainty as primary signal + self-consistency)
      (3) LLM-certain / Easy QC (forced-choice low entropy)
      (4) Diversity (label-aware pooled vectors via prototypes+attention; plus a small prototype-free slice)
  • Retrieval: chunk-level RAG (MMR + optional keyword mix) with CrossEncoder re-rank (cached)
  • Embeddings: FAISS vector store, configurable index (flat/hnsw/ivf), persisted to disk; embeddings mem-mapped
  • Label-aware pooled patient vectors (attention over top-K relevant chunks against per-label prototypes)
  • LLM-first probing:
      - JSON call (rich signal, self-consistency, span logprobs for 'prediction')
      - NEW: forced-choice micro-probe (max_tokens=1) → option entropy & margin
      - Uncertainty U = w1·z(entropy) + w2·z(1 − agreement) when FC available; else fallback to prior method
  • Dependencies among labels (e.g., HTN_PRESENT → {DATE, SEVERITY}) to avoid probing inapplicable children
  • Safe Parquet writing (JSON-encode nested cols)


  To do -
  - Multi-categorical selection support
  - Local LLM with exllamav2 and LMFE forced JSON schemas
"""

from __future__ import annotations
import gzip
import os, re, json, math, time, random, unicodedata
from collections import defaultdict, Counter
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional, Any, Mapping
import numpy as np
import pandas as pd
from .core.data import DataRepository
from .core.embeddings import (
    EmbeddingStore,
    IndexConfig,
    Models,
    build_models_from_env,
)
from .config import (
    DiversityConfig,
    DisagreementConfig,
    LLMConfig,
    LLMFirstConfig,
    OrchestratorConfig,
    Paths,
    RAGConfig,
    SCJitterConfig,
    SelectionConfig,
)
from .services import (
    ActiveLearningSelector,
    ContextBuilder,
    DiversitySelector,
    DisagreementScorer,
    LLMLabeler,
    LLMUncertaintyScorer,
    LLM_RECORDER,
)
from .pipelines.active_learning import ActiveLearningPipeline
from .label_configs import EMPTY_BUNDLE, LabelConfigBundle
from .llm_backends import build_llm_backend
from .services.rag_retriever import RAGRetriever
from .services.disagreement_expander import DisagreementExpander
from .utils.io import atomic_write_bytes, normalize01, read_table, write_table
from .utils.jsonish import _jsonify_cols
from .utils.runtime import (
    CancelledError,
    LOGGER,
    cancellation_scope,
    check_cancelled,
    iter_with_bar,
)
        
"""
Note: Embedding/index primitives (IndexConfig, Models, EmbeddingStore) now live
in ``core.embeddings``.
"""

# ------------------------------
# Label-aware pooling with prototypes
# ------------------------------

class LabelAwarePooler:
    BETA_DEFAULT = 5.0

    def __init__(self, repo: DataRepository, store: EmbeddingStore, models: Models, beta: float = None, kmeans_k: int = 8, persist_dir: str = None, version: str = "v1", use_negative: bool = False, llmfirst_cfg: LLMFirstConfig | None = None):
        self.repo = repo; self.store = store; self.models = models
        self.beta = float(beta) if beta is not None else self.BETA_DEFAULT
        self.kmeans_k = int(kmeans_k)
        self.persist_dir = persist_dir
        self.version = version
        self.use_negative = use_negative
        self.prototypes: Dict[str,np.ndarray] = {}
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
        self._cache_vec: Dict[Tuple[str,str], np.ndarray] = {}
        self.llmfirst_cfg = llmfirst_cfg

    def _save_bank(self, label: str, arr: np.ndarray):
        if not self.persist_dir: return
        path = os.path.join(self.persist_dir, f"prot_{label}_{self.version}.npy")
        np.save(path, arr)

    def _load_bank(self, label: str) -> Optional[np.ndarray]:
        if not self.persist_dir: return None
        path = os.path.join(self.persist_dir, f"prot_{label}_{self.version}.npy")
        if os.path.exists(path):
            try: return np.load(path)
            except: return None
        return None

    def _kmeans_medoids(self, E: np.ndarray, k: int) -> np.ndarray:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(k, max(1, len(E))), n_init=5)
            labels = km.fit_predict(E)
            centers = km.cluster_centers_
            out = []
            for c in range(centers.shape[0]):
                idxs = np.where(labels==c)[0]
                if len(idxs)==0: continue
                sub = E[idxs]
                d = np.linalg.norm(sub - centers[c], axis=1)
                med = idxs[int(np.argmin(d))]
                out.append(E[med])
            return np.vstack(out) if out else E[:min(k,len(E))]
        except Exception:
            if len(E) <= k: return E
            idx = np.linspace(0, len(E)-1, num=k, dtype=int)
            return E[idx]

    def build_prototypes(self):
        """
        Build per-label prototype banks from prior-round rationales.
    
        Type-aware rules:
          - Binary labels:   positive snippets from label_value in {present/yes/1/...};
                             negative snippets ONLY if rationale polarity is explicit ('neg') AND self.use_negative=True.
          - Categorical:     all rationales treated as positive evidence; negatives only if explicit polarity.
          - Numeric/Date:    all rationales treated as positive evidence; negatives only if explicit polarity.
    
        Snippets are de-duplicated, embedded, clustered to medoids, and persisted on disk.
        Previously saved banks are re-used if no fresh snippets are found for a label.
        """
        from collections import defaultdict
    
        # Per-label snippet bags
        pos_snips: dict[str, list[str]] = defaultdict(list)
    
        # 2) Collect snippets (type-aware polarity)
        if "rationales_json" in self.repo.ann.columns:
            for r in self.repo.ann.itertuples(index=False):
                snip_list = getattr(r, "rationales_json", None)
                if not isinstance(snip_list, list):
                    continue
                lid = str(getattr(r, "label_id", ""))
    
                for sp in snip_list:
                    if not (isinstance(sp, dict) and sp.get("snippet")):
                        continue
                    sn = str(sp.get("snippet") or "").strip()
                    if not sn:
                        continue
    
                    pos_snips[lid].append(sn)

        # 3) De-duplicate (keep stable order)
        def _dedup_keep_order(xs: list[str]) -> list[str]:
            seen = set(); out = []
            for s in xs:
                key = s.strip()
                if not key or key in seen:
                    continue
                seen.add(key); out.append(s)
            return out
    
        for lid in list(pos_snips.keys()):
            pos_snips[lid] = _dedup_keep_order(pos_snips[lid])
    
        # 4) For each label: embed, cluster to medoids, persist, or reuse saved banks
        lids = set(pos_snips.keys())
        for lid in lids:
            texts_pos = [s for s in pos_snips.get(lid, []) if isinstance(s, str) and s.strip()]
    
            # Cap optional (keeps memory in check)
            try:
                max_pos = int(getattr(self, "max_snips_per_label", 2000))
            except Exception:
                max_pos = 2000
            if texts_pos and len(texts_pos) > max_pos:
                texts_pos = texts_pos[:max_pos]
    
            # Load previously saved banks (if any)
            P_load = self._load_bank(lid)
    
            # --- Positive prototypes ---
            if texts_pos:
                try:
                    E = self.store._embed(texts_pos)
                    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
                    E = self._kmeans_medoids(E, self.kmeans_k)
                    self.prototypes[lid] = E
                    self._save_bank(lid, E)
                except Exception as e:
                    # Fall back to previously saved if embedding fails
                    if P_load is not None:
                        self.prototypes[lid] = P_load
                    try:
                        print("WARNING: prototype embed failed")
                    except Exception:
                        pass
            elif P_load is not None:
                self.prototypes[lid] = P_load
                

    def pooled_vector(
        self,
        unit_id: str,
        label_id: str,
        retriever: RAGRetriever,
        label_rules: str,
        topk: int = 6,
        context_builder: ContextBuilder | None = None,
    ) -> np.ndarray:
        key = (unit_id, label_id)
        if key in self._cache_vec: return self._cache_vec[key]
        builder = context_builder
        if builder is None:
            builder = getattr(retriever, "context_builder", None)
        if builder is not None:
            ctx = builder.build_context_for_label(
                unit_id,
                label_id,
                label_rules,
                topk_override=topk,
                single_doc_context_mode=getattr(self.llmfirst_cfg, "single_doc_context", "rag"),
                full_doc_char_limit=getattr(self.llmfirst_cfg, "single_doc_full_context_max_chars", None),
            )
        else:
            ctx = retriever.retrieve_for_patient_label(
                unit_id,
                label_id,
                label_rules,
                topk_override=topk,
            )
        if not ctx:
            idxs = retriever.store.get_patient_chunk_indices(unit_id)
            if not idxs:
                v = np.random.randn(retriever.store.X.shape[1]).astype("float32")
                self._cache_vec[key] = v; return v
            V = retriever.store.X[idxs]; v = V.mean(axis=0); self._cache_vec[key] = v; return v
        embs = []
        idxs_u = retriever.store.get_patient_chunk_indices(unit_id)
        for s in ctx:
            did, cid = s.get("doc_id"), s.get("chunk_id")
            found = None
            for ix in idxs_u:
                m = retriever.store.chunk_meta[ix]
                if m["doc_id"]==did and m["chunk_id"]==cid:
                    found = ix; break
            if found is not None: embs.append(retriever.store.X[found])
            else: embs.append(retriever.store._embed([s.get("text","") or ""])[0])
        V = np.vstack(embs); V = V / (np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
        P = self.prototypes.get(label_id)
        if P is None or P.size==0:
            v = V.max(axis=0); self._cache_vec[key] = v; return v
        S = V @ P.T
        w_raw = S.max(axis=1)
        w = w_raw - w_raw.max()
        w = np.exp(5.0 * w)  # beta=5 default
        w_sum = w.sum()
        if w_sum <= 1e-9: v = V.mean(axis=0)
        else: v = (w[:,None]*V).sum(axis=0) / (w_sum+1e-12)
        self._cache_vec[key] = v; return v


# ------------------------------
# K-center (farthest-first)
# ------------------------------
def kcenter_select(vecs: np.ndarray, k: int, seed_idx: Optional[int]=None, preselected: Optional[np.ndarray]=None) -> List[int]:
    if vecs.shape[0]==0 or k<=0: return []
    V = vecs / (np.linalg.norm(vecs,axis=1,keepdims=True)+1e-12)
    N = V.shape[0]; selected = []
    if preselected is not None and preselected.size:
        P = preselected / (np.linalg.norm(preselected,axis=1,keepdims=True)+1e-12)
        d_pre = 1 - (V @ P.T).max(axis=1)
    else:
        d_pre = np.zeros(N, dtype=float)
    if seed_idx is None:
        centroid = V.mean(axis=0, keepdims=True)
        d0 = 1 - (V @ centroid.T).reshape(-1)
        d = d0 + d_pre; i = int(np.argmax(d))
    else:
        i = int(seed_idx)
    selected.append(i)
    sel_mat = V[[i],:]
    if preselected is not None and preselected.size:
        S = np.vstack([sel_mat, preselected])
    else:
        S = sel_mat
    d_to_sel = 1 - (V @ S.T).max(axis=1)
    for _ in range(1, min(k,N)):
        i = int(np.argmax(d_to_sel)); selected.append(i)
        svec = V[i:i+1,:]
        S = np.vstack([S, svec])
        d_to_sel = np.minimum(d_to_sel, 1 - (V @ svec.T).reshape(-1))
    return selected


# ------------------------------
# LLM-first probe + forced-choice + expansion
# ------------------------------
class ActiveLearningLLMFirst:
    def __init__(
        self,
        paths: Paths,
        cfg: OrchestratorConfig,
        label_config_bundle: LabelConfigBundle | None = None,
        *,
        label_config: Optional[dict] = None,
        phenotype_level: str | None = None,
    ):
        import os

        self.paths = paths
        self.cfg = cfg
        notes_df = read_table(paths.notes_path)
        ann_df = read_table(paths.annotations_path)
        self.phenotype_level = (phenotype_level or "multi_doc").strip().lower()
        self.repo = DataRepository(notes_df, ann_df, phenotype_level=self.phenotype_level)

        self.models = build_models_from_env()

        bundle = (label_config_bundle or EMPTY_BUNDLE).with_current_fallback(label_config)
        self.label_config_bundle = bundle
        self.label_config = bundle.current or {}
        self.legacy_label_configs = dict(bundle.legacy)
        self.excluded_unit_ids = set(getattr(cfg, "excluded_unit_ids", []) or [])

        self.store = EmbeddingStore(self.models, cache_dir=self.paths.cache_dir, normalize=self.cfg.rag.normalize_embeddings)
        self.rag = RAGRetriever(
            self.store,
            self.models,
            self.cfg.rag,
            label_configs=self.label_config,
            notes_by_doc=self.repo.notes_by_doc(),
            repo=self.repo,
        )
        self.context_builder = ContextBuilder(
            self.repo,
            self.store,
            self.rag,
            self.cfg.rag,
            self.label_config_bundle,
        )
        try:
            self.rag.context_builder = self.context_builder
        except Exception:
            pass
        self.llm: LLMLabeler | None = None
        self.pooler = LabelAwarePooler(
            self.repo,
            self.store,
            self.models,
            beta=float(os.getenv('POOLER_BETA', 5.0)),
            kmeans_k=int(os.getenv('POOLER_K', 8)),
            persist_dir=os.path.join(self.paths.cache_dir, 'prototypes'),
            version='v1',
            use_negative=bool(int(os.getenv('POOLER_USE_NEG', '0'))),
            llmfirst_cfg=self.cfg.llmfirst,
        )

    def ensure_llm_backend(self) -> None:
        """Lazily construct the LLM backend so embeddings can load first."""

        if self.llm is not None:
            return

        backend = build_llm_backend(self.cfg.llm)
        self.llm = LLMLabeler(
            backend,
            self.label_config_bundle,
            self.cfg.llm,
            sc_cfg=self.cfg.scjitter,
            cache_dir=self.paths.cache_dir,
        )
        # Ensure the annotator has access to the materialised label configuration
        # so that option lookups during JSON prompting succeed.
        self.llm.label_config = self.label_config

    def _build_rerank_rule_overrides(self, rules_map: Mapping[str, str]) -> dict[str, str]:
        """Generate concise rule summaries for re-ranker queries when needed."""

        if not self.llm or not isinstance(rules_map, Mapping):
            return {}
        threshold_default = 150  # ~512-token budget
        try:
            threshold = int(os.getenv("RERANK_RULE_PARAPHRASE_CHARS", str(threshold_default)))
        except Exception:
            threshold = threshold_default
        threshold = max(150, threshold)

        overrides: dict[str, str] = {}
        for label_id, raw_text in rules_map.items():
            text = str(raw_text or "").strip()
            if len(text) <= threshold:
                continue
            try:
                summary = self.llm.summarize_label_rule_for_rerank(str(label_id), text)
            except Exception as exc:
                print(f"Reranker rule paraphrase failed for label {label_id}: {exc}")
                continue
            concise = str(summary or "").strip()
            if concise and len(concise) < len(text):
                overrides[str(label_id)] = concise
        return overrides
    def config_for_labelset(self, labelset_id: Optional[str]) -> Dict[str, object]:
        return self.label_config_bundle.config_for_labelset(labelset_id)

    def config_for_round(self, round_identifier: Optional[str]) -> Dict[str, object]:
        return self.label_config_bundle.config_for_round(round_identifier)

    def _attach_unit_metadata(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if df is None or getattr(self, "repo", None) is None:
            return df
        meta = self.repo.unit_metadata()
        if meta.empty:
            return df
        # Avoid duplicating metadata columns if they already exist in df.
        meta = meta[[c for c in meta.columns if c == "unit_id" or c not in df.columns]]
        if meta.shape[1] <= 1:  # only unit_id
            return df
        merged = df.merge(meta, on="unit_id", how="left")
        return merged

    def _label_maps(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Return legacy and current rule/type maps with latest label config overlays."""

        def _normalize_type(raw: Optional[object]) -> Optional[str]:
            if raw is None:
                return None
            text = str(raw).strip().lower()
            if not text:
                return None
            mapping = {
                "boolean": "binary",
                "bool": "binary",
                "yes/no": "binary",
                "yesno": "binary",
                "y/n": "binary",
                "yn": "binary",
                "binary": "binary",
                "categorical": "categorical",
                "category": "categorical",
                "multiclass": "categorical",
                "multi": "categorical",
                "options": "categorical",
                "option": "categorical",
                "text": "categorical",
                "string": "categorical",
                "free_text": "categorical",
                "numeric": "numeric",
                "number": "numeric",
                "int": "numeric",
                "integer": "numeric",
                "float": "numeric",
                "double": "numeric",
                "date": "date",
                "datetime": "date",
                "timestamp": "date",
            }
            return mapping.get(text, text)

        def _extract_rule_text(entry: Dict[str, object]) -> Optional[str]:
            candidates = []
            for key in ("rule_text", "rules", "prompt"):
                val = entry.get(key) if isinstance(entry, dict) else None
                if isinstance(val, str):
                    val = val.strip()
                    if val:
                        return val
                elif isinstance(val, list):
                    for item in reversed(val):
                        if isinstance(item, str):
                            text = item.strip()
                            if text:
                                return text
                        elif isinstance(item, dict):
                            text = str(item.get("text") or item.get("rule") or "").strip()
                            if text:
                                return text
                elif isinstance(val, dict):
                    text = str(val.get("text") or val.get("rule") or "").strip()
                    if text:
                        return text
            return None

        legacy_rules_map = {str(k): v for k, v in (self.repo.label_rules_by_label or {}).items() if v}
        legacy_label_types = {str(k): str(v) for k, v in (self.repo.label_types() or {}).items()}

        current_rules_map: dict[str, str] = {}
        current_label_types: dict[str, str] = {}

        for key, entry in (self.label_config or {}).items():
            if str(key) == "_meta":
                continue
            label_entry = entry if isinstance(entry, dict) else {}
            raw_id = label_entry.get("label_id") if isinstance(label_entry, dict) else None
            label_id = str(raw_id or key).strip()
            if not label_id:
                continue

            rule_text = _extract_rule_text(label_entry) if isinstance(label_entry, dict) else None
            if rule_text is not None:
                current_rules_map[label_id] = rule_text
            elif label_id not in current_rules_map:
                current_rules_map[label_id] = ""

            normalized_type = _normalize_type(label_entry.get("type") if isinstance(label_entry, dict) else None)
            if normalized_type:
                current_label_types[label_id] = normalized_type
            elif label_id not in current_label_types:
                current_label_types[label_id] = "categorical"

        if not current_rules_map:
            current_rules_map = dict(legacy_rules_map)
        if not current_label_types:
            current_label_types = dict(legacy_label_types)

        return legacy_rules_map, legacy_label_types, current_rules_map, current_label_types


    def _apply_excluded_units(self) -> int:
        removed = self.repo.exclude_units(self.excluded_unit_ids)
        if removed:
            self.rag._notes_by_doc = self.repo.notes_by_doc()
        return removed


    def run(self):
        # Ensure the LLM backend is available after embeddings are loaded.
        self.ensure_llm_backend()

        diversity_selector = DiversitySelector(self.repo, self.store, self.cfg.diversity)
        expander = DisagreementExpander(
            self.cfg.disagree,
            self.repo,
            self.rag,
            label_config_bundle=self.label_config_bundle,
            llmfirst_cfg=self.cfg.llmfirst,
            context_builder=self.context_builder,
        )
        disagreement_scorer = DisagreementScorer(
            self.repo,
            self.cfg.disagree,
            self.cfg.select,
            self.pooler,
            self.rag,
            expander,
            context_builder=self.context_builder,
            kcenter_select_fn=kcenter_select,
        )
        uncertainty_scorer = LLMUncertaintyScorer(self.cfg.llmfirst)
        selector = ActiveLearningSelector(self.cfg.select)

        pipeline = ActiveLearningPipeline(
            data_repo=self.repo,
            emb_store=self.store,
            ctx_builder=self.context_builder,
            llm_labeler=self.llm,
            disagreement_scorer=disagreement_scorer,
            uncertainty_scorer=uncertainty_scorer,
            diversity_selector=diversity_selector,
            selector=selector,
            config=self.cfg,
            paths=self.paths,
            pooler=self.pooler,
            retriever=self.rag,
            label_config=self.label_config,
            label_config_bundle=self.label_config_bundle,
            excluded_unit_ids=self.excluded_unit_ids,
            check_cancelled_fn=check_cancelled,
            iter_with_bar_fn=iter_with_bar,
            jsonify_cols_fn=_jsonify_cols,
            attach_metadata_fn=self._attach_unit_metadata,
            label_maps_fn=self._label_maps,
            rerank_override_fn=self._build_rerank_rule_overrides,
        )

        return pipeline.run()
        
        
# ------------------------------
# CLI
# ------------------------------

def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="LLM-first active learning next-batch selector (forced-choice + expanded disagreement + diversity)")
    ap.add_argument("--notes", required=True, help="Path to notes (csv|tsv|parquet|jsonl) with patienticn, doc_id, text")
    ap.add_argument("--annotations", required=True, help="Path to prior round annotations (csv|tsv|parquet|jsonl)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--label-config", type=str, default=None, help="Optional label config JSON")

    ap.add_argument("--batch-size", type=int, default=300)

    # FAISS index
    ap.add_argument("--index-type", type=str, default="flat", choices=["flat","hnsw","ivf"])
    ap.add_argument("--index-nlist", type=int, default=8192)
    ap.add_argument("--index-nprobe", type=int, default=32)
    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efSearch", type=int, default=64)
    ap.add_argument("--persist-faiss-index", dest="persist_faiss_index", action="store_true", default=True,
                    help="Persist FAISS index to disk (default: true)")
    ap.add_argument("--no-persist-faiss-index", dest="persist_faiss_index", action="store_false",
                    help="Disable persistence; rebuild FAISS index from embeddings each run")

    # LLM-first probe knobs
    ap.add_argument("--probe-per-label", type=int, default=400)
    ap.add_argument("--uncertain-top-pct", type=float, default=0.30)
    ap.add_argument("--certain-bottom-pct", type=float, default=0.15)
    ap.add_argument("--borderline-window", type=float, default=0.10)
    ap.add_argument("--stage-a-topk", type=int, default=2)
    ap.add_argument("--stage-b-topk", type=int, default=6)
    ap.add_argument("--per-label-cap-frac", type=float, default=0.20)
    
    #skip final llm labels
    ap.add_argument("--skip-final-llm-labeling", action='store_true', help='Skip provisional LLM labeling for the final selected batch')

    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    paths = Paths(notes_path=args.notes, annotations_path=args.annotations, outdir=args.outdir)
    cfg = OrchestratorConfig(
        index=IndexConfig(type=args.index_type, nlist=args.index_nlist, nprobe=args.index_nprobe, hnsw_M=args.hnsw_M, hnsw_efSearch=args.hnsw_efSearch, persist=args.persist_faiss_index),
        rag=RAGConfig(),
        llm=LLMConfig(),
        select=SelectionConfig(batch_size=args.batch_size),
        llmfirst=LLMFirstConfig(
            topk=args.topk,
        ),
        disagree=DisagreementConfig(),
        skip_final_llm_labeling=args.skip_final_llm_labeling  
    )
    label_cfg = {}
    if args.label_config and os.path.exists(args.label_config):
        try:
            label_cfg = json.load(open(args.label_config,"r",encoding="utf-8"))
        except Exception as e:
            LOGGER.info(f"[WARN] Failed to load label-config: {e}")
    runner = ActiveLearningLLMFirst(paths, cfg, label_config=label_cfg)
    runner.run()

if __name__ == "__main__":
    main()
