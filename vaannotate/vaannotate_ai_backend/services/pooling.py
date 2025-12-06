"""Label-aware pooling and k-center selection utilities used by Active Learning."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import LLMFirstConfig
from ..core.data import DataRepository
from ..core.embeddings import EmbeddingStore, Models
from ..services.rag_retriever import RAGRetriever
from ..services.context_builder import ContextBuilder


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
