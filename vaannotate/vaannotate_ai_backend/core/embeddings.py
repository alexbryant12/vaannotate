"""Embedding models and index storage primitives for the VAAnnotate AI backend."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from ..utils.runtime import iter_with_bar as _iter_with_bar

if TYPE_CHECKING:
    from ..config import ModelConfig

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        raise ImportError("Please install langchain-text-splitters or langchain to use RecursiveCharacterTextSplitter.")
@dataclass
class IndexConfig:
    type: str = "flat"    # flat | hnsw | ivf
    nlist: int = 2048     # IVF lists
    nprobe: int = 32      # IVF search probes
    hnsw_M: int = 32      # HNSW graph degree
    hnsw_efSearch: int = 64
    persist: bool = True


def _detect_device():
    import os
    # Allow explicit override when the caller wants to pin models to a device.
    explicit = os.getenv("EMBEDDING_DEVICE") or os.getenv("MODEL_DEVICE")
    if explicit:
        return explicit
    if os.getenv("CPU_ONLY", "0") == "1":
        return "cpu"
    try:
        import torch
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            # When multiple GPUs are visible, prefer the second one to avoid
            # colliding with large LLMs that default to GPU-0 when using
            # auto-split or similar heuristics.
            if torch.cuda.device_count() >= 2:
                return "cuda:1"
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _ensure_default_ce_max_length(reranker: CrossEncoder, *, default: int = 512) -> None:
    """Ensure CrossEncoder inputs are truncated when max length is unspecified.

    Some sentence-transformer cross encoders ship without a ``max_length`` defined
    in their configuration. In those cases we fall back to a conservative value
    of 512 tokens to avoid unbounded sequence growth.
    """

    try:
        max_length = getattr(reranker, "max_length", None)
    except Exception:
        max_length = None

    if max_length is None:
        try:
            reranker.max_length = int(default)
        except Exception:
            pass


def build_models_from_env(model_cfg: "ModelConfig | None" = None) -> "Models":
    import os

    embed_name = (model_cfg.embed_model_name if model_cfg else None) or os.getenv(
        "MED_EMBED_MODEL_NAME"
    )
    rerank_name = (model_cfg.rerank_model_name if model_cfg else None) or os.getenv(
        "RERANKER_MODEL_NAME"
    )
    device = _detect_device()
    embedder = SentenceTransformer(embed_name, device=device)
    reranker = CrossEncoder(rerank_name, device=device)
    _ensure_default_ce_max_length(reranker)
    emb_bs = int(os.getenv('EMB_BATCH', '32' if device == "cpu" else "64"))
    rr_bs = int(os.getenv('RERANK_BATCH', '16' if device == "cpu" else "64"))
    return Models(embedder, reranker, device=device, emb_batch=emb_bs, rerank_batch=rr_bs)

class Models:
    def __init__(self, embedder: SentenceTransformer, reranker: CrossEncoder, device: str='cpu', emb_batch: int=64, rerank_batch: int=64):
        self.embedder = embedder; self.reranker = reranker
        self.device = device; self.emb_batch = emb_batch; self.rerank_batch = rerank_batch

class EmbeddingStore:
    def __init__(self, models: Models, cache_dir: str, normalize: bool = True):
        self.models = models; self.cache_dir = cache_dir; os.makedirs(cache_dir, exist_ok=True)
        self.normalize = normalize
        self.faiss_index = None
        self.chunk_meta: List[dict] = []
        self.X = None
        self.unit_to_chunk_idxs: Dict[str,List[int]] = {}
        self.bm25_indices: Dict[str, dict] = {}
        self.idf_global: Dict[str, float] = {}
        self.N_global: int = 0

    def _backfill_chunk_metadata(self, notes_df: "pd.DataFrame") -> None:
        """Ensure cached chunk metadata includes key fields such as notetype."""
        if not isinstance(notes_df, pd.DataFrame) or notes_df.empty or not self.chunk_meta:
            return
        if "notetype" not in notes_df.columns:
            return

        doc_to_notetype: Dict[str, str] = {}
        for row in notes_df.itertuples(index=False):
            if not hasattr(row, "doc_id") or not hasattr(row, "notetype"):
                continue
            doc_id = str(getattr(row, "doc_id"))
            note_type_val = getattr(row, "notetype")
            if note_type_val is None:
                continue
            note_type_str = str(note_type_val).strip()
            if note_type_str:
                doc_to_notetype[doc_id] = note_type_str

        if not doc_to_notetype:
            return

        for meta in self.chunk_meta:
            if meta.get("notetype"):
                continue
            doc_id = str(meta.get("doc_id"))
            note_type_str = doc_to_notetype.get(doc_id)
            if note_type_str:
                meta["notetype"] = note_type_str

        # Nothing else to do; chunk_meta is now enriched in-memory and will be
        # persisted on the next cache save.

    def _remap_unit_ids(self, notes_df: "pd.DataFrame") -> None:
        """Align cached chunk unit IDs with the unit granularity of the notes."""
        if not isinstance(notes_df, pd.DataFrame) or notes_df.empty or not self.chunk_meta:
            return
        if "doc_id" not in notes_df.columns or "unit_id" not in notes_df.columns:
            return

        doc_to_unit: Dict[str, str] = {}
        for row in notes_df.itertuples(index=False):
            doc_val = getattr(row, "doc_id", None)
            unit_val = getattr(row, "unit_id", None)
            if doc_val is None or unit_val is None:
                continue
            doc_str = str(doc_val)
            unit_str = str(unit_val)
            if not doc_str or unit_str.lower() in {"", "nan", "none"}:
                continue
            doc_to_unit[doc_str] = unit_str

        if not doc_to_unit:
            return

        changed = False
        for meta in self.chunk_meta:
            doc_val = meta.get("doc_id")
            if doc_val is None:
                continue
            doc_str = str(doc_val)
            target_unit = doc_to_unit.get(doc_str)
            if not target_unit:
                continue
            current_unit = str(meta.get("unit_id", ""))
            if current_unit != target_unit:
                meta["unit_id"] = target_unit
                changed = True

        if not changed:
            return

        unit_to_idxs = defaultdict(list)
        for idx, meta in enumerate(self.chunk_meta):
            unit_to_idxs[str(meta.get("unit_id", ""))].append(idx)
        self.unit_to_chunk_idxs = dict(unit_to_idxs)

    def _embedder_id(self) -> str:
        try:
            return getattr(self.models.embedder, "name_or_path", "") or str(self.models.embedder)
        except Exception:
            return "unknown_embedder"
    
    def _compute_corpus_fingerprint(self, notes_df: "pd.DataFrame", rag_cfg) -> str:
        """
        Build a stable fingerprint for the (documents + chunker + embedder + normalize) config.
        Prefers notes_df['hash'] if present; falls back to blake2 over doc_id+text.
        """
        h = hashlib.blake2b(digest_size=16)
        # chunker + embedder config
        h.update(f"chunk_size={rag_cfg.chunk_size},overlap={rag_cfg.chunk_overlap}".encode("utf-8"))
        h.update(f",normalize={bool(self.normalize)}".encode("utf-8"))
        h.update(f",embedder={self._embedder_id()}".encode("utf-8"))
    
        # corpus content (fast path uses a precomputed hash column if available)
        if "hash" in notes_df.columns:
            pairs = (notes_df[["doc_id", "hash"]]
                     .astype({"doc_id": str, "hash": str})
                     .sort_values("doc_id"))
            for row in pairs.itertuples(index=False):
                h.update(f"|{row.doc_id}:{row.hash}".encode("utf-8"))
        else:
            # fallback: cheap hash of doc_id + first/last 1024 chars + length (avoid hashing entire text)
            pairs = (notes_df[["doc_id", "text"]]
                     .astype({"doc_id": str, "text": str})
                     .sort_values("doc_id"))
            for row in pairs.itertuples(index=False):
                t = row.text or ""
                tview = (t[:1024] + "…" + t[-1024:]) if len(t) > 2048 else t
                mini = hashlib.blake2b(tview.encode("utf-8"), digest_size=8).hexdigest()
                h.update(f"|{row.doc_id}:{mini}:{len(t)}".encode("utf-8"))
        return h.hexdigest()
    
    def _chunk_cache_dir(self, fingerprint: str) -> str:
        d = os.path.join(self.cache_dir, "chunks", fingerprint)
        os.makedirs(d, exist_ok=True)
        return d
    
    def _manifest_path(self, chunk_dir: str) -> str:
        return os.path.join(chunk_dir, "manifest.json")
    
    def _load_manifest(self, chunk_dir: str) -> dict | None:
        p = self._manifest_path(chunk_dir)
        if os.path.exists(p):
            try:
                return json.load(open(p, "r", encoding="utf-8"))
            except Exception:
                return None
        return None
    
    def _save_manifest(self, chunk_dir: str, data: dict):
        p = self._manifest_path(chunk_dir)
        data = dict(data or {})
        data["saved_at"] = time.time()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def _update_manifest(self, chunk_dir: str, **fields: object) -> None:
        man = self._load_manifest(chunk_dir) or {}
        man.update(fields)
        self._save_manifest(chunk_dir, man)
    
    def _paths_for_cache(self, chunk_dir: str, index_cfg) -> dict:
        idx_name = f"faiss_{getattr(index_cfg, 'type', 'flat')}.index"
        return {
            "meta": os.path.join(chunk_dir, "chunk_meta.json.gz"),
            "emb": os.path.join(chunk_dir, "chunk_embeddings.npz"),
            "emb_legacy": os.path.join(chunk_dir, "chunk_embeddings.npy"),
            "faiss": os.path.join(chunk_dir, idx_name),
            "bm25": self._bm25_index_path(chunk_dir),
        }

    def _load_cached_embeddings(self, emb_path: str, legacy_path: str | None = None):
        candidates = [p for p in (emb_path, legacy_path) if p]
        for p in candidates:
            if not os.path.exists(p):
                continue
            if p.endswith(".npz"):
                with np.load(p) as data:
                    if "embeddings" in data:
                        return data["embeddings"]
                    if "arr_0" in data:
                        return data["arr_0"]
                continue
            return np.load(p, mmap_mode="r")
        raise FileNotFoundError("No cached embeddings found")
    
    def _try_load_cached_chunks(self, chunk_dir: str) -> tuple[list[dict] | None, np.ndarray | None, dict | None]:
        """
        Returns (chunk_meta, X_mmap, manifest) or (None, None, None) if unavailable.
        """
        paths = self._paths_for_cache(chunk_dir, type("Cfg", (), {"type": "flat"}))
        meta_p, emb_p = paths["meta"], paths["emb"]
        emb_legacy = paths.get("emb_legacy")
        man = self._load_manifest(chunk_dir)
        emb_exists = any(os.path.exists(p) for p in (emb_p, emb_legacy) if p)
        if not ((os.path.exists(meta_p) or os.path.exists(meta_p.replace(".gz", ""))) and emb_exists and man):
            return (None, None, None)
        try:
            meta = None
            for mp in (meta_p, meta_p.replace(".gz", "")):
                if not os.path.exists(mp):
                    continue
                try:
                    opener = gzip.open if mp.endswith(".gz") else open
                    meta = json.load(opener(mp, "rt", encoding="utf-8"))
                    break
                except Exception:
                    continue
            if meta is None:
                return (None, None, None)

            X = self._load_cached_embeddings(emb_p, emb_legacy)
            # sanity: manifest rows/dims
            n = int(man.get("n_chunks", -1))
            d = int(man.get("dim", -1))
            if (n > 0 and n != len(meta)) or (d > 0 and d != int(X.shape[1])):
                return (None, None, None)
            return (meta, X, man)
        except Exception:
            return (None, None, None)

    def _bm25_index_path(self, chunk_dir: str) -> str:
        return os.path.join(chunk_dir, "bm25_indices.json.gz")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        if not text:
            return []
        stopwords = {
            "the",
            "and",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "a",
            "an",
        }
        normalized = unicodedata.normalize("NFKC", str(text)).lower()
        tokens = re.findall(r"[a-z0-9_+\-/]+", normalized)
        return [tok for tok in tokens if tok and tok not in stopwords]

    def _build_bm25_indices(self) -> Dict[str, dict]:
        chunk_tokens: list[tuple[dict, List[str]]] = []
        global_df: Counter[str] = Counter()
        for meta in self.chunk_meta:
            toks = self._tokenize_for_bm25(meta.get("text", ""))
            if not toks:
                continue
            chunk_tokens.append((meta, toks))
            global_df.update(set(toks))

        N_global = len(chunk_tokens)
        eps = 1e-12
        # Keep common but clinically important terms (e.g., "pain", "blood") from
        # being downweighted into negative scores by adding a +1 offset, which was
        # used in the previous implementation.
        idf_global = {
            tok: math.log((N_global - freq + 0.5) / (freq + 0.5 + eps)) + 1.0
            for tok, freq in global_df.items()
        }

        unit_docs: Dict[str, list] = defaultdict(list)
        unit_metas: Dict[str, list] = defaultdict(list)
        for meta, toks in chunk_tokens:
            uid = str(meta.get("unit_id", ""))
            unit_docs[uid].append(toks)
            unit_metas[uid].append(meta)

        bm25_units: Dict[str, dict] = {}
        for uid, docs in unit_docs.items():
            if not docs:
                continue
            avgdl = sum(len(toks) for toks in docs) / float(len(docs))
            bm25_units[uid] = {
                "docs": docs,
                "metas": unit_metas.get(uid, []),
                "avgdl": avgdl,
            }

        self.idf_global = idf_global
        self.N_global = N_global
        return {"units": bm25_units, "idf_global": idf_global, "N_global": N_global}

    def _load_bm25_indices(self, chunk_dir: str) -> dict | None:
        primary = self._bm25_index_path(chunk_dir)
        legacy = os.path.join(chunk_dir, "bm25_indices.json")
        for path in (primary, legacy):
            if not os.path.exists(path):
                continue
            try:
                opener = gzip.open if path.endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                units = data.get("units") or data.get("indices") or {}
                if not isinstance(units, dict):
                    continue
                expected_n = data.get("n_chunks")
                if expected_n is not None and expected_n != len(self.chunk_meta):
                    continue
                idf_global = data.get("idf_global") or {}
                N_global = data.get("N_global") or expected_n or 0
                self.idf_global = idf_global
                self.N_global = int(N_global)
                return {"units": units, "idf_global": idf_global, "N_global": N_global}
            except Exception:
                continue
        return None

    def _save_bm25_indices(self, chunk_dir: str, indices: dict) -> None:
        units = indices.get("units") if isinstance(indices, dict) else indices
        payload = {
            "version": "v2",
            "n_chunks": len(self.chunk_meta),
            "units": units,
            "idf_global": indices.get("idf_global", getattr(self, "idf_global", {})) if isinstance(indices, dict) else getattr(self, "idf_global", {}),
            "N_global": indices.get("N_global", getattr(self, "N_global", 0)) if isinstance(indices, dict) else getattr(self, "N_global", 0),
        }
        p = self._bm25_index_path(chunk_dir)
        tmp = p + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, p)
        self._update_manifest(chunk_dir, bm25_version="v2")

    def bm25_index_for_unit(self, unit_id: str) -> dict | None:
        uid = str(unit_id)
        bm25_units = self.bm25_indices.get("units") if isinstance(self.bm25_indices, dict) and "units" in self.bm25_indices else self.bm25_indices
        if isinstance(bm25_units, dict):
            cached = bm25_units.get(uid)
            if cached is not None:
                return cached

        if hasattr(self, "_chunk_cache_dir_path"):
            loaded = self._load_bm25_indices(self._chunk_cache_dir_path)
            if isinstance(loaded, dict):
                self.bm25_indices = loaded
                bm25_units = loaded.get("units") if "units" in loaded else loaded
                hit = bm25_units.get(uid) if isinstance(bm25_units, dict) else None
                if hit is not None:
                    return hit

        if not self.chunk_meta:
            return None

        rebuilt = self._build_bm25_indices()
        if not rebuilt:
            return None

        self.bm25_indices = rebuilt
        self.idf_global = rebuilt.get("idf_global", getattr(self, "idf_global", {}))
        self.N_global = int(rebuilt.get("N_global", getattr(self, "N_global", 0)))
        if hasattr(self, "_chunk_cache_dir_path"):
            try:
                self._save_bm25_indices(self._chunk_cache_dir_path, rebuilt)
            except Exception:
                pass
        bm25_units = rebuilt.get("units") if "units" in rebuilt else rebuilt
        return bm25_units.get(uid) if isinstance(bm25_units, dict) else None
    
    def _save_cached_chunks(self, chunk_dir: str, meta: list[dict], X: np.ndarray, rag_cfg):
        paths = self._paths_for_cache(chunk_dir, type("Cfg", (), {"type": "flat"}))
        meta_p, emb_p = paths["meta"], paths["emb"]
        emb_legacy = paths.get("emb_legacy")
        # Save meta
        tmp = meta_p + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, meta_p)
        # Save embeddings (np.save is atomic-ish via temp file on most FS; to be safe, write to tmp then replace)
        X_to_save = X.astype(np.float16)
        emb_tmp = emb_p + ".tmp.npz"
        np.savez_compressed(emb_tmp, embeddings=X_to_save)
        os.replace(emb_tmp, emb_p)
        if emb_legacy and os.path.exists(emb_legacy):
            try:
                os.remove(emb_legacy)
            except Exception:
                pass
        # Manifest
        self._save_manifest(chunk_dir, {
            "n_chunks": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "chunk_size": int(rag_cfg.chunk_size),
            "chunk_overlap": int(rag_cfg.chunk_overlap),
            "normalize": bool(self.normalize),
            "embedder": self._embedder_id(),
            "emb_dtype": str(X_to_save.dtype),
            "meta_compressed": True,
            "emb_compressed": True,
            "version": "v2",
        })
    
    def _try_load_faiss_index(self, faiss_path: str, expected_n: int) -> "faiss.Index" | None:
        if not (os.path.exists(faiss_path) and "faiss" in globals() and faiss is not None):
            return None
        try:
            idx = faiss.read_index(faiss_path)
            ntotal = getattr(idx, "ntotal", None)
            if ntotal == int(expected_n):
                return idx
            return None
        except Exception:
            return None

    def _build_faiss_index(self, X: np.ndarray, index_cfg) -> "faiss.Index":
        if faiss is None:
            raise ImportError("faiss-cpu is required")
        d = int(X.shape[1])
        if index_cfg.type == "flat":
            idx = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
            idx.add(X)
            return idx
        if index_cfg.type == "hnsw":
            idx = faiss.IndexHNSWFlat(d, index_cfg.hnsw_M)
            idx.hnsw.efSearch = index_cfg.hnsw_efSearch
            idx.add(X)
            return idx
        if index_cfg.type == "ivf":
            quant = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
            idx = faiss.IndexIVFFlat(
                quant,
                d,
                index_cfg.nlist,
                faiss.METRIC_INNER_PRODUCT if self.normalize else faiss.METRIC_L2,
            )
            ntrain = min(X.shape[0], max(10000, index_cfg.nlist * 40))
            samp = X[np.random.choice(X.shape[0], ntrain, replace=False)]
            idx.train(samp)
            idx.add(X)
            idx.nprobe = index_cfg.nprobe
            return idx
        if index_cfg.type == "ivfpq":
            quant = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
            idx = faiss.IndexIVFPQ(quant, d, index_cfg.nlist, index_cfg.pq_m, index_cfg.pq_bits)
            ntrain = min(X.shape[0], max(50000, getattr(index_cfg, "train_size", 100000)))
            samp = X[np.random.choice(X.shape[0], ntrain, replace=False)]
            idx.train(samp)
            idx.add(X)
            idx.nprobe = index_cfg.nprobe
            return idx
        raise ValueError(f"Unknown index type: {index_cfg.type}")

    def _embed(self, texts: List[str], show_bar: Optional[bool] = False) -> np.ndarray:
        if show_bar:
            bs = getattr(self.models, 'emb_batch', 64)
            out = []
            for i in _iter_with_bar("Embedding chunks",
                                   range(0, len(texts), bs),
                                   total=(len(texts)+bs-1)//bs,
                                   min_interval_s=10):
                batch = texts[i:i+bs]
                embs  = self.models.embedder.encode(
                    batch, batch_size=bs, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=self.normalize
                )
                out.append(embs.astype("float32"))
            return np.vstack(out)
        else:
            embs = self.models.embedder.encode(texts, batch_size=getattr(self.models, 'emb_batch', 64), show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=self.normalize)
            return embs.astype("float32")

    def build_chunk_index(
        self,
        notes_df: "pd.DataFrame",
        rag_cfg,
        index_cfg=None,
        *,
        force_rechunk: bool = False,
        force_reembed: bool = False,
        force_reindex: bool = False,
    ):
        """
        Build or reuse cached chunk_meta, embeddings, and FAISS index.
    
        Cache layout: <cache_dir>/chunks/<fingerprint>/{chunk_meta.json, chunk_embeddings.npz, faiss_*.index, manifest.json}
        Fingerprint depends on corpus (doc_id + hash), splitter, embedder, normalize flag.
        """
        index_cfg = index_cfg or IndexConfig()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_cfg.chunk_size, chunk_overlap=rag_cfg.chunk_overlap
        )
    
        # 1) Compute fingerprint and cache dir
        fp = self._compute_corpus_fingerprint(notes_df, rag_cfg)
        chunk_dir = self._chunk_cache_dir(fp)
        paths = self._paths_for_cache(chunk_dir, index_cfg)
    
        # 2) Try reuse chunk_meta + embeddings
        meta = X = man = None
        if not force_rechunk and not force_reembed:
            meta, X, man = self._try_load_cached_chunks(chunk_dir)
    
        # 2a) If no cached chunks (or forced), (re)compute
        if meta is None or X is None:
             # --- chunk from notes_df with progress ---
            chunk_meta = []
            total_docs = int(len(notes_df))
            for row in _iter_with_bar(
                    step="Chunking docs",
                    iterable=notes_df.itertuples(index=False),
                    total=total_docs,
                    min_interval_s=float(getattr(getattr(self, "models", None), "progress_min_interval_s", 0.6) or 0.6)):
                # Chunk the text
                chunks = splitter.split_text(getattr(row, "text"))
                for i, ch in enumerate(chunks):
                    md = {"unit_id": getattr(row, "unit_id"),
                          "doc_id": getattr(row, "doc_id"),
                          "chunk_id": i,
                          "text": ch}
                    # copy a few metadata fields if they exist
                    for k in ("date_note", "notetype", "doc_title", "author", "source_system",
                              "document_metadata_json", "metadata_json"):
                        if hasattr(row, k):
                            v = getattr(row, k)
                            if v is not None:
                                md[k] = str(v) if k in {"date_note", "notetype", "doc_title", "author", "source_system"} else v
                    chunk_meta.append(md)
            
            if not chunk_meta:
                raise RuntimeError("No chunks generated from notes")

            # Embed (even if force_reembed only)
            texts = [m["text"] for m in chunk_meta]
            X = self._embed(texts, show_bar=True)  # float32; normalize handled by self.normalize
            # Save cache
            print("Saving chunks+embeddings to disk...")
            self._save_cached_chunks(chunk_dir, chunk_meta, X, rag_cfg)
            meta = chunk_meta
    
        # 3) Bind to store
        if isinstance(X, np.ndarray):
            self.X = X.astype(np.float32) if X.dtype != np.float32 else X
        elif isinstance(X, np.memmap):
            self.X = np.asarray(X, dtype=np.float32)
        else:
            loaded = self._load_cached_embeddings(paths["emb"], paths.get("emb_legacy"))
            self.X = loaded.astype(np.float32) if loaded.dtype != np.float32 else loaded
        self.chunk_meta = meta
        self._remap_unit_ids(notes_df)
        self._backfill_chunk_metadata(notes_df)
        unit_to_idxs = defaultdict(list)
        for i, m in enumerate(self.chunk_meta):
            unit_to_idxs[m["unit_id"]].append(i)
        self.unit_to_chunk_idxs = dict(unit_to_idxs)

        bm25_loaded = self._load_bm25_indices(chunk_dir)
        if bm25_loaded is None:
            bm25_loaded = self._build_bm25_indices()
            if bm25_loaded:
                try:
                    self._save_bm25_indices(chunk_dir, bm25_loaded)
                except Exception:
                    pass
        self.bm25_indices = bm25_loaded or {}

        # 4) FAISS index: try load; else build and persist
        if faiss is None:
            raise ImportError("faiss-cpu is required")
        persist_index = bool(getattr(index_cfg, "persist", True))
        idx = None
        if persist_index and not force_reindex:
            idx = self._try_load_faiss_index(paths["faiss"], expected_n=int(self.X.shape[0]))
        if idx is None:
            print("Building index...")
            idx = self._build_faiss_index(self.X, index_cfg)
            # Persist index
            if persist_index:
                try:
                    print("Saving index...")
                    faiss.write_index(idx, paths["faiss"])
                except Exception:
                    pass

        self.faiss_index = idx
        # Keep the cache dir for later search() lazy loads
        self._chunk_cache_dir_path = chunk_dir
        return self

    def search(self, query_texts: list[str], topk: int = 50, index_cfg=None) -> tuple[np.ndarray, np.ndarray]:
        index_cfg = index_cfg or IndexConfig()
        if self.faiss_index is None:
            # try to reuse the last chunk cache dir
            if hasattr(self, "_chunk_cache_dir_path"):
                paths = self._paths_for_cache(self._chunk_cache_dir_path, index_cfg)
                # load X + meta
                emb = self._load_cached_embeddings(paths["emb"], paths.get("emb_legacy"))
                self.X = emb.astype(np.float32) if emb.dtype != np.float32 else emb
                meta = None
                for mp in (paths["meta"], paths["meta"].replace(".gz", "")):
                    if not os.path.exists(mp):
                        continue
                    try:
                        opener = gzip.open if mp.endswith(".gz") else open
                        meta = json.load(opener(mp, "rt", encoding="utf-8"))
                        break
                    except Exception:
                        continue
                self.chunk_meta = meta or []
                # load or build index
                persist_index = bool(getattr(index_cfg, "persist", True))
                idx = None
                if persist_index:
                    idx = self._try_load_faiss_index(paths["faiss"], expected_n=int(self.X.shape[0]))
                if idx is None:
                    idx = self._build_faiss_index(self.X, index_cfg)
                self.faiss_index = idx
                # rebuild unit_to_chunk lookup
                unit_to_idxs = defaultdict(list)
                for i, m in enumerate(self.chunk_meta):
                    unit_to_idxs[str(m["unit_id"])].append(i)
                self.unit_to_chunk_idxs = dict(unit_to_idxs)
                bm25_loaded = self._load_bm25_indices(self._chunk_cache_dir_path)
                if bm25_loaded is None:
                    bm25_loaded = self._build_bm25_indices()
                    if bm25_loaded:
                        try:
                            self._save_bm25_indices(self._chunk_cache_dir_path, bm25_loaded)
                        except Exception:
                            pass
                self.bm25_indices = bm25_loaded or {}
            else:
                raise RuntimeError("FAISS index not built; call build_chunk_index() first.")
    
        # set runtime search params if applicable
        try:
            if hasattr(self.faiss_index, "nprobe") and index_cfg:
                self.faiss_index.nprobe = index_cfg.nprobe
            if hasattr(self.faiss_index, "hnsw") and index_cfg:
                self.faiss_index.hnsw.efSearch = index_cfg.hnsw_efSearch
        except Exception:
            pass
    
        Q = self._embed(query_texts)
        sims, idxs = self.faiss_index.search(Q, topk)
        return sims, idxs

    def get_patient_chunk_indices(self, unit_id: str) -> List[int]:
        uid = str(unit_id)
        if self.unit_to_chunk_idxs:
            return self.unit_to_chunk_idxs.get(uid, [])
        if not self.chunk_meta:
            Mp = os.path.join(self.cache_dir, "chunk_meta.json.gz")
            for mp in (Mp, Mp.replace(".gz", "")):
                if not os.path.exists(mp):
                    continue
                try:
                    opener = gzip.open if mp.endswith(".gz") else open
                    self.chunk_meta = json.load(opener(mp, "rt", encoding="utf-8"))
                    break
                except Exception:
                    continue
        return [i for i,m in enumerate(self.chunk_meta) if m.get("unit_id")==uid]


