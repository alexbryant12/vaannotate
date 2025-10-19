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
  • Cost-aware ranking (IG/min) to maximize labels per hour
  • Safe Parquet writing (JSON-encode nested cols)
"""

from __future__ import annotations
import os, re, sys, json, math, time, uuid, random
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any

# ------------------------------
# Logging (JSON) and Telemetry
# ------------------------------
import logging, time, uuid

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        for k, v in getattr(record, "__dict__", {}).items():
            if k in ("args","msg","levelname","levelno","pathname","filename","module","exc_info","exc_text","stack_info","lineno","funcName","created","msecs","relativeCreated","thread","threadName","processName","process"): 
                continue
            if k.startswith("_"): 
                continue
            if k in base: 
                continue
            try:
                json.dumps({k: v}); base[k] = v
            except Exception:
                base[k] = str(v)
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    return logger

LOGGER = setup_logging()

class Telemetry:
    def __init__(self):
        self.t0 = time.time()
        self.times = {}
        self.counts = {}
        self.meta = {"run_id": str(uuid.uuid4())}

    def start(self, stage):
        self.times.setdefault(stage, 0.0)
        self.meta[f"_{stage}_t"] = time.time()

    def end(self, stage):
        t = self.meta.pop(f"_{stage}_t", None)
        if t is not None:
            self.times[stage] = self.times.get(stage, 0.0) + (time.time() - t)

    def tick(self, metric, inc=1):
        self.counts[metric] = self.counts.get(metric, 0) + int(inc)

    def as_dict(self):
        out = {"elapsed": time.time() - self.t0, "times": self.times, "counts": self.counts}
        out.update(self.meta); return out

    def write_json(self, path: str):
        try:
            data = json.dumps(self.as_dict(), ensure_ascii=False, indent=2).encode("utf-8")
            atomic_write_bytes(path, data)
            LOGGER.info("telemetry_write", file=path, elapsed=self.as_dict().get("elapsed"))
        except Exception as e:
            LOGGER.warning("telemetry_write_failed", file=path, error=str(e), exc_info=True)

import numpy as np
import pandas as pd

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None

from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from cachetools import LRUCache
except Exception:
    from collections import OrderedDict
    class LRUCache(dict):
        def __init__(self, maxsize=10000):
            super().__init__(); self._order = OrderedDict(); self._max = maxsize
        def __setitem__(self, k, v):
            if k in self._order: self._order.move_to_end(k)
            self._order[k] = None
            super().__setitem__(k, v)
            if len(self._order) > self._max:
                old, _ = self._order.popitem(last=False)
                super().pop(old, None)
        def get(self, k, default=None):
            if k in self._order: self._order.move_to_end(k)
            return super().get(k, default)


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        raise ImportError("Please install langchain-text-splitters or langchain to use RecursiveCharacterTextSplitter.")

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


# ------------------------------
# Small utilities

# ---------- Stable hashing (reproducible across runs) ----------
def stable_hash_str(s: str, digest_size: int = 8) -> str:
    import hashlib
    if s is None: s = ""
    return hashlib.blake2b(str(s).encode('utf-8'), digest_size=digest_size).hexdigest()

def stable_hash_pair(a: str, b: str, digest_size: int = 12) -> str:
    import hashlib
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    return hashlib.blake2b((a + "\x1f" + b).encode('utf-8'), digest_size=digest_size).hexdigest()

# ------------------------------

def read_table(path: str) -> pd.DataFrame:
    ext = path.lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(path)
    if ext == "tsv":
        return pd.read_csv(path, sep="\t")
    if ext in ("parquet","pq"):
        return pd.read_parquet(path)
    if ext == "jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table extension: {path}")


def atomic_write_bytes(path: str, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def write_table(df: pd.DataFrame, path: str):

    ext = path.lower().split(".")[-1]
    if ext == "csv":
        df.to_csv(path, index=False)
    elif ext in ("parquet","pq"):
        df.to_parquet(path, index=False)
    elif ext == "jsonl":
        df.to_json(path, lines=True, orient="records", force_ascii=False)
    else:
        raise ValueError(f"Unsupported table extension: {path}")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+"," ", s).strip()

def safe_json_loads(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (dict,list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        try:
            import ast
            return ast.literal_eval(x)
        except Exception:
            return None

def normalize01(a: np.ndarray) -> np.ndarray:
    if a.size == 0: return a
    mn, mx = a.min(), a.max()
    if mx <= mn: return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def set_all_seeds(seed: int = 123):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            LOGGER.warning('caught_exception', error=str(e), exc_info=True)
    except Exception as e:
            LOGGER.warning('caught_exception', error=str(e), exc_info=True)


def strings_contain_uncertainty(strings) -> bool:
    if strings is None: return False
    try:
        if hasattr(strings, "dropna") and hasattr(strings, "tolist"):
            iterable = strings.dropna().tolist()
        elif isinstance(strings, (list,tuple,set)):
            iterable = list(strings)
        else:
            iterable = [strings]
    except TypeError:
        iterable = [strings]
    pat = re.compile(r"\b(uncertain|unsure|maybe|ambiguous|not sure|\?|equivocal|difficult)\b", re.I)
    for s in iterable:
        if s is None: continue
        if pat.search(str(s)): return True
    return False

def _jsonify_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return out


# ------------------------------
# Config
# ------------------------------

@dataclass
class IndexConfig:
    type: str = "flat"    # flat | hnsw | ivf
    nlist: int = 8192     # IVF lists
    nprobe: int = 32      # IVF search probes
    hnsw_M: int = 32      # HNSW graph degree
    hnsw_efSearch: int = 64
    persist: bool = True

@dataclass
class CostConfig:
    enable: bool = True
    base_min: float = 0.3           # base minutes per (patient,label)
    per_1k_chars_min: float = 0.06  # minutes per 1k chars
    per_note_min: float = 0.02      # minutes per note
    weight_uncertain: float = 1.0
    weight_certain: float = 0.2
    weight_disagree: float = 0.7
    weight_diversity: float = 0.4

@dataclass
class RAGConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 150
    normalize_embeddings: bool = True
    per_label_topk: int = 6
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    mmr_candidates: int = 200
    use_keywords: bool = True
    keyword_topk: int = 20
    rationale_boost: float = 0.2

@dataclass
class LLMConfig:
    enable: bool = True
    model_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    temperature: float = 0.2
    n_consistency: int = 2
    max_parallel: int = 8
    logprobs: bool = True
    top_logprobs: int = 0           # optional 0..20 for JSON call
    prediction_field: str = "prediction"
    seed: int = 42
    timeout: float = 60.0
    budget_max_calls: int = 8000
    retry_max: int = 3
    retry_backoff: float = 2.0
    max_context_chars: int = 12000
    json_version: str = "1.0"

@dataclass
class SelectionConfig:
    batch_size: int = 300
    pct_disagreement: float = 0.45
    pct_uncertain: float = 0.35      # LLM-uncertain
    pct_easy_qc: float = 0.10        # LLM-certain
    pct_diversity: float = 0.10
    random_seed: int = 123

@dataclass
class LLMFirstConfig:
    enable: bool = True
    per_label_probe_count: int = 400
    uncertain_top_pct: float = 0.3
    certain_bottom_pct: float = 0.15
    borderline_window: float = 0.1
    escalate_borderline: bool = True
    stage_a_topk: int = 2
    stage_b_topk: int = 6
    per_label_cap_fraction: float = 0.2
    kcenter_diversity: bool = True
    diversity_prototype_free_frac: float = 0.25  # diversity portion in prototype-free space
    # forced-choice micro-probe
    fc_enable: bool = True
    fc_temperature: float = 0.0
    fc_top_logprobs: int = 15
    fc_max_tokens: int = 1
    fc_w_entropy: float = 0.65
    fc_w_disagree: float = 0.35

@dataclass
class DisagreementConfig:
    round_policy: str = 'last'       # 'last' | 'all' | 'decay'
    decay_half_life: float = 2.0     # if round_policy='decay'
    uncertainty_note_boost: float = 0.1
    high_entropy_threshold: float = 0.5
    seeds_per_label: int = 20
    snippets_per_seed: int = 3
    similar_chunks_per_seed: int = 100
    expanded_per_label: int = 200
    per_label_cap_fraction: float = 0.25

@dataclass
class Paths:
    notes_path: str
    annotations_path: str
    outdir: str
    cache_dir: str = field(init=False)
    def __post_init__(self):
        self.cache_dir = os.path.join(self.outdir, "cache")
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

@dataclass
class OrchestratorConfig:
    index: IndexConfig = field(default_factory=IndexConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    select: SelectionConfig = field(default_factory=SelectionConfig)
    llmfirst: LLMFirstConfig = field(default_factory=LLMFirstConfig)
    disagree: DisagreementConfig = field(default_factory=DisagreementConfig)


# ------------------------------
# Data repository
# ------------------------------

class DataRepository:
    def __init__(self, notes_df: pd.DataFrame, ann_df: pd.DataFrame):
        required_notes = {"patienticn","doc_id","text"}
        if not required_notes.issubset(set(notes_df.columns)):
            raise ValueError(f"Notes missing {required_notes}")
        required_ann = {"round_id","unit_id","doc_id","label_id","reviewer_id","label_value"}
        if not required_ann.issubset(set(ann_df.columns)):
            raise ValueError(f"Annotations missing {required_ann}")

        self.notes = notes_df.copy()
        self.notes["unit_id"] = self.notes["patienticn"].astype(str)
        self.notes["doc_id"] = self.notes["doc_id"].astype(str)
        self.notes["text"] = self.notes["text"].astype(str).map(normalize_text)

        self.ann = ann_df.copy()
        for col in ("unit_id","doc_id","label_id","reviewer_id"):
            self.ann[col] = self.ann[col].astype(str)
        if "document_text" in self.ann.columns:
            self.ann["document_text"] = self.ann["document_text"].astype(str).map(normalize_text)
        else:
            self.ann["document_text"] = ""
        for col in ("rationales_json","document_metadata_json"):
            if col in self.ann.columns:
                self.ann[col] = self.ann[col].apply(safe_json_loads)
            else:
                self.ann[col] = None
        for col in ("label_rules","reviewer_notes"):
            if col in self.ann.columns:
                self.ann[col] = self.ann[col].astype(str)
            else:
                self.ann[col] = ""

        self.label_rules_by_label = self._collect_label_rules()

    def _collect_label_rules(self) -> Dict[str,str]:
        rules = {}
        if "label_rules" in self.ann.columns:
            df = self.ann[["label_id","label_rules"]].dropna()
            for lid, grp in df.groupby("label_id"):
                vals = [v for v in grp["label_rules"].tolist() if isinstance(v,str) and v.strip()]
                if vals:
                    rules[lid] = vals[-1]
        return rules

    def notes_by_doc(self) -> Dict[str,str]:
        return dict(zip(self.notes["doc_id"].tolist(), self.notes["text"].tolist()))

    def label_types(self) -> Dict[str,str]:
        types = {}
        for lid, grp in self.ann.groupby("label_id"):
            vals = [str(v).lower() for v in grp["label_value"].tolist() if v is not None and not (isinstance(v,float) and math.isnan(v))]
            uniq = set(vals)
            if uniq.issubset({"0","1","true","false","present","absent","yes","no","neg","pos","positive","negative"}):
                types[lid] = "binary"
            else:
                types[lid] = "text"
        return types

    def reviewer_disagreement(self, round_policy: str = 'last', decay_half_life: float = 2.0, uncertainty_note_boost: float = 0.0) -> pd.DataFrame:
        key = ["round_id","unit_id","label_id","reviewer_id"]
        rows = []
        ann = self.ann.copy()
        # Determine round ordering
        try:
            ann["_round_ord"] = pd.to_numeric(ann["round_id"], errors="coerce")
            if ann["_round_ord"].notnull().any():
                ord_series = ann["_round_ord"].fillna(ann["round_id"].astype("category").cat.codes)
            else:
                ord_series = ann["round_id"].astype("category").cat.codes
        except Exception:
            ord_series = ann["round_id"].astype("category").cat.codes
        ann["_round_ord"] = ord_series
        last_ord = int(ann["_round_ord"].max()) if len(ann) else 0

        if round_policy == 'last':
            ann = ann[ann["_round_ord"] == last_ord]
        elif round_policy == 'decay':
            pass  # keep all, weight entropy later
        else:
            pass  # 'all'

        for k, grp in ann.groupby(key):
            vals = grp["label_value"]
            try:
                mc = Counter(vals.tolist()).most_common(1)[0][0]
            except Exception:
                mc = vals.iloc[0] if len(vals) else None
            has_unc = strings_contain_uncertainty(grp.get("reviewer_notes", []))
            rows.append({"round_id":k[0], "unit_id":k[1], "label_id":k[2], "reviewer_id":k[3],
                         "label_value":mc, "has_uncertainty_note":has_unc, "_round_ord": int(grp["_round_ord"].iloc[0])})
        agg = pd.DataFrame(rows)

        out = []
        for (uid,lid), grp in agg.groupby(["unit_id","label_id"]):
            vals = [str(v) for v in grp["label_value"].tolist() if v is not None]
            cnt = Counter(vals)
            total = sum(cnt.values()) if cnt else 0
            probs = np.array([c/total for c in cnt.values()], dtype=float) if total>0 else np.array([])
            ent = -np.sum(probs * np.log2(np.clip(probs,1e-12,1))) if probs.size else 0.0
            ent_norm = float(ent / math.log2(max(len(cnt),2)))
            if round_policy == 'decay' and len(grp) > 0:
                deltas = last_ord - grp["_round_ord"].to_numpy()
                w = float(np.mean(np.exp(-deltas / max(1e-6, decay_half_life))))
                ent_norm *= w
            has_unc_note = bool(any(grp["has_uncertainty_note"])) if "has_uncertainty_note" in grp.columns else False
            if has_unc_note and uncertainty_note_boost > 0.0:
                ent_norm = min(1.0, ent_norm + float(uncertainty_note_boost))
            out.append({"unit_id": uid, "label_id": lid, "disagreement_score": ent_norm,
                        "has_uncertainty_note": has_unc_note, "n_reviewers": int(grp["reviewer_id"].nunique())})
        return pd.DataFrame(out)

    def get_prior_rationales(self, unit_id: str, label_id: str) -> List[dict]:
        sub = self.ann[(self.ann["unit_id"]==unit_id) & (self.ann["label_id"]==label_id)]
        spans = []
        for r in sub.itertuples(index=False):
            lst = r.rationales_json
            if isinstance(lst, list):
                for sp in lst:
                    if isinstance(sp, dict) and sp.get("snippet"):
                        spans.append(sp)
        return spans

    def last_round_consensus(self) -> Dict[Tuple[str,str], str]:
        # {(unit_id,label_id): majority_label_value} for last round
        ann = self.ann.copy()
        try:
            ann["_round_ord"] = pd.to_numeric(ann["round_id"], errors="coerce")
            ord_series = ann["_round_ord"].fillna(ann["round_id"].astype("category").cat.codes)
        except Exception:
            ord_series = ann["round_id"].astype("category").cat.codes
        ann["_round_ord"] = ord_series
        last_ord = int(ann["_round_ord"].max()) if len(ann) else 0
        ann = ann[ann["_round_ord"]==last_ord]
        res = {}
        for (u,l), g in ann.groupby(["unit_id","label_id"]):
            vals = [str(v).lower() for v in g["label_value"].tolist() if v is not None]
            if vals:
                res[(u,l)] = Counter(vals).most_common(1)[0][0]
        return res


# ------------------------------
# Embedding, FAISS, Retrieval
# ------------------------------

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

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.models.embedder.encode(texts, batch_size=getattr(self.models, 'emb_batch', 64), show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return embs.astype("float32")

    def build_chunk_index(self, notes_df: pd.DataFrame, rag_cfg, index_cfg=None):
        index_cfg = index_cfg or IndexConfig()
        splitter = RecursiveCharacterTextSplitter(chunk_size=rag_cfg.chunk_size, chunk_overlap=rag_cfg.chunk_overlap)
        chunk_texts, meta = [], []
        unit_to_idxs = defaultdict(list)
        for row in notes_df.itertuples(index=False):
            chunks = splitter.split_text(row.text)
            for i, ch in enumerate(chunks):
                meta.append({"unit_id": row.unit_id, "doc_id": row.doc_id, "chunk_id": i, "text": ch})
                chunk_texts.append(ch)
        if not chunk_texts: raise RuntimeError("No chunks generated from notes")
        X = self._embed(chunk_texts)
        d = X.shape[1]
        # persist embeddings + meta
        emb_path = os.path.join(self.cache_dir, "chunk_embeddings.npy")
        np.save(emb_path, X)
        json.dump(meta, open(os.path.join(self.cache_dir,"chunk_meta.json"),"w",encoding="utf-8"))
        self.X = np.load(emb_path, mmap_mode='r')
        self.chunk_meta = meta
        for i,m in enumerate(meta):
            unit_to_idxs[m["unit_id"]].append(i)
        self.unit_to_chunk_idxs = dict(unit_to_idxs)
        # build/persist FAISS
        if faiss is None: raise ImportError("faiss-cpu is required")
        index_path = os.path.join(self.cache_dir, f"faiss_{index_cfg.type}.index")
        if index_cfg.persist and os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
        else:
            if index_cfg.type == "flat":
                self.faiss_index = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
                self.faiss_index.add(X)
            elif index_cfg.type == "hnsw":
                self.faiss_index = faiss.IndexHNSWFlat(d, index_cfg.hnsw_M)
                self.faiss_index.hnsw.efSearch = index_cfg.hnsw_efSearch
                self.faiss_index.add(X)
            elif index_cfg.type == "ivf":
                quant = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
                self.faiss_index = faiss.IndexIVFFlat(quant, d, index_cfg.nlist, faiss.METRIC_INNER_PRODUCT if self.normalize else faiss.METRIC_L2)
                ntrain = min(X.shape[0], max(10000, index_cfg.nlist*40))
                samp = X[np.random.choice(X.shape[0], ntrain, replace=False)]
                self.faiss_index.train(samp)
                self.faiss_index.add(X)
                self.faiss_index.nprobe = index_cfg.nprobe
            else:
                raise ValueError(f"Unknown index type: {index_cfg.type}")
            if index_cfg.persist:
                faiss.write_index(self.faiss_index, index_path)
        return self

    def search(self, query_texts: List[str], topk: int=50, index_cfg: Optional[IndexConfig]=None) -> Tuple[np.ndarray,np.ndarray]:

        index_cfg = index_cfg or IndexConfig()
        if self.faiss_index is None:
            Xp = os.path.join(self.cache_dir,"chunk_embeddings.npy")
            Mp = os.path.join(self.cache_dir,"chunk_meta.json")
            if not (os.path.exists(Xp) and os.path.exists(Mp)):
                raise RuntimeError("FAISS index not built")
            self.X = np.load(Xp, mmap_mode='r')
            self.chunk_meta = json.load(open(Mp,"r",encoding="utf-8"))
            if faiss is None: 
                raise ImportError("faiss-cpu is required")
            d = int(self.X.shape[1])
            index_path = os.path.join(self.cache_dir, f"faiss_{index_cfg.type}.index")
            if os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
            else:
                self.faiss_index = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
                self.faiss_index.add(self.X)
        # Override runtime search params
        Q = self._embed(query_texts)
        try:
            if hasattr(self.faiss_index, 'nprobe') and index_cfg:
                self.faiss_index.nprobe = index_cfg.nprobe
            if hasattr(self.faiss_index, 'hnsw') and index_cfg and hasattr(self.faiss_index, 'hnsw'):
                self.faiss_index.hnsw.efSearch = index_cfg.hnsw_efSearch
        except Exception as e:
            LOGGER.warning('caught_exception', error=str(e), exc_info=True)
        sims, idxs = self.faiss_index.search(Q, topk)
        return sims, idxs



    def get_patient_chunk_indices(self, unit_id: str) -> List[int]:
        if self.unit_to_chunk_idxs:
            return self.unit_to_chunk_idxs.get(unit_id, [])
        if not self.chunk_meta:
            Mp = os.path.join(self.cache_dir,"chunk_meta.json")
            if os.path.exists(Mp):
                self.chunk_meta = json.load(open(Mp,"r",encoding="utf-8"))
        return [i for i,m in enumerate(self.chunk_meta) if m.get("unit_id")==unit_id]



class RAGRetriever:
    _RR_CACHE_MAX = 200000
    def __init__(self, store: EmbeddingStore, models: Models, cfg: RAGConfig, label_configs: Optional[dict]=None, notes_by_doc: Optional[Dict[str,str]]=None, repo: Optional[DataRepository]=None):
        self.store = store; self.models = models; self.cfg = cfg
        self.label_configs = label_configs or {}
        self._notes_by_doc = notes_by_doc or {}
        self._repo = repo
        self._rr_cache = LRUCache(maxsize=self._RR_CACHE_MAX)


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

    def _build_query(self, label_id: str, label_rules: Optional[str], rationales: Optional[List[dict]]) -> str:
        base = f"Evidence relevant to patient-level label '{label_id}'. "
        if label_rules and isinstance(label_rules,str) and label_rules.strip():
            base += "Guidelines: " + re.sub(r"\s+"," ",label_rules.strip()) + " "
        if rationales:
            snips = [sp.get("snippet","") for sp in rationales if isinstance(sp,dict) and sp.get("snippet")]
            if snips:
                base += "Known evidence snippets: " + " | ".join(snips[:6]) + " "
        return base.strip()

    def _get_lblcfg(self, label_id: str) -> dict:
        return self.label_configs.get(label_id, {})

    def _keyword_hits_for_patient(self, unit_id: str, keywords: List[str]) -> List[dict]:
        if not keywords: return []
        idxs = self.store.get_patient_chunk_indices(unit_id)
        if not idxs: return []
        pats = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in keywords if isinstance(k,str) and k.strip()]
        out = []
        for ix in idxs:
            txt = self.store.chunk_meta[ix]["text"]
            score = sum(len(p.findall(txt)) for p in pats)
            if score > 0:
                m = self.store.chunk_meta[ix]
                out.append({"doc_id": m["doc_id"], "chunk_id": m["chunk_id"], "text": txt, "score": float(score), "source": "keyword"})
        out.sort(key=lambda d: d["score"], reverse=True)
        return out[: self.cfg.keyword_topk]

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

    def retrieve_for_patient_label(self, unit_id: str, label_id: str, label_rules: Optional[str], topk_override: Optional[int]=None) -> List[dict]:
        lblcfg = self._get_lblcfg(label_id)
        rationales = self._repo.get_prior_rationales(unit_id, label_id) if self._repo is not None else []
        query = self._build_query(label_id, label_rules, rationales=rationales)
        sims, idxs = self.store.search([query], topk=max(self.cfg.mmr_candidates, 50))
        idxs = idxs[0].tolist(); sims = sims[0].tolist()
        cands = [(ix, sc) for ix, sc in zip(idxs, sims) if ix >= 0 and self.store.chunk_meta[ix]["unit_id"] == unit_id]
        kw = lblcfg.get("keywords", []) if self.cfg.use_keywords else []
        items = []
        if cands:
            cands = sorted(cands, key=lambda t: t[1], reverse=True)
            base = [ix for ix,_ in cands[: self.cfg.mmr_candidates]]
            if self.cfg.use_mmr:
                q_emb = self.store._embed([query])[0]
                sel = self._mmr_select(q_emb, base, k=(topk_override or self.cfg.per_label_topk)*3, lam=self.cfg.mmr_lambda)
            else:
                sel = base[: (topk_override or self.cfg.per_label_topk)*3]
            for ix in sel:
                m = self.store.chunk_meta[ix]
                items.append({"doc_id": m["doc_id"], "chunk_id": m["chunk_id"], "text": m["text"], "score": 0.0, "source": "mmr"})
        items += self._keyword_hits_for_patient(unit_id, kw)
        # rationale boost
        rat_snips = [sp.get("snippet","") for sp in (rationales or []) if isinstance(sp,dict) and sp.get("snippet")]
        pats = [re.compile(re.escape(sn), re.I) for sn in rat_snips if sn]
        for it in items:
            bonus = 0.0
            for p in pats:
                if p.search(it.get("text","") or ""): bonus += self.cfg.rationale_boost
            it["score"] = it.get("score",0.0) + bonus
        return self._dedup_rerank(query, items, final_topk=(topk_override or self.cfg.per_label_topk))

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


# ------------------------------
# LLM annotator (JSON call) + forced-choice micro-probe
# ------------------------------


class LLMAnnotator:
    def __init__(self, cfg: LLMConfig, cache_dir: str):
        import threading
        self.cfg = cfg
        self.cache_dir = os.path.join(cache_dir,"llm_cache"); os.makedirs(self.cache_dir, exist_ok=True)
        self.client = None
        self.calls_made = 0
        # Budget guards
        self._budget_lock = threading.Lock()
        # semaphore capacity = budget; each API call consumes one
        self._budget_sem = threading.BoundedSemaphore(value=int(self.cfg.budget_max_calls))
        if self.cfg.enable:
            self._init_client()


    def _init_client(self):
        if AzureOpenAI is None: raise ImportError("Please install openai>=1.0 for Azure")
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION","2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=self.cfg.timeout
        )

    def _cache_key(self, payload: dict) -> str:
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, s))

    def _load(self, key: str):
        p = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(p): return json.load(open(p,"r",encoding="utf-8"))
        return None

    def _save(self, key: str, data: dict):
        p = os.path.join(self.cache_dir, f"{key}.json")
        json.dump(data, open(p,"w",encoding="utf-8"), ensure_ascii=False)

    @staticmethod
    def _avg_token_logprob(resp) -> Optional[float]:
        try:
            choice = resp.choices[0]
            lp = getattr(choice, "logprobs", None)
            items = getattr(lp, "content", None) or []
            vals = []
            for it in items:
                v = getattr(it, "logprob", None)
                if v is None and isinstance(it, dict): v = it.get("logprob")
                if v is not None: vals.append(float(v))
            return float(sum(vals)/len(vals)) if vals else None
        except Exception:
            return None

    def _field_avg_logprob(self, resp, content_str: str, field: str = "prediction") -> Optional[float]:
        """Average logprob over the tokens forming the value of a JSON field."""
        try:
            lp = getattr(resp.choices[0], "logprobs", None)
            items = getattr(lp, "content", None) or []
            if not items or not content_str:
                return None
            m = re.search(r'"%s"\s*:\s*(?P<val>"[^"]*"|\S+)' % re.escape(field), content_str)
            if not m:
                return None
            span = m.span("val")
            pos = 0; s = 0.0; n = 0
            for it in items:
                tok = getattr(it, "token", None) or (it.get("token") if isinstance(it, dict) else None)
                lpv = getattr(it, "logprob", None) or (it.get("logprob") if isinstance(it, dict) else None)
                if tok is None or lpv is None:
                    continue
                tok_start = pos
                pos += len(tok)
                tok_end = pos
                if tok_end <= span[0] or tok_start >= span[1]:
                    continue
                s += float(lpv); n += 1
            return (s / n) if n else None
        except Exception:
            return None

    def annotate(self, unit_id: str, label_id: str, label_type: str, label_rules: str, snippets: List[dict], n_consistency: Optional[int]=None) -> dict:
        if not self.cfg.enable: return {}
        if n_consistency is None: n_consistency = self.cfg.n_consistency
        # Build short context
        ctx, used = [], 0
        for s in snippets:
            frag = f"[doc_id={s.get('doc_id')}, chunk_id={s.get('chunk_id')}] " + (s.get("text","") or "")
            if used + len(frag) > max(1000, self.cfg.max_context_chars): break
            ctx.append(frag); used += len(frag)

        system = ("You are a meticulous clinical annotator for EHR data. "
                  "Follow the label rules precisely. Return strict JSON only.")
        task = (
            f"Label: '{label_id}' (type: {label_type}). "
            "Use the evidence snippets from this patient's notes. "
            "If insufficient evidence, reply with 'unknown' or 'absent' per the rules.\n\n"
            f"Guidelines:\n{label_rules}\n\n"
            "Evidence snippets:\n" + "\n\n".join(ctx) + "\n\n"
            "RESPONSE JSON keys: version, unit_id, label_id, label_type, prediction, confidence, evidence, reasoning"
        )
        messages = [{"role":"system","content":system}, {"role":"user","content":task}]

        payload = {"model": self.cfg.model_name, "messages": messages, "temperature": self.cfg.temperature, "n_consistency": n_consistency}
        key = self._cache_key(payload); cached = self._load(key)
        if cached is not None: return cached
        if self.calls_made + n_consistency > self.cfg.budget_max_calls: return {}

        preds, confs, lps, runs = [], [], [], []
        for i in range(n_consistency):
            attempt = 0
            while attempt <= self.cfg.retry_max:
                try:
                    
                    # budget accounting: acquire 1 slot for this call
                    if not self._budget_sem.acquire(blocking=False):
                        raise RuntimeError("budget_exceeded")
                    with self._budget_lock:
                        self.calls_made += 1
                    kwargs = dict(

                        model=self.cfg.model_name,
                        temperature=self.cfg.temperature,
                        response_format={"type":"json_object"},
                        logprobs=self.cfg.logprobs,
                        n=1,
                        seed=self.cfg.seed + i,
                        messages=messages
                    )
                    if self.cfg.logprobs and int(self.cfg.top_logprobs) > 0:
                        kwargs["top_logprobs"] = int(self.cfg.top_logprobs)
                    resp = self.client.chat.completions.create(**kwargs)
                    content = resp.choices[0].message.content
                    data = json.loads(content)
                    pred = data.get(self.cfg.prediction_field, data.get("prediction"))
                    conf = data.get("confidence")
                    if isinstance(conf, str):
                        try: conf = float(conf)
                        except: conf = None
                    avg_lp = self._avg_token_logprob(resp)
                    pred_lp = self._field_avg_logprob(resp, content, field=self.cfg.prediction_field)
                    preds.append(str(pred) if pred is not None else None)
                    confs.append(conf if isinstance(conf,(int,float)) else None)
                    if avg_lp is not None: lps.append(float(avg_lp))
                    runs.append({"prediction":pred, "confidence":conf, "avg_token_logprob":avg_lp, "prediction_logprob": pred_lp, "raw":data})
                    break
                except Exception as e:
                    if attempt >= self.cfg.retry_max:
                        runs.append({"error": str(e)}); break
                    time.sleep(self.cfg.retry_backoff * (attempt+1)); attempt += 1

        pred_final, conf_final, agree = None, None, 0.0
        if preds:
            cnt = Counter([p for p in preds if p is not None])
            if cnt:
                pred_final = cnt.most_common(1)[0][0]
                agree = cnt.most_common(1)[0][1] / len(preds)
        if confs:
            cs = [c for c in confs if c is not None]
            if cs: conf_final = float(sum(cs)/len(cs))
        avg_lp_final = float(sum(lps)/len(lps)) if lps else None
        pred_lps = [r.get('prediction_logprob') for r in runs if r.get('prediction_logprob') is not None]
        pred_lp_final = float(sum(pred_lps)/len(pred_lps)) if pred_lps else None

        out = {"version": self.cfg.json_version, "unit_id": unit_id, "label_id": label_id, "label_type": label_type,
               "prediction": pred_final, "confidence": conf_final,
               "consistency_agreement": float(agree), "avg_token_logprob": avg_lp_final,
               "prediction_logprob": pred_lp_final, "runs": runs}
        self._save(key, out); return out

    @staticmethod
    def _norm_token(tok: str) -> str:
        if tok is None: return ""
        t = str(tok).strip().strip('"').strip("'").strip()
        if t.startswith(" "): t = t[1:]
        return t

    
    def forced_choice(self, unit_id: str, label_id: str, label_type: str, options: List[str], label_rules: str, snippets: List[dict], *, temperature: float=0.0, top_logprobs: int=15, max_tokens: int=1) -> dict:
            """Single-token forced-choice; returns {'letter','choice','probs','entropy','margin'}.
            Uses Azure Chat Completions logprobs; robust to missing fields.
            """
            try:
                letters = [chr(ord('A')+i) for i in range(len(options))]
                mapping_lines = [f"{letters[i]} = {options[i]}" for i in range(len(options))]
                # short context
                ctx, used = [], 0
                for s in snippets:
                    frag = f"[doc_id={s.get('doc_id')}, chunk_id={s.get('chunk_id')}] " + (s.get("text","") or "")
                    if used + len(frag) > max(800, self.cfg.max_context_chars//2): break
                    ctx.append(frag); used += len(frag)
                system = ("You are a precise clinical annotator. You MUST respond with EXACTLY ONE LETTER from the allowed set; no other text.")
                user = (
                    f"Label: '{label_id}' (type: {label_type}). Choose the best option based on the evidence and rules.\n"
                    f"Guidelines:\n{label_rules}\n\nEvidence:\n" + "\n\n".join(ctx) + "\n\n"
                    + "Allowed outputs: " + ", ".join(letters) + "\n"
                    + "Choices:\n" + "\n".join(mapping_lines) + "\n\n"
                    + "Output strictly one letter (e.g., A)."
                )
                messages = [{"role":"system","content":system},{"role":"user","content":user}]
                # Budget tokens (never released): lifetime cap, not concurrency control
                if not self._budget_sem.acquire(blocking=False):
                    return {"error":"budget_exceeded"}
                with self._budget_lock:
                    self.calls_made += 1
                tlp = max(int(top_logprobs or 0), len(options))
                kwargs = dict(
                    model=self.cfg.model_name,
                    temperature=temperature,
                    logprobs=True,
                    n=1,
                    seed=self.cfg.seed + 999,
                    max_tokens=max_tokens,
                    messages=messages
                )
                if int(tlp) > 0:
                    kwargs["top_logprobs"] = int(tlp)
                resp = self.client.chat.completions.create(**kwargs)
                ch = resp.choices[0]
                lp_block = getattr(ch, "logprobs", None)
                items = getattr(lp_block, "content", None) or []
                # Accumulate best logprob per allowed letter
                import numpy as np
                letter_logps = {L: -1e9 for L in letters}
                for it in items:
                    tops = getattr(it, "top_logprobs", None) or (it.get("top_logprobs") if isinstance(it, dict) else None)
                    if not tops: 
                        continue
                    for cand in tops:
                        tok = getattr(cand, "token", None) or (cand.get("token") if isinstance(cand, dict) else None)
                        lp  = getattr(cand, "logprob", None) or (cand.get("logprob") if isinstance(cand, dict) else None)
                        if tok is None or lp is None:
                            continue
                        t = str(tok).strip().strip('"').strip("'")
                        if t and not t[0].isalnum(): 
                            t = t[1:]
                        t = t[:1].upper() if t else ""
                        if t in letter_logps:
                            letter_logps[t] = max(letter_logps[t], float(lp))
                vals = np.array([letter_logps[L] for L in letters], dtype=float)
                if not np.isfinite(vals).any():
                    return {"error":"no_logprobs"}
                probs = np.exp(vals - np.max(vals)); probs = probs / (probs.sum() + 1e-12)
                best = int(np.argmax(probs))
                second = np.partition(probs, -2)[-2] if len(probs) >= 2 else 0.0
                margin = float(probs[best] - second)
                entropy = float(-(probs * np.log(np.clip(probs,1e-12,None))).sum())
                probs_map = {options[i]: float(probs[i]) for i in range(len(options))}
                return {"letter": letters[best], "choice": options[best], "probs": probs_map, "entropy": entropy, "margin": margin}
            except Exception as e:
                return {"error": str(e)}



    
def annotate_many(self, items: List[dict], max_workers: Optional[int]=None) -> List[dict]:
        """items: list of {unit_id,label_id,label_type,label_rules,snippets,n_consistency}"""
        max_workers = max_workers or self.cfg.max_parallel or 4
        n_cons = max(1, self.cfg.n_consistency)
        # compute allowed items based on remaining budget
        with self._budget_lock:
            remaining = max(0, self.cfg.budget_max_calls - self.calls_made)
        capacity = remaining // n_cons
        if capacity <= 0:
            return [{} for _ in items[:0]]
        if len(items) > capacity:
            items = items[:capacity]
        results = [None]*len(items)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _call(i, it):
            try:
                return i, self.annotate(it['unit_id'], it['label_id'], it['label_type'], it.get('label_rules',''), it.get('snippets',[]), it.get('n_consistency'))
            except Exception as e:
                return i, {"error": str(e)}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
            futs = [ex.submit(_call, i, it) for i,it in enumerate(items)]
            for f in as_completed(futs):
                i, res = f.result()
                results[i] = res
        return results



# ------------------------------
# Label-aware pooling with prototypes
# ------------------------------

class LabelAwarePooler:
    BETA_DEFAULT = 5.0

    def __init__(self, repo: DataRepository, store: EmbeddingStore, models: Models, beta: float = None, kmeans_k: int = 8, persist_dir: str = None, version: str = "v1", use_negative: bool = False):
        self.repo = repo; self.store = store; self.models = models
        self.beta = float(beta) if beta is not None else self.BETA_DEFAULT
        self.kmeans_k = int(kmeans_k)
        self.persist_dir = persist_dir
        self.version = version
        self.use_negative = use_negative
        self.prototypes: Dict[str,np.ndarray] = {}
        self.neg_prototypes: Dict[str,np.ndarray] = {}
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
        self._cache_vec: Dict[Tuple[str,str], np.ndarray] = {}

    def _save_bank(self, label: str, arr: np.ndarray, kind: str = 'pos'):
        if not self.persist_dir: return
        path = os.path.join(self.persist_dir, f"prot_{label}_{kind}_{self.version}.npy")
        np.save(path, arr)

    def _load_bank(self, label: str, kind: str = 'pos') -> Optional[np.ndarray]:
        if not self.persist_dir: return None
        path = os.path.join(self.persist_dir, f"prot_{label}_{kind}_{self.version}.npy")
        if os.path.exists(path):
            try: return np.load(path)
            except: return None
        return None

    def _kmeans_medoids(self, E: np.ndarray, k: int) -> np.ndarray:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(k, max(1, len(E))), n_init=5, random_state=42)
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
        label_to_snips_pos = defaultdict(list)
        # Only incorporate explicit negative rationales if required; default is off for safety
        label_to_snips_neg = defaultdict(list)

        if "rationales_json" in self.repo.ann.columns:
            for r in self.repo.ann.itertuples(index=False):
                if isinstance(r.rationales_json, list):
                    for sp in r.rationales_json:
                        if isinstance(sp, dict) and sp.get("snippet"):
                            val = str(getattr(r, "label_value", "")).strip().lower()
                            is_pos = val in {"1","true","present","yes","pos","positive"}
                            polarity = str(sp.get('polarity','')).lower() if isinstance(sp, dict) else ''
                            if is_pos:
                                label_to_snips_pos[r.label_id].append(sp["snippet"])
                            elif self.use_negative and polarity in {'neg','negative','absent'}:
                                label_to_snips_neg[r.label_id].append(sp["snippet"])

        for lid in set(list(label_to_snips_pos.keys()) + list(label_to_snips_neg.keys())):
            P_load = self._load_bank(lid, 'pos')
            N_load = self._load_bank(lid, 'neg') if self.use_negative else None

            texts_pos = [s for s in label_to_snips_pos.get(lid, []) if isinstance(s,str) and s.strip()]
            texts_neg = [s for s in label_to_snips_neg.get(lid, []) if isinstance(s,str) and s.strip()] if self.use_negative else []

            if texts_pos:
                E = self.store._embed(texts_pos)
                E = E / (np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
                E = self._kmeans_medoids(E, self.kmeans_k)
                self.prototypes[lid] = E; self._save_bank(lid, E, 'pos')
            elif P_load is not None:
                self.prototypes[lid] = P_load

            if self.use_negative and texts_neg:
                EN = self.store._embed(texts_neg)
                EN = EN / (np.linalg.norm(EN,axis=1,keepdims=True)+1e-12)
                EN = self._kmeans_medoids(EN, max(2, self.kmeans_k//2))
                self.neg_prototypes[lid] = EN; self._save_bank(lid, EN, 'neg')
            elif self.use_negative and N_load is not None:
                self.neg_prototypes[lid] = N_load

    def pooled_vector(self, unit_id: str, label_id: str, retriever: RAGRetriever, label_rules: str, topk: int=6) -> np.ndarray:
        key = (unit_id, label_id)
        if key in self._cache_vec: return self._cache_vec[key]
        ctx = retriever.retrieve_for_patient_label(unit_id, label_id, label_rules, topk_override=topk)
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
        PN = self.neg_prototypes.get(label_id)
        if PN is not None and PN.size:
            w_raw = w_raw - (V @ PN.T).max(axis=1)
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

def _options_for_label(label_id: str, label_type: str, label_cfgs: dict) -> Optional[List[str]]:
    cfg = label_cfgs.get(label_id, {}) if isinstance(label_cfgs, dict) else {}
    opts = cfg.get("options")
    if isinstance(opts, list) and 2 <= len(opts) <= 26:
        return [str(o) for o in opts]
    lt = (label_type or "text").lower()
    if lt == "binary":
        return ["present","absent","unknown"]
    return None

class LLMProber:
    def __init__(self, cfg: LLMFirstConfig, llm: LLMAnnotator, retriever: RAGRetriever, repo: DataRepository):
        self.cfg = cfg; self.llm = llm; self.retriever = retriever; self.repo = repo

    def probe_unseen(self, unseen_pairs: List[Tuple[str,str]], label_types: Dict[str,str], per_label_rules: Dict[str,str]) -> pd.DataFrame:
        by_label = defaultdict(list)
        for uid, lid in unseen_pairs:
            by_label[lid].append(uid)

        records = []
        for lid, uids in by_label.items():
            sample = random.sample(uids, min(len(uids), self.cfg.per_label_probe_count))
            # Prebuild contexts
            tasks = []
            for uid in sample:
                rules = per_label_rules.get(lid,"")
                ctx = self.retriever.retrieve_for_patient_label(uid, lid, rules, topk_override=self.cfg.stage_a_topk)
                tasks.append({"unit_id": uid, "label_id": lid, "label_type": label_types.get(lid,"text"),
                              "label_rules": rules, "snippets": ctx, "n_consistency": self.llm.cfg.n_consistency,
                              "_ctx_hash": stable_hash_str("\n".join([c.get("text","") for c in ctx]))})
            # Parallel JSON annotate
            results = self.llm.annotate_many(tasks, max_workers=self.llm.cfg.max_parallel)
            # Forced-choice per item (option-level)
            for it, res in zip(tasks, results):
                avg_lp = res.get("avg_token_logprob"); cons = res.get("consistency_agreement",0.0) or 0.0
                pred_lp = res.get("prediction_logprob")
                # Option set for FC
                fc = None
                fc_opts = _options_for_label(it['label_id'], it['label_type'], self.retriever.label_configs)
                if self.cfg.fc_enable and fc_opts is not None:
                    try:
                        fc = self.llm.forced_choice(it['unit_id'], it['label_id'], it['label_type'], fc_opts, it.get('label_rules',''), it.get('snippets',[]),
                                                    temperature=self.cfg.fc_temperature, top_logprobs=self.cfg.fc_top_logprobs, max_tokens=self.cfg.fc_max_tokens)
                    except Exception as _e:
                        fc = {"error": str(_e)}
                rec = {"unit_id": it["unit_id"], "label_id": it["label_id"], "label_type": it["label_type"],
                       "pred": res.get("prediction"),
                       "avg_token_logprob": avg_lp, "prediction_logprob": pred_lp,
                       "consistency": float(cons),
                       "ctx_hash": int(it["_ctx_hash"]) }
                if isinstance(fc, dict) and 'error' not in fc:
                    rec.update({"fc_choice": fc.get("choice"), "fc_entropy": fc.get("entropy"), "fc_margin": fc.get("margin"), "fc_probs": fc.get("probs")})
                else:
                    rec.update({"fc_choice": None, "fc_entropy": None, "fc_margin": None})
                records.append(rec)

        
        df = pd.DataFrame(records)
        if df.empty: return df
        # Calibrate U with forced-choice entropy (primary) + 1 - consistency (secondary)
        frames = []
        for lid, grp in df.groupby("label_id"):
            grp = grp.copy()
            def _z(a):
                m, s = float(np.nanmean(a)), float(np.nanstd(a)+1e-9)
                return (a - m) / s
            ent = grp.get("fc_entropy") if "fc_entropy" in grp.columns else None
            if ent is not None and not ent.isnull().all():
                z_ent = _z(ent.fillna(ent.mean()).to_numpy())
                u_cons = 1.0 - grp["consistency"].fillna(0.0).to_numpy()
                try:
                    z_cons = _z(u_cons)
                except Exception:
                    z_cons = normalize01(u_cons)
                grp["U"] = float(self.cfg.fc_w_entropy) * z_ent + float(self.cfg.fc_w_disagree) * z_cons
            else:
                # fallback to prior heuristic if FC unavailable
                u_lp = np.array([math.exp(-(x if x is not None else 0.0)) for x in grp.get("prediction_logprob", pd.Series([None]*len(grp))).tolist()], dtype=float)
                try:
                    z_lp = _z(u_lp)
                except Exception:
                    z_lp = normalize01(u_lp)
                u_cons = 1.0 - grp["consistency"].fillna(0.0).to_numpy()
                try:
                    z_cons = _z(u_cons)
                except Exception:
                    z_cons = normalize01(u_cons)
                grp["U"] = 0.6*z_lp + 0.4*z_cons
            frames.append(grp)
        df = pd.concat(frames, ignore_index=True)

        # Slice uncertain/certain per label
        uncertain_top_pct = self.cfg.uncertain_top_pct
        certain_bottom_pct = self.cfg.certain_bottom_pct
        u_rows, c_rows = [], []
        for lid, grp in df.groupby("label_id"):
            if grp.empty: continue
            U = grp["U"].to_numpy()
            lo = np.quantile(U, certain_bottom_pct); hi = np.quantile(U, 1.0 - uncertain_top_pct)
            u_rows.append(grp[grp["U"] >= hi]); c_rows.append(grp[grp["U"] <= lo])
        seeds_unc = pd.concat(u_rows, ignore_index=True) if u_rows else df.head(0)
        seeds_cer = pd.concat(c_rows, ignore_index=True) if c_rows else df.head(0)
        return seeds_unc, seeds_cer

# ------------------------------

class DisagreementExpander:
    def __init__(self, cfg: DisagreementConfig, repo: DataRepository, retriever: RAGRetriever):
        self.cfg = cfg; self.repo = repo; self.retriever = retriever

    def high_entropy_seeds(self) -> pd.DataFrame:
        dis = self.repo.reviewer_disagreement(round_policy=self.cfg.round_policy, decay_half_life=self.cfg.decay_half_life, uncertainty_note_boost=self.cfg.uncertainty_note_boost)
        seeds = dis[dis["disagreement_score"] >= self.cfg.high_entropy_threshold].copy()
        seeds = seeds.sort_values("disagreement_score", ascending=False)
        rows = []
        for lid, grp in seeds.groupby("label_id"):
            rows.append(grp.head(self.cfg.seeds_per_label))
        return pd.concat(rows, ignore_index=True) if rows else seeds.head(0)

    def seed_snippets(self, unit_id: str, label_id: str, label_rules: str) -> List[str]:
        spans = self.repo.get_prior_rationales(unit_id, label_id)
        snips = [sp.get("snippet") for sp in spans if isinstance(sp,dict) and sp.get("snippet")]
        if snips: return snips[: self.cfg.snippets_per_seed]
        ctx = self.retriever.retrieve_for_patient_label(unit_id, label_id, label_rules, topk_override=self.cfg.snippets_per_seed)
        return [c["text"] for c in ctx if isinstance(c.get("text"), str)]

    def expand(self, rules_map: Dict[str,str], seen_pairs: set) -> pd.DataFrame:
        seeds = self.high_entropy_seeds()
        rows = []
        for lid, grp in seeds.groupby("label_id"):
            snips = []
            for r in grp.itertuples(index=False):
                snips.extend(self.seed_snippets(r.unit_id, lid, rules_map.get(lid,"")))
            snips = snips[: self.cfg.seeds_per_label * self.cfg.snippets_per_seed]
            if not snips: continue
            cand = self.retriever.expand_from_snippets(lid, snips, seen_pairs, per_seed_k=self.cfg.similar_chunks_per_seed)
            items = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.expanded_per_label]
            for uid, sc in items:
                rows.append({"unit_id": uid, "label_id": lid, "score": float(sc), "bucket": "disagreement_expanded"})
        df = pd.DataFrame(rows)
        if df.empty: return df
        out = []
        for lid, g in df.groupby("label_id"):
            arr = g["score"].to_numpy()
            g = g.copy(); g["score_n"] = normalize01(arr); out.append(g)
        return pd.concat(out, ignore_index=True)


# ------------------------------
# Cost-aware ranking
# ------------------------------

def estimate_minutes_for_unit(repo: DataRepository, unit_id: str, cfg: CostConfig) -> float:
    notes = repo.notes[repo.notes['unit_id']==unit_id]
    n_notes = len(notes)
    n_chars = int(notes['text'].str.len().sum())
    return float(cfg.base_min + cfg.per_note_min * n_notes + cfg.per_1k_chars_min * (n_chars/1000.0))

def add_ig_over_min(df: pd.DataFrame, bucket: str, repo: DataRepository, cost_cfg: CostConfig) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    if bucket == 'model_uncertain_llm':
        w = cost_cfg.weight_uncertain
        sig = df.get('U', None)
        if sig is None:
            sig = df.get('score_n', None)
        df['_IG'] = (sig if sig is not None else 1.0) * w
    elif bucket == 'easy_qc_llm':
        df['_IG'] = cost_cfg.weight_certain
    elif bucket == 'disagreement_expanded':
        df['_IG'] = df.get('score_n', pd.Series([0.5]*len(df))) * cost_cfg.weight_disagree
    else:  # diversity
        df['_IG'] = cost_cfg.weight_diversity
    mins = [estimate_minutes_for_unit(repo, u, cost_cfg) for u in df['unit_id']]
    df['_min'] = mins
    df['_IG_per_min'] = df['_IG'] / np.maximum(1e-6, df['_min'])
    return df

def cost_aware_trim(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    if df.empty: return df
    return df.sort_values('_IG_per_min', ascending=False).head(target_n).drop(columns=[c for c in ['_IG','_min','_IG_per_min'] if c in df.columns])


# ------------------------------
# Selection helpers

def final_global_topoff(selected_df: pd.DataFrame, union_df: pd.DataFrame, batch_size: int,
                        pooler: LabelAwarePooler, retriever: RAGRetriever, rules_map: Dict[str,str],
                        label_types: Dict[str,str], per_label_cap: int) -> pd.DataFrame:
    # selected_df: already chosen across buckets
    # union_df: all candidate rows from buckets (deduped on unit,label)
    selected = selected_df.copy()
    if len(selected) >= batch_size:
        return selected.head(batch_size)
    # Remaining candidates
    rem = union_df.merge(selected[["unit_id","label_id"]], on=["unit_id","label_id"], how="left", indicator=True)
    rem = rem[rem["_merge"]=="left_only"].drop(columns=["_merge"]).copy()
    if rem.empty: 
        return selected
    # Build vectors
    Vrem = []
    meta = []
    for r in rem.itertuples(index=False):
        Vrem.append(pooler.pooled_vector(r.unit_id, r.label_id, retriever, rules_map.get(r.label_id,"")))
        meta.append((r.unit_id, r.label_id))
    Vrem = np.vstack(Vrem) if Vrem else np.zeros((0, retriever.store.X.shape[1]), dtype="float32")
    # Preselected vectors
    Vsel = []
    if not selected.empty:
        for r in selected.itertuples(index=False):
            Vsel.append(pooler.pooled_vector(r.unit_id, r.label_id, retriever, rules_map.get(r.label_id,"")))
    preV = np.vstack(Vsel) if Vsel else None
    need = batch_size - len(selected)
    idxs = kcenter_select(Vrem, k=min(need, len(rem)), preselected=preV)
    # apply per-label cap while walking idxs
    cnt = selected.groupby("label_id").size().to_dict()
    picks = []
    for i in idxs:
        u,l = meta[i]
        if cnt.get(l,0) >= per_label_cap: 
            continue
        row = rem[(rem["unit_id"]==u) & (rem["label_id"]==l)].head(1).copy()
        if row.empty: 
            continue
        row.loc[:, "selection_reason"] = "final_topoff_kcenter"
        picks.append(row)
        cnt[l] = cnt.get(l,0) + 1
        if len(picks) >= need: break
    if picks:
        selected = pd.concat([selected] + picks, ignore_index=True)
    return selected.head(batch_size)

# ------------------------------

def per_label_kcenter(pool_df: pd.DataFrame, pooler: LabelAwarePooler, retriever: RAGRetriever, rules_map: Dict[str,str],
                      per_label_quota: int, already_selected_vecs: Optional[np.ndarray]=None) -> pd.DataFrame:
    out = []
    for lid, grp in pool_df.groupby("label_id"):
        if grp.empty: continue
        k = min(per_label_quota, len(grp))
        vecs = [pooler.pooled_vector(r.unit_id, r.label_id, retriever, rules_map.get(r.label_id,"")) for r in grp.itertuples(index=False)]
        V = np.vstack(vecs)
        idxs = kcenter_select(V, k=k, seed_idx=None, preselected=already_selected_vecs)
        sel = grp.iloc[idxs].copy()
        sel = sel.reset_index(drop=True)
        sel["kcenter_rank_per_label"] = list(range(1, len(sel)+1))
        out.append(sel)
    return pd.concat(out, ignore_index=True) if out else pool_df.head(0)

def merge_with_global_kcenter(per_label_selected: pd.DataFrame, pool_df: pd.DataFrame,
                              pooler: LabelAwarePooler, retriever: RAGRetriever, rules_map: Dict[str,str],
                              target_n: int, per_label_cap: int,
                              already_selected_vecs: Optional[np.ndarray]=None) -> pd.DataFrame:
    if per_label_selected.empty and pool_df.empty: return per_label_selected
    selected = per_label_selected.copy()
    cnt = Counter(selected["label_id"].tolist()) if not selected.empty else Counter()
    remaining_needed = max(0, target_n - len(selected))
    if remaining_needed == 0: return selected
    if not selected.empty:
        rem = pool_df.merge(selected[["unit_id","label_id"]], on=["unit_id","label_id"], how="left", indicator=True)
        rem = rem[rem["_merge"]=="left_only"].drop(columns=["_merge"])
    else:
        rem = pool_df.copy()
    if rem.empty: return selected

    vecs = [pooler.pooled_vector(r.unit_id, r.label_id, retriever, rules_map.get(r.label_id,"")) for r in rem.itertuples(index=False)]
    V = np.vstack(vecs)

    preV = None
    if already_selected_vecs is not None and already_selected_vecs.size:
        preV = already_selected_vecs
    if not selected.empty:
        Vsel = [pooler.pooled_vector(r.unit_id, r.label_id, retriever, rules_map.get(r.label_id,"")) for r in selected.itertuples(index=False)]
        Vsel = np.vstack(Vsel); preV = Vsel if preV is None else np.vstack([preV, Vsel])

    idx_order = kcenter_select(V, k=min(remaining_needed, len(rem)), seed_idx=None, preselected=preV)

    chosen_rows = []
    for rank_idx, i in enumerate(idx_order, start=1):
        row = rem.iloc[i]
        if cnt[row.label_id] >= per_label_cap:
            continue
        row = row.copy()
        row["kcenter_rank_global"] = rank_idx
        chosen_rows.append(row); cnt[row.label_id] += 1
        if len(chosen_rows) >= remaining_needed: break

    if not chosen_rows: return selected
    chosen_df = pd.DataFrame([r._asdict() if hasattr(r,"_asdict") else dict(r) for r in chosen_rows])
    return pd.concat([selected, chosen_df], ignore_index=True)


def build_diversity_bucket(unseen_pairs: List[Tuple[str,str]], already_selected: List[Tuple[str,str]],
                           n_div: int, per_label_cap: int,
                           pooler: LabelAwarePooler, retriever: RAGRetriever, rules_map: Dict[str,str],
                           label_types: Dict[str,str], sample_cap: int = 2000) -> pd.DataFrame:
    remaining = [(u,l) for (u,l) in unseen_pairs if (u,l) not in set(already_selected)]
    random.shuffle(remaining)
    remaining = remaining[: max(n_div*4, min(sample_cap, len(remaining)))]
    rem_df = pd.DataFrame([{"unit_id": u, "label_id": l, "label_type": label_types.get(l,"text")} for (u,l) in remaining])
    if rem_df.empty or n_div <= 0: return rem_df.head(0)

    # prototype-free vs prototype-based split
    proto_free = int(max(0, round(n_div * 0.25)))
    proto_based = n_div - proto_free

    # Per-label pre-pick (prototype-based)
    per_label_quota = max(1, int(0.25 * max(1, proto_based)))
    per_sel = per_label_kcenter(rem_df, pooler, retriever, rules_map, per_label_quota)

    # Global prototype-based k-center
    preV = []
    for (u,l) in already_selected:
        preV.append(pooler.pooled_vector(u, l, retriever, rules_map.get(l,"")))
    preV = np.vstack(preV) if preV else None
    merged = merge_with_global_kcenter(per_sel, rem_df, pooler, retriever, rules_map, target_n=proto_based, per_label_cap=per_label_cap, already_selected_vecs=preV)
    merged = merged.copy(); merged["selection_reason"] = "diversity_kcenter"

    if proto_free > 0:
        remainder_pairs = [(r.unit_id, r.label_id) for r in rem_df.itertuples(index=False) if (r.unit_id, r.label_id) not in set(zip(merged.unit_id, merged.label_id))]
        if remainder_pairs:
            vecs = []
            idx_map = []
            for (u,l) in remainder_pairs:
                idxs = retriever.store.get_patient_chunk_indices(u)
                if not idxs: continue
                V = retriever.store.X[idxs]
                vecs.append(V.mean(axis=0))
                idx_map.append((u,l))
            if vecs:
                V = np.vstack(vecs)
                pre = []
                for (u,l) in already_selected:
                    idxs = retriever.store.get_patient_chunk_indices(u)
                    if idxs:
                        pre.append(retriever.store.X[idxs].mean(axis=0))
                for r in merged.itertuples(index=False):
                    idxs = retriever.store.get_patient_chunk_indices(r.unit_id)
                    if idxs:
                        pre.append(retriever.store.X[idxs].mean(axis=0))
                pre = np.vstack(pre) if pre else None
                idxs = kcenter_select(V, k=min(proto_free, len(V)), preselected=pre)
                picks = [idx_map[i] for i in idxs]
                add_df = pd.DataFrame([{"unit_id": u, "label_id": l, "label_type": label_types.get(l,"text"), "selection_reason": "diversity_kcenter_protofree"} for (u,l) in picks])
                merged = pd.concat([merged, add_df], ignore_index=True)

    return merged


# ------------------------------
# Orchestrator

def _detect_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'

# ------------------------------

class ActiveLearningLLMFirst:
    def __init__(self, paths: Paths, cfg: OrchestratorConfig, label_config: Optional[dict]=None):
        self.paths = paths; self.cfg = cfg; set_all_seeds(cfg.select.random_seed)
        notes_df = read_table(paths.notes_path); ann_df = read_table(paths.annotations_path)
        self.repo = DataRepository(notes_df, ann_df)

        embed_name = os.getenv("MED_EMBED_MODEL_NAME","pritamdeka/S-Bluebert-snli_MNLI")
        rerank_name = os.getenv("RERANKER_MODEL_NAME","cross-encoder/ms-marco-MiniLM-L-6-v2")
        device = _detect_device()
        embedder = SentenceTransformer(embed_name, device=device)
        reranker = CrossEncoder(rerank_name, device=device)
        emb_bs = int(os.getenv('EMB_BATCH', '64'))
        rr_bs = int(os.getenv('RERANK_BATCH', '64'))
        self.models = Models(embedder, reranker, device=device, emb_batch=emb_bs, rerank_batch=rr_bs)

        self.store = EmbeddingStore(self.models, cache_dir=self.paths.cache_dir, normalize=self.cfg.rag.normalize_embeddings)
        self.rag = RAGRetriever(self.store, self.models, self.cfg.rag, label_configs=label_config or {}, notes_by_doc=self.repo.notes_by_doc(), repo=self.repo)
        self.llm = LLMAnnotator(self.cfg.llm, cache_dir=self.paths.cache_dir)
        self.pooler = LabelAwarePooler(self.repo, self.store, self.models, beta=float(os.getenv('POOLER_BETA', 5.0)), kmeans_k=int(os.getenv('POOLER_K', 8)), persist_dir=os.path.join(self.paths.cache_dir, 'prototypes'), version='v1', use_negative=bool(int(os.getenv('POOLER_USE_NEG', '0'))))
        self.label_config = label_config or {}

    def build_unseen_pairs(self) -> List[Tuple[str,str]]:
        seen = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        all_units = sorted(self.repo.notes["unit_id"].unique().tolist())
        all_labels = sorted(self.repo.ann["label_id"].unique().tolist())
        pairs = [(u,l) for u in all_units for l in all_labels if (u,l) not in seen]
        return pairs

    # ---- Buckets ----

    def build_disagreement_bucket(self, seen_pairs: set, rules_map: Dict[str,str], label_types: Dict[str,str]) -> pd.DataFrame:
        expander = DisagreementExpander(self.cfg.disagree, self.repo, self.rag)
        expanded = expander.expand(rules_map, seen_pairs)
        if expanded.empty: return expanded
        n_dis = int(self.cfg.select.batch_size * self.cfg.select.pct_disagreement)
        per_label_quota = max(1, int(self.cfg.disagree.per_label_cap_fraction * max(1, n_dis)))
        per_sel = per_label_kcenter(expanded, self.pooler, self.rag, rules_map, per_label_quota)
        per_label_cap = per_label_quota
        sel = merge_with_global_kcenter(per_sel, expanded, self.pooler, self.rag, rules_map, target_n=n_dis, per_label_cap=per_label_cap)
        sel = sel.copy(); sel["label_type"] = sel["label_id"].map(lambda x: label_types.get(x,"text"))
        sel["selection_reason"] = "disagreement_expanded"
        if self.cfg.cost.enable:
            sel = add_ig_over_min(sel, 'disagreement_expanded', self.repo, self.cfg.cost)
            sel = cost_aware_trim(sel, n_dis)
        return sel

    def build_llm_uncertain_bucket(self, unseen_pairs: List[Tuple[str,str]], label_types: Dict[str,str], rules_map: Dict[str,str]) -> pd.DataFrame:
        prober = LLMProber(self.cfg.llmfirst, self.llm, self.rag, self.repo)
        probe_df = prober.probe_unseen(unseen_pairs, label_types, rules_map)
        _tmp_probe = _jsonify_cols(probe_df, [c for c in ['fc_probs','rag_context','why'] if c in probe_df.columns])
        _tmp_probe.to_parquet(os.path.join(self.paths.outdir, "llm_probe.parquet"), index=False)
        seeds_unc, seeds_cer = LLMProber.slice_uncertain_certain(probe_df, self.cfg.llmfirst.uncertain_top_pct, self.cfg.llmfirst.certain_bottom_pct)
        seen_pairs_set = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        rows = []
        for lid, grp in seeds_unc.groupby("label_id"):
            snips = []
            for r in grp.itertuples(index=False):
                ctx = self.rag.retrieve_for_patient_label(r.unit_id, r.label_id, rules_map.get(lid,""), topk_override=2)
                for s in ctx[:1]:
                    if isinstance(s.get("text"), str) and s["text"].strip(): snips.append(s["text"][:800])
            cand = self.rag.expand_from_snippets(lid, snips, seen_pairs_set, per_seed_k=100)
            for uid, sc in sorted(cand.items(), key=lambda kv: kv[1], reverse=True):
                rows.append({"unit_id": uid, "label_id": lid, "score": float(sc)})
        expanded_unc = pd.DataFrame(rows)
        if not expanded_unc.empty:
            out = []
            for lid, g in expanded_unc.groupby("label_id"):
                arr = g["score"].to_numpy(); g = g.copy(); g["score_n"] = normalize01(arr); out.append(g)
            expanded_unc = pd.concat(out, ignore_index=True)
        n_unc = int(self.cfg.select.batch_size * self.cfg.select.pct_uncertain)
        per_label_quota = max(1, int(self.cfg.llmfirst.per_label_cap_fraction * max(1, n_unc)))
        per_sel = per_label_kcenter(expanded_unc, self.pooler, self.rag, rules_map, per_label_quota)
        sel_unc = merge_with_global_kcenter(per_sel, expanded_unc, self.pooler, self.rag, rules_map, target_n=n_unc, per_label_cap=per_label_quota)
        if not sel_unc.empty:
            sel_unc["label_type"] = sel_unc["label_id"].map(lambda x: label_types.get(x,"text"))
            sel_unc["selection_reason"] = "model_uncertain_llm"
            if self.cfg.cost.enable:
                seedU = probe_df[['unit_id','label_id','U']].drop_duplicates()
                sel_unc = sel_unc.merge(seedU, on=['unit_id','label_id'], how='left')
                sel_unc = add_ig_over_min(sel_unc, 'model_uncertain_llm', self.repo, self.cfg.cost)
                sel_unc = cost_aware_trim(sel_unc, n_unc)
        # Save seeds for certain bucket reuse
        seeds_cer.to_parquet(os.path.join(self.paths.outdir, "seeds_certain.parquet"), index=False)
        seeds_unc.to_parquet(os.path.join(self.paths.outdir, "seeds_uncertain.parquet"), index=False)
        return sel_unc

    def build_llm_certain_bucket(self, label_types: Dict[str,str], rules_map: Dict[str,str]) -> pd.DataFrame:
        p = os.path.join(self.paths.outdir, "seeds_certain.parquet")
        if os.path.exists(p):
            seeds_cer = pd.read_parquet(p)
        else:
            prober = LLMProber(self.cfg.llmfirst, self.llm, self.rag, self.repo)
            pr = prober.probe_unseen(self.build_unseen_pairs(), label_types, rules_map)
            _, seeds_cer = LLMProber.slice_uncertain_certain(pr, self.cfg.llmfirst.uncertain_top_pct, self.cfg.llmfirst.certain_bottom_pct)
        seen_pairs_set = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        rows = []
        for lid, grp in seeds_cer.groupby("label_id"):
            snips = []
            for r in grp.itertuples(index=False):
                ctx = self.rag.retrieve_for_patient_label(r.unit_id, r.label_id, rules_map.get(lid,""), topk_override=2)
                for s in ctx[:1]:
                    if isinstance(s.get("text"), str) and s["text"].strip(): snips.append(s["text"][:800])
            cand = self.rag.expand_from_snippets(lid, snips, seen_pairs_set, per_seed_k=100)
            for uid, sc in sorted(cand.items(), key=lambda kv: kv[1], reverse=True):
                rows.append({"unit_id": uid, "label_id": lid, "score": float(sc)})
        expanded_cer = pd.DataFrame(rows)
        if not expanded_cer.empty:
            out = []
            for lid, g in expanded_cer.groupby("label_id"):
                arr = g["score"].to_numpy(); g = g.copy(); g["score_n"] = normalize01(arr); out.append(g)
            expanded_cer = pd.concat(out, ignore_index=True)
        n_cer = int(self.cfg.select.batch_size * self.cfg.select.pct_easy_qc)
        per_label_quota = max(1, int(self.cfg.llmfirst.per_label_cap_fraction * max(1, n_cer)))
        per_sel = per_label_kcenter(expanded_cer, self.pooler, self.rag, rules_map, per_label_quota)
        sel_cer = merge_with_global_kcenter(per_sel, expanded_cer, self.pooler, self.rag, rules_map, target_n=n_cer, per_label_cap=per_label_quota)
        if not sel_cer.empty:
            sel_cer["label_type"] = sel_cer["label_id"].map(lambda x: label_types.get(x,"text"))
            sel_cer["selection_reason"] = "easy_qc_llm"
            if self.cfg.cost.enable:
                sel_cer = add_ig_over_min(sel_cer, 'easy_qc_llm', self.repo, self.cfg.cost)
                sel_cer = cost_aware_trim(sel_cer, n_cer)
        return sel_cer

    def build_diversity_bucket(self, already_selected_pairs: List[Tuple[str,str]], unseen_pairs: List[Tuple[str,str]], label_types: Dict[str,str], rules_map: Dict[str,str]) -> pd.DataFrame:
        n_div = int(self.cfg.select.batch_size * self.cfg.select.pct_diversity)
        per_label_cap = max(1, int(self.cfg.llmfirst.per_label_cap_fraction * max(1, n_div)))
        df_div = build_diversity_bucket(unseen_pairs, already_selected_pairs, n_div, per_label_cap, self.pooler, self.rag, rules_map, label_types, sample_cap=2000)
        if self.cfg.cost.enable and not df_div.empty:
            df_div = add_ig_over_min(df_div, 'diversity_kcenter', self.repo, self.cfg.cost)
            df_div = cost_aware_trim(df_div, n_div)
        return df_div

    # ---- Runner ----

    def run(self):
        LOGGER.info("Indexing chunks ...")
        self.store.build_chunk_index(self.repo.notes, self.cfg.rag, self.cfg.index)
        LOGGER.info("Building label prototypes ...")
        self.pooler.build_prototypes()

        rules_map = self.repo.label_rules_by_label
        label_types = self.repo.label_types()

        seen_pairs = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        unseen_pairs = self.build_unseen_pairs()

        # 1) Disagreement bucket
        LOGGER.info("[1/4] Expanded disagreement ...")
        dis_bucket = self.build_disagreement_bucket(seen_pairs, rules_map, label_types)
        dis_bucket.to_parquet(os.path.join(self.paths.outdir, "bucket_disagreement.parquet"), index=False)

        # 2) LLM-uncertain bucket (forced-choice ranking)
        LOGGER.info("[2/4] LLM-uncertain (forced-choice) ...")
        sel_unc = self.build_llm_uncertain_bucket(unseen_pairs, label_types, rules_map)
        sel_unc.to_parquet(os.path.join(self.paths.outdir, "bucket_llm_uncertain.parquet"), index=False)

        # 3) LLM-certain bucket
        LOGGER.info("[3/4] LLM-certain (easy QC) ...")
        sel_cer = self.build_llm_certain_bucket(label_types, rules_map)
        sel_cer.to_parquet(os.path.join(self.paths.outdir, "bucket_llm_certain.parquet"), index=False)

        # 4) Diversity
        LOGGER.info("[4/4] Diversity bucket ...")
        already = []
        for df in (dis_bucket, sel_unc, sel_cer):
            for r in df.itertuples(index=False):
                already.append((r.unit_id, r.label_id))
        sel_div = self.build_diversity_bucket(already, unseen_pairs, label_types, rules_map)
        sel_div.to_parquet(os.path.join(self.paths.outdir, "bucket_diversity.parquet"), index=False)

        # Compose final
        final_union = pd.concat([dis_bucket, sel_unc, sel_cer, sel_div], ignore_index=True, sort=False)
        final_union = final_union.drop_duplicates(subset=["unit_id","label_id"]) 
        # seed with already chosen rows from priority order
        seed = pd.concat([dis_bucket, sel_unc, sel_cer], ignore_index=True, sort=False)
        seed = seed.drop_duplicates(subset=["unit_id","label_id"]) 
        per_label_cap = max(1, int(self.cfg.llmfirst.per_label_cap_fraction * max(1, self.cfg.select.batch_size)))
        final = final_global_topoff(seed, final_union, self.cfg.select.batch_size, self.pooler, self.rag, rules_map, label_types, per_label_cap)
        LOGGER.info('final_topoff')
        self.telemetry.write_json(os.path.join(self.paths.outdir, 'diagnostics.json'))
        final.to_parquet(os.path.join(self.paths.outdir, "final_selection_pre_llm.parquet"), index=False)

        # Provisional LLM labels (reuse not added here to keep changes minimal)
        LOGGER.info("Attaching provisional LLM labels ...")
        rows = []
        for r in final.itertuples(index=False):
            rules = rules_map.get(r.label_id, "")
            ctx = self.rag.retrieve_for_patient_label(r.unit_id, r.label_id, rules, topk_override=self.cfg.rag.per_label_topk)
            res = self.llm.annotate(r.unit_id, r.label_id, r.label_type, rules, ctx, n_consistency=max(2,self.llm.cfg.n_consistency))
            rows.append({
                "unit_id": r.unit_id, "label_id": r.label_id, "label_type": r.label_type,
                "selection_reason": r.selection_reason, "rag_context": ctx,
                "llm_prediction": res.get("prediction"),
                "llm_confidence": res.get("confidence"),
                "llm_consistency": res.get("consistency_agreement"),
                "llm_avg_token_logprob": res.get("avg_token_logprob"),
                "llm_prediction_logprob": res.get("prediction_logprob"),
                "llm_runs": res.get("runs", [])
            })
        llm_df = pd.DataFrame(rows)
        llm_df = _jsonify_cols(llm_df, ["rag_context","llm_runs"])
        llm_df.to_parquet(os.path.join(self.paths.outdir, "final_llm_labels.parquet"), index=False)

        final_out = final.merge(llm_df.drop(columns=["selection_reason"]), on=["unit_id","label_id","label_type"], how="left")
        final_out = _jsonify_cols(final_out, ["rag_context","llm_runs"])
        write_table(final_out, os.path.join(self.paths.outdir, "next_batch.parquet"))
        write_table(final_out, os.path.join(self.paths.outdir, "next_batch.csv"))

        diagnostics = {
            "total_selected": int(len(final)),
            "buckets_counts": {
                "disagreement_expanded": int(len(dis_bucket)),
                "model_uncertain_llm": int(len(sel_unc)),
                "easy_qc_llm": int(len(sel_cer)),
                "diversity_kcenter": int(len(sel_div))
            },
            "llm_calls_made": int(self.llm.calls_made),
            "timestamp": time.time()
        }
        json.dump(diagnostics, open(os.path.join(self.paths.outdir,"diagnostics.json"),"w",encoding="utf-8"), indent=2)
        LOGGER.info("Done. Diagnostics:", diagnostics)


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
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--llm-budget", type=int, default=8000)

    # FAISS index
    ap.add_argument("--index-type", type=str, default="flat", choices=["flat","hnsw","ivf"])
    ap.add_argument("--index-nlist", type=int, default=8192)
    ap.add_argument("--index-nprobe", type=int, default=32)
    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efSearch", type=int, default=64)

    # LLM-first probe knobs
    ap.add_argument("--probe-per-label", type=int, default=400)
    ap.add_argument("--uncertain-top-pct", type=float, default=0.30)
    ap.add_argument("--certain-bottom-pct", type=float, default=0.15)
    ap.add_argument("--borderline-window", type=float, default=0.10)
    ap.add_argument("--stage-a-topk", type=int, default=2)
    ap.add_argument("--stage-b-topk", type=int, default=6)
    ap.add_argument("--per-label-cap-frac", type=float, default=0.20)

    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    paths = Paths(notes_path=args.notes, annotations_path=args.annotations, outdir=args.outdir)
    cfg = OrchestratorConfig(
        index=IndexConfig(type=args.index_type, nlist=args.index_nlist, nprobe=args.index_nprobe, hnsw_M=args.hnsw_M, hnsw_efSearch=args.hnsw_efSearch),
        cost=CostConfig(),
        rag=RAGConfig(),
        llm=LLMConfig(budget_max_calls=args.llm_budget),
        select=SelectionConfig(batch_size=args.batch_size, random_seed=args.seed),
        llmfirst=LLMFirstConfig(
            enable=True,
            per_label_probe_count=args.probe_per_label,
            uncertain_top_pct=args.uncertain_top_pct,
            certain_bottom_pct=args.certain_bottom_pct,
            borderline_window=args.borderline_window,
            escalate_borderline=True,
            stage_a_topk=args.stage_a_topk,
            stage_b_topk=args.stage_b_topk,
            per_label_cap_fraction=args.per_label_cap_frac,
            kcenter_diversity=True
        ),
        disagree=DisagreementConfig()
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
