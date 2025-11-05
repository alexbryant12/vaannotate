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
  - add "cold start" round 0 branch - simple corpus search based on label exemplars -> k-center (basically just a pure diversity run)
  - pure inference branch with query builder: default (zero shot, family-tree traversal, label exemplars). Knobs:
      - zero vs. few shot
      - family-tree traversal vs. single prompt
      - label exemplars for RAG vs. hand-written exemplars
      - Prompt stems - use rules as-is from most recent round, or hand-write
      - checkpointing on inference runs + seamless resume
  
    Final inference query builder
    Final performance checks
    Wire in to app
"""

from __future__ import annotations
import os, re, json, math, time, random, hashlib
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional, Any
import numpy as np
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None
    
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
# Config
# ------------------------------

@dataclass
class IndexConfig:
    type: str = "ivf"    # flat | hnsw | ivf
    nlist: int = 2048     # IVF lists
    nprobe: int = 32      # IVF search probes
    hnsw_M: int = 32      # HNSW graph degree
    hnsw_efSearch: int = 64
    persist: bool = True

@dataclass
class RAGConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 150
    normalize_embeddings: bool = True
    per_label_topk: int = 6
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    mmr_candidates: int = 200
    use_keywords: bool = False
    keyword_topk: int = 20
    min_context_chunks: int = 3
    mmr_multiplier: int = 3
    neighbor_hops: int = 1
        
@dataclass
class LLMConfig:
    model_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    temperature: float = 0.2
    n_consistency: int = 3
    logprobs: bool = True
    top_logprobs: int = 5
    prediction_field: str = "prediction"
    timeout: float = 60.0
    retry_max: int = 3
    retry_backoff: float = 2.0
    max_context_chars: int = 1200000
    rpm_limit: Optional[int] = 30
    
@dataclass
class SelectionConfig:
    batch_size: int = 50
    pct_disagreement: float = 0.3
    pct_uncertain: float = 0.3    # LLM-uncertain
    pct_easy_qc: float = 0.1      # LLM-certain
    pct_diversity: float = 0.3

@dataclass
class LLMFirstConfig:
    use_llm_probe: bool = True
    n_probe_units: int = 100
    topk: int = 6
    json_trace_policy: str = 'fallback'
    progress_min_interval_s: float = 1.0
    exemplar_K: int = 10
    exemplar_generate: bool = True
    exemplar_temperature: float = 0.9
    # forced-choice micro-probe
    fc_enable: bool = True
    #label enrichment for probe
    enrich: bool = True
    probe_enrichment_mix: float = 1.00          # fraction of enriched vs uniform
    probe_enrichment_equalize: bool = True      # equal per parent; else proportional
    probe_ce_unit_sample: int = 1300
    probe_ce_search_topk_per_unit: int = 15
    probe_ce_rerank_m: int = 3        # aggregate top-3 CE
    probe_ce_unit_agg: str = "max"    # or "mean"
    

@dataclass
class DisagreementConfig:
    round_policy: str = 'last'       # 'last' | 'all' | 'decay'
    decay_half_life: float = 2.0     # if round_policy='decay'
    high_entropy_threshold: float = 0.0001 #very low = any disagreements included
    seeds_per_label: int = 20
    snippets_per_seed: int = 3
    similar_chunks_per_seed: int = 150
    expanded_per_label: int = 50
    # Hard-disagreement thresholds
    date_disagree_days: int = 5
    numeric_disagree_abs: float = 1.0
    numeric_disagree_rel: float = 0.20
    
@dataclass
class DiversityConfig:
    rag_k: int = 4
    min_rel_quantile: float = 0.30
    mmr_lambda: float = 0.7
    sample_cap: int = 500
    adaptive_relax: bool = True
    use_proto: bool = False

@dataclass
class SCJitterConfig:
    enable: bool = True
    rag_topk_range: Tuple[int, int] = (4, 10)
    rag_dropout_p: float = 0.20
    temperature_range: Tuple[float, float] = (0.5, 0.9)
    shuffle_context: bool = True 

@dataclass
class Paths:
    notes_path: str
    annotations_path: str
    outdir: str
    cache_dir_override: str | None = None
    cache_dir: str = field(init=False)

    def __post_init__(self):
        outdir_path = Path(self.outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        if self.cache_dir_override:
            cache_dir_path = Path(self.cache_dir_override)
        else:
            cache_dir_path = outdir_path / "cache"
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_dir_path)

@dataclass
class OrchestratorConfig:
    index: IndexConfig = field(default_factory=IndexConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    select: SelectionConfig = field(default_factory=SelectionConfig)
    llmfirst: LLMFirstConfig = field(default_factory=LLMFirstConfig)
    disagree: DisagreementConfig = field(default_factory=DisagreementConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    scjitter: SCJitterConfig = field(default_factory=SCJitterConfig)
    final_llm_labeling: bool = True
    final_llm_labeling_n_consistency: int = 1


# ------------------------------
# Logging (JSON) and Telemetry
# ------------------------------
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

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

class CancelledError(RuntimeError):
    """Raised when a cancellation request is received."""


_cancel_check: Optional[Callable[[], bool]] = None


@contextmanager
def cancellation_scope(callback: Optional[Callable[[], bool]]):
    """Temporarily install a cancellation callback for long-running loops."""
    global _cancel_check
    previous = _cancel_check
    _cancel_check = callback
    try:
        yield
    finally:
        _cancel_check = previous


def check_cancelled() -> None:
    if _cancel_check and _cancel_check():
        raise CancelledError("AI backend run cancelled")


LOGGER = setup_logging()

# ---- Pretty progress logging (ETA) -------------------------------------------
import time as _time
import sys as _sys

def _fmt_hms(_secs: float) -> str:
    if not _secs or _secs == float("inf") or _secs != _secs:  # NaN
        return "--:--:--"
    secs = int(max(0, _secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _bar_str(fraction: float, width: int = 32, ascii_only: bool = False) -> str:
    fraction = max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    if ascii_only:
        mid = ">" if 0 < filled < width else ""
        return "[" + "#" * max(0, filled-1) + mid + "." * (width - filled) + "]"
    # Unicode blocks
    full = "█"; empty = "·"
    return "[" + full * filled + empty * (width - filled) + "]"

def iter_with_bar(step: str, iterable, *, total: int | None = None,
                  bar_width: int = 32, min_interval_s: float = 0.5,
                  ascii_only: bool | None = None, logger=LOGGER):
    """
    Wrap an iterable and render a live progress bar if stderr is a TTY.
    Falls back to periodic log lines otherwise.
    """
    try: 
        if total is None: total = len(iterable)  # may fail for generators
    except Exception:
        total = None

    t0 = _time.time(); last = t0
    tty = hasattr(_sys.stderr, "isatty") and _sys.stderr.isatty()
    if ascii_only is None: ascii_only = not tty  # default: Unicode in TTY

    for i, item in enumerate(iterable, 1):
        check_cancelled()
        now = _time.time()
        if tty and (i == 1 or now - last >= min_interval_s or (total and i == total)):
            last = now
            elapsed = now - t0
            rate = (i / elapsed) if elapsed > 0 else 0.0
            if total:
                frac = i / total
                eta = ((total - i) / rate) if rate > 0 else float("inf")
                bar = _bar_str(frac, width=bar_width, ascii_only=ascii_only)
                msg = f"{step:<14} {bar}  {int(frac*100):3d}%  {i}/{total} • {rate:.2f}/s • ETA {_fmt_hms(eta)}"
            else:
                spinner = "-\\|/"[int((now - t0) * 8) % 4]
                msg = f"{step:<14} [{spinner}]  {i} done • {rate:.2f}/s • elapsed {_fmt_hms(now - t0)}"
            _sys.stderr.write("\r" + msg)
            _sys.stderr.flush()
        elif not tty and (i == 1 or now - last >= min_interval_s or (total and i == total)):
            last = now
            elapsed = now - t0
            rate = (i / elapsed) if elapsed > 0 else 0.0
            if total:
                eta = ((total - i) / rate) if rate > 0 else float("inf")
                logger.info(f"[{step}] {i}/{total} • {rate:.2f}/s • ETA {_fmt_hms(eta)}",
                            extra={"step": step, "done": i, "total": total, "rate_per_s": rate, "eta_s": eta})
            else:
                logger.info(f"[{step}] {i} done • {rate:.2f}/s • elapsed {_fmt_hms(elapsed)}",
                            extra={"step": step, "done": i, "total": None, "rate_per_s": rate, "elapsed_s": elapsed})
        yield item

    # finish line
    if tty:
        if total:
            elapsed = _time.time() - t0
            rate = (total / elapsed) if elapsed > 0 else 0.0
            bar = _bar_str(1.0, width=bar_width, ascii_only=ascii_only)
            _sys.stderr.write("\r" + f"{step:<14} {bar}  100%  {total}/{total} • {rate:.2f}/s • elapsed {_fmt_hms(elapsed)}" + "\n")
        else:
            _sys.stderr.write("\n")
        _sys.stderr.flush()
# ------------------------------------------------------------------------------




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

def _stable_rules_hash(label_id: str, rules: str, K: int, model_sig: str=""):
    s = json.dumps({"label": label_id, "rules": rules or "", "K": int(K), "model": model_sig}, sort_keys=True)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()

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

def _jsonify_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return out


# ------------------------------
# Data repository
# ------------------------------

class DataRepository:
    def __init__(self, notes_df: pd.DataFrame, ann_df: pd.DataFrame):
        required_notes = {"patient_icn","doc_id","text"}
        if not required_notes.issubset(set(notes_df.columns)):
            raise ValueError(f"Notes missing {required_notes}")
        required_ann = {"round_id","unit_id","doc_id","label_id","reviewer_id","label_value"}
        if not required_ann.issubset(set(ann_df.columns)):
            raise ValueError(f"Annotations missing {required_ann}")

        self.notes = notes_df.copy()
        self.notes["unit_id"] = self.notes["patient_icn"].astype(str)
        self.notes["doc_id"] = self.notes["doc_id"].astype(str)
        self.notes["text"] = self.notes["text"].astype(str).map(normalize_text)

        self.ann = ann_df.copy()
        # --- Normalize multi-typed label_value columns ---
        # 1) String label_value (categorical/binary, free text)
        if "label_value" not in self.ann.columns:
            self.ann["label_value"] = ""
        self.ann["label_value"] = self.ann["label_value"].astype(str).str.strip()
        
        # 2) Numeric
        import numpy as _np
        import pandas as _pd
        if "label_value_num" in self.ann.columns:
            self.ann["label_value_num"] = _pd.to_numeric(self.ann["label_value_num"], errors="coerce")
        else:
            self.ann["label_value_num"] = _np.nan
        
        # 3) Date
        if "label_value_date" in self.ann.columns:
            self.ann["label_value_date"] = _pd.to_datetime(self.ann["label_value_date"], errors="coerce", utc=False)
        else:
            self.ann["label_value_date"] = _pd.NaT
            
        
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

    def label_types(self) -> Dict[str, str]:
        """
        Infer label type per label_id using the three input columns:
          - If any non-null in label_value_num   -> "numeric"
          - elif any non-null in label_value_date -> "date"
          - else infer from string: binary vs categorical
        """
        types: Dict[str, str] = {}
        # group once to avoid repeated scans
        for lid, g in self.ann.groupby("label_id", sort=False):
            # numeric wins
            if g["label_value_num"].notna().any():
                types[str(lid)] = "numeric"
                continue
            # date next
            if g["label_value_date"].notna().any():
                types[str(lid)] = "date"
                continue
            # else string; decide binary vs categorical heuristically
            vals = g["label_value"].astype(str).str.lower().str.strip()
            uniq = set(v for v in vals if v not in ("", "nan", "none"))
            # common binary token set
            bin_tokens = {"0","1","true","false","present","absent","yes","no","neg","pos","positive","negative","unknown"}
            if uniq and uniq.issubset(bin_tokens):
                types[str(lid)] = "binary"
            else:
                types[str(lid)] = "categorical"  # use 'categorical' instead of 'text'
        return types

    def reviewer_disagreement(self, round_policy: str = 'last',
                          decay_half_life: float = 2.0) -> pd.DataFrame:
        """
        Compute a per-(unit_id,label_id) disagreement score in [0,1].
          - categorical/binary: normalized entropy
          - numeric: reviewer range / global IQR (clipped to [0,1])
          - date: span_days / global IQR_days (clipped)
        """
        import numpy as _np
        import pandas as _pd
    
        ann = self.ann.copy()
    
        # Round selection
        try:
            ann["_round_ord"] = _pd.to_numeric(ann["round_id"], errors="coerce")
            ord_series = ann["_round_ord"].fillna(ann["round_id"].astype("category").cat.codes)
        except Exception:
            ord_series = ann["round_id"].astype("category").cat.codes
        ann["_round_ord"] = ord_series
        last_ord = int(ann["_round_ord"].max()) if len(ann) else 0
        if round_policy == "last":
            ann = ann[ann["_round_ord"] == last_ord]
        # else 'all' or 'decay' -> keep all; decay applied later
    
        types = self.label_types()
    
        # Precompute global IQR per label for numeric/date
        iqr_num: Dict[str, float] = {}
        iqr_days: Dict[str, float] = {}
    
        for lid, g in ann.groupby("label_id", sort=False):
            t = types.get(str(lid), "categorical")
            if t == "numeric" and g["label_value_num"].notna().any():
                arr = g["label_value_num"].dropna().to_numpy(dtype="float64")
                if arr.size >= 2:
                    iqr = float(_np.subtract(*_np.percentile(arr, [75, 25])))
                    iqr_num[str(lid)] = iqr if iqr > 0 else 1.0
                else:
                    iqr_num[str(lid)] = 1.0
            elif t == "date" and g["label_value_date"].notna().any():
                dt = g["label_value_date"].dropna()
                if not dt.empty:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    iqr = float(_np.subtract(*_np.percentile(ords, [75, 25])))
                    iqr_days[str(lid)] = iqr if iqr > 0 else 1.0
                else:
                    iqr_days[str(lid)] = 1.0
    
        rows = []
        # group by unit, label, reviewer to assemble per (unit,label) reviewer values
        for (uid, lid), g in ann.groupby(["unit_id","label_id"], sort=False):
            t = types.get(str(lid), "categorical")
            score = 0.0
    
            if t in ("categorical","binary"):
                vals = g["label_value"].astype(str).str.lower().str.strip()
                uniq, cnts = _np.unique([v for v in vals if v not in ("","nan","none")], return_counts=True)
                total = cnts.sum() if cnts.size else 0
                if total > 0:
                    p = cnts / total
                    ent = -_np.sum(p * _np.log2(_np.clip(p, 1e-12, 1)))
                    ent_max = _np.log2(max(len(uniq), 2))
                    score = float(ent / ent_max) if ent_max > 0 else 0.0
                else:
                    score = 0.0
    
            elif t == "numeric" and g["label_value_num"].notna().any():
                arr = g["label_value_num"].dropna().to_numpy(dtype="float64")
                if arr.size >= 2:
                    rng = float(_np.nanmax(arr) - _np.nanmin(arr))
                    denom = max(1e-6, iqr_num.get(str(lid), 1.0))
                    score = float(max(0.0, min(1.0, rng / (4.0 * denom))))  # 0..1
                else:
                    score = 0.0
    
            elif t == "date" and g["label_value_date"].notna().any():
                dt = g["label_value_date"].dropna()
                if len(dt) >= 2:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    span = float(_np.nanmax(ords) - _np.nanmin(ords))  # days
                    denom = max(1e-6, iqr_days.get(str(lid), 1.0))
                    score = float(max(0.0, min(1.0, span / (4.0 * denom))))  # 0..1
                else:
                    score = 0.0
            else:
                score = 0.0
    
            # Round decay (if policy='decay')
            if round_policy == "decay":
                delta = (last_ord - int(g["_round_ord"].max())) if len(g) else 0
                w = float(_np.exp(-float(delta) / max(1e-6, float(decay_half_life))))
                score *= w
    
            rows.append({
                "unit_id": str(uid),
                "label_id": str(lid),
                "disagreement_score": float(score),
                "n_reviewers": int(g["reviewer_id"].nunique())
            })
    
        return _pd.DataFrame(rows)

    def hard_disagree(self, label_types: dict, *, date_days: int = 14, num_abs: float = 1.0, num_rel: float = 0.20) -> pd.DataFrame:
        """Return per-(unit_id,label_id) hard disagreement flags based on absolute/relative numeric and date-day spans.
        categorical/binary: hard=True if multiple unique non-empty values
        numeric: max pairwise |Δ| > max(num_abs, num_rel * max(|v_i|))
        date: span_days > date_days
        """
        import numpy as _np
        import pandas as _pd
        ann = self.ann.copy()
        rows = []
        for (uid, lid), g in ann.groupby(["unit_id","label_id"], sort=False):
            t = label_types.get(str(lid), "categorical")
            hard = False
            if t in ("categorical","binary"):
                vals = g["label_value"].astype(str).str.lower().str.strip()
                uniq = [v for v in vals.unique().tolist() if v not in ("","nan","none")]
                hard = (len(uniq) >= 2)
            elif t == "numeric" and g["label_value_num"].notna().any():
                arr = g["label_value_num"].dropna().to_numpy(dtype="float64")
                if arr.size >= 2:
                    vmax = float(_np.nanmax(_np.abs(arr)))
                    thresh = max(float(num_abs), float(num_rel) * max(1.0, vmax))
                    dif = float(_np.nanmax(arr) - _np.nanmin(arr))
                    hard = dif > thresh
            elif t == "date" and g["label_value_date"].notna().any():
                dt = g["label_value_date"].dropna()
                if len(dt) >= 2:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    span = float(_np.nanmax(ords) - _np.nanmin(ords))
                    hard = span > float(date_days)
            rows.append({ "unit_id": str(uid), "label_id": str(lid), "hard_disagree": bool(hard) })
        return _pd.DataFrame(rows)

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
        """
        Returns {(unit_id,label_id): consensus_value_as_string} for the last round.
        Numeric/date are summarized by robust medians; categorical/binary by mode.
        """
        import numpy as _np
        import pandas as _pd
    
        ann = self.ann.copy()
        # establish last round ordering
        try:
            ann["_round_ord"] = _pd.to_numeric(ann["round_id"], errors="coerce")
            ord_series = ann["_round_ord"].fillna(ann["round_id"].astype("category").cat.codes)
        except Exception:
            ord_series = ann["round_id"].astype("category").cat.codes
        ann["_round_ord"] = ord_series
        last_ord = int(ann["_round_ord"].max()) if len(ann) else 0
        ann = ann[ann["_round_ord"] == last_ord]
    
        # Infer types on the last round only (cheaper)
        types = self.label_types()
        out: Dict[Tuple[str,str], str] = {}
        for (uid, lid), g in ann.groupby(["unit_id","label_id"], sort=False):
            t = types.get(str(lid), "categorical")
            if t == "numeric" and g["label_value_num"].notna().any():
                med = _np.nanmedian(g["label_value_num"].to_numpy(dtype="float64"))
                out[(str(uid), str(lid))] = str(med)
            elif t == "date" and g["label_value_date"].notna().any():
                # median date: convert to ordinal days, then back
                dt = g["label_value_date"].dropna()
                if not dt.empty:
                    ords = dt.map(lambda x: x.toordinal()).to_numpy(dtype="int64")
                    med_ord = int(_np.median(ords))
                    out[(str(uid), str(lid))] = str(_pd.Timestamp.fromordinal(med_ord).date())
                else:
                    out[(str(uid), str(lid))] = ""
            else:
                # categorical/binary: mode of string
                vals = g["label_value"].astype(str).str.lower().str.strip()
                if len(vals):
                    mode = vals.mode(dropna=True)
                    out[(str(uid), str(lid))] = str(mode.iloc[0]) if len(mode) else ""
                else:
                    out[(str(uid), str(lid))] = ""
        return out

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
    
    def _paths_for_cache(self, chunk_dir: str, index_cfg) -> dict:
        idx_name = f"faiss_{getattr(index_cfg, 'type', 'flat')}.index"
        return {
            "meta": os.path.join(chunk_dir, "chunk_meta.json"),
            "emb": os.path.join(chunk_dir, "chunk_embeddings.npy"),
            "faiss": os.path.join(chunk_dir, idx_name),
        }
    
    def _try_load_cached_chunks(self, chunk_dir: str) -> tuple[list[dict] | None, np.ndarray | None, dict | None]:
        """
        Returns (chunk_meta, X_mmap, manifest) or (None, None, None) if unavailable.
        """
        paths = self._paths_for_cache(chunk_dir, type("Cfg", (), {"type": "flat"}))
        meta_p, emb_p = paths["meta"], paths["emb"]
        man = self._load_manifest(chunk_dir)
        if not (os.path.exists(meta_p) and os.path.exists(emb_p) and man):
            return (None, None, None)
        try:
            meta = json.load(open(meta_p, "r", encoding="utf-8"))
            X = np.load(emb_p, mmap_mode="r")  # memmap
            # sanity: manifest rows/dims
            n = int(man.get("n_chunks", -1))
            d = int(man.get("dim", -1))
            if (n > 0 and n != len(meta)) or (d > 0 and d != int(X.shape[1])):
                return (None, None, None)
            return (meta, X, man)
        except Exception:
            return (None, None, None)
    
    def _save_cached_chunks(self, chunk_dir: str, meta: list[dict], X: np.ndarray, rag_cfg):
        paths = self._paths_for_cache(chunk_dir, type("Cfg", (), {"type": "flat"}))
        meta_p, emb_p = paths["meta"], paths["emb"]
        # Save meta
        tmp = meta_p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, meta_p)
        # Save embeddings (np.save is atomic-ish via temp file on most FS; to be safe, write to tmp then replace)
        np.save(emb_p, X)
        # Manifest
        self._save_manifest(chunk_dir, {
            "n_chunks": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "chunk_size": int(rag_cfg.chunk_size),
            "chunk_overlap": int(rag_cfg.chunk_overlap),
            "normalize": bool(self.normalize),
            "embedder": self._embedder_id(),
            "version": "v1",
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

    def _embed(self, texts: List[str], show_bar: Optional[bool] = False) -> np.ndarray:
        if show_bar:
            bs = getattr(self.models, 'emb_batch', 64)
            out = []
            for i in iter_with_bar("Embedding chunks",
                                   range(0, len(texts), bs),
                                   total=(len(texts)+bs-1)//bs,
                                   logger=LOGGER,
                                   min_interval_s=1.0):
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
    
        Cache layout: <cache_dir>/chunks/<fingerprint>/{chunk_meta.json, chunk_embeddings.npy, faiss_*.index, manifest.json}
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
            for row in iter_with_bar(
                    step="Chunking docs",
                    iterable=notes_df.itertuples(index=False),
                    total=total_docs,
                    logger=LOGGER,
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
                                md[k] = v
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
        self.X = np.load(paths["emb"], mmap_mode="r") if isinstance(X, np.memmap) or isinstance(X, np.ndarray) else X
        self.chunk_meta = meta
        unit_to_idxs = defaultdict(list)
        for i, m in enumerate(self.chunk_meta):
            unit_to_idxs[m["unit_id"]].append(i)
        self.unit_to_chunk_idxs = dict(unit_to_idxs)
    
        # 4) FAISS index: try load; else build and persist
        if faiss is None:
            raise ImportError("faiss-cpu is required")
        d = int(self.X.shape[1])
        idx = None if force_reindex else self._try_load_faiss_index(paths["faiss"], expected_n=int(self.X.shape[0]))
        if idx is None:
            print("Building index...")
            # (re)build index
            if index_cfg.type == "flat":
                idx = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
                idx.add(self.X)
            elif index_cfg.type == "hnsw":
                idx = faiss.IndexHNSWFlat(d, index_cfg.hnsw_M)
                idx.hnsw.efSearch = index_cfg.hnsw_efSearch
                idx.add(self.X)
            elif index_cfg.type == "ivf":
                quant = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
                idx = faiss.IndexIVFFlat(quant, d, index_cfg.nlist,
                                         faiss.METRIC_INNER_PRODUCT if self.normalize else faiss.METRIC_L2)
                ntrain = min(self.X.shape[0], max(10000, index_cfg.nlist * 40))
                samp = self.X[np.random.choice(self.X.shape[0], ntrain, replace=False)]
                idx.train(samp)
                idx.add(self.X)
                idx.nprobe = index_cfg.nprobe
            elif index_cfg.type == "ivfpq":
                quant = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
                idx = faiss.IndexIVFPQ(quant, d, index_cfg.nlist, index_cfg.pq_m, index_cfg.pq_bits)
                ntrain = min(self.X.shape[0], max(50000, getattr(index_cfg, "train_size", 100000)))
                samp = self.X[np.random.choice(self.X.shape[0], ntrain, replace=False)]
                idx.train(samp)
                idx.add(self.X)
                idx.nprobe = index_cfg.nprobe
            else:
                raise ValueError(f"Unknown index type: {index_cfg.type}")
    
            # Persist index
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
                self.X = np.load(paths["emb"], mmap_mode="r")
                self.chunk_meta = json.load(open(paths["meta"], "r", encoding="utf-8"))
                # load or build index
                idx = self._try_load_faiss_index(paths["faiss"], expected_n=int(self.X.shape[0]))
                if idx is None:
                    # minimal flat index fallback
                    d = int(self.X.shape[1])
                    idx = faiss.IndexFlatIP(d) if self.normalize else faiss.IndexFlatL2(d)
                    idx.add(self.X)
                self.faiss_index = idx
                # rebuild unit_to_chunk lookup
                unit_to_idxs = defaultdict(list)
                for i, m in enumerate(self.chunk_meta):
                    unit_to_idxs[str(m["unit_id"])].append(i)
                self.unit_to_chunk_idxs = dict(unit_to_idxs)
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
            Mp = os.path.join(self.cache_dir,"chunk_meta.json")
            if os.path.exists(Mp):
                self.chunk_meta = json.load(open(Mp,"r",encoding="utf-8"))
        return [i for i,m in enumerate(self.chunk_meta) if m.get("unit_id")==uid]


class DisagreementExpander:
    def __init__(self, cfg: DisagreementConfig, repo: DataRepository, retriever: RAGRetriever, label_config: Optional[dict]=None):
        self.cfg = cfg; self.repo = repo; self.retriever = retriever; self.label_config = label_config or {}

    def high_entropy_seeds(self) -> pd.DataFrame:
        dis = self.repo.reviewer_disagreement(round_policy=self.cfg.round_policy, decay_half_life=self.cfg.decay_half_life)
        seeds = dis[dis["disagreement_score"] >= self.cfg.high_entropy_threshold].copy()
        types = self.repo.label_types()
        hard_df = self.repo.hard_disagree(types, date_days=int(self.cfg.date_disagree_days), num_abs=float(self.cfg.numeric_disagree_abs), num_rel=float(self.cfg.numeric_disagree_rel))
        if not hard_df.empty:
            hard_pairs = set(zip(hard_df.loc[hard_df['hard_disagree'],'unit_id'].astype(str), hard_df.loc[hard_df['hard_disagree'],'label_id'].astype(str)))
            if hard_pairs:
                add = dis[[ (str(r.unit_id), str(r.label_id)) in hard_pairs for r in dis.itertuples(index=False) ]].copy()
                seeds = pd.concat([seeds, add], ignore_index=True).drop_duplicates(subset=['unit_id','label_id'])
        seeds = seeds.sort_values("disagreement_score", ascending=False)
        
        # ---- Parent→child gating on seeds (use prior-round consensus) ----
        try:
            parent_to_children, child_to_parents, roots = build_label_dependencies(self.label_config)
        except Exception:
            parent_to_children, child_to_parents, roots = {}, {}, []
        roots = set(str(x) for x in (roots or []))
        types = self.repo.label_types()
        consensus = self.repo.last_round_consensus()  # {(unit_id,label_id)-> value str}
        
        def _gate_seed(uid: str, lid: str) -> bool:
            # Parents (roots) are always eligible; children need parent gate pass
            if str(lid) in roots:
                return True
            parents = child_to_parents.get(str(lid), [])
            if not parents:
                return True  # not marked as child → treat as eligible
            parent_preds = {(str(uid), str(p)): consensus.get((str(uid), str(p)), None) for p in parents}
            # IMPORTANT: evaluate by prior-round consensus; robust evaluator handles casing/types
            return evaluate_gating(str(lid), str(uid), parent_preds, types, self.label_config)
        
        if not seeds.empty:
            seeds["unit_id"] = seeds["unit_id"].astype(str)
            seeds["label_id"] = seeds["label_id"].astype(str)
            seeds = seeds[seeds.apply(lambda r: _gate_seed(r["unit_id"], r["label_id"]), axis=1)]
        
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


class RAGRetriever:
    _RR_CACHE_MAX = 200000
    def __init__(self, store: EmbeddingStore, models: Models, cfg: RAGConfig, label_configs: Optional[dict]=None, notes_by_doc: Optional[Dict[str,str]]=None, repo: Optional[DataRepository]=None):
        self.store = store; self.models = models; self.cfg = cfg
        self.label_configs = label_configs or {}
        self._notes_by_doc = notes_by_doc or {}
        self._repo = repo
        self._rr_cache = LRUCache(maxsize=self._RR_CACHE_MAX)
        self._label_query_texts = {}   # (label_id, rules_hash, K) -> List[str]
        self._label_query_embs  = {}   # (label_id, rules_hash, K) -> np.ndarray[K,d]

    def set_label_exemplars(self, label_id: str, rules: str, K: int, texts: list[str]):
        key = (str(label_id), _stable_rules_hash(label_id, rules, K, getattr(self.models.embedder, "name_or_path", "")), int(K))
        self._label_query_texts[key] = [t for t in texts if isinstance(t, str) and t.strip()]
        if self._label_query_texts[key]:
            E = self.store._embed(self._label_query_texts[key])   # (K, d)
            self._label_query_embs[key] = E
    
    def _get_label_query_embs(self, label_id: str, rules: str, K: int):
        key = (str(label_id), _stable_rules_hash(label_id, rules, K, getattr(self.models.embedder, "name_or_path", "")), int(K))
        return self._label_query_embs.get(key)
    
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
        if "notetype" in m and m["notetype"]:
            out["note_type"] = str(m["notetype"])
    
        # Try to parse richer metadata JSON if present
        meta_raw = m.get("document_metadata_json") or m.get("metadata_json")
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
        base = f"Evidence relevant to patient-level label '{label_id}'. "
        if label_rules and isinstance(label_rules,str) and label_rules.strip():
            base += "Guidelines: " + re.sub(r"\s+"," ",label_rules.strip()) + " "
        return base.strip()

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
                out.append({"doc_id": m["doc_id"], "chunk_id": m["chunk_id"], 'metadata': self._extract_meta(m),
                            "text": txt, "score": float(score), "source": "keyword"})
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

    def retrieve_for_patient_label(
        self,
        unit_id: str | None,
        label_id: str,
        label_rules: str | None,
        topk_override: int | None = None,
        min_k_override: int | None = None,
        mmr_lambda_override: float | None = None,   
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
        final_k   = int(topk_override or getattr(cfg_rag, "per_label_topk", 6))
        min_k     = min_k_override or max(1, getattr(cfg_rag, "min_context_chunks", 3))
        mmr_mult  = max(1, getattr(cfg_rag, "mmr_multiplier", 3))  # pool size before CE = final_k * mmr_mult
        hops      = int(getattr(cfg_rag, "neighbor_hops", 1))
        use_kw    = bool(getattr(cfg_rag, "use_keywords", True))
        
        # λ (0..1)
        lam = mmr_lambda_override
        if lam is None: lam = getattr(cfg_rag, "mmr_lambda", None)
        lam = None if lam is None else float(lam)
        if lam is not None: lam = max(0.0, min(1.0, lam))

        # ---- label-aware query + embedding ----
        K = int(getattr(self.cfg, "exemplar_K", 6) or 6)
        Q = self._get_label_query_embs(label_id, label_rules, K=K)
        mmr_select_k = final_k * mmr_mult
        
        if Q is not None and getattr(Q, "ndim", 1) == 2 and Q.shape[0] > 0:
            # multi-vector preselect by max-sim across exemplars
            items = _patient_local_rank_multi(str(unit_id), Q, need=mmr_select_k * 2)
            # centroid for MMR
            q_emb = Q.mean(axis=0)
            query = f"[exemplar centroid for {label_id}]"
        else:
            # fallback to rules-string query
            query = self._build_query(label_id, label_rules)
            q_emb = self.store._embed([query])[0]
            items = _patient_local_rank(str(unit_id), q_emb, need=mmr_select_k * 2)
            
        #  + (optional) keywords
        if use_kw:
            lblcfg = self.label_configs.get(label_id, {}) if isinstance(self.label_configs, dict) else {}
            kw = lblcfg.get("keywords", [])
            items += self._keyword_hits_for_patient(str(unit_id), kw)

       # Dedup
        pool = _dedup_only(items)
    
        # Neighbor padding for short pools
        if len(pool) < max(min_k, final_k):
            extra = _neighbors(pool, hops=hops)
            pool = _dedup_only(pool + extra)
        if not pool:
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
    
        # CE fallback if mapping failed
        if not cand_idxs:
            texts = [it["text"] for it in pool]
            rr = self._cross_scores_cached(query, texts)
            for it, s in zip(pool, rr):
                it["score"] = float(s)
            pool.sort(key=lambda d: d["score"], reverse=True)
            return pool[:final_k]
    
        # Preselect for CE (MMR or head)
        k_pre = min(len(cand_items), max(final_k, min_k, mmr_select_k))
        if lam is not None:
            sel = self._mmr_select(q_emb, cand_idxs, k=k_pre, lam=lam)
            idx_to_item = {ix: it for ix, it in zip(cand_idxs, cand_items)}
            pre = [idx_to_item[ix] for ix in sel if ix in idx_to_item]
        else:
            pre = cand_items[:k_pre]
    
        # CE last, CE-only score
        texts = [it["text"] for it in pre]
        rr = self._cross_scores_cached(query, texts)
        for it, s in zip(pre, rr):
            it["score"] = float(s)
    
        pre.sort(key=lambda d: d["score"], reverse=True)
        out = pre[:final_k]
    
        # Top-off to min_k using CE on the rest
        if len(out) < min_k:
            picked = set((o["doc_id"], int(o["chunk_id"])) for o in out)
            rest = [it for it in pool if (it["doc_id"], int(it["chunk_id"])) not in picked]
            if rest:
                rr2 = self._cross_scores_cached(query, [it["text"] for it in rest])
                for it, s in zip(rest, rr2):
                    it["score"] = float(s)
                rest.sort(key=lambda d: d["score"], reverse=True)
                need = min_k - len(out)
                out.extend(rest[:need])
    
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


# ---- In-memory per-run LLM call recorder ----
class LLMRunRecorder:
    def __init__(self):
        self.calls = []
        self.run_meta = {}
        self.run_id = None
        self.outdir = None

    def start(self, outdir: str, run_id: Optional[str] = None, meta: Optional[dict] = None):
        import time as _time, os
        self.calls = []
        self.run_meta = meta or {}
        self.run_id = run_id or _time.strftime("%Y%m%d-%H%M%S")
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def record(self, kind: str, payload: dict):
        import time as _time
        self.calls.append({"ts": _time.time(), "kind": kind, **(payload or {})})

    def flush(self) -> Optional[str]:
        import json, os
        if not self.outdir: 
            return None
        path = os.path.join(self.outdir, f"llm_calls_{self.run_id}.json")
        data = {
            "run_id": self.run_id,
            "meta": self.run_meta,
            "count": len(self.calls),
            "calls": self.calls,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)  # pretty, human-readable
        return path

# Global recorder (simple and sufficient for single-process run())
LLM_RECORDER = LLMRunRecorder()

# ------------------------------
# LLM annotator (JSON call) + forced-choice micro-probe
# ------------------------------
class LLMAnnotator:
    def __init__(self, cfg: LLMConfig, scCfg: SCJitterConfig, cache_dir: str):
        self.cfg = cfg
        self.scCfg = scCfg
        self.cache_dir = os.path.join(cache_dir,"llm_cache"); os.makedirs(self.cache_dir, exist_ok=True)
        self.client = None
        self._init_client()

    def _init_client(self):
        if AzureOpenAI is None: raise ImportError("Please install openai>=1.0 for Azure")
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION","2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=self.cfg.timeout
        )

    def _save(self, key: str, data: dict):
        # Record final aggregate per-call object in memory; the single file is written at the end of run()
        try:
            LLM_RECORDER.record("aggregate", {"cache_key": key, "out": data})
        except Exception:
            pass

    # ---- normalization & parsing ----
    @staticmethod
    def _norm_token(x) -> str:
        if x is None:
            return ""
        s = str(x)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _parse_float(value):
        try:
            if value is None:
                return None
            s = str(value).strip()
            s = re.sub(r"[\s,]", "", s)
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _parse_date(value):
        from datetime import datetime, date
        if value is None:
            return None
        s = str(value).strip()
        fmts = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d %b %Y", "%b %d %Y"]
        for f in fmts:
            try:
                return datetime.strptime(s, f).date()
            except Exception:
                continue
        m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
        if m:
            try:
                y, mo, da = map(int, m.groups())
                return date(y, mo, da)
            except Exception:
                return None
        return None

    # ---- unknown-aware pairwise agreement ----
    @staticmethod
    def _is_unknown_token(s: str) -> bool:
        if s is None:
            return True
        t = str(s).strip().lower()
        return t in {"", "na", "n/a", "none", "null", "unknown", "not evaluable", "not_evaluable", "absent", "missing"}

    @staticmethod
    def _pairwise_agree_numeric(preds: List, abs_scale: float = 1.0, rel_scale: float = 0.05,
                                w_uu: float = 0.5, w_uk: float = 0.0) -> float:
        n = len(preds)
        if n <= 1:
            return 1.0
        vals = [LLMAnnotator._parse_float(p) if not LLMAnnotator._is_unknown_token(p) else None for p in preds]
        known_vals = [v for v in vals if v is not None]
        if known_vals:
            med = float(np.median(known_vals))
            tau = max(abs_scale, rel_scale * (abs(med) if med != 0 else 1.0))
        else:
            tau = abs_scale
        sim_sum, pairs = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                ui = vals[i] is None
                uj = vals[j] is None
                if ui and uj:
                    s = w_uu
                elif ui != uj:
                    s = w_uk
                else:
                    s = math.exp(-abs(vals[i] - vals[j]) / tau)
                sim_sum += s
                pairs += 1
        return float(sim_sum / max(1, pairs))

    @staticmethod
    def _pairwise_agree_date(preds: List, tau_days: int = 14,
                             w_uu: float = 0.5, w_uk: float = 0.0) -> float:
        n = len(preds)
        if n <= 1:
            return 1.0
        ords = []
        for p in preds:
            if LLMAnnotator._is_unknown_token(p):
                ords.append(None)
            else:
                d = LLMAnnotator._parse_date(p)
                ords.append(d.toordinal() if d is not None else None)
        sim_sum, pairs = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                ui = ords[i] is None
                uj = ords[j] is None
                if ui and uj:
                    s = w_uu
                elif ui != uj:
                    s = w_uk
                else:
                    s = math.exp(-abs(ords[i] - ords[j]) / float(tau_days))
                sim_sum += s
                pairs += 1
        return float(sim_sum / max(1, pairs))

    def annotate(self, unit_id: str, label_id: str, label_type: str, label_rules: str, snippets: List[dict], n_consistency: int=1, 
                 jitter_params: bool=False) -> dict:
        import json, time
        
        rag_topk_range     = self.scCfg.rag_topk_range
        rag_dropout_p      = self.scCfg.rag_dropout_p
        temp_range         = self.scCfg.temperature_range
        shuffle_context    = self.scCfg.shuffle_context
    
        # Jitter RNG
        rng = random.Random()
    
        # --- helper: build context text from a candidate chunk list with char budget ---
        def _build_context_text(_snips: List[dict]) -> str:
            ctx, used = [], 0
            budget = max(1000, getattr(self.cfg, "max_context_chars", 4000))
            for s in _snips:
                md = s.get("metadata") or {}
                hdr_bits = [f"doc_id={s.get('doc_id')}", f"chunk_id={s.get('chunk_id')}"]
                if md.get("date"):      hdr_bits.append(f"date={md['date']}")
                if md.get("note_type"): hdr_bits.append(f"type={md['note_type']}")
                header = "[" + ", ".join(hdr_bits) + "] "
                text_body = (s.get("text", "") or "")
                frag = header + text_body
                if used + len(frag) > budget:
                    break
                ctx.append(frag)
                used += len(frag)
            return "\n\n".join(ctx)
    

        # --- run n_consistency votes (each vote = fresh jitter if enabled) ---
        preds, runs = [], []
        time_last_call = time.time()
    
        # Prebuild immutable "system" header (we add a tiny meta line per vote to avoid LLM-side cache collisions)
        system_base = ("You are a meticulous clinical annotator for EHR data. "
                       "Follow the label rules precisely. Return strict JSON only.")
    
        for i in range(n_consistency):
            # ----- sample jitter for this vote -----
            if jitter_params:
                # top-K
                kmin, kmax = rag_topk_range
                kmin = max(1, int(kmin or 1))
                kmax = max(kmin, int(kmax or kmin))
                k = min(len(snippets), rng.randint(kmin, kmax))
                # dropout & shuffle
                drop_p = max(0.0, min(1.0, rag_dropout_p))
                cand = list(snippets[:k]) if k > 0 else list(snippets)
                if drop_p > 0.0:
                    cand = [s for s in cand if rng.random() > drop_p] or [snippets[0]]
                if shuffle_context and len(cand) > 1:
                    rng.shuffle(cand)
                # temperature
                t_lo, t_hi = temp_range
                t = rng.uniform(float(t_lo), float(t_hi))
                # meta string to perturb the prompt slightly (transparent to the schema)
                sc_meta = f"<!-- sc:vote={i};k={k};drop={drop_p:.2f};shuf={int(shuffle_context)};temp={t:.2f} -->"
                system = system_base + "\n" + sc_meta
                ctx_text = _build_context_text(cand)
                temperature_this_vote = t
            else:
                # legacy behavior: use original ranked snippets (no jitter), fixed temperature
                system = system_base
                ctx_text = _build_context_text(snippets)
                temperature_this_vote = self.cfg.temperature
    
            # ----- build prompt for this vote -----
            task = (
                f"Label: '{label_id}' (type: {label_type}). "
                "Use the evidence snippets from this patient's notes. "
                "If insufficient evidence, reply with 'unknown'.\n\n"
                f"Guidelines:\n{label_rules}\n\n"
                "Evidence snippets:\n" + ctx_text + "\n\n"
                "RESPONSE JSON keys: prediction, reasoning"
            )
            messages = [{"role": "system", "content": system},
                        {"role": "user",   "content": task}]
    
            # ----- per-vote LLM call -----
            attempt = 0
            while attempt <= self.cfg.retry_max:
                try:
                    kwargs = dict(
                        model=self.cfg.model_name,
                        temperature=temperature_this_vote,
                        response_format={"type": "json_object"},
                        logprobs=self.cfg.logprobs,
                        n=1,
                        messages=messages,
                    )
                    if self.cfg.logprobs and int(self.cfg.top_logprobs) > 0:
                        kwargs["top_logprobs"] = int(self.cfg.top_logprobs)
    
                    # simple RPM limiter (your original approach)
                    if self.cfg.rpm_limit is not None:
                        min_spacing = float(60 / self.cfg.rpm_limit)
                        since = time.time() - time_last_call
                        if since < min_spacing:
                            time.sleep(min_spacing - since)
    
                    t0 = time.time()
                    resp = self.client.chat.completions.create(**kwargs)
                    time_last_call = time.time()
                    dt = time.time() - t0
                    content = resp.choices[0].message.content
                    data = json.loads(content)
    
                    pred = data.get(self.cfg.prediction_field, data.get("prediction"))
    
                    preds.append(str(pred) if pred is not None else None)

                    runs.append({
                        "prediction": pred,
                        "raw": data,
                        "jitter": ({"k": k, "drop": drop_p, "shuffle": shuffle_context, "temperature": temperature_this_vote}
                                   if jitter_params else None),
                    })
                    
                    try:
                        # Capture the exact prompt + minimal context identifiers used this vote
                        LLM_RECORDER.record("json_vote", {
                            "unit_id": unit_id,
                            "label_id": label_id,
                            "label_type": label_type,
                            "vote_idx": i,
                            "prompt": {"system": system, "user": task},
                            "params": {
                                "temperature": temperature_this_vote,
                                "n_consistency": int(n_consistency),
                            },
                            "snippets": [{"doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")} for c in (cand if jitter_params else snippets)],
                            "output": {"prediction": pred, "raw": data},
                        })
                    except Exception:
                        LLM_RECORDER.record("json_vote_error", {})
                        pass
                    
                    break  # success
                except Exception as e:
                    if attempt >= self.cfg.retry_max:
                        runs.append({"error": str(e)})
                        break
                    time.sleep(self.cfg.retry_backoff * (attempt + 1))
                    attempt += 1
    
        # --- Distance-aware, unknown-aware self-consistency aggregation ---
        pred_final, cons = None, 0.0
        norm_preds = [self._norm_token(p) for p in preds]
        lt = (label_type or "").lower()

        if lt in ("number", "numeric", "float", "int"):
            cons = self._pairwise_agree_numeric(norm_preds, abs_scale=1.0, rel_scale=0.05, w_uu=0.8, w_uk=0.0)
            vals = [self._parse_float(p) for p in norm_preds if not self._is_unknown_token(p)]
            if vals:
                pred_final = str(float(np.median(vals)))
            else:
                pred_final = "unknown"

        elif "date" in lt:
            cons = self._pairwise_agree_date(norm_preds, tau_days=14, w_uu=0.8, w_uk=0.0)
            dvals = [self._parse_date(p) for p in norm_preds if not self._is_unknown_token(p)]
            dvals = [d for d in dvals if d is not None]
            if dvals:
                ords = sorted([d.toordinal() for d in dvals])
                med_ord = ords[len(ords) // 2]
                from datetime import date
                pred_final = str(date.fromordinal(med_ord))
            else:
                pred_final = "unknown"

        else:
            # Categorical/text → majority vote
            cnt = {p: norm_preds.count(p) for p in set(norm_preds) if p is not None}
            if cnt:
                pred_final = max(cnt.items(), key=lambda kv: kv[1])[0]
                cons = cnt[pred_final] / max(1, len(norm_preds))
    
        out = {
            "unit_id": unit_id, "label_id": label_id, "label_type": label_type,
            "prediction": pred_final, 
            "consistency_agreement": float(cons),
             "runs": runs
        }
        return out


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
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
        self._cache_vec: Dict[Tuple[str,str], np.ndarray] = {}

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
    return None

# ------------------------------
# Selection helpers
# ------------------------------
# Parent-Child Gating + Family Labeling
# ------------------------------

def _parse_date(x):
    try:
        import pandas as _pd
        if x is None or (isinstance(x, float) and math.isnan(x)): return None
        dt = _pd.to_datetime(x, errors="coerce")
        if dt is _pd.NaT: return None
        return dt
    except Exception:
        return None

def _to_number(x):
    try:
        if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return float(x)
        s = str(x).strip()
        s = s.replace(',', '')
        return float(s)
    except Exception:
        return None

def _canon_str(x):
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canon_cat(x):
    s = _canon_str(x).lower()
    # normalize common truthy/falsey
    if s in {"y","yes","true","1","present","positive","pos"}:
        return "yes"
    if s in {"n","no","false","0","absent","negative","neg"}:
        return "no"
    return s

def build_label_dependencies(label_config: dict) -> tuple[dict, dict, list]:
    """Return (parent->children, child->parents, roots) from label_config.
    Supports either:
      label_config[<lid>]['gated_by'] = <parent or [parents]>
      label_config[<lid>]['gating_rules'] = list of rule dicts with keys:
          - parent: parent label_id (optional if 'gated_by' provided)
          - type: 'categorical'|'numeric'|'date'
          - op: one of 'in','notin','==','!=','>','>=','<','<=','between','outside','exists','notnull'
          - values/value/low/high/inclusive (depending on op)
      label_config[<lid>]['gating_logic'] = 'AND'|'OR' (default AND) across provided rules
    """
    parent_to_children: dict[str, list[str]] = {}
    child_to_parents: dict[str, list[str]] = {}
    if not isinstance(label_config, dict):
        return {}, {}, []
    for lid, cfg in label_config.items():
        if not isinstance(cfg, dict): 
            continue
        gb = cfg.get('gated_by')
        # Normalize to list
        parents = []
        if gb:
            if isinstance(gb, (list, tuple, set)):
                parents.extend([str(x) for x in gb])
            else:
                parents.append(str(gb))
        # Also scan rules to pick parent fields
        rules = cfg.get('gating_rules') or []
        for r in (rules if isinstance(rules, list) else [rules]):
            p = None
            if isinstance(r, dict):
                p = r.get('parent') or r.get('gated_by') or r.get('field')
            if p:
                p = str(p)
                if p not in parents:
                    parents.append(p)
        if parents:
            for p in parents:
                parent_to_children.setdefault(str(p), []).append(str(lid))
                child_to_parents.setdefault(str(lid), []).append(str(p))
    all_labels = {str(k) for k in label_config.keys()}
    roots = [lid for lid in all_labels if lid not in child_to_parents]
    return parent_to_children, child_to_parents, roots

def _check_rule(parent_value, parent_type: str, rule: dict) -> bool:
    """Evaluate a single gating rule for a parent value of a given type."""
    if parent_type == 'numeric':
        v = _to_number(parent_value)
        if v is None: 
            return False
        op = str(rule.get('op','in')).lower()
        if op == 'between':
            lo = _to_number(rule.get('low', None))
            hi = _to_number(rule.get('high', None))
            inc = bool(rule.get('inclusive', True))
            if lo is None or hi is None: 
                return False
            return (lo <= v <= hi) if inc else (lo < v < hi)
        elif op in ('>','>=','<','<=','==','!='):
            val = _to_number(rule.get('value', rule.get('values', [None])[0] if isinstance(rule.get('values'), list) else None))
            if val is None: 
                return False
            if op == '>':  return v >  val
            if op == '>=': return v >= val
            if op == '<':  return v <  val
            if op == '<=': return v <= val
            if op == '==': return v == val
            if op == '!=': return v != val
            return False
        elif op in ('in','notin'):
            vals = rule.get('values', [])
            vals = [ _to_number(x) for x in (vals if isinstance(vals, list) else [vals]) ]
            vals = [x for x in vals if x is not None]
            ok = v in vals
            return ok if op == 'in' else (not ok)
        elif op in ('exists','notnull'):
            return v is not None
        else:
            return False
    elif parent_type == 'date':
        d = _parse_date(parent_value)
        if d is None: 
            return False
        op = str(rule.get('op','>')).lower()
        if op == 'between':
            lo = _parse_date(rule.get('low'))
            hi = _parse_date(rule.get('high'))
            inc = bool(rule.get('inclusive', True))
            if lo is None or hi is None: 
                return False
            return (lo <= d <= hi) if inc else (lo < d < hi)
        elif op in ('>','>=','<','<=','==','!='):
            val = _parse_date(rule.get('value', rule.get('values', [None])[0] if isinstance(rule.get('values'), list) else None))
            if val is None: 
                return False
            if op == '>':  return d >  val
            if op == '>=': return d >= val
            if op == '<':  return d <  val
            if op == '<=': return d <= val
            if op == '==': return d == val
            if op == '!=': return d != val
            return False
        elif op in ('exists','notnull'):
            return d is not None
        else:
            return False
    else:
        s = _canon_cat(parent_value)
        op = str(rule.get('op','in')).lower()
        if op in ('in','notin'):
            vals = rule.get('values', [])
            vals = [ _canon_cat(x) for x in (vals if isinstance(vals, list) else [vals]) ]
            ok = s in vals
            return ok if op == 'in' else (not ok)
        elif op in ('==','!='):
            val = _canon_cat(rule.get('value', rule.get('values', [None])[0] if isinstance(rule.get('values'), list) else None))
            if op == '==': return s == val
            else: return s != val
        elif op in ('exists','notnull'):
            return len(s) > 0
        else:
            # default: treat as truthy yes
            return s in {'yes','present','true','1'}

def _gating_for_child(child_id: str, label_config: dict) -> dict:
    cfg = (label_config or {}).get(child_id, {}) if isinstance(label_config, dict) else {}
    rules = cfg.get('gating_rules') or []
    if isinstance(rules, dict):
        rules = [rules]
    logic = str(cfg.get('gating_logic','AND')).upper()
    parents_declared = cfg.get('gated_by')
    parents = []
    if parents_declared:
        parents = parents_declared if isinstance(parents_declared, list) else [parents_declared]
        # If no explicit rules, build a default categorical truthy rule for each parent
        if not rules:
            rules = [ {'parent': p, 'type': 'categorical', 'op': 'in', 'values': ['yes','present','true','1']} for p in parents ]
    # ensure parent field filled in rules
    for r in rules:
        if isinstance(r, dict) and 'parent' not in r:
            if parents:
                r['parent'] = parents[0]
    return {'rules': rules, 'logic': logic}

def evaluate_gating(child_id: str, unit_id: str, parent_preds: dict, label_types: dict, label_config: dict) -> bool:
    """Return True if child is eligible given current parent predictions."""
    g = _gating_for_child(child_id, label_config)
    rules = g.get('rules') or []
    if not rules:
        return True
    logic = g.get('logic','AND').upper()
    outcomes = []
    for r in rules:
        if not isinstance(r, dict): 
            continue
        p = str(r.get('parent'))
        p_type = label_types.get(p, 'categorical')
        val = parent_preds.get((unit_id, p))
        outcomes.append(_check_rule(val, p_type, r))
    if not outcomes:
        return True
    return all(outcomes) if logic != 'OR' else any(outcomes)

class FamilyLabeler:
    """Label entire family tree per unit, honoring parent-child gating."""
    def __init__(self, llm: LLMAnnotator, retriever: RAGRetriever, repo: DataRepository, label_config: dict, scCfg: SCJitterConfig, llmfirst_cfg: LLMFirstConfig):
        self.llm = llm
        self.retriever = retriever
        self.repo = repo
        self.label_config = label_config or {}
        self.scCfg = scCfg
        self.cfg = llmfirst_cfg
        self.parent_to_children, self.child_to_parents, self.roots = build_label_dependencies(self.label_config)
        
    
    def ensure_label_exemplars(self, rules_map: dict[str, str], K: int = 6):
        for lid, rules in (rules_map or {}).items():
            K_use = int(getattr(self.cfg, "exemplar_K", K) or K)
            # skip if already cached
            if self.retriever._get_label_query_embs(lid, rules, K_use) is not None:
                continue
            t = self._generate_label_exemplars(lid, rules, K_use)
            if t:
                self.retriever.set_label_exemplars(lid, rules, K_use, t)

    def _generate_label_exemplars(
        self,
        label_id: str,
        rules: str,
        K: int,
        options: list[str] | None = None,
    ) -> list[str]:
        """
        Generate K realistic clinical snippets (1–3 sentences) that satisfy the rule.
        - If the label has options (categorical/binary), request JSON objects with {"option","text"}
          and return a mix across options when feasible.
        - If no options (numeric/date/etc.), request JSON with {"snippets": [<string>, ...]}.
        - Azure OpenAI JSON mode (json_object); robust parsing and de-dup; returns List[str].
        """
        import time as _time
        import json as _json
    
        # Resolve options from label_config if not explicitly provided
        opts = options
        if not opts:
            try:
                lblcfgs = getattr(self.retriever, "label_configs", {}) or {}
                cfg = lblcfgs.get(label_id, {}) if isinstance(lblcfgs, dict) else {}
                raw = cfg.get("options")
                if isinstance(raw, list) and 2 <= len(raw) <= 26:
                    opts = [str(x) for x in raw]
            except Exception:
                opts = None
        has_opts = bool(opts)
    
        # ----- Build prompt (conditional on options) -----
        if has_opts:
            menu_text = "\n".join(f"- {o}" for o in opts)
            system = (
                "You write realistic clinical note snippets.\n"
                "Output policy:\n"
                "- Return ONLY a valid JSON object with a single key 'snippets', whose value is an array of objects.\n"
                "- Each object MUST have: {\"option\": <string>, \"text\": <string>}.\n"
                "- No extra keys, no commentary."
            )
            user = f"""
                Produce {K} diverse clinical note snippets (2-4 sentences each) that would *satisfy* this label rule.
                
                Label: {label_id}
                Rule: {rules or '(no extra rules)'}
                Options:
                {menu_text}
                
                Constraints:
                - Style: realistic, diverse, clinician-authored note sentences. No headers, no lists, no commentary.
                - No meta text (no “the patient meets criteria because...”, no references to “label” or “rule”).
                - Vary phrasing, timing (present/past/relative dates), and synonyms.
                - Keep PHI generic.
                
                Provide a MIX across options so each option appears at least once if feasible.
                Set "option" to exactly one of: {", ".join(opts)}.
                
                Return ONLY:
                {{
                  "snippets": [
                    {{"option": "<one of the options>", "text": "<2–4 sentence snippet>"}}
                  ]
                }}
                """.strip()
        else:
            system = (
                "You write realistic clinical note snippets.\n"
                "Output policy:\n"
                "- Return ONLY a valid JSON object with a single key 'snippets', whose value is an array of strings.\n"
                "- Do NOT include an 'option' field.\n"
                "- No extra keys, no commentary."
            )
            user = f"""
                Produce {K} diverse clinical note snippets (2–4 sentences each) that would *satisfy* this label rule.
                
                Label: {label_id}
                Rule: {rules or '(no extra rules)'}
                
                Constraints:
                - Style: realistic, diverse, clinician-authored note sentences. No headers, no lists, no commentary.
                - No meta text (no “the patient meets criteria because...”, no references to “label” or “rule”).
                - Vary phrasing, timing (present/past/relative dates), and synonyms.
                - Keep PHI generic.
                
                Return ONLY:
                {{
                  "snippets": ["<2–4 sentence snippet>", "..."]
                }}
                """.strip()
    
        # ----- Azure OpenAI JSON-mode call -----
        temp = float(getattr(self.cfg, "exemplar_temperature", 0.7) or 0.7)
        kwargs = dict(
            model=self.llm.cfg.model_name,
            temperature=temp,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
    
        # Optional RPM limiter (matches your pattern elsewhere)
        try:
            if getattr(self.llm.cfg, "rpm_limit", None):
                _since = getattr(self.llm, "_last_call", 0.0)
                _min_spacing = float(60.0 / float(self.llm.cfg.rpm_limit))
                now = _time.time()
                if now - _since < _min_spacing:
                    _time.sleep(_min_spacing - (now - _since))
        except Exception:
            pass
    
        try:
            resp = self.llm.client.chat.completions.create(**kwargs)
            setattr(self.llm, "_last_call", _time.time())
            ch = resp.choices[0]
            content = (
                getattr(getattr(ch, "message", None), "content", None)
                or getattr(ch, "content", None)
                or ""
            )
            # Capture the exact prompt + minimal context identifiers used this vote
            LLM_RECORDER.record("label_exemplar", {
                "label_id": label_id,
                "prompt": {"system": system, "user": user},
                "params": {
                    "temperature": temp,
                },
                "output": content,
            })
        except Exception:
            LLM_RECORDER.record("label_exemplar_error", {})
            return []
        
        # ----- Parse JSON -----
        texts_by_opt: dict[str, list[str]] = {}
        texts: list[str] = []
    
        def _clean_text(s):
            return str(s).strip()
    
        def _coerce_option(o: str) -> str:
            if not has_opts or not isinstance(o, str):
                return str(o or "")
            lut = {s.lower(): s for s in opts}
            return lut.get(o.lower(), str(o).strip())
    
        try:
            data = _json.loads(content) if isinstance(content, str) else content
            if has_opts:
                objs = (data or {}).get("snippets", []) if isinstance(data, dict) else []
                for it in objs:
                    if not isinstance(it, dict): 
                        continue
                    txt = _clean_text(it.get("text", ""))
                    opt = _coerce_option(it.get("option"))
                    if len(txt) > 20 and ((not has_opts) or (opt in (opts or []))):
                        texts_by_opt.setdefault(opt, []).append(txt)
            else:
                arr = (data or {}).get("snippets", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                for s in arr:
                    txt = _clean_text(s)
                    if len(txt) > 20:
                        texts.append(txt)
        except Exception:
            # Fallback: try to pull the first JSON array
            try:
                i = content.find("["); j = content.rfind("]")
                arr = _json.loads(content[i:j+1]) if (i != -1 and j != -1 and j > i) else []
                for s in arr:
                    txt = _clean_text(s)
                    if len(txt) > 20:
                        texts.append(txt)
            except Exception:
                pass
    
        # ----- De-dup and balance / limit to K -----
        if has_opts:
            # De-dup within each option bucket while preserving order
            for o in list(texts_by_opt.keys()):
                seen = set(); ded = []
                for t in texts_by_opt[o]:
                    if t not in seen:
                        seen.add(t); ded.append(t)
                texts_by_opt[o] = ded
    
            # Round-robin across options to get a mix, then fill remaining
            out: list[str] = []
            # one per option if feasible
            for o in opts:
                if len(out) >= K: break
                if texts_by_opt.get(o):
                    out.append(texts_by_opt[o].pop(0))
            # fill remaining round-robin
            while len(out) < K:
                progressed = False
                for o in opts:
                    if len(out) >= K: break
                    bucket = texts_by_opt.get(o) or []
                    if bucket:
                        out.append(bucket.pop(0)); progressed = True
                if not progressed:
                    break
            # if still short, append any leftover
            if len(out) < K:
                for o in opts:
                    for t in texts_by_opt.get(o, []):
                        if len(out) >= K: break
                        out.append(t)
                    if len(out) >= K: break
            return out[:K]
        else:
            # De-dup and cap
            seen = set(); out = []
            for t in texts:
                if t not in seen:
                    seen.add(t); out.append(t)
                if len(out) >= K:
                    break
            return out


    def _fc_probe(self, unit_id: str, label_id: str, label_type: str, label_rules: str, options: list[str]) -> dict:
        """Run a 1-token forced-choice micro-probe over provided options and return probs+entropy with reasoning."""
        # Map options to A/B/C/... tokens
        letters = [chr(ord('A') + i) for i in range(len(options))]
        option_lines = [f"{letters[i]}. {options[i]}" for i in range(len(options))]
        system = "You are a careful clinical information extraction assistant."
        ctx = "\n\n".join([c.get('text','') for c in self.retriever.retrieve_for_patient_label(unit_id, label_id, label_rules, topk_override=self.cfg.topk)])
        user = (
            f"Task: Choose the single best option for label '{label_id}' given the context snippets.\n" +
            (f"Label rules/hints: {label_rules}\n" if label_rules else "") +
            "Options:\n" + "\n".join(option_lines) + "\n" +
            "Return ONLY the option letter.\n\n" +
            "Context:\n" + ctx
        )
        kwargs = dict(
            model=self.llm.cfg.model_name,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
            max_tokens=1,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        # RPM limiter
        if self.llm.cfg.rpm_limit:
            import time as _time
            _since = getattr(self.llm, "_last_call", 0.0)
            _min_spacing = float(60.0 / self.llm.cfg.rpm_limit)
            now = _time.time()
            if now - _since < _min_spacing:
                _time.sleep(_min_spacing - (now - _since))
        t0 = time.time()
        resp = self.llm.client.chat.completions.create(**kwargs)
        dt = time.time() - t0
        setattr(self.llm, "_last_call", time.time())
        ch = resp.choices[0]
        lp = getattr(ch, "logprobs", None)
        items = getattr(lp, "content", None) or []
        letter_logps = {L: -1e9 for L in letters}
        for it in items:
            tops = getattr(it, "top_logprobs", None) or (it.get("top_logprobs") if isinstance(it, dict) else None)
            if not tops:
                continue
            for cand in tops:
                tok = getattr(cand, "token", None) or (cand.get("token") if isinstance(cand, dict) else None)
                val = getattr(cand, "logprob", None) or (cand.get("logprob") if isinstance(cand, dict) else None)
                if tok is None or val is None: 
                    continue
                t = str(tok).strip().strip('"').strip("'")
                if t and not t[0].isalnum():
                    t = t[1:]
                t = t[:1].upper() if t else ""
                if t in letter_logps:
                    letter_logps[t] = max(letter_logps[t], float(val))
        # convert to probs
        logits = np.array([letter_logps[L] for L in letters], dtype="float64")
        # stabilize
        m = logits.max()
        probs = np.exp(logits - m); probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)
        # entropy
        ent = float(-(probs * np.log(probs + 1e-12)).sum())
        # map back to option labels
        opt_probs = {options[i]: float(probs[i]) for i in range(len(options))}
        pred = options[int(np.argmax(probs))]
        
        try:
            LLM_RECORDER.record("forced_choice", {
                "unit_id": unit_id,
                "label_id": label_id,
                "label_type": label_type,
                "prompt": {"system": system, "user": user},
                "snippets": ctx,
                "fc_output": {"fc_probs": opt_probs, "fc_entropy": ent, "prediction": pred},
            })
        except Exception:
            pass

        return {"fc_probs": opt_probs, "fc_entropy": ent, "prediction": pred}

    def label_family_for_unit(self, unit_id: str, label_types: dict[str,str], per_label_rules: dict[str,str], *,
                              json_only: bool=False, json_n_consistency: int=1, json_jitter: bool=False) -> list[dict]:
        """Label all parents then gated children for a unit. Returns list of probe rows."""
        self.ensure_label_exemplars(per_label_rules)
        results = []
        # We'll process in a BFS: queue starts with roots, then expand children once parent gating satisfied
        # Maintain map of parent predictions for gating evaluation
        parent_preds: dict[tuple[str,str], Any] = {}
        # Determine an evaluation order by repeatedly scanning for labels whose parents are satisfied or none
        # But to preserve minimal disruption, we simply try roots then others with gating check at point of execution.
        all_labels = list({*self.roots, *[str(lid) for lid in label_types.keys()]})
        # We'll ensure unique and stable order
        seen = set()
        order = []
        for lid in self.roots + [l for l in all_labels if l not in self.roots]:
            if lid not in seen:
                seen.add(lid); order.append(lid)
        for lid in order:
            ltype = label_types.get(lid, 'categorical')
            rules = per_label_rules.get(lid, "")
            # if this label is gated (is a child), check gating
            allowed = evaluate_gating(lid, unit_id, parent_preds, label_types, self.label_config)
            if not allowed:
                # record gated-out for transparency but do not include in probe_df to avoid selection noise
                continue
            # Build a small context
            ctx = self.retriever.retrieve_for_patient_label(unit_id, lid, rules, topk_override=self.cfg.topk)
            # Decide FC vs JSON
            opts = _options_for_label(lid, ltype, getattr(self.retriever, 'label_configs', {}))
            used_fc = False
            row = {"unit_id": unit_id, "label_id": lid, "label_type": ltype, "rag_context": ctx}
            
            if json_only:
                res = self.llm.annotate(unit_id, lid, ltype, rules, ctx, n_consistency=max(1, int(json_n_consistency)), jitter_params=bool(json_jitter))
                row["prediction"] = res.get("prediction")
                row["consistency"] = res.get("consistency_agreement")
                row["runs"] = res.get("runs")
                row["U"] = (1.0 - float(row["consistency"])) if row.get("consistency") is not None else None
                # Prediction value for parent gating
                parent_preds[(unit_id, lid)] = row.get("prediction")
            else:
                if ltype in ('categorical','binary') or (opts is not None and self.cfg.fc_enable):
                    if opts is None:
                        # degrade to JSON if no options configured
                        pass
                    else:
                        try:
                            fc = self._fc_probe(unit_id, lid, ltype, rules, opts)
                            row.update(fc); used_fc = True
                            # Uncertainty from entropy
                            row["U"] = float(fc.get("fc_entropy", np.nan))
                            # Prediction value for parent gating
                            parent_preds[(unit_id, lid)] = fc.get("prediction")
                        except Exception as e:
                            # fallback to JSON if FC fails
                            used_fc = False
                if not used_fc:
                    # JSON with self-consistency if configured
                    res = self.llm.annotate(unit_id, lid, ltype, rules, ctx, n_consistency=self.llm.cfg.n_consistency, jitter_params=True)
                    row["prediction"] = res.get("prediction")
                    row["consistency"] = res.get("consistency_agreement")
                    row["runs"] = res.get("runs")
                    # Uncertainty = 1 - consistency
                    try:
                        row["U"] = float(1.0 - float(res.get("consistency_agreement") or 0.0))
                    except Exception:
                        row["U"] = np.nan
                    parent_preds[(unit_id, lid)] = row.get("prediction")
            results.append(row)
        return results

    def sample_units_for_probe_enriched(self, per_label_rules: dict[str, str], exclude_units: Optional[set[str]] = None,) -> List[str]:
        """
        CE-ranked enrichment that reuses retriever.retrieve_for_patient_label:
          For each parent P:
            - Randomly sample units from unseen pool
            - Call retrieve_for_patient_label(unit, P, rule) to get top-k patient snippets (already CE-ranked)
            - Unit score = max/mean CE score over top m snippets
          Then allocate quotas across parents, resolve overlaps by best parent, mix with uniform, and top up.
        """
        import numpy as np
    
        rng = np.random.default_rng()
        ex = set(exclude_units or [])
        
        #universe after exclusion
        all_units = [str(u) for u in sorted(self.repo.notes["unit_id"].unique().tolist()) if str(u) not in ex]
        if not all_units:
            return []

        # --- overall mix ---
        # uniform slice only from non-excluded
        n_total  = int(getattr(self.cfg, "n_probe_units", 100) or 100)
        mix      = float(getattr(self.cfg, "probe_enrichment_mix", 0.90) or 0.90)
        n_enr    = int(round(n_total * mix))
        n_unif   = max(0, n_total - n_enr)
        uniform_units = rng.choice(all_units, size=min(n_unif, len(all_units)), replace=False).tolist() if n_unif > 0 else []

    
        # --- parents & pool ---
        parents = list(getattr(self, "roots", []) or list((per_label_rules or {}).keys()))
        if not parents or not all_units:
            # fall back to uniform if nothing sensible
            print('falling back to uniform')
            return list(np.random.default_rng().choice(all_units, size=min(n_total, len(all_units)), replace=False)) if all_units else []
        parent_scores: dict[str, list[tuple[str, float]]] = {}
        best_parent: dict[str, tuple[str, float]] = {}
        
        # --- knobs for CE enrichment ---
        per_parent_sample = int(getattr(self.cfg, "probe_ce_unit_sample", 800) or 800)
        equalize         = bool(getattr(self.cfg, "probe_enrichment_equalize", True))
    
        # ---- score units per parent using your retrieve_for_patient_label ----
        for p in parents:
            rule = (per_label_rules or {}).get(p) or p
            # candidate pool excludes both uniform slice and excluded units
            pool_space = [u for u in all_units if (u not in set(uniform_units))]
            if not pool_space:
                parent_scores[p] = []
                continue
    
            per_parent_sample = int(getattr(self.cfg, "probe_ce_unit_sample", 800) or 800)
            cand_units = rng.choice(pool_space, size=min(per_parent_sample, len(pool_space)), replace=False).tolist()
            
            _progress_every = float(self.cfg.progress_min_interval_s or 1.0)
            unit_chunks, total_pairs = [], 0
            
            for uid in iter_with_bar(
                    step = "Enriching probe pool",
                    iterable = cand_units, 
                    total = len(cand_units), 
                    logger=LOGGER,
                    min_interval_s=_progress_every):
                if uid in ex:             # redundant but explicit
                    continue
                ctx = self.retriever.retrieve_for_patient_label(uid, p, rule, topk_override=int(getattr(self.cfg, "probe_ce_search_topk_per_unit", 24) or 24))
                if not ctx:
                    continue
                unit_chunks.append((uid, ctx))
                total_pairs += len(ctx)
    
            if not unit_chunks:
                parent_scores[p] = []
                continue
    
            # CE scores already in ctx['score']; aggregate top-m
            m = int(getattr(self.cfg, "probe_ce_rerank_m", 3) or 3)
            unit_rows = []
            for uid, ctx in unit_chunks:
                top = ctx[:max(1, m)]
                vals = [float(it.get("score", 0.0)) for it in top]
                unit_score = float(np.mean(vals)) if str(getattr(self.cfg, "probe_ce_unit_agg", "max")).lower() == "mean" else float(np.max(vals))
                unit_rows.append((uid, unit_score))
                prev = best_parent.get(uid)
                if (prev is None) or (unit_score > prev[1]):
                    best_parent[uid] = (p, unit_score)
    
            # keep only units best assigned to this parent
            parent_scores[p] = [(u, sc) for (u, sc) in unit_rows if best_parent.get(u, (None, -1e9))[0] == p]
    
        # ---- quotas across parents (equal or proportional) ----
        n_avail = sum(len(v) for v in parent_scores.values())
        n_enriched = min(n_enr, n_avail)
        quotas: dict[str, int] = {}
        if equalize:
            base = (n_enriched // max(1, len(parents)))
            quotas = {p: base for p in parents}
            rem = n_enriched - base * len(parents)
            if rem > 0:
                order = sorted(parents, key=lambda L: len(parent_scores.get(L, [])), reverse=True)
                for p in order[:rem]:
                    quotas[p] = quotas.get(p, 0) + 1
        else:
            import math
            tot = max(1, sum(len(parent_scores.get(p, [])) for p in parents))
            raw = {p: (len(parent_scores.get(p, [])) * n_enriched) / tot for p in parents}
            quotas = {p: int(math.floor(v)) for p, v in raw.items()}
            give = n_enriched - sum(quotas.values())
            if give > 0:
                frac = sorted(((p, raw[p] - quotas[p]) for p in parents), key=lambda x: x[1], reverse=True)
                for p, _ in frac[:give]:
                    quotas[p] += 1
    
        # ---- pick by CE score within each parent; merge with uniform; top up uniformly if needed ----
        chosen, used = [], set(uniform_units)
        for p in parents:
            pool = sorted(parent_scores.get(p, []), key=lambda t: t[1], reverse=True)
            need = int(quotas.get(p, 0))
            for (u, _sc) in pool:
                if u in used: 
                    continue
                chosen.append(u); used.add(u)
                if len(chosen) >= n_enriched or sum(1 for x in chosen if best_parent[x][0] == p) >= need:
                    break
    
        merged = (uniform_units + chosen)
        out, used = [], set()
        for u in merged:
            if u in used or u in ex:
                continue
            used.add(u); out.append(u)
            if len(out) >= n_total: break
        if len(out) < n_total:
            rest = [u for u in all_units if u not in used]
            if rest:
                extra = rng.choice(rest, size=min(n_total - len(out), len(rest)), replace=False).tolist()
                out.extend(extra)
        return out[:n_total]


    def sample_units_for_probe(self, exclude_units: Optional[set[str]] = None) -> List[str]:
        import numpy as np
        ex = set(exclude_units or [])
        n_total = int(getattr(self.cfg, "n_probe_units", 100) or 100)
        uids = [str(u) for u in sorted(self.repo.notes["unit_id"].unique().tolist()) if str(u) not in ex]
        if not uids:
            return []
        rng = np.random.default_rng()
        return rng.choice(uids, size=min(n_total, len(uids)), replace=False).tolist()


    def probe_units_label_tree(self, enrich: bool, label_types: dict[str,str], 
                               per_label_rules: dict[str,str], seed: Optional[int]=None,
                               exclude_units: Optional[set[str]] = None) -> pd.DataFrame:
        ex = set(exclude_units or [])
        if enrich:
            sample = self.sample_units_for_probe_enriched(per_label_rules, exclude_units=ex)
        else: sample = self.sample_units_for_probe(exclude_units=ex)
        rows = []
        _progress_every = float(self.cfg.progress_min_interval_s or 1.0)
        for uid in iter_with_bar(
                step="LLM probe",
                iterable=sample,
                total=len(sample),
                logger=LOGGER,
                min_interval_s=_progress_every):
            rows.extend(self.label_family_for_unit(uid, label_types, per_label_rules))

        df = pd.DataFrame(rows)
        # Compute U column if using FC or JSON: ensure presence
        if "U" not in df.columns:
            df["U"] = np.nan
        # Harmonize U like LLMProber.probe_unseen does: FC entropy preferred, else 1-consistency
        if "fc_entropy" in df.columns:
            # keep U as already set; ensure fallback for rows lacking fc
            idx = df["U"].isna()
            if "consistency" in df.columns:
                df.loc[idx, "U"] = 1.0 - pd.to_numeric(df.loc[idx, "consistency"], errors="coerce").fillna(0.0)
        elif "consistency" in df.columns:
            df["U"] = 1.0 - pd.to_numeric(df["consistency"], errors="coerce").fillna(0.0)
        # Return aligned columns
        return df

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
                              target_n: int,
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
        row = row.copy()
        row["kcenter_rank_global"] = rank_idx
        chosen_rows.append(row); cnt[row.label_id] += 1
        if len(chosen_rows) >= remaining_needed: break

    if not chosen_rows: return selected
    chosen_df = pd.DataFrame([r._asdict() if hasattr(r,"_asdict") else dict(r) for r in chosen_rows])
    return pd.concat([selected, chosen_df], ignore_index=True)


def _mmr_select_simple(vecs, rel_scores, k, lam=0.7, preselected=None):
    """Greedy MMR: argmax lam*rel – (1–lam)*max_sim_to_selected (no label caps)."""
    if k <= 0 or vecs is None or len(vecs) == 0:
        return []
    V = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    rel = np.asarray(rel_scores, dtype="float32")
    N = V.shape[0]
    selected = []
    if preselected is not None and getattr(preselected, "size", 0):
        P = preselected / (np.linalg.norm(preselected, axis=1, keepdims=True) + 1e-12)
        sim_pre = (V @ P.T).max(axis=1)
    else:
        sim_pre = np.zeros(N, dtype="float32")
    used = np.zeros(N, dtype=bool)
    for _ in range(min(k, N)):
        if selected:
            S = V[selected, :]
            sim_sel = (V @ S.T).max(axis=1)
            sim = np.maximum(sim_pre, sim_sel)
        else:
            sim = sim_pre
        score = lam * rel - (1.0 - lam) * sim
        score[used] = -1e9
        i = int(np.argmax(score))
        if score[i] <= -1e8:
            break
        selected.append(i)
        used[i] = True
    return selected

def build_diversity_bucket(
    unseen_pairs,
    already_selected,
    n_div,
    pooler,
    retriever,
    rules_map,
    label_types,
    label_config=None,
    rag_k=4,
    min_rel_quantile=0.30,
    mmr_lambda=0.7,
    sample_cap=2000,
    adaptive_relax=True,
    relax_steps=(0.20, 0.10, 0.05),
    pool_factor=4.0,
    use_proto=False,                      # if True, pooler prototype beats exemplars
    *,
    family_labeler=None,                  # NEW: for ensure_label_exemplars
    ensure_exemplars: bool = True,        # NEW: call ensure_label_exemplars first
    exclude_units: set[str] | None = None # (keep if you already added this earlier)
):
    """
    Diversity bucket without CE gating:
      label-aware MMR using rel-to-anchor where anchor is:
        (1) pooler prototype (if use_proto=True),
        (2) else exemplar centroid (preferred fallback),
        (3) else rule-text query embedding.
      Then pooled candidates -> unit-level k-center.
    """
    import numpy as np
    import pandas as pd

    # --- ensure exemplars once (cheap if already cached) ---
    if ensure_exemplars and family_labeler is not None:
        try:
            K_use = int(getattr(getattr(family_labeler, "cfg", None), "exemplar_K", 6) or 6)
            family_labeler.ensure_label_exemplars(rules_map, K=K_use)
        except Exception:
            pass

    def _l2(v):
        v = np.asarray(v, dtype="float32")
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def _cos(a, b): return float(np.dot(_l2(a), _l2(b)))

    def _kcenter_greedy(U: np.ndarray, k: int, seed_indices=None) -> list[int]:
        if U.size == 0 or k <= 0: return []
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        n = Un.shape[0]
        sel, dmin = [], np.ones(n) * np.inf
        if seed_indices:
            seeds = [s for s in seed_indices if 0 <= s < n]
            if seeds:
                sel.extend(seeds)
                sims = Un @ Un[seeds].T
                dmin = 1 - sims.max(axis=1); dmin[seeds] = 0.0
        if not sel:
            s0 = int(np.random.randint(0, n))
            sel.append(s0); dmin = 1 - (Un @ Un[s0])
        while len(sel) < min(k, n):
            i = int(np.argmax(dmin))
            sel.append(i)
            dmin = np.minimum(dmin, 1 - (Un @ Un[i]))
        return sel

    def _equal_quota(group_sizes: dict[str, int], total: int) -> dict[str, int]:
        labs = list(group_sizes.keys())
        if not labs or total <= 0: return {lab: 0 for lab in labs}
        base, rem = total // len(labs), total % len(labs)
        q = {lab: min(group_sizes[lab], base) for lab in labs}
        order = sorted(labs, key=lambda x: group_sizes[x] - q[x], reverse=True)
        for lab in order:
            if rem <= 0: break
            if q[lab] < group_sizes[lab]: q[lab] += 1; rem -= 1
        while rem > 0:
            progressed = False
            for lab in order:
                if q[lab] < group_sizes[lab]:
                    q[lab] += 1; rem -= 1; progressed = True
                    if rem == 0: break
            if not progressed: break
        return q

    # --- early exits & candidate sampling ---
    if n_div <= 0:
        return pd.DataFrame(columns=["unit_id","label_id","label_type","selection_reason"])

    rem_all = [(u,l) for (u,l) in unseen_pairs if (u,l) not in set(already_selected)]
    if exclude_units:
        rem_all = [(u,l) for (u,l) in rem_all if u not in exclude_units]
    np.random.shuffle(rem_all)
    rem_all = rem_all[: max(n_div*8, min(sample_cap, len(rem_all)))]
    if not rem_all:
        return pd.DataFrame(columns=["unit_id","label_id","label_type","selection_reason"])

    rem_df = pd.DataFrame([{"unit_id": u, "label_id": str(l), "label_type": label_types.get(l,"text")}
                           for (u,l) in rem_all])

    # --- anchor cache (label -> vector) ---
    proto_cache: dict[str, np.ndarray] = {}

    def _label_anchor(lid: str) -> np.ndarray:
        """
        (1) pooler prototype (if use_proto),
        (2) exemplar centroid if cached,
        (3) rule-text query embedding fallback.
        """
        if lid in proto_cache:
            return proto_cache[lid]

        proto = None

        # 1) pooler prototype (explicit opt-in)
        if hasattr(pooler, "label_prototype") and use_proto:
            try:
                proto = pooler.label_prototype(lid, retriever, rules_map.get(lid, ""))
            except Exception:
                proto = None

        # 2) exemplar centroid (preferred fallback)
        if proto is None:
            Q = None
            try:
                K = int(getattr(getattr(retriever, "cfg", None), "exemplar_K", 6) or 6)
                getQ = getattr(retriever, "_get_label_query_embs", None)
                if callable(getQ):
                    Q = getQ(lid, rules_map.get(lid, ""), K)
            except Exception:
                Q = None
            if Q is not None and getattr(Q, "ndim", 1) == 2 and Q.shape[0] > 0:
                proto = Q.mean(axis=0)

        # 3) rule-text embedding fallback
        if proto is None:
            q = retriever._build_query(lid, rules_map.get(lid, ""))
            proto = retriever.store._embed([q])[0]

        proto_cache[lid] = np.asarray(proto, dtype="float32")
        return proto_cache[lid]

    # --- build label-aware vectors & rel-to-anchor ---
    vecs, rels = [], []
    _progress_every = float(family_labeler.cfg.progress_min_interval_s or 1)
    for r in iter_with_bar(
            step="Diversity: pooling vectors",
            iterable=rem_df.itertuples(index=False),
            total=len(rem_df),
            logger=LOGGER,
            min_interval_s=_progress_every):
        v = pooler.pooled_vector(r.unit_id, r.label_id, retriever, rules_map.get(r.label_id, ""))
        vecs.append(v)
        rels.append(_cos(v, _label_anchor(r.label_id)))
    rem_df["vec"] = vecs
    rem_df["rel"] = np.array(rels, dtype="float32")

    # --- per-label quantile keep (with relax) ---
    def _keep_by_q(df, q):
        kept = []
        for lid, g in df.groupby("label_id", sort=False):
            if g.empty: continue
            thr = float(np.quantile(g["rel"].to_numpy(), q)) if len(g) > 1 else g["rel"].min()
            kept.append(g[g["rel"] >= thr])
        return pd.concat(kept, ignore_index=True) if kept else df.head(0)

    if min_rel_quantile is not None:
        gated = _keep_by_q(rem_df, float(min_rel_quantile))
        if adaptive_relax and len(gated) < n_div:
            for q in relax_steps:
                gated = _keep_by_q(rem_df, float(q))
                if len(gated) >= n_div:
                    break
    else:
        gated = rem_df
    if gated.empty:
        return gated

    # --- per-label MMR → pooled candidates ---
    pool_total = int(min(len(gated), max(n_div, int(n_div * pool_factor))))
    sizes = {lid: len(g) for lid, g in gated.groupby("label_id", sort=False)}
    quotas = _equal_quota(sizes, pool_total)

    pools, sel_pairs = [], set()
    preV_by_label: dict[str, np.ndarray] = {}  # keep empty unless you add warm-start vectors

    for lid, g in gated.groupby("label_id", sort=False):
        k_lab = quotas.get(lid, 0)
        if k_lab <= 0 or g.empty: continue
        V   = np.stack(g["vec"].to_list()).astype("float32")
        rel = g["rel"].to_numpy().astype("float32")
        preV = preV_by_label.get(lid)
        order = _mmr_select_simple(V, rel, k=k_lab, lam=mmr_lambda, preselected=preV)
        choice = g.iloc[order].head(k_lab) if order else g.sort_values("rel", ascending=False).head(k_lab)
        for r in choice.itertuples(index=False):
            key = (r.unit_id, r.label_id)
            if key in sel_pairs: continue
            sel_pairs.add(key); pools.append(r)

    if not pools:
        add_df = gated.sort_values("rel", ascending=False).head(n_div).copy()
        add_df["selection_reason"] = "diversity_toprel_fallback"
        return add_df[["unit_id","label_id","label_type","selection_reason"]]

    pool_df = pd.DataFrame(pools)
    if len(pool_df) > pool_total:
        pool_df = pool_df.sample(pool_total).reset_index(drop=True)

    # --- unit-level k-center over averaged vectors ---
    by_unit = {}
    for r in pool_df.itertuples(index=False):
        by_unit.setdefault(r.unit_id, {"labels": [], "vecs": [], "best_label": None, "best_rel": -1.0})
        by_unit[r.unit_id]["labels"].append(r.label_id)
        by_unit[r.unit_id]["vecs"].append(np.asarray(r.vec, dtype="float32"))
        if float(r.rel) > by_unit[r.unit_id]["best_rel"]:
            by_unit[r.unit_id]["best_rel"] = float(r.rel)
            by_unit[r.unit_id]["best_label"] = r.label_id

    units = list(by_unit.keys())
    if not units:
        return pd.DataFrame(columns=["unit_id","label_id","label_type","selection_reason"])

    U = np.stack([np.mean(np.stack(v["vecs"], axis=0), axis=0) for v in by_unit.values()], axis=0)
    idx_map = {u: i for i, u in enumerate(units)}
    seed_units = {u for (u, _l) in already_selected}
    seeds = [idx_map[u] for u in seed_units if u in idx_map]

    picks = _kcenter_greedy(U, k=n_div, seed_indices=seeds)

    rows = []
    for i in picks:
        u = units[i]
        lab = by_unit[u]["best_label"]
        rows.append({
            "unit_id": u,
            "label_id": lab,
            "label_type": label_types.get(lab, "text"),
            "selection_reason": "diversity_mmr_kcenter"
        })
    return pd.DataFrame(rows)



def direct_uncertainty_selection(
    probe_df: pd.DataFrame,
    n_unc: int,
    *,
    label_col: str = "label_id",
    uncertainty_col: str = "U",
    higher_is_more_uncertain: bool = True,   # e.g., U = entropy → higher = more uncertain
    select_most_certain: bool = False,       # flip to pick the most certain items
    add_selection_reason: bool = True,
) -> pd.DataFrame:
    """
    Fill purely by round-robin across labels:
      1) Sort within each label by the criterion (uncertainty or certainty).
      2) Build a FIFO queue per label.
      3) Pop one from each label in turn until n_unc is reached (or queues empty).

    Notes:
      - NaN uncertainty sinks to the end for the chosen ordering.
    """
    # Basic guards
    if probe_df is None or not isinstance(probe_df, pd.DataFrame) or probe_df.empty or n_unc <= 0:
        return (probe_df.head(0).copy()
                if isinstance(probe_df, pd.DataFrame) else pd.DataFrame())

    df = probe_df.copy()
    if label_col not in df.columns or uncertainty_col not in df.columns:
        raise ValueError(f"probe_df must contain '{label_col}' and '{uncertainty_col}'")

    # Normalize uncertainty column and handle NaNs so they sink.
    u = pd.to_numeric(df[uncertainty_col], errors="coerce")

    # asc_u=True means "smaller score is better".
    # We want:
    #   - Most-uncertain:  desc if higher_is_more_uncertain else asc
    #   - Most-certain:    asc  if higher_is_more_uncertain else desc
    asc_u = (select_most_certain == higher_is_more_uncertain)

    # Fill NaNs so they sort to the end given asc/desc choice
    u = u.fillna(np.inf if asc_u else -np.inf)

    # Tiny jitter for stable tie-breaking (optional)
    rng = np.random.default_rng()
    u = u + (rng.random(len(u)) * 1e-9)

    df["_score"] = u

    # Build per-label queues: each label sorted by the criterion
    # (We sort by label then by score so groupby preserves label order;
    #  optionally shuffle label order 
    per_label_sorted = df.sort_values([label_col, "_score"], ascending=[True, asc_u])
    label_groups = list(per_label_sorted.groupby(label_col, sort=False))

    # Convert to queues (row-index lists)
    queues = {lab: grp.index.to_list() for lab, grp in label_groups}
    labels_order = [lab for lab, _ in label_groups if len(queues.get(lab, [])) > 0]

    # Optional: shuffle label start order for fairness when a seed is provided
    rng.shuffle(labels_order)

    # Round-robin selection
    chosen_idx = []
    need = int(n_unc)

    while need > 0 and labels_order:
        progressed = False
        new_order = []
        for lab in labels_order:
            q = queues.get(lab, [])
            if q:
                chosen_idx.append(q.pop(0))
                need -= 1
                progressed = True
                if q:  # keep label in the rotation if it still has items
                    new_order.append(lab)
            if need <= 0:
                break
        labels_order = new_order if progressed else []

    if not chosen_idx:
        out = df.head(0).copy()
        if add_selection_reason and "selection_reason" not in out.columns:
            out["selection_reason"] = ("model_certain_direct" if select_most_certain
                                       else "model_uncertain_direct")
        return out

    chosen = df.loc[chosen_idx].copy()

    # Decorate
    reason = "model_certain_direct" if select_most_certain else "model_uncertain_direct"
    if add_selection_reason and "selection_reason" not in chosen.columns:
        chosen["selection_reason"] = reason

    prefix = "certain" if select_most_certain else "uncertain"
    # Per-label rank by the criterion (for inspection)
    chosen[f"{prefix}_rank_per_label"] = (
        chosen.sort_values([label_col, "_score"], ascending=[True, asc_u])
              .groupby(label_col, sort=False).cumcount() + 1
    )
    # Global rank by the criterion
    chosen[f"{prefix}_rank_global"] = (
        chosen.sort_values("_score", ascending=asc_u)
              .reset_index(drop=True).index + 1
    )

    # Return ordered by the chosen criterion (you could return in selection order if preferred)
    out = (chosen.sort_values("_score", ascending=asc_u)
                 .drop(columns=["_score"])
                 .reset_index(drop=True))
    return out



# ------------------------------
# Orchestrator
def _detect_device():
    import os
    if os.getenv("CPU_ONLY", "0") == "1":
        return "cpu"
    try:
        import torch
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

# ------------------------------

class ActiveLearningLLMFirst:
    def __init__(self, paths: Paths, cfg: OrchestratorConfig, label_config: Optional[dict]=None):
        import os
        self.paths = paths; self.cfg = cfg
        notes_df = read_table(paths.notes_path); ann_df = read_table(paths.annotations_path)
        self.repo = DataRepository(notes_df, ann_df)

        embed_name = os.getenv("MED_EMBED_MODEL_NAME")
        rerank_name = os.getenv("RERANKER_MODEL_NAME")
        device = _detect_device()
        embedder = SentenceTransformer(embed_name, device=device)
        reranker = CrossEncoder(rerank_name, device=device)
        emb_bs = int(os.getenv('EMB_BATCH', '32' if device == "cpu" else "64"))
        rr_bs = int(os.getenv('RERANK_BATCH', '16' if device == "cpu" else "64"))
        self.models = Models(embedder, reranker, device=device, emb_batch=emb_bs, rerank_batch=rr_bs)

        self.store = EmbeddingStore(self.models, cache_dir=self.paths.cache_dir, normalize=self.cfg.rag.normalize_embeddings)
        self.rag = RAGRetriever(self.store, self.models, self.cfg.rag, label_configs=label_config or {}, notes_by_doc=self.repo.notes_by_doc(), repo=self.repo)
        self.llm = LLMAnnotator(self.cfg.llm, self.cfg.scjitter, cache_dir=self.paths.cache_dir)
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
        """
        Disagreement bucket where apportionment targets come from LAST-ROUND VALID DISAGREEMENTS:
          • parents: all parent labels with last-round disagreement >= threshold
          • children: last-round disagreement >= threshold AND parent's aggregated label passes gating by consensus
        Then parent-first per label, fill with children for that label, and top-up (children-first).
        """
        import pandas as pd
        from collections import defaultdict
    
        # ---------- helpers ----------
        def _apportion_by_mix_fair(counts: pd.Series, total: int) -> Dict[str, int]:
            """
            Hamilton apportionment with fairness base:
              - If total >= #present labels, give each present label 1 base seat.
              - Distribute remainder by largest remainders of (counts / sum(counts)).
            """
            counts = counts[counts > 0]
            if total <= 0 or counts.empty:
                return {}
            labels = counts.index.tolist()
            L = len(labels)
    
            base_each = 1 if total >= L else 0
            base = {str(l): base_each for l in labels}
            remaining = total - base_each * L
            if remaining < 0:
                base = {str(l): 0 for l in labels}
                remaining = total
    
            weights = counts / counts.sum()
            exact = weights * remaining
            floors = exact.apply(int)
            tgt = {str(l): int(base[str(l)] + floors.get(l, 0)) for l in labels}
            give = int(remaining - floors.sum())
            if give > 0:
                rema = (exact - floors).sort_values(ascending=False)
                for l in rema.index[:give]:
                    tgt[str(l)] = tgt.get(str(l), 0) + 1
            return {k: int(max(0, v)) for k, v in tgt.items()}
    
        def _select_for_label(pool_df: pd.DataFrame, label_id: str, k: int) -> pd.DataFrame:
            """Run per-label k-center on a single-label subset and return ≤k rows."""
            if k <= 0:
                return pool_df.head(0)
            sub = pool_df[pool_df["label_id"].astype(str) == str(label_id)]
            if sub.empty:
                return sub
            sel = per_label_kcenter(sub, self.pooler, self.rag, rules_map, per_label_quota=int(k))
            return sel.head(k)
    
        # ---------- expand candidates (assumes seed-time gating in expander) ----------
        expander = DisagreementExpander(self.cfg.disagree, self.repo, self.rag, self.label_config)
        expanded = expander.expand(rules_map, seen_pairs)
        if expanded is None or expanded.empty:
            return expanded
    
        # ---------- dependencies & (fail-open) gating for expanded pool ----------
        try:
            parent_to_children, child_to_parents, roots = build_label_dependencies(self.label_config)
        except Exception:
            parent_to_children, child_to_parents, roots = {}, {}, []
        roots = set(str(x) for x in (roots or []))
        consensus = self.repo.last_round_consensus()  # {(unit_id,label_id)-> str}
        types = self.repo.label_types()
    
        def _gate_ok_expanded(uid: str, lid: str) -> bool:
            """
            Fail-open for expanded pool:
              - parents: True
              - children: if ANY parent consensus exists, evaluate; if none, allow.
            """
            parents = child_to_parents.get(str(lid), [])
            if not parents:
                return True
            parent_preds = {}
            have_any = False
            for p in parents:
                key = (str(uid), str(p))
                val = consensus.get(key, None)
                parent_preds[key] = val
                if val is not None and str(val).strip() != "":
                    have_any = True
            if not have_any:
                return True
            return evaluate_gating(str(lid), str(uid), parent_preds, types, self.label_config)
    
        df = expanded.copy()
        df["unit_id"] = df["unit_id"].astype(str)
        df["label_id"] = df["label_id"].astype(str)
        df["is_root_parent"] = df["label_id"].isin(roots)
        df = df[df["is_root_parent"] | df.apply(lambda r: _gate_ok_expanded(r["unit_id"], r["label_id"]), axis=1)].reset_index(drop=True)
        if df.empty:
            return df
    
        # ---------- APPORTIONMENT FROM LAST-ROUND *VALID DISAGREEMENTS* ----------
        # Last round disagreement table
        last_dis = self.repo.reviewer_disagreement(round_policy='last')  # has 'disagreement_score'
        thr = float(getattr(self.cfg.disagree, "high_entropy_threshold", 0.5))
        if "disagreement_score" in last_dis.columns:
            valid = last_dis[last_dis["disagreement_score"] >= thr].copy()
        else:
            valid = last_dis.copy()  # fallback if column name differs
    
        # Count by label from last-round disagreements:
        #   - parents: any valid disagreement on that parent label
        #   - children: valid disagreement AND gate(child) passes by consensus
        valid["unit_id"] = valid["unit_id"].astype(str)
        valid["label_id"] = valid["label_id"].astype(str)
    
        def _gate_by_cons(uid: str, lid: str) -> bool:
            parents = child_to_parents.get(str(lid), [])
            if not parents:
                return True
            parent_preds = {(uid, str(p)): consensus.get((uid, str(p)), None) for p in parents}
            return evaluate_gating(str(lid), uid, parent_preds, types, self.label_config)
    
        counts_dict: Dict[str, int] = {}
        present_labels = set(df["label_id"].unique().tolist())  # only apportion among labels actually available in this round
        for lid, grp in valid.groupby("label_id"):
            lid = str(lid)
            if lid not in present_labels:
                continue
            if lid in roots or not child_to_parents.get(lid, []):
                cnt = int(len(grp))
            else:
                # children: only those units whose parent(s) pass the gate by LAST round consensus
                c = 0
                for uid in grp["unit_id"].unique():
                    if _gate_by_cons(str(uid), lid):
                        c += 1
                cnt = c
            counts_dict[lid] = cnt
    
        counts_series = pd.Series(counts_dict, dtype="int64")
        n_dis = int(self.cfg.select.batch_size * self.cfg.select.pct_disagreement)
    
        # Edge case: if no counts (e.g., all disagreement below threshold), fall back to pool frequencies
        if counts_series.sum() <= 0:
            counts_series = df["label_id"].value_counts()
    
        target_per_label = _apportion_by_mix_fair(counts_series, n_dis)
    
        # Clamp targets by actual availability in df
        avail = df["label_id"].value_counts()
        for lid in list(target_per_label.keys()):
            target_per_label[lid] = int(min(target_per_label[lid], int(avail.get(lid, 0))))
    
        # ---------- parent-first per label, then children to fill ----------
        parents_pool = df[df["is_root_parent"]].copy()
        children_pool = df[~df["is_root_parent"]].copy()
    
        selected_parts = []
        taken_by_label: Dict[str, int] = defaultdict(int)
    
        # parents
        par_chunks = []
        for lid, tgt in target_per_label.items():
            if tgt <= 0:
                continue
            s = _select_for_label(parents_pool, lid, tgt)
            if not s.empty:
                s = s.copy()
                s["selection_reason"] = "disagreement_parent"
                par_chunks.append(s)
                taken_by_label[lid] += int(len(s))
        if par_chunks:
            selected_parts.append(pd.concat(par_chunks, ignore_index=True))
    
        # children per label (fill to target)
        ch_chunks = []
        for lid, tgt in target_per_label.items():
            rem = tgt - int(taken_by_label.get(lid, 0))
            if rem <= 0:
                continue
            s = _select_for_label(children_pool, lid, rem)
            if not s.empty:
                s = s.copy()
                s["selection_reason"] = "disagreement_child"
                ch_chunks.append(s)
                taken_by_label[lid] += int(len(s))
        if ch_chunks:
            selected_parts.append(pd.concat(ch_chunks, ignore_index=True))
    
        if not selected_parts:
            return df.head(0)
    
        sel = pd.concat(selected_parts, ignore_index=True)
    
        # ---------- global top-up (ADD ONLY) to n_dis: children-first, then parents ----------
        if len(sel) < n_dis:
            remaining_pool = df.merge(sel[["unit_id","label_id"]], on=["unit_id","label_id"], how="left", indicator=True)
            remaining_pool = remaining_pool[remaining_pool["_merge"] == "left_only"].drop(columns=["_merge"])
    
            need = n_dis - len(sel)
    
            # children first
            if need > 0:
                child_rem = remaining_pool[~remaining_pool["is_root_parent"]]
                if not child_rem.empty:
                    fill = merge_with_global_kcenter(
                        sel, child_rem, self.pooler, self.rag, rules_map,
                        target_n=len(sel) + min(need, len(child_rem))
                    )
                    added = fill.merge(sel[["unit_id","label_id"]], on=["unit_id","label_id"], how="left", indicator=True)
                    added = added[added["_merge"] == "left_only"].drop(columns=["_merge"])
                    if not added.empty:
                        added = added.copy()
                        added["selection_reason"] = "disagreement_child_topup"
                        sel = pd.concat([sel, added], ignore_index=True)
                        need = n_dis - len(sel)
    
            # then parents
            if need > 0:
                par_rem = remaining_pool[remaining_pool["is_root_parent"]]
                if not par_rem.empty:
                    fill = merge_with_global_kcenter(
                        sel, par_rem, self.pooler, self.rag, rules_map,
                        target_n=len(sel) + min(need, len(par_rem))
                    )
                    added = fill.merge(sel[["unit_id","label_id"]], on=["unit_id","label_id"], how="left", indicator=True)
                    added = added[added["_merge"] == "left_only"].drop(columns=["_merge"])
                    if not added.empty:
                        added = added.copy()
                        added["selection_reason"] = "disagreement_parent_topup"
                        sel = pd.concat([sel, added], ignore_index=True)
    
        sel["label_type"] = sel["label_id"].map(lambda x: label_types.get(x, "text"))
        return sel

    def build_llm_uncertain_bucket(self, label_types: Dict[str,str], rules_map: Dict[str,str], exclude_units: Optional[set[str]]=None) -> pd.DataFrame:
        fam = FamilyLabeler(self.llm, self.rag, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
        probe_df = fam.probe_units_label_tree(self.cfg.llmfirst.enrich, label_types, rules_map, exclude_units = exclude_units)
        # Save the raw probe for inspection / later reuse
        safe_cols = [c for c in ["fc_probs","rag_context","why","runs"] if c in probe_df.columns]
        _jsonify_cols(probe_df, safe_cols).to_parquet(os.path.join(self.paths.outdir, "llm_probe.parquet"), index=False)
        # select most uncertain
        n_unc = int(self.cfg.select.batch_size * self.cfg.select.pct_uncertain)
        unc_df = direct_uncertainty_selection(probe_df, n_unc, select_most_certain=False)
        return unc_df

    def build_llm_certain_bucket(self, label_types: Dict[str,str], rules_map: Dict[str,str], exclude_units: Optional[set[str]] = None) -> pd.DataFrame:
        p = os.path.join(self.paths.outdir, "llm_probe.parquet")
        if os.path.exists(p):
            probe_df = pd.read_parquet(p)
        else:
            fam = FamilyLabeler(self.llm, self.rag, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
            probe_df = fam.probe_units_label_tree(self.cfg.llmfirst.enrich, label_types, rules_map, exclude_units = exclude_units)
        n_cer = int(self.cfg.select.batch_size * self.cfg.select.pct_easy_qc)
        cer_df = direct_uncertainty_selection(probe_df, n_cer,select_most_certain=True)
        return cer_df
    
    def top_off_random(
        self,
        current_sel: pd.DataFrame,
        unseen_pairs: list[tuple[str, str]],
        label_types: dict[str, str],
        target_n: int,
    ) -> pd.DataFrame:
        """
        Add-only random top-off from unseen_pairs to reach target_n.
        Returns an updated DataFrame with new rows labeled 'random_topoff'.
        """
        import random
        import pandas as pd
    
        sel = current_sel.copy()
        need = int(target_n) - len(sel)
        if need <= 0:
            return sel
    
        # Exclude already-selected (unit,label) pairs
        taken = set(zip(sel["unit_id"].astype(str), sel["label_id"].astype(str)))
        cand = [(str(u), str(l)) for (u, l) in unseen_pairs if (str(u), str(l)) not in taken]
        if not cand:
            return sel
    
        rng = random.Random()
        rng.shuffle(cand)
    
        take = cand[:need]
        if not take:
            return sel
    
        add = pd.DataFrame(
            [{
                "unit_id": u,
                "label_id": l,
                "label_type": label_types.get(l, "text"),
                "selection_reason": "random_topoff",
            } for (u, l) in take]
        )
    
        return pd.concat([sel, add], ignore_index=True)

    
    def run(self):
        import time, os, pandas as pd
        
        t0 = time.time()
        
        run_id = time.strftime("%Y%m%d-%H%M%S")
        LLM_RECORDER.start(
            outdir=self.paths.outdir,
            run_id=run_id,
            meta={
                "model": getattr(self.cfg.llm, "model_name", None),
                "project": os.path.basename(self.paths.outdir.rstrip(os.sep)),
            },
        )

        print("Indexing chunks ...")
        check_cancelled()
        self.store.build_chunk_index(self.repo.notes, self.cfg.rag, self.cfg.index)
        print("Building label prototypes ...")
        check_cancelled()
        self.pooler.build_prototypes()
    
        rules_map   = self.repo.label_rules_by_label
        label_types = self.repo.label_types()
    
        # ---------- small helpers ----------
        def _to_unit_only(df: "pd.DataFrame") -> "pd.DataFrame":
            if df is None or df.empty:
                return pd.DataFrame(columns=["unit_id","label_id","label_type","selection_reason"])
            cols = [c for c in ["unit_id","label_id","label_type","selection_reason"] if c in df.columns]
            return df[cols].drop_duplicates(subset=["unit_id"], keep="first").copy()
    
        def _filter_units(df: "pd.DataFrame", excluded: set[str]) -> "pd.DataFrame":
            if df is None or df.empty or not excluded:
                return df if df is not None else pd.DataFrame(columns=["unit_id"])
            return df[~df["unit_id"].isin(excluded)].copy()
    
        def _head_units(df: "pd.DataFrame", k: int) -> "pd.DataFrame":
            if df is None or df.empty or k <= 0:
                return pd.DataFrame(columns=["unit_id","label_id","label_type","selection_reason"])
            return df.drop_duplicates(subset=["unit_id"], keep="first").head(k).copy()

    
        # ---------- seen/unseen and quotas ----------
        seen_units = set(self.repo.ann["unit_id"].unique().tolist())
        seen_pairs = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        unseen_pairs_all = self.build_unseen_pairs()
    
        total = int(self.cfg.select.batch_size)
        n_dis = int(total * self.cfg.select.pct_disagreement)
        n_div = int(total * self.cfg.select.pct_diversity)
        n_unc = int(total * self.cfg.select.pct_uncertain) if self.cfg.llmfirst.use_llm_probe else 0
        n_cer = int(total * self.cfg.select.pct_easy_qc)   if self.cfg.llmfirst.use_llm_probe else 0
    
        selected_rows: list[pd.DataFrame] = []
        selected_units: set[str] = set()

        run_id = time.strftime("%Y%m%d-%H%M%S")
        LLM_RECORDER.start(outdir=self.paths.outdir, run_id=run_id)
    
        # progressively exclude units
        selected_units: set[str] = set()
        selected_rows: list[pd.DataFrame] = []
    
        # 1) Disagreement (unit-level, excluding seen + already-picked)
        print("[1/4] Expanded disagreement ...")
        check_cancelled()
        dis_pairs = self.build_disagreement_bucket(seen_pairs, rules_map, label_types)
        dis_pairs = _filter_units(dis_pairs, seen_units | selected_units)
        dis_units = _head_units(_to_unit_only(dis_pairs), n_dis)
        dis_units.to_parquet(os.path.join(self.paths.outdir, "bucket_disagreement.parquet"), index=False)
        selected_rows.append(dis_units)
        selected_units |= set(dis_units["unit_id"])

        # 2) Diversity (exclude seen + already-picked via both unseen_pairs filter and seed set)
        print("[2/4] Diversity ...")
        check_cancelled()
        want_div = min(n_div, max(0, total - len(selected_units)))
        fam = FamilyLabeler(self.llm, self.rag, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
        sel_div_pairs = build_diversity_bucket(
            unseen_pairs=unseen_pairs_all,
            already_selected=[(r.unit_id, getattr(r, "label_id", "")) for r in dis_units.itertuples(index=False)],
            n_div=want_div,
            pooler=self.pooler,
            retriever=self.rag,
            rules_map=rules_map,
            label_types=label_types,
            label_config=self.label_config,
            rag_k=getattr(self.cfg.diversity, "rag_topk", 4),
            min_rel_quantile=getattr(self.cfg.diversity, "min_rel_quantile", 0.30),
            mmr_lambda=getattr(self.cfg.diversity, "mmr_lambda", 0.7),
            sample_cap=getattr(self.cfg.diversity, "sample_cap", 2000),
            adaptive_relax=getattr(self.cfg.diversity, "adaptive_relax", True),
            relax_steps=getattr(self.cfg.diversity, "relax_steps", (0.20, 0.10, 0.05)),
            pool_factor=getattr(self.cfg.diversity, "pool_factor", 4.0),
            use_proto=getattr(self.cfg.diversity, "use_proto", False),
            family_labeler=fam,
            ensure_exemplars=True,
            exclude_units=seen_units | selected_units,
        )
        sel_div_pairs = _filter_units(sel_div_pairs, seen_units | selected_units)
        sel_div_units = _head_units(_to_unit_only(sel_div_pairs), want_div)
        sel_div_units.to_parquet(os.path.join(self.paths.outdir, "bucket_diversity.parquet"), index=False)
        selected_rows.append(sel_div_units)
        selected_units |= set(sel_div_units["unit_id"])
    
        # 3) LLM-uncertain (gated); exclude seen + prior-picked
        if self.cfg.llmfirst.use_llm_probe:
            print("[3/4] LLM-uncertain ...")
            check_cancelled()
            want_unc = min(n_unc, max(0, total - len(selected_units)))
            if want_unc > 0:
                sel_unc_pairs = self.build_llm_uncertain_bucket(
                    label_types, rules_map,
                    exclude_units=seen_units | selected_units,   # <- sampler-level exclusion
                )
                sel_unc_pairs = _filter_units(sel_unc_pairs, seen_units | selected_units)
                sel_unc_units = _head_units(_to_unit_only(sel_unc_pairs), want_unc)
                sel_unc_units.to_parquet(os.path.join(self.paths.outdir, "bucket_llm_uncertain.parquet"), index=False)
                selected_rows.append(sel_unc_units)
                selected_units |= set(sel_unc_units["unit_id"])
            else:
                print("Uncertain bucket skipped: no remaining quota.")
    
        # 4) LLM-certain (gated); exclude seen + prior-picked
        if self.cfg.llmfirst.use_llm_probe:
            print("[4/4] LLM-certain ...")
            check_cancelled()
            want_cer = min(n_cer, max(0, total - len(selected_units)))
            if want_cer > 0:
                sel_cer_pairs = self.build_llm_certain_bucket(
                    label_types, rules_map,
                    exclude_units=seen_units | selected_units,   # <- sampler-level exclusion
                )
                sel_cer_pairs = _filter_units(sel_cer_pairs, seen_units | selected_units)
                sel_cer_units = _head_units(_to_unit_only(sel_cer_pairs), want_cer)
                sel_cer_units.to_parquet(os.path.join(self.paths.outdir, "bucket_llm_certain.parquet"), index=False)
                selected_rows.append(sel_cer_units)
                selected_units |= set(sel_cer_units["unit_id"])
            else:
                print("Certain bucket skipped: no remaining quota.")

        # ---------- Compose final (units only) + top-off ----------
        final = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(columns=["unit_id","label_id","label_type","selection_reason"])
        final = final.drop_duplicates(subset=["unit_id"], keep="first").copy()

        if len(final) < total:
            print("Topping off to target batch_size ...")
            excluded = seen_units | set(final["unit_id"])
            unseen_pairs_topoff = [(u, l) for (u, l) in self.build_unseen_pairs() if u not in excluded]
            final = self.top_off_random(
                current_sel=final,
                unseen_pairs=unseen_pairs_topoff,
                label_types=label_types,
                target_n=total,
            ).drop_duplicates(subset=["unit_id"], keep="first")
    
        final.to_parquet(os.path.join(self.paths.outdir, "final_selection.parquet"), index=False)

        # (Optional) run family labeling on the final units for transparency/audit
        if self.cfg.final_llm_labeling:
            fam_rows = []
            fam = FamilyLabeler(self.llm, self.rag, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
            unit_ids = final["unit_id"].tolist()
            rules_map = self.repo.label_rules_by_label
            types = self.repo.label_types()
            _progress_every = float(fam.cfg.progress_min_interval_s or 1)
            for uid in iter_with_bar(
                    step="Final family labeling",
                    iterable=unit_ids,
                    total=len(unit_ids),
                    logger=LOGGER,
                    min_interval_s=_progress_every):
                fam_rows.extend(
                    fam.label_family_for_unit(uid, types, rules_map,
                                              json_only=True,
                                              json_n_consistency=getattr(self.cfg.llmfirst, "final_llm_label_consistency", 1),
                                              json_jitter=False)
                )
            fam_df = pd.DataFrame(fam_rows)
            if not fam_df.empty:
                if "runs" in fam_df.columns: fam_df.rename(columns={"runs":"llm_runs"}, inplace=True)
                if "consistency" in fam_df.columns: fam_df.rename(columns={"consistency":"llm_consistency"}, inplace=True)
                if "prediction" in fam_df.columns: fam_df.rename(columns={"prediction":"llm_prediction"}, inplace=True)
                if "llm_runs" in fam_df.columns:
                    fam_df["llm_reasoning"] = fam_df["llm_runs"].map(
                        lambda rs: (rs[0].get("raw", {}).get("reasoning") if isinstance(rs, list) and rs else None)
                    )
                fam_df = _jsonify_cols(fam_df, [c for c in ["rag_context","llm_runs","fc_probs"] if c in fam_df.columns])
            fam_df.to_parquet(os.path.join(self.paths.outdir, "final_llm_family_probe.parquet"), index=False)
    
            # Build wide convenience view & save merged
            if not fam_df.empty:
                pv = fam_df[["unit_id","label_id","llm_prediction"]].copy()
                pv["col"] = pv["label_id"].astype(str) + "_llm"
                fam_wide = (
                    pv.pivot_table(index="unit_id", columns="col", values="llm_prediction", aggfunc="first")
                      .reset_index()
                )
                if "llm_reasoning" in fam_df.columns:
                    rv = fam_df[["unit_id","label_id","llm_reasoning"]].copy()
                    rv["colr"] = rv["label_id"].astype(str) + "_llm_reason"
                    fam_reason_wide = (
                        rv.pivot_table(index="unit_id", columns="colr", values="llm_reasoning", aggfunc="first")
                          .reset_index()
                    )
                    fam_wide = fam_wide.merge(fam_reason_wide, on="unit_id", how="left")
                final_units = final[["unit_id", "label_id", "label_type", "selection_reason"]].drop_duplicates()
                final_out = final_units.merge(fam_wide, on="unit_id", how="left")
                final_out.to_parquet(os.path.join(self.paths.outdir, "final_selection_with_llm.parquet"), index=False)
        
        diagnostics = {
            "total_selected": int(len(final)),
            "bucket_sizes": {
                "disagreement": int(len(dis_units) if 'dis_bucket_units' in locals() else 0),
                "diversity":    int(len(sel_div_units)   if 'sel_div_units' in locals() else 0),
                "uncertain":    int(len(sel_unc_units)   if 'sel_unc_units' in locals() else 0),
                "certain":      int(len(sel_cer_units)   if 'sel_cer_units' in locals() else 0),
            },
            "unique_units": int(len(final["unit_id"].unique())),
        }
        
        try:
            rec_path = LLM_RECORDER.flush()
            if rec_path:
                LOGGER.info("llm_run_log_written", extra={"path": rec_path, "n_calls": len(LLM_RECORDER.calls)})
        except Exception as e:
            LOGGER.warning("llm_run_log_write_failed", extra={"error": str(e)})

        total_seconds = time.time() - t0
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Done. Total elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        
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
        index=IndexConfig(type=args.index_type, nlist=args.index_nlist, nprobe=args.index_nprobe, hnsw_M=args.hnsw_M, hnsw_efSearch=args.hnsw_efSearch),
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
