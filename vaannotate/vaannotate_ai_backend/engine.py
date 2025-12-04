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
 - pure inference branch with query builder: default (zero shot, family-tree traversal, label exemplars). Knobs:
      - zero vs. few shot
      - family-tree traversal vs. single prompt
      - label exemplars for RAG vs. hand-written exemplars
      - Prompt stems - use rules as-is from most recent round, or hand-write
      - checkpointing on inference runs + seamless resume
"""

from __future__ import annotations
import gzip
import os, re, json, math, time, random, hashlib, unicodedata
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional, Any, Mapping
import numpy as np
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from .label_configs import EMPTY_BUNDLE, LabelConfigBundle
from .llm_backends import JSONCallResult, ForcedChoiceResult, build_llm_backend

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

# ------------------------------
# Config
# ------------------------------

@dataclass
class IndexConfig:
    type: str = "flat"    # flat | hnsw | ivf
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
    keywords: List[str] = field(default_factory=list)
    label_keywords: dict[str, list[str]] = field(default_factory=dict)
    min_context_chunks: int = 3
    mmr_multiplier: int = 3
    neighbor_hops: int = 1
        
def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class LLMConfig:
    model_name: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    )
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "azure"))
    temperature: float = 0.2
    n_consistency: int = 3
    logprobs: bool = True
    top_logprobs: int = 5
    few_shot_examples: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    prediction_field: str = "prediction"
    timeout: float = 60.0
    retry_max: int = 3
    retry_backoff: float = 2.0
    max_context_chars: int = 1200000
    rpm_limit: Optional[int] = 30
    include_reasoning: bool = False
    # Azure OpenAI specific knobs
    azure_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY")
    )
    azure_api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    )
    azure_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    # Local ExLlamaV2 specific knobs
    local_model_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_MODEL_DIR")
    )
    local_max_seq_len: Optional[int] = field(
        default_factory=lambda: _env_int("LOCAL_LLM_MAX_SEQ_LEN")
    )
    local_max_new_tokens: Optional[int] = field(
        default_factory=lambda: _env_int("LOCAL_LLM_MAX_NEW_TOKENS")
    )
    # Context ordering for snippets
    context_order: str = "relevance"  # relevance | chronological
    
@dataclass
class SelectionConfig:
    batch_size: int = 10
    pct_disagreement: float = 0.3
    pct_uncertain: float = 0.3    # LLM-uncertain
    pct_easy_qc: float = 0.1      # LLM-certain
    pct_diversity: float = 0.3

@dataclass
class LLMFirstConfig:
    n_probe_units: int = 10
    topk: int = 6
    json_trace_policy: str = 'fallback'
    progress_min_interval_s: float = 10.0
    exemplar_K: int = 1
    exemplar_generate: bool = True
    exemplar_temperature: float = 0.9
    # forced-choice micro-probe
    fc_enable: bool = True
    #label enrichment for probe
    enrich: bool = True
    probe_enrichment_mix: float = 1.00          # fraction of enriched vs uniform
    probe_enrichment_equalize: bool = True      # equal per parent; else proportional
    probe_ce_unit_sample: int = 75
    probe_ce_search_topk_per_unit: int = 15
    probe_ce_rerank_m: int = 3        # aggregate top-3 CE
    probe_ce_unit_agg: str = "max"    # or "mean"
    single_doc_context: str = "rag"
    single_doc_full_context_max_chars: int = 12000
    context_order: str = "relevance"  # relevance | chronological
    

@dataclass
class DisagreementConfig:
    round_policy: str = 'last'       # 'last' | 'all' | 'decay'
    decay_half_life: float = 2.0     # if round_policy='decay'
    high_entropy_threshold: float = 0.0001 #very low = any disagreements included
    seeds_per_label: int = 5
    snippets_per_seed: int = 3
    similar_chunks_per_seed: int = 50
    expanded_per_label: int = 10
    # Hard-disagreement thresholds
    date_disagree_days: int = 5
    numeric_disagree_abs: float = 1.0
    numeric_disagree_rel: float = 0.20
    
@dataclass
class DiversityConfig:
    rag_k: int = 4
    min_rel_quantile: float = 0.30
    mmr_lambda: float = 0.7
    sample_cap: int = 50
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
    excluded_unit_ids: set[str] = field(default_factory=set)


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
                  bar_width: int = 32, min_interval_s: float = 10,
                  ascii_only: bool | None = None):
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

    wrote_progress = False
    last_render_len = 0
    last_tty_msg = ""
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
            if wrote_progress and last_render_len:
                _sys.stderr.write("\r" + " " * last_render_len)
            _sys.stderr.write("\r" + msg)
            last_render_len = len(msg)
            last_tty_msg = msg
            wrote_progress = True
            _sys.stderr.flush()
        elif not tty and (i == 1 or now - last >= min_interval_s or (total and i == total)):
            last = now
            elapsed = now - t0
            rate = (i / elapsed) if elapsed > 0 else 0.0
            if total:
                eta = ((total - i) / rate) if rate > 0 else float("inf")
                msg = f"[{step}] {i}/{total} • {rate:.2f}/s • ETA {_fmt_hms(eta)}"
            else:
                msg = f"[{step}] {i} done • {rate:.2f}/s • elapsed {_fmt_hms(elapsed)}"
            _sys.stderr.write(msg + "\n")
            _sys.stderr.flush()
        yield item

    # finish line
    if tty:
        if total:
            elapsed = _time.time() - t0
            rate = (total / elapsed) if elapsed > 0 else 0.0
            bar = _bar_str(1.0, width=bar_width, ascii_only=ascii_only)
            if wrote_progress and last_render_len:
                _sys.stderr.write("\r" + " " * last_render_len)
            final = f"{step:<14} {bar}  100%  {total}/{total} • {rate:.2f}/s • elapsed {_fmt_hms(elapsed)}"
            _sys.stderr.write("\r" + final + "\n")
            last_tty_msg = final
        else:
            if wrote_progress and last_render_len:
                _sys.stderr.write("\r" + " " * last_render_len)
            if wrote_progress and last_tty_msg:
                _sys.stderr.write("\r" + last_tty_msg)
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


def _maybe_parse_jsonish(value):
    """Best-effort JSON (or literal) parser that tolerates legacy metadata strings."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            return json.loads(txt)
        except Exception:  # noqa: BLE001
            try:
                import ast

                return ast.literal_eval(txt)
            except Exception:  # noqa: BLE001
                return None
    return None


# ------------------------------
# Data repository
# ------------------------------

class DataRepository:
    def __init__(
        self,
        notes_df: pd.DataFrame,
        ann_df: pd.DataFrame,
        *,
        phenotype_level: str | None = None,
    ):
        level = (phenotype_level or "multi_doc").strip().lower()
        if level not in {"single_doc", "multi_doc"}:
            level = "multi_doc"
        self.phenotype_level = level
        required_notes = {"patient_icn","doc_id","text"}
        if not required_notes.issubset(set(notes_df.columns)):
            raise ValueError(f"Notes missing {required_notes}")
        required_ann = {"round_id","unit_id","doc_id","label_id","reviewer_id","label_value"}
        if not required_ann.issubset(set(ann_df.columns)):
            raise ValueError(f"Annotations missing {required_ann}")

        self.notes = notes_df.copy()
        self.notes["patient_icn"] = self.notes["patient_icn"].astype(str)
        self.notes["doc_id"] = self.notes["doc_id"].astype(str)
        self.notes["text"] = self.notes["text"].astype(str).map(normalize_text)
        original_unit_ids = None
        if "unit_id" in self.notes.columns:
            original_unit_ids = self.notes["unit_id"].astype(str)
        self._notes_by_doc_cache: Optional[Dict[str, str]] = None
        self._unit_doc_lookup: dict[str, str] = {}

        if "notetype" not in self.notes.columns:
            self.notes["notetype"] = ""
        else:
            self.notes["notetype"] = self.notes["notetype"].fillna("").astype(str)

        if self.phenotype_level == "single_doc":
            unit_ids = self.notes["doc_id"]
            if original_unit_ids is not None:
                doc_ids = self.notes["doc_id"].astype(str)
                for raw, doc in zip(original_unit_ids.tolist(), doc_ids.tolist()):
                    if raw:
                        self._unit_doc_lookup.setdefault(raw, doc)
        else:
            unit_ids = self.notes["patient_icn"]
        self.notes["unit_id"] = unit_ids.astype(str)

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
        if "labelset_id" in self.ann.columns:
            self.ann["labelset_id"] = self.ann["labelset_id"].astype(str)
        else:
            self.ann["labelset_id"] = ""
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
        self._round_labelset_map = self._collect_round_labelsets()

    def unit_metadata(self) -> pd.DataFrame:
        columns = ["unit_id", "patient_icn"]
        if self.phenotype_level == "single_doc":
            columns.append("doc_id")
        meta = self.notes[columns].drop_duplicates(subset=["unit_id"]).copy()
        return meta

    def _collect_label_rules(self) -> Dict[str,str]:
        rules = {}
        if "label_rules" in self.ann.columns:
            df = self.ann[["label_id","label_rules"]].dropna()
            for lid, grp in df.groupby("label_id"):
                vals = [v for v in grp["label_rules"].tolist() if isinstance(v,str) and v.strip()]
                if vals:
                    rules[lid] = vals[-1]
        return rules

    def _collect_round_labelsets(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if {"round_id", "labelset_id"}.issubset(self.ann.columns):
            subset = self.ann[["round_id", "labelset_id"]].dropna()
            if not subset.empty:
                subset = subset.astype({"round_id": str, "labelset_id": str})
                for row in subset.drop_duplicates().itertuples(index=False):
                    if row.labelset_id:
                        mapping[str(row.round_id)] = str(row.labelset_id)
        return mapping

    def labelset_for_round(self, round_identifier: str) -> Optional[str]:
        return self._round_labelset_map.get(str(round_identifier))

    def exclude_units(self, unit_ids: set[str] | list[str]) -> int:
        """Remove excluded units from notes/annotations and reset caches.

        Returns the number of notes rows removed.
        """
        if not unit_ids:
            return 0

        excluded = {str(u) for u in unit_ids if str(u)}
        if self.notes.empty or not excluded:
            return 0

        mask = ~self.notes["unit_id"].astype(str).isin(excluded)
        removed_notes = int(len(self.notes) - mask.sum())
        if removed_notes:
            self.notes = self.notes.loc[mask].reset_index(drop=True)
            self._notes_by_doc_cache = None

        if not self.ann.empty and "unit_id" in self.ann.columns and excluded:
            ann_mask = ~self.ann["unit_id"].astype(str).isin(excluded)
            if int(len(self.ann) - ann_mask.sum()):
                self.ann = self.ann.loc[ann_mask].reset_index(drop=True)

        # Trim any stale unit->doc lookup entries for single_doc mode
        if self._unit_doc_lookup:
            for uid in list(self._unit_doc_lookup.keys()):
                if uid in excluded:
                    self._unit_doc_lookup.pop(uid, None)

        return removed_notes

    def labelset_for_annotation(
        self,
        unit_id: str,
        label_id: str,
        *,
        round_id: Optional[str] = None,
    ) -> Optional[str]:
        sub = self.ann
        if round_id is not None:
            sub = sub[sub["round_id"].astype(str) == str(round_id)]
        sub = sub[(sub["unit_id"].astype(str) == str(unit_id)) & (sub["label_id"].astype(str) == str(label_id))]
        if sub.empty:
            return None
        first = sub["labelset_id"].dropna().astype(str)
        for value in first:
            text = value.strip()
            if text:
                return text
        return None

    def notes_by_doc(self) -> Dict[str,str]:
        if self._notes_by_doc_cache is None:
            self._notes_by_doc_cache = dict(zip(self.notes["doc_id"].tolist(), self.notes["text"].tolist()))
        return self._notes_by_doc_cache

    def doc_id_for_unit(self, unit_id: str) -> Optional[str]:
        if self.phenotype_level != "single_doc":
            return None
        unit_str = str(unit_id)
        if not unit_str:
            return None
        notes = self.notes_by_doc()
        if unit_str in notes:
            return unit_str
        doc = self._unit_doc_lookup.get(unit_str)
        if doc:
            return doc
        matches = self.ann[self.ann["unit_id"] == unit_str]
        if not matches.empty:
            docs = matches["doc_id"].dropna().astype(str)
            if not docs.empty:
                return docs.iloc[0]
        return None

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

            labelset_value = ""
            if "labelset_id" in g.columns:
                non_empty = g["labelset_id"].dropna().astype(str).str.strip()
                if not non_empty.empty:
                    labelset_value = str(non_empty.iloc[-1])

            round_value = ""
            if "round_id" in g.columns:
                round_non_empty = g["round_id"].dropna().astype(str).str.strip()
                if not round_non_empty.empty:
                    round_value = str(round_non_empty.iloc[-1])

            rows.append({
                "unit_id": str(uid),
                "label_id": str(lid),
                "disagreement_score": float(score),
                "n_reviewers": int(g["reviewer_id"].nunique()),
                "round_id": round_value,
                "labelset_id": labelset_value,
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
            for i in iter_with_bar("Embedding chunks",
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
            for row in iter_with_bar(
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


_FULL_DOC_CONTEXT_FALLBACK_CHARS = 12000


def _contexts_for_unit_label(
    retriever: "RAGRetriever",
    repo: DataRepository,
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
    if getattr(repo, "phenotype_level", "").strip().lower() == "single_doc":
        resolver = getattr(repo, "doc_id_for_unit", None)
        if callable(resolver):
            resolved = resolver(unit_id)
            if resolved:
                resolved_unit_id = str(resolved)
    if repo.phenotype_level == "single_doc" and mode == "full":
        doc_id = resolved_unit_id
        text = repo.notes_by_doc().get(doc_id)
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
            idxs = retriever.store.get_patient_chunk_indices(doc_id)
        except Exception:
            idxs = []
        if idxs:
            try:
                chunk_meta = retriever.store.chunk_meta[idxs[0]]
                metadata = retriever._extract_meta(chunk_meta) or {}
            except Exception:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("other_meta", "")

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

    return retriever.retrieve_for_patient_label(
        resolved_unit_id,
        label_id,
        label_rules,
        topk_override=topk_override,
        min_k_override=min_k_override,
        mmr_lambda_override=mmr_lambda_override,
    )


class DisagreementExpander:
    def __init__(
        self,
        cfg: DisagreementConfig,
        repo: DataRepository,
        retriever: RAGRetriever,
        label_config_bundle: LabelConfigBundle | None = None,
        llmfirst_cfg: LLMFirstConfig | None = None,
    ):
        self.cfg = cfg
        self.repo = repo
        self.retriever = retriever
        self.label_config_bundle = label_config_bundle or EMPTY_BUNDLE
        self._dependency_cache: Dict[str, tuple[dict, dict, list]] = {}
        self.llmfirst_cfg = llmfirst_cfg

    def _dependencies_for(self, labelset_id: Optional[str]) -> tuple[dict, dict, list]:
        key = str(labelset_id or "__current__")
        if key not in self._dependency_cache:
            config = self.label_config_bundle.config_for_labelset(labelset_id)
            try:
                deps = build_label_dependencies(config)
            except Exception:
                deps = ({}, {}, [])
            self._dependency_cache[key] = deps
        return self._dependency_cache[key]

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
        seeds = seeds.copy()
        if "labelset_id" not in seeds.columns:
            seeds["labelset_id"] = seeds.apply(
                lambda r: self.repo.labelset_for_annotation(
                    r.get("unit_id", ""),
                    r.get("label_id", ""),
                    round_id=r.get("round_id"),
                )
                or "",
                axis=1,
            )
        else:
            seeds["labelset_id"] = seeds["labelset_id"].fillna("").astype(str)
        seeds["round_id"] = seeds.get("round_id", "").astype(str)
        types = self.repo.label_types()
        consensus = self.repo.last_round_consensus()  # {(unit_id,label_id)-> value str}

        def _gate_seed(row: pd.Series) -> bool:
            uid = str(row["unit_id"])
            lid = str(row["label_id"])
            labelset_id = str(row.get("labelset_id") or "").strip() or None
            parent_to_children, child_to_parents, roots = self._dependencies_for(labelset_id)
            roots_set = set(str(x) for x in (roots or []))
            # Parents (roots) are always eligible; children need parent gate pass
            if str(lid) in roots_set:
                return True
            parents = child_to_parents.get(str(lid), [])
            if not parents:
                return True  # not marked as child → treat as eligible
            parent_preds = {(str(uid), str(p)): consensus.get((str(uid), str(p)), None) for p in parents}
            # IMPORTANT: evaluate by prior-round consensus; robust evaluator handles casing/types
            config = self.label_config_bundle.config_for_labelset(labelset_id)
            return evaluate_gating(str(lid), str(uid), parent_preds, types, config)

        if not seeds.empty:
            seeds["unit_id"] = seeds["unit_id"].astype(str)
            seeds["label_id"] = seeds["label_id"].astype(str)
            seeds = seeds[seeds.apply(_gate_seed, axis=1)]
        
        rows = []
        for lid, grp in seeds.groupby("label_id"):
            rows.append(grp.head(self.cfg.seeds_per_label))
        return pd.concat(rows, ignore_index=True) if rows else seeds.head(0)

    def seed_snippets(self, unit_id: str, label_id: str, label_rules: str) -> List[str]:
        spans = self.repo.get_prior_rationales(unit_id, label_id)
        snips = [sp.get("snippet") for sp in spans if isinstance(sp,dict) and sp.get("snippet")]
        if snips: return snips[: self.cfg.snippets_per_seed]
        ctx = _contexts_for_unit_label(
            self.retriever,
            self.repo,
            unit_id,
            label_id,
            label_rules,
            topk_override=self.cfg.snippets_per_seed,
            single_doc_context_mode=getattr(self.llmfirst_cfg, "single_doc_context", "rag"),
            full_doc_char_limit=getattr(self.llmfirst_cfg, "single_doc_full_context_max_chars", None),
        )
        return [c["text"] for c in ctx if isinstance(c.get("text"), str)]

    def expand(self, rules_map: Dict[str,str], seen_pairs: set) -> pd.DataFrame:
        seeds = self.high_entropy_seeds()
        print('disagreement seeds: ', seeds)
        rows = []
        for lid, grp in seeds.groupby("label_id"):
            labelset_value = ""
            if "labelset_id" in grp.columns:
                non_empty = grp["labelset_id"].dropna().astype(str).str.strip()
                if not non_empty.empty:
                    labelset_value = str(non_empty.iloc[0])
            round_value = ""
            if "round_id" in grp.columns:
                round_non_empty = grp["round_id"].dropna().astype(str).str.strip()
                if not round_non_empty.empty:
                    round_value = str(round_non_empty.iloc[0])
            snips = []
            for r in grp.itertuples(index=False):
                snips.extend(self.seed_snippets(r.unit_id, lid, rules_map.get(lid,"")))
            snips = snips[: self.cfg.seeds_per_label * self.cfg.snippets_per_seed]
            if not snips: continue
            cand = self.retriever.expand_from_snippets(lid, snips, seen_pairs, per_seed_k=self.cfg.similar_chunks_per_seed)
            items = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.expanded_per_label]
            for uid, sc in items:
                rows.append({
                    "unit_id": uid,
                    "label_id": lid,
                    "score": float(sc),
                    "bucket": "disagreement_expanded",
                    "labelset_id": labelset_value,
                    "round_id": round_value,
                })
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
        self._bm25_cache: Dict[str, dict] = {}
        self._label_query_texts = {}   # (label_id, rules_hash, K) -> List[str]
        self._label_query_embs  = {}   # (label_id, rules_hash, K) -> np.ndarray[K,d]
        # Optional concise rule text to keep re-ranker context compact
        self.rerank_rule_overrides: dict[str, str] = {}

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
        base = f"Evidence relevant to patient-level label '{label_id}'. "
        if label_rules and isinstance(label_rules,str) and label_rules.strip():
            base += "Guidelines: " + re.sub(r"\s+"," ",label_rules.strip()) + " "
        return base.strip()

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
        if hasattr(self.store, "bm25_index_for_unit"):
            idx = self.store.bm25_index_for_unit(uid)
            if idx is not None:
                self._bm25_cache[uid] = idx
                return idx
        idxs = self.store.get_patient_chunk_indices(uid)
        if not idxs:
            return None
        docs, metas = [], []
        for ix in idxs:
            meta = self.store.chunk_meta[ix]
            tokens = self._tokenize_for_bm25(meta.get("text", ""))
            if not tokens:
                continue
            docs.append(tokens)
            metas.append(meta)
        if not docs:
            return None
        avgdl = sum(len(toks) for toks in docs) / float(len(docs))
        index = {"docs": docs, "metas": metas, "avgdl": avgdl}
        self._bm25_cache[uid] = index
        if hasattr(self.store, "bm25_indices"):
            try:
                existing_indices = getattr(self.store, "bm25_indices", {})
                if not isinstance(existing_indices, dict):
                    existing_indices = {}
                units = existing_indices.get("units") if "units" in existing_indices else existing_indices
                if not isinstance(units, dict):
                    units = {}
                units = dict(units)
                units[uid] = index
                if "units" in existing_indices or not existing_indices:
                    existing_indices = dict(existing_indices)
                    existing_indices["units"] = units
                else:
                    existing_indices = units
                self.store.bm25_indices = existing_indices
                if hasattr(self.store, "_chunk_cache_dir_path"):
                    self.store._save_bm25_indices(self.store._chunk_cache_dir_path, self.store.bm25_indices)
            except Exception:
                pass
        return index

    def _bm25_hits_for_patient(self, unit_id: str, keywords: List[str]) -> List[dict]:
        if not keywords:
            return []
        index = self._bm25_index_for_patient(unit_id)
        if not index:
            return []
        query_tokens = []
        for kw in keywords:
            if isinstance(kw, str):
                query_tokens.extend(self._tokenize_for_bm25(kw))
        if not query_tokens:
            return []
        docs = index["docs"]
        idf = getattr(self.store, "idf_global", {}) or {}
        if not idf and isinstance(getattr(self.store, "bm25_indices", None), dict):
            indices_dict = getattr(self.store, "bm25_indices", {})
            idf = indices_dict.get("idf_global", {}) if isinstance(indices_dict, dict) else {}
        if not idf and hasattr(self.store, "_build_bm25_indices"):
            built = self.store._build_bm25_indices()
            if isinstance(built, dict):
                self.store.bm25_indices = built
                idf = built.get("idf_global", {})
                if hasattr(self.store, "_chunk_cache_dir_path"):
                    try:
                        self.store._save_bm25_indices(self.store._chunk_cache_dir_path, built)
                    except Exception:
                        pass
        if not idf:
            return []
        avgdl = index["avgdl"] or 1.0
        metas = index["metas"]
        k1, b = 1.5, 0.75
        scores: List[tuple[float, dict]] = []
        for toks, meta in zip(docs, metas):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in query_tokens:
                if term not in tf or term not in idf:
                    continue
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * dl / avgdl)
                score += idf[term] * (freq * (k1 + 1)) / (denom + 1e-12)
            if score <= 0:
                continue
            scores.append((score, meta))
        scores.sort(key=lambda pair: pair[0], reverse=True)
        out = []
        for score, meta in scores[: self.cfg.keyword_topk]:
            out.append({
                "doc_id": meta.get("doc_id"),
                "chunk_id": meta.get("chunk_id"),
                "metadata": self._extract_meta(meta),
                "text": meta.get("text", ""),
                "score": float(score),
                "source": "bm25",
            })
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
        Q = self._get_label_query_embs(label_id, label_rules, K=K_use)
        mmr_select_k = final_k * mmr_mult

        rule_query = self._build_query(label_id, label_rules)
        rerank_query = self._build_query(label_id, self._rerank_rules_text(label_id, label_rules))
        rule_emb = self.store._embed([rule_query])[0]
        mmr_query_embs: list[np.ndarray] = [rule_emb]
        semantic_runs: list[list[dict]] = []

        rule_hits = _patient_local_rank(str(unit_id), rule_emb, need=mmr_select_k * 2)
        for it in rule_hits:
            it["source"] = "patient_rule"
        semantic_runs.append(rule_hits)

        if Q is not None and getattr(Q, "ndim", 1) == 2 and Q.shape[0] > 0:
            for i in range(Q.shape[0]):
                ex_hits = _patient_local_rank(str(unit_id), Q[i], need=mmr_select_k * 2)
                for it in ex_hits:
                    it["source"] = "patient_exemplar"
                semantic_runs.append(ex_hits)
                mmr_query_embs.append(Q[i])

        if len(semantic_runs) == 1:
            items = semantic_runs[0]
        else:
            items = self._reciprocal_rank_fusion(semantic_runs)

        q_emb = np.mean(np.vstack(mmr_query_embs), axis=0) if mmr_query_embs else rule_emb
        query = rerank_query

        #  + (optional) keywords
        bm25_hits: List[dict] = []
        if use_kw:
            lblcfg = self.label_configs.get(label_id, {}) if isinstance(self.label_configs, dict) else {}
            cfg_keywords = getattr(cfg_rag, "keywords", [])
            keywords: list[str] = []
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
            if keywords:
                # de-duplicate while preserving order
                seen_kw = set()
                uniq_keywords = []
                for kw in keywords:
                    if kw in seen_kw:
                        continue
                    seen_kw.add(kw)
                    uniq_keywords.append(kw)
                bm25_hits = self._bm25_hits_for_patient(str(unit_id), uniq_keywords)
        if bm25_hits:
            if items:
                items = self._reciprocal_rank_fusion([items, bm25_hits])
            else:
                items = bm25_hits

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
        self.backend = None
        # The label configuration is injected by the orchestrator once it has been
        # materialised.  Default to an empty mapping so that downstream calls that
        # access ``self.label_config`` degrade gracefully when no configuration has
        # been supplied (e.g. during unit tests or CLI usage without overrides).
        self.label_config: dict[str, object] = {}
        self.backend = build_llm_backend(self.cfg)

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

    def _few_shot_messages(self, label_id: str) -> list[dict[str, str]]:
        examples_cfg = getattr(self.cfg, "few_shot_examples", {}) or {}
        if not isinstance(examples_cfg, Mapping):
            return []
        label_key = str(label_id)
        label_examples = examples_cfg.get(label_key) or examples_cfg.get(label_key.lower())
        if not isinstance(label_examples, (list, tuple)):
            return []
        messages: list[dict[str, str]] = []
        for entry in label_examples:
            if not isinstance(entry, Mapping):
                continue
            answer = entry.get("answer")
            context = entry.get("context")
            if context is not None:
                ctx_text = str(context)
                if ctx_text.strip():
                    ctx_text = f"EHR context:\n{ctx_text}"
                messages.append({"role": "user", "content": ctx_text})
            if answer is not None:
                messages.append({"role": "assistant", "content": str(answer)})
        return messages

    def summarize_label_rule_for_rerank(self, label_id: str, label_rules: str, max_sentences: int = 2) -> str:
        """Generate a concise paraphrase of a label rule for re-ranker queries."""

        if not getattr(self, "backend", None):
            return ""
        system_msg = (
            "You condense clinical labeling guidelines for a cross-encoder reranker. "
            "Return a succinct 1-2 sentence summary that preserves the key inclusion/" \
            "exclusion cues. Avoid meta commentary."
        )
        user_msg = (
            "Label ID: {label_id}\n"
            "Rewrite the following label rule into at most {limit} sentences capturing the essence:\n{rules}"
        ).format(label_id=label_id, limit=max_sentences, rules=label_rules)
        result = self.backend.json_call(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            logprobs=False,
            top_logprobs=None,
        )
        text = re.sub(r"\s+", " ", str(getattr(result, "content", "") or "")).strip()
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        trimmed = " ".join(sentences[:max_sentences]).strip()
        return trimmed or text

    def annotate(
        self,
        unit_id: str,
        label_id: str,
        label_type: str,
        label_rules: str,
        snippets: List[dict],
        n_consistency: int = 1,
        jitter_params: bool = False,
    ) -> dict:
        import json, time

        rag_topk_range = self.scCfg.rag_topk_range
        rag_dropout_p = self.scCfg.rag_dropout_p
        temp_range = self.scCfg.temperature_range
        shuffle_context = self.scCfg.shuffle_context
        context_order = getattr(self.cfg, "context_order", "relevance") or "relevance"

        def _ordered_snippets(items: List[dict]) -> List[dict]:
            if str(context_order).lower() != "chronological":
                return list(items)
            sortable: list[tuple[str, int, dict]] = []
            for idx, snip in enumerate(items):
                meta = snip.get("metadata") if isinstance(snip, Mapping) else {}
                date_val = ""
                if isinstance(meta, Mapping):
                    date_val = str(meta.get("date") or "")
                sortable.append((date_val, idx, snip))
            sortable.sort(key=lambda t: (t[0], t[1]))
            return [entry[2] for entry in sortable]

        snippets = _ordered_snippets(snippets)

        # Jitter RNG
        rng = random.Random()
        include_reasoning = bool(getattr(self.cfg, "include_reasoning", True))
    
        # --- helper: build context text from a candidate chunk list with char budget ---
        def _build_context_text(_snips: List[dict]) -> str:
            ctx, used = [], 0
            budget = max(1000, getattr(self.cfg, "max_context_chars", 4000))
            for s in _snips:
                md = s.get("metadata") or {}
                hdr_bits = [f"doc_id={s.get('doc_id')}", f"chunk_id={s.get('chunk_id')}"]
                if md.get("date"):      hdr_bits.append(f"date={md['date']}")
                note_type = md.get("note_type") or md.get("notetype")
                if note_type:
                    hdr_bits.append(f"type={note_type}")
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
    
        # Prebuild immutable introduction (per-vote metadata added below to avoid cache collisions)
        system_intro = "You are a meticulous clinical annotator for EHR data."
    
        for i in range(n_consistency):
            # ----- sample jitter for this vote -----
            sc_meta = ""
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
                ctx_text = _build_context_text(cand)
                temperature_this_vote = t
            else:
                # legacy behavior: use original ranked snippets (no jitter), fixed temperature
                sc_meta = ""
                ctx_text = _build_context_text(snippets)
                temperature_this_vote = self.cfg.temperature
    
            # ----- build prompt for this vote -----
            opts = _options_for_label(label_id, label_type, self.label_config)
            lt_norm = (label_type or "").strip().lower()
            categorical_types = {
                "binary",
                "boolean",
                "categorical",
                "categorical_single",
                "ordinal",
            }
            use_options = bool(opts) and lt_norm in categorical_types
            option_values = [str(opt) for opt in (opts or [])]

            guideline_text = label_rules if label_rules else "(no additional guidelines)"
            response_keys = "reasoning, prediction" if include_reasoning else "prediction"

            system_segments: list[str] = [
                system_intro,
                f"Your task: label '{label_id}' (type: {label_type}). Use the evidence snippets from this patient's notes.",
                f"Label rules:\n{guideline_text}",
            ]
            if use_options:
                system_segments.append("Choose the single best option from the list below based on the evidence.")
                system_segments.append("Options:")
                system_segments.extend(f"- {opt}" for opt in option_values)
                system_segments.append(
                    f"Set prediction to exactly one of: {', '.join(option_values)}. Do not invent new options."
                )
            else:
                system_segments.append("If insufficient evidence, reply with 'unknown'.")

            if include_reasoning:
                system_segments.append(
                    "Think step-by-step citing specific evidence, and keep the reasoning concise."
                )
            system_segments.append(f"Return strict JSON only with keys: {response_keys}.")
            system_segments.append("No additional keys or text.")

            system_body = "\n\n".join(system_segments)
            system = system_body + ("\n" + sc_meta if sc_meta else "")

            user_content = f"EHR context:\n{ctx_text}" if ctx_text else "EHR context: (no snippets)"

            task = user_content
            few_shot_messages = self._few_shot_messages(label_id)
            messages = [
                {"role": "system", "content": system},
                *few_shot_messages,
                {"role": "user", "content": task},
            ]

            if include_reasoning:
                response_schema = {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": ["string", "null"]},
                        "prediction": {"type": ["string", "number", "boolean", "null"]},
                    },
                    "required": ["reasoning", "prediction"],
                    "additionalProperties": False,
                }
            else:
                response_schema = {
                    "type": "object",
                    "properties": {
                        "prediction": {"type": ["string", "number", "boolean", "null"]},
                    },
                    "required": ["prediction"],
                    "additionalProperties": False,
                }

            # ----- per-vote LLM call -----
            attempt = 0
            while attempt <= self.cfg.retry_max:
                try:
                    result = self.backend.json_call(
                        messages=messages,
                        temperature=temperature_this_vote,
                        logprobs=self.cfg.logprobs,
                        top_logprobs=int(self.cfg.top_logprobs) if self.cfg.logprobs and int(self.cfg.top_logprobs) > 0 else None,
                        response_format={
                            "type": "json_object",
                            "json_schema": response_schema,
                        },
                    )
                    content = result.content
                    data_map: dict[str, Any]
                    raw_data = result.data
                    if isinstance(raw_data, Mapping):
                        data_map = dict(raw_data.items())
                    else:
                        data_map = {"prediction": raw_data}
                    if not include_reasoning:
                        data_map.pop("reasoning", None)

                    pred = data_map.get(self.cfg.prediction_field, data_map.get("prediction"))

                    preds.append(str(pred) if pred is not None else None)

                    run_entry = {
                        "prediction": pred,
                        "raw": data_map,
                        "jitter": ({"k": k, "drop": drop_p, "shuffle": shuffle_context, "temperature": temperature_this_vote}
                                   if jitter_params else None),
                    }
                    if result.logprobs is not None:
                        run_entry["logprobs"] = result.logprobs
                    run_entry["response_latency_s"] = result.latency_s
                    runs.append(run_entry)

                    try:
                        # Capture the exact prompt + minimal context identifiers used this vote
                        LLM_RECORDER.record("json_vote", {
                            "unit_id": unit_id,
                            "label_id": label_id,
                            "label_type": label_type,
                            "vote_idx": i,
                            "prompt": {"system": system, "user": task, "few_shot": few_shot_messages},
                            "params": {
                                "temperature": temperature_this_vote,
                                "n_consistency": int(n_consistency),
                            },
                            "snippets": [{"doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")} for c in (cand if jitter_params else snippets)],
                            "output": {"prediction": pred, "raw": data_map, "content": content},
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
                

    def pooled_vector(self, unit_id: str, label_id: str, retriever: RAGRetriever, label_rules: str, topk: int=6) -> np.ndarray:
        key = (unit_id, label_id)
        if key in self._cache_vec: return self._cache_vec[key]
        ctx = _contexts_for_unit_label(
            retriever,
            self.repo,
            unit_id,
            label_id,
            label_rules,
            topk_override=topk,
            single_doc_context_mode=getattr(self.llmfirst_cfg, "single_doc_context", "rag"),
            full_doc_char_limit=getattr(self.llmfirst_cfg, "single_doc_full_context_max_chars", None),
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
        if str(lid) == "_meta":
            continue
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
    all_labels = {str(k) for k in label_config.keys() if str(k) != "_meta"}
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
        try:
            label_types = self.repo.label_types()
        except Exception:
            label_types = {}
        for lid, rules in (rules_map or {}).items():
            base_k = getattr(self.cfg, "exemplar_K", None)
            K_use = int(base_k if base_k is not None else K)
            if K_use <= 0:
                K_use = K
            opts = _options_for_label(lid, label_types.get(str(lid), "categorical"), self.label_config) or []
            if opts:
                K_use = max(K_use, len(opts))
            # skip if already cached
            if self.retriever._get_label_query_embs(lid, rules, K_use) is not None:
                continue
            t = self._generate_label_exemplars(lid, rules, K_use)
            if not t:
                t = self._fallback_label_exemplars(lid, rules, K_use, opts)
            if t:
                self.retriever.set_label_exemplars(lid, rules, K_use, t)

    def _fallback_label_exemplars(self, label_id: str, rules: str, K: int, options: list[str] | None) -> list[str]:
        """Graceful fallback if exemplar generation fails (e.g., Azure JSON mode issues).

        We synthesize minimal exemplars using the label rules so downstream RAG still
        has deterministic label-aware queries instead of aborting with
        "label_exemplar_error". These are intentionally simple and will be embedded
        just like LLM-generated snippets.
        """
        K = max(1, int(K))
        base = self.retriever._build_query(label_id, rules)
        texts: list[str] = []
        opts = options or []
        if opts:
            for o in opts:
                texts.append(f"{base} Example of option '{o}'.")
        if not texts:
            texts.append(base)
        while len(texts) < K:
            texts.append(base)
        return texts[:K]

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
    
        # ----- Backend JSON call -----
        temp = float(getattr(self.cfg, "exemplar_temperature", 0.7) or 0.7)

        if has_opts:
            snippets_schema: Mapping[str, Any] = {
                "type": "object",
                "properties": {
                    "snippets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "option": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["option", "text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["snippets"],
                "additionalProperties": False,
            }
        else:
            snippets_schema = {
                "type": "object",
                "properties": {
                    "snippets": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["snippets"],
                "additionalProperties": False,
            }

        try:
            result = self.llm.backend.json_call(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temp,
                logprobs=False,
                top_logprobs=None,
                response_format={
                    "type": "json_object",
                    "json_schema": snippets_schema,
                },
            )
            content = result.content
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
        snippets = _contexts_for_unit_label(
            self.retriever,
            self.repo,
            unit_id,
            label_id,
            label_rules,
            topk_override=self.cfg.topk,
            single_doc_context_mode=getattr(self.cfg, "single_doc_context", "rag"),
            full_doc_char_limit=getattr(self.cfg, "single_doc_full_context_max_chars", None),
        )
        ctx_lines = []
        for snip in snippets:
            md = snip.get("metadata") or {}
            hdr_bits = [f"doc_id={snip.get('doc_id')}", f"chunk_id={snip.get('chunk_id')}"]
            if md.get("date"):
                hdr_bits.append(f"date={md['date']}")
            note_type = md.get("note_type") or md.get("notetype")
            if note_type:
                hdr_bits.append(f"type={note_type}")
            header = "[" + ", ".join(hdr_bits) + "] "
            ctx_lines.append(header + (snip.get('text', '') or ''))
        ctx = "\n\n".join(ctx_lines)
        user = (
            f"Task: Choose the single best option for label '{label_id}' given the context snippets.\n" +
            (f"Label rules/hints: {label_rules}\n" if label_rules else "") +
            "Options:\n" + "\n".join(option_lines) + "\n" +
            "Return ONLY the option letter.\n\n" +
            "Context:\n" + ctx
        )
        result = self.llm.backend.forced_choice(
            system=system,
            user=user,
            options=options,
            letters=letters,
            top_logprobs=int(self.llm.cfg.top_logprobs) if int(self.llm.cfg.top_logprobs) > 0 else 5,
        )
        opt_probs = dict(result.option_probs)
        ent = float(result.entropy)
        pred = result.prediction

        try:
            LLM_RECORDER.record("forced_choice", {
                "unit_id": unit_id,
                "label_id": label_id,
                "label_type": label_type,
                "prompt": {"system": system, "user": user},
                "snippets": ctx,
                "fc_output": {
                    "fc_probs": opt_probs,
                    "fc_entropy": ent,
                    "prediction": pred,
                    "latency_s": result.latency_s,
                },
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
            ctx = _contexts_for_unit_label(
                self.retriever,
                self.repo,
                unit_id,
                lid,
                rules,
                topk_override=self.cfg.topk,
                single_doc_context_mode=getattr(self.cfg, "single_doc_context", "rag"),
                full_doc_char_limit=getattr(self.cfg, "single_doc_full_context_max_chars", None),
            )
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
                    min_interval_s=_progress_every):
                if uid in ex:             # redundant but explicit
                    continue
                ctx = _contexts_for_unit_label(
                    self.retriever,
                    self.repo,
                    uid,
                    p,
                    rule,
                    topk_override=int(getattr(self.cfg, "probe_ce_search_topk_per_unit", 24) or 24),
                    single_doc_context_mode=getattr(self.cfg, "single_doc_context", "rag"),
                    full_doc_char_limit=getattr(self.cfg, "single_doc_full_context_max_chars", None),
                )
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
                base_k = getattr(getattr(retriever, "cfg", None), "exemplar_K", None)
                K_use = int(base_k if base_k is not None else 6)
                if K_use <= 0:
                    K_use = 6
                opts = _options_for_label(
                    lid,
                    label_types.get(str(lid), "categorical"),
                    getattr(retriever, "label_configs", {}),
                ) or []
                if opts:
                    K_use = max(K_use, len(opts))
                getQ = getattr(retriever, "_get_label_query_embs", None)
                if callable(getQ):
                    Q = getQ(lid, rules_map.get(lid, ""), K_use)
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

        embed_name = os.getenv("MED_EMBED_MODEL_NAME")
        rerank_name = os.getenv("RERANKER_MODEL_NAME")
        device = _detect_device()
        embedder = SentenceTransformer(embed_name, device=device)
        reranker = CrossEncoder(rerank_name, device=device)
        _ensure_default_ce_max_length(reranker)
        emb_bs = int(os.getenv('EMB_BATCH', '32' if device == "cpu" else "64"))
        rr_bs = int(os.getenv('RERANK_BATCH', '16' if device == "cpu" else "64"))
        self.models = Models(embedder, reranker, device=device, emb_batch=emb_bs, rerank_batch=rr_bs)

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
        self.llm: LLMAnnotator | None = None
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

        self.llm = LLMAnnotator(self.cfg.llm, self.cfg.scjitter, cache_dir=self.paths.cache_dir)
        # Ensure the annotator has access to the materialised label configuration
        # so that option lookups during JSON prompting succeed.
        self.llm.label_config = self.label_config

    def _build_rerank_rule_overrides(self, rules_map: Mapping[str, str]) -> dict[str, str]:
        """Generate concise rule summaries for re-ranker queries when needed."""

        if not self.llm or not isinstance(rules_map, Mapping):
            return {}
        threshold_default = 1200  # ~512-token budget
        try:
            threshold = int(os.getenv("RERANK_RULE_PARAPHRASE_CHARS", str(threshold_default)))
        except Exception:
            threshold = threshold_default
        threshold = max(200, threshold)

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

    def build_unseen_pairs(self, label_ids: Optional[set[str]] = None) -> List[Tuple[str,str]]:
        seen = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        all_units = sorted(self.repo.notes["unit_id"].unique().tolist())

        if label_ids is None:
            legacy_labels = {str(l) for l in self.repo.ann["label_id"].unique().tolist()}
            config_labels: set[str] = set()
            for key, entry in (self.label_config or {}).items():
                if str(key) == "_meta":
                    continue
                if isinstance(entry, dict):
                    lid = str(entry.get("label_id") or key).strip()
                else:
                    lid = str(key).strip()
                if lid:
                    config_labels.add(lid)
            use_labels = legacy_labels | config_labels
        else:
            use_labels = {str(l) for l in label_ids if str(l)}

        all_labels = sorted(use_labels)
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
        expander = DisagreementExpander(
            self.cfg.disagree,
            self.repo,
            self.rag,
            label_config_bundle=self.label_config_bundle,
            llmfirst_cfg=self.cfg.llmfirst,
        )
        expanded = expander.expand(rules_map, seen_pairs)
        if expanded is None or expanded.empty:
            return expanded
    
        dep_cache: Dict[str, tuple[dict, dict, list]] = {}

        def _dependencies_for(labelset_id: Optional[str]) -> tuple[dict, dict, list]:
            key = str(labelset_id or "__current__")
            if key not in dep_cache:
                config = self.label_config_bundle.config_for_labelset(labelset_id)
                try:
                    dep_cache[key] = build_label_dependencies(config)
                except Exception:
                    dep_cache[key] = ({}, {}, [])
            return dep_cache[key]

        consensus = self.repo.last_round_consensus()  # {(unit_id,label_id)-> str}
        types = self.repo.label_types()

        def _resolve_labelset(row: pd.Series) -> str:
            existing = str(row.get("labelset_id") or "").strip()
            if existing:
                return existing
            fallback = self.repo.labelset_for_annotation(
                row.get("unit_id", ""),
                row.get("label_id", ""),
                round_id=row.get("round_id"),
            )
            return str(fallback).strip() if fallback else ""

        df = expanded.copy()
        df["unit_id"] = df["unit_id"].astype(str)
        df["label_id"] = df["label_id"].astype(str)
        if "round_id" in df.columns:
            df["round_id"] = df["round_id"].fillna("").astype(str)
        else:
            df["round_id"] = ""
        df["labelset_id"] = df.apply(_resolve_labelset, axis=1)

        def _is_root(row: pd.Series) -> bool:
            _, _, roots = _dependencies_for(str(row["labelset_id"]) or None)
            roots_set = set(str(x) for x in (roots or []))
            return str(row["label_id"]) in roots_set

        def _gate_ok_expanded(row: pd.Series) -> bool:
            """
            Fail-open for expanded pool:
              - parents: True
              - children: if ANY parent consensus exists, evaluate; if none, allow.
            """

            labelset_id = str(row.get("labelset_id") or "").strip() or None
            _, child_to_parents, _ = _dependencies_for(labelset_id)
            parents = child_to_parents.get(str(row["label_id"]), [])
            if not parents:
                return True
            parent_preds = {}
            have_any = False
            for p in parents:
                key = (str(row["unit_id"]), str(p))
                val = consensus.get(key, None)
                parent_preds[key] = val
                if val is not None and str(val).strip() != "":
                    have_any = True
            if not have_any:
                return True
            config = self.label_config_bundle.config_for_labelset(labelset_id)
            return evaluate_gating(str(row["label_id"]), str(row["unit_id"]), parent_preds, types, config)

        df["is_root_parent"] = df.apply(_is_root, axis=1)
        df = df[df["is_root_parent"] | df.apply(_gate_ok_expanded, axis=1)].reset_index(drop=True)
        
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

        if "round_id" in valid.columns:
            valid["round_id"] = valid["round_id"].fillna("").astype(str)
        else:
            valid["round_id"] = ""
        if "labelset_id" in valid.columns:
            valid["labelset_id"] = valid["labelset_id"].fillna("").astype(str)
        else:
            valid["labelset_id"] = valid.apply(_resolve_labelset, axis=1)

        def _gate_by_cons(uid: str, lid: str, labelset_id: Optional[str]) -> bool:
            _, child_to_parents, _ = _dependencies_for(labelset_id)
            parents = child_to_parents.get(str(lid), [])
            if not parents:
                return True
            parent_preds = {(uid, str(p)): consensus.get((uid, str(p)), None) for p in parents}
            config = self.label_config_bundle.config_for_labelset(labelset_id)
            return evaluate_gating(str(lid), uid, parent_preds, types, config)

        counts_dict: Dict[str, int] = {}
        present_labels = set(df["label_id"].unique().tolist())  # only apportion among labels actually available in this round
        for (lid, lset), grp in valid.groupby(["label_id", "labelset_id"], dropna=False):
            lid = str(lid)
            if lid not in present_labels:
                continue
            labelset_id = str(lset).strip() or None
            _, child_to_parents, roots = _dependencies_for(labelset_id)
            roots_set = set(str(x) for x in (roots or []))
            if lid in roots_set or not child_to_parents.get(lid, []):
                cnt = int(len(grp))
            else:
                # children: only those units whose parent(s) pass the gate by LAST round consensus
                c = 0
                for uid in grp["unit_id"].astype(str).unique():
                    if _gate_by_cons(str(uid), lid, labelset_id):
                        c += 1
                cnt = c
            counts_dict[lid] = counts_dict.get(lid, 0) + cnt
    
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


    def _apply_excluded_units(self) -> int:
        removed = self.repo.exclude_units(self.excluded_unit_ids)
        if removed:
            self.rag._notes_by_doc = self.repo.notes_by_doc()
        return removed


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
        removed = self._apply_excluded_units()
        if removed:
            print(f"Excluded {removed} previously reviewed units from candidate pool")
        if self.repo.notes.empty:
            raise RuntimeError("No candidate units remain after excluding previously reviewed units.")
        print("Building label prototypes ...")
        check_cancelled()
        self.pooler.build_prototypes()
        # Load the LLM backend only after embeddings/cross-encoders have been loaded and
        # used, so GPU memory can be allocated to them before the ExLlamaV2 model.
        self.ensure_llm_backend()

        (
            legacy_rules_map,
            legacy_label_types,
            current_rules_map,
            current_label_types,
        ) = self._label_maps()

        legacy_label_ids = {str(l) for l in self.repo.ann["label_id"].unique().tolist()}
        if not legacy_label_ids:
            legacy_label_ids = set(legacy_rules_map.keys())
        current_label_ids = set(current_rules_map.keys())
        if not current_label_ids:
            current_label_ids = set(legacy_label_ids)

        # Build concise rule overrides for reranker-only queries to preserve context
        # budget without altering retrieval embeddings.
        overrides = self._build_rerank_rule_overrides(current_rules_map)
        self.rag.rerank_rule_overrides = overrides or {}
    
        # ---------- small helpers ----------
        def _empty_unit_frame() -> "pd.DataFrame":
            return pd.DataFrame(
                {
                    "unit_id": pd.Series(dtype="string"),
                    "label_id": pd.Series(dtype="string"),
                    "label_type": pd.Series(dtype="string"),
                    "selection_reason": pd.Series(dtype="string"),
                }
            )

        def _ensure_unit_schema(df: "pd.DataFrame" | None) -> "pd.DataFrame":
            base = _empty_unit_frame()
            if df is None:
                return base
            result = df.copy()
            for col in base.columns:
                if col not in result.columns:
                    result[col] = pd.Series(dtype=base[col].dtype)
            return result[base.columns.tolist()]

        def _to_unit_only(df: "pd.DataFrame") -> "pd.DataFrame":
            if df is None:
                return _empty_unit_frame()
            cols = [c for c in ["unit_id","label_id","label_type","selection_reason"] if c in df.columns]
            if not cols:
                return _empty_unit_frame()
            subset = df[cols].drop_duplicates(subset=["unit_id"], keep="first")
            return _ensure_unit_schema(subset)
    
        def _filter_units(df: "pd.DataFrame", excluded: set[str]) -> "pd.DataFrame":
            if df is None or df.empty or not excluded:
                return df if df is not None else pd.DataFrame(columns=["unit_id"])
            return df[~df["unit_id"].isin(excluded)].copy()
    
        def _head_units(df: "pd.DataFrame", k: int) -> "pd.DataFrame":
            ensured = _ensure_unit_schema(df)
            if ensured.empty or k <= 0:
                return ensured.iloc[0:0].copy()
            return ensured.head(k).copy()

    
        # ---------- seen/unseen and quotas ----------
        seen_units = set(self.repo.ann["unit_id"].unique().tolist())
        seen_pairs = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        unseen_pairs_current = self.build_unseen_pairs(label_ids=current_label_ids)
    
        total = int(self.cfg.select.batch_size)
        n_dis = int(total * self.cfg.select.pct_disagreement)
        n_div = int(total * self.cfg.select.pct_diversity)
        n_unc = int(total * self.cfg.select.pct_uncertain)
        n_cer = int(total * self.cfg.select.pct_easy_qc)
    
        selected_rows: list[pd.DataFrame] = []
        selected_units: set[str] = set()

        run_id = time.strftime("%Y%m%d-%H%M%S")
        LLM_RECORDER.start(outdir=self.paths.outdir, run_id=run_id)
    
        # progressively exclude units
        selected_units: set[str] = set()
        selected_rows: list[pd.DataFrame] = []
    
        # 1) Disagreement (unit-level, excluding seen + already-picked)
        dis_units = _empty_unit_frame()
        dis_path = os.path.join(self.paths.outdir, "bucket_disagreement.parquet")
        if self.repo.ann.empty:
            print("[1/4] Skipping disagreement bucket (no prior rounds or quota is zero)")
        else:
            if n_dis > 0:
                print("[1/4] Expanded disagreement ...")
            else:
                print("[1/4] Disagreement quota is zero; refreshing schema only ...")
            check_cancelled()
            dis_pairs = self.build_disagreement_bucket(seen_pairs, legacy_rules_map, legacy_label_types)
            # Keep prior-round units eligible for disagreement reviews while still
            # preventing duplicates within the current batch.
            dis_pairs = _filter_units(dis_pairs, selected_units)
            dis_units = _head_units(_to_unit_only(dis_pairs), n_dis)
        dis_units.to_parquet(dis_path, index=False)
        selected_rows.append(dis_units)
        selected_units |= set(dis_units["unit_id"])

        # 2) Diversity (exclude seen + already-picked via both unseen_pairs filter and seed set)
        print("[2/4] Diversity ...")
        check_cancelled()
        want_div = min(n_div, max(0, total - len(selected_units)))
        fam = FamilyLabeler(self.llm, self.rag, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
        sel_div_pairs = build_diversity_bucket(
            unseen_pairs=unseen_pairs_current,
            already_selected=[(r.unit_id, getattr(r, "label_id", "")) for r in dis_units.itertuples(index=False)],
            n_div=want_div,
            pooler=self.pooler,
            retriever=self.rag,
            rules_map=current_rules_map,
            label_types=current_label_types,
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
        if n_unc > 0:
            print("[3/4] LLM-uncertain ...")
            check_cancelled()
            want_unc = min(n_unc, max(0, total - len(selected_units)))
            if want_unc > 0:
                sel_unc_pairs = self.build_llm_uncertain_bucket(
                    current_label_types, current_rules_map,
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
        if n_cer > 0:
            print("[4/4] LLM-certain ...")
            check_cancelled()
            want_cer = min(n_cer, max(0, total - len(selected_units)))
            if want_cer > 0:
                sel_cer_pairs = self.build_llm_certain_bucket(
                    current_label_types, current_rules_map,
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
            unseen_pairs_topoff = [
                (u, l) for (u, l) in self.build_unseen_pairs(label_ids=current_label_ids) if u not in excluded
            ]
            final = self.top_off_random(
                current_sel=final,
                unseen_pairs=unseen_pairs_topoff,
                label_types=current_label_types,
                target_n=total,
            ).drop_duplicates(subset=["unit_id"], keep="first")
    
        final.to_parquet(os.path.join(self.paths.outdir, "final_selection.parquet"), index=False)
        result_df = final

        # (Optional) run family labeling on the final units for transparency/audit
        final_out = None
        if self.cfg.final_llm_labeling:
            fam_rows = []
            fam = FamilyLabeler(self.llm, self.rag, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
            unit_ids = final["unit_id"].tolist()
            rules_map = current_rules_map
            types = current_label_types
            _progress_every = float(fam.cfg.progress_min_interval_s or 1)
            for uid in iter_with_bar(
                    step="Final family labeling",
                    iterable=unit_ids,
                    total=len(unit_ids),
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

                labels_path = Path(self.paths.outdir) / "final_llm_labels.parquet"
                try:
                    fam_wide.to_parquet(labels_path, index=False)
                except Exception:
                    pass
                else:
                    try:
                        labels_path.with_suffix(".json").write_text(
                            fam_wide.to_json(orient="records", indent=2, force_ascii=False),
                            encoding="utf-8",
                        )
                    except TypeError:
                        labels_path.with_suffix(".json").write_text(
                            fam_wide.to_json(orient="records"),
                            encoding="utf-8",
                        )

            try:
                probe_json_path = Path(self.paths.outdir) / "final_llm_family_probe.json"
                fam_df.to_json(probe_json_path, orient="records", indent=2, force_ascii=False)
            except TypeError:
                fam_df.to_json(probe_json_path, orient="records")
        if final_out is not None:
            result_df = final_out
        
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

        result_df = self._attach_unit_metadata(result_df)
        return result_df
        
        
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
