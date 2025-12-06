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
import os, re, json, math, time, random, hashlib, unicodedata
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional, Any, Mapping
import numpy as np
import logging
import pandas as pd
from .core.data import DataRepository
from .core.embeddings import (
    EmbeddingStore,
    IndexConfig,
    Models,
    build_models_from_env,
)
from .services import ContextBuilder, LLMLabeler, LLM_RECORDER
from .services.disagreement import DisagreementScorer
from .label_configs import EMPTY_BUNDLE, LabelConfigBundle
from .llm_backends import build_llm_backend
from .core.retrieval import RetrievalCoordinator, SemanticQuery

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
        
"""
Note: Embedding/index primitives (IndexConfig, Models, EmbeddingStore) now live
in ``core.embeddings``.
"""

# ------------------------------
# Config
# ------------------------------

@dataclass
class RAGConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 150
    normalize_embeddings: bool = True
    per_label_topk: int = 6
    top_k_final: int = 6
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    mmr_candidates: int = 200
    use_keywords: bool = True
    keyword_topk: int = 20
    keyword_fraction: float = 0.3
    keywords: List[str] = field(default_factory=list)
    label_keywords: dict[str, list[str]] = field(default_factory=dict)
    min_context_chunks: int = 3
    mmr_multiplier: int = 3
    neighbor_hops: int = 1
    pool_factor: int = 3
    pool_oversample: float = 1.5
        
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


class DisagreementExpander:
    def __init__(
        self,
        cfg: DisagreementConfig,
        repo: DataRepository,
        retriever: RAGRetriever,
        label_config_bundle: LabelConfigBundle | None = None,
        llmfirst_cfg: LLMFirstConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ):
        self.cfg = cfg
        self.repo = repo
        self.retriever = retriever
        self.label_config_bundle = label_config_bundle or EMPTY_BUNDLE
        self._dependency_cache: Dict[str, tuple[dict, dict, list]] = {}
        self.llmfirst_cfg = llmfirst_cfg
        self.context_builder = context_builder

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
        if self.context_builder is not None:
            ctx = self.context_builder.build_context_for_label(
                unit_id,
                label_id,
                label_rules,
                topk_override=self.cfg.snippets_per_seed,
                single_doc_context_mode=getattr(self.llmfirst_cfg, "single_doc_context", "rag"),
                full_doc_char_limit=getattr(self.llmfirst_cfg, "single_doc_full_context_max_chars", None),
            )
        else:
            ctx = self.retriever.retrieve_for_patient_label(
                unit_id,
                label_id,
                label_rules,
                topk_override=self.cfg.snippets_per_seed,
                min_k_override=None,
                mmr_lambda_override=None,
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

        diagnostics["retrieval_coordinator"] = pool_diag
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
    def __init__(self, llm: LLMLabeler, retriever: RAGRetriever, repo: DataRepository, label_config: dict, scCfg: SCJitterConfig, llmfirst_cfg: LLMFirstConfig):
        self.llm = llm
        self.retriever = retriever
        self.repo = repo
        self.label_config = label_config or {}
        self.scCfg = scCfg
        self.cfg = llmfirst_cfg
        self.context_builder = getattr(retriever, "context_builder", None)
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
            label_rows = self.llm.label_unit(
                unit_id,
                [lid],
                label_types=label_types,
                per_label_rules=per_label_rules,
                context_builder=self.context_builder,
                retriever=self.retriever,
                llmfirst_cfg=self.cfg,
                json_only=json_only,
                json_n_consistency=json_n_consistency,
                json_jitter=json_jitter,
            )
            if not label_rows:
                continue
            row = label_rows[0]
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
                ctx = self.context_builder.build_context_for_label(
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
        vecs = [
            pooler.pooled_vector(
                r.unit_id,
                r.label_id,
                retriever,
                rules_map.get(r.label_id, ""),
                context_builder=self.context_builder,
            )
            for r in grp.itertuples(index=False)
        ]
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

    vecs = [
        pooler.pooled_vector(
            r.unit_id,
            r.label_id,
            retriever,
            rules_map.get(r.label_id, ""),
            context_builder=context_builder,
        )
        for r in rem.itertuples(index=False)
    ]
    V = np.vstack(vecs)

    preV = None
    if already_selected_vecs is not None and already_selected_vecs.size:
        preV = already_selected_vecs
    if not selected.empty:
        Vsel = [
            pooler.pooled_vector(
                r.unit_id,
                r.label_id,
                retriever,
                rules_map.get(r.label_id, ""),
                context_builder=context_builder,
            )
            for r in selected.itertuples(index=False)
        ]
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
        expander = DisagreementExpander(
            self.cfg.disagree,
            self.repo,
            self.rag,
            label_config_bundle=self.label_config_bundle,
            llmfirst_cfg=self.cfg.llmfirst,
            context_builder=self.context_builder,
        )
        scorer = DisagreementScorer(
            self.repo,
            self.cfg.disagree,
            self.cfg.select,
            self.pooler,
            self.rag,
            expander,
            context_builder=self.context_builder,
            kcenter_select_fn=kcenter_select,
        )
        return scorer.compute_disagreement(seen_pairs, rules_map, label_types)

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
