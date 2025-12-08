"""Configuration dataclasses for AI backend pipelines and services."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .core.embeddings import IndexConfig


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class RAGConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 150
    normalize_embeddings: bool = True
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
    """Knobs for the LLM-first labeling pipeline.

    Attributes:
        inference_labeling_mode: how to label units during inference; "family"
            uses the existing FamilyLabeler, "single_prompt" will use a single
            multi-label LLM call per unit.
        single_prompt_max_labels: safety clamp on how many labels can be included
            in a single-prompt call.
        single_prompt_max_chars: approximate cap on total characters of the
            merged context passed to the LLM in single-prompt mode.
    """

    n_probe_units: int = 10
    topk: int = 6
    json_trace_policy: str = 'fallback'
    progress_min_interval_s: float = 10.0
    exemplar_K: int = 1
    exemplar_generate: bool = True
    exemplar_temperature: float = 0.9
    inference_labeling_mode: str = "family"  # "family" | "single_prompt"
    single_prompt_max_labels: int = 64
    single_prompt_max_chars: int = 16000
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
class ModelConfig:
    embed_model_name: str | None = None
    rerank_model_name: str | None = None


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
    models: ModelConfig = field(default_factory=ModelConfig)
    final_llm_labeling: bool = True
    final_llm_labeling_n_consistency: int = 1
    excluded_unit_ids: set[str] = field(default_factory=set)
