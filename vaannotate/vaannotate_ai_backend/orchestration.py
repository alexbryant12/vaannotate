"""Helpers to assemble active learning and inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence

from .config import OrchestratorConfig, Paths
from .core.data import DataRepository
from .core.embeddings import EmbeddingStore, Models, build_models_from_env
from .label_configs import LabelConfigBundle
from .llm_backends import build_llm_backend
from .pipelines.active_learning import ActiveLearningPipeline
from .pipelines.inference import InferencePipeline
from .services import (
    ActiveLearningSelector,
    ContextBuilder,
    DiversitySelector,
    DisagreementScorer,
    LLMLabeler,
    LLMUncertaintyScorer,
)
from .services.disagreement_expander import DisagreementExpander
from .services.pooling import LabelAwarePooler, kcenter_select
from .services.rag_retriever import RAGRetriever
from .utils.io import read_table
from .utils.jsonish import _jsonify_cols
from .utils.runtime import check_cancelled, iter_with_bar


def _build_shared_components(
    paths: Paths,
    cfg: OrchestratorConfig,
    label_config_bundle: LabelConfigBundle,
    phenotype_level: str | None = None,
    *,
    include_pooler: bool = False,
    models: Models | None = None,
    store: EmbeddingStore | None = None,
):
    notes_df = read_table(paths.notes_path)
    ann_df = read_table(paths.annotations_path)
    repo = DataRepository(notes_df, ann_df, phenotype_level=phenotype_level)

    if models is None:
        models = build_models_from_env(cfg.models)
    if store is None:
        store = EmbeddingStore(
            models,
            cache_dir=paths.cache_dir,
            normalize=cfg.rag.normalize_embeddings,
        )

    # If the LLM config does not already include few-shot examples, fall back to
    # any examples embedded in the current label configuration. This keeps the
    # prompting layer aligned with label-set metadata for both active learning
    # and inference pipelines.
    few_shot_cfg = getattr(cfg.llm, "few_shot_examples", {}) or {}
    few_shot_explicit = getattr(cfg.llm, "_few_shot_examples_overridden", False)
    if not few_shot_cfg and not few_shot_explicit:
        label_config = label_config_bundle.current or {}
        extracted: dict[str, list[dict[str, str]]] = {}
        for label_id, entry in label_config.items():
            if str(label_id) == "_meta" or not isinstance(entry, Mapping):
                continue
            raw_examples = entry.get("few_shot_examples")
            if not isinstance(raw_examples, Sequence):
                continue
            parsed: list[dict[str, str]] = []
            for example in raw_examples:
                if not isinstance(example, Mapping):
                    continue
                payload: dict[str, str] = {}
                if example.get("context") is not None:
                    payload["context"] = str(example.get("context"))
                if example.get("answer") is not None:
                    payload["answer"] = str(example.get("answer"))
                if payload:
                    parsed.append(payload)
            if parsed:
                key = str(entry.get("label_id") or label_id)
                if key:
                    extracted[key] = parsed
        if extracted:
            try:
                setattr(cfg.llm, "few_shot_examples", extracted)
            except Exception:
                pass
    rag = RAGRetriever(
        store,
        models,
        cfg.rag,
        label_configs=label_config_bundle.current or {},
        notes_by_doc=repo.notes_by_doc(),
        repo=repo,
    )
    context_builder = ContextBuilder(repo, store, rag, cfg.rag, label_config_bundle)
    try:
        rag.context_builder = context_builder
    except Exception:
        pass

    backend = build_llm_backend(cfg.llm)
    llm_labeler = LLMLabeler(
        backend,
        label_config_bundle,
        cfg.llm,
        sc_cfg=cfg.scjitter,
        cache_dir=paths.cache_dir,
    )
    llm_labeler.label_config = label_config_bundle.current or {}

    pooler = None
    if include_pooler:
        pooler = LabelAwarePooler(
            repo,
            store,
            models,
            beta=float(os.getenv("POOLER_BETA", 5.0)),
            kmeans_k=int(os.getenv("POOLER_K", 8)),
            persist_dir=os.path.join(paths.cache_dir, "prototypes"),
            version="v1",
            use_negative=bool(int(os.getenv("POOLER_USE_NEG", "0"))),
            llmfirst_cfg=cfg.llmfirst,
        )

    return {
        "repo": repo,
        "store": store,
        "models": models,
        "rag": rag,
        "context_builder": context_builder,
        "llm_labeler": llm_labeler,
        "pooler": pooler,
        "label_config": label_config_bundle.current or {},
    }


def _build_rerank_rule_overrides(
    llm_labeler: LLMLabeler, rules_map: Mapping[str, str]
) -> dict[str, str]:
    if not llm_labeler or not isinstance(rules_map, Mapping):
        return {}
    threshold_default = 150
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
            summary = llm_labeler.summarize_label_rule_for_rerank(str(label_id), text)
        except Exception:
            continue
        concise = str(summary or "").strip()
        if concise and len(concise) < len(text):
            overrides[str(label_id)] = concise
    return overrides


def _attach_unit_metadata(repo: DataRepository, df):
    if df is None or repo is None:
        return df
    meta = repo.unit_metadata()
    if meta.empty:
        return df
    meta = meta[[c for c in meta.columns if c == "unit_id" or c not in df.columns]]
    if meta.shape[1] <= 1:
        return df
    return df.merge(meta, on="unit_id", how="left")


def build_active_learning_runner(
    paths: Paths,
    cfg: OrchestratorConfig,
    label_config_bundle: LabelConfigBundle,
    phenotype_level: str | None = None,
    *,
    models: Models | None = None,
    store: EmbeddingStore | None = None,
) -> ActiveLearningPipeline:
    components = _build_shared_components(
        paths,
        cfg,
        label_config_bundle,
        phenotype_level=phenotype_level,
        include_pooler=True,  # active learning uses pooler
        models=models,
        store=store,
    )

    repo = components["repo"]
    store = components["store"]
    rag = components["rag"]
    context_builder = components["context_builder"]
    llm_labeler = components["llm_labeler"]
    pooler = components["pooler"]
    label_config = components["label_config"]

    diversity_selector = DiversitySelector(repo, store, cfg.diversity)
    expander = DisagreementExpander(
        cfg.disagree,
        repo,
        rag,
        label_config_bundle=label_config_bundle,
        llmfirst_cfg=cfg.llmfirst,
        context_builder=context_builder,
    )
    disagreement_scorer = DisagreementScorer(
        repo,
        cfg.disagree,
        cfg.select,
        pooler,
        rag,
        expander,
        context_builder=context_builder,
        kcenter_select_fn=kcenter_select,
    )
    uncertainty_scorer = LLMUncertaintyScorer(cfg.llmfirst)
    selector = ActiveLearningSelector(cfg.select)
    excluded_unit_ids = set(getattr(cfg, "excluded_unit_ids", []) or [])

    return ActiveLearningPipeline(
        data_repo=repo,
        emb_store=store,
        ctx_builder=context_builder,
        llm_labeler=llm_labeler,
        disagreement_scorer=disagreement_scorer,
        uncertainty_scorer=uncertainty_scorer,
        diversity_selector=diversity_selector,
        selector=selector,
        config=cfg,
        paths=paths,
        pooler=pooler,
        retriever=rag,
        label_config=label_config,
        label_config_bundle=label_config_bundle,
        excluded_unit_ids=excluded_unit_ids,
        check_cancelled_fn=check_cancelled,
        iter_with_bar_fn=iter_with_bar,
        jsonify_cols_fn=_jsonify_cols,
        attach_metadata_fn=lambda df: _attach_unit_metadata(repo, df),
        rerank_override_fn=lambda rules_map: _build_rerank_rule_overrides(llm_labeler, rules_map),
    )


def build_inference_runner(
    paths: Paths,
    cfg: OrchestratorConfig,
    label_config_bundle: LabelConfigBundle,
    phenotype_level: str | None = None,
    *,
    models: Models | None = None,
    store: EmbeddingStore | None = None,
) -> InferencePipeline:
    components = _build_shared_components(
        paths,
        cfg,
        label_config_bundle,
        phenotype_level=phenotype_level,
        include_pooler=False,  # inference does not use pooler
        models=models,
        store=store,
    )

    return InferencePipeline(
        data_repo=components["repo"],
        emb_store=components["store"],
        ctx_builder=components["context_builder"],
        llm_labeler=components["llm_labeler"],
        config=cfg,
        paths=paths,
    )


@dataclass
class BackendSession:
    """Lightweight holder for shared embedding models and stores.

    This is useful for parameter sweeps or sharded inference runs where you
    want to reuse the same Models and EmbeddingStore across multiple
    ActiveLearning or Inference pipelines.
    """

    models: Models
    store: EmbeddingStore

    @classmethod
    def from_env(cls, paths: Paths, cfg: OrchestratorConfig) -> "BackendSession":
        """Build a session by loading models from the current environment.

        This mirrors the logic used in _build_shared_components for
        constructing the EmbeddingStore, but does not instantiate any
        downstream components (retriever, labeler, etc.).
        """
        models = build_models_from_env(cfg.models)
        store = EmbeddingStore(
            models,
            cache_dir=paths.cache_dir,
            normalize=cfg.rag.normalize_embeddings,
        )
        return cls(models=models, store=store)

    def build_active_learning_pipeline(
        self,
        paths: Paths,
        cfg: OrchestratorConfig,
        label_config_bundle: LabelConfigBundle,
        phenotype_level: str | None = None,
    ) -> ActiveLearningPipeline:
        """Construct an ActiveLearningPipeline reusing this session's models."""
        return build_active_learning_runner(
            paths=paths,
            cfg=cfg,
            label_config_bundle=label_config_bundle,
            phenotype_level=phenotype_level,
            models=self.models,
            store=self.store,
        )

    def build_inference_pipeline(
        self,
        paths: Paths,
        cfg: OrchestratorConfig,
        label_config_bundle: LabelConfigBundle,
        phenotype_level: str | None = None,
    ) -> InferencePipeline:
        """Construct an InferencePipeline reusing this session's models."""
        return build_inference_runner(
            paths=paths,
            cfg=cfg,
            label_config_bundle=label_config_bundle,
            phenotype_level=phenotype_level,
            models=self.models,
            store=self.store,
        )


__all__ = [
    "build_active_learning_runner",
    "build_inference_runner",
    "BackendSession",
]
