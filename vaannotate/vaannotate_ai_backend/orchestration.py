"""Helpers to assemble active learning and inference pipelines."""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional

from .config import OrchestratorConfig, Paths
from .core.data import DataRepository
from .core.embeddings import EmbeddingStore, build_models_from_env
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
):
    notes_df = read_table(paths.notes_path)
    ann_df = read_table(paths.annotations_path)
    repo = DataRepository(notes_df, ann_df, phenotype_level=phenotype_level)

    models = build_models_from_env()
    store = EmbeddingStore(models, cache_dir=paths.cache_dir, normalize=cfg.rag.normalize_embeddings)
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


def _label_maps(bundle: LabelConfigBundle, label_config: Mapping[str, object]):
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
            "decimal": "numeric",
            "ordinal": "ordinal",
            "rank": "ordinal",
            "ranking": "ordinal",
            "date": "date",
            "datetime": "date",
            "timestamp": "date",
        }
        return mapping.get(text, "categorical")

    def _extract_rule_text(entry: Mapping[str, object] | None) -> Optional[str]:
        if not isinstance(entry, Mapping):
            return None
        for key in ("rule", "rules", "why", "query", "text"):
            val = entry.get(key)
            if isinstance(val, str):
                text = val.strip()
                if text:
                    return text
            elif isinstance(val, list):
                for item in reversed(val):
                    if isinstance(item, str):
                        text = item.strip()
                        if text:
                            return text
                    elif isinstance(item, Mapping):
                        text = str(item.get("text") or item.get("rule") or "").strip()
                        if text:
                            return text
            elif isinstance(val, Mapping):
                text = str(val.get("text") or val.get("rule") or "").strip()
                if text:
                    return text
        return None

    legacy_rules_map = bundle.legacy_rules_map()
    legacy_label_types = bundle.legacy_label_types()
    current_rules_map: Dict[str, str] = {}
    current_label_types: Dict[str, str] = {}

    for key, entry in (label_config or {}).items():
        if str(key) == "_meta":
            continue
        label_entry = entry if isinstance(entry, Mapping) else {}
        raw_id = label_entry.get("label_id") if isinstance(label_entry, Mapping) else None
        label_id = str(raw_id or key).strip()
        if not label_id:
            continue

        rule_text = _extract_rule_text(label_entry) if isinstance(label_entry, Mapping) else None
        if rule_text is not None:
            current_rules_map[label_id] = rule_text
        elif label_id not in current_rules_map:
            current_rules_map[label_id] = ""

        normalized_type = _normalize_type(label_entry.get("type") if isinstance(label_entry, Mapping) else None)
        if normalized_type:
            current_label_types[label_id] = normalized_type
        elif label_id not in current_label_types:
            current_label_types[label_id] = "categorical"

    if not current_rules_map:
        current_rules_map = dict(legacy_rules_map)
    if not current_label_types:
        current_label_types = dict(legacy_label_types)

    return legacy_rules_map, legacy_label_types, current_rules_map, current_label_types


def build_active_learning_runner(
    paths: Paths,
    cfg: OrchestratorConfig,
    label_config_bundle: LabelConfigBundle,
    phenotype_level: str | None = None,
) -> ActiveLearningPipeline:
    components = _build_shared_components(
        paths,
        cfg,
        label_config_bundle,
        phenotype_level=phenotype_level,
        include_pooler=True,
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
        label_maps_fn=lambda: _label_maps(label_config_bundle, label_config),
        rerank_override_fn=lambda rules_map: _build_rerank_rule_overrides(llm_labeler, rules_map),
    )


def build_inference_runner(
    paths: Paths,
    cfg: OrchestratorConfig,
    label_config_bundle: LabelConfigBundle,
    phenotype_level: str | None = None,
) -> InferencePipeline:
    components = _build_shared_components(
        paths,
        cfg,
        label_config_bundle,
        phenotype_level=phenotype_level,
        include_pooler=False,
    )

    return InferencePipeline(
        data_repo=components["repo"],
        emb_store=components["store"],
        ctx_builder=components["context_builder"],
        llm_labeler=components["llm_labeler"],
        config=cfg,
        paths=paths,
    )


__all__ = [
    "build_active_learning_runner",
    "build_inference_runner",
]
