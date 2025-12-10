from __future__ import annotations

"""Utilities for constructing RAG contexts for unit/label pairs."""

from typing import Dict, Mapping

from ..core.data import DataRepository
from ..core.embeddings import EmbeddingStore
from ..label_configs import LabelConfigBundle


_FULL_DOC_CONTEXT_FALLBACK_CHARS = 12000


class ContextBuilder:
    """Canonical builder for RAG contexts spanning notes, annotations, and label metadata."""

    def __init__(
        self,
        data_repo: DataRepository,
        emb_store: EmbeddingStore,
        retriever,
        rag_config,
        label_config_bundle: LabelConfigBundle,
    ):
        self.repo = data_repo
        self.store = emb_store
        self.retriever = retriever
        self.cfg = rag_config
        self.label_config_bundle = label_config_bundle

    def build_context_for_label(
        self,
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
        if getattr(self.repo, "phenotype_level", "").strip().lower() == "single_doc":
            resolver = getattr(self.repo, "doc_id_for_unit", None)
            if callable(resolver):
                resolved = resolver(unit_id)
                if resolved:
                    resolved_unit_id = str(resolved)
        if self.repo.phenotype_level == "single_doc" and mode == "full":
            doc_id = resolved_unit_id
            text = self.repo.notes_by_doc().get(doc_id)
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
                idxs = self.retriever.store.get_patient_chunk_indices(doc_id)
            except Exception:
                idxs = []
            if idxs:
                try:
                    chunk_meta = self.retriever.store.chunk_meta[idxs[0]]
                    metadata = self.retriever._extract_meta(chunk_meta) or {}
                except Exception:
                    metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("other_meta", "")

            try:
                self.retriever.set_last_diagnostics(
                    resolved_unit_id,
                    label_id,
                    {
                        "rag_mode": "full_doc",
                        "unit_id": resolved_unit_id,
                        "label_id": str(label_id),
                        "stage": "full_doc",
                        "final_selection": {"count": 1, "score_stats": {"min": 1.0, "max": 1.0, "mean": 1.0}},
                    },
                    original_unit_id=str(unit_id),
                )
            except Exception:
                pass

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

        snippets = self.retriever.retrieve_for_patient_label(
            unit_id=resolved_unit_id,
            label_id=label_id,
            label_rules=label_rules,
            topk_override=topk_override,
            min_k_override=min_k_override,
            mmr_lambda_override=mmr_lambda_override,
            original_unit_id=str(unit_id),
        )

        if snippets is None:
            return []
        if not isinstance(snippets, list):
            try:
                snippets = list(snippets)
            except Exception:
                return []

        return snippets

    def build_context_for_family(
        self,
        unit_id: str,
        label_ids: list[str],
        rules_map: Mapping[str, str],
        *,
        topk_per_label: int | None = None,
        max_snippets: int | None = None,
        max_chars: int | None = None,
    ) -> list[dict]:
        """
        Build a single merged context for a unit across many labels.

        - For each label_id in label_ids:
            - Look up rules_text = rules_map.get(label_id, "")
            - Call build_context_for_label(unit_id, label_id, rules_text, ...)
            - Take up to topk_per_label snippets (if provided), otherwise keep all.
        - Deduplicate snippets across labels (e.g. by (doc_id, chunk_id) or another stable key in the snippet dict).
        - Sort snippets in a stable way (e.g. by descending score, then by chronological metadata if available).
        - Apply max_snippets to truncate the list if provided.
        - Apply a character budget if max_chars is not None, by walking the sorted snippets and stopping when the cumulative length of snippet["text"] (or their rendered form) exceeds max_chars.
        - Return the final list of snippet dicts, retaining the same shape as build_context_for_label.
        """

        collected: list[dict] = []
        topk: int | None = None
        if topk_per_label is not None:
            try:
                topk = max(0, int(topk_per_label))
            except Exception:
                topk = None

        for label_id in label_ids:
            snippets = self.build_context_for_label(unit_id, label_id, rules_map.get(label_id, ""))
            if topk is not None:
                snippets = snippets[:topk]
            collected.extend(snippets)

        seen: dict[tuple[object, object], dict] = {}
        ordered: list[dict] = []
        for snip in collected:
            key = (snip.get("doc_id"), snip.get("chunk_id"))
            if key in seen:
                continue
            seen[key] = snip
            ordered.append(snip)

        def _score_key(item: dict, idx: int) -> tuple:
            raw_score = item.get("score")
            try:
                score = float(raw_score)
            except Exception:
                score = 0.0
            return (-score, idx)

        scored = sorted(enumerate(ordered), key=lambda t: _score_key(t[1], t[0]))
        sorted_snippets = [ordered[idx] for idx, _ in scored]

        if max_snippets is not None:
            try:
                limit = int(max_snippets)
            except Exception:
                limit = None
            if limit is not None and limit >= 0:
                sorted_snippets = sorted_snippets[:limit]

        if max_chars is not None:
            try:
                char_budget = int(max_chars)
            except Exception:
                char_budget = None
            if char_budget is not None and char_budget > 0:
                total = 0
                budgeted: list[dict] = []
                for snip in sorted_snippets:
                    text = snip.get("text", "")
                    length = len(text) if isinstance(text, str) else len(str(text))
                    if total + length > char_budget:
                        break
                    budgeted.append(snip)
                    total += length
                sorted_snippets = budgeted

        return sorted_snippets

def _options_for_label(label_id: str, label_type: str, label_config: dict) -> list[str]:
    cfg = label_config.get(label_id, {}) if isinstance(label_config, dict) else {}
    if label_type in {"categorical", "categorical_multi"}:
        return cfg.get("options", []) or []
    elif label_type == "binary":
        return ["yes", "no"]
    return []
