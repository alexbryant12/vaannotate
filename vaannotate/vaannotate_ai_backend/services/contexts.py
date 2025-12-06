from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from ..core.data import DataRepository


def _contexts_for_unit_label(
    retriever: Any,
    repo: DataRepository,
    unit_id: str,
    label_id: str,
    rules_text: str,
    *,
    topk_override: int | None = None,
    single_doc_context_mode: str = "rag",
    full_doc_char_limit: int | None = None,
) -> list[Mapping[str, Any] | MutableMapping[str, Any]]:
    """Build contexts for a unit/label pair.

    Supports two modes for single-doc phenotypes:
    - "full": return the entire note text (optionally truncated) with metadata
      derived from the first matching chunk.
    - "rag": delegate to the retriever for chunk-level contexts.

    For multi-document phenotypes, RAG retrieval is always used.
    """

    mode = str(single_doc_context_mode or "rag").strip().lower()
    unit_str = str(unit_id)

    doc_id = unit_str
    if getattr(repo, "phenotype_level", "multi_doc") == "single_doc":
        mapped = repo.doc_id_for_unit(unit_str)
        if mapped:
            doc_id = mapped

    if mode == "full" and getattr(repo, "phenotype_level", "multi_doc") == "single_doc":
        notes_by_doc = repo.notes_by_doc()
        text = str(notes_by_doc.get(doc_id, ""))
        if not text:
            return []

        limit = full_doc_char_limit
        if limit is not None:
            try:
                max_chars = int(limit)
                if max_chars > 0:
                    text = text[:max_chars]
            except Exception:
                pass

        metadata: dict[str, Any] = {}
        chunk_id = None
        try:
            for meta in getattr(getattr(retriever, "store", None), "chunk_meta", []) or []:
                if str(meta.get("doc_id")) == doc_id or str(meta.get("unit_id", "")) == doc_id:
                    chunk_id = meta.get("chunk_id")
                    try:
                        metadata = retriever._extract_meta(meta) if hasattr(retriever, "_extract_meta") else {}
                    except Exception:
                        metadata = {}
                    break
        except Exception:
            metadata = {}

        return [
            {
                "source": "full_doc",
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "metadata": metadata,
                "text": text,
            }
        ]

    contexts = retriever.retrieve_for_patient_label(
        doc_id,
        label_id,
        rules_text,
        topk_override=topk_override,
        original_unit_id=unit_str,
    )
    return contexts or []
