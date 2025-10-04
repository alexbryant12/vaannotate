"""Helpers for working with document metadata."""
from __future__ import annotations

from typing import Dict, Mapping, Any


_METADATA_EXCLUDE_KEYS = {
    "doc_id",
    "hash",
    "text",
    "order_index",
    "documents",
    "metadata",
    "metadata_json",
}


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def extract_document_metadata(source: Mapping[str, object] | None) -> Dict[str, object]:
    """Return a normalized metadata dictionary for a document payload.

    The helper merges explicit ``metadata`` dictionaries with any scalar
    attributes present on the document payload while skipping core fields like
    the note text and document identifier. Empty strings and ``None`` values
    are omitted so the caller receives only meaningful metadata entries.
    """

    metadata: Dict[str, object] = {}
    if not source:
        return metadata
    raw_metadata = source.get("metadata") if isinstance(source, Mapping) else None
    if isinstance(raw_metadata, Mapping):
        for key, value in raw_metadata.items():
            if _is_meaningful(value):
                metadata[key] = value
    for key, value in source.items():
        if key in _METADATA_EXCLUDE_KEYS:
            continue
        if isinstance(value, Mapping):
            continue
        if isinstance(value, (list, tuple, set)):
            continue
        if not _is_meaningful(value):
            continue
        metadata.setdefault(key, value)
    return metadata
