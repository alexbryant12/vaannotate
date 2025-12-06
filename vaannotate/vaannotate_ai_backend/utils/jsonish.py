"""Helpers for JSON-friendly column handling and parsing."""

from __future__ import annotations

import json
from typing import List


def _jsonify_cols(df, cols: List[str]):
    if df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
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


__all__ = ["_jsonify_cols", "_maybe_parse_jsonish"]
