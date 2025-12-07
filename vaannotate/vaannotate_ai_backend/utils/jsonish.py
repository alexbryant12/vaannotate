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
                lambda x: _jsonify_value(x)
            )
    return out


def _jsonify_value(value):
    """Normalize structured values to JSON strings.

    PyArrow/parquet writes will fail when a column mixes structured objects
    (lists, dicts, tuples, sets) with plain scalars. Converting the structured
    values to JSON strings keeps the column homogenous and avoids the
    "cannot mix struct and non-struct" error.
    """

    # Preserve common scalars and null-ish values unchanged
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Normalize common container types into JSON strings
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, tuple):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, set):
        return json.dumps(sorted(value), ensure_ascii=False)

    # Fallback: stringify any other object to avoid struct/scalar mixing
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


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
