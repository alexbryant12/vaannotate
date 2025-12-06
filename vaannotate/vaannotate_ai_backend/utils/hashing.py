"""Stable hashing utilities for reproducible identifiers."""

from __future__ import annotations

import hashlib
import json


def stable_hash_str(s: str, digest_size: int = 8) -> str:
    if s is None:
        s = ""
    return hashlib.blake2b(str(s).encode("utf-8"), digest_size=digest_size).hexdigest()


def stable_hash_pair(a: str, b: str, digest_size: int = 12) -> str:
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    return hashlib.blake2b((a + "\x1f" + b).encode("utf-8"), digest_size=digest_size).hexdigest()


def _stable_rules_hash(label_id: str, rules: str, K: int, model_sig: str = ""):
    payload = {
        "label": label_id,
        "rules": rules or "",
        "K": int(K),
        "model": model_sig,
    }
    return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=12).hexdigest()


__all__ = [
    "stable_hash_pair",
    "stable_hash_str",
    "_stable_rules_hash",
]
