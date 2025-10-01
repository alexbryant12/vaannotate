"""Utility helpers for VAAnnotate."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

NEWLINE_PATTERN = re.compile(r"\r\n?|\f")
WHITESPACE_PATTERN = re.compile(r"[\t\x0b\x0c\r]+")


def canonicalize_text(text: str) -> str:
    """Canonicalize clinical note text according to the specification.

    - Normalizes newlines to ``\n``.
    - Collapses miscellaneous whitespace to single spaces while preserving
      intentional double spacing already using spaces.
    - Ensures the text is NFC normalized.
    """

    import unicodedata

    normalized = unicodedata.normalize("NFC", text)
    normalized = NEWLINE_PATTERN.sub("\n", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized


def text_hash(text: str) -> str:
    """Return a SHA256 hash for ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create ``path`` if it does not exist and return a :class:`Path`."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def utcnow_ts() -> str:
    """Return a UTC timestamp string with second precision."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def deterministic_seed(*parts: object, base_seed: int | None = None) -> int:
    """Combine ``parts`` into a deterministic seed.

    Args:
        *parts: Arbitrary objects that will be JSON encoded.
        base_seed: Optional integer seed that is mixed in.
    """

    payload = json.dumps([base_seed, *parts], sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def shuffled(seq: Sequence[object], seed: int) -> list[object]:
    """Return a deterministically shuffled copy of ``seq`` using ``seed``."""

    rng = random.Random(seed)
    copy = list(seq)
    rng.shuffle(copy)
    return copy


def chunks(values: Sequence[object], n: int) -> Iterable[list[object]]:
    """Yield ``values`` split into ``n`` near-equal chunks preserving order."""

    if n <= 0:
        raise ValueError("chunk count must be positive")
    length = len(values)
    base, extra = divmod(length, n)
    start = 0
    for idx in range(n):
        size = base + (1 if idx < extra else 0)
        yield list(values[start : start + size])
        start += size


def stable_write(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically."""

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(path)


def stable_json_dump(path: Path, payload: Mapping[str, object]) -> None:
    """Serialize ``payload`` as pretty JSON at ``path`` atomically."""

    text = json.dumps(payload, indent=2, sort_keys=True)
    stable_write(path, text + "\n")
