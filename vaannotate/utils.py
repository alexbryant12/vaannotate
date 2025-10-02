"""Utility helpers for VAAnnotate."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Ensure directory exists and return its :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def canonical_json(data: object) -> str:
    """Return a deterministic JSON representation."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Sequence | str | bytes) -> str:
    """Return SHA256 hash built from provided parts."""
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            hasher.update(part)
        elif isinstance(part, str):
            hasher.update(part.encode("utf-8"))
        elif isinstance(part, Iterable):
            for item in part:
                hasher.update(str(item).encode("utf-8"))
        else:
            hasher.update(str(part).encode("utf-8"))
    return hasher.hexdigest()


def deterministic_choice(items: Sequence, seed: int | str) -> list:
    """Return shuffled copy of items using deterministic seed."""
    rnd = random.Random()
    if isinstance(seed, str):
        seed_int = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % (2**32)
    else:
        seed_int = seed
    rnd.seed(seed_int)
    copy = list(items)
    rnd.shuffle(copy)
    return copy


def validate_file_exists(path: os.PathLike[str] | str) -> Path:
    """Raise ``FileNotFoundError`` if file missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def copy_sqlite_database(
    source: os.PathLike[str] | str, target: os.PathLike[str] | str
) -> Path:
    """Safely copy a SQLite database including WAL contents."""

    src = Path(source)
    dst = Path(target)
    ensure_dir(dst.parent)
    with sqlite3.connect(src) as source_conn:
        source_conn.execute("PRAGMA wal_checkpoint(FULL)")
        with sqlite3.connect(dst) as dest_conn:
            source_conn.backup(dest_conn)
    try:
        shutil.copystat(src, dst, follow_symlinks=False)
    except OSError:
        # Copying metadata can fail on some filesystems; ignore best-effort.
        pass
    return dst
