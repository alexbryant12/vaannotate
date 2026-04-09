"""Lightweight index configuration types for retrieval backends.

This module intentionally avoids importing heavyweight ML dependencies so it can
be imported safely by CLI/config code paths that do not need model loading.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndexConfig:
    type: str = "flat"    # flat | hnsw | ivf
    nlist: int = 2048     # IVF lists
    nprobe: int = 32      # IVF search probes
    hnsw_M: int = 32      # HNSW graph degree
    hnsw_efSearch: int = 64
    persist: bool = True
