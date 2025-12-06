"""Lightweight I/O helpers for tables and numeric arrays."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    ext = path.lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(path)
    if ext == "tsv":
        return pd.read_csv(path, sep="\t")
    if ext in ("parquet", "pq"):
        return pd.read_parquet(path)
    if ext == "jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table extension: {path}")


def atomic_write_bytes(path: str, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_table(df: pd.DataFrame, path: str):
    ext = path.lower().split(".")[-1]
    if ext == "csv":
        df.to_csv(path, index=False)
    elif ext in ("parquet", "pq"):
        df.to_parquet(path, index=False)
    elif ext == "jsonl":
        df.to_json(path, lines=True, orient="records", force_ascii=False)
    else:
        raise ValueError(f"Unsupported table extension: {path}")


def normalize01(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    mn, mx = a.min(), a.max()
    if mx <= mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


__all__ = ["atomic_write_bytes", "normalize01", "read_table", "write_table"]
