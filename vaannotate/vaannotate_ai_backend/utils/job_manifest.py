from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def read_manifest(path: Path) -> dict:
    """Best-effort loader for small JSON job manifests."""

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        LOGGER.warning("Manifest at %s is not a JSON object (got %s)", path, type(data).__name__)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read manifest %s: %s", path, exc)
    return {}


def write_manifest_atomic(path: Path, data: dict) -> None:
    """Persist a manifest atomically, logging and swallowing any errors."""

    payload = dict(data or {})
    payload["saved_at"] = time.time()
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to write manifest %s: %s", path, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
