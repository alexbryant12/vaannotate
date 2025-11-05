from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Tuple
import contextlib
import io
import logging
import sys
import pandas as pd

from . import engine


def _default_paths(outdir: Path, cache_dir: Path | None = None) -> engine.Paths:
    notes_path = outdir / "notes.parquet"
    annotations_path = outdir / "annotations.parquet"
    return engine.Paths(
        notes_path=str(notes_path),
        annotations_path=str(annotations_path),
        outdir=str(outdir),
        cache_dir_override=str(cache_dir) if cache_dir else None,
    )


def _apply_overrides(target: object, overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            current = getattr(target, key, None)
            if current is not None and not isinstance(current, (str, bytes, int, float, bool)):
                _apply_overrides(current, value)
            else:
                setattr(target, key, dict(value))
        else:
            setattr(target, key, value)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

class _CallbackTee(io.TextIOBase):
    """Write to the original stream and emit complete lines to a callback."""

    def __init__(self, stream: io.TextIOBase, callback: Callable[[str], None]):
        super().__init__()
        self._stream = stream
        self._callback = callback
        self._current_line = ""
        self._last_was_cr = False
        self._last_emitted_progress = ""

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        self._stream.write(data)
        self._stream.flush()

        text = data
        while text:
            next_newline = text.find("\n")
            next_return = text.find("\r")
            if next_newline == -1 and next_return == -1:
                self._current_line += text
                if self._last_was_cr and self._current_line.strip():
                    text_line = self._current_line.strip()
                    self._callback("\r" + text_line)
                    self._last_emitted_progress = text_line
                break

            if next_newline == -1 or (0 <= next_return < next_newline):
                idx = next_return
                marker = "\r"
            else:
                idx = next_newline
                marker = "\n"

            segment = text[:idx]
            text = text[idx + 1 :]
            if segment:
                self._current_line += segment
                if self._last_was_cr and self._current_line.strip():
                    text_line = self._current_line.strip()
                    self._callback("\r" + text_line)
                    self._last_emitted_progress = text_line

            if marker == "\n":
                if self._current_line.strip():
                    self._callback(self._current_line.strip())
                self._last_emitted_progress = ""
                self._current_line = ""
                self._last_was_cr = False
            else:  # carriage return → in-place update
                text_line = self._current_line.strip()
                if text_line and text_line != self._last_emitted_progress:
                    self._callback("\r" + text_line)
                    self._last_emitted_progress = text_line
                self._current_line = ""
                self._last_was_cr = True

        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        self._stream.flush()
        if self._current_line.strip():
            text_line = self._current_line.strip()
            if self._last_was_cr:
                if text_line != self._last_emitted_progress:
                    self._callback("\r" + text_line)
                    self._last_emitted_progress = text_line
            else:
                self._callback(text_line)
                self._last_emitted_progress = ""
        self._current_line = ""
        self._last_was_cr = False


class _CallbackHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            message = record.getMessage()
        message = message.strip()
        if not message:
            return

        step = getattr(record, "step", None)
        done = getattr(record, "done", None)
        total = getattr(record, "total", None)

        if step is not None and done is not None:
            self._callback("\r" + message)
            if total is not None and done == total:
                # Emit a trailing message without the carriage return so the
                # UI can finalize the in-place progress line without adding a
                # duplicate entry.
                self._callback(message)
            return

        self._callback(message)


@contextlib.contextmanager
def _capture_logs(callback: Optional[Callable[[str], None]]):
    if not callback:
        yield
        return

    stack = contextlib.ExitStack()
    stdout_tee = _CallbackTee(sys.__stdout__, callback)
    stderr_tee = _CallbackTee(sys.__stderr__, callback)
    stack.enter_context(contextlib.redirect_stdout(stdout_tee))
    stack.enter_context(contextlib.redirect_stderr(stderr_tee))

    handler = _CallbackHandler(callback)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()

    def _remove_handler() -> None:
        root_logger.removeHandler(handler)

    root_logger.addHandler(handler)
    try:
        yield
    finally:
        try:
            stack.close()
        finally:
            _remove_handler()


def build_next_batch(
    notes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    outdir: Path,
    label_config: Optional[dict] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    cache_dir: Path | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """High-level entrypoint consumed by AdminApp/RoundBuilder.
    Writes intermediates under outdir and returns (final_df, artifacts).
    """
    outdir = Path(outdir)
    _ensure_dir(outdir)
    cache_dir_path = Path(cache_dir) if cache_dir else None
    paths = _default_paths(outdir, cache_dir=cache_dir_path)
    # Persist inputs to the files expected by the engine
    notes_path = Path(paths.notes_path)
    ann_path = Path(paths.annotations_path)
    notes_df.to_parquet(notes_path, index=False)
    ann_df.to_parquet(ann_path, index=False)

    # Build config
    cfg = engine.OrchestratorConfig()
    if cfg_overrides:
        _apply_overrides(cfg, cfg_overrides)

    with _capture_logs(log_callback):
        with engine.cancellation_scope(cancel_callback):
            orch = engine.ActiveLearningLLMFirst(paths=paths, cfg=cfg, label_config=label_config or {})
            final_df = orch.run()  # engine returns DataFrame

    normalized = final_df.copy()
    rename_map = {}
    if "patienticn" in normalized.columns:
        rename_map["patienticn"] = "patient_icn"
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    csv_path = outdir / "ai_next_batch.csv"
    csv_columns = [
        col
        for col in [
            "unit_id",
            "patient_icn",
            "doc_id",
            "label_id",
            "selection_reason",
            "strata_key",
        ]
        if col in normalized.columns
    ]
    if csv_columns:
        normalized.to_csv(csv_path, columns=csv_columns, index=False)
    else:
        normalized.to_csv(csv_path, index=False)

    artifacts = {
        "ai_next_batch_csv": str(csv_path),
        "buckets": {
            "disagreement": str(Path(outdir) / "bucket_disagreement.parquet"),
            "llm_uncertain": str(Path(outdir) / "bucket_llm_uncertain.parquet"),
            "llm_certain": str(Path(outdir) / "bucket_llm_certain.parquet"),
            "diversity": str(Path(outdir) / "bucket_diversity.parquet"),
        },
        "final_labels": str(Path(outdir) / "final_llm_labels.parquet")
        if (Path(outdir) / "final_llm_labels.parquet").exists()
        else None,
    }
    return normalized, artifacts
