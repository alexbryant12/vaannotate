from __future__ import annotations
from collections.abc import Mapping
import copy
import os
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Tuple
from dataclasses import replace
import contextlib
import io
import logging
import sys
import pandas as pd

from .config import OrchestratorConfig, Paths
from .orchestration import build_active_learning_runner, build_inference_runner, BackendSession
from .utils.runtime import cancellation_scope
from .label_configs import EMPTY_BUNDLE, LabelConfigBundle


def _default_paths(outdir: Path, cache_dir: Path | None = None) -> Paths:
    notes_path = outdir / "notes.parquet"
    annotations_path = outdir / "annotations.parquet"
    return Paths(
        notes_path=str(notes_path),
        annotations_path=str(annotations_path),
        outdir=str(outdir),
        cache_dir_override=str(cache_dir) if cache_dir else None,
    )


def _apply_overrides(target: object, overrides: Mapping[str, Any]) -> None:
    if isinstance(target, dict):
        for key, value in overrides.items():
            if isinstance(value, Mapping):
                current = target.get(key)
                if isinstance(current, Mapping):
                    nested = dict(current)
                    _apply_overrides(nested, value)
                    target[key] = nested
                else:
                    target[key] = dict(value)
            else:
                target[key] = value
        return

    for key, value in overrides.items():
        if key == "few_shot_examples":
            try:
                setattr(target, "_few_shot_examples_overridden", True)
            except Exception:
                pass

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


def _apply_label_queries(
    bundle: LabelConfigBundle, label_queries: Mapping[str, str] | None
) -> LabelConfigBundle:
    """Return a bundle whose current config includes manual label queries.

    The AI backend accepts manual query strings via ``rag.label_queries`` in the
    configuration overrides, but the retriever consumes them from the
    materialised label configuration. This helper mirrors the RoundBuilder logic
    to merge queries into the current label config before instantiating the
    engine.
    """

    if not label_queries:
        return bundle

    parsed: Dict[str, str] = {}
    for label_id, query in label_queries.items():
        if isinstance(query, str) and query.strip():
            parsed[str(label_id)] = query.strip()

    if not parsed:
        return bundle

    current_config = dict(bundle.current or {})
    for label_id, query in parsed.items():
        entry = current_config.get(label_id)
        entry_payload = dict(entry) if isinstance(entry, Mapping) else {}
        entry_payload["search_query"] = query
        current_config[label_id] = entry_payload

    return replace(
        bundle,
        current=current_config,
    )


def _runtime_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate per-unit runtime statistics for logging and UI display."""

    if df is None or df.empty or "runtime_s" not in df.columns:
        return {}

    series = df["runtime_s"]
    if "unit_id" in df.columns:
        series = df.groupby("unit_id")["runtime_s"].first()

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}

    return {
        "count": int(numeric.count()),
        "avg_s": float(numeric.mean()),
        "median_s": float(numeric.median()),
        "p95_s": float(numeric.quantile(0.95)),
        "min_s": float(numeric.min()),
        "max_s": float(numeric.max()),
    }


def _build_test_stub_active_runner(paths: Paths):
    class _StubPipeline:
        def __init__(self, paths: Paths):
            self.paths = paths

        def run(self):
            parquet_notes = pd.read_parquet(self.paths.notes_path)
            outdir = Path(self.paths.outdir)

            buckets = {
                "bucket_disagreement.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
                "bucket_llm_uncertain.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
                "bucket_llm_certain.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
                "bucket_diversity.parquet": pd.DataFrame({"unit_id": [], "label_id": []}),
            }
            for name, df in buckets.items():
                df.to_parquet(outdir / name, index=False)

            return pd.DataFrame(
                {
                    "unit_id": parquet_notes["unit_id"].astype(str),
                    "doc_id": parquet_notes.get("doc_id", parquet_notes.get("note_id")),
                    "label_id": "pneumonitis",
                    "selection_reason": "dummy",
                }
            )

    return _StubPipeline(paths)


def _build_test_stub_inference_runner(paths: Paths):
    class _StubInference:
        def __init__(self, paths: Paths):
            self.paths = paths

        def run(self, unit_ids=None):  # noqa: ANN001 - match real signature
            parquet_notes = pd.read_parquet(self.paths.notes_path)
            if unit_ids:
                unit_set = {str(u) for u in unit_ids}
                parquet_notes = parquet_notes[parquet_notes["unit_id"].astype(str).isin(unit_set)]

            outdir = Path(self.paths.outdir)
            df = pd.DataFrame(
                {
                    "unit_id": parquet_notes["unit_id"].astype(str),
                    "doc_id": parquet_notes.get("doc_id", parquet_notes.get("note_id")),
                    "label_id": "pneumonitis",
                    "label_option_id": "yes",
                }
            )
            df.to_parquet(outdir / "inference_predictions.parquet", index=False)
            df.to_json(outdir / "inference_predictions.json", orient="records", lines=True)
            return df

    return _StubInference(paths)

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
    label_config_bundle: LabelConfigBundle | None = None,
    *,
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
    cfg = OrchestratorConfig()
    overrides = dict(cfg_overrides or {})
    phenotype_level = overrides.pop("phenotype_level", None)
    rag_overrides = overrides.get("rag") if isinstance(overrides.get("rag"), Mapping) else {}
    label_queries_override = None
    if isinstance(rag_overrides, Mapping):
        candidate = rag_overrides.get("label_queries")
        if isinstance(candidate, Mapping):
            label_queries_override = candidate
    if overrides:
        _apply_overrides(cfg, overrides)

    bundle = (label_config_bundle or EMPTY_BUNDLE).with_current_fallback(label_config)
    bundle = _apply_label_queries(bundle, label_queries_override)

    with _capture_logs(log_callback):
        with cancellation_scope(cancel_callback):
            if os.getenv("VAANNOTATE_TEST_STUB_ACTIVE_RUNNER") == "1":
                pipeline = _build_test_stub_active_runner(paths)
            else:
                pipeline = build_active_learning_runner(
                    paths=paths,
                    cfg=cfg,
                    label_config_bundle=bundle,
                    phenotype_level=phenotype_level,
                )
            final_df = pipeline.run()

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

    outdir_path = Path(outdir)
    artifacts = {
        "ai_next_batch_csv": str(csv_path),
        "buckets": {
            "disagreement": str(outdir_path / "bucket_disagreement.parquet"),
            "llm_uncertain": str(outdir_path / "bucket_llm_uncertain.parquet"),
            "llm_certain": str(outdir_path / "bucket_llm_certain.parquet"),
            "diversity": str(outdir_path / "bucket_diversity.parquet"),
        },
        "final_labels": str(outdir_path / "final_llm_labels.parquet")
        if (outdir_path / "final_llm_labels.parquet").exists()
        else None,
        "final_labels_json": str(outdir_path / "final_llm_labels.json")
        if (outdir_path / "final_llm_labels.json").exists()
        else None,
        "final_family_probe": str(outdir_path / "final_llm_family_probe.parquet")
        if (outdir_path / "final_llm_family_probe.parquet").exists()
        else None,
        "final_family_probe_json": str(outdir_path / "final_llm_family_probe.json")
        if (outdir_path / "final_llm_family_probe.json").exists()
        else None,
    }
    return normalized, artifacts


def run_inference(
    notes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    outdir: Path,
    label_config_bundle: LabelConfigBundle | None = None,
    *,
    label_config: Optional[dict] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    cfg: OrchestratorConfig | None = None,
    unit_ids: Optional[list[str]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    cache_dir: Path | None = None,
    session: BackendSession | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """Run inference-only labeling across the corpus or a provided subset.

    If ``session`` is provided, its models and ``EmbeddingStore`` are reused when
    building the inference pipeline. If ``session`` is ``None``, models and
    stores are constructed for this call as before.
    """

    outdir = Path(outdir)
    _ensure_dir(outdir)
    cache_dir_path = Path(cache_dir) if cache_dir else None
    paths = _default_paths(outdir, cache_dir=cache_dir_path)

    notes_path = Path(paths.notes_path)
    ann_path = Path(paths.annotations_path)
    notes_df.to_parquet(notes_path, index=False)
    ann_df.to_parquet(ann_path, index=False)

    cfg = copy.deepcopy(cfg) if cfg is not None else OrchestratorConfig()
    overrides = dict(cfg_overrides or {})
    phenotype_level = overrides.pop("phenotype_level", None)
    rag_overrides = overrides.get("rag") if isinstance(overrides.get("rag"), Mapping) else {}
    label_queries_override = None
    if isinstance(rag_overrides, Mapping):
        candidate = rag_overrides.get("label_queries")
        if isinstance(candidate, Mapping):
            label_queries_override = candidate
    if overrides:
        _apply_overrides(cfg, overrides)
        
    print("=== DEBUG orchestrator.run_inference ===")
    rag_overrides = (cfg_overrides.get("rag") or {}) if isinstance(cfg_overrides, dict) else {}
    print(
        "  cfg_overrides.rag:",
        "top_k_final=", rag_overrides.get("top_k_final"),
        "per_label_topk=", rag_overrides.get("per_label_topk"),
        "chunk_size=", rag_overrides.get("chunk_size"),
    )
    print(
        "  cfg.rag after overrides:",
        "top_k_final=", getattr(cfg.rag, "top_k_final", None),
        "per_label_topk=", getattr(cfg.rag, "per_label_topk", None),
        "chunk_size=", getattr(cfg.rag, "chunk_size", None),
    )
    bundle = (label_config_bundle or EMPTY_BUNDLE).with_current_fallback(label_config)
    bundle = _apply_label_queries(bundle, label_queries_override)

    with _capture_logs(log_callback):
        with cancellation_scope(cancel_callback):
            if os.getenv("VAANNOTATE_TEST_STUB_INFERENCE_RUNNER") == "1":
                pipeline = _build_test_stub_inference_runner(paths)
            else:
                models = session.models if session is not None else None
                store = session.store if session is not None else None
                pipeline = build_inference_runner(
                    paths=paths,
                    cfg=cfg,
                    label_config_bundle=bundle,
                    phenotype_level=phenotype_level,
                    models=models,
                    store=store,
                )
            result_df = pipeline.run(unit_ids=unit_ids)

    runtime_stats = _runtime_summary(result_df)

    artifacts = {
        "predictions": str(outdir / "inference_predictions.parquet"),
        "predictions_json": str(outdir / "inference_predictions.json"),
        "runtime_summary": runtime_stats,
    }

    return result_df, artifacts
