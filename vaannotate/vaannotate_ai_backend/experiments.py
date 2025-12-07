from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import json
import pandas as pd

from .config import OrchestratorConfig, Paths
from .label_configs import LabelConfigBundle, EMPTY_BUNDLE
from .orchestration import BackendSession
from .orchestrator import run_inference


@dataclass
class InferenceExperimentResult:
    """Container for the outputs of a single inference experiment.

    Attributes
    ----------
    name:
        Human-readable experiment name (key from the sweeps mapping).
    cfg_overrides:
        The configuration overrides used for this experiment, in the same
        nested format accepted by ``run_inference``.
    outdir:
        Directory where this experiment's outputs were written.
    dataframe:
        The predictions DataFrame returned by ``run_inference``.
    artifacts:
        The artifacts dict returned by ``run_inference`` (e.g. file paths).
    """

    name: str
    cfg_overrides: Dict[str, Any]
    outdir: Path
    dataframe: pd.DataFrame
    artifacts: Dict[str, Any]


def run_inference_experiments(
    notes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    base_outdir: Path,
    sweeps: Mapping[str, Mapping[str, Any]],
    *,
    label_config_bundle: LabelConfigBundle | None = None,
    label_config: Optional[Dict[str, Any]] = None,
    unit_ids: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    session: Optional[BackendSession] = None,
) -> Dict[str, InferenceExperimentResult]:
    """Run multiple inference configurations (sweeps) and collect their outputs.

    Each entry in ``sweeps`` is a mapping from experiment name to a nested
    ``cfg_overrides`` dict in the same format accepted by :func:`run_inference`.

    For each experiment ``name``, this function:

    * creates an output directory at ``base_outdir / name``
    * calls :func:`run_inference` with that directory and the provided overrides
    * stores the resulting predictions DataFrame and artifacts

    The embeddings cache directory can be shared across experiments via the
    ``cache_dir`` argument; if not provided, a ``cache`` subdirectory under
    ``base_outdir`` is used by default.

    Parameters
    ----------
    notes_df:
        Notes DataFrame in the same format expected by :func:`run_inference`.
    ann_df:
        Annotations DataFrame in the same format expected by :func:`run_inference`.
    base_outdir:
        Directory under which per-experiment subdirectories will be created.
    sweeps:
        Mapping from experiment name to ``cfg_overrides`` dictionaries.
    label_config_bundle:
        Optional pre-materialised :class:`LabelConfigBundle`. If not provided,
        :func:`run_inference` will fall back to an empty bundle.
    label_config:
        Optional label_config overlay passed through to :func:`run_inference`.
    unit_ids:
        Optional list of unit IDs to restrict inference to.
    cache_dir:
        Optional shared cache directory for embeddings. If omitted, a
        ``cache`` subdirectory under ``base_outdir`` is used.
    cancel_callback:
        Optional cancellation callback passed through to :func:`run_inference`.
    log_callback:
        Optional logging callback; if provided, messages are prefixed with the
        experiment name.
    session:
        Optional BackendSession to reuse a single set of models + EmbeddingStore
        across all sweeps. If ``None``, a fresh session is created using the
        shared cache directory.

    Returns
    -------
    Dict[str, InferenceExperimentResult]
        Mapping from experiment name to result objects.
    """
    base_outdir = Path(base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    shared_cache_dir: Optional[Path] = cache_dir or (base_outdir / "cache")
    if shared_cache_dir is not None:
        shared_cache_dir.mkdir(parents=True, exist_ok=True)

    bundle = label_config_bundle or EMPTY_BUNDLE
    results: Dict[str, InferenceExperimentResult] = {}

    # Optionally scope the corpus to the evaluation units for indexing.
    # When unit_ids is provided, we restrict the notes/annotations that will be
    # written to parquet and indexed, so experiment sweeps do not embed the
    # entire base corpus if we only care about a gold-standard subset.
    index_notes_df = notes_df
    index_ann_df = ann_df

    if unit_ids is not None:
        unit_set = {str(u) for u in unit_ids if str(u)}
        if "unit_id" in index_notes_df.columns:
            index_notes_df = index_notes_df[
                index_notes_df["unit_id"].astype(str).isin(unit_set)
            ].copy()
        if "unit_id" in index_ann_df.columns:
            index_ann_df = index_ann_df[
                index_ann_df["unit_id"].astype(str).isin(unit_set)
            ].copy()

    # Build a shared BackendSession if one was not provided.
    if session is None:
        # Use a dedicated outdir under base_outdir for the session; only the
        # cache_dir matters for model/index reuse.
        session_paths = Paths(
            notes_path=str(base_outdir / "_session_notes.parquet"),
            annotations_path=str(base_outdir / "_session_annotations.parquet"),
            outdir=str(base_outdir / "_session"),
            cache_dir_override=str(shared_cache_dir),
        )
        base_cfg = OrchestratorConfig()
        session = BackendSession.from_env(session_paths, base_cfg)

    def _make_log_callback(name: str) -> Optional[Callable[[str], None]]:
        if log_callback is None:
            return None

        def _wrapped(message: str) -> None:
            log_callback(f"[{name}] {message}")

        return _wrapped

    for name, overrides in sweeps.items():
        exp_outdir = base_outdir / name
        exp_log_cb = _make_log_callback(name)

        df, artifacts = run_inference(
            notes_df=index_notes_df,
            ann_df=index_ann_df,
            outdir=exp_outdir,
            label_config_bundle=bundle,
            label_config=label_config,
            cfg_overrides=dict(overrides),
            unit_ids=unit_ids,
            cancel_callback=cancel_callback,
            log_callback=exp_log_cb,
            cache_dir=shared_cache_dir,
            session=session,
        )

        results[name] = InferenceExperimentResult(
            name=name,
            cfg_overrides=dict(overrides),
            outdir=exp_outdir,
            dataframe=df,
            artifacts=artifacts,
        )

    # Persist a simple manifest for downstream tooling or inspection.
    manifest_path = base_outdir / "experiments.json"
    try:
        manifest_payload = {
            name: {
                "cfg_overrides": result.cfg_overrides,
                "outdir": str(result.outdir),
                "artifacts": result.artifacts,
            }
            for name, result in results.items()
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    except Exception:
        # The experiments themselves have already completed; a failure to write
        # the manifest should not cause the sweep to be treated as failed.
        pass

    return results


__all__ = [
    "InferenceExperimentResult",
    "run_inference_experiments",
]
