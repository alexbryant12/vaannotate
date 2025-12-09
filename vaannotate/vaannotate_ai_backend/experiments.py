from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
import pandas as pd

from .config import OrchestratorConfig, Paths
from .label_configs import LabelConfigBundle, EMPTY_BUNDLE
from .orchestration import BackendSession
from .orchestrator import _apply_overrides, run_inference


def _normalize_local_model_overrides(
    cfg_overrides: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Ensure local model directories are passed via the models/llm configs."""

    normalized = copy.deepcopy(cfg_overrides or {})

    models_cfg = (
        dict(normalized.get("models"))
        if isinstance(normalized.get("models"), Mapping)
        else {}
    )
    embed_dir = normalized.get("embedding_model_dir")
    rerank_dir = normalized.get("reranker_model_dir")

    if isinstance(embed_dir, str) and embed_dir.strip():
        models_cfg.setdefault("embed_model_name", embed_dir.strip())
    if isinstance(rerank_dir, str) and rerank_dir.strip():
        models_cfg.setdefault("rerank_model_name", rerank_dir.strip())
    if models_cfg:
        normalized["models"] = models_cfg

    llm_cfg = (
        dict(normalized.get("llm")) if isinstance(normalized.get("llm"), Mapping) else {}
    )
    top_level_local_llm = normalized.get("local_model_dir")
    if isinstance(top_level_local_llm, str) and top_level_local_llm.strip():
        llm_cfg.setdefault("local_model_dir", top_level_local_llm.strip())
    if llm_cfg:
        normalized["llm"] = llm_cfg

    # RAG top-k compatibility shim:
    # - Older configs may set rag.per_label_topk only.
    # - New code and Admin sweeps should set rag.top_k_final.
    # Here we canonicalize:
    #   * If only per_label_topk is set, promote it into top_k_final.
    #   * Then mirror top_k_final back into per_label_topk so any legacy code
    #     inspecting that field sees the same value, but it never drives behavior.
    rag_cfg = (
        dict(normalized.get("rag")) if isinstance(normalized.get("rag"), Mapping) else {}
    )

    top_k_final = rag_cfg.get("top_k_final")
    per_label_topk = rag_cfg.get("per_label_topk")

    # Prefer explicit per_label_topk overrides over inherited top_k_final values
    # (e.g., when a prior-round baseline included top_k_final but the sweep only
    # sets per_label_topk). If per_label_topk is present and differs, promote it
    # into top_k_final to avoid carrying forward stale baselines.
    per_label_specified = "per_label_topk" in rag_cfg
    top_final_specified = "top_k_final" in rag_cfg

    if per_label_specified and (not top_final_specified or per_label_topk != top_k_final):
        try:
            rag_cfg["top_k_final"] = int(per_label_topk)
        except Exception:
            rag_cfg["top_k_final"] = per_label_topk
        top_k_final = rag_cfg["top_k_final"]

    # If only the legacy knob is set, promote it to top_k_final.
    if top_k_final is None and per_label_topk is not None:
        try:
            rag_cfg["top_k_final"] = int(per_label_topk)
        except Exception:
            rag_cfg["top_k_final"] = per_label_topk
        top_k_final = rag_cfg["top_k_final"]

    # If top_k_final is set, treat it as canonical and keep per_label_topk in sync.
    if top_k_final is not None:
        try:
            rag_cfg["per_label_topk"] = int(top_k_final)
        except Exception:
            rag_cfg["per_label_topk"] = top_k_final

    if rag_cfg:
        normalized["rag"] = rag_cfg

    return normalized


def _overrides_affect_backend(overrides: Mapping[str, Any] | None) -> bool:
    """Return True when overrides change any backend-affecting knobs (embedder, re-ranker, or RAG config)."""

    if overrides is None:
        return False

    models_cfg = overrides.get("models") if isinstance(overrides, Mapping) else None
    if isinstance(models_cfg, Mapping) and any(
        key in models_cfg for key in ("embed_model_name", "rerank_model_name")
    ):
        return True

    if isinstance(overrides, Mapping) and any(
        key in overrides for key in ("embedding_model_dir", "reranker_model_dir")
    ):
        return True

    rag_cfg = overrides.get("rag") if isinstance(overrides, Mapping) else None
    if isinstance(rag_cfg, Mapping) and rag_cfg:
        return True

    return False


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
    sweep_cfgs: Optional[Mapping[str, OrchestratorConfig]] = None,
    normalized_sweeps: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
    sweep_cfgs:
        Optional mapping of experiment name to fully prepared
        :class:`OrchestratorConfig` objects. When provided, these configs are
        deep-copied per sweep instead of being reconstructed from the overrides.
    normalized_sweeps:
        Optional mapping of experiment name to pre-normalized overrides. When
        supplied, these overrides are forwarded to :func:`run_inference` without
        additional normalization.
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
        Optional :class:`BackendSession` to reuse a single set of models and
        EmbeddingStore across sweeps. A shared session is reused only when
        ``_overrides_affect_backend(normalized_overrides)`` is ``False`` for a
        given sweep; otherwise a new session is built for that sweep. Any
        ``rag.*`` override (e.g., ``chunk_size``, ``top_k_final``,
        ``normalize_embeddings``, ``use_mmr``, etc.) is treated as
        backend-affecting, so those sweeps receive their own session even if the
        embedder and re-ranker names are unchanged. If ``session`` is ``None``,
        a fresh session is created using the shared cache directory.

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

    def _build_cfg_for_overrides(
        overrides: Mapping[str, Any], *, normalize: bool = True
    ) -> tuple[OrchestratorConfig, Dict[str, Any]]:
        """Prepare a config + overrides for a sweep."""

        sweep_cfg = OrchestratorConfig()
        normalized_overrides = (
            _normalize_local_model_overrides(dict(overrides)) if normalize else dict(overrides)
        )
        if normalized_overrides:
            _apply_overrides(sweep_cfg, normalized_overrides)

        return sweep_cfg, normalized_overrides

    def _build_session_for_cfg(name: str, sweep_cfg: OrchestratorConfig) -> BackendSession:
        """Construct a BackendSession that respects experiment overrides."""

        session_paths = Paths(
            notes_path=str(base_outdir / f"_{name}_session_notes.parquet"),
            annotations_path=str(base_outdir / f"_{name}_session_annotations.parquet"),
            outdir=str(base_outdir / f"_{name}_session"),
            cache_dir_override=str(shared_cache_dir),
        )

        return BackendSession.from_env(session_paths, sweep_cfg)

    def _make_log_callback(name: str) -> Optional[Callable[[str], None]]:
        if log_callback is None:
            return None

        def _wrapped(message: str) -> None:
            log_callback(f"[{name}] {message}")

        return _wrapped

    for name, overrides in sweeps.items():
        normalized_overrides = _normalize_local_model_overrides(dict(overrides))
        if normalized_sweeps is not None and name in normalized_sweeps:
            normalized_overrides = copy.deepcopy(normalized_sweeps[name])
        if sweep_cfgs is not None and name in sweep_cfgs:
            sweep_cfg = copy.deepcopy(sweep_cfgs[name])
        else:
            sweep_cfg, normalized_overrides = _build_cfg_for_overrides(
                normalized_overrides, normalize=False
            )

        exp_outdir = base_outdir / name
        exp_log_cb = _make_log_callback(name)

        backend_overrides = _overrides_affect_backend(normalized_overrides)

        if session is not None and backend_overrides:
            exp_session = _build_session_for_cfg(name, sweep_cfg)
        else:
            exp_session = session or _build_session_for_cfg(name, sweep_cfg)

        df, artifacts = run_inference(
            notes_df=index_notes_df,
            ann_df=index_ann_df,
            outdir=exp_outdir,
            label_config_bundle=bundle,
            label_config=label_config,
            cfg_overrides=normalized_overrides,
            unit_ids=unit_ids,
            cancel_callback=cancel_callback,
            log_callback=exp_log_cb,
            cache_dir=shared_cache_dir,
            session=exp_session,
            cfg=sweep_cfg,
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
