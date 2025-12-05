"""Prompt builder and large-batch inference helpers for the AdminApp.

This module provides a backend-only implementation of the prompt builder and
large-batch inference workflow described in the Admin UI requirements.  The
classes here are UI-agnostic so that both the desktop AdminApp and the CLI can
share a single implementation for experiment sweeps, logging, and resumable
inference jobs.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from vaannotate.utils import ensure_dir
from vaannotate.vaannotate_ai_backend import run_ai_backend_and_collect


@dataclass
class PromptBuilderConfig:
    """User-selected knobs for prompt building and inference."""

    labelset_id: str
    system_prompt: str = ""
    use_few_shot: bool = False
    few_shot_examples: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    label_rule_overrides: Dict[str, str] = field(default_factory=dict)
    inference_mode: str = "family_tree"  # "family_tree" | "single_shot"
    backend: str = "default"  # "default" | "openai" | "local"
    azure_api_key: str = ""
    azure_api_version: str = ""
    azure_endpoint: str = ""
    local_model_dir: str = ""
    local_max_seq_len: int = 0
    local_max_new_tokens: int = 0
    embedding_model_dir: str = ""
    reranker_model_dir: str = ""
    context_order: str = "relevance"
    rag_chunk_size: int = 1500
    rag_num_chunks: int = 6
    rag_mmr_lambda: float = 0.7
    backend_overrides: Dict[str, object] = field(default_factory=dict)

    def label_config_payload(self) -> Dict[str, object]:
        """Build a label_config-like payload consumed by the AI backend."""

        payload: Dict[str, object] = {
            "_meta": {
                "inference_mode": self.inference_mode,
                "backend": self.backend,
                "system_prompt": self.system_prompt,
            }
        }
        if self.use_few_shot and self.few_shot_examples:
            payload["few_shot_examples"] = self.few_shot_examples
        if self.label_rule_overrides:
            payload["rules"] = self.label_rule_overrides
        return payload

    def cfg_overrides(self) -> Dict[str, object]:
        """Translate UI settings into ``cfg_overrides`` for the backend."""

        llm_cfg: Dict[str, object] = {"backend": self.backend}
        if self.azure_api_key:
            llm_cfg["azure_api_key"] = self.azure_api_key
        if self.azure_api_version:
            llm_cfg["azure_api_version"] = self.azure_api_version
        if self.azure_endpoint:
            llm_cfg["azure_endpoint"] = self.azure_endpoint
        if self.local_model_dir:
            llm_cfg["local_model_dir"] = self.local_model_dir
        if self.local_max_seq_len:
            llm_cfg["local_max_seq_len"] = self.local_max_seq_len
        if self.local_max_new_tokens:
            llm_cfg["local_max_new_tokens"] = self.local_max_new_tokens
        if self.use_few_shot and self.few_shot_examples:
            llm_cfg["few_shot_examples"] = self.few_shot_examples
        if self.embedding_model_dir:
            llm_cfg["embedding_model_dir"] = self.embedding_model_dir
        if self.reranker_model_dir:
            llm_cfg["reranker_model_dir"] = self.reranker_model_dir
        if self.context_order:
            llm_cfg["context_order"] = self.context_order

        rag_cfg: Dict[str, object] = {
            "chunk_size": self.rag_chunk_size,
            "per_label_topk": self.rag_num_chunks,
            "mmr_lambda": self.rag_mmr_lambda,
            "use_mmr": True,
        }

        prompt_cfg: Dict[str, object] = {
            "inference_mode": self.inference_mode,
            "system_prompt": self.system_prompt,
        }

        return {
            "llm": _deep_update_dict(llm_cfg, self.backend_overrides.get("llm", {})),
            "rag": _deep_update_dict(rag_cfg, self.backend_overrides.get("rag", {})),
            "prompt_builder": _deep_update_dict(
                prompt_cfg, self.backend_overrides.get("prompt_builder", {})
            ),
            "llmfirst": _deep_update_dict({}, self.backend_overrides.get("llmfirst", {})),
        }


@dataclass
class PromptExperimentConfig:
    """Configuration for a single experiment sweep entry."""

    name: str
    config: PromptBuilderConfig


@dataclass
class PromptExperimentResult:
    """Lightweight record of an experiment evaluation."""

    name: str
    agreement: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "name": self.name,
            "agreement": self.agreement,
            "metrics": self.metrics,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
        }
        return payload


@dataclass
class PromptExperimentSweep:
    """Helper that expands user-specified sweeps into concrete configs."""

    base: PromptBuilderConfig
    chunk_sizes: Sequence[int] = (1500,)
    num_chunks: Sequence[int] = (6,)

    def variants(self) -> List[PromptExperimentConfig]:
        entries: List[PromptExperimentConfig] = []
        for chunk_size in self.chunk_sizes:
            for num_chunks in self.num_chunks:
                cfg = PromptBuilderConfig(
                    labelset_id=self.base.labelset_id,
                    system_prompt=self.base.system_prompt,
                    use_few_shot=self.base.use_few_shot,
                    few_shot_examples=self.base.few_shot_examples,
                    label_rule_overrides=self.base.label_rule_overrides,
                    inference_mode=self.base.inference_mode,
                    backend=self.base.backend,
                    azure_api_key=self.base.azure_api_key,
                    azure_api_version=self.base.azure_api_version,
                    azure_endpoint=self.base.azure_endpoint,
                    local_model_dir=self.base.local_model_dir,
                    local_max_seq_len=self.base.local_max_seq_len,
                    local_max_new_tokens=self.base.local_max_new_tokens,
                    embedding_model_dir=self.base.embedding_model_dir,
                    reranker_model_dir=self.base.reranker_model_dir,
                    context_order=self.base.context_order,
                    rag_chunk_size=chunk_size,
                    rag_num_chunks=num_chunks,
                    rag_mmr_lambda=self.base.rag_mmr_lambda,
                    backend_overrides=self.base.backend_overrides,
                )
                name = f"chunk{chunk_size}-k{num_chunks}"
                entries.append(PromptExperimentConfig(name=name, config=cfg))
        return entries


@dataclass
class PromptInferenceCheckpoint:
    """Checkpoint payload for resumable inference jobs."""

    run_id: str
    created_at: str
    project_root: str
    pheno_id: str
    labelset_id: str
    corpus_id: Optional[str]
    corpus_path: Optional[str]
    phenotype_level: str
    adjudicated_rounds: List[int]
    variants: List[Dict[str, object]]
    completed: int = 0
    results: List[Dict[str, object]] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        *,
        project_root: Path,
        pheno_id: str,
        labelset_id: str,
        corpus_id: Optional[str],
        corpus_path: Optional[Path],
        phenotype_level: str,
        adjudicated_rounds: Sequence[int],
        variants: Iterable[PromptExperimentConfig],
    ) -> "PromptInferenceCheckpoint":
        now = datetime.utcnow().isoformat()
        return cls(
            run_id=str(uuid.uuid4()),
            created_at=now,
            project_root=str(project_root),
            pheno_id=pheno_id,
            labelset_id=labelset_id,
            corpus_id=corpus_id,
            corpus_path=str(corpus_path) if corpus_path else None,
            phenotype_level=phenotype_level,
            adjudicated_rounds=list(adjudicated_rounds),
            variants=[{"name": v.name, "config": asdict(v.config)} for v in variants],
        )

    def write(self, path: Path) -> None:
        ensure_dir(path.parent)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PromptInferenceCheckpoint":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


class PromptInferenceJob:
    """Manage prompt experiments and resumable inference runs."""

    def __init__(
        self,
        project_root: Path,
        pheno_id: str,
        labelset_id: str,
        phenotype_level: str,
        adjudicated_rounds: Sequence[int],
        *,
        corpus_id: Optional[str] = None,
        corpus_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.pheno_id = pheno_id
        self.labelset_id = labelset_id
        self.phenotype_level = phenotype_level
        self.adjudicated_rounds = list(adjudicated_rounds)
        self.corpus_id = corpus_id
        self.corpus_path = Path(corpus_path) if corpus_path else None
        self.run_root = Path(run_root) if run_root else Path(project_root) / "prompt_runs"
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path
            else self.run_root / f"{pheno_id}_prompt_checkpoint.json"
        )

    def _load_or_create_checkpoint(
        self, variants: Iterable[PromptExperimentConfig]
    ) -> PromptInferenceCheckpoint:
        if self.checkpoint_path.exists():
            return PromptInferenceCheckpoint.load(self.checkpoint_path)
        return PromptInferenceCheckpoint.new(
            project_root=self.project_root,
            pheno_id=self.pheno_id,
            labelset_id=self.labelset_id,
            corpus_id=self.corpus_id,
            corpus_path=self.corpus_path,
            phenotype_level=self.phenotype_level,
            adjudicated_rounds=self.adjudicated_rounds,
            variants=variants,
        )

    def _write_checkpoint(self, ckpt: PromptInferenceCheckpoint) -> None:
        ckpt.write(self.checkpoint_path)

    def run(
        self,
        variants: Iterable[PromptExperimentConfig],
        *,
        user: str,
        log_callback: Optional[Callable[[str], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> List[PromptExperimentResult]:
        """Run the experiment sweep with checkpointing and resume support."""

        log = log_callback or (lambda msg: None)
        ckpt = self._load_or_create_checkpoint(variants)
        results: List[PromptExperimentResult] = []

        for idx, variant in enumerate(
            [PromptExperimentConfig(name=v["name"], config=PromptBuilderConfig(**v["config"])) for v in ckpt.variants]
        ):
            if idx < ckpt.completed:
                # Already completed in a previous run
                results.append(
                    PromptExperimentResult(
                        name=variant.name,
                        agreement=None,
                        metrics={},
                        artifacts_dir=None,
                    )
                )
                continue

            run_dir = ensure_dir(self.run_root / ckpt.run_id / variant.name)
            log(f"Running prompt experiment {variant.name} → {run_dir}")
            try:
                backend_result = run_ai_backend_and_collect(
                    self.project_root,
                    self.pheno_id,
                    self.labelset_id,
                    self.adjudicated_rounds,
                    run_dir,
                    self.phenotype_level,
                    user,
                    cfg_overrides=variant.config.cfg_overrides(),
                    label_config=variant.config.label_config_payload(),
                    log_callback=log_callback,
                    cancel_callback=cancel_callback,
                    corpus_id=self.corpus_id,
                    corpus_path=str(self.corpus_path) if self.corpus_path else None,
                    scope_corpus_to_annotations=True,
                    consensus_only=True,
                )
                result = PromptExperimentResult(
                    name=variant.name,
                    agreement=backend_result.metrics.get("agreement"),
                    metrics=backend_result.metrics,
                    artifacts_dir=Path(backend_result.artifacts_dir)
                    if backend_result.artifacts_dir
                    else run_dir,
                )
                results.append(result)
                ckpt.results.append(result.to_dict())
                ckpt.completed = idx + 1
                self._write_checkpoint(ckpt)
            except Exception:
                self._write_checkpoint(ckpt)
                raise
        return results

def _deep_update_dict(target: Dict[str, object], updates: Mapping[str, object]) -> Dict[str, object]:
    """Recursively merge ``updates`` into ``target`` and return the target dict."""

    merged = dict(target)
    for key, value in updates.items():
        if isinstance(value, Mapping):
            current = merged.get(key)
            if isinstance(current, Mapping):
                merged[key] = _deep_update_dict(dict(current), value)
            else:
                merged[key] = dict(value)
        else:
            merged[key] = value
    return merged
