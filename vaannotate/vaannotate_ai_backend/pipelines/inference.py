from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from ..services import LLM_RECORDER
from ..utils.jsonish import _jsonify_cols


class InferencePipeline:
    """Inference runner supporting family or single-prompt labeling modes.

    - cfg.llmfirst.inference_labeling_mode="family" uses the existing
      FamilyLabeler with per-label retrieval and gating.
    - cfg.llmfirst.inference_labeling_mode="single_prompt" builds one
      merged context per unit, calls annotate_multi, and applies label
      dependencies as post-hoc gating. This mode is intended for
      inference experiments and large-scale corpus labeling.
    """
    def __init__(self, data_repo, emb_store, ctx_builder, llm_labeler, config, paths):
        self.repo = data_repo
        self.store = emb_store
        self.ctx_builder = ctx_builder
        self.llm = llm_labeler
        self.cfg = config
        self.paths = paths
        self.retriever = getattr(ctx_builder, "retriever", None)
        self.label_config = getattr(llm_labeler, "label_config", {})
        self.label_config_bundle = getattr(ctx_builder, "label_config_bundle", None)

    def _label_maps(self) -> Tuple[dict[str, str], dict[str, str]]:
        bundle = getattr(self, "label_config_bundle", None)
        if bundle is None:
            return {}, {}

        try:
            current_rules_map = bundle.current_rules_map(self.label_config)
        except TypeError:
            current_rules_map = bundle.current_rules_map()

        try:
            current_label_types = bundle.current_label_types(self.label_config)
        except TypeError:
            current_label_types = bundle.current_label_types()

        if not current_rules_map:
            current_rules_map = {}
        if not current_label_types:
            current_label_types = {}
        return current_rules_map, current_label_types

    def _label_units(self, unit_ids: Iterable[str], label_types: dict[str, str], rules_map: dict[str, str]) -> pd.DataFrame:
        mode = getattr(self.cfg.llmfirst, "inference_labeling_mode", "family")
        if mode == "single_prompt":
            return self._label_units_single_prompt(unit_ids, label_types, rules_map)
        return self._label_units_family(unit_ids, label_types, rules_map)

    def _label_units_family(
        self, unit_ids: Iterable[str], label_types: dict[str, str], rules_map: dict[str, str]
    ) -> pd.DataFrame:
        from ..services.family_labeler import build_family_labeler, run_family_labeling_for_units

        fam = build_family_labeler(
            self.llm,
            self.retriever,
            self.repo,
            self.label_config,
            self.cfg.scjitter,
            self.cfg.llmfirst,
        )

        json_n_consistency = int(
            getattr(self.cfg.llmfirst, "final_llm_label_consistency", 1) or 1
        )

        df = run_family_labeling_for_units(
            fam,
            unit_ids=unit_ids,
            label_types=label_types,
            per_label_rules=rules_map,
            json_n_consistency=json_n_consistency,
            json_jitter=False,
            iter_with_bar_fn=None,
            progress_step="Inference family labeling",
        )

        # The JSON-safe view is still handled by _jsonify_cols in run(), so we
        # simply return the tall DataFrame here.
        return df

    def _label_units_single_prompt(
        self, unit_ids: Iterable[str], label_types: dict[str, str], rules_map: dict[str, str]
    ) -> pd.DataFrame:
        label_ids = sorted(label_types.keys())
        max_labels = int(getattr(self.cfg.llmfirst, "single_prompt_max_labels", 64) or 64)
        label_ids = label_ids[:max_labels]

        rows: list[dict] = []

        from ..services.label_dependencies import build_label_dependencies, evaluate_gating

        _parent_to_children, _child_to_parents, _roots = build_label_dependencies(self.label_config)

        for uid in unit_ids:
            ctx = self.ctx_builder.build_context_for_family(
                uid,
                label_ids=label_ids,
                rules_map=rules_map,
                topk_per_label=self.cfg.rag.top_k_final,
                max_snippets=None,
                max_chars=getattr(self.cfg.llmfirst, "single_prompt_max_chars", None),
            )

            res = self.llm.annotate_multi(
                unit_id=uid,
                label_ids=label_ids,
                label_types=label_types,
                rules_map=rules_map,
                ctx_snippets=ctx,
            )
            preds = res.get("predictions") or {}

            parent_preds: dict[tuple[str, str], str] = {}
            for lid, info in preds.items():
                parent_preds[(uid, str(lid))] = info.get("prediction") if isinstance(info, dict) else None

            for lid in label_ids:
                info = preds.get(str(lid)) or {}
                value = info.get("prediction") if isinstance(info, dict) else None
                gated_ok = evaluate_gating(
                    label_id=str(lid),
                    unit_id=uid,
                    parent_preds=parent_preds,
                    label_types=label_types,
                    label_config=self.label_config,
                )
                if not gated_ok:
                    value = None

                rows.append(
                    {
                        "unit_id": uid,
                        "label_id": str(lid),
                        "label_type": label_types.get(str(lid), "categorical"),
                        "llm_prediction": value,
                        "llm_consistency": None,
                        "llm_reasoning": info.get("reasoning") if isinstance(info, dict) else None,
                        "rag_context": ctx,
                        "llm_runs": res.get("runs", []),
                    }
                )

        return pd.DataFrame(rows)

    def run(self, unit_ids: Optional[List[str]] = None) -> pd.DataFrame:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        LLM_RECORDER.start(
            outdir=self.paths.outdir,
            run_id=run_id,
            meta={"mode": "inference", "model": getattr(self.cfg.llm, "model_name", None)},
        )

        self.store.build_chunk_index(self.repo.notes, self.cfg.rag, self.cfg.index)

        all_units = [str(u) for u in self.repo.notes["unit_id"].unique().tolist()]
        selected_units = all_units if unit_ids is None else [str(u) for u in unit_ids if str(u)]

        if not selected_units:
            LLM_RECORDER.flush()
            return pd.DataFrame(columns=["unit_id", "label_id", "llm_prediction"])

        rules_map, label_types = self._label_maps()
        label_df = self._label_units(selected_units, label_types, rules_map)

        if label_df.empty:
            LLM_RECORDER.flush()
            return label_df

        jsonified = _jsonify_cols(label_df, [c for c in ["rag_context", "llm_runs", "fc_probs"] if c in label_df.columns])
        out_path = os.path.join(self.paths.outdir, "inference_predictions.parquet")
        jsonified.to_parquet(out_path, index=False)

        json_path = os.path.join(self.paths.outdir, "inference_predictions.json")
        try:
            jsonified.to_json(json_path, orient="records", lines=True, force_ascii=False)
        except Exception:
            pass

        LLM_RECORDER.flush()
        return label_df


__all__ = ["InferencePipeline"]
