from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from ..services import LLM_RECORDER
from ..utils.jsonish import _jsonify_cols


class InferencePipeline:
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
        from ..services.family_labeler import build_family_labeler

        fam = build_family_labeler(
            self.llm,
            self.retriever,
            self.repo,
            self.label_config,
            self.cfg.scjitter,
            self.cfg.llmfirst,
        )

        rows: List[dict] = []
        for uid in unit_ids:
            rows.extend(
                fam.label_family_for_unit(
                    uid,
                    label_types,
                    rules_map,
                    json_only=True,
                    json_n_consistency=getattr(self.cfg.llmfirst, "final_llm_label_consistency", 1),
                    json_jitter=False,
                )
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if "runs" in df.columns:
            df.rename(columns={"runs": "llm_runs"}, inplace=True)
        if "consistency" in df.columns:
            df.rename(columns={"consistency": "llm_consistency"}, inplace=True)
        if "prediction" in df.columns:
            df.rename(columns={"prediction": "llm_prediction"}, inplace=True)
        if "llm_runs" in df.columns:
            df["llm_reasoning"] = df["llm_runs"].map(
                lambda rs: (rs[0].get("raw", {}).get("reasoning") if isinstance(rs, list) and rs else None)
            )
        return df

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
