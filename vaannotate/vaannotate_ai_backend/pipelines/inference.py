from __future__ import annotations

import os
import time
from typing import Iterable, List, Mapping, Optional, Tuple

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
        def _normalize_type(raw_type: object) -> str:
            if raw_type is None:
                return ""
            t = str(raw_type).strip().lower()
            if not t:
                return ""
            if t in {"binary", "categorical", "categorical_single", "categorical_multi", "ordinal", "date", "numeric"}:
                return t
            return "categorical"

        def _extract_rule_text(entry: Mapping[str, object] | None) -> str:
            if not isinstance(entry, Mapping):
                return ""
            for key in ("rule", "rules", "why", "query", "text"):
                val = entry.get(key)
                if isinstance(val, str):
                    val = val.strip()
                    if val:
                        return val
                elif isinstance(val, list):
                    for item in reversed(val):
                        if isinstance(item, str):
                            text = item.strip()
                            if text:
                                return text
                        elif isinstance(item, Mapping):
                            text = str(item.get("text") or item.get("rule") or "").strip()
                            if text:
                                return text
                elif isinstance(val, Mapping):
                    text = str(val.get("text") or val.get("rule") or "").strip()
                    if text:
                        return text
            return ""

        legacy_rules_map = (
            getattr(self.label_config_bundle, "legacy_rules_map", lambda: {})() if self.label_config_bundle else {}
        )
        legacy_label_types = (
            getattr(self.label_config_bundle, "legacy_label_types", lambda: {})() if self.label_config_bundle else {}
        )

        current_rules_map: dict[str, str] = {}
        current_label_types: dict[str, str] = {}

        for key, entry in (self.label_config or {}).items():
            if str(key) == "_meta":
                continue
            label_entry = entry if isinstance(entry, Mapping) else {}
            raw_id = label_entry.get("label_id") if isinstance(label_entry, Mapping) else None
            label_id = str(raw_id or key).strip()
            if not label_id:
                continue

            rule_text = _extract_rule_text(label_entry) if isinstance(label_entry, Mapping) else ""
            if label_id not in current_rules_map:
                current_rules_map[label_id] = rule_text

            normalized_type = _normalize_type(label_entry.get("type") if isinstance(label_entry, Mapping) else None)
            if normalized_type:
                current_label_types[label_id] = normalized_type
            elif label_id not in current_label_types:
                current_label_types[label_id] = "categorical"

        if not current_rules_map:
            current_rules_map = dict(legacy_rules_map)
        if not current_label_types:
            current_label_types = dict(legacy_label_types)

        return current_rules_map, current_label_types

    def _label_units(self, unit_ids: Iterable[str], label_types: dict[str, str], rules_map: dict[str, str]) -> pd.DataFrame:
        fam_cls = getattr(self.llm, "family_labeler_cls", None)
        fam = fam_cls(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst) if fam_cls else None
        if fam is None:
            from ..services.family_labeler import FamilyLabeler

            fam = FamilyLabeler(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)

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
