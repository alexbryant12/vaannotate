from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from ..services import LLM_RECORDER
from ..utils.runtime import check_cancelled as default_check_cancelled
from ..utils.runtime import iter_with_bar as default_iter_with_bar


class ActiveLearningPipeline:
    def __init__(
        self,
        data_repo,
        emb_store,
        ctx_builder,
        llm_labeler,
        disagreement_scorer,
        uncertainty_scorer,
        diversity_selector,
        selector,
        config,
        paths,
        *,
        pooler=None,
        retriever=None,
        label_config=None,
        label_config_bundle=None,
        excluded_unit_ids: Optional[Set[str]] = None,
        check_cancelled_fn: Optional[Callable[[], None]] = None,
        iter_with_bar_fn: Optional[Callable[..., object]] = None,
        jsonify_cols_fn: Optional[Callable[[pd.DataFrame, List[str]], pd.DataFrame]] = None,
        attach_metadata_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        label_maps_fn: Optional[Callable[[], Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]]] = None,
        unseen_pairs_fn: Optional[Callable[[Optional[Set[str]]], List[Tuple[str, str]]]] = None,
        disagreement_builder_fn: Optional[Callable[[Set[Tuple[str, str]], Dict[str, str], Dict[str, str]], pd.DataFrame]] = None,
        uncertain_builder_fn: Optional[Callable[[Dict[str, str], Dict[str, str], Optional[Set[str]]], pd.DataFrame]] = None,
        certain_builder_fn: Optional[Callable[[Dict[str, str], Dict[str, str], Optional[Set[str]]], pd.DataFrame]] = None,
        rerank_override_fn: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None,
    ):
        self.repo = data_repo
        self.store = emb_store
        self.ctx_builder = ctx_builder
        self.llm = llm_labeler
        self.disagreement_scorer = disagreement_scorer
        self.uncertainty_scorer = uncertainty_scorer
        self.diversity_selector = diversity_selector
        self.selector = selector
        self.cfg = config
        self.paths = paths
        self.pooler = pooler
        self.retriever = retriever or getattr(ctx_builder, "retriever", None)
        self.label_config = label_config if label_config is not None else getattr(llm_labeler, "label_config", None)
        self.label_config_bundle = label_config_bundle if label_config_bundle is not None else getattr(ctx_builder, "label_config_bundle", None)
        self.excluded_unit_ids = excluded_unit_ids or set()
        self.check_cancelled = check_cancelled_fn
        self.iter_with_bar = iter_with_bar_fn
        self.jsonify_cols = jsonify_cols_fn
        self.attach_metadata = attach_metadata_fn
        self.label_maps_fn = label_maps_fn
        self.unseen_pairs_fn = unseen_pairs_fn
        self.disagreement_builder_fn = disagreement_builder_fn
        self.uncertain_builder_fn = uncertain_builder_fn
        self.certain_builder_fn = certain_builder_fn
        self.rerank_override_fn = rerank_override_fn

    def _apply_excluded_units(self) -> int:
        removed = self.repo.exclude_units(self.excluded_unit_ids)
        if removed and self.retriever is not None:
            self.retriever._notes_by_doc = self.repo.notes_by_doc()
        return removed

    def _label_maps(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
        if self.label_maps_fn:
            return self.label_maps_fn()
        legacy_rules_map = getattr(self.label_config_bundle, "legacy_rules_map", lambda: {})()
        legacy_label_types = getattr(self.label_config_bundle, "legacy_label_types", lambda: {})()
        current_rules_map = getattr(self.label_config_bundle, "current_rules_map", lambda: {})()
        current_label_types = getattr(self.label_config_bundle, "current_label_types", lambda: {})()

        if not current_rules_map and legacy_rules_map:
            current_rules_map = dict(legacy_rules_map)
        if not current_label_types and legacy_label_types:
            current_label_types = dict(legacy_label_types)

        return legacy_rules_map, legacy_label_types, current_rules_map, current_label_types

    def _build_disagreement_bucket(self, seen_pairs: Set[Tuple[str, str]], rules_map: Dict[str, str], label_types: Dict[str, str]) -> pd.DataFrame:
        if self.disagreement_builder_fn:
            return self.disagreement_builder_fn(seen_pairs, rules_map, label_types)
        return self.disagreement_scorer.compute_disagreement(seen_pairs, rules_map, label_types)

    def _build_unseen_pairs(self, label_ids: Optional[Set[str]] = None) -> List[Tuple[str, str]]:
        if self.unseen_pairs_fn:
            return self.unseen_pairs_fn(label_ids)
        seen = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        all_units = sorted(self.repo.notes["unit_id"].unique().tolist())

        if label_ids is None:
            legacy_labels = {str(l) for l in self.repo.ann["label_id"].unique().tolist()}
            config_labels: set[str] = set()
            for key, entry in (self.label_config or {}).items():
                if str(key) == "_meta":
                    continue
                if isinstance(entry, dict):
                    lid = str(entry.get("label_id") or key).strip()
                else:
                    lid = str(key).strip()
                if lid:
                    config_labels.add(lid)
            use_labels = legacy_labels | config_labels
        else:
            use_labels = {str(l) for l in label_ids if str(l)}

        all_labels = sorted(use_labels)
        return [(u, l) for u in all_units for l in all_labels if (u, l) not in seen]

    def _build_llm_uncertain_bucket(self, label_types: Dict[str, str], rules_map: Dict[str, str], exclude_units: Optional[Set[str]] = None) -> pd.DataFrame:
        if self.uncertain_builder_fn:
            return self.uncertain_builder_fn(label_types, rules_map, exclude_units)
        fam_cls = getattr(self.llm, "family_labeler_cls", None)
        fam = fam_cls(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst) if fam_cls else None
        if fam is None:
            from ..engine import FamilyLabeler  # type: ignore

            fam = FamilyLabeler(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
        probe_df = fam.probe_units_label_tree(self.cfg.llmfirst.enrich, label_types, rules_map, exclude_units=exclude_units)
        probe_df = self.uncertainty_scorer.score_probe_results(probe_df)
        if self.jsonify_cols:
            safe_cols = [c for c in ["fc_probs", "rag_context", "why", "runs"] if c in probe_df.columns]
            self.jsonify_cols(probe_df, safe_cols).to_parquet(os.path.join(self.paths.outdir, "llm_probe.parquet"), index=False)
        from ..engine import direct_uncertainty_selection  # type: ignore

        n_unc = int(self.cfg.select.batch_size * self.cfg.select.pct_uncertain)
        return direct_uncertainty_selection(probe_df, n_unc, select_most_certain=False)

    def _build_llm_certain_bucket(self, label_types: Dict[str, str], rules_map: Dict[str, str], exclude_units: Optional[Set[str]] = None) -> pd.DataFrame:
        if self.certain_builder_fn:
            return self.certain_builder_fn(label_types, rules_map, exclude_units)
        p = os.path.join(self.paths.outdir, "llm_probe.parquet")
        if os.path.exists(p):
            probe_df = pd.read_parquet(p)
        else:
            fam_cls = getattr(self.llm, "family_labeler_cls", None)
            fam = fam_cls(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst) if fam_cls else None
            if fam is None:
                from ..engine import FamilyLabeler  # type: ignore

                fam = FamilyLabeler(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
            probe_df = fam.probe_units_label_tree(self.cfg.llmfirst.enrich, label_types, rules_map, exclude_units=exclude_units)
        probe_df = self.uncertainty_scorer.score_probe_results(probe_df)
        from ..engine import direct_uncertainty_selection  # type: ignore

        n_cer = int(self.cfg.select.batch_size * self.cfg.select.pct_easy_qc)
        return direct_uncertainty_selection(probe_df, n_cer, select_most_certain=True)

    def run(self) -> pd.DataFrame:
        import pandas as pd

        t0 = time.time()
        check_cancelled = self.check_cancelled or default_check_cancelled
        iter_with_bar = self.iter_with_bar or default_iter_with_bar

        run_id = time.strftime("%Y%m%d-%H%M%S")
        LLM_RECORDER.start(
            outdir=self.paths.outdir,
            run_id=run_id,
            meta={
                "model": getattr(self.cfg.llm, "model_name", None),
                "project": os.path.basename(self.paths.outdir.rstrip(os.sep)),
            },
        )

        print("Indexing chunks ...")
        check_cancelled()
        self.store.build_chunk_index(self.repo.notes, self.cfg.rag, self.cfg.index)
        removed = self._apply_excluded_units()
        if removed:
            print(f"Excluded {removed} previously reviewed units from candidate pool")
        if self.repo.notes.empty:
            raise RuntimeError("No candidate units remain after excluding previously reviewed units.")
        print("Building label prototypes ...")
        check_cancelled()
        if self.pooler is not None:
            self.pooler.build_prototypes()

        legacy_rules_map, legacy_label_types, current_rules_map, current_label_types = self._label_maps()
        legacy_label_ids = {str(l) for l in self.repo.ann["label_id"].unique().tolist()}
        if not legacy_label_ids:
            legacy_label_ids = set(legacy_rules_map.keys())
        current_label_ids = set(current_rules_map.keys())
        if not current_label_ids:
            current_label_ids = set(legacy_label_ids)

        if hasattr(self.retriever, "rerank_rule_overrides"):
            overrides = self.rerank_override_fn(current_rules_map) if self.rerank_override_fn else {}
            self.retriever.rerank_rule_overrides = overrides or getattr(self.retriever, "rerank_rule_overrides", {}) or {}

        seen_units = set(self.repo.ann["unit_id"].unique().tolist())
        seen_pairs = set(zip(self.repo.ann["unit_id"], self.repo.ann["label_id"]))
        unseen_pairs_current = self._build_unseen_pairs(label_ids=current_label_ids)

        selector = self.selector
        selector.label_types = current_label_types
        selector.current_label_ids = current_label_ids
        selector.seen_units = seen_units
        selector.unseen_pairs = unseen_pairs_current

        total = int(self.cfg.select.batch_size)
        n_dis = int(total * self.cfg.select.pct_disagreement)
        n_div = int(total * self.cfg.select.pct_diversity)
        n_unc = int(total * self.cfg.select.pct_uncertain)
        n_cer = int(total * self.cfg.select.pct_easy_qc)

        run_id = time.strftime("%Y%m%d-%H%M%S")
        LLM_RECORDER.start(outdir=self.paths.outdir, run_id=run_id)

        selected_units: set[str] = set()
        dis_units = selector._empty_unit_frame()
        dis_path = os.path.join(self.paths.outdir, "bucket_disagreement.parquet")
        if self.repo.ann.empty:
            print("[1/4] Skipping disagreement bucket (no prior rounds or quota is zero)")
        else:
            if n_dis > 0:
                print("[1/4] Expanded disagreement ...")
            else:
                print("[1/4] Disagreement quota is zero; refreshing schema only ...")
            check_cancelled()
            dis_pairs = self._build_disagreement_bucket(seen_pairs, legacy_rules_map, legacy_label_types)
            dis_units = selector.select_disagreement(dis_pairs, selected_units=selected_units)
        dis_units.to_parquet(dis_path, index=False)
        selected_units |= set(dis_units["unit_id"])

        print("[2/4] Diversity ...")
        check_cancelled()
        want_div = min(n_div, max(0, total - len(selected_units)))
        fam_cls = getattr(self.llm, "family_labeler_cls", None)
        fam = fam_cls(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst) if fam_cls else None
        if fam is None:
            from ..engine import FamilyLabeler  # type: ignore

            fam = FamilyLabeler(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
        div_candidates = pd.DataFrame(unseen_pairs_current, columns=["unit_id", "label_id"])
        sel_div_pairs = self.diversity_selector.select_diverse_units(
            div_candidates,
            n_div=want_div,
            already_selected=[(r.unit_id, getattr(r, "label_id", "")) for r in dis_units.itertuples(index=False)],
            pooler=self.pooler,
            retriever=self.retriever,
            rules_map=current_rules_map,
            label_types=current_label_types,
            label_config=self.label_config,
            rag_k=getattr(self.cfg.diversity, "rag_topk", 4),
            min_rel_quantile=getattr(self.cfg.diversity, "min_rel_quantile", 0.30),
            mmr_lambda=getattr(self.cfg.diversity, "mmr_lambda", 0.7),
            sample_cap=getattr(self.cfg.diversity, "sample_cap", 2000),
            adaptive_relax=getattr(self.cfg.diversity, "adaptive_relax", True),
            relax_steps=getattr(self.cfg.diversity, "relax_steps", (0.20, 0.10, 0.05)),
            pool_factor=getattr(self.cfg.diversity, "pool_factor", 4.0),
            use_proto=getattr(self.cfg.diversity, "use_proto", False),
            family_labeler=fam,
            ensure_exemplars=True,
            exclude_units=seen_units | selected_units,
            iter_with_bar_fn=iter_with_bar,
        )
        sel_div_pairs = selector._filter_units(sel_div_pairs, seen_units | selected_units)
        sel_div_units = selector._head_units(selector._to_unit_only(sel_div_pairs), want_div)
        sel_div_units.to_parquet(os.path.join(self.paths.outdir, "bucket_diversity.parquet"), index=False)
        selected_units |= set(sel_div_units["unit_id"])

        if n_unc > 0:
            print("[3/4] LLM-uncertain ...")
            check_cancelled()
            want_unc = min(n_unc, max(0, total - len(selected_units)))
            if want_unc > 0:
                sel_unc_pairs = self._build_llm_uncertain_bucket(
                    current_label_types,
                    current_rules_map,
                    exclude_units=seen_units | selected_units,
                )
                sel_unc_pairs = selector._filter_units(sel_unc_pairs, seen_units | selected_units)
                sel_unc_units = selector._head_units(selector._to_unit_only(sel_unc_pairs), want_unc)
                sel_unc_units.to_parquet(os.path.join(self.paths.outdir, "bucket_llm_uncertain.parquet"), index=False)
                selected_units |= set(sel_unc_units["unit_id"])
            else:
                print("Uncertain bucket skipped: no remaining quota.")

        if n_cer > 0:
            print("[4/4] LLM-certain ...")
            check_cancelled()
            want_cer = min(n_cer, max(0, total - len(selected_units)))
            if want_cer > 0:
                sel_cer_pairs = self._build_llm_certain_bucket(
                    current_label_types,
                    current_rules_map,
                    exclude_units=seen_units | selected_units,
                )
                sel_cer_pairs = selector._filter_units(sel_cer_pairs, seen_units | selected_units)
                sel_cer_units = selector._head_units(selector._to_unit_only(sel_cer_pairs), want_cer)
                sel_cer_units.to_parquet(os.path.join(self.paths.outdir, "bucket_llm_certain.parquet"), index=False)
                selected_units |= set(sel_cer_units["unit_id"])
            else:
                print("Certain bucket skipped: no remaining quota.")

        final = selector.build_next_batch(
            disagree_df=dis_units,
            uncertainty_df=sel_unc_units if "sel_unc_units" in locals() else selector._empty_unit_frame(),
            easy_df=sel_cer_units if "sel_cer_units" in locals() else selector._empty_unit_frame(),
            diversity_df=sel_div_units,
            prefiltered=True,
        )

        final.to_parquet(os.path.join(self.paths.outdir, "final_selection.parquet"), index=False)
        result_df = final

        final_out = None
        if getattr(self.cfg, "final_llm_labeling", False):
            fam_cls = getattr(self.llm, "family_labeler_cls", None)
            fam = fam_cls(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst) if fam_cls else None
            if fam is None:
                from ..engine import FamilyLabeler  # type: ignore

                fam = FamilyLabeler(self.llm, self.retriever, self.repo, self.label_config, self.cfg.scjitter, self.cfg.llmfirst)
            unit_ids = final["unit_id"].tolist()
            rules_map = current_rules_map
            types = current_label_types
            _progress_every = float(getattr(fam.cfg, "progress_min_interval_s", 1) or 1)
            fam_rows = []
            for uid in iter_with_bar(
                step="Final family labeling",
                iterable=unit_ids,
                total=len(unit_ids),
                min_interval_s=_progress_every,
            ):
                fam_rows.extend(
                    fam.label_family_for_unit(
                        uid,
                        types,
                        rules_map,
                        json_only=True,
                        json_n_consistency=getattr(self.cfg.llmfirst, "final_llm_label_consistency", 1),
                        json_jitter=False,
                    )
                )
            fam_df = pd.DataFrame(fam_rows)
            if not fam_df.empty:
                if "runs" in fam_df.columns:
                    fam_df.rename(columns={"runs": "llm_runs"}, inplace=True)
                if "consistency" in fam_df.columns:
                    fam_df.rename(columns={"consistency": "llm_consistency"}, inplace=True)
                if "prediction" in fam_df.columns:
                    fam_df.rename(columns={"prediction": "llm_prediction"}, inplace=True)
                if "llm_runs" in fam_df.columns:
                    fam_df["llm_reasoning"] = fam_df["llm_runs"].map(
                        lambda rs: (rs[0].get("raw", {}).get("reasoning") if isinstance(rs, list) and rs else None)
                    )
                if self.jsonify_cols:
                    fam_df = self.jsonify_cols(fam_df, [c for c in ["rag_context", "llm_runs", "fc_probs"] if c in fam_df.columns])
            fam_df.to_parquet(os.path.join(self.paths.outdir, "final_llm_family_probe.parquet"), index=False)

            if not fam_df.empty:
                pv = fam_df[["unit_id", "label_id", "llm_prediction"]].copy()
                pv["col"] = pv["label_id"].astype(str) + "_llm"
                fam_wide = pv.pivot_table(index="unit_id", columns="col", values="llm_prediction", aggfunc="first").reset_index()
                if "llm_reasoning" in fam_df.columns:
                    rv = fam_df[["unit_id", "label_id", "llm_reasoning"]].copy()
                    rv["colr"] = rv["label_id"].astype(str) + "_llm_reason"
                    fam_reason_wide = rv.pivot_table(index="unit_id", columns="colr", values="llm_reasoning", aggfunc="first").reset_index()
                    fam_wide = fam_wide.merge(fam_reason_wide, on="unit_id", how="left")
                final_units = final[["unit_id", "label_id", "label_type", "selection_reason"]].drop_duplicates()
                final_out = final_units.merge(fam_wide, on="unit_id", how="left")
                final_out.to_parquet(os.path.join(self.paths.outdir, "final_selection_with_llm.parquet"), index=False)

                labels_path = Path(self.paths.outdir) / "final_llm_labels.parquet"
                try:
                    fam_wide.to_parquet(labels_path, index=False)
                except Exception:
                    pass
                else:
                    try:
                        labels_path.with_suffix(".json").write_text(
                            fam_wide.to_json(orient="records", indent=2, force_ascii=False),
                            encoding="utf-8",
                        )
                    except TypeError:
                        labels_path.with_suffix(".json").write_text(
                            fam_wide.to_json(orient="records"),
                            encoding="utf-8",
                        )

            try:
                probe_json_path = Path(self.paths.outdir) / "final_llm_family_probe.json"
                fam_df.to_json(probe_json_path, orient="records", indent=2, force_ascii=False)
            except TypeError:
                fam_df.to_json(probe_json_path, orient="records")
        if final_out is not None:
            result_df = final_out

        diagnostics = {
            "total_selected": int(len(final)),
            "bucket_sizes": {
                "disagreement": int(len(dis_units) if "dis_units" in locals() else 0),
                "diversity": int(len(sel_div_units) if "sel_div_units" in locals() else 0),
                "uncertain": int(len(sel_unc_units) if "sel_unc_units" in locals() else 0),
                "certain": int(len(sel_cer_units) if "sel_cer_units" in locals() else 0),
            },
            "unique_units": int(len(final["unit_id"].unique())),
        }

        try:
            rec_path = LLM_RECORDER.flush()
            if rec_path:
                import logging

                LOGGER = logging.getLogger(__name__)
                LOGGER.info("llm_run_log_written", extra={"path": rec_path, "n_calls": len(LLM_RECORDER.calls)})
        except Exception:
            import logging

            LOGGER = logging.getLogger(__name__)
            LOGGER.warning("llm_run_log_write_failed")

        total_seconds = time.time() - t0
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Done. Total elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

        if self.attach_metadata:
            result_df = self.attach_metadata(result_df)
        return result_df
