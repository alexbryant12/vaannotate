"""Seed expansion utilities for disagreement-based sampling."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..config import DisagreementConfig, LLMFirstConfig
from ..core.data import DataRepository
from ..label_configs import EMPTY_BUNDLE, LabelConfigBundle
from ..services.label_dependencies import build_label_dependencies, evaluate_gating
from ..services.rag_retriever import RAGRetriever
from ..utils.io import normalize01
from .context_builder import ContextBuilder


class DisagreementExpander:
    def __init__(
        self,
        cfg: DisagreementConfig,
        repo: DataRepository,
        retriever: RAGRetriever,
        label_config_bundle: LabelConfigBundle | None = None,
        llmfirst_cfg: LLMFirstConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ):
        self.cfg = cfg
        self.repo = repo
        self.retriever = retriever
        self.label_config_bundle = label_config_bundle or EMPTY_BUNDLE
        self._dependency_cache: Dict[str, tuple[dict, dict, list]] = {}
        self.llmfirst_cfg = llmfirst_cfg
        self.context_builder = context_builder

    def _dependencies_for(self, labelset_id: Optional[str]) -> tuple[dict, dict, list]:
        key = str(labelset_id or "__current__")
        if key not in self._dependency_cache:
            config = self.label_config_bundle.config_for_labelset(labelset_id)
            try:
                deps = build_label_dependencies(config)
            except Exception:
                deps = ({}, {}, [])
            self._dependency_cache[key] = deps
        return self._dependency_cache[key]

    def high_entropy_seeds(self) -> pd.DataFrame:
        dis = self.repo.reviewer_disagreement(round_policy=self.cfg.round_policy, decay_half_life=self.cfg.decay_half_life)
        seeds = dis[dis["disagreement_score"] >= self.cfg.high_entropy_threshold].copy()
        types = self.repo.label_types()
        hard_df = self.repo.hard_disagree(types, date_days=int(self.cfg.date_disagree_days), num_abs=float(self.cfg.numeric_disagree_abs), num_rel=float(self.cfg.numeric_disagree_rel))
        if not hard_df.empty:
            hard_pairs = set(zip(hard_df.loc[hard_df['hard_disagree'],'unit_id'].astype(str), hard_df.loc[hard_df['hard_disagree'],'label_id'].astype(str)))
            if hard_pairs:
                add = dis[[ (str(r.unit_id), str(r.label_id)) in hard_pairs for r in dis.itertuples(index=False) ]].copy()
                seeds = pd.concat([seeds, add], ignore_index=True).drop_duplicates(subset=['unit_id','label_id'])
        seeds = seeds.sort_values("disagreement_score", ascending=False)

        # ---- Parent→child gating on seeds (use prior-round consensus) ----
        seeds = seeds.copy()
        if "labelset_id" not in seeds.columns:
            seeds["labelset_id"] = seeds.apply(
                lambda r: self.repo.labelset_for_annotation(
                    r.get("unit_id", ""),
                    r.get("label_id", ""),
                    round_id=r.get("round_id"),
                )
                or "",
                axis=1,
            )
        else:
            seeds["labelset_id"] = seeds["labelset_id"].fillna("").astype(str)
        seeds["round_id"] = seeds.get("round_id", "").astype(str)
        types = self.repo.label_types()
        consensus = self.repo.last_round_consensus()  # {(unit_id,label_id)-> value str}

        def _gate_seed(row: pd.Series) -> bool:
            uid = str(row["unit_id"])
            lid = str(row["label_id"])
            labelset_id = str(row.get("labelset_id") or "").strip() or None
            parent_to_children, child_to_parents, roots = self._dependencies_for(labelset_id)
            roots_set = set(str(x) for x in (roots or []))
            # Parents (roots) are always eligible; children need parent gate pass
            if str(lid) in roots_set:
                return True
            parents = child_to_parents.get(str(lid), [])
            if not parents:
                return True  # not marked as child → treat as eligible
            parent_preds = {(str(uid), str(p)): consensus.get((str(uid), str(p)), None) for p in parents}
            # IMPORTANT: evaluate by prior-round consensus; robust evaluator handles casing/types
            config = self.label_config_bundle.config_for_labelset(labelset_id)
            return evaluate_gating(str(lid), str(uid), parent_preds, types, config)

        if not seeds.empty:
            seeds["unit_id"] = seeds["unit_id"].astype(str)
            seeds["label_id"] = seeds["label_id"].astype(str)
            seeds = seeds[seeds.apply(_gate_seed, axis=1)]

        rows = []
        for lid, grp in seeds.groupby("label_id"):
            rows.append(grp.head(self.cfg.seeds_per_label))
        return pd.concat(rows, ignore_index=True) if rows else seeds.head(0)

    def seed_snippets(self, unit_id: str, label_id: str, label_rules: str) -> List[str]:
        spans = self.repo.get_prior_rationales(unit_id, label_id)
        snips = [sp.get("snippet") for sp in spans if isinstance(sp,dict) and sp.get("snippet")]
        if snips: return snips[: self.cfg.snippets_per_seed]
        if self.context_builder is not None:
            ctx = self.context_builder.build_context_for_label(
                unit_id,
                label_id,
                label_rules,
                topk_override=self.cfg.snippets_per_seed,
                single_doc_context_mode=getattr(self.llmfirst_cfg, "single_doc_context", "rag"),
                full_doc_char_limit=getattr(self.llmfirst_cfg, "single_doc_full_context_max_chars", None),
            )
        else:
            ctx = self.retriever.retrieve_for_patient_label(
                unit_id,
                label_id,
                label_rules,
                topk_override=self.cfg.snippets_per_seed,
                min_k_override=None,
                mmr_lambda_override=None,
            )
        return [c["text"] for c in ctx if isinstance(c.get("text"), str)]

    def expand(self, rules_map: Dict[str,str], seen_pairs: set) -> pd.DataFrame:
        seeds = self.high_entropy_seeds()
        print('disagreement seeds: ', seeds)
        rows = []
        for lid, grp in seeds.groupby("label_id"):
            labelset_value = ""
            if "labelset_id" in grp.columns:
                non_empty = grp["labelset_id"].dropna().astype(str).str.strip()
                if not non_empty.empty:
                    labelset_value = str(non_empty.iloc[0])
            round_value = ""
            if "round_id" in grp.columns:
                round_non_empty = grp["round_id"].dropna().astype(str).str.strip()
                if not round_non_empty.empty:
                    round_value = str(round_non_empty.iloc[0])
            snips = []
            for r in grp.itertuples(index=False):
                snips.extend(self.seed_snippets(r.unit_id, lid, rules_map.get(lid,"")))
            snips = snips[: self.cfg.seeds_per_label * self.cfg.snippets_per_seed]
            if not snips: continue
            cand = self.retriever.expand_from_snippets(lid, snips, seen_pairs, per_seed_k=self.cfg.similar_chunks_per_seed)
            items = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.expanded_per_label]
            for uid, sc in items:
                rows.append({
                    "unit_id": uid,
                    "label_id": lid,
                    "score": float(sc),
                    "bucket": "disagreement_expanded",
                    "labelset_id": labelset_value,
                    "round_id": round_value,
                })
        df = pd.DataFrame(rows)
        if df.empty: return df
        out = []
        for lid, g in df.groupby("label_id"):
            arr = g["score"].to_numpy()
            g = g.copy(); g["score_n"] = normalize01(arr); out.append(g)
        return pd.concat(out, ignore_index=True)
