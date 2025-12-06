"""Disagreement scoring helpers for active learning selection."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from vaannotate.vaannotate_ai_backend.core.data import DataRepository
from vaannotate.vaannotate_ai_backend.services.disagreement_expander import DisagreementExpander
from vaannotate.vaannotate_ai_backend.services.label_dependencies import build_label_dependencies


class DisagreementScorer:
    def __init__(
        self,
        data_repo: DataRepository,
        disagree_config,
        select_config,
        pooler,
        retriever,
        expander: DisagreementExpander,
        *,
        context_builder=None,
        kcenter_select_fn: Optional[Callable] = None,
    ):
        self.repo = data_repo
        self.cfg = disagree_config
        self.select_cfg = select_config
        self.pooler = pooler
        self.retriever = retriever
        self.expander = expander
        self.context_builder = context_builder
        self.kcenter_select = kcenter_select_fn

    def _per_label_kcenter(
        self,
        pool_df: pd.DataFrame,
        rules_map: Dict[str, str],
        per_label_quota: int,
        already_selected_vecs: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        out = []
        for lid, grp in pool_df.groupby("label_id"):
            if grp.empty:
                continue
            k = min(per_label_quota, len(grp))
            vecs = [
                self.pooler.pooled_vector(
                    r.unit_id,
                    r.label_id,
                    self.retriever,
                    rules_map.get(r.label_id, ""),
                    context_builder=self.context_builder,
                )
                for r in grp.itertuples(index=False)
            ]
            V = np.vstack(vecs)
            idxs = self.kcenter_select(V, k=k, seed_idx=None, preselected=already_selected_vecs)
            sel = grp.iloc[idxs].copy()
            sel = sel.reset_index(drop=True)
            sel["kcenter_rank_per_label"] = list(range(1, len(sel) + 1))
            out.append(sel)
        return pd.concat(out, ignore_index=True) if out else pool_df.head(0)

    def _merge_with_global_kcenter(
        self,
        per_label_selected: pd.DataFrame,
        pool_df: pd.DataFrame,
        rules_map: Dict[str, str],
        target_n: int,
        already_selected_vecs: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if per_label_selected.empty and pool_df.empty:
            return per_label_selected
        selected = per_label_selected.copy()
        cnt = Counter(selected["label_id"].tolist()) if not selected.empty else Counter()
        remaining_needed = max(0, target_n - len(selected))
        if remaining_needed == 0:
            return selected
        if not selected.empty:
            rem = pool_df.merge(selected[["unit_id", "label_id"]], on=["unit_id", "label_id"], how="left", indicator=True)
            rem = rem[rem["_merge"] == "left_only"].drop(columns=["_merge"])
        else:
            rem = pool_df.copy()
        if rem.empty:
            return selected

        vecs = [
            self.pooler.pooled_vector(
                r.unit_id,
                r.label_id,
                self.retriever,
                rules_map.get(r.label_id, ""),
                context_builder=self.context_builder,
            )
            for r in rem.itertuples(index=False)
        ]
        V = np.vstack(vecs)

        preV = None
        if already_selected_vecs is not None and already_selected_vecs.size:
            preV = already_selected_vecs
        if not selected.empty:
            Vsel = [
                self.pooler.pooled_vector(
                    r.unit_id,
                    r.label_id,
                    self.retriever,
                    rules_map.get(r.label_id, ""),
                    context_builder=self.context_builder,
                )
                for r in selected.itertuples(index=False)
            ]
            Vsel = np.vstack(Vsel)
            preV = Vsel if preV is None else np.vstack([preV, Vsel])

        idx_order = self.kcenter_select(V, k=min(remaining_needed, len(rem)), seed_idx=None, preselected=preV)

        chosen_rows = []
        for rank_idx, i in enumerate(idx_order, start=1):
            row = rem.iloc[i]
            row = row.copy()
            row["kcenter_rank_global"] = rank_idx
            chosen_rows.append(row)
            cnt[row.label_id] += 1
            if len(chosen_rows) >= remaining_needed:
                break

        if not chosen_rows:
            return selected
        chosen_df = pd.DataFrame([r._asdict() if hasattr(r, "_asdict") else dict(r) for r in chosen_rows])
        return pd.concat([selected, chosen_df], ignore_index=True)

    def compute_disagreement(
        self,
        seen_pairs: Set[Tuple[str, str]],
        rules_map: Dict[str, str],
        label_types: Dict[str, str],
    ) -> pd.DataFrame:
        import pandas as pd

        def _apportion_by_mix_fair(counts: pd.Series, total: int) -> Dict[str, int]:
            counts = counts[counts > 0]
            if total <= 0 or counts.empty:
                return {}
            labels = counts.index.tolist()
            L = len(labels)

            base_each = 1 if total >= L else 0
            base = {str(l): base_each for l in labels}
            remaining = total - base_each * L
            if remaining < 0:
                base = {str(l): 0 for l in labels}
                remaining = total

            weights = counts / counts.sum()
            exact = weights * remaining
            floors = exact.apply(int)
            tgt = {str(l): int(base[str(l)] + floors.get(l, 0)) for l in labels}
            give = int(remaining - floors.sum())
            if give > 0:
                rema = (exact - floors).sort_values(ascending=False)
                for l in rema.index[:give]:
                    tgt[str(l)] = tgt.get(str(l), 0) + 1
            return {k: int(max(0, v)) for k, v in tgt.items()}

        def _select_for_label(pool_df: pd.DataFrame, label_id: str, k: int) -> pd.DataFrame:
            if k <= 0:
                return pool_df.head(0)
            sub = pool_df[pool_df["label_id"].astype(str) == str(label_id)]
            if sub.empty:
                return sub
            sel = self._per_label_kcenter(sub, rules_map, per_label_quota=int(k))
            return sel.head(k)

        expanded = self.expander.expand(rules_map, seen_pairs)
        if expanded is None or expanded.empty:
            return expanded

        dep_cache: Dict[str, tuple[dict, dict, list]] = {}

        def _dependencies_for(labelset_id: Optional[str]) -> tuple[dict, dict, list]:
            key = str(labelset_id or "__current__")
            if key not in dep_cache:
                config = self.expander.label_config_bundle.config_for_labelset(labelset_id)
                try:
                    deps = build_label_dependencies(config)
                except Exception:
                    deps = ({}, {}, [])
                dep_cache[key] = deps
            return dep_cache[key]

        expanded["labelset_id"] = expanded.get("labelset_id", "").fillna("").astype(str)
        expanded["is_root_parent"] = False
        for lsid, grp in expanded.groupby("labelset_id"):
            parent_to_children, child_to_parents, roots = _dependencies_for(lsid)
            roots_set = set(str(x) for x in (roots or []))
            mask = grp["label_id"].astype(str).isin(roots_set)
            expanded.loc[mask.index, "is_root_parent"] = mask

        df = expanded
        dep_cache = {}
        counts_dict: Dict[str, int] = {}
        for uid, grp in df.groupby("unit_id"):
            labelset_id = None
            if "labelset_id" in grp.columns:
                non_empty = grp["labelset_id"].dropna().astype(str).str.strip()
                if not non_empty.empty:
                    labelset_id = str(non_empty.iloc[0]) or None
            parent_to_children, child_to_parents, roots = _dependencies_for(labelset_id)
            gr = grp.copy()
            gr["label_id"] = gr["label_id"].astype(str)
            roots_set = set(str(x) for x in (roots or []))
            for lid, g in gr.groupby("label_id"):
                cnt = len(g)
                if lid in roots_set:
                    pass
                else:
                    parents = child_to_parents.get(lid, [])
                    if parents:
                        c = 0
                        for p in parents:
                            if parent_to_children.get(p, []):
                                c += 1
                        cnt = c
                counts_dict[lid] = counts_dict.get(lid, 0) + cnt

        counts_series = pd.Series(counts_dict, dtype="int64")
        n_dis = int(self.select_cfg.batch_size * self.select_cfg.pct_disagreement)

        if counts_series.sum() <= 0:
            counts_series = df["label_id"].value_counts()

        target_per_label = _apportion_by_mix_fair(counts_series, n_dis)

        avail = df["label_id"].value_counts()
        for lid in list(target_per_label.keys()):
            target_per_label[lid] = int(min(target_per_label[lid], int(avail.get(lid, 0))))

        parents_pool = df[df["is_root_parent"]].copy()
        children_pool = df[~df["is_root_parent"]].copy()

        selected_parts = []
        taken_by_label: Dict[str, int] = defaultdict(int)

        par_chunks = []
        for lid, tgt in target_per_label.items():
            if tgt <= 0:
                continue
            s = _select_for_label(parents_pool, lid, tgt)
            if not s.empty:
                s = s.copy()
                s["selection_reason"] = "disagreement_parent"
                par_chunks.append(s)
                taken_by_label[lid] += int(len(s))
        if par_chunks:
            selected_parts.append(pd.concat(par_chunks, ignore_index=True))

        ch_chunks = []
        for lid, tgt in target_per_label.items():
            rem = tgt - int(taken_by_label.get(lid, 0))
            if rem <= 0:
                continue
            s = _select_for_label(children_pool, lid, rem)
            if not s.empty:
                s = s.copy()
                s["selection_reason"] = "disagreement_child"
                ch_chunks.append(s)
                taken_by_label[lid] += int(len(s))
        if ch_chunks:
            selected_parts.append(pd.concat(ch_chunks, ignore_index=True))

        if not selected_parts:
            return df.head(0)

        sel = pd.concat(selected_parts, ignore_index=True)

        if len(sel) < n_dis:
            remaining_pool = df.merge(sel[["unit_id", "label_id"]], on=["unit_id", "label_id"], how="left", indicator=True)
            remaining_pool = remaining_pool[remaining_pool["_merge"] == "left_only"].drop(columns=["_merge"])

            need = n_dis - len(sel)

            if need > 0:
                child_rem = remaining_pool[~remaining_pool["is_root_parent"]]
                if not child_rem.empty:
                    fill = self._merge_with_global_kcenter(
                        sel, child_rem, rules_map, target_n=len(sel) + min(need, len(child_rem))
                    )
                    added = fill.merge(sel[["unit_id", "label_id"]], on=["unit_id", "label_id"], how="left", indicator=True)
                    added = added[added["_merge"] == "left_only"].drop(columns=["_merge"])
                    if not added.empty:
                        added = added.copy()
                        added["selection_reason"] = "disagreement_child_topup"
                        sel = pd.concat([sel, added], ignore_index=True)
                        need = n_dis - len(sel)

            if need > 0:
                par_rem = remaining_pool[remaining_pool["is_root_parent"]]
                if not par_rem.empty:
                    fill = self._merge_with_global_kcenter(
                        sel, par_rem, rules_map, target_n=len(sel) + min(need, len(par_rem))
                    )
                    added = fill.merge(sel[["unit_id", "label_id"]], on=["unit_id", "label_id"] , how="left", indicator=True)
                    added = added[added["_merge"] == "left_only"].drop(columns=["_merge"])
                    if not added.empty:
                        added = added.copy()
                        added["selection_reason"] = "disagreement_parent_topup"
                        sel = pd.concat([sel, added], ignore_index=True)

        sel["label_type"] = sel["label_id"].map(lambda x: label_types.get(x, "text"))
        return sel
