from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple

import pandas as pd
import random


@dataclass
class ActiveLearningSelector:
    select_config: object
    label_types: Optional[dict[str, str]] = None
    current_label_ids: Optional[Set[str]] = None
    seen_units: Optional[Set[str]] = None
    unseen_pairs: Optional[Sequence[Tuple[str, str]]] = None

    def _empty_unit_frame(self) -> "pd.DataFrame":
        return pd.DataFrame(
            {
                "unit_id": pd.Series(dtype="string"),
                "label_id": pd.Series(dtype="string"),
                "label_type": pd.Series(dtype="string"),
                "selection_reason": pd.Series(dtype="string"),
            }
        )

    def _ensure_unit_schema(self, df: "pd.DataFrame" | None) -> "pd.DataFrame":
        base = self._empty_unit_frame()
        if df is None:
            return base
        result = df.copy()
        for col in base.columns:
            if col not in result.columns:
                result[col] = pd.Series(dtype=base[col].dtype)
        return result[base.columns.tolist()]

    def _to_unit_only(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if df is None:
            return self._empty_unit_frame()
        cols = [c for c in ["unit_id", "label_id", "label_type", "selection_reason"] if c in df.columns]
        if not cols:
            return self._empty_unit_frame()
        subset = df[cols].drop_duplicates(subset=["unit_id"], keep="first")
        return self._ensure_unit_schema(subset)

    def _filter_units(self, df: "pd.DataFrame", excluded: Set[str]) -> "pd.DataFrame":
        if df is None or df.empty or not excluded:
            return df if df is not None else pd.DataFrame(columns=["unit_id"])
        return df[~df["unit_id"].isin(excluded)].copy()

    def _head_units(self, df: "pd.DataFrame", k: int) -> "pd.DataFrame":
        ensured = self._ensure_unit_schema(df)
        if ensured.empty or k <= 0:
            return ensured.iloc[0:0].copy()
        return ensured.head(k).copy()

    def _select_bucket(self, df: "pd.DataFrame", k: int, excluded: Iterable[str] | None = None) -> "pd.DataFrame":
        excluded_set = set(excluded or [])
        filtered = self._filter_units(df, excluded_set)
        return self._head_units(self._to_unit_only(filtered), k)

    def select_disagreement(self, disagree_df: "pd.DataFrame", *, selected_units: Optional[Set[str]] = None) -> "pd.DataFrame":
        total = int(getattr(self.select_config, "batch_size", 0))
        n_dis = int(total * getattr(self.select_config, "pct_disagreement", 0))
        return self._select_bucket(disagree_df, n_dis, excluded=selected_units or set())

    def top_off_random(self, current_sel: pd.DataFrame, target_n: int) -> pd.DataFrame:
        sel = current_sel.copy()
        need = int(target_n) - len(sel)
        if need <= 0:
            return sel

        label_types = self.label_types or {}
        seen_units = self.seen_units or set()
        taken = set(zip(sel["unit_id"].astype(str), sel["label_id"].astype(str)))
        excluded_units = seen_units | set(sel["unit_id"].astype(str))
        cand = [
            (str(u), str(l))
            for (u, l) in (self.unseen_pairs or [])
            if (str(u), str(l)) not in taken and str(u) not in excluded_units
        ]
        if not cand:
            return sel

        rng = random.Random()
        rng.shuffle(cand)

        take = cand[:need]
        if not take:
            return sel

        add = pd.DataFrame(
            [
                {
                    "unit_id": u,
                    "label_id": l,
                    "label_type": label_types.get(l, "text"),
                    "selection_reason": "random_topoff",
                }
                for (u, l) in take
            ]
        )

        return pd.concat([sel, add], ignore_index=True)

    def build_next_batch(
        self,
        disagree_df: pd.DataFrame,
        uncertainty_df: pd.DataFrame,
        easy_df: pd.DataFrame,
        diversity_df: pd.DataFrame,
        *,
        prefiltered: bool = False,
    ) -> pd.DataFrame:
        total = int(getattr(self.select_config, "batch_size", 0))
        n_dis = int(total * getattr(self.select_config, "pct_disagreement", 0))
        n_div = int(total * getattr(self.select_config, "pct_diversity", 0))
        n_unc = int(total * getattr(self.select_config, "pct_uncertain", 0))
        n_cer = int(total * getattr(self.select_config, "pct_easy_qc", 0))

        selected_rows: list[pd.DataFrame] = []
        selected_units: set[str] = set()
        seen_units = self.seen_units or set()

        if prefiltered:
            dis_units = self._ensure_unit_schema(disagree_df)
        else:
            dis_units = self._select_bucket(disagree_df, n_dis, excluded=selected_units)
        selected_rows.append(dis_units)
        selected_units |= set(dis_units["unit_id"])

        want_div = min(n_div, max(0, total - len(selected_units)))
        if prefiltered:
            sel_div_units = self._ensure_unit_schema(diversity_df)
        else:
            sel_div_units = self._select_bucket(
                diversity_df,
                want_div,
                excluded=seen_units | selected_units,
            )
        selected_rows.append(sel_div_units)
        selected_units |= set(sel_div_units["unit_id"])

        want_unc = min(n_unc, max(0, total - len(selected_units)))
        if prefiltered:
            sel_unc_units = self._ensure_unit_schema(uncertainty_df)
        else:
            sel_unc_units = self._select_bucket(
                uncertainty_df,
                want_unc,
                excluded=seen_units | selected_units,
            )
        if not sel_unc_units.empty:
            selected_rows.append(sel_unc_units)
            selected_units |= set(sel_unc_units["unit_id"])

        want_cer = min(n_cer, max(0, total - len(selected_units)))
        if prefiltered:
            sel_cer_units = self._ensure_unit_schema(easy_df)
        else:
            sel_cer_units = self._select_bucket(
                easy_df,
                want_cer,
                excluded=seen_units | selected_units,
            )
        if not sel_cer_units.empty:
            selected_rows.append(sel_cer_units)
            selected_units |= set(sel_cer_units["unit_id"])

        final = (
            pd.concat(selected_rows, ignore_index=True)
            if selected_rows
            else self._empty_unit_frame()
        )
        final = final.drop_duplicates(subset=["unit_id"], keep="first").copy()

        if len(final) < total:
            print("Topping off to target batch_size ...")
            final = self.top_off_random(final, target_n=total)
            final = final.drop_duplicates(subset=["unit_id"], keep="first")

        return final
