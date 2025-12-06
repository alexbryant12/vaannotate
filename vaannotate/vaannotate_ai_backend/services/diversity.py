"""Diversity-focused selection helpers for active learning batches."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def _options_for_label(label_id: str, label_type: str, label_cfgs: dict) -> Optional[List[str]]:
    cfg = label_cfgs.get(label_id, {}) if isinstance(label_cfgs, dict) else {}
    opts = cfg.get("options")
    if isinstance(opts, list) and 2 <= len(opts) <= 26:
        return [str(o) for o in opts]
    return None


class DiversitySelector:
    """Select a diverse subset of candidate (unit, label) pairs."""

    def __init__(self, data_repo, emb_store, diversity_config):
        self.repo = data_repo
        self.store = emb_store
        self.cfg = diversity_config

    def select_diverse_units(
        self,
        candidate_df: pd.DataFrame,
        *,
        n_div: int,
        already_selected=None,
        pooler=None,
        retriever=None,
        rules_map=None,
        label_types=None,
        label_config=None,
        rag_k: int = 4,
        min_rel_quantile: float = 0.30,
        mmr_lambda: float = 0.7,
        sample_cap: int = 2000,
        adaptive_relax: bool = True,
        relax_steps=(0.20, 0.10, 0.05),
        pool_factor: float = 4.0,
        use_proto: bool = False,
        family_labeler=None,
        ensure_exemplars: bool = True,
        exclude_units: set[str] | None = None,
        iter_with_bar_fn=None,
    ) -> pd.DataFrame:
        """Diversity bucket without CE gating (mirrors the prior inline logic)."""

        def _mmr_select_simple(vecs, rel_scores, k, lam=0.7, preselected=None):
            """Greedy MMR: argmax lam*rel – (1–lam)*max_sim_to_selected (no label caps)."""
            if k <= 0 or vecs is None or len(vecs) == 0:
                return []
            V = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            rel = np.asarray(rel_scores, dtype="float32")
            N = V.shape[0]
            selected = []
            if preselected is not None and getattr(preselected, "size", 0):
                P = preselected / (np.linalg.norm(preselected, axis=1, keepdims=True) + 1e-12)
                sim_pre = (V @ P.T).max(axis=1)
            else:
                sim_pre = np.zeros(N, dtype="float32")
            used = np.zeros(N, dtype=bool)
            for _ in range(min(k, N)):
                if selected:
                    S = V[selected, :]
                    sim_sel = (V @ S.T).max(axis=1)
                    sim = np.maximum(sim_pre, sim_sel)
                else:
                    sim = sim_pre
                score = lam * rel - (1.0 - lam) * sim
                score[used] = -1e9
                i = int(np.argmax(score))
                if score[i] <= -1e8:
                    break
                selected.append(i)
                used[i] = True
            return selected

        def _l2(v):
            v = np.asarray(v, dtype="float32")
            n = np.linalg.norm(v) + 1e-12
            return v / n

        def _cos(a, b):
            return float(np.dot(_l2(a), _l2(b)))

        def _kcenter_greedy(U: np.ndarray, k: int, seed_indices=None) -> list[int]:
            if U.size == 0 or k <= 0:
                return []
            Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
            n = Un.shape[0]
            sel, dmin = [], np.ones(n) * np.inf
            if seed_indices:
                seeds = [s for s in seed_indices if 0 <= s < n]
                if seeds:
                    sel.extend(seeds)
                    sims = Un @ Un[seeds].T
                    dmin = 1 - sims.max(axis=1)
                    dmin[seeds] = 0.0
            if not sel:
                s0 = int(np.random.randint(0, n))
                sel.append(s0)
                dmin = 1 - (Un @ Un[s0])
            while len(sel) < min(k, n):
                i = int(np.argmax(dmin))
                sel.append(i)
                dmin = np.minimum(dmin, 1 - (Un @ Un[i]))
            return sel

        def _equal_quota(group_sizes: dict[str, int], total: int) -> dict[str, int]:
            labs = list(group_sizes.keys())
            if not labs or total <= 0:
                return {lab: 0 for lab in labs}
            base, rem = total // len(labs), total % len(labs)
            q = {lab: min(group_sizes[lab], base) for lab in labs}
            order = sorted(labs, key=lambda x: group_sizes[x] - q[x], reverse=True)
            for lab in order:
                if rem <= 0:
                    break
                if q[lab] < group_sizes[lab]:
                    q[lab] += 1
                    rem -= 1
            while rem > 0:
                progressed = False
                for lab in order:
                    if q[lab] < group_sizes[lab]:
                        q[lab] += 1
                        rem -= 1
                        progressed = True
                        if rem == 0:
                            break
                if not progressed:
                    break
            return q

        def _empty():
            return pd.DataFrame(columns=["unit_id", "label_id", "label_type", "selection_reason"])

        rules_map = rules_map or {}
        label_types = label_types or {}
        exclude_units = exclude_units or set()
        already_selected = already_selected or []
        pooler = pooler or getattr(self, "pooler", None)
        retriever = retriever or getattr(self, "retriever", None)

        if (
            candidate_df is None
            or not isinstance(candidate_df, pd.DataFrame)
            or candidate_df.empty
            or pooler is None
            or retriever is None
        ):
            return _empty()

        if ensure_exemplars and family_labeler is not None:
            try:
                K_use = int(getattr(getattr(family_labeler, "cfg", None), "exemplar_K", 6) or 6)
                family_labeler.ensure_label_exemplars(rules_map, K=K_use)
            except Exception:
                pass

        if {"unit_id", "label_id"}.issubset(set(candidate_df.columns)):
            unseen_pairs = [
                (str(getattr(r, "unit_id", "")), str(getattr(r, "label_id", "")))
                for r in candidate_df.itertuples(index=False)
                if getattr(r, "unit_id", None) is not None and getattr(r, "label_id", None) is not None
            ]
        else:
            unseen_pairs = []

        if n_div <= 0 or not unseen_pairs:
            return _empty()

        rem_all = [(u, l) for (u, l) in unseen_pairs if (u, l) not in set(already_selected)]
        if exclude_units:
            rem_all = [(u, l) for (u, l) in rem_all if u not in exclude_units]
        np.random.shuffle(rem_all)
        rem_all = rem_all[: max(n_div * 8, min(sample_cap, len(rem_all)))]
        if not rem_all:
            return _empty()

        rem_df = pd.DataFrame([
            {"unit_id": u, "label_id": str(l), "label_type": label_types.get(l, "text")}
            for (u, l) in rem_all
        ])

        proto_cache: dict[str, np.ndarray] = {}

        def _label_anchor(lid: str) -> np.ndarray:
            if lid in proto_cache:
                return proto_cache[lid]

            proto = None

            if hasattr(pooler, "label_prototype") and use_proto:
                try:
                    proto = pooler.label_prototype(lid, retriever, rules_map.get(lid, ""))
                except Exception:
                    proto = None

            if proto is None:
                Q = None
                try:
                    base_k = getattr(getattr(retriever, "cfg", None), "exemplar_K", None)
                    K_use = int(base_k if base_k is not None else 6)
                    if K_use <= 0:
                        K_use = 6
                    opts = _options_for_label(
                        lid,
                        label_types.get(str(lid), "categorical"),
                        getattr(retriever, "label_configs", {}),
                    ) or []
                    if opts:
                        K_use = max(K_use, len(opts))
                    getQ = getattr(retriever, "_get_label_query_embs", None)
                    if callable(getQ):
                        Q = getQ(lid, rules_map.get(lid, ""), K_use)
                except Exception:
                    Q = None
                if Q is not None and getattr(Q, "ndim", 1) == 2 and Q.shape[0] > 0:
                    proto = Q.mean(axis=0)

            if proto is None:
                q = retriever._build_query(lid, rules_map.get(lid, ""))
                proto = retriever.store._embed([q])[0]

            proto_cache[lid] = np.asarray(proto, dtype="float32")
            return proto_cache[lid]

        iter_fn = iter_with_bar_fn or (lambda step, iterable, **kwargs: iterable)
        vecs, rels = [], []
        _progress_every = float(getattr(getattr(family_labeler, "cfg", None), "progress_min_interval_s", 1) or 1)
        for r in iter_fn(
            step="Diversity: pooling vectors",
            iterable=rem_df.itertuples(index=False),
            total=len(rem_df),
            min_interval_s=_progress_every,
        ):
            v = pooler.pooled_vector(r.unit_id, r.label_id, retriever, rules_map.get(r.label_id, ""))
            vecs.append(v)
            rels.append(_cos(v, _label_anchor(r.label_id)))
        rem_df["vec"] = vecs
        rem_df["rel"] = np.array(rels, dtype="float32")

        def _keep_by_q(df, q):
            kept = []
            for lid, g in df.groupby("label_id", sort=False):
                if g.empty:
                    continue
                thr = float(np.quantile(g["rel"].to_numpy(), q)) if len(g) > 1 else g["rel"].min()
                kept.append(g[g["rel"] >= thr])
            return pd.concat(kept, ignore_index=True) if kept else df.head(0)

        if min_rel_quantile is not None:
            gated = _keep_by_q(rem_df, float(min_rel_quantile))
            if adaptive_relax and len(gated) < n_div:
                for q in relax_steps:
                    gated = _keep_by_q(rem_df, float(q))
                    if len(gated) >= n_div:
                        break
        else:
            gated = rem_df
        if gated.empty:
            return gated

        pool_total = int(min(len(gated), max(n_div, int(n_div * pool_factor))))
        sizes = {lid: len(g) for lid, g in gated.groupby("label_id", sort=False)}
        quotas = _equal_quota(sizes, pool_total)

        pools, sel_pairs = [], set()
        preV_by_label: dict[str, np.ndarray] = {}

        for lid, g in gated.groupby("label_id", sort=False):
            k_lab = quotas.get(lid, 0)
            if k_lab <= 0 or g.empty:
                continue
            V = np.stack(g["vec"].to_list()).astype("float32")
            rel = g["rel"].to_numpy().astype("float32")
            preV = preV_by_label.get(lid)
            order = _mmr_select_simple(V, rel, k=k_lab, lam=mmr_lambda, preselected=preV)
            choice = g.iloc[order].head(k_lab) if order else g.sort_values("rel", ascending=False).head(k_lab)
            for r in choice.itertuples(index=False):
                key = (r.unit_id, r.label_id)
                if key in sel_pairs:
                    continue
                sel_pairs.add(key)
                pools.append(r)

        if not pools:
            add_df = gated.sort_values("rel", ascending=False).head(n_div).copy()
            add_df["selection_reason"] = "diversity_toprel_fallback"
            return add_df[["unit_id", "label_id", "label_type", "selection_reason"]]

        pool_df = pd.DataFrame(pools)
        if len(pool_df) > pool_total:
            pool_df = pool_df.sample(pool_total).reset_index(drop=True)

        by_unit = {}
        for r in pool_df.itertuples(index=False):
            by_unit.setdefault(r.unit_id, {"labels": [], "vecs": [], "best_label": None, "best_rel": -1.0})
            by_unit[r.unit_id]["labels"].append(r.label_id)
            by_unit[r.unit_id]["vecs"].append(np.asarray(r.vec, dtype="float32"))
            if float(r.rel) > by_unit[r.unit_id]["best_rel"]:
                by_unit[r.unit_id]["best_rel"] = float(r.rel)
                by_unit[r.unit_id]["best_label"] = r.label_id

        units = list(by_unit.keys())
        if not units:
            return _empty()

        U = np.stack([np.mean(np.stack(v["vecs"], axis=0), axis=0) for v in by_unit.values()], axis=0)
        idx_map = {u: i for i, u in enumerate(units)}
        seed_units = {u for (u, _l) in already_selected}
        seeds = [idx_map[u] for u in seed_units if u in idx_map]

        picks = _kcenter_greedy(U, k=n_div, seed_indices=seeds)

        rows = []
        for i in picks:
            u = units[i]
            lab = by_unit[u]["best_label"]
            rows.append({
                "unit_id": u,
                "label_id": lab,
                "label_type": label_types.get(lab, "text"),
                "selection_reason": "diversity_mmr_kcenter",
            })
        return pd.DataFrame(rows)


__all__ = ["DiversitySelector"]
