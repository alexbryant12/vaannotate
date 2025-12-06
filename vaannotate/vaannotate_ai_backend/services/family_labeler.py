from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..config import LLMFirstConfig, SCJitterConfig
from ..core.data import DataRepository
from ..services import ContextBuilder, LLMLabeler, LLM_RECORDER
from ..services.label_dependencies import build_label_dependencies, evaluate_gating
from ..services.rag_retriever import RAGRetriever
from ..utils.runtime import iter_with_bar

# ------------------------------
def _options_for_label(label_id: str, label_type: str, label_cfgs: dict) -> Optional[List[str]]:
    cfg = label_cfgs.get(label_id, {}) if isinstance(label_cfgs, dict) else {}
    opts = cfg.get("options")
    if isinstance(opts, list) and 2 <= len(opts) <= 26:
        return [str(o) for o in opts]
    return None

class FamilyLabeler:
    """Label entire family tree per unit, honoring parent-child gating."""
    def __init__(self, llm: LLMLabeler, retriever: RAGRetriever, repo: DataRepository, label_config: dict, scCfg: SCJitterConfig, llmfirst_cfg: LLMFirstConfig):
        self.llm = llm
        self.retriever = retriever
        self.repo = repo
        self.label_config = label_config or {}
        self.scCfg = scCfg
        self.cfg = llmfirst_cfg
        self.context_builder = getattr(retriever, "context_builder", None)
        self.parent_to_children, self.child_to_parents, self.roots = build_label_dependencies(self.label_config)
        
    
    def ensure_label_exemplars(self, rules_map: dict[str, str], K: int = 6):
        try:
            label_types = self.repo.label_types()
        except Exception:
            label_types = {}
        for lid, rules in (rules_map or {}).items():
            base_k = getattr(self.cfg, "exemplar_K", None)
            K_use = int(base_k if base_k is not None else K)
            if K_use <= 0:
                K_use = K
            opts = _options_for_label(lid, label_types.get(str(lid), "categorical"), self.label_config) or []
            if opts:
                K_use = max(K_use, len(opts))
            # skip if already cached
            if self.retriever._get_label_query_embs(lid, rules, K_use) is not None:
                continue
            t = self._generate_label_exemplars(lid, rules, K_use)
            if not t:
                t = self._fallback_label_exemplars(lid, rules, K_use, opts)
            if t:
                self.retriever.set_label_exemplars(lid, rules, K_use, t)

    def _fallback_label_exemplars(self, label_id: str, rules: str, K: int, options: list[str] | None) -> list[str]:
        """Graceful fallback if exemplar generation fails (e.g., Azure JSON mode issues).

        We synthesize minimal exemplars using the label rules so downstream RAG still
        has deterministic label-aware queries instead of aborting with
        "label_exemplar_error". These are intentionally simple and will be embedded
        just like LLM-generated snippets.
        """
        K = max(1, int(K))
        base = self.retriever._build_query(label_id, rules)
        texts: list[str] = []
        opts = options or []
        if opts:
            for o in opts:
                texts.append(f"{base} Example of option '{o}'.")
        if not texts:
            texts.append(base)
        while len(texts) < K:
            texts.append(base)
        return texts[:K]

    def _generate_label_exemplars(
        self,
        label_id: str,
        rules: str,
        K: int,
        options: list[str] | None = None,
    ) -> list[str]:
        """
        Generate K realistic clinical snippets (1–3 sentences) that satisfy the rule.
        - If the label has options (categorical/binary), request JSON objects with {"option","text"}
          and return a mix across options when feasible.
        - If no options (numeric/date/etc.), request JSON with {"snippets": [<string>, ...]}.
        - Azure OpenAI JSON mode (json_object); robust parsing and de-dup; returns List[str].
        """
        import time as _time
        import json as _json
    
        # Resolve options from label_config if not explicitly provided
        opts = options
        if not opts:
            try:
                lblcfgs = getattr(self.retriever, "label_configs", {}) or {}
                cfg = lblcfgs.get(label_id, {}) if isinstance(lblcfgs, dict) else {}
                raw = cfg.get("options")
                if isinstance(raw, list) and 2 <= len(raw) <= 26:
                    opts = [str(x) for x in raw]
            except Exception:
                opts = None
        has_opts = bool(opts)
    
        # ----- Build prompt (conditional on options) -----
        if has_opts:
            menu_text = "\n".join(f"- {o}" for o in opts)
            system = (
                "You write realistic clinical note snippets.\n"
                "Output policy:\n"
                "- Return ONLY a valid JSON object with a single key 'snippets', whose value is an array of objects.\n"
                "- Each object MUST have: {\"option\": <string>, \"text\": <string>}.\n"
                "- No extra keys, no commentary."
            )
            user = f"""
                Produce {K} diverse clinical note snippets (2-4 sentences each) that would *satisfy* this label rule.
                
                Label: {label_id}
                Rule: {rules or '(no extra rules)'}
                Options:
                {menu_text}
                
                Constraints:
                - Style: realistic, diverse, clinician-authored note sentences. No headers, no lists, no commentary.
                - No meta text (no “the patient meets criteria because...”, no references to “label” or “rule”).
                - Vary phrasing, timing (present/past/relative dates), and synonyms.
                - Keep PHI generic.
                
                Provide a MIX across options so each option appears at least once if feasible.
                Set "option" to exactly one of: {", ".join(opts)}.
                
                Return ONLY:
                {{
                  "snippets": [
                    {{"option": "<one of the options>", "text": "<2–4 sentence snippet>"}}
                  ]
                }}
                """.strip()
        else:
            system = (
                "You write realistic clinical note snippets.\n"
                "Output policy:\n"
                "- Return ONLY a valid JSON object with a single key 'snippets', whose value is an array of strings.\n"
                "- Do NOT include an 'option' field.\n"
                "- No extra keys, no commentary."
            )
            user = f"""
                Produce {K} diverse clinical note snippets (2–4 sentences each) that would *satisfy* this label rule.
                
                Label: {label_id}
                Rule: {rules or '(no extra rules)'}
                
                Constraints:
                - Style: realistic, diverse, clinician-authored note sentences. No headers, no lists, no commentary.
                - No meta text (no “the patient meets criteria because...”, no references to “label” or “rule”).
                - Vary phrasing, timing (present/past/relative dates), and synonyms.
                - Keep PHI generic.
                
                Return ONLY:
                {{
                  "snippets": ["<2–4 sentence snippet>", "..."]
                }}
                """.strip()
    
        # ----- Backend JSON call -----
        temp = float(getattr(self.cfg, "exemplar_temperature", 0.7) or 0.7)

        if has_opts:
            snippets_schema: Mapping[str, Any] = {
                "type": "object",
                "properties": {
                    "snippets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "option": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["option", "text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["snippets"],
                "additionalProperties": False,
            }
        else:
            snippets_schema = {
                "type": "object",
                "properties": {
                    "snippets": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["snippets"],
                "additionalProperties": False,
            }

        try:
            result = self.llm.backend.json_call(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temp,
                logprobs=False,
                top_logprobs=None,
                response_format={
                    "type": "json_object",
                    "json_schema": snippets_schema,
                },
            )
            content = result.content
            # Capture the exact prompt + minimal context identifiers used this vote
            LLM_RECORDER.record("label_exemplar", {
                "label_id": label_id,
                "prompt": {"system": system, "user": user},
                "params": {
                    "temperature": temp,
                },
                "output": content,
            })
        except Exception:
            LLM_RECORDER.record("label_exemplar_error", {})
            return []
        
        # ----- Parse JSON -----
        texts_by_opt: dict[str, list[str]] = {}
        texts: list[str] = []
    
        def _clean_text(s):
            return str(s).strip()
    
        def _coerce_option(o: str) -> str:
            if not has_opts or not isinstance(o, str):
                return str(o or "")
            lut = {s.lower(): s for s in opts}
            return lut.get(o.lower(), str(o).strip())
    
        try:
            data = _json.loads(content) if isinstance(content, str) else content
            if has_opts:
                objs = (data or {}).get("snippets", []) if isinstance(data, dict) else []
                for it in objs:
                    if not isinstance(it, dict): 
                        continue
                    txt = _clean_text(it.get("text", ""))
                    opt = _coerce_option(it.get("option"))
                    if len(txt) > 20 and ((not has_opts) or (opt in (opts or []))):
                        texts_by_opt.setdefault(opt, []).append(txt)
            else:
                arr = (data or {}).get("snippets", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                for s in arr:
                    txt = _clean_text(s)
                    if len(txt) > 20:
                        texts.append(txt)
        except Exception:
            # Fallback: try to pull the first JSON array
            try:
                i = content.find("["); j = content.rfind("]")
                arr = _json.loads(content[i:j+1]) if (i != -1 and j != -1 and j > i) else []
                for s in arr:
                    txt = _clean_text(s)
                    if len(txt) > 20:
                        texts.append(txt)
            except Exception:
                pass
    
        # ----- De-dup and balance / limit to K -----
        if has_opts:
            # De-dup within each option bucket while preserving order
            for o in list(texts_by_opt.keys()):
                seen = set(); ded = []
                for t in texts_by_opt[o]:
                    if t not in seen:
                        seen.add(t); ded.append(t)
                texts_by_opt[o] = ded
    
            # Round-robin across options to get a mix, then fill remaining
            out: list[str] = []
            # one per option if feasible
            for o in opts:
                if len(out) >= K: break
                if texts_by_opt.get(o):
                    out.append(texts_by_opt[o].pop(0))
            # fill remaining round-robin
            while len(out) < K:
                progressed = False
                for o in opts:
                    if len(out) >= K: break
                    bucket = texts_by_opt.get(o) or []
                    if bucket:
                        out.append(bucket.pop(0)); progressed = True
                if not progressed:
                    break
            # if still short, append any leftover
            if len(out) < K:
                for o in opts:
                    for t in texts_by_opt.get(o, []):
                        if len(out) >= K: break
                        out.append(t)
                    if len(out) >= K: break
            return out[:K]
        else:
            # De-dup and cap
            seen = set(); out = []
            for t in texts:
                if t not in seen:
                    seen.add(t); out.append(t)
                if len(out) >= K:
                    break
            return out


    def label_family_for_unit(self, unit_id: str, label_types: dict[str,str], per_label_rules: dict[str,str], *,
                              json_only: bool=False, json_n_consistency: int=1, json_jitter: bool=False) -> list[dict]:
        """Label all parents then gated children for a unit. Returns list of probe rows."""
        self.ensure_label_exemplars(per_label_rules)
        results = []
        # We'll process in a BFS: queue starts with roots, then expand children once parent gating satisfied
        # Maintain map of parent predictions for gating evaluation
        parent_preds: dict[tuple[str,str], Any] = {}
        # Determine an evaluation order by repeatedly scanning for labels whose parents are satisfied or none
        # But to preserve minimal disruption, we simply try roots then others with gating check at point of execution.
        all_labels = list({*self.roots, *[str(lid) for lid in label_types.keys()]})
        # We'll ensure unique and stable order
        seen = set()
        order = []
        for lid in self.roots + [l for l in all_labels if l not in self.roots]:
            if lid not in seen:
                seen.add(lid); order.append(lid)
        for lid in order:
            ltype = label_types.get(lid, 'categorical')
            rules = per_label_rules.get(lid, "")
            # if this label is gated (is a child), check gating
            allowed = evaluate_gating(lid, unit_id, parent_preds, label_types, self.label_config)
            if not allowed:
                # record gated-out for transparency but do not include in probe_df to avoid selection noise
                continue
            label_rows = self.llm.label_unit(
                unit_id,
                [lid],
                label_types=label_types,
                per_label_rules=per_label_rules,
                context_builder=self.context_builder,
                retriever=self.retriever,
                llmfirst_cfg=self.cfg,
                json_only=json_only,
                json_n_consistency=json_n_consistency,
                json_jitter=json_jitter,
            )
            if not label_rows:
                continue
            row = label_rows[0]
            parent_preds[(unit_id, lid)] = row.get("prediction")
            results.append(row)
        return results

    def sample_units_for_probe_enriched(self, per_label_rules: dict[str, str], exclude_units: Optional[set[str]] = None,) -> List[str]:
        """
        CE-ranked enrichment that reuses retriever.retrieve_for_patient_label:
          For each parent P:
            - Randomly sample units from unseen pool
            - Call retrieve_for_patient_label(unit, P, rule) to get top-k patient snippets (already CE-ranked)
            - Unit score = max/mean CE score over top m snippets
          Then allocate quotas across parents, resolve overlaps by best parent, mix with uniform, and top up.
        """
        import numpy as np
    
        rng = np.random.default_rng()
        ex = set(exclude_units or [])
        
        #universe after exclusion
        all_units = [str(u) for u in sorted(self.repo.notes["unit_id"].unique().tolist()) if str(u) not in ex]
        if not all_units:
            return []

        # --- overall mix ---
        # uniform slice only from non-excluded
        n_total  = int(getattr(self.cfg, "n_probe_units", 100) or 100)
        mix      = float(getattr(self.cfg, "probe_enrichment_mix", 0.90) or 0.90)
        n_enr    = int(round(n_total * mix))
        n_unif   = max(0, n_total - n_enr)
        uniform_units = rng.choice(all_units, size=min(n_unif, len(all_units)), replace=False).tolist() if n_unif > 0 else []

    
        # --- parents & pool ---
        parents = list(getattr(self, "roots", []) or list((per_label_rules or {}).keys()))
        if not parents or not all_units:
            # fall back to uniform if nothing sensible
            print('falling back to uniform')
            return list(np.random.default_rng().choice(all_units, size=min(n_total, len(all_units)), replace=False)) if all_units else []
        parent_scores: dict[str, list[tuple[str, float]]] = {}
        best_parent: dict[str, tuple[str, float]] = {}
        
        # --- knobs for CE enrichment ---
        per_parent_sample = int(getattr(self.cfg, "probe_ce_unit_sample", 800) or 800)
        equalize         = bool(getattr(self.cfg, "probe_enrichment_equalize", True))
    
        # ---- score units per parent using your retrieve_for_patient_label ----
        for p in parents:
            rule = (per_label_rules or {}).get(p) or p
            # candidate pool excludes both uniform slice and excluded units
            pool_space = [u for u in all_units if (u not in set(uniform_units))]
            if not pool_space:
                parent_scores[p] = []
                continue
    
            per_parent_sample = int(getattr(self.cfg, "probe_ce_unit_sample", 800) or 800)
            cand_units = rng.choice(pool_space, size=min(per_parent_sample, len(pool_space)), replace=False).tolist()
            
            _progress_every = float(self.cfg.progress_min_interval_s or 1.0)
            unit_chunks, total_pairs = [], 0
            
            for uid in iter_with_bar(
                    step = "Enriching probe pool",
                    iterable = cand_units,
                    total = len(cand_units),
                    min_interval_s=_progress_every):
                if uid in ex:             # redundant but explicit
                    continue
                ctx = self.context_builder.build_context_for_label(
                    uid,
                    p,
                    rule,
                    topk_override=int(getattr(self.cfg, "probe_ce_search_topk_per_unit", 24) or 24),
                    single_doc_context_mode=getattr(self.cfg, "single_doc_context", "rag"),
                    full_doc_char_limit=getattr(self.cfg, "single_doc_full_context_max_chars", None),
                )
                if not ctx:
                    continue
                unit_chunks.append((uid, ctx))
                total_pairs += len(ctx)
    
            if not unit_chunks:
                parent_scores[p] = []
                continue
    
            # CE scores already in ctx['score']; aggregate top-m
            m = int(getattr(self.cfg, "probe_ce_rerank_m", 3) or 3)
            unit_rows = []
            for uid, ctx in unit_chunks:
                top = ctx[:max(1, m)]
                vals = [float(it.get("score", 0.0)) for it in top]
                unit_score = float(np.mean(vals)) if str(getattr(self.cfg, "probe_ce_unit_agg", "max")).lower() == "mean" else float(np.max(vals))
                unit_rows.append((uid, unit_score))
                prev = best_parent.get(uid)
                if (prev is None) or (unit_score > prev[1]):
                    best_parent[uid] = (p, unit_score)
    
            # keep only units best assigned to this parent
            parent_scores[p] = [(u, sc) for (u, sc) in unit_rows if best_parent.get(u, (None, -1e9))[0] == p]
    
        # ---- quotas across parents (equal or proportional) ----
        n_avail = sum(len(v) for v in parent_scores.values())
        n_enriched = min(n_enr, n_avail)
        quotas: dict[str, int] = {}
        if equalize:
            base = (n_enriched // max(1, len(parents)))
            quotas = {p: base for p in parents}
            rem = n_enriched - base * len(parents)
            if rem > 0:
                order = sorted(parents, key=lambda L: len(parent_scores.get(L, [])), reverse=True)
                for p in order[:rem]:
                    quotas[p] = quotas.get(p, 0) + 1
        else:
            import math
            tot = max(1, sum(len(parent_scores.get(p, [])) for p in parents))
            raw = {p: (len(parent_scores.get(p, [])) * n_enriched) / tot for p in parents}
            quotas = {p: int(math.floor(v)) for p, v in raw.items()}
            give = n_enriched - sum(quotas.values())
            if give > 0:
                frac = sorted(((p, raw[p] - quotas[p]) for p in parents), key=lambda x: x[1], reverse=True)
                for p, _ in frac[:give]:
                    quotas[p] += 1
    
        # ---- pick by CE score within each parent; merge with uniform; top up uniformly if needed ----
        chosen, used = [], set(uniform_units)
        for p in parents:
            pool = sorted(parent_scores.get(p, []), key=lambda t: t[1], reverse=True)
            need = int(quotas.get(p, 0))
            for (u, _sc) in pool:
                if u in used: 
                    continue
                chosen.append(u); used.add(u)
                if len(chosen) >= n_enriched or sum(1 for x in chosen if best_parent[x][0] == p) >= need:
                    break
    
        merged = (uniform_units + chosen)
        out, used = [], set()
        for u in merged:
            if u in used or u in ex:
                continue
            used.add(u); out.append(u)
            if len(out) >= n_total: break
        if len(out) < n_total:
            rest = [u for u in all_units if u not in used]
            if rest:
                extra = rng.choice(rest, size=min(n_total - len(out), len(rest)), replace=False).tolist()
                out.extend(extra)
        return out[:n_total]


    def sample_units_for_probe(self, exclude_units: Optional[set[str]] = None) -> List[str]:
        import numpy as np
        ex = set(exclude_units or [])
        n_total = int(getattr(self.cfg, "n_probe_units", 100) or 100)
        uids = [str(u) for u in sorted(self.repo.notes["unit_id"].unique().tolist()) if str(u) not in ex]
        if not uids:
            return []
        rng = np.random.default_rng()
        return rng.choice(uids, size=min(n_total, len(uids)), replace=False).tolist()


    def probe_units_label_tree(self, enrich: bool, label_types: dict[str,str], 
                               per_label_rules: dict[str,str], seed: Optional[int]=None,
                               exclude_units: Optional[set[str]] = None) -> pd.DataFrame:
        ex = set(exclude_units or [])
        if enrich:
            sample = self.sample_units_for_probe_enriched(per_label_rules, exclude_units=ex)
        else: sample = self.sample_units_for_probe(exclude_units=ex)
        rows = []
        _progress_every = float(self.cfg.progress_min_interval_s or 1.0)
        for uid in iter_with_bar(
                step="LLM probe",
                iterable=sample,
                total=len(sample),
                min_interval_s=_progress_every):
            rows.extend(self.label_family_for_unit(uid, label_types, per_label_rules))

        df = pd.DataFrame(rows)
        return df

# ------------------------------
def per_label_kcenter(pool_df: pd.DataFrame, pooler: LabelAwarePooler, retriever: RAGRetriever, rules_map: Dict[str,str],
                      per_label_quota: int, already_selected_vecs: Optional[np.ndarray]=None) -> pd.DataFrame:
    out = []
    for lid, grp in pool_df.groupby("label_id"):
        if grp.empty: continue
        k = min(per_label_quota, len(grp))
        vecs = [
            pooler.pooled_vector(
                r.unit_id,
                r.label_id,
                retriever,
                rules_map.get(r.label_id, ""),
                context_builder=self.context_builder,
            )
            for r in grp.itertuples(index=False)
        ]
        V = np.vstack(vecs)
        idxs = kcenter_select(V, k=k, seed_idx=None, preselected=already_selected_vecs)
        sel = grp.iloc[idxs].copy()
        sel = sel.reset_index(drop=True)
        sel["kcenter_rank_per_label"] = list(range(1, len(sel)+1))
        out.append(sel)
    return pd.concat(out, ignore_index=True) if out else pool_df.head(0)


def merge_with_global_kcenter(per_label_selected: pd.DataFrame, pool_df: pd.DataFrame,
                              pooler: LabelAwarePooler, retriever: RAGRetriever, rules_map: Dict[str,str],
                              target_n: int,
                              already_selected_vecs: Optional[np.ndarray]=None) -> pd.DataFrame:
    if per_label_selected.empty and pool_df.empty: return per_label_selected
    selected = per_label_selected.copy()
    cnt = Counter(selected["label_id"].tolist()) if not selected.empty else Counter()
    remaining_needed = max(0, target_n - len(selected))
    if remaining_needed == 0: return selected
    if not selected.empty:
        rem = pool_df.merge(selected[["unit_id","label_id"]], on=["unit_id","label_id"], how="left", indicator=True)
        rem = rem[rem["_merge"]=="left_only"].drop(columns=["_merge"])
    else:
        rem = pool_df.copy()
    if rem.empty: return selected

    vecs = [
        pooler.pooled_vector(
            r.unit_id,
            r.label_id,
            retriever,
            rules_map.get(r.label_id, ""),
            context_builder=context_builder,
        )
        for r in rem.itertuples(index=False)
    ]
    V = np.vstack(vecs)

    preV = None
    if already_selected_vecs is not None and already_selected_vecs.size:
        preV = already_selected_vecs
    if not selected.empty:
        Vsel = [
            pooler.pooled_vector(
                r.unit_id,
                r.label_id,
                retriever,
                rules_map.get(r.label_id, ""),
                context_builder=context_builder,
            )
            for r in selected.itertuples(index=False)
        ]
        Vsel = np.vstack(Vsel); preV = Vsel if preV is None else np.vstack([preV, Vsel])

    idx_order = kcenter_select(V, k=min(remaining_needed, len(rem)), seed_idx=None, preselected=preV)

    chosen_rows = []
    for rank_idx, i in enumerate(idx_order, start=1):
        row = rem.iloc[i]
        row = row.copy()
        row["kcenter_rank_global"] = rank_idx
        chosen_rows.append(row); cnt[row.label_id] += 1
        if len(chosen_rows) >= remaining_needed: break

    if not chosen_rows: return selected
    chosen_df = pd.DataFrame([r._asdict() if hasattr(r,"_asdict") else dict(r) for r in chosen_rows])
    return pd.concat([selected, chosen_df], ignore_index=True)


def direct_uncertainty_selection(
    probe_df: pd.DataFrame,
    n_unc: int,
    *,
    label_col: str = "label_id",
    uncertainty_col: str = "U",
    higher_is_more_uncertain: bool = True,   # e.g., U = entropy → higher = more uncertain
    select_most_certain: bool = False,       # flip to pick the most certain items
    add_selection_reason: bool = True,
) -> pd.DataFrame:
    """
    Fill purely by round-robin across labels:
      1) Sort within each label by the criterion (uncertainty or certainty).
      2) Build a FIFO queue per label.
      3) Pop one from each label in turn until n_unc is reached (or queues empty).

    Notes:
      - NaN uncertainty sinks to the end for the chosen ordering.
    """
    # Basic guards
    if probe_df is None or not isinstance(probe_df, pd.DataFrame) or probe_df.empty or n_unc <= 0:
        return (probe_df.head(0).copy()
                if isinstance(probe_df, pd.DataFrame) else pd.DataFrame())

    df = probe_df.copy()
    if label_col not in df.columns or uncertainty_col not in df.columns:
        raise ValueError(f"probe_df must contain '{label_col}' and '{uncertainty_col}'")

    # Normalize uncertainty column and handle NaNs so they sink.
    u = pd.to_numeric(df[uncertainty_col], errors="coerce")

    # asc_u=True means "smaller score is better".
    # We want:
    #   - Most-uncertain:  desc if higher_is_more_uncertain else asc
    #   - Most-certain:    asc  if higher_is_more_uncertain else desc
    asc_u = (select_most_certain == higher_is_more_uncertain)

    # Fill NaNs so they sort to the end given asc/desc choice
    u = u.fillna(np.inf if asc_u else -np.inf)

    # Tiny jitter for stable tie-breaking (optional)
    rng = np.random.default_rng()
    u = u + (rng.random(len(u)) * 1e-9)

    df["_score"] = u

    # Build per-label queues: each label sorted by the criterion
    # (We sort by label then by score so groupby preserves label order;
    #  optionally shuffle label order 
    per_label_sorted = df.sort_values([label_col, "_score"], ascending=[True, asc_u])
    label_groups = list(per_label_sorted.groupby(label_col, sort=False))

    # Convert to queues (row-index lists)
    queues = {lab: grp.index.to_list() for lab, grp in label_groups}
    labels_order = [lab for lab, _ in label_groups if len(queues.get(lab, [])) > 0]

    # Optional: shuffle label start order for fairness when a seed is provided
    rng.shuffle(labels_order)

    # Round-robin selection
    chosen_idx = []
    need = int(n_unc)

    while need > 0 and labels_order:
        progressed = False
        new_order = []
        for lab in labels_order:
            q = queues.get(lab, [])
            if q:
                chosen_idx.append(q.pop(0))
                need -= 1
                progressed = True
                if q:  # keep label in the rotation if it still has items
                    new_order.append(lab)
            if need <= 0:
                break
        labels_order = new_order if progressed else []

    if not chosen_idx:
        out = df.head(0).copy()
        if add_selection_reason and "selection_reason" not in out.columns:
            out["selection_reason"] = ("model_certain_direct" if select_most_certain
                                       else "model_uncertain_direct")
        return out

    chosen = df.loc[chosen_idx].copy()

    # Decorate
    reason = "model_certain_direct" if select_most_certain else "model_uncertain_direct"
    if add_selection_reason and "selection_reason" not in chosen.columns:
        chosen["selection_reason"] = reason

    prefix = "certain" if select_most_certain else "uncertain"
    # Per-label rank by the criterion (for inspection)
    chosen[f"{prefix}_rank_per_label"] = (
        chosen.sort_values([label_col, "_score"], ascending=[True, asc_u])
              .groupby(label_col, sort=False).cumcount() + 1
    )
    # Global rank by the criterion
    chosen[f"{prefix}_rank_global"] = (
        chosen.sort_values("_score", ascending=asc_u)
              .reset_index(drop=True).index + 1
    )

    # Return ordered by the chosen criterion (you could return in selection order if preferred)
    out = (chosen.sort_values("_score", ascending=asc_u)
                 .drop(columns=["_score"])
                 .reset_index(drop=True))
    return out



