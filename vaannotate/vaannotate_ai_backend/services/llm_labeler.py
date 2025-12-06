"""LLM labeling service with prompt construction and call orchestration."""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter
from typing import Any, Iterable, List, Mapping, Optional

import numpy as np

from ..label_configs import LabelConfigBundle
from ..llm_backends import ForcedChoiceResult, JSONCallResult, build_llm_backend


class LLMRunRecorder:
    """Simple per-run recorder that writes JSON traces for LLM calls."""

    def __init__(self):
        self.calls = []
        self.run_meta = {}
        self.run_id = None
        self.outdir = None

    def start(self, outdir: str, run_id: Optional[str] = None, meta: Optional[dict] = None):
        import time as _time, os

        self.calls = []
        self.run_meta = meta or {}
        self.run_id = run_id or _time.strftime("%Y%m%d-%H%M%S")
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def record(self, kind: str, payload: dict):
        import time as _time

        self.calls.append({"ts": _time.time(), "kind": kind, **(payload or {})})

    def flush(self) -> Optional[str]:
        import os

        if not self.outdir:
            return None
        path = os.path.join(self.outdir, f"llm_calls_{self.run_id}.json")
        data = {
            "run_id": self.run_id,
            "meta": self.run_meta,
            "count": len(self.calls),
            "calls": self.calls,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path


LLM_RECORDER = LLMRunRecorder()


def _canon_str(x):
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _options_for_label(label_id: str, label_type: str, label_cfgs: dict) -> Optional[List[str]]:
    cfg = (label_cfgs or {}).get(label_id, {}) if isinstance(label_cfgs, dict) else {}
    if not isinstance(cfg, dict):
        return None
    opts = cfg.get("options")
    if not isinstance(opts, (list, tuple)):
        return None
    if str(label_type).lower() in {"categorical", "categorical_single", "categorical_multi", "ordinal", "binary", "boolean"}:
        return [str(x) for x in opts]
    return None


class LLMLabeler:
    """Builds prompts, calls the backend, and normalizes LLM label outputs."""

    def __init__(
        self,
        llm_backend,
        label_config_bundle: LabelConfigBundle,
        llm_config,
        sc_cfg=None,
        cache_dir: str | None = None,
    ):
        self.cfg = llm_config
        self.scCfg = sc_cfg
        self.cache_dir = os.path.join(cache_dir or "", "llm_cache") if cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.backend = llm_backend or build_llm_backend(self.cfg)
        self.label_config: dict[str, object] = (label_config_bundle or LabelConfigBundle()).current or {}

    def _save(self, key: str, data: dict):
        try:
            LLM_RECORDER.record("aggregate", {"cache_key": key, "out": data})
        except Exception:
            pass

    @staticmethod
    def _norm_token(x) -> str:
        if x is None:
            return ""
        s = str(x)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _is_yes(value: Any) -> bool:
        if isinstance(value, bool):
            return bool(value)
        if value is None:
            return False
        s = _canon_str(value).lower()
        return s in {"yes", "y", "true", "1", "present", "selected", "include"}

    @staticmethod
    def _canon_multi_selection(prediction: Any, option_values: list[str]) -> str | None:
        opts = [str(o) for o in (option_values or [])]
        option_lookup = {_canon_str(o).lower(): o for o in opts}
        selected: list[str] = []

        def _add_if_selected(key: Any, val: Any = True) -> None:
            canon_key = _canon_str(key).lower()
            opt_value = option_lookup.get(canon_key)
            if not opt_value:
                return
            if not LLMLabeler._is_yes(val):
                return
            if opt_value not in selected:
                selected.append(opt_value)

        if isinstance(prediction, Mapping):
            for k, v in prediction.items():
                _add_if_selected(k, v)
        elif isinstance(prediction, (list, tuple, set)):
            for item in prediction:
                _add_if_selected(item)
        elif isinstance(prediction, str):
            parts = re.split(r"[,;\n]\s*", prediction)
            for p in parts:
                if p:
                    _add_if_selected(p)
        else:
            _add_if_selected(prediction)

        return ",".join(selected) if selected else None

    @staticmethod
    def _parse_float(value):
        try:
            if value is None:
                return None
            s = str(value).strip()
            s = re.sub(r"[\s,]", "", s)
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _parse_date(value):
        from datetime import datetime, date

        if value is None:
            return None
        s = str(value).strip()
        fmts = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d %b %Y", "%b %d %Y"]
        for f in fmts:
            try:
                return datetime.strptime(s, f).date()
            except Exception:
                continue
        m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
        if m:
            try:
                y, mo, da = map(int, m.groups())
                return date(y, mo, da)
            except Exception:
                return None
        return None

    @staticmethod
    def _is_unknown_token(s: str) -> bool:
        if s is None:
            return True
        t = str(s).strip().lower()
        return t in {"", "na", "n/a", "none", "null", "unknown", "not evaluable", "not_evaluable", "absent", "missing"}

    @staticmethod
    def _pairwise_agree_numeric(preds: List, abs_scale: float = 1.0, rel_scale: float = 0.05, w_uu: float = 0.5, w_uk: float = 0.0) -> float:
        n = len(preds)
        if n <= 1:
            return 1.0
        vals = [LLMLabeler._parse_float(p) if not LLMLabeler._is_unknown_token(p) else None for p in preds]
        known_vals = [v for v in vals if v is not None]
        if known_vals:
            med = float(np.median(known_vals))
            tau = max(abs_scale, rel_scale * (abs(med) if med != 0 else 1.0))
        else:
            tau = abs_scale
        sim_sum, pairs = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                ui = vals[i] is None
                uj = vals[j] is None
                if ui and uj:
                    s = w_uu
                elif ui != uj:
                    s = w_uk
                else:
                    s = math.exp(-abs(vals[i] - vals[j]) / tau)
                sim_sum += s
                pairs += 1
        return float(sim_sum / max(1, pairs))

    @staticmethod
    def _pairwise_agree_date(preds: List, tau_days: int = 14, w_uu: float = 0.5, w_uk: float = 0.0) -> float:
        n = len(preds)
        if n <= 1:
            return 1.0
        ords = []
        for p in preds:
            if LLMLabeler._is_unknown_token(p):
                ords.append(None)
            else:
                d = LLMLabeler._parse_date(p)
                ords.append(d.toordinal() if d is not None else None)
        sim_sum, pairs = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                ui = ords[i] is None
                uj = ords[j] is None
                if ui and uj:
                    s = w_uu
                elif ui != uj:
                    s = w_uk
                else:
                    s = math.exp(-abs(ords[i] - ords[j]) / float(tau_days))
                sim_sum += s
                pairs += 1
        return float(sim_sum / max(1, pairs))

    def _few_shot_messages(self, label_id: str) -> list[dict[str, str]]:
        examples_cfg = getattr(self.cfg, "few_shot_examples", {}) or {}
        if not isinstance(examples_cfg, Mapping):
            return []
        label_key = str(label_id)
        label_examples = examples_cfg.get(label_key) or examples_cfg.get(label_key.lower())
        if not isinstance(label_examples, (list, tuple)):
            return []
        messages: list[dict[str, str]] = []
        for entry in label_examples:
            if not isinstance(entry, Mapping):
                continue
            answer = entry.get("answer")
            context = entry.get("context")
            if context is not None:
                ctx_text = str(context)
                if ctx_text.strip():
                    ctx_text = f"EHR context:\n{ctx_text}"
                messages.append({"role": "user", "content": ctx_text})
            if answer is not None:
                messages.append({"role": "assistant", "content": str(answer)})
        return messages

    def summarize_label_rule_for_rerank(self, label_id: str, label_rules: str, max_sentences: int = 2) -> str:
        if not getattr(self, "backend", None):
            return ""
        system_msg = (
            "You condense clinical labeling guidelines for a cross-encoder reranker. "
            "Return a succinct 1-2 sentence summary that preserves the key inclusion/"
            "exclusion cues. Avoid meta commentary."
        )
        user_msg = (
            "Label ID: {label_id}\n"
            "Rewrite the following label rule into at most {limit} sentences capturing the essence:\n{rules}"
        ).format(label_id=label_id, limit=max_sentences, rules=label_rules)
        result = self.backend.json_call(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            logprobs=False,
            top_logprobs=None,
        )
        text = re.sub(r"\s+", " ", str(getattr(result, "content", "") or "")).strip()
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        trimmed = " ".join(sentences[:max_sentences]).strip()
        return trimmed or text

    def annotate(
        self,
        unit_id: str,
        label_id: str,
        label_type: str,
        label_rules: str,
        snippets: List[dict],
        n_consistency: int = 1,
        jitter_params: bool = False,
        rag_diagnostics: Optional[dict] = None,
    ) -> dict:
        rag_topk_range = getattr(self.scCfg, "rag_topk_range", (6, 6))
        rag_dropout_p = getattr(self.scCfg, "rag_dropout_p", 0.0)
        temp_range = getattr(self.scCfg, "temperature_range", (self.cfg.temperature, self.cfg.temperature))
        shuffle_context = getattr(self.scCfg, "shuffle_context", False)
        context_order = getattr(self.cfg, "context_order", "relevance") or "relevance"

        def _ordered_snippets(items: List[dict]) -> List[dict]:
            if str(context_order).lower() != "chronological":
                return list(items)
            sortable: list[tuple[str, int, dict]] = []
            for idx, snip in enumerate(items):
                meta = snip.get("metadata") if isinstance(snip, Mapping) else {}
                date_val = ""
                if isinstance(meta, Mapping):
                    date_val = str(meta.get("date") or "")
                sortable.append((date_val, idx, snip))
            sortable.sort(key=lambda t: (t[0], t[1]))
            return [entry[2] for entry in sortable]

        snippets = _ordered_snippets(snippets)

        rng = random.Random()
        include_reasoning = bool(getattr(self.cfg, "include_reasoning", True))

        def _build_context_text(_snips: List[dict]) -> str:
            ctx, used = [], 0
            budget = max(1000, getattr(self.cfg, "max_context_chars", 4000))
            for s in _snips:
                md = s.get("metadata") or {}
                hdr_bits = [f"doc_id={s.get('doc_id')}", f"chunk_id={s.get('chunk_id')}"]
                if md.get("date"):
                    hdr_bits.append(f"date={md['date']}")
                note_type = md.get("note_type") or md.get("notetype")
                if note_type:
                    hdr_bits.append(f"type={note_type}")
                header = "[" + ", ".join(hdr_bits) + "] "
                text_body = (s.get("text", "") or "")
                frag = header + text_body
                if used + len(frag) > budget:
                    break
                ctx.append(frag)
                used += len(frag)
            return "\n\n".join(ctx)

        preds, runs = [], []
        system_intro = "You are a meticulous clinical annotator for EHR data."

        for i in range(n_consistency):
            sc_meta = ""
            if jitter_params:
                kmin, kmax = rag_topk_range
                kmin = max(1, int(kmin or 1))
                kmax = max(kmin, int(kmax or kmin))
                k = min(len(snippets), rng.randint(kmin, kmax))
                drop_p = max(0.0, min(1.0, rag_dropout_p))
                cand = list(snippets[:k]) if k > 0 else list(snippets)
                if drop_p > 0.0:
                    cand = [s for s in cand if rng.random() > drop_p] or [snippets[0]]
                if shuffle_context and len(cand) > 1:
                    rng.shuffle(cand)
                t_lo, t_hi = temp_range
                t = rng.uniform(float(t_lo), float(t_hi))
                sc_meta = f"<!-- sc:vote={i};k={k};drop={drop_p:.2f};shuf={int(shuffle_context)};temp={t:.2f} -->"
                ctx_text = _build_context_text(cand)
                temperature_this_vote = t
            else:
                sc_meta = ""
                ctx_text = _build_context_text(snippets)
                temperature_this_vote = self.cfg.temperature

            opts = _options_for_label(label_id, label_type, self.label_config)
            lt_norm = (label_type or "").strip().lower()
            categorical_types = {
                "binary",
                "boolean",
                "categorical",
                "categorical_single",
                "categorical_multi",
                "ordinal",
            }
            use_options = bool(opts) and lt_norm in categorical_types
            option_values = [str(opt) for opt in (opts or [])]
            is_multi_select = lt_norm == "categorical_multi"
            unknown_option_configured = any(_canon_str(o).lower() == "unknown" for o in option_values)

            guideline_text = label_rules if label_rules else "(no additional guidelines)"
            response_keys = "reasoning, prediction" if include_reasoning else "prediction"

            system_segments: list[str] = [
                system_intro,
                f"Your task: label '{label_id}' (type: {label_type}). Use the evidence snippets from this patient's notes.",
                f"Label rules:\n{guideline_text}",
            ]
            if use_options:
                if is_multi_select:
                    system_segments.append("Select every option that is supported by the evidence from the list below.")
                    system_segments.append("Options:")
                    system_segments.extend(f"- {opt}" for opt in option_values)
                    deny_unknown = " Do NOT answer 'unknown' unless it appears in the options above." if not unknown_option_configured else ""
                    system_segments.append(
                        "Return a compact JSON object where each included option key is set to 'Yes' if it applies (omit or set to 'No' when it does not)." + deny_unknown
                    )
                else:
                    system_segments.append("Choose the single best option from the list below based on the evidence.")
                    system_segments.append("Options:")
                    system_segments.extend(f"- {opt}" for opt in option_values)
                    system_segments.append(
                        f"Set prediction to exactly one of: {', '.join(option_values)}. Do not invent new options."
                    )
            else:
                system_segments.append("If insufficient evidence, reply with 'unknown'.")

            if include_reasoning:
                system_segments.append(
                    "Think step-by-step citing specific evidence, and keep the reasoning concise."
                )
            system_segments.append(f"Return strict JSON only with keys: {response_keys}.")
            system_segments.append("No additional keys or text.")

            system_body = "\n\n".join(system_segments)
            system = system_body + ("\n" + sc_meta if sc_meta else "")

            few_shot_msgs = self._few_shot_messages(label_id)
            messages = ([{"role": "system", "content": system}] + few_shot_msgs + [{"role": "user", "content": ctx_text}])
            schema = {"type": "object", "properties": {"prediction": {"type": "string"}}, "required": ["prediction"], "additionalProperties": include_reasoning}
            if include_reasoning:
                schema["properties"]["reasoning"] = {"type": "string"}
            try:
                result = self.backend.json_call(
                    messages,
                    temperature=temperature_this_vote,
                    logprobs=bool(self.cfg.logprobs),
                    top_logprobs=int(self.cfg.top_logprobs) if int(self.cfg.top_logprobs) > 0 else None,
                    response_format={"type": "json_object", "json_schema": schema},
                    timeout=self.cfg.timeout,
                    retry_max=self.cfg.retry_max,
                    retry_backoff=self.cfg.retry_backoff,
                )
                content = result.content
                LLM_RECORDER.record(
                    "json_vote",
                    {
                        "unit_id": unit_id,
                        "label_id": label_id,
                        "label_type": label_type,
                        "prompt": {"system": system, "messages": few_shot_msgs, "user": ctx_text},
                        "params": {"temperature": temperature_this_vote},
                        "output": content,
                        "rag_diagnostics": rag_diagnostics or {},
                    },
                )
            except Exception:
                LLM_RECORDER.record("json_vote_error", {"rag_diagnostics": rag_diagnostics or {}})
                continue

            if isinstance(content, str):
                try:
                    obj = json.loads(content)
                except Exception:
                    obj = None
            else:
                obj = content if isinstance(content, Mapping) else None

            pred_raw = (obj or {}).get("prediction")
            reasoning = (obj or {}).get("reasoning") if include_reasoning else None
            pred_norm = None
            if use_options:
                if is_multi_select:
                    pred_norm = self._canon_multi_selection(pred_raw, option_values)
                    if pred_norm is None and isinstance(obj, Mapping):
                        pred_norm = self._canon_multi_selection(obj, option_values)
                else:
                    lut = {_canon_str(o).lower(): o for o in option_values}
                    pred_norm = lut.get(_canon_str(pred_raw).lower())
            if pred_norm is None:
                pred_norm = self._norm_token(pred_raw)
            raw_prediction = pred_raw if pred_raw is not None else pred_norm
            preds.append(pred_norm)
            runs.append({
                "prediction": pred_norm,
                "raw_prediction": raw_prediction,
                "prob": getattr(result, "top_logprobs", None),
                "logprobs": getattr(result, "logprobs", None),
                "reasoning": reasoning,
            })

        if not preds:
            return {"unit_id": unit_id, "label_id": label_id, "label_type": label_type, "prediction": None, "consistency_agreement": None, "runs": runs}

        pred_final = None
        cons = 1.0
        lt = (label_type or "").strip().lower()
        if lt in {"numeric", "number", "int", "integer", "float", "double"}:
            vals = [self._parse_float(p) for p in preds if not self._is_unknown_token(p)]
            if vals:
                pred_final = float(np.median([v for v in vals if v is not None]))
            cons = self._pairwise_agree_numeric(preds, abs_scale=1.0, rel_scale=0.05, w_uu=0.5, w_uk=0.1)
        elif lt in {"date", "datetime", "timestamp"}:
            vals = [self._parse_date(p) for p in preds if not self._is_unknown_token(p)]
            if vals:
                med = np.median([v.toordinal() for v in vals if v is not None])
                try:
                    from datetime import date

                    pred_final = date.fromordinal(int(med))
                except Exception:
                    pred_final = None
            cons = self._pairwise_agree_date(preds, tau_days=14, w_uu=0.5, w_uk=0.1)
        elif lt == "categorical_multi":
            cnt = Counter([p for p in preds if p is not None])
            if cnt:
                pred_final = max(cnt.items(), key=lambda kv: kv[1])[0]
                cons = cnt[pred_final] / max(1, len([p for p in preds if p is not None]))
        else:
            cnt = Counter([p for p in preds if p is not None])
            if cnt:
                pred_final = max(cnt.items(), key=lambda kv: kv[1])[0]
                cons = cnt[pred_final] / max(1, len([p for p in preds if p is not None]))

        out = {
            "unit_id": unit_id,
            "label_id": label_id,
            "label_type": label_type,
            "prediction": pred_final,
            "consistency_agreement": float(cons),
            "runs": runs,
        }
        return out

    def forced_choice_probe(
        self,
        unit_id: str,
        label_id: str,
        label_type: str,
        label_rules: str,
        options: list[str],
        context_builder,
        retriever,
        llmfirst_cfg,
    ) -> dict:
        letters = [chr(ord("A") + i) for i in range(len(options))]
        option_lines = [f"{letters[i]}. {options[i]}" for i in range(len(options))]
        system = "You are a careful clinical information extraction assistant."
        snippets = context_builder.build_context_for_label(
            unit_id,
            label_id,
            label_rules,
            topk_override=llmfirst_cfg.topk,
            single_doc_context_mode=getattr(llmfirst_cfg, "single_doc_context", "rag"),
            full_doc_char_limit=getattr(llmfirst_cfg, "single_doc_full_context_max_chars", None),
        )
        rag_diag = retriever.get_last_diagnostics(unit_id, label_id)
        ctx_lines = []
        for snip in snippets:
            md = snip.get("metadata") or {}
            hdr_bits = [f"doc_id={snip.get('doc_id')}", f"chunk_id={snip.get('chunk_id')}"]
            if md.get("date"):
                hdr_bits.append(f"date={md['date']}")
            note_type = md.get("note_type") or md.get("notetype")
            if note_type:
                hdr_bits.append(f"type={note_type}")
            header = "[" + ", ".join(hdr_bits) + "] "
            ctx_lines.append(header + (snip.get("text", "") or ""))
        ctx = "\n\n".join(ctx_lines)
        user = (
            f"Task: Choose the single best option for label '{label_id}' given the context snippets.\n"
            + (f"Label rules/hints: {label_rules}\n" if label_rules else "")
            + "Options:\n"
            + "\n".join(option_lines)
            + "\n"
            + "Return ONLY the option letter.\n\n"
            + "Context:\n"
            + ctx
        )
        result = self.backend.forced_choice(
            system=system,
            user=user,
            options=options,
            letters=letters,
            top_logprobs=int(self.cfg.top_logprobs) if int(self.cfg.top_logprobs) > 0 else 5,
        )
        opt_probs = dict(result.option_probs)
        ent = float(result.entropy)
        pred = result.prediction

        try:
            LLM_RECORDER.record(
                "forced_choice",
                {
                    "unit_id": unit_id,
                    "label_id": label_id,
                    "label_type": label_type,
                    "prompt": {"system": system, "user": user},
                    "snippets": ctx,
                    "fc_output": {
                        "fc_probs": opt_probs,
                        "fc_entropy": ent,
                        "prediction": pred,
                        "latency_s": result.latency_s,
                    },
                    "rag_diagnostics": rag_diag or {},
                    "rag_queries": {
                        "manual_query": rag_diag.get("manual_query") if isinstance(rag_diag, Mapping) else None,
                        "query_source": rag_diag.get("query_source") if isinstance(rag_diag, Mapping) else None,
                        "queries": rag_diag.get("queries") if isinstance(rag_diag, Mapping) else None,
                        "query_sources": rag_diag.get("query_sources") if isinstance(rag_diag, Mapping) else None,
                    },
                },
            )
        except Exception:
            pass

        return {"fc_probs": opt_probs, "fc_entropy": ent, "prediction": pred, "rag_context": snippets}

    def label_unit(
        self,
        unit_id: str,
        label_ids: list[str],
        *,
        label_types: Mapping[str, str],
        per_label_rules: Mapping[str, str],
        context_builder,
        retriever,
        llmfirst_cfg,
        json_only: bool = False,
        json_n_consistency: int = 1,
        json_jitter: bool = False,
    ) -> list[dict]:
        results: list[dict] = []
        for lid in label_ids:
            ltype = label_types.get(lid, "categorical") if isinstance(label_types, Mapping) else "categorical"
            rules = per_label_rules.get(lid, "") if isinstance(per_label_rules, Mapping) else ""
            opts = _options_for_label(lid, ltype, getattr(retriever, "label_configs", {}))
            row = {"unit_id": unit_id, "label_id": lid, "label_type": ltype}
            try:
                rag_context = context_builder.build_context_for_label(
                    unit_id,
                    lid,
                    rules,
                    topk_override=llmfirst_cfg.topk,
                    single_doc_context_mode=getattr(llmfirst_cfg, "single_doc_context", "rag"),
                    full_doc_char_limit=getattr(llmfirst_cfg, "single_doc_full_context_max_chars", None),
                )
            except Exception:
                rag_context = []
            rag_diag = retriever.get_last_diagnostics(unit_id, lid)
            row["rag_context"] = rag_context

            if json_only:
                res = self.annotate(
                    unit_id,
                    lid,
                    ltype,
                    rules,
                    rag_context,
                    n_consistency=max(1, int(json_n_consistency)),
                    jitter_params=bool(json_jitter),
                    rag_diagnostics=rag_diag,
                )
                row["prediction"] = res.get("prediction")
                row["consistency"] = res.get("consistency_agreement")
                row["runs"] = res.get("runs")
                row["U"] = (1.0 - float(row["consistency"])) if row.get("consistency") is not None else None
            else:
                used_fc = False
                if ltype in ("categorical", "binary") or (opts is not None and getattr(llmfirst_cfg, "fc_enable", False)):
                    if opts is not None:
                        try:
                            fc = self.forced_choice_probe(
                                unit_id,
                                lid,
                                ltype,
                                rules,
                                opts,
                                context_builder,
                                retriever,
                                llmfirst_cfg,
                            )
                            row.update(fc)
                            used_fc = True
                            row["U"] = float(fc.get("fc_entropy", np.nan))
                        except Exception:
                            used_fc = False
                if not used_fc:
                    res = self.annotate(
                        unit_id,
                        lid,
                        ltype,
                        rules,
                        rag_context,
                        n_consistency=getattr(self.cfg, "n_consistency", 1),
                        jitter_params=True,
                        rag_diagnostics=rag_diag,
                    )
                    row["prediction"] = res.get("prediction")
                    row["consistency"] = res.get("consistency_agreement")
                    row["runs"] = res.get("runs")
                    try:
                        row["U"] = float(1.0 - float(res.get("consistency_agreement") or 0.0))
                    except Exception:
                        row["U"] = np.nan
            results.append(row)

        return results


__all__ = ["LLMLabeler", "LLM_RECORDER"]
