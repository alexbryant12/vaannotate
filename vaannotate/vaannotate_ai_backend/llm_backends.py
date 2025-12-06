"""Pluggable backends for active-learning LLM calls.

This module provides a light abstraction around the two inference backends we
currently support:

* Azure OpenAI Chat Completions (the historical default)
* Local ExLlamaV2 inference with tolerant JSON parsing

Backends expose only the operations required by the active-learning engine:

``json_call``
    Execute a JSON-oriented chat completion and return the parsed payload plus
    any ancillary metadata (raw text, latency, logprobs when available).

``forced_choice``
    Run a 1-token forced-choice probe and return per-option probabilities and
    entropy metrics.

The intent is to keep the orchestration layer agnostic to the underlying
inference engine so that switching between Azure-hosted and local inference is
purely a configuration concern.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - avoid runtime circular import
    from .config import LLMConfig

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - handled when backend is constructed
    AzureOpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
except Exception:  # pragma: no cover - handled during backend construction
    ExLlamaV2 = None  # type: ignore
    ExLlamaV2Cache = None  # type: ignore
    ExLlamaV2Config = None  # type: ignore
    ExLlamaV2Tokenizer = None  # type: ignore
    ExLlamaV2DynamicGenerator = None  # type: ignore
    ExLlamaV2Sampler = None  # type: ignore

def _to_serializable(value: Any) -> Any:
    """Convert SDK response objects into plain Python structures."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, MappingABC):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_serializable(v) for v in value]

    for attr in ("model_dump", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):  # pragma: no branch - best effort serialisation
            try:
                return _to_serializable(method())
            except Exception:  # noqa: BLE001 - fallback to next strategy
                pass

    json_method = getattr(value, "model_dump_json", None)
    if callable(json_method):
        try:
            return json.loads(json_method())
        except Exception:  # noqa: BLE001 - fallback to repr
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _to_serializable(v)
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
        except Exception:  # noqa: BLE001 - fallback to repr
            pass

    return str(value)


def _reasoning_first(obj: Mapping[str, Any] | Any) -> Mapping[str, Any] | Any:
    """Return a shallow copy with the reasoning key (if present) first."""

    if not isinstance(obj, MappingABC):
        return obj

    if "reasoning" not in obj:
        return obj

    ordered: Dict[str, Any] = {"reasoning": obj.get("reasoning")}
    for key, value in obj.items():
        if key == "reasoning":
            continue
        ordered[key] = value
    return ordered


@dataclass
class JSONCallResult:
    """Structured result returned by :meth:`LLMBackend.json_call`."""

    data: Mapping[str, Any]
    content: str
    raw_response: Any
    latency_s: float
    logprobs: Any | None = None


@dataclass
class ForcedChoiceResult:
    """Structured result returned by :meth:`LLMBackend.forced_choice`."""

    option_probs: Mapping[str, float]
    option_logprobs: Mapping[str, float]
    prediction: str
    entropy: float
    latency_s: float
    raw_response: Any | None = None


class LLMBackend:
    """Base class shared by all inference backends."""

    def __init__(self, cfg: "LLMConfig"):
        self.cfg = cfg
        self._last_call_ts: float = 0.0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _respect_rpm_limit(self) -> None:
        rpm = getattr(self.cfg, "rpm_limit", None)
        if not rpm:
            return
        min_spacing = float(60.0 / float(rpm))
        now = time.time()
        delta = now - self._last_call_ts
        if delta < min_spacing:
            time.sleep(min_spacing - delta)

    def _post_call(self) -> None:
        self._last_call_ts = time.time()

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    def json_call(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        logprobs: bool,
        top_logprobs: Optional[int],
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> JSONCallResult:
        raise NotImplementedError

    def forced_choice(
        self,
        system: str,
        user: str,
        *,
        options: Sequence[str],
        letters: Sequence[str],
        top_logprobs: int = 5,
    ) -> ForcedChoiceResult:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - hook for specialised backends
        """Allow backends with external resources to provide an explicit close."""
        return None


class AzureOpenAIBackend(LLMBackend):
    """Backend that wraps Azure OpenAI chat completions."""

    def __init__(self, cfg: LLMConfig):
        if AzureOpenAI is None:  # pragma: no cover - runtime guard
            raise ImportError("Please install openai>=1.0 to use the Azure backend.")
        super().__init__(cfg)
        api_key = cfg.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = cfg.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        endpoint = cfg.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise ValueError("Azure backend requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            timeout=cfg.timeout,
        )

    def json_call(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        logprobs: bool,
        top_logprobs: Optional[int],
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> JSONCallResult:
        self._respect_rpm_limit()
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model_name,
            "temperature": float(temperature),
            "messages": list(messages),
            "n": 1,
            "logprobs": bool(logprobs),
        }
        if response_format:
            # Azure currently rejects nested JSON schema fields ("response_format.json_schema")
            # with a 400 error. We therefore strip the schema detail and rely on the
            # model's default JSON mode behaviour instead.
            if "json_schema" in response_format:
                rf_clean = dict(response_format)
                rf_clean.pop("json_schema", None)
                kwargs["response_format"] = rf_clean
            else:
                kwargs["response_format"] = response_format
        if logprobs and top_logprobs:
            kwargs["top_logprobs"] = int(top_logprobs)
        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.time() - t0
        self._post_call()
        choice = resp.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message else None
        parsed = getattr(message, "parsed", None) if message else None

        # Azure JSON mode may populate `message.parsed` instead of `content`
        if parsed is not None:
            data_candidate = parsed
            if not content:
                try:
                    content = json.dumps(parsed)
                except Exception:
                    content = str(parsed)
        else:
            if content is None:
                content = getattr(choice, "content", "")
            content = content or ""
            data_candidate = content
            if content:
                try:
                    data_candidate = json.loads(content)
                except Exception as exc:
                    if response_format:
                        snippet = content[:2000]
                        raise ValueError(f"Failed to parse JSON response: {snippet}") from exc

        if response_format and not isinstance(data_candidate, MappingABC):
            raise ValueError(f"Expected JSON object but received: {str(data_candidate)[:2000]}")

        data = _reasoning_first(data_candidate)
        logprob_info = getattr(choice, "logprobs", None)
        if logprob_info is not None:
            logprob_info = _to_serializable(logprob_info)
        return JSONCallResult(
            data=data,
            content=content,
            raw_response=resp,
            latency_s=float(latency),
            logprobs=logprob_info,
        )

    def forced_choice(
        self,
        system: str,
        user: str,
        *,
        options: Sequence[str],
        letters: Sequence[str],
        top_logprobs: int = 5,
    ) -> ForcedChoiceResult:
        self._respect_rpm_limit()
        kwargs = dict(
            model=self.cfg.model_name,
            temperature=0.0,
            logprobs=True,
            top_logprobs=int(top_logprobs),
            max_tokens=1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.time() - t0
        self._post_call()
        choice = resp.choices[0]
        logprobs_obj = getattr(choice, "logprobs", None)
        items = getattr(logprobs_obj, "content", None) or []
        letter_logps = {letter: -1e9 for letter in letters}
        for item in items:
            tops = getattr(item, "top_logprobs", None)
            if tops is None and isinstance(item, Mapping):
                tops = item.get("top_logprobs")
            tops = tops or []
            for cand in tops:
                token = getattr(cand, "token", None)
                if token is None and isinstance(cand, Mapping):
                    token = cand.get("token")
                logprob = getattr(cand, "logprob", None)
                if logprob is None and isinstance(cand, Mapping):
                    logprob = cand.get("logprob")
                if token is None or logprob is None:
                    continue
                token_text = str(token).strip().strip('"').strip("'")
                if token_text and not token_text[0].isalnum():
                    token_text = token_text[1:]
                key = token_text[:1].upper() if token_text else ""
                if key in letter_logps:
                    letter_logps[key] = max(letter_logps[key], float(logprob))
        logits = [letter_logps[letter] for letter in letters]
        m = max(logits)
        probs = [math.exp(x - m) for x in logits]
        denom = sum(probs)
        if denom <= 0:
            probs = [1.0 / len(probs) for _ in probs]
        else:
            probs = [p / denom for p in probs]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        option_probs = {options[i]: float(probs[i]) for i in range(len(options))}
        option_logps = {options[i]: float(letter_logps[letters[i]]) for i in range(len(options))}
        pred_idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
        prediction = options[pred_idx]
        return ForcedChoiceResult(
            option_probs=option_probs,
            option_logprobs=option_logps,
            prediction=prediction,
            entropy=float(entropy),
            latency_s=float(latency),
            raw_response=resp,
        )


class ExLlamaV2Backend(LLMBackend):  # pragma: no cover - requires heavy optional deps
    """Backend that runs inference locally via ExLlamaV2."""

    def __init__(self, cfg: LLMConfig):
        if ExLlamaV2 is None or ExLlamaV2Config is None or ExLlamaV2Tokenizer is None:
            raise ImportError(
                "ExLlamaV2 backend requested but exllamav2 is not installed."
            )
        super().__init__(cfg)
        model_dir = cfg.local_model_dir or os.getenv("LOCAL_LLM_MODEL_DIR")
        if not model_dir:
            raise ValueError("ExLlamaV2 backend requires local_model_dir to be set.")

        config = ExLlamaV2Config()
        config.model_dir = str(model_dir)
        if cfg.local_max_seq_len:
            config.max_seq_len = int(cfg.local_max_seq_len)
        if hasattr(config, "no_flash_attn"):
            setattr(config, "no_flash_attn", True)
        elif hasattr(config, "use_flash_attn"):
            setattr(config, "use_flash_attn", False)
        elif hasattr(config, "flash_attn"):
            setattr(config, "flash_attn", False)
        config.prepare()

        self.model = ExLlamaV2(config)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        max_seq = int(getattr(cfg, "local_max_seq_len", 0) or config.max_seq_len)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=max_seq, lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        if ExLlamaV2DynamicGenerator is None:
            raise ImportError(
                "ExLlamaV2DynamicGenerator is unavailable even though exllamav2 was imported."
            )
        try:
            # Prefer keyword arguments to match current releases.
            self.generator = ExLlamaV2DynamicGenerator(
                model=self.model,
                cache=self.cache,
                tokenizer=self.tokenizer,
                paged=False,
            )
        except TypeError:  # pragma: no cover - signature drift across releases
            try:
                self.generator = ExLlamaV2DynamicGenerator(self.model, self.cache, self.tokenizer)
            except TypeError:
                self.generator = ExLlamaV2DynamicGenerator(self.model, self.tokenizer, self.cache)

        warmup = getattr(self.generator, "warmup", None)
        if callable(warmup):  # pragma: no cover - optional optimisation
            warmup()

        self._json_stop_conditions: Sequence[int] = tuple(
            getattr(cfg, "local_json_stop_tokens", None) or (128001, 128009, 78191)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _format_messages(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Render messages using the Meta Llama 3.x chat template."""

        def _role(role_name: str) -> str:
            normalized = role_name.lower()
            if normalized not in {"system", "user", "assistant", "tool"}:
                return "user"
            return normalized

        parts: List[str] = ["<|begin_of_text|>"]
        for msg in messages:
            role = _role(msg.get("role", "user"))
            content = msg.get("content", "")
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n{content.strip()}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        return "".join(parts)

    def _tolerant_json_loads(self, text: str) -> Mapping[str, Any]:
        """
        Best-effort JSON parsing for locally generated completions.

        The ExLlamaV2 generator is not guaranteed to emit perfectly formatted
        JSON. This helper tries multiple repair strategies before falling back
        to regex-based extraction of key fields like ``prediction`` and
        ``reasoning``.
        """

        def _strip_code_fences(payload: str) -> str:
            stripped = payload.strip()
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
            return stripped.strip()

        def _balance_pairs(payload: str, open_char: str, close_char: str) -> str:
            opens = payload.count(open_char)
            closes = payload.count(close_char)
            if opens > closes:
                payload = payload + (close_char * (opens - closes))
            return payload

        cleaned = _strip_code_fences(text)
        candidates: List[str] = []
        if cleaned:
            candidates.append(cleaned)

        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            candidates.append(cleaned[brace_start : brace_end + 1])

        balanced_candidates: List[str] = []
        for cand in candidates:
            balanced_candidates.append(cand)
            balanced = _balance_pairs(cand, "{", "}")
            if balanced != cand:
                balanced_candidates.append(balanced)

        decoder = json.JSONDecoder()
        for cand in balanced_candidates:
            try:
                obj, _ = decoder.raw_decode(cand)
                return obj
            except Exception:
                continue

        salvage: Dict[str, Any] = {}

        def _capture_scalar(key: str) -> None:
            pattern = re.compile(
                rf"\b{re.escape(key)}\b\s*[:=]\s*(?:'([^']*)'|\"([^\"]*)\"|([^,}}\n]+))",
                re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(cleaned)
            if match:
                for group in match.groups():
                    if group is not None:
                        salvage[key] = group.strip()
                        break

        for key in ("reasoning", "prediction"):
            _capture_scalar(key)

        snippet_match = re.search(
            r"\b(snippets)\b\s*[:=]\s*(\[[\s\S]*?)(\]|$)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if snippet_match:
            snippets_text = snippet_match.group(2)
            snippets_balanced = _balance_pairs(snippets_text, "[", "]")
            try:
                salvage["snippets"] = json.loads(snippets_balanced)
            except Exception:
                # best-effort: split by line/item delimiters
                rough_items = re.split(r"\s*[,\n]\s*", snippets_balanced.strip("[] \n\t"))
                salvage["snippets"] = [item for item in (s.strip() for s in rough_items) if item]

        if salvage:
            return salvage

        raise ValueError(f"Local backend produced invalid JSON: {text}")

    def _generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        filters: Optional[Sequence[Any]] = None,
        stop_conditions: Optional[Sequence[int]] = None,
        logits_processors: Optional[Sequence[Any]] = None,
    ) -> tuple[str, List[int], List[float]]:
        if ExLlamaV2Sampler is None:
            raise ImportError(
                "ExLlamaV2Sampler is unavailable even though exllamav2 was imported."
            )
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = max(0.0, float(temperature))
        settings.top_p = 1.0
        settings.top_k = 0
        settings.token_repetition_penalty = 1.0
        settings.max_new_tokens = int(max_new_tokens)
        if filters:
            print('filters applied:')
            print(filters)
            settings.filters = filters
            settings.filter_prefer_eos = True

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "gen_settings": settings,
            "add_bos": False,
            "add_eos": False,
            "completion_only": True,
            "encode_special_tokens": True,
        }
        if stop_conditions:
            kwargs["stop_conditions"] = list(stop_conditions)

        generator = self.generator
        clear_logits = getattr(generator, "clear_logits_processors", None)
        remove_logits = getattr(generator, "remove_logits_processor", None)
        add_logits = getattr(generator, "add_logits_processor", None)

        if logits_processors and callable(clear_logits):  # pragma: no cover - optional API
            clear_logits()

        added_processors: List[Any] = []
        if logits_processors:
            if not callable(add_logits):  # pragma: no cover - defensive
                raise RuntimeError(
                    "ExLlamaV2DynamicGenerator does not support logits processors"
                )
            for processor in logits_processors:
                add_logits(processor)
                added_processors.append(processor)

        try:
            result = generator.generate(**kwargs)
        finally:  # pragma: no cover - cleanup path
            if logits_processors and callable(remove_logits):
                for processor in added_processors:
                    remove_logits(processor)
            elif logits_processors and callable(clear_logits):
                clear_logits()

        text: str
        token_ids: List[int]
        token_logprobs: List[float]

        if isinstance(result, tuple):
            # Older versions may return (text, token_ids, logprobs)
            if len(result) == 3:
                text = str(result[0])
                token_ids = list(result[1] or [])
                token_logprobs = list(result[2] or [])
            else:  # pragma: no cover - defensive
                text = str(result[0])
                token_ids = list(result[1] or []) if len(result) > 1 else []
                token_logprobs = list(result[2] or []) if len(result) > 2 else []
        elif isinstance(result, dict):
            text = str(result.get("text") or result.get("output") or "")
            token_ids = list(result.get("token_ids") or result.get("tokens") or [])
            token_logprobs = list(
                result.get("token_logprobs") or result.get("logprobs") or []
            )
        else:
            text = str(getattr(result, "text", result))
            token_ids = list(getattr(result, "token_ids", getattr(result, "tokens", [])))
            token_logprobs = list(
                getattr(result, "token_logprobs", getattr(result, "logprobs", []))
            )

        return text, [int(t) for t in token_ids], [float(lp) for lp in token_logprobs]

    # ------------------------------------------------------------------
    # Backend API implementation
    # ------------------------------------------------------------------
    def json_call(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        logprobs: bool,
        top_logprobs: Optional[int],
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> JSONCallResult:
        prompt = self._format_messages(messages)
        max_new = int(self.cfg.local_max_new_tokens or 1024)
        self._respect_rpm_limit()
        attempts = 0
        last_error: Exception | None = None
        text = ""
        token_ids: List[int] = []
        token_logprobs: List[float] = []
        data: Mapping[str, Any] | None = None
        latency = 0.0
        while attempts < 2:
            t0 = time.time()
            text, token_ids, token_logprobs = self._generate(
                prompt,
                max_new_tokens=max_new,
                temperature=temperature,
                stop_conditions=self._json_stop_conditions,
            )
            latency = time.time() - t0
            self._post_call()
            try:
                data = _reasoning_first(self._tolerant_json_loads(text))
                break
            except Exception as exc:
                last_error = exc
                attempts += 1
                if attempts >= 2:
                    raise
                continue
        else:  # pragma: no cover - defensive
            latency = time.time() - t0
            self._post_call()
            if last_error:
                raise last_error

        assert data is not None  # for type checkers; parsed JSON required

        logprob_payload = None
        if logprobs:
            logprob_payload = {
                "tokens": token_ids,
                "logprobs": token_logprobs,
            }
        return JSONCallResult(
            data=data,
            content=text,
            raw_response={"tokens": token_ids},
            latency_s=float(latency),
            logprobs=logprob_payload,
        )

    def forced_choice(
        self,
        system: str,
        user: str,
        *,
        options: Sequence[str],
        letters: Sequence[str],
        top_logprobs: int = 5,  # kept for interface symmetry; unused here
    ) -> ForcedChoiceResult:
        """
        Forced-choice micro-probe for local ExLlamaV2.

        Workaround for upstream ExLlamaV2 not exposing logits processors or
        returning token_logprobs:

        - Build the full chat prompt using the Llama 3.x template.
        - Encode it with the ExLlamaV2 tokenizer.
        - Run ExLlamaV2.forward manually to get logits for the *next* token.
        - Convert to log-softmax and read off the logprob for each answer letter.
        """

        import torch
        import torch.nn.functional as F  # type: ignore

        # ---- Build prompt via existing chat template -----------------------
        prompt = self._format_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

        self._respect_rpm_limit()
        t0 = time.time()

        # ---- Encode prompt -------------------------------------------------
        tokens = self.tokenizer.encode(
            prompt,
            add_bos=False,
            encode_special_tokens=True,
        )
        if hasattr(tokens, "dim") and tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # ---- Forward pass to get next-token logits ------------------------
        self.cache.current_seq_len = 0

        with torch.inference_mode():
            # Prime KV cache with all but the last token
            if tokens.size(1) > 1:
                _ = self.model.forward(
                    tokens[:, :-1],
                    self.cache,
                    input_mask=None,
                    preprocess_only=True,
                )

            logits = self.model.forward(
                tokens[:, -1:],  # last token
                self.cache,
                input_mask=None,
            ).float()

        # logits: [1, 1, vocab] or [1, vocab]
        if logits.dim() == 3:
            scores = logits[0, -1, :]
        else:
            scores = logits[0, :]

        log_probs = F.log_softmax(scores, dim=-1).reshape(-1)  # [vocab]

        # ---- Helpers to map letters -> token IDs --------------------------
        def _flatten_ids(enc) -> list[int]:
            """
            Robustly flatten encode output to a 1D list of ints.

            Handles:
            - torch tensors (1D or 2D)
            - list-of-lists ([[...]])
            """
            if hasattr(enc, "dim"):  # torch tensor
                if enc.dim() > 1:
                    enc = enc.view(-1)
                return [int(x) for x in enc.tolist()]

            ids = enc
            # Peel off nesting like [[1, 2, 3]] -> [1, 2, 3]
            while isinstance(ids, (list, tuple)) and ids and isinstance(
                ids[0], (list, tuple)
            ):
                ids = ids[0]
            return [int(x) for x in ids]

        def _best_letter_logprob(letter_text: str) -> float:
            """
            Best-effort log p(letter) by trying both 'A' and ' A' style encodings
            and taking the logprob of the *last* token.
            """
            best = float("-1e9")
            for variant in (letter_text, " " + letter_text):
                enc = self.tokenizer.encode(
                    variant,
                    add_bos=False,
                    encode_special_tokens=False,
                )
                ids = _flatten_ids(enc)
                if not ids:
                    continue
                tok_id = ids[-1]
                if 0 <= tok_id < log_probs.shape[0]:
                    lp = float(log_probs[tok_id].item())
                    if lp > best:
                        best = lp
            return best

        # ---- Letter logprobs and aggregate by option ----------------------
        letter_logps: Dict[str, float] = {}
        for letter in letters:
            letter_logps[letter] = _best_letter_logprob(letter)

        option_logps: Dict[str, float] = {}
        for opt, letter in zip(options, letters):
            option_logps[opt] = letter_logps.get(letter, float("-1e9"))

        logits_list = list(option_logps.values())
        if not logits_list:
            latency = time.time() - t0
            self._post_call()
            return ForcedChoiceResult(
                option_probs={},
                option_logprobs=option_logps,
                prediction="",
                entropy=0.0,
                latency_s=float(latency),
                raw_response=option_logps,
            )

        # ---- Softmax-normalise + entropy + prediction ---------------------
        m = max(logits_list)
        probs = [math.exp(v - m) for v in logits_list]
        denom = sum(probs) or 1.0
        probs = [p / denom for p in probs]

        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        option_probs = {opt: float(p) for opt, p in zip(options, probs)}
        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        prediction = options[pred_idx]

        latency = time.time() - t0
        self._post_call()

        print('FC call:')
        print(option_probs)
        print(prediction)
        return ForcedChoiceResult(
            option_probs=option_probs,
            option_logprobs=option_logps,
            prediction=prediction,
            entropy=float(entropy),
            latency_s=float(latency),
            raw_response=option_logps,
        )

      
def build_llm_backend(cfg: LLMConfig) -> LLMBackend:
    """Factory helper that instantiates the requested backend."""

    backend_name = (cfg.backend or "azure").lower()
    if backend_name == "azure":
        return AzureOpenAIBackend(cfg)
    if backend_name in {"exllama", "exllamav2", "local"}:
        return ExLlamaV2Backend(cfg)
    raise ValueError(f"Unsupported LLM backend: {cfg.backend}")

