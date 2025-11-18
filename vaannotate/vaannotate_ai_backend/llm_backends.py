"""Pluggable backends for active-learning LLM calls.

This module provides a light abstraction around the two inference backends we
currently support:

* Azure OpenAI Chat Completions (the historical default)
* Local ExLlamaV2 inference with LM Format Enforcer (LMFE) for strict JSON
  schema control

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
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - avoid runtime circular import
    from .engine import LLMConfig

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - handled when backend is constructed
    AzureOpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
    from exllamav2.generator.filters import ExLlamaV2PrefixFilter
except Exception:  # pragma: no cover - handled during backend construction
    ExLlamaV2 = None  # type: ignore
    ExLlamaV2Cache = None  # type: ignore
    ExLlamaV2Config = None  # type: ignore
    ExLlamaV2Tokenizer = None  # type: ignore
    ExLlamaV2DynamicGenerator = None  # type: ignore
    ExLlamaV2Sampler = None  # type: ignore
    ExLlamaV2PrefixFilter = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
except Exception:  # pragma: no cover - handled when backend initialises
    JsonSchemaParser = None  # type: ignore
    ExLlamaV2TokenEnforcerFilter = None  # type: ignore


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
        if content is None:
            content = getattr(choice, "content", "")
        content = content or ""
        data = json.loads(content)
        logprob_info = getattr(choice, "logprobs", None)
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
    """Backend that runs inference locally via ExLlamaV2 with LMFE."""

    def __init__(self, cfg: LLMConfig):
        if ExLlamaV2 is None or ExLlamaV2Config is None or ExLlamaV2Tokenizer is None:
            raise ImportError(
                "ExLlamaV2 backend requested but exllamav2 is not installed."
            )
        if JsonSchemaParser is None or ExLlamaV2TokenEnforcerFilter is None:
            raise ImportError(
                "ExLlamaV2 backend requires lm-format-enforcer with the exllamav2 integration."
            )
        if ExLlamaV2PrefixFilter is None:
            raise ImportError(
                "ExLlamaV2 backend requires the prefix filter from exllamav2.generator.filters."
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

    def _build_json_filters(self, response_format: Optional[Mapping[str, Any]]):
        schema: Mapping[str, Any]
        if response_format and response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}) or {"type": "object"}
        else:
            schema = {"type": "object"}
        parser = JsonSchemaParser(schema)
        lmfe_filter = ExLlamaV2TokenEnforcerFilter(parser, self.tokenizer)
        prefix_filter = ExLlamaV2PrefixFilter(self.model, self.tokenizer, "{")
        return [lmfe_filter, prefix_filter]

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
            settings.filters = list(filters)
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
        filters = self._build_json_filters(response_format)
        t0 = time.time()
        text, token_ids, token_logprobs = self._generate(
            prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            filters=filters,
            stop_conditions=self._json_stop_conditions,
        )
        latency = time.time() - t0
        self._post_call()
        try:
            data = json.loads(text)
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Local backend produced invalid JSON: {exc}\n{text}") from exc
        logprob_payload = None
        if logprobs:
            logprob_payload = {
                "tokens": token_ids,
                "logprobs": token_logprobs,
            }
        print(text)
        print(logprob_payload)
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
        top_logprobs: int = 5,
    ) -> ForcedChoiceResult:
        prompt = self._format_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

        import torch
        import torch.nn.functional as F  # type: ignore

        option_logps: Dict[str, float] = {}
        latency_total = 0.0

        class _Processor:
            """Logits processor that records the logprob of a single target token."""

            def __init__(self, target_id: int):
                self._target = int(target_id)
                self.last_logprob: float = float("-1e9")

            def __call__(self, logits: torch.Tensor) -> torch.Tensor:
                if logits.dim() == 3:
                    scores = logits[:, -1, :]
                else:
                    scores = logits
                log_probs = F.log_softmax(scores, dim=-1)
                lp = log_probs[..., self._target]
                self.last_logprob = float(lp.reshape(-1)[0].item())
                mask = torch.full_like(logits, float("-inf"))
                mask[..., self._target] = logits[..., self._target]
                return mask

        def _processor_for_letter(letter_text: str):
            encoded = self.tokenizer.encode(letter_text)
            token_ids = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
            if not token_ids:
                return None
            return _Processor(int(token_ids[0]))

        for letter, option in zip(letters, options):
            processor = _processor_for_letter(letter)
            if processor is None:
                option_logps[option] = -1e9
                continue
            self._respect_rpm_limit()
            t0 = time.time()
            self._generate(
                prompt,
                max_new_tokens=1,
                temperature=0.0,
                logits_processors=[processor],
            )
            latency = time.time() - t0
            latency_total += latency
            self._post_call()
            option_logps[option] = float(getattr(processor, "last_logprob", -1e9))
        logits = list(option_logps.values())
        m = max(logits)
        probs = [math.exp(v - m) for v in logits]
        denom = sum(probs)
        if denom <= 0:
            probs = [1.0 / len(probs) for _ in probs]
        else:
            probs = [p / denom for p in probs]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        option_probs = {options[i]: float(probs[i]) for i in range(len(options))}
        prediction = options[max(range(len(probs)), key=lambda idx: probs[idx])]
        avg_latency = latency_total / max(1, len(option_logps))
        return ForcedChoiceResult(
            option_probs=option_probs,
            option_logprobs=option_logps,
            prediction=prediction,
            entropy=float(entropy),
            latency_s=float(avg_latency),
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

