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
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
    from exllamav2.tokenizer import ExLlamaV2Tokenizer
    from exllamav2.generator import (
        ExLlamaV2Generator,
        ExLlamaV2StreamingGenerator,
        SequenceGeneratorSettings,
    )
except Exception:  # pragma: no cover - handled during backend construction
    ExLlamaV2 = None  # type: ignore
    ExLlamaV2Cache = None  # type: ignore
    ExLlamaV2Config = None  # type: ignore
    ExLlamaV2Tokenizer = None  # type: ignore
    ExLlamaV2Generator = None  # type: ignore
    ExLlamaV2StreamingGenerator = None  # type: ignore
    SequenceGeneratorSettings = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.exllamav2 import (
        ExLlamaV2LogitsProcessor,
        ExLlamaV2TokenizerProxy,
    )
except Exception:  # pragma: no cover - handled when backend initialises
    JsonSchemaParser = None  # type: ignore
    ExLlamaV2LogitsProcessor = None  # type: ignore
    ExLlamaV2TokenizerProxy = None  # type: ignore


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
        if JsonSchemaParser is None or ExLlamaV2LogitsProcessor is None:
            raise ImportError(
                "ExLlamaV2 backend requires lm-format-enforcer with the exllamav2 integration."
            )
        super().__init__(cfg)
        model_dir = cfg.local_model_dir or os.getenv("LOCAL_LLM_MODEL_DIR")
        if not model_dir:
            raise ValueError("ExLlamaV2 backend requires local_model_dir to be set.")

        config = ExLlamaV2Config()
        config.model_dir = str(model_dir)
        if cfg.local_max_seq_len:
            config.max_seq_len = int(cfg.local_max_seq_len)
        if cfg.local_gpu_split:
            config.set_auto_map(cfg.local_gpu_split)
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
        self.generator = ExLlamaV2Generator(self.model, self.tokenizer, self.cache)

        # LMFE integration keeps strict JSON outputs
        self._tokenizer_proxy = ExLlamaV2TokenizerProxy(self.tokenizer)

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

    def _build_json_logits_processor(self, response_format: Optional[Mapping[str, Any]]):
        schema: Mapping[str, Any]
        if response_format and response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}) or {"type": "object"}
        else:
            schema = {"type": "object"}
        parser = JsonSchemaParser(schema)
        return ExLlamaV2LogitsProcessor(parser, self._tokenizer_proxy)

    def _generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        logits_processors: Optional[Sequence[Any]] = None,
    ) -> tuple[str, List[int], List[float]]:
        settings = SequenceGeneratorSettings()
        settings.temperature = max(0.0, float(temperature))
        settings.top_p = 1.0
        settings.top_k = 0
        settings.token_repetition_penalty = 1.0
        settings.max_new_tokens = int(max_new_tokens)

        stream_gen = ExLlamaV2StreamingGenerator(self.generator)
        if logits_processors:
            for processor in logits_processors:
                stream_gen.add_logits_processor(processor)
        stream_gen.begin_stream(prompt)
        collected_tokens: List[int] = []
        collected_logprobs: List[float] = []
        output_chunks: List[str] = []
        while True:
            token, text = stream_gen.stream_next(settings)
            if token is None:
                break
            collected_tokens.append(int(token))
            collected_logprobs.append(float(stream_gen.last_token_logprob()))
            if text:
                output_chunks.append(text)
            if len(collected_tokens) >= max_new_tokens:
                break
        stream_gen.end_stream()
        return "".join(output_chunks), collected_tokens, collected_logprobs

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
        logits_processor = self._build_json_logits_processor(response_format)
        t0 = time.time()
        text, token_ids, token_logprobs = self._generate(
            prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            logits_processors=[logits_processor],
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
        option_logps: Dict[str, float] = {}
        latency_total = 0.0

        def _processor_for_letter(letter_text: str):
            import torch

            encoded = self.tokenizer.encode(letter_text)
            token_ids = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
            if not token_ids:
                return None

            class _Processor:
                def __init__(self, target: int):
                    self._target = target

                def __call__(self, logits):
                    mask = torch.full_like(logits, float("-inf"))
                    mask[..., self._target] = logits[..., self._target]
                    return mask

            return _Processor(int(token_ids[0]))

        for letter, option in zip(letters, options):
            processor = _processor_for_letter(letter)
            if processor is None:
                option_logps[option] = -1e9
                continue
            self._respect_rpm_limit()
            t0 = time.time()
            _, _, token_logprobs = self._generate(
                prompt,
                max_new_tokens=1,
                temperature=0.0,
                logits_processors=[processor],
            )
            latency = time.time() - t0
            latency_total += latency
            self._post_call()
            option_logps[option] = float(token_logprobs[0]) if token_logprobs else -1e9
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

