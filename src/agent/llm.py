"""Thin provider wrapper: complete(messages) -> LLMResponse. Keeps the agent provider-agnostic."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Protocol

from dotenv import load_dotenv

Message = dict  # {"role": "user" | "assistant", "content": str}

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class LLMError(RuntimeError):
    """The provider call failed or returned nothing usable."""


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


class LLM(Protocol):
    def complete(self, messages: list[Message], system: str | None = None) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 4096):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, messages: list[Message], system: str | None = None) -> LLMResponse:
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        start = time.perf_counter()
        try:
            if self.model in _FALLBACK_MODELS:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                resp = self.client.messages.create(**kwargs)
        except self._anthropic.APIError as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        latency = int((time.perf_counter() - start) * 1000)
        if resp.stop_reason == "refusal":
            raise LLMError(f"model refused: {getattr(resp.stop_details, 'category', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(text, resp.usage.input_tokens, resp.usage.output_tokens, latency)


class OpenAILLM:
    def __init__(self, model: str, max_tokens: int = 4096):
        import openai

        self._openai = openai
        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, messages: list[Message], system: str | None = None) -> LLMResponse:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, max_completion_tokens=self.max_tokens
            )
        except self._openai.OpenAIError as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        latency = int((time.perf_counter() - start) * 1000)
        usage = resp.usage
        return LLMResponse(
            resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            latency,
        )


class FakeLLM:
    """Returns scripted outputs in order. Records every prompt it was sent. For tests."""

    def __init__(self, outputs: Iterable[str]):
        self.outputs = list(outputs)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], system: str | None = None) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.outputs:
            raise LLMError("FakeLLM ran out of scripted outputs")
        text = self.outputs.pop(0)
        return LLMResponse(text, tokens_in=sum(len(m["content"]) for m in messages) // 4,
                           tokens_out=len(text) // 4, latency_ms=1)


def get_llm() -> LLM:
    """Build the LLM configured by LLM_PROVIDER / LLM_MODEL."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicLLM(os.getenv("LLM_MODEL", "claude-sonnet-5-5"))
    if provider == "openai":
        model = os.getenv("LLM_MODEL")
        if not model:
            raise LLMError("LLM_MODEL must be set when LLM_PROVIDER=openai")
        return OpenAILLM(model)
    raise LLMError(f"Unknown LLM_PROVIDER: {provider!r}")


def complete(messages: list[Message], system: str | None = None) -> str:
    """Convenience one-shot call with the configured provider."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return get_llm().complete(messages, system).text
