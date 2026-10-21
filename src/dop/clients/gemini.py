"""Gemini client (Google AI Studio / google-genai SDK).

Install:  pip install google-genai
Auth:     export GEMINI_API_KEY=...

Why thinking tokens are handled explicitly
------------------------------------------
Gemini 2.5+ models emit hidden reasoning tokens, reported as
`usage_metadata.thoughts_token_count`, separate from
`candidates_token_count`. Charging only prompt + candidates silently
undercounts every reasoning token, which is precisely the accounting
error this project exists to correct. We charge all three.

`thinking_budget` must be identical across all four architectures within
a comparison, or thinking becomes an uncontrolled variable that differs
by topology and the comparison is no longer token-matched. Set it once
in the run config and never per-architecture.
"""

from __future__ import annotations

import os
from typing import Sequence

from ..ledger import Usage
from .base import Message, Response


def _to_contents(messages: Sequence[Message]):
    """Map our messages onto Gemini contents + system instruction."""
    system_bits: list[str] = []
    contents = []
    for m in messages:
        if m.role == "system":
            system_bits.append(m.content)
            continue
        role = "model" if m.role == "assistant" else "user"
        text = m.content
        if m.role == "tool":
            text = f"[tool:{m.name or 'unknown'}]\n{m.content}"
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents, ("\n\n".join(system_bits) or None)


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install google-genai") from e
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY")
        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.model = model

    def count_tokens(self, messages: Sequence[Message]) -> int:
        contents, system = _to_contents(messages)
        n = self._client.models.count_tokens(model=self.model, contents=contents)
        total = int(getattr(n, "total_tokens", 0) or 0)
        if system:
            # count_tokens does not include the system instruction; approximate
            # it generously so the reservation never understates the prompt.
            total += len(system) // 3 + 16
        return total

    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        thinking_budget: int = 0,
        seed: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> Response:
        from google.genai import types

        contents, system = _to_contents(messages)
        cfg: dict = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if system:
            cfg["system_instruction"] = system
        if seed is not None:
            cfg["seed"] = seed
        if stop:
            cfg["stop_sequences"] = list(stop)
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**cfg),
        )

        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(um, "thoughts_token_count", 0) or 0),
        )

        finish = "stop"
        cands = getattr(resp, "candidates", None) or []
        if cands and getattr(cands[0], "finish_reason", None) is not None:
            finish = str(cands[0].finish_reason)

        return Response(
            text=(getattr(resp, "text", None) or ""),
            usage=usage,
            model=self.model,
            finish_reason=finish,
            raw=resp,
        )
