"""Gemini client (Google AI Studio / google-genai SDK).

Install:  pip install google-genai
Auth:     GEMINI_API_KEY in the environment or in .env at the repo root

Why thinking tokens are handled explicitly
------------------------------------------
Gemini 2.5+ models emit hidden reasoning tokens, reported as
`usage_metadata.thoughts_token_count`, separate from
`candidates_token_count`. Charging only prompt + candidates silently
undercounts every reasoning token, which is precisely the accounting
error this project exists to correct. We charge all three.

Gemini counts thinking tokens inside its own max_output_tokens, so
generate() sends max_output_tokens + thinking_budget, and the reservation
(prompt + max output + thinking budget) stays an exact upper bound.
Gemini 3.x prefers `thinking_level`, but still accepts `thinking_budget`,
and only a budget gives the meter a bound to reserve. How strictly the
budget is kept varies by model (checked October 2026):
  - gemini-3.6-flash: respected; 0 disables thinking.
  - gemini-3.5-flash-lite: 0 is rejected (400), and the budget is only a
    hint (128 -> 364 thinking tokens). The total cap still holds, so the
    reservation stays an upper bound, but thinking can crowd out the
    visible answer: watch finish_reason for MAX_TOKENS.

`thinking_budget` must be identical across all four topologies within
a comparison, or thinking becomes an uncontrolled variable that differs
by topology and the comparison is no longer token-matched. Set it once
in the run config and never per-topology.

Tools are declared as native functions, with function calling mode ANY
(or the allowed names of `tool_choice`), so every reply is a call.
Described only in the prompt, as text tags, tools made Gemini attempt
native calls anyway and fail with MALFORMED_FUNCTION_CALL.

Prompt estimates: the Developer API's count_tokens counts contents only,
not the system instruction or function declarations, nor the thinking of
earlier replies, which Gemini bills again as prompt tokens on every later
call (it travels in the thought signatures we must replay). The thinking
is added from each reply's reported usage. Until calibrate()
has seen a real call with the same system instruction and tools, that
overhead is approximated generously; after, it is the measured
difference between Gemini's reported prompt and count_tokens.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence

from .base import Message, Response, ToolCall, ToolSpec, Usage

# Retried: rate limits and transient server errors. None of these bill
# tokens, so a retry never double-charges; the reservation is simply held
# a little longer. Gives up after `retry_attempts` and raises, which
# run_one records as status "error", to be re-run.
RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)


def _to_contents(messages: Sequence[Message]):
    """Map our messages onto Gemini contents + system instruction.

    Assistant turns are replayed in Gemini's own form when we have it, so
    thought signatures go back unchanged. Consecutive turns of the same
    role (a function response, then a user note) are merged into one."""
    system_bits: list[str] = []
    contents: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_bits.append(m.content)
            continue
        if m.role == "assistant":
            if m.replay is not None:
                contents.append(m.replay)
                continue
            parts: list[dict] = []
            if m.content:
                parts.append({"text": m.content})
            if m.tool_call:
                parts.append(
                    {
                        "function_call": {
                            "name": m.tool_call.name,
                            "args": m.tool_call.args,
                        }
                    }
                )
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            continue
        if m.role == "tool":
            part = {
                "function_response": {
                    "name": m.name or "unknown",
                    "response": {"output": m.content},
                }
            }
        else:
            part = {"text": m.content}
        last = contents[-1] if contents else None
        if isinstance(last, dict) and last["role"] == "user":
            last["parts"].append(part)
        else:
            contents.append({"role": "user", "parts": [part]})
    return contents, ("\n\n".join(system_bits) or None)


def _declarations(tools: Sequence[ToolSpec]) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters_json_schema": t.parameters,
        }
        for t in tools
    ]


def _overhead_key(system: str | None, tools: Sequence[ToolSpec]) -> str:
    return json.dumps([system, _declarations(tools)], sort_keys=True)


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        api_key: str | None = None,
        *,
        retry_attempts: int = 8,
        retry_initial_delay: float = 2.0,
        retry_max_delay: float = 90.0,
    ) -> None:
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install google-genai") from e
        from dotenv import load_dotenv
        from google.genai import types

        load_dotenv()  # real environment variables win over .env
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY")
        retry = types.HttpRetryOptions(
            attempts=retry_attempts,
            initial_delay=retry_initial_delay,
            max_delay=retry_max_delay,
            exp_base=2,
            jitter=1,
            http_status_codes=list(RETRY_STATUS_CODES),
        )
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(retry_options=retry)
        )
        self.model = model
        # measured system instruction + declarations overhead, by content
        self._overhead: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- prompt estimates ------------------------------------------------

    def _count_contents(self, contents: list) -> int:
        if not contents:
            return 0
        n = self._client.models.count_tokens(model=self.model, contents=contents)
        return int(getattr(n, "total_tokens", 0) or 0)

    def count_tokens(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> int:
        contents, system = _to_contents(messages)
        total = self._count_contents(contents)
        # replayed replies re-bill their thinking, which count_tokens misses
        total += sum(m.thinking_tokens for m in messages if m.replay is not None)
        with self._lock:
            measured = self._overhead.get(_overhead_key(system, tools))
        if measured is not None:
            return total + measured
        # Not yet calibrated: approximate generously (~3 chars per token,
        # where English text runs ~4), so the hold never understates it.
        extra = len(system or "") + len(json.dumps(_declarations(tools)))
        return total + (extra // 3 + 16 if extra else 0)

    def calibrate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        prompt_tokens: int,
    ) -> None:
        """Record the exact overhead of this system instruction and tool set:
        Gemini's reported prompt minus count_tokens of the contents. Once
        per distinct overhead; it is the same for every call that shares
        it, e.g. all calls of one agent role."""
        contents, system = _to_contents(messages)
        key = _overhead_key(system, tools)
        with self._lock:
            if key in self._overhead:
                return
        overhead = max(prompt_tokens - self._count_contents(contents), 0)
        with self._lock:
            self._overhead.setdefault(key, overhead)

    # -- calls -----------------------------------------------------------

    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int,
        temperature: float | None = None,  # None: the provider's default
        thinking_budget: int = 0,
        seed: int | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: Sequence[str] | None = None,
    ) -> Response:
        from google.genai import types

        contents, system = _to_contents(messages)
        cfg: dict = {
            # Gemini counts thinking inside max_output_tokens; ours means
            # visible output, and the meter reserves thinking separately.
            "max_output_tokens": max_output_tokens + thinking_budget,
            "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
        }
        if temperature is not None:
            cfg["temperature"] = temperature
        if system:
            cfg["system_instruction"] = system
        if seed is not None:
            cfg["seed"] = seed
        if tools:
            cfg["tools"] = [{"function_declarations": _declarations(tools)}]
            cfg["tool_config"] = {
                "function_calling_config": {
                    "mode": "ANY",
                    **(
                        {"allowed_function_names": list(tool_choice)}
                        if tool_choice
                        else {}
                    ),
                }
            }
            cfg["automatic_function_calling"] = {"disable": True}

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

        cands = getattr(resp, "candidates", None) or []
        cand = cands[0] if cands else None
        finish = str(cand.finish_reason) if cand and cand.finish_reason else "stop"
        content = cand.content if cand else None
        parts = (content.parts if content else None) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        call = next((p.function_call for p in parts if p.function_call), None)
        tool_call = None
        if call is not None and call.name:
            tool_call = ToolCall(call.name, dict(call.args or {}), id=call.id)

        return Response(
            text=text,
            usage=usage,
            # the model that actually served the call: deprecated names are
            # silently routed to newer models
            model=getattr(resp, "model_version", None) or self.model,
            finish_reason=finish,
            tool_call=tool_call,
            replay=content if parts else None,
            raw=resp,
        )
