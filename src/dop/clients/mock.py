"""Deterministic mock client for testing the meter without spending money.

A real model is the wrong thing to test a ledger against: it is
nondeterministic, slow, and costs money. The mock lets a test say
"return exactly 4000 completion tokens and 500 thinking tokens" and then
assert the ledger did the right thing.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence

from .base import Message, Response, ToolCall, ToolSpec, Usage

Reply = str | ToolCall | tuple[str, ToolCall | None]

_WORD = re.compile(r"\S+")


def naive_token_count(text: str) -> int:
    """Stand-in tokenizer: 1 token per whitespace-delimited chunk, min 1."""
    return max(1, len(_WORD.findall(text)))


class MockClient:
    """Configurable fake provider.

    Parameters
    ----------
    completion_tokens
        Tokens of visible output to report per call. Either a fixed int or
        a callable taking the call index.
    thinking_tokens
        Tokens of hidden reasoning to report per call. Same forms.
    responder
        Produces the reply from the message list and call index: a string
        (plain text), a ToolCall, or a (text, ToolCall) pair. Defaults to a
        short canned text reply. The tools offered, and any tool_choice,
        are recorded in `offered`, but not enforced.
    undercount
        Tokens by which count_tokens should UNDERSTATE the true prompt, to
        simulate an inaccurate tokenizer and exercise overshoot tracking.
    """

    def __init__(
        self,
        *,
        model: str = "mock-1",
        completion_tokens: int | Callable[[int], int] = 50,
        thinking_tokens: int | Callable[[int], int] = 0,
        responder: Callable[[Sequence[Message], int], Reply] | None = None,
        undercount: int = 0,
    ) -> None:
        self.model = model
        self._completion = completion_tokens
        self._thinking = thinking_tokens
        self._responder = responder
        self._undercount = undercount
        self.call_count = 0
        self.seen: list[list[Message]] = []
        self.offered: list[tuple[list[str], list[str] | None]] = []
        self._lock = threading.Lock()  # topologies call from several threads

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _resolve(spec, idx: int) -> int:
        return spec(idx) if callable(spec) else int(spec)

    def _true_prompt_tokens(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> int:
        n = 0
        for m in messages:
            n += naive_token_count(m.content) + 3
            if m.tool_call:
                n += naive_token_count(json.dumps(m.tool_call.args))
        for t in tools:
            n += naive_token_count(
                f"{t.name} {t.description} {json.dumps(t.parameters)}"
            )
        return n

    # -- LLMClient -----------------------------------------------------

    def count_tokens(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> int:
        return max(1, self._true_prompt_tokens(messages, tools) - self._undercount)

    def calibrate(self, messages, tools, prompt_tokens) -> None:
        """Nothing to learn: count_tokens is already exact (or deliberately
        wrong, with `undercount`)."""

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
        with self._lock:
            idx = self.call_count
            self.call_count += 1
            self.seen.append(list(messages))
            self.offered.append(
                ([t.name for t in tools], list(tool_choice) if tool_choice else None)
            )

        completion = min(self._resolve(self._completion, idx), max_output_tokens)
        thinking = min(self._resolve(self._thinking, idx), thinking_budget)

        reply = (
            self._responder(messages, idx)
            if self._responder is not None
            else f"mock response {idx}"
        )
        if isinstance(reply, ToolCall):
            text, call = "", reply
        elif isinstance(reply, tuple):
            text, call = reply
        else:
            text, call = reply, None

        return Response(
            text=text,
            tool_call=call,
            usage=Usage(
                prompt_tokens=self._true_prompt_tokens(messages, tools),
                completion_tokens=completion,
                thinking_tokens=thinking,
            ),
            model=self.model,
            finish_reason="length" if completion >= max_output_tokens else "stop",
        )
