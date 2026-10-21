"""Deterministic mock client for testing the meter without spending money.

A real model is the wrong thing to test a ledger against: it is
nondeterministic, slow, and costs money. The mock lets a test say
"return exactly 4000 completion tokens and 500 thinking tokens" and then
assert the ledger did the right thing.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from ..ledger import Usage
from .base import Message, Response

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
        Produces the response text from the message list. Defaults to a
        short canned reply.
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
        responder: Callable[[Sequence[Message], int], str] | None = None,
        undercount: int = 0,
    ) -> None:
        self.model = model
        self._completion = completion_tokens
        self._thinking = thinking_tokens
        self._responder = responder
        self._undercount = undercount
        self.call_count = 0
        self.seen: list[list[Message]] = []

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _resolve(spec, idx: int) -> int:
        return spec(idx) if callable(spec) else int(spec)

    def _true_prompt_tokens(self, messages: Sequence[Message]) -> int:
        return sum(naive_token_count(m.content) + 3 for m in messages)

    # -- LLMClient -----------------------------------------------------

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return max(1, self._true_prompt_tokens(messages) - self._undercount)

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
        idx = self.call_count
        self.call_count += 1
        self.seen.append(list(messages))

        completion = min(self._resolve(self._completion, idx), max_output_tokens)
        thinking = min(self._resolve(self._thinking, idx), thinking_budget)

        if self._responder is not None:
            text = self._responder(messages, idx)
        else:
            text = f"mock response {idx}"

        return Response(
            text=text,
            usage=Usage(
                prompt_tokens=self._true_prompt_tokens(messages),
                completion_tokens=completion,
                thinking_tokens=thinking,
            ),
            model=self.model,
            finish_reason="length" if completion >= max_output_tokens else "stop",
        )
