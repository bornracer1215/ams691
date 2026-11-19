"""Provider-agnostic model client interface.

Every topology talks to a MeteredClient, which wraps one of these.
No code path may reach a provider SDK directly: if it can, the token
numbers are wrong and the experiment is invalid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Role = str  # "system" | "user" | "assistant" | "tool"


@dataclass(frozen=True)
class Usage:
    """Provider-reported token usage for a single model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.thinking_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.thinking_tokens + other.thinking_tokens,
        )


@dataclass(frozen=True)
class ToolSpec:
    """A function the model may call, declared to the provider natively.
    `parameters` is a JSON schema for the call's arguments, an object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None  # the provider's call id, echoed in the result


@dataclass
class Message:
    role: Role
    content: str
    name: str | None = None  # tool name, when role == "tool"
    tool_call: ToolCall | None = None  # the call an assistant message made
    # The provider's own form of an assistant message, sent back unchanged
    # on later turns. Gemini 3 requires this: its replies carry thought
    # signatures that must be returned with the function calls.
    replay: Any = field(default=None, repr=False, compare=False)
    # An assistant reply's hidden reasoning. Gemini bills it again as prompt
    # tokens on every later call (it travels in the replayed thought
    # signature), and count_tokens cannot see it, so estimates add it.
    thinking_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call:
            d["tool_call"] = {"name": self.tool_call.name, "args": self.tool_call.args}
        return d


@dataclass
class Response:
    text: str
    usage: Usage
    model: str
    finish_reason: str = "stop"
    tool_call: ToolCall | None = None
    replay: Any = field(default=None, repr=False)  # see Message.replay
    raw: Any = field(default=None, repr=False)

    def message(self) -> Message:
        """This reply as the assistant turn of a conversation."""
        return Message(
            "assistant",
            self.text,
            tool_call=self.tool_call,
            replay=self.replay,
            thinking_tokens=self.usage.thinking_tokens,
        )


class LLMClient(Protocol):
    """Minimal surface an agent needs from a model provider."""

    model: str

    def count_tokens(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> int:
        """Prompt token count, tool declarations included, used to size the
        reservation before calling.

        Must be exact or an overestimate. An underestimate lets a run
        exceed the cap, which shows up as ledger.overshoot > 0.
        """
        ...

    def calibrate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        prompt_tokens: int,
    ) -> None:
        """Learn from a finished call's provider-reported prompt size, for
        any part of the prompt count_tokens can only approximate."""
        ...

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
        """One completion. Must report real provider usage, including
        thinking tokens where the model produces them.

        With tools, the reply must be a call to one of them (or to one of
        `tool_choice`, when given), returned as Response.tool_call. The
        provider may still fail to produce one; tool_call is then None."""
        ...
