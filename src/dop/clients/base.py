"""Provider-agnostic model client interface.

Every architecture talks to a MeteredClient, which wraps one of these.
No code path may reach a provider SDK directly: if it can, the token
numbers are wrong and the experiment is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..ledger import Usage

Role = str  # "system" | "user" | "assistant" | "tool"


@dataclass
class Message:
    role: Role
    content: str
    name: str | None = None  # tool name, when role == "tool"

    def to_dict(self) -> dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Response:
    text: str
    usage: Usage
    model: str
    finish_reason: str = "stop"
    raw: Any = field(default=None, repr=False)


class LLMClient(Protocol):
    """Minimal surface an architecture needs from a model provider."""

    model: str

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Prompt token count, used to size the reservation before calling.

        Must be exact or an overestimate. An underestimate lets a run
        exceed the cap, which shows up as ledger.overshoot > 0.
        """
        ...

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
        """One completion. Must report real provider usage, including
        thinking tokens where the model produces them."""
        ...
