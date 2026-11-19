"""MeteredClient: the only way an agent is allowed to call a model."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

from ..clients.base import LLMClient, Message, Response, ToolSpec
from .ledger import BudgetExhausted, BudgetLedger
from .trace import TraceWriter


class MeteredClient:
    """Wraps an LLMClient so every call is reserved and charged.

    Each topology receives a MeteredClient bound to the run's single
    ledger. Sub-agents get a child via `for_agent()`, which shares that
    ledger (or a slice of it from `ledger.allocate()`), so a shared cap is
    enforced across the whole system rather than per agent.
    """

    def __init__(
        self,
        client: LLMClient,
        ledger: BudgetLedger,
        *,
        agent_id: str = "root",
        trace: TraceWriter | None = None,
        default_max_output_tokens: int = 1024,
        thinking_budget: int = 0,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.client = client
        self.ledger = ledger
        self.agent_id = agent_id
        self.trace = trace
        self.default_max_output_tokens = default_max_output_tokens
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.seed = seed

    def for_agent(
        self, agent_id: str, *, ledger: BudgetLedger | None = None
    ) -> MeteredClient:
        """A view for a sub-agent, sharing this run's trace and settings.

        It charges this client's ledger unless given another, normally a
        slice from `ledger.allocate()`. Its seed is derived from the run
        seed and agent_id: agents sampling the same prompt at temperature
        > 0 must not share a seed, or Best-of-N draws N copies of one
        sample, yet the run as a whole must stay reproducible.
        """
        seed = None
        if self.seed is not None:
            seed = zlib.crc32(f"{self.seed}/{agent_id}".encode()) & 0x7FFFFFFF
        return MeteredClient(
            self.client,
            ledger if ledger is not None else self.ledger,
            agent_id=agent_id,
            trace=self.trace,
            default_max_output_tokens=self.default_max_output_tokens,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            seed=seed,
        )

    def estimate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        max_output_tokens: int | None = None,
    ) -> int:
        """Tokens generate() would reserve for these messages and tools."""
        max_out = max_output_tokens or self.default_max_output_tokens
        return (
            self.client.count_tokens(messages, tools) + max_out + self.thinking_budget
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        tool_choice: Sequence[str] | None = None,
        max_output_tokens: int | None = None,
        step: str = "",
    ) -> Response:
        """Reserve, call, commit. Raises BudgetExhausted if it will not fit.

        Callers must treat BudgetExhausted as "halt and return best answer
        so far", never as a crash. Tool declarations are prompt tokens, so
        they are part of the estimate; after the call the client calibrates
        that estimate against the provider's reported prompt size.
        """
        max_out = max_output_tokens or self.default_max_output_tokens
        prompt_est = self.client.count_tokens(messages, tools)
        hold = prompt_est + max_out + self.thinking_budget

        reservation = self.ledger.reserve(hold, agent_id=self.agent_id)
        try:
            resp = self.client.generate(
                messages,
                max_output_tokens=max_out,
                temperature=self.temperature,
                thinking_budget=self.thinking_budget,
                seed=self.seed,
                tools=tools,
                tool_choice=tool_choice,
            )
        except BaseException:
            reservation.release()
            raise

        reservation.commit(resp.usage)
        self.client.calibrate(messages, tools, resp.usage.prompt_tokens)

        if self.trace is not None:
            self.trace.call(
                agent_id=self.agent_id,
                step=step,
                model=resp.model,
                prompt_est=prompt_est,
                reserved=hold,
                usage=resp.usage,
                spent_after=self.ledger.spent,
                cap=self.ledger.cap,
                finish_reason=resp.finish_reason,
                n_messages=len(messages),
                messages=messages,
                text=resp.text,
                tool_call=resp.tool_call,
            )
        return resp

    def try_generate(self, messages: Sequence[Message], **kw) -> Response | None:
        """generate(), but returns None instead of raising on exhaustion."""
        try:
            return self.generate(messages, **kw)
        except BudgetExhausted:
            return None
