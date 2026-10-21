"""MeteredClient: the only way an agent is allowed to call a model."""

from __future__ import annotations

from typing import Sequence

from .clients.base import LLMClient, Message, Response
from .ledger import BudgetExhausted, BudgetLedger
from .trace import TraceWriter


class MeteredClient:
    """Wraps an LLMClient so every call is reserved and charged.

    Each architecture receives a MeteredClient bound to the run's single
    ledger. Sub-agents get a child via `for_agent()`, which shares that
    ledger, so a shared cap is enforced across the whole system rather
    than per agent.
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
        temperature: float = 0.0,
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

    def for_agent(self, agent_id: str) -> "MeteredClient":
        """A view for a sub-agent, sharing this run's ledger and trace."""
        return MeteredClient(
            self.client,
            self.ledger,
            agent_id=agent_id,
            trace=self.trace,
            default_max_output_tokens=self.default_max_output_tokens,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            seed=self.seed,
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        step: str = "",
    ) -> Response:
        """Reserve, call, commit. Raises BudgetExhausted if it will not fit.

        Callers must treat BudgetExhausted as "halt and return best answer
        so far", never as a crash.
        """
        max_out = max_output_tokens or self.default_max_output_tokens
        prompt_est = self.client.count_tokens(messages)
        hold = prompt_est + max_out + self.thinking_budget

        reservation = self.ledger.reserve(hold, agent_id=self.agent_id)
        try:
            resp = self.client.generate(
                messages,
                max_output_tokens=max_out,
                temperature=self.temperature,
                thinking_budget=self.thinking_budget,
                seed=self.seed,
                stop=stop,
            )
        except BaseException:
            reservation.release()
            raise

        reservation.commit(resp.usage)

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
            )
        return resp

    def try_generate(self, messages: Sequence[Message], **kw) -> Response | None:
        """generate(), but returns None instead of raising on exhaustion."""
        try:
            return self.generate(messages, **kw)
        except BudgetExhausted:
            return None
