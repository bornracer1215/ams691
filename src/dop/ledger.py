"""Budget ledger: the single source of truth for token spend in a run.

Design notes
------------
Completion length is not known before a call returns, so a cap cannot be
enforced by charging after the fact without overshooting. We use
reserve-then-commit:

    1. Estimate the prompt locally and reserve
       (prompt_est + max_output_tokens + thinking_budget).
    2. If the reservation does not fit in the remaining budget, raise
       BudgetExhausted. The caller halts and is scored as-is.
    3. After the call returns, commit the provider-reported usage and
       release the unused hold.

Every token in the system is charged here, including prompt tokens
(which re-include the whole conversation on every turn), thinking
tokens, and tool output (which enters the budget as prompt tokens on the
following call).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Usage:
    """Provider-reported token usage for a single model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.thinking_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.thinking_tokens + other.thinking_tokens,
        )


class BudgetExhausted(Exception):
    """Raised when a reservation does not fit in the remaining budget.

    Architectures must catch this and return their best answer so far with
    halted=True, rather than letting it propagate as a crash. A halted run
    is scored as-is, never as zero.
    """

    def __init__(self, requested: int, remaining: int, agent_id: str = "") -> None:
        self.requested = requested
        self.remaining = remaining
        self.agent_id = agent_id
        super().__init__(
            f"budget exhausted: agent={agent_id or '?'} "
            f"requested={requested} remaining={remaining}"
        )


@dataclass
class LedgerEvent:
    seq: int
    kind: str  # "reserve" | "commit" | "release" | "refused"
    agent_id: str
    amount: int
    spent_after: int
    held_after: int
    wall_time: float
    detail: dict[str, Any] = field(default_factory=dict)


class Reservation:
    """A hold on part of the budget. Commit it or release it, exactly once."""

    __slots__ = ("_ledger", "amount", "agent_id", "_closed")

    def __init__(self, ledger: "BudgetLedger", amount: int, agent_id: str) -> None:
        self._ledger = ledger
        self.amount = amount
        self.agent_id = agent_id
        self._closed = False

    def commit(self, usage: Usage) -> None:
        if self._closed:
            raise RuntimeError("reservation already closed")
        self._closed = True
        self._ledger._commit(self, usage)

    def release(self) -> None:
        """Cancel without charging. Use when the call failed before billing."""
        if self._closed:
            return
        self._closed = True
        self._ledger._release(self)

    def __enter__(self) -> "Reservation":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._closed:
            self.release()
        return False


class BudgetLedger:
    """One hard shared cap for an entire run, across every agent in it.

    Thread-safe. Concurrency policy for parallel topologies
    (Supervisor-Worker, Debate) is first-come-first-served under a single
    lock: an agent that cannot reserve gets BudgetExhausted and must return
    its partial state. No agent is starved deliberately and none can
    collectively overspend the cap.
    """

    def __init__(self, cap: int, run_id: str = "run") -> None:
        if cap <= 0:
            raise ValueError("cap must be positive")
        self.cap = cap
        self.run_id = run_id
        self._spent = 0
        self._held = 0
        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.time()
        self._parent: "BudgetLedger | None" = None
        self._parent_res: "Reservation | None" = None
        self._parent_agent = ""
        self._settled = False
        self.events: list[LedgerEvent] = []
        self.usage = Usage()
        self.calls = 0
        self.refusals = 0
        # Charged beyond what was reserved, i.e. prompt estimator error.
        # Should stay at 0 with an exact tokenizer; tracked to prove it.
        self.overshoot = 0

    # -- introspection -------------------------------------------------

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def held(self) -> int:
        with self._lock:
            return self._held

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.cap - self._spent - self._held

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    # -- core ----------------------------------------------------------

    def reserve(self, amount: int, agent_id: str = "root") -> Reservation:
        if amount <= 0:
            raise ValueError("reservation must be positive")
        with self._lock:
            available = self.cap - self._spent - self._held
            if amount > available:
                self.refusals += 1
                self._record("refused", agent_id, amount)
                raise BudgetExhausted(amount, available, agent_id)
            self._held += amount
            self._record("reserve", agent_id, amount)
            return Reservation(self, amount, agent_id)

    def _commit(self, res: Reservation, usage: Usage) -> None:
        with self._lock:
            self._held -= res.amount
            self._spent += usage.total
            self.usage = self.usage + usage
            self.calls += 1
            if usage.total > res.amount:
                self.overshoot += usage.total - res.amount
            self._record(
                "commit",
                res.agent_id,
                usage.total,
                detail={
                    "reserved": res.amount,
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "thinking": usage.thinking_tokens,
                },
            )

    def _release(self, res: Reservation) -> None:
        with self._lock:
            self._held -= res.amount
            self._record("release", res.agent_id, res.amount)

    # -- deliberate sub-allocation -------------------------------------

    def allocate(self, amount: int, agent_id: str = "worker") -> "BudgetLedger":
        """Carve a guaranteed sub-budget out of this ledger.

        Pure first-come-first-served lets one agent drain the cap before a
        concurrently-started sibling makes its first reservation, so a
        worker can end up with zero calls for reasons of thread scheduling
        rather than anything about the architecture. That is
        nondeterministic variance injected straight into the measurement.

        Topologies that fan out should therefore allocate explicitly: the
        supervisor decides each worker's slice, the worker spends only
        from it, and whatever it does not spend returns to the parent on
        settle(). The allocation policy becomes a documented property of
        the architecture rather than an artifact of the GIL.
        """
        res = self.reserve(amount, agent_id=agent_id)
        child = BudgetLedger(amount, run_id=f"{self.run_id}/{agent_id}")
        child._parent = self
        child._parent_res = res
        child._parent_agent = agent_id
        return child

    def settle(self) -> None:
        """Return a sub-budget's unspent remainder to its parent."""
        if self._parent is None or self._settled:
            return
        self._settled = True
        with self._lock:
            usage, calls, refusals = self.usage, self.calls, self.refusals
            overshoot = self.overshoot
        self._parent._absorb(
            self._parent_res, usage, calls, refusals, overshoot, self._parent_agent
        )

    def _absorb(self, res, usage: Usage, calls: int, refusals: int,
                overshoot: int, agent_id: str) -> None:
        with self._lock:
            self._held -= res.amount
            self._spent += usage.total
            self.usage = self.usage + usage
            self.calls += calls
            self.refusals += refusals
            self.overshoot += overshoot
            self._record(
                "absorb", agent_id, usage.total,
                detail={"allocated": res.amount, "calls": calls,
                        "unused": res.amount - usage.total},
            )
        res._closed = True

    def __enter__(self) -> "BudgetLedger":
        return self

    def __exit__(self, *exc) -> bool:
        self.settle()
        return False

    def _record(self, kind: str, agent_id: str, amount: int, detail=None) -> None:
        # caller holds the lock
        self._seq += 1
        self.events.append(
            LedgerEvent(
                seq=self._seq,
                kind=kind,
                agent_id=agent_id,
                amount=amount,
                spent_after=self._spent,
                held_after=self._held,
                wall_time=round(time.time() - self._t0, 4),
                detail=detail or {},
            )
        )

    # -- reporting -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "cap": self.cap,
                "spent": self._spent,
                "remaining": self.cap - self._spent - self._held,
                "utilization": round(self._spent / self.cap, 4),
                "calls": self.calls,
                "refusals": self.refusals,
                "overshoot": self.overshoot,
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "thinking_tokens": self.usage.thinking_tokens,
            }

    def event_dicts(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]
