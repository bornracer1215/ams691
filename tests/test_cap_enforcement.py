"""Exit criterion for WP1 week 1.

Three properties must hold before any architecture work begins:

  1. A runaway agent halts at the cap and never exceeds it.
  2. Ledger total equals the sum of provider-reported usage, exactly.
  3. Parallel agents sharing one cap never collectively overspend.

A metering bug found after the full runs invalidates every result and
cannot be fixed by re-running a subset, which is why these come first.
"""

from __future__ import annotations

import concurrent.futures as cf

import pytest

from dop.clients.base import Message
from dop.clients.mock import MockClient
from dop.ledger import BudgetExhausted, BudgetLedger, Usage
from dop.metering import MeteredClient


def msgs(n_words: int = 20) -> list[Message]:
    return [Message(role="user", content=" ".join(["word"] * n_words))]


# ---------------------------------------------------------------- 1 ---

def test_runaway_agent_halts_at_cap():
    """An agent instructed to loop forever stops at the cap, never above."""
    cap = 9_000
    ledger = BudgetLedger(cap, run_id="runaway")
    client = MockClient(completion_tokens=200, thinking_tokens=0)
    mc = MeteredClient(client, ledger, default_max_output_tokens=256)

    history = msgs(30)
    halted = False
    for _ in range(10_000):  # would never terminate on its own
        try:
            resp = mc.generate(history)
        except BudgetExhausted:
            halted = True
            break
        # grow the context the way a real ReAct loop does
        history = history + [Message(role="assistant", content="tok " * 150)]

    assert halted, "loop should have been stopped by the budget"
    assert ledger.spent <= cap, f"spent {ledger.spent} exceeded cap {cap}"
    assert ledger.refusals >= 1
    assert ledger.calls > 1


def test_single_call_larger_than_cap_is_refused_not_truncated():
    ledger = BudgetLedger(100)
    mc = MeteredClient(MockClient(), ledger, default_max_output_tokens=4096)
    with pytest.raises(BudgetExhausted):
        mc.generate(msgs(10))
    assert ledger.spent == 0
    assert ledger.calls == 0


def test_context_regrowth_is_charged_every_turn():
    """Prompt tokens re-include the whole conversation on every turn.

    This is the accounting most prior work gets wrong: a 20-step loop
    pays for its history 20 times. Prompt spend must dominate completion
    spend in a long loop.
    """
    ledger = BudgetLedger(200_000)
    mc = MeteredClient(MockClient(completion_tokens=50), ledger,
                       default_max_output_tokens=64)
    history = msgs(50)
    for _ in range(20):
        mc.generate(history)
        history = history + [Message(role="assistant", content="tok " * 50)]

    assert ledger.usage.prompt_tokens > ledger.usage.completion_tokens * 5


# ---------------------------------------------------------------- 2 ---

def test_ledger_total_equals_sum_of_provider_usage():
    ledger = BudgetLedger(500_000)
    client = MockClient(completion_tokens=lambda i: 40 + i,
                        thinking_tokens=lambda i: 10 * i)
    mc = MeteredClient(client, ledger, default_max_output_tokens=512,
                       thinking_budget=512)

    manual = Usage()
    history = msgs(25)
    for _ in range(25):
        resp = mc.generate(history)
        manual = manual + resp.usage
        history = history + [Message(role="assistant", content=resp.text)]

    assert ledger.usage == manual
    assert ledger.spent == manual.total
    commits = [e for e in ledger.events if e.kind == "commit"]
    assert sum(e.amount for e in commits) == manual.total
    assert ledger.held == 0, "all reservations should be settled"


def test_thinking_tokens_are_charged():
    """Charging prompt + completion but not thinking is the exact error
    this project exists to correct."""
    ledger = BudgetLedger(100_000)
    client = MockClient(completion_tokens=100, thinking_tokens=400)
    mc = MeteredClient(client, ledger, default_max_output_tokens=128,
                       thinking_budget=512)
    resp = mc.generate(msgs(10))

    assert resp.usage.thinking_tokens == 400
    assert ledger.usage.thinking_tokens == 400
    assert ledger.spent == resp.usage.total
    naive = resp.usage.prompt_tokens + resp.usage.completion_tokens
    assert ledger.spent > naive, "thinking tokens must be inside the total"


def test_failed_call_releases_its_hold():
    class Boom(MockClient):
        def generate(self, *a, **kw):
            raise RuntimeError("provider 500")

    ledger = BudgetLedger(10_000)
    mc = MeteredClient(Boom(), ledger, default_max_output_tokens=256)
    with pytest.raises(RuntimeError):
        mc.generate(msgs(10))
    assert ledger.held == 0
    assert ledger.spent == 0
    assert ledger.remaining == 10_000


def test_tokenizer_undercount_is_recorded_as_overshoot():
    """With an exact tokenizer overshoot stays 0. If a provider's counter
    understates the prompt, we want that visible rather than silent."""
    ledger = BudgetLedger(100_000)
    client = MockClient(completion_tokens=10, undercount=50)
    mc = MeteredClient(client, ledger, default_max_output_tokens=16)
    mc.generate(msgs(60))
    assert ledger.overshoot > 0


# ---------------------------------------------------------------- 3 ---

def test_parallel_agents_never_collectively_overspend():
    """Supervisor-Worker and Debate run agents concurrently against one
    shared cap. First-come-first-served under a single lock."""
    cap = 50_000
    ledger = BudgetLedger(cap, run_id="parallel")
    client = MockClient(completion_tokens=300)
    root = MeteredClient(client, ledger, default_max_output_tokens=384)

    def worker(wid: int) -> int:
        mc = root.for_agent(f"worker-{wid}")
        history = msgs(40)
        calls = 0
        while True:
            try:
                mc.generate(history, step="work")
            except BudgetExhausted:
                return calls
            calls += 1
            history = history + [Message(role="assistant", content="tok " * 100)]

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))

    assert ledger.spent <= cap, f"spent {ledger.spent} exceeded cap {cap}"
    assert ledger.held == 0
    assert sum(results) == ledger.calls
    # NOTE: raw FCFS bounds the total but does NOT guarantee fairness. A
    # worker can get zero calls purely from thread scheduling. Fan-out
    # topologies must use allocate() instead; see the test below.


def test_allocation_guarantees_every_worker_a_slice():
    """Deliberate sub-budgets remove scheduling nondeterminism from the
    measurement: each worker gets exactly its slice regardless of which
    thread wins the race, and unspent budget returns to the parent."""
    cap = 60_000
    ledger = BudgetLedger(cap, run_id="allocated")
    client = MockClient(completion_tokens=300)
    root = MeteredClient(client, ledger, default_max_output_tokens=384)

    slice_size = 12_000  # supervisor keeps 12k for itself

    def worker(wid: int) -> tuple[int, int]:
        child = ledger.allocate(slice_size, agent_id=f"worker-{wid}")
        mc = MeteredClient(client, child, agent_id=f"worker-{wid}",
                           default_max_output_tokens=384)
        history = msgs(40)
        calls = 0
        try:
            while True:
                try:
                    mc.generate(history, step="work")
                except BudgetExhausted:
                    break
                calls += 1
                history = history + [Message(role="assistant", content="tok " * 100)]
        finally:
            child.settle()
        return calls, child.spent

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))

    calls = [c for c, _ in results]
    spends = [s for _, s in results]

    assert all(c > 0 for c in calls), "allocation must not starve a worker"
    assert all(s <= slice_size for s in spends), "a worker overran its slice"
    assert ledger.spent == sum(spends)
    assert ledger.spent <= cap
    assert ledger.held == 0, "all allocations settled"
    # unspent remainder came back rather than being lost
    assert ledger.remaining == cap - sum(spends)


def test_sub_agents_share_the_parent_ledger():
    ledger = BudgetLedger(100_000)
    root = MeteredClient(MockClient(completion_tokens=20), ledger,
                         default_max_output_tokens=32)
    a, b = root.for_agent("a"), root.for_agent("b")
    a.generate(msgs(5))
    b.generate(msgs(5))

    assert a.ledger is b.ledger is ledger
    assert ledger.calls == 2
    agents = {e.agent_id for e in ledger.events if e.kind == "commit"}
    assert agents == {"a", "b"}
