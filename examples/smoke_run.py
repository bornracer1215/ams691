"""End-to-end smoke test of the meter + trace, with no API calls.

    python examples/smoke_run.py

Simulates a supervisor fanning out to three workers under one 40k cap,
writes a JSONL trace, and prints the budget breakdown by agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dop.clients.base import Message
from dop.clients.mock import MockClient
from dop.meter import (
    BudgetExhausted,
    BudgetLedger,
    MeteredClient,
    RunMeta,
    TraceWriter,
    read_trace,
)

CAP = 40_000
OUT = Path(__file__).parent / "smoke_trace.jsonl"


def main() -> None:
    ledger = BudgetLedger(CAP, run_id="smoke")
    client = MockClient(completion_tokens=250, thinking_tokens=120)
    trace = TraceWriter(
        OUT,
        RunMeta(
            run_id="smoke",
            topology="supervisor_worker",
            task_id="demo-001",
            task_category="database",
            task_shape="tool_feedback",
            cap=CAP,
            model=client.model,
            seed=7,
            thinking_budget=256,
        ),
    )

    sup = MeteredClient(
        client,
        ledger,
        agent_id="supervisor",
        trace=trace,
        default_max_output_tokens=320,
        thinking_budget=256,
    )

    plan = sup.generate(
        [Message(role="user", content="Solve the task. " * 40)], step="plan"
    )
    print(f"supervisor planned: {plan.text!r}")

    halted = False
    for wid in range(3):
        name = f"worker-{wid}"
        trace.message(frm="supervisor", to=name, chars=len(plan.text))
        child = ledger.allocate(9_000, agent_id=name)
        mc = MeteredClient(
            client,
            child,
            agent_id=name,
            trace=trace,
            default_max_output_tokens=320,
            thinking_budget=256,
        )
        history = [Message(role="user", content="subtask " * 60)]
        try:
            for step in range(20):
                try:
                    r = mc.generate(history, step=f"step-{step}")
                except BudgetExhausted as e:
                    trace.halt(
                        agent_id=name, requested=e.requested, remaining=e.remaining
                    )
                    break
                history += [
                    Message(role="assistant", content=r.text),
                    Message(
                        role="tool",
                        name="sqlite",
                        content="ERROR: no such column " * 30,
                    ),
                ]
                trace.tool(
                    agent_id=name,
                    tool="sqlite",
                    ok=False,
                    output_chars=len(history[-1].content),
                )
        finally:
            child.settle()

    try:
        final = sup.generate(
            [Message(role="user", content="Synthesize. " * 30)], step="synthesize"
        )
        answer = final.text
    except BudgetExhausted:
        answer, halted = None, True

    s = ledger.summary()
    trace.finish(answer=answer, halted=halted, ledger_summary=s, score=None)

    print("\n--- ledger ---")
    for k, v in s.items():
        print(f"  {k:18} {v}")
    assert s["spent"] <= CAP, "cap violated"

    rows = [r for r in read_trace(OUT) if r["kind"] == "call"]
    by_agent: dict[str, int] = {}
    for r in rows:
        by_agent[r["agent_id"]] = by_agent.get(r["agent_id"], 0) + r["call_total"]
    print("\n--- spend by agent ---")
    for a, t in sorted(by_agent.items(), key=lambda kv: -kv[1]):
        print(f"  {a:14} {t:7,}  ({t / s['spent']:.0%})")
    print(f"\ntrace: {OUT}")


if __name__ == "__main__":
    main()
