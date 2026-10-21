# Does Orchestration Pay?

Token-matched ablation of multi-agent LLM architectures.
AMS 691.01, Fall 2026. Parth Kambli, Viraj Chogle, Hritwik Biyani.

This repo currently contains **D1 part 1: the token meter**. Architectures,
tasks and graders are not built yet.

## Quick start

```bash
pip install pytest
python -m pytest -q          # 10 tests, no API key needed
python examples/smoke_run.py # simulated supervisor + 3 workers under one cap
```

For real runs:

```bash
pip install google-genai
export GEMINI_API_KEY=...
```

## Layout

```
src/dop/
  ledger.py          BudgetLedger, Reservation, Usage, BudgetExhausted
  metering.py        MeteredClient, the only sanctioned path to a model
  trace.py           TraceWriter, JSONL run records (input to D3/D4)
  clients/
    base.py          LLMClient protocol, Message, Response
    mock.py          deterministic fake provider for tests
    gemini.py        Google AI Studio client
tests/
  test_cap_enforcement.py   WP1 week-1 exit criterion
examples/
  smoke_run.py       end-to-end demo, no API calls
```

## Design decisions worth knowing

**Reserve-then-commit.** Completion length is unknown before a call returns,
so charging after the fact overshoots the cap. Before each call we estimate
the prompt locally and reserve `prompt_est + max_output_tokens +
thinking_budget`. If it does not fit, `BudgetExhausted` is raised and the run
halts. After the call we commit real provider usage and release the unused
hold.

**Everything is charged.** Prompt tokens (which re-include the full
conversation every turn), completion tokens, and thinking tokens. Tool output
enters the budget as prompt tokens on the following call and is additionally
logged by size so coordination overhead is attributable.

**Thinking tokens are not optional.** Gemini reports
`thoughts_token_count` separately from `candidates_token_count`. Charging only
prompt + candidates silently undercounts every reasoning token, which is the
exact accounting error this project exists to correct. `thinking_budget` must
be identical across all four architectures in a comparison, or thinking
becomes an uncontrolled variable that differs by topology.

**Allocation, not a race.** Pure first-come-first-served bounds total spend but
does not guarantee fairness: a worker can get zero calls because of thread
scheduling, injecting nondeterministic variance into the measurement. Fan-out
topologies use `ledger.allocate(n, agent_id=...)` to carve a guaranteed slice,
and `settle()` returns the remainder. The allocation policy is then a
documented property of the architecture rather than an artifact of the GIL.
See `test_allocation_guarantees_every_worker_a_slice`.

**Halted runs are scored as-is, never zero.** Each architecture must catch
`BudgetExhausted` and return its best answer so far. Scoring halted runs as
zero measures which architecture hits the cap first, not which uses a budget
well.

## WP1 week-1 exit criterion

All three hold (`pytest`):

1. A runaway agent halts at the cap and never exceeds it.
2. Ledger total equals the sum of provider-reported usage, exactly.
3. Parallel agents sharing one cap never collectively overspend.

Plus: failed calls release their hold, thinking tokens are charged, context
re-growth dominates a long loop, and tokenizer undercount surfaces as
`ledger.overshoot` rather than silently breaching the cap.

## Not done yet

- Tool layer (python exec, SQLite) for ClassEval and BIRD-SQL
- Base ReAct loop
- The four topologies: Single, Best-of-N, Supervisor-Worker, Debate
- Per-topology "best answer so far" extraction rules
- Task suite and graders (D2)
