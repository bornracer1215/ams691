# Does Orchestration Pay?

Token-matched ablation of multi-agent LLM topologies.
AMS 691.01, Fall 2026. Parth Kambli, Viraj Chogle, Hritwik Biyani.

This repo contains the token meter, the ClassEval-Pro and LiveSQLBench
loaders, graders and agent tools, one shared ReAct loop, the four
topologies (Single, Best-of-N, Supervisor-Worker, Debate) and `run_one`.

## Quick start

```bash
uv sync                          
uv run pytest -q                 # no API key needed; fetches ClassEval-Pro once
uv run python examples/smoke_run.py  # simulated supervisor + 3 workers under one cap
```

Grading and code execution need [podman](https://podman.io/docs/installation)
with its API socket: on Linux `systemctl --user enable --now podman.socket`;
on macOS/Windows `podman machine init` once, then `podman machine start`.
The sandbox image builds itself on first use. Tests that need podman are
skipped without it.

Once per git clone, install the git hook that runs ruff (lint + format) on
every commit:

```bash
uv run pre-commit install
```

For real runs, put `GEMINI_API_KEY=...` in `.env` at the repo root
(gitignored), or export it.

## Benchmarks

Both are pinned (commit/revision hash + sha256 per file) and download
themselves on first use into `~/.cache/dop` (override with `DOP_CACHE_DIR`).
Each uses a hand-picked 60-task subset, listed with its selection rules in
its module.

- **ClassEval-Pro** ([repo](https://github.com/ian-Kappa/ClassEval-Pro),
  arXiv:2604.26923): class-level code generation, 20 small / medium / large
  classes by method count. `src/dop/tasks/classeval_pro/`.
- **LiveSQLBench-Base-Lite-SQLite**
  ([HF](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite),
  CC BY-SA 4.0): text-to-SQL, 20 Simple / Moderate / Challenging SELECT
  tasks over 18 databases. Graded by BIRD's own evaluator
  (bird-bench/mini_dev, pinned), run inside the sandbox.
  `src/dop/tasks/livesqlbench/`.

## Layout

```
src/dop/
  clients/           model providers; know nothing about budgets
    base.py          LLMClient protocol, Message, Response, Usage
    mock.py          deterministic fake provider for tests
    gemini.py        Google AI Studio client (default gemini-3.5-flash-lite, retries 429/5xx)
  meter/             token accounting, on top of clients/
    ledger.py        BudgetLedger, Reservation, BudgetExhausted
    client.py        MeteredClient, the only sanctioned path to a model
    trace.py         TraceWriter, JSONL run records (input to D3/D4)
  sandbox.py         run untrusted Python in a fresh podman container (llm-sandbox)
  agent.py           the one ReAct loop: protocol prompt, tool calls, forced final
  topologies/
    base.py          Topology, Outcome, parallel fan-out, Universal Self-Consistency
    single.py        one agent
    best_of_n.py     N independent agents, USC picks
    supervisor_worker.py  supervisor with a delegate tool; workers see only the subtask
    debate.py        independent answers, then rounds of revision, USC picks
  runner.py          run_one: one topology x task x seed -> graded trace
  tasks/
    base.py          Task, Tool, GradeResult, Benchmark protocol
    fetch.py         pinned, hash-checked downloads into the cache
    classeval_pro/   loader, subset, grader, python tool + Containerfile (Python + numpy)
    livesqlbench/    loader, subset, grader, sql tool + Containerfile + manifest.json
tests/
  test_cap_enforcement.py   WP1 week-1 exit criterion
  test_sandbox.py           sandbox isolation (needs podman)
  test_classeval_pro.py     subset, extraction, grading
  test_livesqlbench.py      subset, prompt, grading with synthetic gold
  test_agent.py             the ReAct loop against the mock client
  test_topologies.py        all four topologies and run_one, mock client
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
documented property of the topology rather than an artifact of the GIL.
See `test_allocation_guarantees_every_worker_a_slice`.

**One loop, one protocol.** Every agent in every topology runs
`agent.react`, with the same protocol text, step limit and tool output
bound. A topology adds only role text and, for Supervisor-Worker, the
`delegate` tool. Tools are declared as native functions and every reply
must call one; agents answer by calling `submit`. (Tools described as text
tags made Gemini attempt native calls anyway and fail.) The USC judge's
`submit` takes an enum of response numbers.

**Exact holds after one call per role.** Free-tier `count_tokens` cannot
count the system instruction or function declarations, so their size is
approximated generously until `calibrate()` has seen one real call with
the same system prompt and tools, and measured exactly after.

**Halted agents get one forced answer.** Before each step the loop checks
that the budget covers that step plus one more; if not, or after
`MAX_STEPS`, the agent is told to answer now and may call only `submit`.
Without this, scoring would measure which topology hits the cap first, not
which uses a budget well. An agent that never submits has no answer, and the
run scores 0 without grading.

**Same sampling for every topology.** The provider's default temperature
(1.0 for Gemini, which Google recommends keeping for 3.x) everywhere, so
Single is not compared at a different temperature than Best-of-N. `for_agent()`
derives each agent's seed from the run seed and its id, so parallel samples
differ but a run is reproducible.

**Selection never sees the hidden tests.** Best-of-N and Debate pick their
answer with Universal Self-Consistency, one judge call charged to the run.
`run_one` also grades every candidate (`candidate_scores`, oracle pass@N), as
a diagnostic only.

## WP1 week-1 exit criterion

All three hold (`pytest`):

1. A runaway agent halts at the cap and never exceeds it.
2. Ledger total equals the sum of provider-reported usage, exactly.
3. Parallel agents sharing one cap never collectively overspend.

Plus: failed calls release their hold, thinking tokens are charged, context
re-growth dominates a long loop, and tokenizer undercount surfaces as
`ledger.overshoot` rather than silently breaching the cap.

## Not done yet

- A grid over topology x task x seed, with resume
