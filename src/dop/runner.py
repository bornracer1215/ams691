"""run_one: one topology on one task under one cap, traced and graded.

Everything a comparison must hold equal between topologies lives here
and in RunConfig, not in the topologies: the ledger, the model settings
(temperature, thinking budget, output limit, seed) and the grading.
Topologies never see the grader.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clients.base import LLMClient
from .meter import BudgetLedger, MeteredClient, RunMeta, TraceWriter
from .tasks.base import Benchmark, GradeResult, Task
from .topologies.base import Topology


@dataclass(frozen=True)
class RunConfig:
    cap: int
    seed: int = 0
    # None: the provider's default, 1.0 for Gemini, which Google recommends
    # keeping for 3.x. Must be > 0 so Best-of-N and Debate samples differ.
    temperature: float | None = None
    # Not 0: gemini-3.5-flash-lite cannot disable thinking (clients/gemini.py)
    thinking_budget: int = 1024
    max_output_tokens: int = 4096  # visible output per model call
    out_dir: Path = Path("runs")


@dataclass
class RunRecord:
    run_id: str
    trace_path: Path
    answer: str | None
    score: float
    solved: bool
    halted: bool
    ledger: dict[str, Any]


def run_id(topology: Topology, task: Task, cfg: RunConfig) -> str:
    return f"{topology.name}__{task.id.replace('/', '__')}__s{cfg.seed}"


def run_one(
    topology: Topology,
    task: Task,
    benchmark: Benchmark,
    llm: LLMClient,
    cfg: RunConfig,
) -> RunRecord:
    """Run, grade, and write the trace to cfg.out_dir/<run_id>.jsonl.

    A run with no answer scores 0 without being graded. Topologies with
    several candidate answers (Best-of-N, Debate) also get every
    candidate graded, as a diagnostic upper bound (oracle pass@N); that
    score never feeds back into which answer is picked.

    If the topology or grader raises, run_end is written with status
    "error" and the exception propagates: such a run is re-run, not
    scored.
    """
    rid = run_id(topology, task, cfg)
    path = cfg.out_dir / f"{rid}.jsonl"
    ledger = BudgetLedger(cfg.cap, run_id=rid)
    meta = RunMeta(
        run_id=rid,
        topology=topology.name,
        task_id=task.id,
        task_category=task.category,
        task_shape=task.shape,
        cap=cfg.cap,
        model=llm.model,
        seed=cfg.seed,
        temperature=cfg.temperature,
        thinking_budget=cfg.thinking_budget,
        extra={
            "topology_params": dataclasses.asdict(topology),
            "max_output_tokens": cfg.max_output_tokens,
        },
    )
    trace = TraceWriter(path, meta)
    client = MeteredClient(
        llm,
        ledger,
        agent_id="root",
        trace=trace,
        default_max_output_tokens=cfg.max_output_tokens,
        thinking_budget=cfg.thinking_budget,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )

    try:
        outcome = topology.run(task, client)
        grade = _grade(benchmark, task, outcome.answer)
        detail = dict(outcome.detail)
        if "candidates" in detail:
            scores: dict[str | None, float] = {}  # candidates often repeat
            for c in detail["candidates"]:
                if c not in scores:
                    scores[c] = _grade(benchmark, task, c).score
            detail["candidate_scores"] = [scores[c] for c in detail["candidates"]]
    except BaseException as e:
        trace.finish(
            answer=None,
            halted=False,
            ledger_summary=ledger.summary(),
            status="error",
            topology_detail={"error": f"{type(e).__name__}: {e}"},
        )
        raise

    summary = ledger.summary()
    trace.finish(
        answer=outcome.answer,
        halted=outcome.halted,
        ledger_summary=summary,
        score=grade.score,
        grader_detail=grade.detail,
        topology_detail=detail,
    )
    return RunRecord(
        run_id=rid,
        trace_path=path,
        answer=outcome.answer,
        score=grade.score,
        solved=grade.solved,
        halted=outcome.halted,
        ledger=summary,
    )


def _grade(benchmark: Benchmark, task: Task, answer: str | None) -> GradeResult:
    if answer is None:
        return GradeResult(0.0, False, {"error": "no answer"})
    return benchmark.grade(task, answer)
