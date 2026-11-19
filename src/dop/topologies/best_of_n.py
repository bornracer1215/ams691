"""N independent agents, then Universal Self-Consistency picks one answer.

The control for Debate: the same agents and selection step with no
discussion between them (cf. Choi et al., "Debate or Vote", 2025).
Diversity comes from temperature > 0 and the distinct per-agent seeds
`for_agent()` derives.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent import conversation
from ..meter import MeteredClient
from ..tasks.base import Task
from .base import Outcome, Topology, agent_summary, fan_out, usc_select


@dataclass
class BestOfN(Topology):
    name = "best_of_n"

    n: int = 3
    judge_share: float = 0.15  # of the budget, held back for the USC call

    def run(self, task: Task, client: MeteredClient) -> Outcome:
        ids = [f"sample-{i}" for i in range(self.n)]
        budget = int(client.ledger.remaining * (1 - self.judge_share))
        start = conversation("", task.prompt)
        results = fan_out(client, ids, budget, [start] * self.n, task.tools)

        candidates = [r.answer for r in results]
        answer, usc = usc_select(task, client.for_agent("judge"), candidates)
        judge_halted = usc.get("judge", {}).get("source", "final") != "final"
        return Outcome(
            answer,
            halted=judge_halted or any(r.halted for r in results),
            detail={
                "agents": {
                    a: agent_summary(r) for a, r in zip(ids, results, strict=True)
                },
                "candidates": candidates,
                **usc,
            },
        )
