"""Multi-agent debate (Du et al., 2023): agents answer independently, then
in each further round see the others' latest answers and revise. The
final answers are combined by Universal Self-Consistency, the same
selection Best-of-N uses, so the two differ only by the discussion.

Each agent keeps its whole conversation across rounds, tool calls
included. Every round gets an equal share of the budget still left
(after the judge's share), split equally between agents; what a round
does not spend rolls over to later rounds. An agent that gives no
answer in a round keeps its previous one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent import AgentResult, conversation
from ..clients.base import Message
from ..meter import MeteredClient
from ..tasks.base import Task
from .base import Outcome, Topology, agent_summary, fan_out, usc_select

UPDATE_PROMPT = """Other agents answered the same task:

{others}

Using their answers as additional information, check your own answer and submit an updated final answer. You may use your other functions first."""


@dataclass
class Debate(Topology):
    name = "debate"

    n_agents: int = 2
    rounds: int = 2  # discussion rounds after the independent first answer
    judge_share: float = 0.15  # of the budget, held back for the USC call

    def run(self, task: Task, client: MeteredClient) -> Outcome:
        ids = [f"debater-{i}" for i in range(self.n_agents)]
        judge_reserve = int(client.ledger.remaining * self.judge_share)
        convs = [conversation("", task.prompt)] * self.n_agents
        answers: list[str | None] = [None] * self.n_agents
        history: list[list[AgentResult]] = []

        for rnd in range(self.rounds + 1):
            if rnd > 0:
                convs = [
                    [*r.messages, Message("user", _update(answers, i))]
                    for i, r in enumerate(history[-1])
                ]
            left = client.ledger.remaining - judge_reserve
            budget = max(left, 0) // (self.rounds + 1 - rnd)
            if budget < self.n_agents:  # nothing left to discuss with
                break
            results = fan_out(client, ids, budget, convs, task.tools)
            history.append(results)
            answers = [
                r.answer if r.answer is not None else prev
                for r, prev in zip(results, answers, strict=True)
            ]

        answer, usc = usc_select(task, client.for_agent("judge"), answers)
        judge_halted = usc.get("judge", {}).get("source", "final") != "final"
        return Outcome(
            answer,
            halted=judge_halted or any(r.halted for rnd in history for r in rnd),
            detail={
                "agents": {
                    a: [agent_summary(rnd[i]) for rnd in history]
                    for i, a in enumerate(ids)
                },
                "rounds_run": len(history),
                "candidates": answers,
                **usc,
            },
        )


def _update(answers: list[str | None], me: int) -> str:
    others = "\n\n".join(
        f"Agent {j}'s answer:\n{a}" if a is not None else f"Agent {j} gave no answer."
        for j, a in enumerate(answers)
        if j != me
    )
    return UPDATE_PROMPT.format(others=others)
