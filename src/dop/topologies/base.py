"""What every topology provides, plus Universal Self-Consistency, the
selection step Best-of-N and Debate share.

A topology takes a task and a MeteredClient bound to the run's ledger and
returns one answer. It builds no ledgers, traces or graders: sub-agents
come from `client.for_agent()`, budget slices from
`client.ledger.allocate()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..agent import AgentResult, conversation, react, submit_tool
from ..clients.base import Message
from ..meter import MeteredClient
from ..tasks.base import Task, Tool


@dataclass
class Outcome:
    answer: str | None  # None scores 0 without grading
    halted: bool  # some agent ran out of budget or steps
    detail: dict[str, Any] = field(default_factory=dict)  # -> run_end


class Topology(ABC):
    name: ClassVar[str]

    @abstractmethod
    def run(self, task: Task, client: MeteredClient) -> Outcome: ...


def agent_summary(r: AgentResult) -> dict[str, Any]:
    return {"source": r.source, "steps": r.steps}


def fan_out(
    client: MeteredClient,
    agent_ids: list[str],
    budget: int,
    conversations: list[list[Message]],
    tools: Sequence[Tool],
) -> list[AgentResult]:
    """Run agent i on conversations[i], all in parallel, each on its own
    equal slice of `budget`. Slices are carved before any agent starts,
    so no agent's share depends on thread scheduling; what an agent does
    not spend returns to the parent ledger."""
    each = budget // len(agent_ids)
    slices = [client.ledger.allocate(each, agent_id=a) for a in agent_ids]

    def one(i: int) -> AgentResult:
        try:
            agent = client.for_agent(agent_ids[i], ledger=slices[i])
            return react(agent, conversations[i], tools)
        finally:
            slices[i].settle()

    with ThreadPoolExecutor(len(agent_ids)) as pool:
        return list(pool.map(one, range(len(agent_ids))))


# Universal Self-Consistency (Chen et al., 2023): the model itself picks
# the candidate most consistent with the others. Their prompt, adapted to
# answering through submit, whose argument is constrained to the response
# numbers.
USC_PROMPT = """I have generated the following responses to the task below.

# Task
{task}

# Responses
{responses}

Evaluate these responses. Select the most consistent response based on majority consensus, and submit its number."""


def usc_select(
    task: Task, client: MeteredClient, candidates: list[str | None]
) -> tuple[str | None, dict[str, Any]]:
    """Pick one of `candidates` (None entries are agents that gave no
    answer). Skips the model call when there is nothing to choose
    between, and falls back to the first candidate if the judge gives no
    valid choice: never to the best-graded one, as selection must not see
    the hidden tests."""
    options = [c for c in candidates if c is not None]
    if not options:
        return None, {"usc": "no_candidates"}
    if len(set(options)) == 1:
        return options[0], {"usc": "unanimous"}

    responses = "\n\n".join(f"Response {i}:\n{c}" for i, c in enumerate(options))
    prompt = USC_PROMPT.format(task=task.prompt, responses=responses)
    numbers = [str(i) for i in range(len(options))]
    submit = submit_tool({"type": "string", "enum": numbers})
    judge = react(client, conversation("", prompt), submit=submit, max_steps=2)
    detail: dict[str, Any] = {"judge": agent_summary(judge)}
    if judge.answer not in numbers:  # no submit, or the enum was not enforced
        return options[0], {**detail, "usc": "fallback", "judge_answer": judge.answer}
    pick = int(judge.answer)
    return options[pick], {**detail, "usc": "picked", "pick": pick}
