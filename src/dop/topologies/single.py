"""One agent with the task's tools: the baseline every other topology
has to beat at the same budget."""

from __future__ import annotations

from dataclasses import dataclass

from ..agent import conversation, react
from ..meter import MeteredClient
from ..tasks.base import Task
from .base import Outcome, Topology, agent_summary


@dataclass
class Single(Topology):
    name = "single"

    def run(self, task: Task, client: MeteredClient) -> Outcome:
        agent = client.for_agent("agent")
        r = react(agent, conversation("", task.prompt), task.tools)
        return Outcome(r.answer, r.halted, {"agents": {"agent": agent_summary(r)}})
