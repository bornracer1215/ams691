"""A supervisor that can hand self-contained subtasks to workers.

The supervisor is an ordinary agent with one extra tool, `delegate`.
Each call starts a fresh worker that sees only the subtask text, has the
task's own tools, and returns its final answer as the tool's output. So
all coordination is visible in the trace: the subtask and the report are
agent_message records, and the report enters the supervisor's budget as
prompt tokens on its next call.

The supervisor keeps the task's tools too, so it can check the pieces
it assembles. One delegation per turn, so workers run one at a time.
Each worker gets a slice of at most `worker_share` of the cap, and never
more than half of what is left, so the supervisor can still answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agent import conversation, react
from ..meter import MeteredClient
from ..tasks.base import Task, Tool, string_param
from .base import Outcome, Topology, agent_summary

SUPERVISOR_ROLE = """You are a supervisor. You can hand self-contained subtasks to worker agents with the delegate tool, and combine what they return into your final answer. Workers spend from the same token budget as you."""

DELEGATE_DESCRIPTION = """Starts a new worker agent on a subtask and returns what the worker submits. The worker sees only the subtask text, not the task or your conversation, so include everything it needs. Workers have the same other functions you have, except delegate."""

WORKER_ROLE = """You are a worker agent. A supervisor gave you the subtask below. Complete it and submit your result; the supervisor sees only what you submit."""

NO_REPORT = "The worker ran out of budget without answering."


@dataclass
class SupervisorWorker(Topology):
    name = "supervisor_worker"

    worker_share: float = 0.25  # of the cap, at most, per delegation

    def run(self, task: Task, client: MeteredClient) -> Outcome:
        supervisor = client.for_agent("supervisor")
        workers: dict[str, dict[str, Any]] = {}

        def delegate(args: dict) -> str:
            subtask = args["subtask"]
            agent_id = f"worker-{len(workers)}"
            remaining = client.ledger.remaining
            amount = min(int(client.ledger.cap * self.worker_share), remaining // 2)
            if amount < 2 * client.default_max_output_tokens:
                return "error: not enough budget left to start a worker."
            if client.trace is not None:
                client.trace.message(frm="supervisor", to=agent_id, chars=len(subtask))
            sub = client.ledger.allocate(amount, agent_id=agent_id)
            try:
                worker = client.for_agent(agent_id, ledger=sub)
                r = react(worker, conversation(WORKER_ROLE, subtask), task.tools)
            finally:
                sub.settle()
            workers[agent_id] = agent_summary(r)
            report = r.answer if r.answer is not None else NO_REPORT
            if client.trace is not None:
                client.trace.message(
                    frm=agent_id, to="supervisor", chars=len(report), kind="report"
                )
            return report

        delegate_tool = Tool(
            "delegate",
            DELEGATE_DESCRIPTION,
            string_param("subtask", "Everything the worker needs to do its part."),
            delegate,
        )
        tools = (*task.tools, delegate_tool)
        r = react(supervisor, conversation(SUPERVISOR_ROLE, task.prompt), tools)
        return Outcome(
            r.answer,
            halted=r.halted or any(w["source"] != "final" for w in workers.values()),
            detail={"agents": {"supervisor": agent_summary(r), **workers}},
        )
