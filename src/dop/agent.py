"""The ReAct loop that every agent in every topology runs.

Topologies differ only in how many agents they run, what each is told
(its role text) and which tools it gets. The protocol below, the step
limit and the forced final answer are the same for every agent, so they
cannot differ by topology.

Protocol: tools are declared to the model natively and every reply must
call exactly one (the client sets function calling to "any"). An agent
answers by calling `submit`, which ends its work. Tools described only
as text in the prompt made Gemini attempt native calls anyway and fail
(MALFORMED_FUNCTION_CALL), so there is no text protocol.

Forced final answer: before each step the loop checks that the budget
covers that step plus one more. If it does not, or the agent has used
max_steps, it is told to answer now and may call only `submit`. An agent
that never submits has no answer, and its run scores 0.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .clients.base import Message, ToolCall, ToolSpec
from .meter import BudgetExhausted, MeteredClient
from .tasks.base import Tool

MAX_STEPS = 20

# Headroom kept for the forced final answer beyond 2x the next step's
# reservation: the step's reply and tool output both land in the final
# call's prompt.
FINAL_MARGIN = 2048

PROTOCOL = """You work in turns. In each turn, call exactly one of your functions; its result comes back in the next message. When you are done, call submit with your final answer, which ends your work.

Only the submitted answer is kept, so it must be complete on its own and follow any output format the task asks for. If your budget runs low you will be asked to submit immediately."""

NUDGE = "Your reply did not call a function. Call exactly one of your functions."

FORCE_FINAL = "You are out of budget. Call submit now with your final answer."

SUBMITTED = "Answer received."  # submit's result, if the conversation goes on


def submit_tool(answer: dict[str, Any] | None = None) -> ToolSpec:
    """The function an agent answers with. `answer` is the JSON schema of
    its one argument; a plain string unless the caller constrains it, as
    the USC judge does with an enum of response numbers."""
    return ToolSpec(
        "submit",
        "Submit your final answer. This ends your work.",
        {
            "type": "object",
            "properties": {
                "answer": answer
                or {"type": "string", "description": "Your complete final answer."}
            },
            "required": ["answer"],
        },
    )


SUBMIT = submit_tool()


@dataclass
class AgentResult:
    answer: str | None  # None: the agent never submitted one
    source: str  # "final" | "forced" | "none"
    steps: int  # model calls made
    messages: list[Message]  # the whole conversation, to continue it later

    @property
    def halted(self) -> bool:
        """Ran out of budget or steps instead of submitting on its own."""
        return self.source != "final"


def system_prompt(role: str) -> str:
    return f"{role}\n\n{PROTOCOL}" if role else PROTOCOL


def conversation(role: str, prompt: str) -> list[Message]:
    """A fresh conversation: system prompt, then the task as the user turn.

    Task prompts must not contradict the protocol: a one-shot order like
    "DO NOT include any text outside the code block" in the user turn
    wins over the system prompt, and the agent never calls a tool."""
    return [Message("system", system_prompt(role)), Message("user", prompt)]


def react(
    client: MeteredClient,
    messages: Sequence[Message],
    tools: Sequence[Tool] = (),
    *,
    submit: ToolSpec = SUBMIT,
    max_steps: int = MAX_STEPS,
) -> AgentResult:
    """Run one agent until it submits, runs out of steps, or runs out of
    budget. `messages` is a conversation() or a previous result's
    messages plus a new user turn. Never raises BudgetExhausted."""
    messages = list(messages)
    declared: list[ToolSpec] = [*tools, submit]
    by_name = {t.name: t for t in tools}
    steps = 0
    try:
        while steps < max_steps:
            need = client.estimate(messages, declared)
            if client.ledger.remaining < 2 * need + FINAL_MARGIN:
                break
            resp = client.generate(messages, tools=declared, step=f"react:{steps}")
            steps += 1
            messages.append(resp.message())
            call = resp.tool_call
            if call is None:
                messages.append(Message("user", NUDGE))
            elif call.name == submit.name:
                messages.append(Message("tool", SUBMITTED, name=submit.name))
                return AgentResult(_answer(call), "final", steps, messages)
            elif call.name not in by_name:
                error = f"error: no function named {call.name}"
                messages.append(Message("tool", error, name=call.name))
            else:
                output = _call(client, by_name[call.name], call)
                messages.append(Message("tool", output, name=call.name))

        messages.append(Message("user", FORCE_FINAL))
        resp = client.generate(
            messages, tools=declared, tool_choice=[submit.name], step="react:forced"
        )
        steps += 1
        messages.append(resp.message())
        call = resp.tool_call
        if call is not None and call.name == submit.name:
            messages.append(Message("tool", SUBMITTED, name=submit.name))
            return AgentResult(_answer(call), "forced", steps, messages)
        return AgentResult(None, "none", steps, messages)
    except BudgetExhausted as e:
        if client.trace is not None:
            client.trace.halt(
                agent_id=client.agent_id, requested=e.requested, remaining=e.remaining
            )
        return AgentResult(None, "none", steps, messages)


def _answer(call: ToolCall) -> str | None:
    answer = call.args.get("answer")
    return None if answer is None else str(answer)


def _call(client: MeteredClient, tool: Tool, call: ToolCall) -> str:
    t0 = time.monotonic()
    try:
        output, ok = tool.run(call.args), True
    except Exception as e:  # a broken call is the agent's problem to work around
        output, ok = f"error: {type(e).__name__}: {e}", False
    if client.trace is not None:
        client.trace.tool(
            agent_id=client.agent_id,
            tool=tool.name,
            ok=ok,
            output_chars=len(output),
            latency_s=round(time.monotonic() - t0, 3),
        )
    return output
