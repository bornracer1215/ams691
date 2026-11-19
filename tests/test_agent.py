"""The ReAct loop, against the mock client and a fake tool."""

from dop.agent import FORCE_FINAL, NUDGE, SUBMITTED, conversation, react
from dop.clients.base import ToolCall
from dop.clients.mock import MockClient
from dop.meter import BudgetLedger, MeteredClient
from dop.tasks.base import Tool, string_param

CALC = Tool(
    "calc",
    "Evaluates an arithmetic expression.",
    string_param("expr", "The expression."),
    lambda args: str(eval(args["expr"])),
)


def calc(expr):
    return ToolCall("calc", {"expr": expr})


def submit(answer):
    return ToolCall("submit", {"answer": answer})


def scripted(*replies):
    """A responder that returns `replies` in order, then repeats the last."""
    return lambda messages, i: replies[min(i, len(replies) - 1)]


def client(responder, cap=100_000, max_out=64):
    ledger = BudgetLedger(cap)
    mock = MockClient(completion_tokens=20, responder=responder)
    return MeteredClient(mock, ledger, default_max_output_tokens=max_out), mock


def test_tool_round_trip_then_submit():
    mc, mock = client(scripted(("Let me compute.", calc("6*7")), submit("42")))
    r = react(mc, conversation("", "What is 6*7?"), [CALC])

    assert (r.answer, r.source, r.steps) == ("42", "final", 2)
    assert r.messages[2].tool_call == calc("6*7")
    assert (r.messages[3].role, r.messages[3].name) == ("tool", "calc")
    assert r.messages[3].content == "42"
    assert mock.seen[1][-1].content == "42"  # the model saw the tool output
    # every call offers the task's tools plus submit, unrestricted
    assert mock.offered == [(["calc", "submit"], None)] * 2
    # submit gets a result too, so the conversation can be continued
    assert (r.messages[-1].name, r.messages[-1].content) == ("submit", SUBMITTED)


def test_reply_without_a_call_is_nudged():
    mc, mock = client(scripted("hmm", submit("x")))
    r = react(mc, conversation("", "q"), [CALC])
    assert r.answer == "x"
    assert mock.seen[1][-1].content == NUDGE


def test_unknown_function_and_tool_errors_go_back_to_the_agent():
    mc, mock = client(scripted(ToolCall("nope", {}), calc("1/0"), submit("?")))
    react(mc, conversation("", "q"), [CALC])
    assert mock.seen[1][-1].content == "error: no function named nope"
    assert mock.seen[2][-1].content.startswith("error: ZeroDivisionError")


def stubborn(messages, i):
    """Calls calc until forced, then submits."""
    return submit("forced") if messages[-1].content == FORCE_FINAL else calc("1+1")


def test_step_limit_forces_a_final_answer():
    mc, mock = client(stubborn)
    r = react(mc, conversation("", "q"), [CALC], max_steps=3)
    assert (r.answer, r.source, r.steps) == ("forced", "forced", 4)
    assert r.halted
    assert mock.offered[-1] == (["calc", "submit"], ["submit"])  # only submit allowed


def test_low_budget_forces_a_final_answer_within_the_cap():
    mc, _ = client(stubborn, cap=4_000)
    r = react(mc, conversation("", "q " * 200), [CALC])
    assert (r.answer, r.source) == ("forced", "forced")
    assert mc.ledger.spent <= mc.ledger.cap
    assert mc.ledger.refusals == 0


def test_no_answer_when_the_budget_cannot_cover_even_the_final_call():
    mc, _ = client(scripted(submit("never seen")), cap=50)
    r = react(mc, conversation("", "q " * 100))
    assert (r.answer, r.source, r.steps) == (None, "none", 0)
    assert mc.ledger.spent == 0


def test_forced_reply_without_submit_is_no_answer():
    mc, _ = client(scripted(calc("1"), "I think it is 1"))
    r = react(mc, conversation("", "q"), [CALC], max_steps=1)
    assert (r.answer, r.source) == (None, "none")


def test_tool_declarations_are_part_of_the_hold():
    mc, mock = client(scripted(submit("x")))
    bare = mc.estimate(conversation("", "q"))
    assert mc.estimate(conversation("", "q"), [CALC]) > bare


def test_sub_agents_get_distinct_reproducible_seeds():
    root = MeteredClient(MockClient(), BudgetLedger(1000), seed=7)
    a, b = root.for_agent("a"), root.for_agent("b")
    assert a.seed != b.seed
    assert a.seed == root.for_agent("a").seed
    unseeded = MeteredClient(MockClient(), BudgetLedger(1000))
    assert unseeded.for_agent("a").seed is None
