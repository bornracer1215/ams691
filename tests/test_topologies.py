"""The four topologies and run_one, against the mock client.

The mock's responder plays every agent: it tells them apart by their
system prompt and by the agent-specific text in their conversation.
"""

import re

from dop.agent import FORCE_FINAL
from dop.clients.base import ToolCall
from dop.clients.mock import MockClient
from dop.meter import read_trace
from dop.runner import RunConfig, run_one
from dop.tasks.base import GradeResult, Task, Tool, string_param
from dop.topologies.best_of_n import BestOfN
from dop.topologies.debate import Debate
from dop.topologies.single import Single
from dop.topologies.supervisor_worker import (
    SUPERVISOR_ROLE,
    WORKER_ROLE,
    SupervisorWorker,
)

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


def delegate(subtask):
    return ToolCall("delegate", {"subtask": subtask})


TASK = Task(
    id="toy/1",
    benchmark="toy",
    prompt="What is 6*7?",
    category="easy",
    shape="toy",
    data={"gold": "42"},
    tools=(CALC,),
)


class Toy:
    name = "toy"

    def __init__(self):
        self.graded = []

    def load(self):
        return [TASK]

    def grade(self, task, answer):
        self.graded.append(answer)
        ok = answer.strip() == task.data["gold"]
        return GradeResult(float(ok), ok)


def run(topology, responder, cap=200_000, tmp_path=None):
    bench = Toy()
    cfg = RunConfig(cap=cap, seed=1, max_output_tokens=64, out_dir=tmp_path)
    mock = MockClient(completion_tokens=20, responder=responder)
    rec = run_one(topology, TASK, bench, mock, cfg)
    return rec, bench, mock, read_trace(rec.trace_path)


def is_judge(messages):
    return "Select the most consistent response" in messages[1].content


def test_single(tmp_path):
    def responder(messages, i):
        return calc("6*7") if i == 0 else submit("42")

    rec, bench, _, trace = run(Single(), responder, tmp_path=tmp_path)
    assert (rec.answer, rec.score, rec.solved, rec.halted) == ("42", 1.0, True, False)
    end = trace[-1]
    assert end["kind"] == "run_end" and end["status"] == "ok"
    assert trace[0]["meta"]["topology"] == "single"
    assert [r["kind"] for r in trace].count("tool") == 1


def test_no_answer_scores_zero_without_grading(tmp_path):
    rec, bench, _, _ = run(Single(), lambda m, i: "thinking...", tmp_path=tmp_path)
    assert (rec.answer, rec.score, rec.halted) == (None, 0.0, True)
    assert bench.graded == []


def test_best_of_n_judge_picks_and_candidates_are_graded(tmp_path):
    def responder(messages, i):
        if is_judge(messages):
            return ("Response 1 agrees with the majority.", submit("1"))
        return submit(["41", "42", "42"][i % 3])

    rec, bench, mock, trace = run(BestOfN(n=3), responder, tmp_path=tmp_path)
    detail = trace[-1]["topology_detail"]
    assert sorted(detail["candidates"]) == ["41", "42", "42"]
    assert detail["usc"] == "picked"
    assert rec.answer == detail["candidates"][1]
    assert sorted(detail["candidate_scores"]) == [0.0, 1.0, 1.0]
    agents = {r["agent_id"] for r in trace if r["kind"] == "call"}
    assert agents == {"sample-0", "sample-1", "sample-2", "judge"}
    assert rec.ledger["spent"] <= rec.ledger["cap"]


def test_best_of_n_skips_the_judge_when_unanimous(tmp_path):
    rec, _, _, trace = run(BestOfN(n=3), lambda m, i: submit("42"), tmp_path=tmp_path)
    assert rec.answer == "42"
    assert trace[-1]["topology_detail"]["usc"] == "unanimous"
    assert "judge" not in {r["agent_id"] for r in trace if r["kind"] == "call"}


def test_supervisor_delegates_and_uses_the_report(tmp_path):
    def responder(messages, i):
        system, last = messages[0].content, messages[-1]
        if system.startswith(WORKER_ROLE):
            return submit("the product is 42")
        assert system.startswith(SUPERVISOR_ROLE)
        if last.role == "tool":
            return submit(re.search(r"\d+", last.content).group())
        return delegate("Compute 6*7.")

    rec, _, mock, trace = run(SupervisorWorker(), responder, tmp_path=tmp_path)
    assert rec.answer == "42"
    msgs = [r for r in trace if r["kind"] == "agent_message"]
    assert [(m["from"], m["to"]) for m in msgs] == [
        ("supervisor", "worker-0"),
        ("worker-0", "supervisor"),
    ]
    # the worker sees only the subtask, not the task
    worker_calls = [s for s in mock.seen if s[0].content.startswith(WORKER_ROLE)]
    assert worker_calls[0][1].content == "Compute 6*7."
    assert TASK.prompt not in worker_calls[0][1].content
    detail = trace[-1]["topology_detail"]
    assert set(detail["agents"]) == {"supervisor", "worker-0"}


def test_debate_runs_rounds_and_agents_see_each_other(tmp_path):
    def responder(messages, i):
        if is_judge(messages):
            return submit("0")
        if "Other agents answered" in messages[-1].content:
            return submit("42")  # everyone converges after one look
        return submit(str(40 + i % 3))

    rec, _, mock, trace = run(
        Debate(n_agents=3, rounds=2), responder, tmp_path=tmp_path
    )
    detail = trace[-1]["topology_detail"]
    assert detail["rounds_run"] == 3
    assert detail["candidates"] == ["42", "42", "42"]
    assert detail["usc"] == "unanimous"
    assert rec.answer == "42"
    # in round 1 an agent's prompt quotes the other two agents, not itself
    update = next(s for s in mock.seen if "Other agents" in s[-1].content)
    assert update[-1].content.count("answer:") == 2
    assert rec.ledger["spent"] <= rec.ledger["cap"]


def test_every_topology_stays_under_a_tight_cap(tmp_path):
    def stubborn(messages, i):  # never answers until forced
        if messages[-1].content == FORCE_FINAL:
            return submit("42")
        if messages[0].content.startswith(SUPERVISOR_ROLE) and i % 2:
            return delegate("Compute 6*7.")
        return calc("6*7")

    for topo in (Single(), BestOfN(n=3), SupervisorWorker(), Debate()):
        out = tmp_path / topo.name
        rec, _, _, trace = run(topo, stubborn, cap=20_000, tmp_path=out)
        assert rec.ledger["spent"] <= 20_000, topo.name
        assert rec.ledger["overshoot"] == 0, topo.name
        assert trace[-1]["status"] == "ok"


def test_crash_is_recorded_and_reraised(tmp_path):
    def responder(messages, i):
        raise RuntimeError("provider down")

    try:
        run(Single(), responder, tmp_path=tmp_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
    (path,) = tmp_path.glob("*.jsonl")
    end = read_trace(path)[-1]
    assert (end["kind"], end["status"]) == ("run_end", "error")
    assert "provider down" in end["topology_detail"]["error"]


def test_trace_rebuilds_every_prompt_exactly(tmp_path):
    def responder(messages, i):
        if is_judge(messages):
            return submit("0")
        if messages[-1].role == "tool" and messages[-1].name == "calc":
            return submit("42")
        return calc("6*7")

    _, _, mock, trace = run(Debate(), responder, tmp_path=tmp_path)
    convs: dict[str, list[dict]] = {}
    rebuilt = []
    for r in (r for r in trace if r["kind"] == "call"):
        conv = convs.setdefault(r["agent_id"], [])
        if r["input_from"] == 0:
            conv.clear()
        assert len(conv) == r["input_from"]
        conv += r["input"]
        rebuilt.append([m["content"] for m in conv])
    sent = [[m.content for m in seen] for seen in mock.seen]
    assert sorted(rebuilt) == sorted(sent)


def test_judge_answers_through_an_enum_and_falls_back_blind(tmp_path):
    def responder(messages, i):
        if is_judge(messages):
            return submit("7")  # out of range: the mock does not enforce the enum
        return submit(["41", "42", "43"][i % 3])

    rec, _, mock, trace = run(BestOfN(n=3), responder, tmp_path=tmp_path)
    detail = trace[-1]["topology_detail"]
    assert detail["usc"] == "fallback"
    assert detail["judge_answer"] == "7"
    # the fallback is the first candidate, never the best-graded one
    assert rec.answer == detail["candidates"][0]
    judge_offers = [
        o for o, s in zip(mock.offered, mock.seen, strict=True) if is_judge(s)
    ]
    assert judge_offers[0] == (["submit"], None)  # the judge has no other tools
