"""What every benchmark module provides.

Benchmarks are split by benchmark, not by stage: one module owns loading,
prompting, answer extraction, and grading for its benchmark, because
those all depend on the same task format. What they share is this
interface, so a runner can iterate tasks and score answers without
knowing which benchmark it is driving.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..clients.base import ToolSpec


@dataclass(frozen=True)
class Tool(ToolSpec):
    """A function an agent can call: declared to the model natively
    (name, description, JSON-schema parameters), and run here with the
    call's arguments. Implementations bound their own output size, so
    every topology sees the same tool output."""

    run: Callable[[dict[str, Any]], str] = field(repr=False)


def string_param(name: str, description: str) -> dict[str, Any]:
    """Parameters schema for a tool taking one required string."""
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
    }


@dataclass(frozen=True)
class Task:
    id: str  # globally unique, "<benchmark>/<native id>"
    benchmark: str
    prompt: str  # problem statement handed to the agents
    category: str  # stratum, recorded in RunMeta.task_category
    shape: str  # RunMeta.task_shape
    data: dict[str, Any] = field(repr=False)  # raw benchmark record, for the grader
    tools: tuple[Tool, ...] = field(default=(), repr=False)  # given to every agent


@dataclass
class GradeResult:
    score: float  # in [0, 1]; partial credit, so halted runs are not zeroed
    solved: bool  # the benchmark's own all-or-nothing criterion
    detail: dict[str, Any] = field(default_factory=dict)  # -> run_end grader_detail


class Benchmark(Protocol):
    name: str

    def load(self) -> list[Task]: ...

    def grade(self, task: Task, answer: str) -> GradeResult:
        """Score a topology's final answer. Must not raise on bad
        answers (empty, unparsable, hanging); those score 0."""
        ...
