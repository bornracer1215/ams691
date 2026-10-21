"""Run traces: the input to D3 and D4.

One JSONL file per run. Every model call is one record, tagged with the
agent that made it, so post-hoc analysis can answer "how much of the
Supervisor-Worker budget went to coordination messages rather than task
work" without re-running anything.

Design the schema before the pilot. Changing it afterward means
re-running every configuration.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ledger import Usage

SCHEMA_VERSION = 1


@dataclass
class RunMeta:
    run_id: str
    architecture: str          # single | best_of_n | supervisor_worker | debate
    task_id: str
    task_category: str         # coding | reasoning | database
    task_shape: str            # decomposable | sequential | tool_feedback
    cap: int
    model: str
    seed: int | None = None
    temperature: float = 0.0
    thinking_budget: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class TraceWriter:
    def __init__(self, path: str | Path, meta: RunMeta) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.time()
        self._fh = self.path.open("w", encoding="utf-8")
        self._emit(
            "run_start",
            {"schema_version": SCHEMA_VERSION, "meta": meta.__dict__},
        )

    # -- events --------------------------------------------------------

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            rec = {
                "seq": self._seq,
                "kind": kind,
                "t": round(time.time() - self._t0, 4),
                **payload,
            }
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()

    def call(
        self,
        *,
        agent_id: str,
        step: str,
        model: str,
        prompt_est: int,
        reserved: int,
        usage: Usage,
        spent_after: int,
        cap: int,
        finish_reason: str,
        n_messages: int,
    ) -> None:
        self._emit(
            "call",
            {
                "agent_id": agent_id,
                "step": step,
                "model": model,
                "prompt_est": prompt_est,
                "reserved": reserved,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "thinking_tokens": usage.thinking_tokens,
                "call_total": usage.total,
                "spent_after": spent_after,
                "remaining_after": cap - spent_after,
                "finish_reason": finish_reason,
                "n_messages": n_messages,
            },
        )

    def tool(
        self,
        *,
        agent_id: str,
        tool: str,
        ok: bool,
        output_chars: int,
        truncated_to: int | None = None,
        latency_s: float | None = None,
    ) -> None:
        """Tool output enters the budget as prompt tokens on the NEXT call.
        Logged separately so 'budget spent re-reading SQL errors' is
        measurable."""
        self._emit(
            "tool",
            {
                "agent_id": agent_id,
                "tool": tool,
                "ok": ok,
                "output_chars": output_chars,
                "truncated_to": truncated_to,
                "latency_s": latency_s,
            },
        )

    def message(self, *, frm: str, to: str, chars: int, kind: str = "delegate") -> None:
        """Agent-to-agent message. Charged as prompt tokens when it lands in
        a context; recorded here so coordination overhead is attributable."""
        self._emit(
            "agent_message",
            {"from": frm, "to": to, "chars": chars, "msg_kind": kind},
        )

    def halt(self, *, agent_id: str, requested: int, remaining: int) -> None:
        self._emit(
            "halt",
            {"agent_id": agent_id, "requested": requested, "remaining": remaining,
             "reason": "budget_exhausted"},
        )

    def finish(
        self,
        *,
        answer: str | None,
        halted: bool,
        ledger_summary: dict[str, Any],
        score: float | None = None,
        grader_detail: dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            "run_end",
            {
                "answer": answer,
                "halted": halted,
                "score": score,
                "grader_detail": grader_detail or {},
                "ledger": ledger_summary,
            },
        )
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
