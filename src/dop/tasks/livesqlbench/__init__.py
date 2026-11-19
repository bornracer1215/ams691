"""LiveSQLBench-Base-Lite-SQLite: text-to-SQL over 18 SQLite databases.

BIRD team, https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite
(CC BY-SA 4.0). 270 tasks: 180 SELECT ("Query") and 90 data-management
ones. We use a hand-picked 60-task subset of the Query tasks, below.

Gold SQL and test cases are not public. Request them by emailing
bird.bench25@gmail.com with the subject
"[livesqlbench-base-lite-SQLite GT&Test Cases]", and save the JSONL you
get back at gold_path() (default ~/.cache/dop/livesqlbench/gold.jsonl,
or set DOP_LIVESQLBENCH_GOLD). Never commit it.

Grading reuses the upstream evaluator from bird-bench/mini_dev, fetched
at a pinned commit (manifest.json). Its code never runs in this process:
the grader ships it into a fresh podman container together with a
throwaway copy of the task's database, and a small driver there calls
upstream's run_preprocessing / run_evaluation_phase, which exec the gold
test cases (or upstream's default Soft-EX test) against the model's SQL.
The SQL itself is extracted from the answer by upstream's post_process.

TODO(agent prompt), once the gold SQL arrives: follow BIRD-Interact's
agentic mode (a-Interact, bird-bench/BIRD-Interact,
bird_interact_agent/experiments/utils/prompts.py). The prompt below is
upstream's one-shot one and does not work in the agent loop:
  - it is 15-18k tokens, almost all schema, column meanings and knowledge
    base, and re-sent on every turn, so re-reading it would dominate any
    token cap and penalise topologies with more agents;
  - it ends in an open ```sqlite fence, a completion-style cue that made
    Gemini reply with bare SQL instead of following the agent protocol.
Instead, give the agent the question, database name and table names,
and lookup tools next to `sql`: schema(table), column_meanings(table),
knowledge() (names only) and knowledge_definition(name). Same data,
fetched on demand. No ask/submit: their feedback comes from the hidden
tests.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ...sandbox import describe, run_python
from ..base import GradeResult, Task, Tool, string_param
from ..fetch import cache_root, fetch_pinned

HERE = Path(__file__).parent
CONTAINERFILE = HERE / "Containerfile"
MANIFEST = json.loads((HERE / "manifest.json").read_text())

GRADE_TIMEOUT = 60.0  # upstream's own per-query busy timeout is 30s
SQL_TOOL_TIMEOUT = 30.0
SQL_TOOL_ROWS = 50

SQL_TOOL_DESCRIPTION = f"""Runs SQLite statements against the task's database and returns the rows of the last one (at most {SQL_TOOL_ROWS}) or the error. Each call starts from a fresh copy of the database, so changes are discarded. Times out after {SQL_TOOL_TIMEOUT:g}s."""

# Hand-picked subset: 20 Query tasks per difficulty tier, spread over as
# many databases as each tier covers (Simple has Query tasks in 12 of the
# 18 databases, Moderate in all 18, Challenging in 16), with at most two
# per database except where a tier needs extras to reach 20 (Challenging:
# insider, mental, robot, vaccine). Within a tier we mixed high-level and
# low-level phrasings and order-sensitive and order-free results.
SUBSET: dict[str, list[str]] = {
    "simple": [
        "alien_2", "alien_8", "archeology_1", "credit_2", "credit_3",
        "crypto_1", "crypto_3", "cybermarket_2", "cybermarket_4", "disaster_1",
        "disaster_6", "gaming_4", "gaming_9", "museum_3", "museum_6",
        "news_7", "news_9", "polar_6", "solar_3", "virtual_3",
    ],
    "moderate": [
        "alien_4", "alien_10", "archeology_7", "credit_10", "cross_db_1",
        "cross_db_6", "crypto_6", "cybermarket_5", "disaster_2", "fake_2",
        "gaming_3", "insider_7", "mental_3", "museum_8", "news_2",
        "polar_1", "robot_4", "solar_5", "vaccine_4", "virtual_6",
    ],
    "challenging": [
        "archeology_8", "cross_db_8", "crypto_7", "cybermarket_8", "disaster_7",
        "fake_3", "gaming_8", "insider_1", "insider_10", "mental_2",
        "mental_5", "museum_1", "news_4", "polar_3", "robot_1",
        "robot_2", "solar_9", "vaccine_1", "vaccine_2", "virtual_7",
    ],
}  # fmt: skip

# Upstream's baseline prompt (live_sql_bench_sqlite/prompt/baseline.py),
# verbatim, including the trailing open ```sqlite fence it ends with.
# (\x20 is a trailing space upstream has; spelled out so editors keep it.)
PROMPT_TEMPLATE = """# Database Schema:
{schema}

# Column Meanings:
{column_meanings}

# External Knowledge:
{knowledge}

# User Task:
{query}

Generate the correct SQLite to handle the user task above:
(FORMAT: You should enclose your final SQLite in '```sqlite\\n[Your Generated SQLs]\\n```' in the end. Could use semicolon to separate multiple statements.)

# Your Generated SQL:\x20
```sqlite"""

_RESULT_TAG = "__DOP_RESULT__"

# Runs inside the container, next to upstream's evaluator files, db.sqlite
# (a copy of the task's database) and task.json. Mirrors upstream's
# evaluate_instance() minus its file paths, logging and database reset.
_GRADE = f"""
import contextlib, json, sys
from db_utils import get_connection_for_phase
from logger import NullLogger
from post_process import extract_sql_from_response
from single_instance_eval_sqlite import run_evaluation_phase, run_preprocessing
from test_utils import TEST_CASE_DEFAULT
from utils import split_field

task = json.load(open("task.json"))
out = {{"pred_sqls": [], "passed": 0, "total": 0, "error": None}}
try:
    with contextlib.redirect_stdout(sys.stderr):  # upstream prints freely
        pred_sqls = extract_sql_from_response(task["answer"])
        out["pred_sqls"] = pred_sqls
        test_cases = task.get("test_cases") or []
        kwargs = {{}}
        if not test_cases:
            test_cases = [TEST_CASE_DEFAULT]
            kwargs = {{"conditions": task.get("conditions") or {{}}}}
        out["total"] = len(test_cases)
        logger = NullLogger()
        conn = get_connection_for_phase("db.sqlite", logger)
        run_preprocessing(split_field(task, "preprocess_sql"), "db.sqlite", logger, conn)
        exec_err, timeout_err, _, passed, failed, msg = run_evaluation_phase(
            pred_sqls, split_field(task, "sol_sql"), "db.sqlite", test_cases,
            logger, conn, False, kwargs,
        )
        conn.close()
    out.update(passed=passed, failed=failed, execution_error=exec_err,
               timeout_error=timeout_err, error=msg[:2000] or None)
except Exception as e:
    out["error"] = f"{{type(e).__name__}}: {{e}}"[:2000]
print("{_RESULT_TAG}" + json.dumps(out))
"""


# Runs inside the container next to db.sqlite (a copy of the task's
# database) and query.sql. Statements are split with complete_statement,
# so a semicolon inside a string literal does not end one. No
# preprocessing: preprocess_sql is empty for every subset task.
_QUERY = f"""
import sqlite3, sys
rows_shown = {SQL_TOOL_ROWS}
sql, stmts, buf = open("query.sql").read(), [], ""
for piece in sql.split(";"):
    buf += piece + ";"
    if sqlite3.complete_statement(buf):
        if buf.strip(" \\t\\n;"):
            stmts.append(buf)
        buf = ""
if buf.strip(" \\t\\n;"):
    stmts.append(buf)
conn = sqlite3.connect("db.sqlite")
try:
    cur = None
    for s in stmts:
        cur = conn.execute(s)
    if cur is None or cur.description is None:
        print("OK, no rows returned.")
        sys.exit(0)
    rows = cur.fetchmany(rows_shown + 1)
except sqlite3.Error as e:
    print(f"error: {{e}}")
    sys.exit(1)
print(" | ".join(d[0] for d in cur.description))
for r in rows[:rows_shown]:
    print(" | ".join(map(str, r)))
more = len(rows) > rows_shown
print(f"(first {{rows_shown}} rows; there are more)" if more else f"({{len(rows)}} rows)")
"""


def gold_path() -> Path:
    env = os.environ.get("DOP_LIVESQLBENCH_GOLD")
    return Path(env) if env else cache_root() / "livesqlbench" / "gold.jsonl"


class LiveSQLBench:
    name = "livesqlbench"

    def load(self, require_gold: bool = True) -> list[Task]:
        """The 60 subset tasks. Without the gold file (require_gold=False)
        tasks still carry their prompts but cannot be graded."""
        rows = {
            r["instance_id"]: r
            for r in _jsonl(self._dataset_file("livesqlbench_data_sqlite.jsonl"))
        }
        gold = self._load_gold(require_gold)
        tasks = []
        for tier, ids in SUBSET.items():
            for iid in ids:
                row = {**rows[iid], **gold.get(iid, {})}
                tasks.append(
                    Task(
                        id=f"{self.name}/{iid}",
                        benchmark=self.name,
                        prompt=self._prompt(row),
                        category=tier,
                        shape="tool_feedback",
                        data=row,
                        tools=(self._sql_tool(row["selected_database"]),),
                    )
                )
        return tasks

    def grade(self, task: Task, answer: str) -> GradeResult:
        if "sol_sql" not in task.data or not task.data["sol_sql"]:
            raise RuntimeError(f"no gold SQL for {task.id}; see {gold_path()}")
        db = task.data["selected_database"]
        files: dict[str, str | bytes] = {
            Path(p).name: self._evaluator_file(p).read_bytes()
            for p in MANIFEST["evaluator"]["sha256"]
        }
        files["db.sqlite"] = self._dataset_file(
            f"{db}/{db}_template.sqlite"
        ).read_bytes()
        files["task.json"] = json.dumps(
            {
                "answer": answer,
                "preprocess_sql": task.data.get("preprocess_sql", []),
                "sol_sql": task.data["sol_sql"],
                "test_cases": task.data.get("test_cases", []),
                "conditions": task.data.get("conditions", {}),
            }
        )
        files["_grade.py"] = _GRADE
        res = run_python(
            files, ["_grade.py"], containerfile=CONTAINERFILE, timeout=GRADE_TIMEOUT
        )

        line = next(
            (
                ln
                for ln in reversed(res.stdout.splitlines())
                if ln.startswith(_RESULT_TAG)
            ),
            None,
        )
        if line is None:
            error = (
                "timeout"
                if res.timed_out
                else f"grading run crashed: {res.stderr[-500:]}"
            )
            return GradeResult(0.0, False, {"error": error})
        out = json.loads(line[len(_RESULT_TAG) :])
        # Upstream counts an instance as solved only if the predicted SQL
        # executes without error or timeout and every test case passes.
        crashed = "execution_error" not in out  # the driver raised early
        ran = not (crashed or out["execution_error"] or out["timeout_error"])
        passed = out["passed"] if ran else 0
        total = out["total"]
        return GradeResult(
            score=passed / total if total else 0.0,
            solved=ran and total > 0 and passed == total,
            detail=out,
        )

    def _prompt(self, row: dict) -> str:
        db = row["selected_database"]
        schema = self._dataset_file(f"{db}/{db}_schema.txt").read_text()
        meanings = json.loads(
            self._dataset_file(f"{db}/{db}_column_meaning_base.json").read_text()
        )
        # Upstream keys knowledge by name (later entries win) and shows
        # only these fields.
        kb = {
            k["knowledge"]: k for k in _jsonl(self._dataset_file(f"{db}/{db}_kb.jsonl"))
        }
        visible = ["id", "knowledge", "description", "definition"]
        knowledge = [{f: k[f] for f in visible if f in k} for k in kb.values()]
        return PROMPT_TEMPLATE.format(
            schema=schema,
            column_meanings=json.dumps(
                {k.lower(): v for k, v in meanings.items()}, indent=2
            ),
            knowledge=json.dumps(knowledge, indent=2),
            query=row["query"],
        )

    def _sql_tool(self, db: str) -> Tool:
        def run(args: dict) -> str:
            files = {
                "db.sqlite": self._dataset_file(
                    f"{db}/{db}_template.sqlite"
                ).read_bytes(),
                "query.sql": _strip_fence(args["query"]),
                "_query.py": _QUERY,
            }
            res = run_python(
                files,
                ["_query.py"],
                containerfile=CONTAINERFILE,
                timeout=SQL_TOOL_TIMEOUT,
            )
            return describe(res)

        return Tool(
            "sql",
            SQL_TOOL_DESCRIPTION,
            string_param("query", "SQLite statements, separated by semicolons."),
            run,
        )

    def _load_gold(self, required: bool) -> dict[str, dict]:
        path = gold_path()
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"LiveSQLBench gold SQL not found at {path}. Email "
                    "bird.bench25@gmail.com with subject "
                    "'[livesqlbench-base-lite-SQLite GT&Test Cases]', then save the "
                    "JSONL reply there (or set DOP_LIVESQLBENCH_GOLD)."
                )
            return {}
        fields = ("sol_sql", "external_knowledge", "test_cases")
        return {
            g["instance_id"]: {f: g.get(f, []) for f in fields} for g in _jsonl(path)
        }

    def _dataset_file(self, path: str) -> Path:
        return _pinned(MANIFEST["dataset"], path, cache_root() / self.name / "data")

    def _evaluator_file(self, path: str) -> Path:
        return _pinned(
            MANIFEST["evaluator"], path, cache_root() / self.name / "evaluator"
        )


def _pinned(source: dict, path: str, root: Path) -> Path:
    url = source["url"].format(rev=source["rev"], path=path)
    return fetch_pinned(url, source["sha256"][path], root / source["rev"][:12] / path)


def _strip_fence(sql: str) -> str:
    """The inside of a ```sqlite fence, if the agent wrote one."""
    m = re.search(r"```[^\n]*\n(.*?)(?:```|\Z)", sql, re.DOTALL)
    return m.group(1) if m else sql


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
