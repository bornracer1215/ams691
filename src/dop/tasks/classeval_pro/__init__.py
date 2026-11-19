"""ClassEval-Pro: class-level code generation (Chen et al., AIware 2026).

We use a hand-picked 60-task subset, below.

Grading follows upstream common/evaluate.py: the answer is concatenated
with the task's test code and each test class in `test_classes` is run
separately. A task is `solved` when every test passes, which is upstream's
class-level success. `score` is the fraction of tests passed, so a run
that halts at the cap with half a class still gets credit.

Differences from upstream, all deliberate:
  - grading runs in a fresh podman container (sandbox.run_python), and
    inside it each test class runs in its own subprocess, so a hang or
    crash costs only that class;
  - timeout is 10s per test class instead of 5s, to absorb machine noise;
  - no `@staticmethod` insertion heuristic on the answer.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path

from ...sandbox import describe, run_python
from ..base import GradeResult, Task, Tool, string_param
from ..fetch import cache_root, fetch_pinned

CONTAINERFILE = Path(__file__).with_name("Containerfile")

COMMIT = "6ec785c7068e0e7d36374e2890ebf473f88dd306"
DATA_URL = (
    f"https://raw.githubusercontent.com/ian-Kappa/ClassEval-Pro/{COMMIT}/data.json"
)
DATA_SHA256 = "f6bcb748819f6f2e39f3d6635c76a8310f5cbee55f08154f9dd726522decb2e3"

# Hand-picked subset: 20 tasks per size bucket, by method count of the
# reference class (small <= 7, medium 8-10, large >= 11; roughly the
# 30th/77th percentiles of all 300). Within a bucket we picked for
# spread of method count and subject matter, and avoided near-duplicates:
# upstream reuses a source class in up to 8 tasks, e.g. the dependency-
# resolution reporter (starting/pinning/backtracking) family. Each bucket
# has 5 tasks whose tests use unittest.mock (about the benchmark-wide
# 23%), since mocks can pin implementation details like os.makedirs vs
# Path.mkdir and fail correct code; spreading them evenly keeps that
# noise out of the size comparison.
#
# Excluded on purpose: ClassEval_102, whose test expects HTTPStatus(422)
# to read "Unprocessable Entity"; Python 3.13 renamed it.
SUBSET: dict[str, list[int]] = {
    "small": [
        0,
        4,
        6,
        7,
        11,
        13,
        17,
        19,
        22,
        26,
        37,
        39,
        45,
        47,
        75,
        81,
        105,
        126,
        186,
        273,
    ],
    "medium": [
        15,
        31,
        32,
        36,
        66,
        80,
        92,
        108,
        120,
        151,
        158,
        166,
        177,
        179,
        193,
        210,
        213,
        235,
        248,
        257,
    ],
    "large": [
        21,
        46,
        58,
        69,
        72,
        74,
        76,
        93,
        139,
        145,
        146,
        152,
        173,
        199,
        205,
        237,
        253,
        278,
        289,
        299,
    ],
}

# Upstream's holistic prompt with its output format rewritten for an agent.
# Upstream's one-shot order ("DO NOT include any text outside the code
# block") stops agents from calling tools at all; the skeleton and the
# "NO comments, NO docstrings, NO doctests" rule, which keeps answers
# short, are unchanged. Upstream's version, for reference:
#   Please complete the class in following skeleton:
#   ## Skeleton
#   {skeleton}
#   ## Output Format
#   Output the complete code in a single ```python``` code block. DO NOT
#   include any text outside the code block. NO comments, NO docstrings,
#   NO doctests.
PROMPT_TEMPLATE = """Please complete the class in following skeleton:

## Skeleton
{skeleton}

## Output Format
You can run code with the python tool to check your implementation before answering. Your final answer is the complete code, including imports, in a single ```python``` code block. NO comments, NO docstrings, NO doctests."""

TEST_CLASS_TIMEOUT = 10.0
TOOL_TIMEOUT = 10.0

PYTHON_TOOL_DESCRIPTION = f"""Runs a Python 3.14 program and returns its exit code, stdout and stderr. Each call starts from scratch in a fresh sandbox, so include everything the program needs, e.g. your class and the code exercising it. numpy is installed; there is no network. Times out after {TOOL_TIMEOUT:g}s."""
_RESULT_TAG = "__DOP_RESULT__"

# Runs one unittest class from the module "solution" and prints a JSON
# summary on a tagged line, so test output on stdout cannot be mistaken
# for it.
_RUN_ONE = f"""
import json, sys, unittest
cls = sys.argv[1]
out = {{"run": 0, "failed": [], "import_error": None}}
try:
    import solution
    suite = unittest.TestLoader().loadTestsFromTestCase(getattr(solution, cls))
except BaseException as e:
    out["import_error"] = f"{{type(e).__name__}}: {{e}}"[:500]
else:
    r = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(suite)
    out["run"] = r.testsRun
    out["failed"] = [t.id().rsplit(".", 1)[-1] for t, _ in r.failures + r.errors]
print("{_RESULT_TAG}" + json.dumps(out))
"""

# Runs every graded test class through _run_one.py with its own timeout,
# all inside one container, and prints {class: outcome} on a tagged line.
_RUN_ALL = f"""
import json, subprocess, sys
timeout, classes = float(sys.argv[1]), sys.argv[2:]
results = {{}}
for cls in classes:
    try:
        p = subprocess.run([sys.executable, "_run_one.py", cls],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        results[cls] = {{"error": "timeout"}}
        continue
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("{_RESULT_TAG}")), None)
    if line is None:
        tail = p.stderr.strip().splitlines()[-1:] or ["no output"]
        results[cls] = {{"error": "runner crashed: " + tail[0][:300]}}
    else:
        results[cls] = json.loads(line[len("{_RESULT_TAG}"):])
print("{_RESULT_TAG}" + json.dumps(results))
"""


class ClassEvalPro:
    name = "classeval_pro"

    def load(self) -> list[Task]:
        dest = cache_root() / self.name / COMMIT[:12] / "data.json"
        raw = fetch_pinned(DATA_URL, DATA_SHA256, dest).read_bytes()
        records = {r["task_id"]: r for r in json.loads(raw)}
        tasks = []
        for bucket, ids in SUBSET.items():
            for n in ids:
                r = records[f"ClassEval_{n}"]
                tasks.append(
                    Task(
                        id=f"{self.name}/{n}",
                        benchmark=self.name,
                        prompt=PROMPT_TEMPLATE.format(skeleton=r["skeleton"]),
                        category=bucket,
                        shape="class_codegen",
                        data=r,
                        tools=(PYTHON_TOOL,),
                    )
                )
        return tasks

    def grade(self, task: Task, answer: str) -> GradeResult:
        code = extract_code(answer)
        test_counts = count_tests(task.data["test"], task.data["test_classes"])
        files = {
            "solution.py": code + "\n\n" + task.data["test"],
            "_run_one.py": _RUN_ONE,
            "_run_all.py": _RUN_ALL,
        }
        res = run_python(
            files,
            ["_run_all.py", str(TEST_CLASS_TIMEOUT), *test_counts],
            containerfile=CONTAINERFILE,
            timeout=TEST_CLASS_TIMEOUT * len(test_counts) + 10,
        )
        outcomes = _parse_outcomes(res)
        per_class = {
            cls: _summarize(outcomes.get(cls), n) for cls, n in test_counts.items()
        }

        total = sum(test_counts.values())
        passed = sum(c["passed"] for c in per_class.values())
        return GradeResult(
            score=passed / total if total else 0.0,
            solved=total > 0 and passed == total,
            detail={"passed": passed, "total": total, "test_classes": per_class},
        )


def _run_python_tool(args: dict) -> str:
    res = run_python(
        {"main.py": extract_code(args["code"])},
        ["main.py"],
        containerfile=CONTAINERFILE,
        timeout=TOOL_TIMEOUT,
    )
    return describe(res)


# The agents' only tool: no test code, so nothing hidden leaks to them.
PYTHON_TOOL = Tool(
    "python",
    PYTHON_TOOL_DESCRIPTION,
    string_param("code", "The complete Python program to run."),
    _run_python_tool,
)


def extract_code(answer: str) -> str:
    """The last fenced block (```python preferred), else the raw answer.

    Last, not first: an agent's final answer may quote
    an earlier draft before its revision.
    """
    for pattern in (r"```python\s*\n(.*?)```", r"```[^\n]*\n(.*?)```"):
        blocks = re.findall(pattern, answer, re.DOTALL)
        if blocks:
            return blocks[-1].strip()
    return answer.strip()


def count_tests(test_code: str, test_classes: list[str]) -> dict[str, int]:
    """Number of test_* methods per graded test class, read statically.

    Needed so an answer that fails to import still has a denominator.
    """
    with warnings.catch_warnings():  # upstream tests have invalid escapes like "\d"
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(test_code)
    defs = {
        n.name: sum(
            isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            and m.name.startswith("test")
            for m in n.body
        )
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
    }
    return {c: defs.get(c, 0) for c in test_classes}


def _parse_outcomes(res) -> dict:
    line = next(
        (ln for ln in reversed(res.stdout.splitlines()) if ln.startswith(_RESULT_TAG)),
        None,
    )
    return json.loads(line[len(_RESULT_TAG) :]) if line else {}


def _summarize(out: dict | None, n_tests: int) -> dict:
    if out is None:  # the whole grading run died, e.g. the memory limit
        return {"passed": 0, "total": n_tests, "error": "grading run crashed"}
    if out.get("error"):
        return {"passed": 0, "total": n_tests, "error": out["error"]}
    if out["import_error"]:
        return {"passed": 0, "total": n_tests, "error": out["import_error"]}
    return {
        "passed": max(out["run"] - len(out["failed"]), 0),
        "total": n_tests,
        "failed": out["failed"],
    }
