"""ClassEval-Pro loader and grader.

The first tests need no network. The rest download the pinned data.json
once (cached under ~/.cache/dop) and are skipped if that fails.
"""

import ast

import pytest
from conftest import requires_podman

from dop.tasks.classeval_pro import SUBSET, ClassEvalPro, count_tests, extract_code


def test_extract_code_prefers_last_python_block():
    answer = "draft:\n```python\nx = 1\n```\nfinal:\n```python\nx = 2\n```"
    assert extract_code(answer) == "x = 2"


def test_extract_code_falls_back_to_raw_text():
    assert extract_code("x = 3\n") == "x = 3"


def test_count_tests_only_counts_graded_classes():
    src = "class A:\n def test_a(self): pass\n def helper(self): pass\nclass B:\n def test_b(self): pass"
    assert count_tests(src, ["A"]) == {"A": 1}


def test_subset_is_20_per_bucket_without_duplicates():
    ids = [n for bucket in SUBSET.values() for n in bucket]
    assert {k: len(v) for k, v in SUBSET.items()} == {
        "small": 20,
        "medium": 20,
        "large": 20,
    }
    assert len(set(ids)) == 60
    assert 102 not in ids


@pytest.fixture(scope="module")
def bench_and_tasks():
    bench = ClassEvalPro()
    try:
        tasks = bench.load()
    except OSError as e:
        pytest.skip(f"ClassEval-Pro data unavailable: {e}")
    return bench, {t.id: t for t in tasks}


def _n_methods(task) -> int:
    cls = next(
        n
        for n in ast.parse(task.data["solution_code"]).body
        if isinstance(n, ast.ClassDef)
    )
    return sum(isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) for m in cls.body)


def test_buckets_match_method_counts(bench_and_tasks):
    _, tasks = bench_and_tasks
    bounds = {"small": (0, 7), "medium": (8, 10), "large": (11, 10**6)}
    for t in tasks.values():
        lo, hi = bounds[t.category]
        assert lo <= _n_methods(t) <= hi, t.id


@requires_podman
@pytest.mark.parametrize("task_id", ["classeval_pro/0", "classeval_pro/278"])
def test_reference_solution_is_solved(bench_and_tasks, task_id):
    bench, tasks = bench_and_tasks
    t = tasks[task_id]
    g = bench.grade(t, f"```python\n{t.data['solution_code']}\n```")
    assert g.solved and g.score == 1.0, g.detail


@requires_podman
def test_bad_answers_score_zero_without_raising(bench_and_tasks, monkeypatch):
    monkeypatch.setattr(
        "dop.tasks.classeval_pro.TEST_CLASS_TIMEOUT", 1.0
    )  # the hang hits every class
    bench, tasks = bench_and_tasks
    t = tasks["classeval_pro/0"]
    for answer in [
        "",
        "this is not python (",
        "raise SystemExit(1)",
        "while True: pass",
    ]:
        g = bench.grade(t, answer)
        assert g.score == 0.0 and not g.solved


@requires_podman
def test_python_tool_runs_code_and_reports_errors(bench_and_tasks):
    _, tasks = bench_and_tasks
    (tool,) = next(iter(tasks.values())).tools
    assert tool.name == "python"
    ok = tool.run({"code": "```python\nimport numpy\nprint(numpy.__name__)\n```"})
    assert ok.startswith("exit code 0") and "numpy" in ok
    bad = tool.run({"code": 'raise ValueError("boom")'})
    assert bad.startswith("exit code 1") and "ValueError: boom" in bad
    assert len(tool.run({"code": 'print("x" * 100_000)'})) < 5_000
