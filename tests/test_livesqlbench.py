"""LiveSQLBench loader and grader.

The real gold SQL is emailed, not downloaded, so grading is checked with
synthetic gold built from the task's own database: the mechanics (SQL
extraction, upstream's Soft-EX default test, exec'd custom test cases,
error handling) are the same. Downloads the pinned data once (~40 MB).
"""

import dataclasses
import sqlite3

import pytest
from conftest import requires_podman

from dop.tasks.livesqlbench import SUBSET, LiveSQLBench, gold_path


@pytest.fixture(scope="module")
def bench_and_task():
    bench = LiveSQLBench()
    try:
        tasks = bench.load(require_gold=False)
    except OSError as e:
        pytest.skip(f"LiveSQLBench data unavailable: {e}")
    return bench, tasks


def test_subset_is_20_per_tier_without_duplicates(bench_and_task):
    _, tasks = bench_and_task
    assert {k: len(v) for k, v in SUBSET.items()} == {
        "simple": 20,
        "moderate": 20,
        "challenging": 20,
    }
    assert len({t.id for t in tasks}) == 60
    for t in tasks:
        assert t.data["category"] == "Query"
        assert t.data["difficulty_tier"].lower() == t.category


def test_prompt_matches_upstream_shape(bench_and_task):
    _, tasks = bench_and_task
    p = tasks[0].prompt
    assert p.startswith("# Database Schema:\n")
    assert tasks[0].data["query"] in p
    assert p.endswith("# Your Generated SQL: \n```sqlite")


def test_missing_gold_explains_how_to_get_it(monkeypatch, tmp_path):
    monkeypatch.setenv("DOP_LIVESQLBENCH_GOLD", str(tmp_path / "nope.jsonl"))
    with pytest.raises(FileNotFoundError, match="bird.bench25@gmail.com"):
        LiveSQLBench().load()
    assert gold_path() == tmp_path / "nope.jsonl"


def _with_gold(bench, task, test_cases=()):
    db = task.data["selected_database"]
    path = bench._dataset_file(f"{db}/{db}_template.sqlite")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' LIMIT 1"
    ).fetchone()[0]
    gold = f'SELECT COUNT(*) FROM "{table}"'
    data = {**task.data, "sol_sql": [gold], "test_cases": list(test_cases)}
    return dataclasses.replace(task, data=data), gold, table


@requires_podman
def test_correct_sql_is_solved_in_either_answer_format(bench_and_task):
    bench, tasks = bench_and_task
    task, gold, _ = _with_gold(bench, tasks[0])
    for answer in [f"```sqlite\n{gold};\n```", f"{gold};\n```"]:
        g = bench.grade(task, answer)
        assert g.solved and g.score == 1.0, g.detail


@requires_podman
def test_wrong_broken_and_empty_answers_score_zero(bench_and_task):
    bench, tasks = bench_and_task
    task, _, table = _with_gold(bench, tasks[0])
    for answer in [
        f'```sqlite\nSELECT COUNT(*) + 1 FROM "{table}";\n```',
        "```sqlite\nSELEC nope;\n```",
        "",
    ]:
        g = bench.grade(task, answer)
        assert g.score == 0.0 and not g.solved, (answer, g.detail)


@requires_podman
def test_custom_test_cases_are_executed(bench_and_task):
    bench, tasks = bench_and_task
    case = (
        "def test_case(pred_sqls, sol_sqls, db_path, conn):\n"
        "    r, _ = perform_query_on_sqlite_databases(pred_sqls[0], db_path, conn=conn)\n"
        "    assert r[0][0] < 0\n"
    )
    task, gold, _ = _with_gold(bench, tasks[0], [case])
    g = bench.grade(task, f"```sqlite\n{gold};\n```")
    assert not g.solved and g.detail["failed"] == ["test_1"]


@requires_podman
def test_sql_tool_returns_rows_errors_and_discards_writes(bench_and_task):
    _, tasks = bench_and_task
    (tool,) = tasks[0].tools
    assert tool.name == "sql"
    out = tool.run({"query": "```sqlite\nSELECT 1 AS a, 'x;y' AS b;\n```"})
    assert "a | b\n1 | x;y\n(1 rows)" in out
    assert "error:" in tool.run({"query": "SELECT * FROM no_such_table"})
    tool.run({"query": "CREATE TABLE scratch(x)"})
    assert "no such table" in tool.run({"query": "SELECT * FROM scratch"})
