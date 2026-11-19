"""The sandbox's isolation guarantees. Needs podman; skipped without it."""

import hashlib

from conftest import requires_podman

from dop.sandbox import run_python
from dop.tasks.classeval_pro import CONTAINERFILE

pytestmark = requires_podman


def run(code: str, timeout: float = 10.0):
    return run_python(
        {"main.py": code}, ["main.py"], containerfile=CONTAINERFILE, timeout=timeout
    )


def test_runs_code_and_reports_exit_code():
    r = run("import numpy; print(numpy.__version__); raise SystemExit(3)")
    assert r.returncode == 3 and r.stdout.strip() and not r.timed_out


def test_hang_is_killed_at_timeout():
    r = run("while True: pass", timeout=1)
    assert r.timed_out and r.returncode is None


def test_no_network():
    r = run("import socket; socket.create_connection(('1.1.1.1', 80), 2)")
    assert r.returncode != 0 and "unreachable" in r.stderr.lower()


def test_no_host_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-leak")
    r = run("import os; print('GEMINI_API_KEY' in os.environ, os.getuid())")
    assert r.stdout.split() == ["False", "65534"]


def test_root_filesystem_is_read_only():
    r = run("open('/usr/x', 'w')")
    assert "Read-only file system" in r.stderr


def test_calls_do_not_share_state():
    run("open('/sandbox/left.txt', 'w').write('x')")
    r = run("import os; print(os.path.exists('/sandbox/left.txt'))")
    assert r.stdout.strip() == "False"


def test_binary_files_round_trip():
    blob = bytes(range(256)) * 1000
    code = "import hashlib; print(hashlib.sha256(open('blob.bin', 'rb').read()).hexdigest())"
    r = run_python(
        {"blob.bin": blob, "main.py": code}, ["main.py"], containerfile=CONTAINERFILE
    )
    assert r.stdout.strip() == hashlib.sha256(blob).hexdigest()
