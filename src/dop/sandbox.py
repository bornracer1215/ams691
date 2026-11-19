"""Run untrusted Python in a throwaway podman container, via llm-sandbox.

Used by graders now and by the agents' tools later. Each call gets a fresh
container that is removed afterwards, so nothing carries over between
calls: no files, no processes, no installed packages, no database writes.

The container has no network, no host environment (so no API keys), a
read-only root filesystem with in-memory /sandbox and /tmp, memory and
process limits, no capabilities, and runs as nobody. Each benchmark brings
its own image, defined by a Containerfile next to its module in
dop.tasks; the harness itself never installs benchmark dependencies.

Timeouts are enforced inside the container by a small driver that runs
the program as a subprocess. llm-sandbox's own run() timeout only stops
waiting; the process keeps running until the session closes, so it is
kept as a backstop with some slack.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxTimeoutError

RUNTIME_CONFIGS = {
    "init": True,  # PID 1 forwards SIGTERM; without it every close waits 10s
    "network_mode": "none",
    "mem_limit": "512m",
    "pids_limit": 128,
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges"],
    "read_only": True,
    "user": "65534:65534",
    "mounts": [
        {"type": "tmpfs", "target": t, "size": "64m", "chown": True}
        for t in ("/sandbox", "/tmp")
    ],
}

_BACKSTOP_SLACK = 15.0  # seconds on top of the in-container timeout
_RESULT_TAG = "__DOP_SANDBOX__"

# Runs inside the container: write the files, run the program with a real
# timeout, print the outcome on one tagged line.
_DRIVER = """
import base64, json, pathlib, subprocess, sys
files, argv, timeout = json.loads(base64.b64decode({payload!r}))
for name, content in files.items():
    pathlib.Path(name).write_bytes(base64.b64decode(content))
try:
    p = subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, timeout=timeout
    )
    out = {{"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
           "timed_out": False}}
except subprocess.TimeoutExpired as e:
    def text(b):
        return b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
    out = {{"returncode": None, "stdout": text(e.stdout), "stderr": text(e.stderr),
           "timed_out": True}}
print({tag!r} + json.dumps(out))
"""


@dataclass
class ExecResult:
    returncode: int | None  # None when the program was killed on timeout
    stdout: str
    stderr: str
    timed_out: bool = False


def run_python(
    files: Mapping[str, str | bytes],
    argv: Sequence[str],
    *,
    containerfile: Path,
    timeout: float = 10.0,
) -> ExecResult:
    """Write `files` into a fresh container's /sandbox and run `python *argv`.

    The container runs the image built from `containerfile`.
    """
    image = _ensure_image(containerfile)
    encoded = {
        name: base64.b64encode(c.encode() if isinstance(c, str) else c).decode()
        for name, c in files.items()
    }
    payload = base64.b64encode(json.dumps([encoded, list(argv), timeout]).encode())
    code = _DRIVER.format(payload=payload.decode(), tag=_RESULT_TAG)
    try:
        with _session(image=image) as session:
            result = session.run(code, timeout=timeout + _BACKSTOP_SLACK)
    except SandboxTimeoutError:
        return ExecResult(None, "", "sandbox backstop timeout", timed_out=True)

    line = next(
        (
            ln
            for ln in reversed(result.stdout.splitlines())
            if ln.startswith(_RESULT_TAG)
        ),
        None,
    )
    if line is None:  # the driver itself died, e.g. killed by the memory limit
        return ExecResult(result.exit_code, result.stdout, result.stderr)
    return ExecResult(**json.loads(line[len(_RESULT_TAG) :]))


def _session(**source: str) -> SandboxSession:
    """A session on `image=...`, or on `dockerfile=...` to build one."""
    return SandboxSession(
        backend="podman",
        lang="python",
        skip_environment_setup=True,  # the image already has everything
        keep_template=True,  # never delete the shared image on close
        runtime_configs={**RUNTIME_CONFIGS, "mounts": list(RUNTIME_CONFIGS["mounts"])},
        **source,
    )


_image_lock = threading.Lock()
_images_ready: set[Path] = set()


def _ensure_image(containerfile: Path) -> str:
    """Build `containerfile`'s image once per process and return its tag.

    Podman's layer cache makes the rebuild fast when the Containerfile is
    unchanged, and picks up edits when it is not.
    """
    # llm-sandbox tags dockerfile builds as "sandbox-<lang>-<context dir name>",
    # so each benchmark's package directory names its image.
    image = f"sandbox-python-{containerfile.parent.name}"
    with _image_lock:
        if containerfile in _images_ready:
            return image
        if shutil.which("podman") is None:
            raise RuntimeError(
                "podman not found. Install it (https://podman.io/docs/installation); "
                "on macOS/Windows also run `podman machine init` once."
            )
        logging.getLogger("llm_sandbox").setLevel(logging.WARNING)
        try:
            with _session(dockerfile=str(containerfile)):
                pass
        except Exception as e:
            raise RuntimeError(
                "could not build or start the sandbox image. If podman is not "
                "running: `systemctl --user enable --now podman.socket` on Linux, "
                f"`podman machine start` on macOS/Windows. ({type(e).__name__}: {e})"
            ) from e
        _images_ready.add(containerfile)
        return image


# What an agent's tool call shows it: shared by every benchmark's tools so
# all of them bound their output the same way. Head and tail, because the
# error a program dies with is at the end of its output.
TOOL_OUTPUT_CHARS = 4000


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} chars cut ...]\n{text[-half:]}"


def describe(res: ExecResult, limit: int = TOOL_OUTPUT_CHARS) -> str:
    """An ExecResult as tool output: status line, then stdout and stderr."""
    status = "timed out" if res.timed_out else f"exit code {res.returncode}"
    parts = [status]
    for label, text in (("stdout", res.stdout), ("stderr", res.stderr)):
        if text.strip():
            parts.append(f"{label}:\n{clip(text, limit // 2)}")
    return "\n".join(parts)
