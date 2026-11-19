"""Download benchmark files once, pinned and hash-checked, into a local cache.

Every remote file a benchmark needs is named by an immutable URL (a commit
or revision hash, never a branch) plus its sha256, so every machine ends
up with byte-identical inputs. The cache is ~/.cache/dop unless
DOP_CACHE_DIR is set; nothing here belongs in git.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path

import certifi


def cache_root() -> Path:
    return Path(os.environ.get("DOP_CACHE_DIR") or Path.home() / ".cache" / "dop")


def fetch_pinned(url: str, sha256: str, dest: Path) -> Path:
    """Return `dest`, downloading `url` there first if it is missing.

    Raises if the bytes on disk or on the wire do not hash to `sha256`.
    """
    if dest.exists():
        raw = dest.read_bytes()
    else:
        # uv's standalone Pythons cannot find the CA bundle on NixOS.
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
            raw = resp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"sha256 mismatch for {url}: got {digest}")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(raw)
        tmp.replace(dest)  # atomic, so a killed download never looks complete
    return dest
