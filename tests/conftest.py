import shutil

import pytest

requires_podman = pytest.mark.skipif(
    shutil.which("podman") is None, reason="podman not installed"
)
