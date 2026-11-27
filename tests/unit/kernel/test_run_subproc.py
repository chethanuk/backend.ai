from __future__ import annotations

import os
import shlex
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, override

import pytest
import zmq
import zmq.asyncio

from ai.backend.kernel.base import BaseRunner

pytestmark = pytest.mark.skipif(not os.access("/bin/sh", os.X_OK), reason="needs /bin/sh")


class _StubRunner(BaseRunner):
    log_prefix = "test-runner"

    def __init__(self, shell: str, zctx: zmq.asyncio.Context) -> None:
        # The real __init__ requires /home/work and the container layout.
        self.subproc = None
        self.child_env = {"PATH": "/usr/bin:/bin", "SHELL": shell}
        # PUB drops messages when nobody subscribes, so output never blocks.
        self.outsock = zctx.socket(zmq.PUB)

    @override
    async def init_with_loop(self) -> None:
        pass

    @override
    async def build_heuristic(self) -> int:
        return 0

    @override
    async def execute_heuristic(self) -> int:
        return 0

    @override
    async def start_service(self, service_info: Any) -> Any:
        return None


@pytest.fixture
def zctx() -> Iterator[zmq.asyncio.Context]:
    ctx = zmq.asyncio.Context()
    yield ctx
    ctx.destroy(linger=0)


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    try:
        subprocess.run([str(path), "-c", "true"], check=False)
    except PermissionError:
        pytest.skip("tmp_path does not allow executing files")


async def test_batch_command_runs_under_configured_shell(
    tmp_path: Path, zctx: zmq.asyncio.Context
) -> None:
    # A wrapper that leaves a marker proves which shell ran the command,
    # independent of what /bin/sh is on the test machine.
    marker = tmp_path / "marker"
    wrapper = tmp_path / "custom-shell"
    _make_executable(
        wrapper, f'#!/bin/sh\necho "$0" > {shlex.quote(str(marker))}\nexec /bin/sh "$@"\n'
    )
    marker.unlink()

    retcode = await _StubRunner(str(wrapper), zctx).run_subproc("true", batch=True)

    assert retcode == 0
    assert marker.exists(), "command did not run under the configured SHELL"
    assert marker.read_text().strip() == str(wrapper)


@pytest.mark.parametrize(
    "shell_kind",
    ["empty", "missing", "not_executable", "directory", "relative", "unsearchable_dir"],
)
async def test_batch_command_falls_back_to_bin_sh_for_unusable_shell(
    tmp_path: Path,
    zctx: zmq.asyncio.Context,
    monkeypatch: pytest.MonkeyPatch,
    shell_kind: str,
) -> None:
    # An executable relative path must still be rejected, so it exits non-zero if used.
    monkeypatch.chdir(tmp_path)
    _make_executable(tmp_path / "relative-shell", "#!/bin/sh\nexit 7\n")
    not_executable = tmp_path / "not-executable"
    not_executable.write_text("#!/bin/sh\nexit 1\n")
    not_executable.chmod(0o644)
    locked = tmp_path / "locked"
    locked.mkdir()
    _make_executable(locked / "shell", "#!/bin/sh\nexit 7\n")
    shell = {
        "empty": "",
        "missing": str(tmp_path / "no-such-shell"),
        "not_executable": str(not_executable),
        "directory": str(tmp_path),
        "relative": "./relative-shell",
        "unsearchable_dir": str(locked / "shell"),
    }[shell_kind]
    marker = tmp_path / "marker"
    if shell_kind == "unsearchable_dir":
        if os.geteuid() == 0:
            pytest.skip("root bypasses directory permissions")
        locked.chmod(0)

    try:
        # With shell=True, $0 of the "-c" command is the executable that ran it.
        retcode = await _StubRunner(shell, zctx).run_subproc(
            f'echo "$0" > {shlex.quote(str(marker))}', batch=True
        )
    finally:
        locked.chmod(0o700)

    assert retcode == 0
    assert marker.read_text().strip() == "/bin/sh"
