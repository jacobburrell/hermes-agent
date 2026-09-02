"""Regression tests: the WhatsApp stale-bridge cleanup must never kill a stranger.

The bridge records its PID in ``bridge.pid``. On the next start the gateway
SIGTERMs that PID to reap an orphaned bridge. The original code checked only
that the PID was *alive* — but once the bridge exits and is reaped the kernel
can recycle its number onto an unrelated process. Because the WhatsApp bridge
crash-loops, this cleanup ran constantly, and a recycled PID that had landed on
the user's browser main process got SIGTERMed, closing the browser at irregular
intervals (no crash, no coredump — a clean kill of a stranger).

These tests prove the identity guard: a PID is only signalled when it is still
our bridge (kernel start time matches and its command line names node + this
session; legacy pidfiles retain the command-line check). A recycled PID is
left alone.
"""

import subprocess
import sys
import time

import pytest

from hermes_constants import find_node_executable
from plugins.platforms.whatsapp.adapter import (
    _bridge_pid_is_ours,
    _kill_stale_bridge_by_pidfile,
    _write_bridge_pidfile,
)
from gateway.status import get_process_start_time


def _spawn_sleeper(*extra_argv) -> subprocess.Popen:
    """Spawn a real, short-lived process; optional extra argv shapes its cmdline."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)", *extra_argv]
    )


def _spawn_node_bridge(session_path) -> subprocess.Popen:
    """Spawn a real Node-shaped bridge command line for identity testing."""
    node = find_node_executable("node")
    if not node:
        pytest.skip("Node is required for the live bridge PID identity check")
    return subprocess.Popen(
        [
            node,
            "-e",
            "setTimeout(() => {}, 1000)",
            "bridge.js",
            "--session",
            str(session_path),
        ]
    )


def _wait_dead(proc: subprocess.Popen, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return False


class TestWriteAndRoundTrip:
    def test_pidfile_records_pid_and_start_time(self, tmp_path):
        proc = _spawn_sleeper()
        try:
            _write_bridge_pidfile(tmp_path, proc.pid)
            lines = (tmp_path / "bridge.pid").read_text(encoding="utf-8").split("\n")
            assert int(lines[0]) == proc.pid
            # Some platforms cannot expose a process start time; the helper
            # intentionally retains a one-line legacy-safe pidfile there.
            start_time = get_process_start_time(proc.pid)
            if start_time is None:
                assert len(lines) == 1
            else:
                assert int(lines[1]) == start_time
        finally:
            # The helper process exits by itself.  Do not signal it here: the
            # repository-wide live-system guard can observe a just-reparented
            # child as outside this test's subtree even though ``poll()`` has
            # not caught up yet.
            proc.wait(timeout=5)


class TestIdentityGuard:
    def test_kills_when_start_time_matches(self, tmp_path):
        """A genuine bridge (recorded start time matches) IS reaped."""
        proc = _spawn_node_bridge(tmp_path)
        try:
            _write_bridge_pidfile(tmp_path, proc.pid)
            _kill_stale_bridge_by_pidfile(tmp_path)
            assert _wait_dead(proc), "the real bridge process should be killed"
            assert not (tmp_path / "bridge.pid").exists()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


    def test_legacy_pidfile_kills_matching_bridge_cmdline(self, tmp_path):
        """Legacy pidfile: a PID whose cmdline names node + session IS reaped."""
        # Shape the cmdline to look like the node bridge for this session.
        proc = _spawn_node_bridge(tmp_path)
        try:
            (tmp_path / "bridge.pid").write_text(
                str(proc.pid), encoding="utf-8"
            )  # legacy: pid only
            _kill_stale_bridge_by_pidfile(tmp_path)
            assert _wait_dead(proc), "a cmdline-confirmed bridge should be killed"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_recycled_node_pid_is_not_killed_when_start_time_changes(self, tmp_path, monkeypatch):
        """A recycled PID remains unsafe even if it now names another node."""
        from gateway import status

        (tmp_path / "bridge.pid").write_text("4242\n100", encoding="utf-8")
        monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(status, "get_process_start_time", lambda _pid: 101)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda _pid: f"node bridge.js --session {tmp_path}",
        )
        killed = []
        monkeypatch.setattr(
            "plugins.platforms.whatsapp.adapter.os.kill",
            lambda pid, sig: killed.append((pid, sig)),
        )

        _kill_stale_bridge_by_pidfile(tmp_path)

        assert killed == []

    def test_matching_start_time_still_requires_session_command_line(self, tmp_path, monkeypatch):
        """A PID/start-time match alone cannot authorize a signal."""
        from gateway import status

        (tmp_path / "bridge.pid").write_text("4242\n100", encoding="utf-8")
        monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(status, "get_process_start_time", lambda _pid: 100)
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda _pid: "node unrelated.js"
        )
        killed = []
        monkeypatch.setattr(
            "plugins.platforms.whatsapp.adapter.os.kill",
            lambda pid, sig: killed.append((pid, sig)),
        )

        _kill_stale_bridge_by_pidfile(tmp_path)

        assert killed == []

    @pytest.mark.parametrize(
        ("pidfile", "expected_start"),
        [("4242\n100", 100), ("4242", None)],
        ids=["start-time", "legacy"],
    )
    @pytest.mark.parametrize(
        "cmdline_builder",
        [
            lambda session: f"node bridge.js --session {session}-other",
            lambda session: (
                f"node bridge.js --session "
                f"{session.parent / ('other-' + session.name)}"
            ),
            lambda session: f"python node bridge.js --session {session}",
            lambda session: (
                f"node bridge.js --session {session}-other --session {session}"
            ),
            lambda session: (
                f"node bridge.js --session {session} --session {session}-other"
            ),
        ],
        ids=[
            "session-suffix",
            "session-prefix",
            "fake-node-argument",
            "duplicate-foreign-first",
            "duplicate-foreign-second",
        ],
    )
    def test_pidfile_never_kills_prefix_suffix_or_fake_node(
        self, tmp_path, monkeypatch, pidfile, expected_start, cmdline_builder
    ):
        """Only Node's exact ``--session <path>`` argv pair can authorize SIGTERM."""
        from gateway import status

        session = tmp_path / "wa-session"
        session.mkdir()
        (session / "bridge.pid").write_text(pidfile, encoding="utf-8")
        monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(status, "get_process_start_time", lambda _pid: expected_start)
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda _pid: cmdline_builder(session)
        )
        killed = []
        monkeypatch.setattr(
            "plugins.platforms.whatsapp.adapter.os.kill",
            lambda pid, sig: killed.append((pid, sig)),
        )

        _kill_stale_bridge_by_pidfile(session)

        assert killed == []

    @pytest.mark.parametrize(
        ("pidfile", "expected_start"),
        [("4242\n100", 100), ("4242", None)],
        ids=["start-time", "legacy"],
    )
    def test_pidfile_kills_only_exact_node_session_argv(
        self, tmp_path, monkeypatch, pidfile, expected_start
    ):
        """Both modern and legacy PID files retain an exact positive path."""
        from gateway import status

        session = tmp_path / "wa-session"
        session.mkdir()
        (session / "bridge.pid").write_text(pidfile, encoding="utf-8")
        monkeypatch.setattr(status, "_pid_exists", lambda _pid: True)
        monkeypatch.setattr(status, "get_process_start_time", lambda _pid: expected_start)
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda _pid: f"/usr/local/bin/node bridge.js --session {session}",
        )
        killed = []
        monkeypatch.setattr(
            "plugins.platforms.whatsapp.adapter.os.kill",
            lambda pid, sig: killed.append((pid, sig)),
        )

        _kill_stale_bridge_by_pidfile(session)

        assert killed == [(4242, 15)]
