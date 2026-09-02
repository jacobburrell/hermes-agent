"""Ownership boundaries for the local WhatsApp Node bridge.

The bridge is one local TCP service.  Its session lock alone cannot stop two
multiplexed profiles from treating the same port as theirs, and a health
response from another profile must never be reused or killed.
"""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


class _AsyncCM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


class _RaisingAsyncCM:
    """Async context manager which fails before a health response exists."""

    async def __aenter__(self):
        raise OSError("bridge unavailable")

    async def __aexit__(self, *exc):
        return False


def _adapter(session_path: Path, bridge_script: Path, *, port: int = 19876):
    """Build the narrow adapter state used by connect ownership tests."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = port
    adapter._bridge_script = str(bridge_script)
    adapter._session_path = session_path
    adapter._bridge_process = None
    adapter._bridge_port_lock_identity = None
    adapter._bridge_port_local_claim = False
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._reply_prefix = None
    adapter._send_read_receipts = False
    adapter._running = False
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = None
    return adapter


def _bridge_tree(tmp_path: Path, session_name: str) -> tuple[Path, Path]:
    """Create the smallest fresh paired bridge tree for ``connect()``."""
    from plugins.platforms.whatsapp.adapter import _file_content_hash

    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir(exist_ok=True)
    bridge = bridge_dir / "bridge.js"
    bridge.write_text("// ownership test bridge\n", encoding="utf-8")
    package = bridge_dir / "package.json"
    package.write_text('{"name":"bridge"}\n', encoding="utf-8")
    node_modules = bridge_dir / "node_modules"
    node_modules.mkdir(exist_ok=True)
    (node_modules / ".hermes-pkg-hash").write_text(
        _file_content_hash(package), encoding="utf-8"
    )
    session = tmp_path / session_name
    session.mkdir()
    (session / "creds.json").write_text("{}", encoding="utf-8")
    return bridge, session


def _health_client(payload: dict):
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value=payload)
    session = MagicMock()
    session.get = MagicMock(return_value=_AsyncCM(response))
    return MagicMock(return_value=_AsyncCM(session))


def _fake_aiohttp(payload: dict):
    """Minimal importable aiohttp shape for this adapter-only test path."""
    return SimpleNamespace(
        ClientSession=_health_client(payload),
        ClientTimeout=lambda **_kwargs: object(),
    )


def _health_session(payload: dict):
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value=payload)
    session = MagicMock()
    session.get = MagicMock(return_value=_AsyncCM(response))
    return _AsyncCM(session)


def _discard_background_coroutine(coro):
    """Avoid leaving the reuse-path poll coroutine unawaited in a unit test."""
    coro.close()
    return MagicMock()


@pytest.fixture(autouse=True)
def _clear_in_process_port_owners():
    """Do not let an assertion failure leak a synthetic owner between tests."""
    from plugins.platforms.whatsapp.adapter import _BRIDGE_PORT_OWNERS

    _BRIDGE_PORT_OWNERS.clear()
    yield
    _BRIDGE_PORT_OWNERS.clear()


class TestBridgePortOwnership:
    def test_session_fingerprint_matches_node_through_symlink_and_missing_leaf(self, tmp_path):
        """Python and Node canonicalize the same not-yet-created session path."""
        from plugins.platforms.whatsapp.adapter import (
            _canonical_session_path,
            _session_path_fingerprint,
        )

        node = shutil.which("node")
        if node is None:
            pytest.skip("Node is required for the cross-language bridge contract")
        canonical_parent = tmp_path / "canonical-parent"
        canonical_parent.mkdir()
        alias_parent = tmp_path / "session-alias"
        try:
            alias_parent.symlink_to(canonical_parent, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink unavailable: {exc}")
        missing_session = alias_parent / "not-created-yet"
        helper = (
            Path(__file__).resolve().parents[2]
            / "scripts/whatsapp-bridge/bridge_helpers.js"
        )
        script = (
            "import { sessionPathFingerprint } from "
            f"{json.dumps(helper.as_uri())}; "
            "console.log(sessionPathFingerprint(process.argv[1]));"
        )
        result = subprocess.run(
            [node, "--input-type=module", "-e", script, str(missing_session)],
            check=True,
            capture_output=True,
            text=True,
        )

        assert _canonical_session_path(missing_session) == str(
            canonical_parent / "not-created-yet"
        )
        assert result.stdout.strip() == _session_path_fingerprint(missing_session)

    def test_foreign_port_lock_is_retryable_without_implicit_takeover(self, tmp_path):
        """An ordinary port conflict does not signal the bridge owner."""
        bridge, session_path = _bridge_tree(tmp_path, "profile-a")
        adapter = _adapter(session_path, bridge)
        existing = {"pid": 42125}

        with patch(
            "gateway.status.acquire_scoped_lock", return_value=(False, existing)
        ), patch("gateway.status.take_over_scoped_lock_holder") as take_over:
            assert adapter._acquire_bridge_port_lock() is False

        take_over.assert_not_called()
        assert adapter._fatal_error_code is None

    def test_explicit_replace_can_take_over_port_lock_once(self, tmp_path):
        """The established ``--replace`` authority remains the only takeover."""
        bridge, canonical_session = _bridge_tree(tmp_path, "profile-a")
        session_path = tmp_path / "profile-a-alias"
        session_path.symlink_to(canonical_session, target_is_directory=True)
        adapter = _adapter(session_path, bridge)
        adapter._platform_lock_takeover_allowed = True
        adapter._platform_lock_takeover_attempted = False
        existing = {"pid": 42126}

        with patch(
            "gateway.status.acquire_scoped_lock",
            side_effect=[(False, existing), (True, None)],
        ) as acquire, patch(
            "gateway.status.take_over_scoped_lock_holder", return_value=42126
        ) as take_over, patch("gateway.status.release_scoped_lock"):
            assert adapter._acquire_bridge_port_lock() is True
            assert adapter._platform_lock_takeover_allowed is False
            assert adapter._platform_lock_takeover_attempted is True
            take_over.assert_called_once_with(existing)
            assert acquire.call_count == 2
            adapter._release_bridge_port_lock()

    def test_two_profile_sessions_cannot_claim_one_port_in_one_gateway(self, tmp_path):
        """The first profile remains owner; the second fails before bridge I/O."""
        bridge, first_session = _bridge_tree(tmp_path, "profile-a")
        _, second_session = _bridge_tree(tmp_path, "profile-b")
        first = _adapter(first_session, bridge)
        second = _adapter(second_session, bridge)

        with patch(
            "gateway.status.acquire_scoped_lock", return_value=(True, None)
        ) as acquire, patch("gateway.status.release_scoped_lock"):
            assert first._acquire_bridge_port_lock() is True
            assert second._acquire_bridge_port_lock() is False
            assert first._bridge_port_local_claim is True
            assert second._bridge_port_local_claim is False
            acquire.assert_called_once()
            first._release_bridge_port_lock()

    @pytest.mark.asyncio
    async def test_conflicting_profile_returns_before_probe_spawn_or_kill(self, tmp_path):
        """A same-process port conflict cannot disturb the original bridge."""
        bridge, first_session = _bridge_tree(tmp_path, "profile-a")
        _, second_session = _bridge_tree(tmp_path, "profile-b")
        first = _adapter(first_session, bridge)
        second = _adapter(second_session, bridge)

        with patch(
            "gateway.status.acquire_scoped_lock", return_value=(True, None)
        ), patch("gateway.status.release_scoped_lock"), patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ), patch.dict(sys.modules, {"aiohttp": _fake_aiohttp({})}), patch(
            "subprocess.Popen"
        ) as popen, patch.object(
            second, "_acquire_platform_lock", return_value=True
        ) as session_lock:
            assert first._acquire_bridge_port_lock() is True
            assert await second.connect() is False

        popen.assert_not_called()
        session_lock.assert_not_called()
        assert first._bridge_port_local_claim is True
        first._release_bridge_port_lock()

    @pytest.mark.asyncio
    async def test_reuses_only_health_with_matching_session_fingerprint(self, tmp_path):
        """A matching healthy bridge is reused without a restart."""
        from plugins.platforms.whatsapp.adapter import (
            _file_content_hash,
            _session_path_fingerprint,
        )

        bridge, canonical_session = _bridge_tree(tmp_path, "profile-a")
        session_path = tmp_path / "profile-a-alias-for-reuse"
        session_path.symlink_to(canonical_session, target_is_directory=True)
        adapter = _adapter(session_path, bridge)
        health = {
            "status": "connected",
            "scriptHash": _file_content_hash(bridge),
            "sendReadReceipts": False,
            "sessionFingerprint": _session_path_fingerprint(session_path),
            "pid": 42123,
        }

        with patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ), patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(health)}), patch(
            "subprocess.Popen"
        ) as popen, patch.object(
            adapter, "_acquire_bridge_port_lock", return_value=True
        ), patch.object(
            adapter, "_acquire_platform_lock", return_value=True
        ), patch.object(adapter, "_wire_plugin_handlers"), patch(
            "plugins.platforms.whatsapp.adapter.asyncio.create_task",
            side_effect=_discard_background_coroutine,
        ):
            assert await adapter.connect() is True

        popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_mismatched_health_never_reuses_or_kills_other_profile(self, tmp_path):
        """A port owner's different session is a retryable, silent conflict."""
        from plugins.platforms.whatsapp.adapter import _session_path_fingerprint

        bridge, session_path = _bridge_tree(tmp_path, "profile-a")
        _, foreign_session = _bridge_tree(tmp_path, "profile-b")
        adapter = _adapter(session_path, bridge)
        foreign_health = {
            "status": "connected",
            "sessionFingerprint": _session_path_fingerprint(foreign_session),
            "pid": 42124,
        }

        with patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ), patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(foreign_health)}), patch(
            "subprocess.Popen"
        ) as popen, patch(
            "plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"
        ) as kill_pidfile, patch.object(
            adapter, "_acquire_bridge_port_lock", return_value=True
        ), patch.object(
            adapter, "_acquire_platform_lock", return_value=True
        ):
            assert await adapter.connect() is False

        popen.assert_not_called()
        kill_pidfile.assert_not_called()
        assert adapter._fatal_error_code is None

    @pytest.mark.asyncio
    async def test_missing_health_fingerprint_fails_closed_without_kill(self, tmp_path):
        """A legacy or malformed health response cannot be reused or restarted."""
        bridge, canonical_session = _bridge_tree(tmp_path, "profile-a")
        session_path = tmp_path / "profile-a-alias-for-reuse"
        session_path.symlink_to(canonical_session, target_is_directory=True)
        adapter = _adapter(session_path, bridge)
        legacy_health = {"status": "connected", "pid": 42124}

        with patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ), patch.dict(sys.modules, {"aiohttp": _fake_aiohttp(legacy_health)}), patch(
            "subprocess.Popen"
        ) as popen, patch(
            "plugins.platforms.whatsapp.adapter._kill_stale_bridge_by_pidfile"
        ) as kill_pidfile, patch.object(
            adapter, "_acquire_bridge_port_lock", return_value=True
        ), patch.object(
            adapter, "_acquire_platform_lock", return_value=True
        ):
            assert await adapter.connect() is False

        popen.assert_not_called()
        kill_pidfile.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("matching_session", [False, True], ids=["foreign-session", "wrong-pid"])
    async def test_startup_refuses_foreign_health_after_child_bind_failure(
        self, tmp_path, matching_session
    ):
        """Readiness needs both our session fingerprint and child PID."""
        from plugins.platforms.whatsapp.adapter import _session_path_fingerprint

        bridge, session_path = _bridge_tree(tmp_path, "profile-a")
        _, foreign_session = _bridge_tree(tmp_path, "profile-b")
        adapter = _adapter(session_path, bridge)
        foreign_health = {
            "status": "connected",
            "sessionFingerprint": _session_path_fingerprint(
                session_path if matching_session else foreign_session
            ),
            "pid": 42126,
        }
        initial_session = MagicMock()
        initial_session.get = MagicMock(return_value=_RaisingAsyncCM())
        client_factory = MagicMock(
            side_effect=[_AsyncCM(initial_session), _health_session(foreign_health)]
        )
        fake_aiohttp = SimpleNamespace(
            ClientSession=client_factory,
            ClientTimeout=lambda **_kwargs: object(),
        )
        child = MagicMock()
        child.pid = 42127
        child.poll.return_value = None

        with patch(
            "plugins.platforms.whatsapp.adapter.check_whatsapp_requirements",
            return_value=True,
        ), patch.dict(sys.modules, {"aiohttp": fake_aiohttp}), patch(
            "plugins.platforms.whatsapp.adapter.asyncio.sleep", new_callable=AsyncMock
        ), patch(
            "subprocess.Popen", return_value=child
        ) as popen, patch.object(
            adapter, "_acquire_bridge_port_lock", return_value=True
        ), patch.object(
            adapter, "_acquire_platform_lock", return_value=True
        ):
            assert await adapter.connect() is False

        popen.assert_called_once()
        assert adapter._running is False
