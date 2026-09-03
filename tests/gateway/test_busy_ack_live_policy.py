"""Real-path coverage for live, source-aware busy-ack visibility policy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource, build_session_key
from gateway.run import GatewayRunner


def _source(
    *,
    platform: Platform = Platform.WHATSAPP,
    profile: str | None = None,
    chat_type: str = "dm",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_type=chat_type,
        user_id="user-1",
        profile=profile,
    )


def _write_config(home, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")


def test_busy_ack_resolution_precedence_keeps_legacy_env_as_fallback_only():
    """Chat/platform/global YAML each beat a conflicting legacy environment."""
    from gateway.display_config import resolve_busy_ack_enabled

    config = {
        "display": {
            "busy_ack_enabled": True,
            "platforms": {"whatsapp": {"busy_ack_enabled": True}},
            "chat_types": {"group": {"busy_ack_enabled": False}},
        }
    }
    assert (
        resolve_busy_ack_enabled(
            config, "whatsapp", chat_type="group", legacy_env="true"
        )
        is False
    )

    del config["display"]["chat_types"]
    assert (
        resolve_busy_ack_enabled(
            config, "whatsapp", chat_type="group", legacy_env="false"
        )
        is True
    )

    del config["display"]["platforms"]
    assert resolve_busy_ack_enabled(config, "whatsapp", legacy_env="false") is True
    assert (
        resolve_busy_ack_enabled(
            {"display": {"busy_ack_enabled": None}},
            "whatsapp",
            legacy_env="false",
        )
        is False
    )
    assert resolve_busy_ack_enabled({}, "whatsapp", legacy_env="true") is True
    assert resolve_busy_ack_enabled({}, "whatsapp", legacy_env="false") is False
    # Busy acknowledgements use the same conservative boolean normalisation
    # as the other display flags: an explicit unrecognised value is false,
    # rather than silently falling through to the stale legacy environment.
    assert (
        resolve_busy_ack_enabled(
            {"display": {"busy_ack_enabled": "unexpected"}},
            "telegram",
            legacy_env="true",
        )
        is False
    )


def test_whatsapp_builtins_are_quiet_but_other_platforms_keep_existing_default():
    from gateway.display_config import resolve_busy_ack_enabled

    assert resolve_busy_ack_enabled({}, "whatsapp") is False
    assert resolve_busy_ack_enabled({}, "whatsapp_cloud") is False
    assert resolve_busy_ack_enabled({}, "telegram") is True
    assert resolve_busy_ack_enabled({}, "discord") is True


def test_live_profile_config_managed_overlay_and_lkg(tmp_path, monkeypatch):
    """Profiles re-read policy independently; a torn edit retains only its LKG."""
    from hermes_cli.managed_scope import invalidate_managed_cache

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    _write_config(
        tmp_path / "profiles" / "research",
        "display:\n  busy_ack_enabled: true\n",
    )
    _write_config(
        tmp_path / "profiles" / "ops",
        "display:\n  busy_ack_enabled: false\n",
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    research = _source(profile="research")
    ops = _source(profile="ops")
    assert runner._busy_ack_enabled_for_source(research) is True
    assert runner._busy_ack_enabled_for_source(ops) is False

    # A managed leaf wins over the user-owned per-profile YAML on every live
    # read. It is not retained as an environment side effect.
    managed = tmp_path / "managed"
    _write_config(
        managed,
        "display:\n  platforms:\n    whatsapp:\n      busy_ack_enabled: false\n",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    invalidate_managed_cache()
    assert runner._busy_ack_enabled_for_source(research) is False
    assert runner._busy_ack_enabled_for_source(ops) is False

    # Remove the overlay, then capture research's valid policy once more.
    monkeypatch.delenv("HERMES_MANAGED_DIR")
    invalidate_managed_cache()
    assert runner._busy_ack_enabled_for_source(research) is True
    (tmp_path / "profiles" / "research" / "config.yaml").write_text(
        "display: [broken", encoding="utf-8"
    )
    assert runner._busy_ack_enabled_for_source(research) is True

    # There is no valid snapshot for this profile. WhatsApp fails closed even
    # though the legacy env would otherwise opt it in; existing platforms use
    # that compatibility fallback normally.
    _write_config(tmp_path / "profiles" / "new", "display: [broken")
    assert runner._busy_ack_enabled_for_source(_source(profile="new")) is False
    assert (
        runner._busy_ack_enabled_for_source(
            _source(platform=Platform.TELEGRAM, profile="new")
        )
        is False
    )
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    assert (
        runner._busy_ack_enabled_for_source(
            _source(platform=Platform.TELEGRAM, profile="new")
        )
        is True
    )


def test_malformed_managed_config_before_busy_ack_snapshot_fails_closed(tmp_path, monkeypatch):
    """A corrupt managed policy cannot fall through to a user-enabled ack."""
    from hermes_cli.managed_scope import invalidate_managed_cache

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    _write_config(tmp_path, "display:\n  busy_ack_enabled: true\n")
    managed = tmp_path / "managed"
    _write_config(managed, "display: [broken")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    invalidate_managed_cache()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()

    assert runner._busy_ack_enabled_for_source(_source()) is False


def test_malformed_managed_config_retains_combined_busy_ack_snapshot(tmp_path, monkeypatch):
    """A corrupt managed edit retains the prior overlay instead of re-enabling acks."""
    from hermes_cli.managed_scope import invalidate_managed_cache

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    _write_config(tmp_path, "display:\n  busy_ack_enabled: true\n")
    managed = tmp_path / "managed"
    _write_config(managed, "display:\n  busy_ack_enabled: false\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    invalidate_managed_cache()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    source = _source()
    assert runner._busy_ack_enabled_for_source(source) is False

    _write_config(managed, "display: [broken")
    invalidate_managed_cache()

    assert runner._busy_ack_enabled_for_source(source) is False


def test_multiplex_busy_ack_rejects_unserved_or_mismatched_profile_homes(
    tmp_path, monkeypatch
):
    """A foreign source profile cannot select another profile's live policy."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    _write_config(
        tmp_path / "profiles" / "served",
        "display:\n  busy_ack_enabled: true\n",
    )
    _write_config(
        tmp_path / "profiles" / "foreign",
        "display:\n  busy_ack_enabled: true\n",
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=["served"],
    )

    # The foreign directory exists and would opt in if it were trusted, but
    # the multiplex runner serves only ``served`` (plus the default profile).
    assert runner._busy_ack_enabled_for_source(_source(profile="foreign")) is False

    # A served name must resolve to that exact served home, not merely any
    # other served-looking path selected by a faulty adapter/routing seam.
    runner._resolve_profile_home_for_source = lambda _source: (
        tmp_path / "profiles" / "foreign"
    )
    assert runner._busy_ack_enabled_for_source(_source(profile="served")) is False


def _busy_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._sessions = {}
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner._restart_requested = False
    runner._external_drain_active = False
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    return runner


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config.extra = {}
    adapter.platform = Platform.WHATSAPP
    return adapter


@pytest.mark.asyncio
async def test_whatsapp_busy_and_restart_drain_mutate_state_without_sending(tmp_path, monkeypatch):
    """Normal and direct restart/drain rails honor live WhatsApp quiet policy."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    runner = _busy_runner()
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    source = _source()
    event = MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="message-1",
    )
    key = build_session_key(source)
    agent = MagicMock()
    agent._active_children = []
    runner._running_agents[key] = agent

    assert await runner._handle_active_session_busy_message(event, key) is True
    agent.interrupt.assert_called_once_with("follow up")
    adapter._send_with_retry.assert_not_called()

    # Isolate the drain assertion from the ordinary busy queue we just proved.
    adapter._pending_messages.clear()
    runner._session_state(key).conversation.queued_events.clear()
    runner._draining = True
    runner._restart_requested = True
    runner._busy_input_mode = "queue"
    drain_event = MessageEvent(
        text="during restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="message-2",
    )
    assert await runner._handle_active_session_busy_message(drain_event, key) is True
    assert adapter._pending_messages[key] is drain_event
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_priority_drain_returns_no_platform_message_when_whatsapp_is_quiet(
    tmp_path, monkeypatch
):
    """The priority fast-path cannot leak its returned restart/drain text."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    runner = _busy_runner()
    runner._draining = True
    runner._restart_requested = True
    runner._busy_input_mode = "queue"
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    source = _source()
    key = build_session_key(source)
    agent = MagicMock()
    agent._active_children = []
    runner._running_agents[key] = agent
    event = MessageEvent(
        text="during priority drain",
        message_type=MessageType.TEXT,
        source=source,
        message_id="message-3",
    )

    assert await runner._handle_message(event) is None
    assert adapter._pending_messages[key] is event
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_external_drain_returns_no_platform_message_when_whatsapp_is_quiet(
    tmp_path, monkeypatch
):
    """The external-drain returned response stays silent without claiming work."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    runner = _busy_runner()
    runner._external_drain_active = True
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    source = _source()
    key = build_session_key(source)
    event = MessageEvent(
        text="during external drain",
        message_type=MessageType.TEXT,
        source=source,
        message_id="message-4",
    )

    assert await runner._handle_message(event) is None
    assert key not in runner._running_agents
    assert key not in adapter._pending_messages
    adapter._send_with_retry.assert_not_called()
