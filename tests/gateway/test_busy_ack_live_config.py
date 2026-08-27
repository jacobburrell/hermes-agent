"""Live, profile-aware busy-ack suppression at the outbound boundary."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner


class _WhatsAppAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(*, mode: str = "interrupt", multiplex: bool = False) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner._busy_input_mode = mode
    # Keep text inside the busy handler even for queue mode; this is a valid
    # legacy busy_text_mode combination and lets the test pin its queue state.
    runner._busy_text_mode = "interrupt"
    runner._profile_adapters = {}
    runner.adapters = {}
    runner._sessions = {}
    runner._draining = False
    runner._restart_requested = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    return runner


def _adapter() -> _WhatsAppAdapter:
    adapter = _WhatsAppAdapter(
        PlatformConfig(enabled=True, token="test-token"),
        Platform.WHATSAPP,
    )
    adapter.set_message_handler(AsyncMock(return_value="unused"))
    adapter._send_with_retry = AsyncMock(wraps=adapter._send_with_retry)
    return adapter


def _real_whatsapp_adapter():
    """Real ingress with mocked Node transport; text delivery stays unused."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra={"session_name": "busy-ack-test", "bridge_port": 19876},
        )
    )
    node_transport = MagicMock()
    node_transport.closed = False
    node_transport.post = MagicMock()
    adapter._running = True
    adapter._http_session = node_transport
    adapter._send_with_retry = AsyncMock(wraps=adapter._send_with_retry)
    return adapter, node_transport


def _event(
    *,
    message_id: str = "message-1",
    profile: str | None = None,
    platform: Platform = Platform.WHATSAPP,
    text: str = "follow up",
    message_type: MessageType = MessageType.TEXT,
    media_urls: list[str] | None = None,
    chat_type: str = "group",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=SessionSource(
            platform=platform,
            chat_id="group-1",
            chat_type=chat_type,
            user_id="user-1",
            profile=profile,
        ),
        message_id=message_id,
        media_urls=list(media_urls or []),
    )


def _write_config(home, *, busy_ack_enabled=..., display_config=None):
    display = dict(display_config or {})
    if busy_ack_enabled is not ...:
        display["busy_ack_enabled"] = busy_ack_enabled
    config = {
        "display": display,
        "onboarding": {"seen": {"busy_input_prompt": True}},
    }
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def _write_malformed_config(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("display: [\n", encoding="utf-8")


def _seed_busy_turn(runner, adapter, event):
    session_key = runner._session_key_for_source(event.source)
    agent = MagicMock()
    agent._active_children = []
    agent.steer.return_value = True
    agent.redirect.return_value = True
    runner._running_agents[session_key] = agent
    return session_key, agent


@pytest.mark.asyncio
async def test_whatsapp_busy_ingress_honors_yaml_false_over_stale_true_env(
    tmp_path,
    monkeypatch,
):
    """The adapter busy ingress stays silent while preserving the interrupt."""
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _write_config(tmp_path, busy_ack_enabled=False)

    runner = _runner()
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    event = _event()
    session_key, agent = _seed_busy_turn(runner, adapter, event)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    adapter._send_with_retry.assert_not_awaited()
    agent.interrupt.assert_called_once_with("follow up")
    assert adapter._pending_messages[session_key] is event
    assert session_key not in runner._busy_ack_ts
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved == {
        "display": {"busy_ack_enabled": False},
        "onboarding": {"seen": {"busy_input_prompt": True}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["queue", "steer", "interrupt", "redirect"])
async def test_yaml_false_suppresses_only_ack_after_busy_state_mutation(
    tmp_path,
    monkeypatch,
    mode,
):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _write_config(tmp_path, busy_ack_enabled=False)

    effective_mode = "interrupt" if mode == "redirect" else mode
    runner = _runner(mode=effective_mode)
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    event = _event()
    session_key, agent = _seed_busy_turn(runner, adapter, event)
    if mode == "redirect":
        agent._supports_active_turn_redirect = True

    assert await runner._handle_active_session_busy_message(event, session_key) is True

    adapter._send_with_retry.assert_not_awaited()
    assert session_key not in runner._busy_ack_ts
    if mode == "queue":
        assert adapter._pending_messages[session_key] is event
        agent.steer.assert_not_called()
        agent.interrupt.assert_not_called()
    elif mode == "steer":
        agent.steer.assert_called_once_with("follow up")
        agent.interrupt.assert_not_called()
        assert session_key not in adapter._pending_messages
    elif mode == "redirect":
        agent.redirect.assert_called_once_with("follow up")
        agent.interrupt.assert_not_called()
        assert session_key not in adapter._pending_messages
    else:
        assert adapter._pending_messages[session_key] is event
        agent.steer.assert_not_called()
        agent.interrupt.assert_called_once_with("follow up")


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_during_drain", [True, False])
async def test_yaml_false_suppresses_drain_notice_after_preserving_state(
    tmp_path,
    monkeypatch,
    queue_during_drain,
):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _write_config(tmp_path, busy_ack_enabled=False)

    runner = _runner(mode="queue")
    runner._draining = True
    runner._queue_during_drain_enabled = MagicMock(
        return_value=queue_during_drain
    )
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    event = _event()
    session_key, agent = _seed_busy_turn(runner, adapter, event)

    assert await runner._handle_active_session_busy_message(event, session_key) is True

    runner._queue_during_drain_enabled.assert_called_once_with("queue")
    adapter._send_with_retry.assert_not_awaited()
    agent.steer.assert_not_called()
    agent.interrupt.assert_not_called()
    assert (adapter._pending_messages.get(session_key) is event) is queue_during_drain


@pytest.mark.asyncio
async def test_explicit_queue_command_result_remains_user_visible(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _write_config(tmp_path, busy_ack_enabled=False)

    runner = _runner(mode="queue")
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    event = _event(text="/queue follow later")
    session_key = runner._session_key_for_source(event.source)

    result = await runner._busy_queue_command(event, session_key, event.source)

    assert result == "Queued for the next turn."
    assert adapter._pending_messages[session_key].text == "follow later"
    adapter._send_with_retry.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "queued"),
    [("queue", True), ("interrupt", False)],
)
async def test_priority_restart_drain_return_is_silent_after_state_decision(
    tmp_path,
    monkeypatch,
    mode,
    queued,
):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _write_config(tmp_path, busy_ack_enabled=False)

    runner = _runner(mode=mode)
    runner._draining = True
    runner._restart_requested = True
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    event = _event()
    session_key, agent = _seed_busy_turn(runner, adapter, event)

    assert await runner._handle_message(event) is None

    assert (adapter._pending_messages.get(session_key) is event) is queued
    adapter._send_with_retry.assert_not_awaited()
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_busy_event_observes_yaml_edit_without_gateway_restart(
    tmp_path,
    monkeypatch,
):
    """A post-startup YAML mute wins before the existing debounce timestamp."""
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _write_config(tmp_path)

    runner = _runner()
    adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    first = _event(message_id="message-1")
    session_key, _agent = _seed_busy_turn(runner, adapter, first)

    assert await runner._handle_active_session_busy_message(first, session_key) is True
    assert adapter._send_with_retry.await_count == 1
    first_ack_ts = runner._busy_ack_ts[session_key]

    _write_config(tmp_path, busy_ack_enabled=False)
    second = _event(message_id="message-2")
    assert await runner._handle_active_session_busy_message(second, session_key) is True

    assert adapter._send_with_retry.await_count == 1
    assert runner._busy_ack_ts[session_key] == first_ack_ts


@pytest.mark.asyncio
async def test_real_whatsapp_seven_photo_burst_falls_back_to_silent_queue(
    tmp_path,
    monkeypatch,
):
    """One owner plus six non-steerable photos form one silent queued album."""
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    _write_config(tmp_path, busy_ack_enabled=False)

    runner = _runner(mode="steer")
    adapter, node_transport = _real_whatsapp_adapter()
    runner.adapters[Platform.WHATSAPP] = adapter
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    owner_started = asyncio.Event()
    release_owner = asyncio.Event()
    handled_events = []

    async def _blocking_owner(handled_event):
        handled_events.append(handled_event)
        owner_started.set()
        await release_owner.wait()
        return ""

    adapter.set_message_handler(_blocking_owner)
    burst = [
        _event(
            message_id=f"album-{index}",
            text="",
            message_type=MessageType.PHOTO,
            media_urls=[f"/tmp/album-{index}.jpg"],
        )
        for index in range(1, 8)
    ]

    await adapter.handle_message(burst[0])
    await asyncio.wait_for(owner_started.wait(), timeout=1)
    session_key, agent = _seed_busy_turn(runner, adapter, burst[0])
    owner_task = adapter._session_tasks[session_key]
    assert list(adapter._session_tasks) == [session_key]

    for follow_up in burst[1:]:
        await adapter.handle_message(follow_up)

    assert list(adapter._pending_messages) == [session_key]
    pending = adapter._pending_messages[session_key]
    assert pending.message_type == MessageType.PHOTO
    assert pending.media_urls == [
        f"/tmp/album-{index}.jpg" for index in range(2, 8)
    ]
    agent.steer.assert_not_called()
    agent.interrupt.assert_not_called()
    adapter._send_with_retry.assert_not_awaited()
    assert not any(
        str(call.args[0]).endswith(("/send", "/edit"))
        for call in node_transport.post.call_args_list
        if call.args
    )

    release_owner.set()
    await asyncio.wait_for(owner_task, timeout=1)
    for _ in range(10):
        if len(handled_events) == 2 and session_key not in adapter._session_tasks:
            break
        await asyncio.sleep(0)
    assert handled_events[0] is burst[0]
    assert len(handled_events) == 2
    assert handled_events[1].media_urls == [
        f"/tmp/album-{index}.jpg" for index in range(2, 8)
    ]
    adapter._send_with_retry.assert_not_awaited()
    assert not any(
        str(call.args[0]).endswith(("/send", "/edit"))
        for call in node_transport.post.call_args_list
        if call.args
    )


@pytest.mark.asyncio
async def test_multiplex_busy_ack_reads_source_profile_without_cross_profile_bleed(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run

    primary_home = tmp_path / "primary"
    secondary_home = primary_home / "profiles" / "secondary"
    _write_config(primary_home, busy_ack_enabled=True)
    _write_config(secondary_home, busy_ack_enabled=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", primary_home)
    monkeypatch.setenv("HERMES_HOME", str(primary_home))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    runner = _runner(multiplex=True)
    primary_adapter = _adapter()
    secondary_adapter = _adapter()
    runner.adapters[Platform.WHATSAPP] = primary_adapter
    runner._profile_adapters = {
        "secondary": {Platform.WHATSAPP: secondary_adapter}
    }

    secondary = _event(message_id="secondary-1", profile="secondary")
    secondary_key, secondary_agent = _seed_busy_turn(
        runner, secondary_adapter, secondary
    )
    assert await runner._handle_active_session_busy_message(
        secondary, secondary_key
    ) is True
    secondary_adapter._send_with_retry.assert_not_awaited()
    secondary_agent.interrupt.assert_called_once_with("follow up")

    primary = _event(message_id="primary-1")
    primary_key, primary_agent = _seed_busy_turn(runner, primary_adapter, primary)
    assert await runner._handle_active_session_busy_message(primary, primary_key) is True
    primary_adapter._send_with_retry.assert_awaited_once()
    primary_agent.interrupt.assert_called_once_with("follow up")

    secondary_again = _event(message_id="secondary-2", profile="secondary")
    assert await runner._handle_active_session_busy_message(
        secondary_again, secondary_key
    ) is True
    secondary_adapter._send_with_retry.assert_not_awaited()
    primary_adapter._send_with_retry.assert_awaited_once()


def test_chat_type_override_wins_then_dm_falls_back_to_platform(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run

    _write_config(
        tmp_path,
        display_config={
            "busy_ack_enabled": True,
            "platforms": {
                "whatsapp": {
                    "busy_ack_enabled": False,
                    "chat_types": {
                        "group": {"busy_ack_enabled": True},
                    },
                },
            },
        },
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    runner = _runner()
    assert runner._busy_ack_enabled_for_source(_event(chat_type="group").source)
    assert not runner._busy_ack_enabled_for_source(_event(chat_type="dm").source)


def test_platform_override_wins_over_global_and_legacy_env(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    _write_config(
        tmp_path,
        display_config={
            "busy_ack_enabled": False,
            "platforms": {"whatsapp": {"busy_ack_enabled": True}},
        },
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    assert _runner()._busy_ack_enabled_for_source(_event().source)


def test_valid_false_then_malformed_yaml_keeps_path_last_known_good(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    _write_config(tmp_path, busy_ack_enabled=False)
    runner = _runner()

    assert not runner._busy_ack_enabled_for_source(_event().source)
    _write_malformed_config(tmp_path)
    assert not runner._busy_ack_enabled_for_source(_event().source)


def test_valid_managed_false_then_malformed_keeps_path_last_known_good(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run
    from hermes_cli import managed_scope

    user_home = tmp_path / "user"
    managed_home = tmp_path / "managed"
    _write_config(user_home)
    _write_config(managed_home, busy_ack_enabled=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", user_home)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    managed_scope.invalidate_managed_cache()

    runner = _runner()
    assert not runner._busy_ack_enabled_for_source(_event().source)

    _write_malformed_config(managed_home)
    managed_scope.invalidate_managed_cache()
    assert not runner._busy_ack_enabled_for_source(_event().source)


@pytest.mark.parametrize(
    "document",
    ["display: [\n", "- list-item\n", "scalar\n", "null\n", ""],
)
def test_first_invalid_whatsapp_config_fails_closed(
    tmp_path,
    monkeypatch,
    document,
):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    (tmp_path / "config.yaml").write_text(document, encoding="utf-8")

    assert not _runner()._busy_ack_enabled_for_source(_event().source)


def test_first_missing_whatsapp_config_fails_closed(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    assert not (tmp_path / "config.yaml").exists()
    assert not _runner()._busy_ack_enabled_for_source(_event().source)


def test_managed_overlay_wins_user_yaml_and_is_valid_without_user_file(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run
    from hermes_cli import managed_scope

    user_home = tmp_path / "user"
    managed_home = tmp_path / "managed"
    _write_config(managed_home, busy_ack_enabled=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", user_home)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    managed_scope.invalidate_managed_cache()

    runner = _runner()
    assert runner._busy_ack_enabled_for_source(_event().source)

    _write_config(user_home, busy_ack_enabled=False)
    assert runner._busy_ack_enabled_for_source(_event().source)


def test_malformed_multiplex_profile_does_not_bleed_from_valid_peer(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run

    primary_home = tmp_path / "primary"
    secondary_home = primary_home / "profiles" / "secondary"
    _write_config(primary_home, busy_ack_enabled=True)
    _write_malformed_config(secondary_home)
    monkeypatch.setattr(gateway_run, "_hermes_home", primary_home)
    monkeypatch.setenv("HERMES_HOME", str(primary_home))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    runner = _runner(multiplex=True)
    secondary = _event(profile="secondary")
    primary = _event(profile="default")

    assert not runner._busy_ack_enabled_for_source(secondary.source)
    assert runner._busy_ack_enabled_for_source(primary.source)
    assert not runner._busy_ack_enabled_for_source(secondary.source)


@pytest.mark.parametrize("profile", ["missing", "unserved"])
def test_unknown_or_unserved_routed_whatsapp_profile_never_borrows_primary_policy(
    tmp_path,
    monkeypatch,
    profile,
):
    """The real profile resolver's fallback cannot leak primary visibility."""
    import gateway.run as gateway_run

    primary_home = tmp_path / "primary"
    _write_config(primary_home, busy_ack_enabled=True)
    _write_config(primary_home / "profiles" / "served", busy_ack_enabled=False)
    _write_config(primary_home / "profiles" / "unserved", busy_ack_enabled=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", primary_home)
    monkeypatch.setenv("HERMES_HOME", str(primary_home))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    runner = _runner(multiplex=True)
    runner.config.multiplex_profile_allowlist = ["served"]
    source = _event(profile=profile).source

    resolved = runner._resolve_profile_home_for_source(source)
    if profile == "missing":
        assert resolved == primary_home
    else:
        assert resolved == primary_home / "profiles" / "unserved"
    assert not runner._busy_ack_enabled_for_source(source)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("false", False), ("true", True), (None, True)],
)
def test_valid_empty_mapping_preserves_env_then_builtin_fallback(
    tmp_path,
    monkeypatch,
    env_value,
    expected,
):
    import gateway.run as gateway_run

    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    if env_value is None:
        monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    else:
        monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", env_value)

    assert _runner()._busy_ack_enabled_for_source(_event().source) is expected


@pytest.mark.parametrize("env_value", ["true", "false"])
def test_first_read_failure_other_platform_preserves_legacy_behavior(
    tmp_path,
    monkeypatch,
    env_value,
):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", env_value)
    source = _event(platform=Platform.API_SERVER).source

    assert _runner()._busy_ack_enabled_for_source(source) is (env_value == "true")


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("false", False), ("true", True)],
)
def test_absent_yaml_busy_ack_key_preserves_env_fallback(
    tmp_path,
    monkeypatch,
    env_value,
    expected,
):
    import gateway.run as gateway_run

    _write_config(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", env_value)

    assert _runner()._busy_ack_enabled_for_source(_event().source) is expected


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [
        (False, False),
        ("false", False),
        ("off", False),
        (True, True),
        ("true", True),
        ("yes", True),
    ],
)
def test_yaml_busy_ack_bool_and_string_values_override_stale_env(
    tmp_path,
    monkeypatch,
    yaml_value,
    expected,
):
    import gateway.run as gateway_run

    _write_config(tmp_path, busy_ack_enabled=yaml_value)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv(
        "HERMES_GATEWAY_BUSY_ACK_ENABLED",
        "false" if expected else "true",
    )

    assert _runner()._busy_ack_enabled_for_source(_event().source) is expected


def test_documented_busy_ack_key_is_recognized_by_config_validation():
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key("display.busy_ack_enabled") == (True, None)
