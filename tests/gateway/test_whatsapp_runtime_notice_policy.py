"""WhatsApp's internal-notice boundary is quiet without changing final delivery.

The production incidents were emitted by several independent callback rails.
These tests deliberately exercise those rails through the real mixin methods,
not a wording-based outbound filter.
"""

from __future__ import annotations

import contextlib
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner, _sanitize_gateway_final_response
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


def _source(platform: Platform = Platform.WHATSAPP) -> SessionSource:
    return SessionSource(platform=platform, chat_id="chat-a", chat_type="group", user_id="owner")


def _runner(config: dict, *, valid: bool = True):
    runner = object.__new__(GatewayRunner)
    runner._notice_policy_config_for_source = lambda _source: (config, valid)
    return runner


def test_whatsapp_defaults_are_final_answer_first() -> None:
    runner = _runner({})
    source = _source()

    assert runner._transient_notice_enabled_for_source(source) is False
    assert runner._memory_notification_mode_for_source(source) == "off"
    assert runner._progress_notices_enabled_for_source(source) is False
    assert runner._long_running_notifications_enabled_for_source(source) is False
    assert runner._busy_ack_enabled_for_source(source) is False


def test_whatsapp_runtime_notice_opt_in_is_source_platform_scoped() -> None:
    runner = _runner({"display": {"runtime_notices": True, "platforms": {"whatsapp": {
        "runtime_notices": True, "memory_notifications": "verbose", "tool_progress": "new",
        "thinking_progress": True, "long_running_notifications": True, "busy_ack_enabled": True,
    }}}})
    source = _source()

    assert runner._transient_notice_enabled_for_source(source) is True
    assert runner._memory_notification_mode_for_source(source) == "verbose"
    assert runner._progress_notices_enabled_for_source(source) is True
    assert runner._long_running_notifications_enabled_for_source(source) is True
    assert runner._busy_ack_enabled_for_source(source) is True


def test_whatsapp_global_legacy_settings_do_not_reenable_notice_rails() -> None:
    runner = _runner({"display": {
        "runtime_notices": True, "memory_notifications": "on", "tool_progress": "all",
        "long_running_notifications": True, "busy_ack_enabled": True,
    }})
    source = _source()

    # New WhatsApp policy requires explicit platform opt-in for the noisy rails;
    # an existing global display preference remains meaningful for other platforms.
    assert runner._memory_notification_mode_for_source(source) == "off"
    assert runner._progress_notices_enabled_for_source(source) is False
    assert runner._transient_notice_enabled_for_source(source) is True
    assert runner._busy_ack_enabled_for_source(source) is True


def test_malformed_first_config_fails_closed_for_whatsapp() -> None:
    runner = _runner({}, valid=False)
    source = _source()

    assert runner._transient_notice_enabled_for_source(source) is False
    assert runner._memory_notification_mode_for_source(source) == "off"
    assert runner._busy_ack_enabled_for_source(source) is False


def test_persisted_restart_target_uses_the_same_quiet_policy() -> None:
    runner = _runner({})

    assert runner._transient_notice_enabled_for_target(
        Platform.WHATSAPP, "chat-a", profile="jackwhatsapp"
    ) is False


@pytest.mark.asyncio
async def test_platform_notice_boundary_suppresses_provider_billing_and_memory_text() -> None:
    source = _source()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner = _runner({})
    runner._adapter_for_source = lambda _source: adapter
    runner._thread_metadata_for_source = lambda _source: None
    runner.config = None

    for text in (
        "⚠ Empty response from model — retrying (1/3)",
        "⚠ Model fallback: provider quota exhausted",
        "💾 Self-improvement review: Memory updated",
    ):
        await runner._deliver_platform_notice(source, text)
    adapter.send.assert_not_awaited()


def test_status_and_notice_callbacks_do_not_schedule_whatsapp_retry_provider_or_credit_notice() -> None:
    source = _source()
    runner = _runner({})
    runner._transient_notice_enabled_for_source = lambda _source: False
    scheduled: list[object] = []
    ctx = SimpleNamespace(
        source=source,
        _status_adapter=MagicMock(),
        _run_still_current=lambda: True,
        _loop_for_step=None,
        _status_chat_id="chat-a",
        _status_thread_metadata=None,
        _cleanup_progress=False,
        mute_notification_reply=False,
    )
    turn = TurnRunner(runner, ctx)
    turn._schedule = lambda coro, *_args: scheduled.append(coro)

    turn._status_callback_sync("provider.retry", "Empty response from model — retrying (1/3)")
    turn._notice_callback_sync(SimpleNamespace(text="⚠ Credits 90% used"))
    assert scheduled == []


def test_whatsapp_provider_failure_keeps_one_sanitized_final_and_local_keeps_raw() -> None:
    raw = "API call failed: HTTP 429 quota exhausted; request_id=secret-ish"
    whatsapp = _sanitize_gateway_final_response(Platform.WHATSAPP, raw)

    assert whatsapp == "I couldn’t complete your request this time. Please try again later."
    assert _sanitize_gateway_final_response(Platform.LOCAL, raw) == raw


def test_busy_policy_only_changes_ack_visibility_not_event_admission() -> None:
    """The busy handler checks this gate only after it has queued/steered the event."""
    runner = _runner({})
    event = MessageEvent(text="follow-up", message_type=MessageType.TEXT, source=_source(), message_id="m1")
    assert runner._busy_ack_enabled_for_source(event.source) is False


@pytest.mark.asyncio
async def test_busy_drain_mutates_queue_but_keeps_whatsapp_quiet() -> None:
    """A restart/drain must retain the pending turn without manufacturing an ack."""
    runner = _runner({})
    event = MessageEvent(text="follow-up", message_type=MessageType.TEXT, source=_source(), message_id="m-drain")
    adapter = SimpleNamespace(_send_with_retry=AsyncMock())
    queued: list[tuple[str, MessageEvent]] = []
    runner._adapter_for_source = lambda _source: adapter
    runner._queue_during_drain_enabled = lambda _mode: True
    runner._queue_or_replace_pending_event = lambda key, pending: queued.append((key, pending))
    runner._status_action_gerund = lambda: "restarting"

    await runner._send_busy_drain_notice(event, "session-a", "queue")

    assert queued == [("session-a", event)]
    adapter._send_with_retry.assert_not_awaited()


def test_status_transport_rechecks_live_source_policy() -> None:
    """Pre-gated memory/interim callbacks cannot send after a live WhatsApp mute."""
    source = _source()
    runner = _runner({})
    runner._transient_notice_enabled_for_source = lambda _source: False
    scheduled: list[object] = []
    ctx = SimpleNamespace(
        source=source,
        _status_adapter=SimpleNamespace(send=AsyncMock()),
        _run_still_current=lambda: True,
        _status_chat_id="chat-a",
    )
    turn = TurnRunner(runner, ctx)
    turn._schedule = lambda coro, *_args: scheduled.append(coro)

    turn._send_status_text("💾 Memory updated", None, "status send")

    assert scheduled == []
    ctx._status_adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_transport_keeps_other_platforms_compatible() -> None:
    source = _source(Platform.TELEGRAM)
    runner = _runner({})
    runner._transient_notice_enabled_for_source = lambda _source: True
    scheduled: list[object] = []
    adapter = SimpleNamespace(send=AsyncMock())
    ctx = SimpleNamespace(
        source=source,
        _status_adapter=adapter,
        _run_still_current=lambda: True,
        _status_chat_id="chat-a",
    )
    turn = TurnRunner(runner, ctx)
    turn._schedule = lambda coro, *_args: scheduled.append(coro)

    turn._send_status_text("Working", None, "status send")
    assert len(scheduled) == 1
    await scheduled.pop()
    adapter.send.assert_awaited_once_with("chat-a", "Working", metadata=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("revoked", ("policy", "ownership"))
async def test_status_transport_rechecks_policy_and_ownership_after_scheduling(revoked: str) -> None:
    """The scheduled coroutine, rather than only its producer, owns the final send decision."""
    source = _source()
    state = {"policy": True, "ownership": True}
    runner = _runner({})
    runner._transient_notice_enabled_for_source = lambda _source: state["policy"]
    scheduled: list[object] = []
    adapter = SimpleNamespace(send=AsyncMock())
    ctx = SimpleNamespace(
        source=source,
        _status_adapter=adapter,
        _run_still_current=lambda: state["ownership"],
        _status_chat_id="chat-a",
    )
    turn = TurnRunner(runner, ctx)
    turn._schedule = lambda coro, *_args: scheduled.append(coro)

    turn._send_status_text("Working", None, "status send")
    assert len(scheduled) == 1
    state[revoked] = False
    await scheduled.pop()

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_inactivity_warning_respects_whatsapp_transient_policy() -> None:
    source = _source()
    runner = _runner({})
    adapter = SimpleNamespace(send=AsyncMock())
    runner._adapter_for_source = lambda _source: adapter
    runner._transient_notice_enabled_for_source = lambda _source: False
    worker = SimpleNamespace(agent_warning=60, agent_timeout=120)

    await runner._run_agent_inactivity_warning(worker, source, None)

    adapter.send.assert_not_awaited()


def _progress_turn(policy):
    """Build the actual editable-progress path with a minimal native adapter."""
    source = _source()
    ctx = TurnContext(
        source=source,
        _run_still_current=lambda: True,
        progress_queue=queue.Queue(),
        _progress_metadata=None,
        _progress_reply_to=None,
    )
    ctx.agent_holder[0] = SimpleNamespace(is_interrupted=False)
    runner = _runner({})
    runner._progress_notices_enabled_for_source = policy
    return TurnRunner(runner, ctx), ctx


@pytest.mark.asyncio
async def test_whatsapp_progress_policy_gates_real_send_and_drains_queue() -> None:
    """A live quiet policy prevents a queued tool event reaching adapter.send."""
    turn, ctx = _progress_turn(lambda _source: False)
    adapter = SimpleNamespace(
        name="whatsapp",
        MAX_MESSAGE_LENGTH=4000,
        send=AsyncMock(return_value=SendResult(success=True, message_id="p1")),
        edit_message=AsyncMock(return_value=SendResult(success=True, message_id="p1")),
    )
    turn._runner._adapter_for_source = lambda _source: adapter
    ctx.progress_queue.put("🔎 searching")

    await turn.send_progress_messages()

    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
async def test_whatsapp_progress_rechecks_policy_before_edit_failure_fallback_send() -> None:
    """A policy change while an edit fails must not create a fresh progress bubble."""
    decisions = iter((True, True, False))
    turn, _ctx = _progress_turn(lambda _source: next(decisions))
    adapter = SimpleNamespace(
        name="whatsapp",
        MAX_MESSAGE_LENGTH=4000,
        send=AsyncMock(return_value=SendResult(success=True, message_id="new")),
        edit_message=AsyncMock(return_value=SendResult(success=False, error="not editable")),
    )
    st = turn._progress_edit_state(adapter)
    st.progress_msg_id = "old"
    st.progress_lines = ["🔎 searching"]

    assert await turn._progress_send_or_edit(st, "🔎 searching") is True
    adapter.edit_message.assert_awaited_once()
    adapter.send.assert_not_awaited()


def test_notice_policy_uses_scoped_resolver_and_retains_last_known_good(monkeypatch, tmp_path) -> None:
    """Malformed YAML or managed-overlay failure never turns an unseen policy into `{}`."""
    import gateway.run as run_module
    import hermes_cli.config as config_module

    (tmp_path / "config.yaml").write_text("display: {}\n", encoding="utf-8")
    source = _source()
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    monkeypatch.setattr(run_module, "_profile_runtime_scope", lambda _home: contextlib.nullcontext())

    resolved = {"display": {"platforms": {"whatsapp": {"tool_progress": "new"}}}}
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: resolved)
    monkeypatch.setattr(config_module, "get_active_config_parse_failure", lambda: None)
    assert runner._notice_policy_config_for_source(source) == (resolved, True)

    # The official resolver reports a malformed scalar/list YAML while
    # preserving its own cache.  This source cache retains only our previous
    # good mapping; a fresh source would fail closed.
    monkeypatch.setattr(config_module, "get_active_config_parse_failure", lambda: "top-level YAML must be a mapping")
    assert runner._notice_policy_config_for_source(source) == (resolved, True)
    fresh = object.__new__(GatewayRunner)
    fresh.config = SimpleNamespace(multiplex_profiles=False)
    fresh._resolve_profile_home_for_source = lambda _source: tmp_path
    assert fresh._notice_policy_config_for_source(source) == ({}, False)

    # An overlay/resolver failure is likewise not an implicit empty config.
    monkeypatch.setattr(config_module, "get_active_config_parse_failure", lambda: None)
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: (_ for _ in ()).throw(RuntimeError("overlay failed")))
    assert runner._notice_policy_config_for_source(source) == (resolved, True)


@pytest.mark.parametrize("invalid_root", ("[]\n", "false\n", "0\n"))
def test_notice_policy_real_profile_rejects_falsey_non_mapping_roots(tmp_path, invalid_root) -> None:
    """The global resolver's compatibility coercion cannot validate notice policy."""
    source = _source()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("display:\n  platforms:\n    whatsapp:\n      tool_progress: new\n", encoding="utf-8")
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path

    known_good, valid = runner._notice_policy_config_for_source(source)
    assert valid is True
    assert known_good["display"]["platforms"]["whatsapp"]["tool_progress"] == "new"

    config_path.write_text(invalid_root, encoding="utf-8")
    assert runner._notice_policy_config_for_source(source) == (known_good, True)

    unseen = object.__new__(GatewayRunner)
    unseen.config = SimpleNamespace(multiplex_profiles=False)
    unseen._resolve_profile_home_for_source = lambda _source: tmp_path
    assert unseen._notice_policy_config_for_source(source) == ({}, False)


def test_notice_policy_real_profile_accepts_empty_mapping(tmp_path) -> None:
    """An explicitly empty mapping remains a valid, deliberately quiet policy."""
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path

    config, valid = runner._notice_policy_config_for_source(_source())
    assert valid is True
    assert isinstance(config, dict)


@pytest.mark.asyncio
async def test_heartbeat_rechecks_live_policy_before_edit_and_fallback_send(monkeypatch) -> None:
    """A live WhatsApp mute takes effect between heartbeat transport operations."""
    import gateway.run as run_module

    # initial admission; first loop/send; second loop/edit; fallback send
    # (the final false is observed immediately after the failed edit).
    decisions = iter((True, True, True, True, True, False))
    runner = object.__new__(GatewayRunner)
    runner._long_running_notifications_enabled_for_source = lambda _source: next(decisions)
    runner._should_emit_long_running_notification = lambda *_args: True
    runner._agent_activity_summary = lambda _agent: None
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SendResult(success=True, message_id="hb-1")),
        edit_message=AsyncMock(return_value=SendResult(success=False, error="cannot edit")),
    )
    runner._adapter_for_source = lambda _source: adapter
    monkeypatch.setattr(run_module, "_float_env", lambda *_args: 0.001)
    disp = SimpleNamespace(
        _display_surface_mode=lambda *_args, **_kwargs: "on",
        resolve_display_setting=lambda *_args, **_kwargs: False,
        user_config={},
        platform_key="whatsapp",
        _generic_status_phrase=lambda _kind: "Working",
    )
    ctx = TurnContext(source=_source(), session_key="session")
    ctx.agent_holder[0] = SimpleNamespace()

    await runner._run_agent_notify_long_running(disp, ctx, [None])

    adapter.send.assert_awaited_once()  # first heartbeat only
    adapter.edit_message.assert_awaited_once()
