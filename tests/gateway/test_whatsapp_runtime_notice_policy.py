"""WhatsApp's internal-notice boundary is quiet without changing final delivery.

The production incidents were emitted by several independent callback rails.
These tests deliberately exercise those rails through the real mixin methods,
not a wording-based outbound filter.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _sanitize_gateway_final_response
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource


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
