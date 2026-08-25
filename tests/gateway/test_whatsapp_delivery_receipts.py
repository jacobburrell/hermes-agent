import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent, MessageType, ProcessingOutcome, SendResult,
    merge_pending_message_event,
)
from gateway.session import SessionSource, build_session_key
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


def _event(receipt, kind=MessageType.TEXT):
    return MessageEvent(text="x", message_type=kind, delivery_receipts=[receipt])


def _routed_event(text="x", *, receipt_id="one"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363000000000000@g.us",
            chat_type="group",
            user_id="15550000000@s.whatsapp.net",
        ),
        message_id=f"message-{receipt_id}",
        delivery_receipts=[{"id": receipt_id, "receipt": f"token-{receipt_id}"}],
    )


def _branch_adapter(monkeypatch):
    """Build a WhatsApp adapter that exercises BasePlatformAdapter branches."""
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientTimeout=lambda **_kwargs: None),
    )
    adapter = _make_adapter()
    adapter._bridge_token = "consumer-token"
    adapter._session_tasks = {}
    adapter._expected_cancelled_tasks = set()
    adapter._pending_delivery_terminal_ops = {}
    adapter._inflight_deliveries = {}
    adapter._busy_text_mode = ""
    adapter._busy_session_handler = None
    adapter._text_debounce = {}
    adapter._text_debounce_tasks = {}
    adapter._message_handler = AsyncMock(return_value="")
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="response")
    )
    adapter._http_session = MagicMock()
    adapter._http_session.post = MagicMock(
        return_value=_AsyncCM(MagicMock(status=200))
    )
    return adapter


def _mark_active(adapter, event):
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    receipt = event.delivery_receipts[0]
    adapter._inflight_deliveries[receipt["id"]] = receipt
    return session_key


@pytest.mark.parametrize("kind", [MessageType.TEXT, MessageType.PHOTO, MessageType.VOICE])
def test_base_merge_preserves_every_delivery_receipt(kind):
    pending = {}
    first, second = _event({"id": "one", "receipt": "a"}, kind), _event({"id": "two", "receipt": "b"}, kind)
    merge_pending_message_event(pending, "session", first, merge_text=True)
    merge_pending_message_event(pending, "session", second, merge_text=True)
    assert pending["session"].delivery_receipts == [{"id": "one", "receipt": "a"}, {"id": "two", "receipt": "b"}]


@pytest.mark.asyncio
async def test_success_ack_and_failure_release_retry(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientTimeout=lambda **_kwargs: None))
    adapter = _make_adapter()
    adapter._http_session = MagicMock()
    adapter._bridge_token = "consumer-token"
    adapter._inflight_deliveries = {"one": {"id": "one", "receipt": "a"}}
    adapter._pending_delivery_terminal_ops = {}
    response = MagicMock(status=200)
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(response))
    event = _event({"id": "one", "receipt": "a"})
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert adapter._http_session.post.call_args.args[0].endswith("/ack")
    assert not adapter._inflight_deliveries
    adapter._inflight_deliveries = {"two": {"id": "two", "receipt": "b"}}
    adapter._http_session.post = MagicMock(side_effect=OSError("network"))
    await adapter.on_processing_complete(_event({"id": "two", "receipt": "b"}), ProcessingOutcome.FAILURE)
    assert "two" in adapter._pending_delivery_terminal_ops
    assert "two" in adapter._inflight_deliveries


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [ProcessingOutcome.FAILURE, ProcessingOutcome.CANCELLED])
async def test_failure_and_cancel_release_with_consumer(monkeypatch, outcome):
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientTimeout=lambda **_kwargs: None))
    adapter = _make_adapter(); adapter._http_session = MagicMock(); adapter._bridge_token = "consumer-token"
    adapter._inflight_deliveries = {"one": {"id": "one", "receipt": "a"}}; adapter._pending_delivery_terminal_ops = {}
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(MagicMock(status=200)))
    await adapter.on_processing_complete(_event({"id": "one", "receipt": "a"}), outcome)
    call = adapter._http_session.post.call_args
    assert call.args[0].endswith("/release") and call.kwargs["json"]["consumerId"] == "consumer-token"


@pytest.mark.asyncio
async def test_terminal_retry_flushes_after_network_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientTimeout=lambda **_kwargs: None))
    adapter = _make_adapter(); adapter._http_session = MagicMock(); adapter._bridge_token = "consumer"
    adapter._inflight_deliveries = {"one": {"id": "one", "receipt": "a"}}; adapter._pending_delivery_terminal_ops = {}
    adapter._http_session.post = MagicMock(side_effect=OSError("network"))
    await adapter.on_processing_complete(_event({"id": "one", "receipt": "a"}), ProcessingOutcome.SUCCESS)
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(MagicMock(status=200)))
    await adapter._flush_delivery_terminal_ops()
    assert not adapter._pending_delivery_terminal_ops and not adapter._inflight_deliveries


@pytest.mark.asyncio
async def test_batched_receipts_use_one_ack_request(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientTimeout=lambda **_kwargs: None))
    adapter = _make_adapter(); adapter._http_session = MagicMock(); adapter._bridge_token = "consumer"
    adapter._inflight_deliveries = {str(i): {"id": str(i), "receipt": str(i)} for i in range(2)}
    adapter._pending_delivery_terminal_ops = {str(i): ("ack", {"id": str(i), "receipt": str(i)}) for i in range(2)}
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(MagicMock(status=200)))
    await adapter._flush_delivery_terminal_ops()
    assert adapter._http_session.post.call_count == 1
    assert len(adapter._http_session.post.call_args.kwargs["json"]["deliveries"]) == 2


def test_whatsapp_debounce_merges_all_receipts(monkeypatch):
    adapter = _make_adapter(); adapter._text_batch_key = lambda _event: "s"; adapter._pending_text_batches = {}; adapter._pending_text_batch_tasks = {}
    def close_task(coro):
        coro.close()
        return MagicMock(done=lambda: True, cancel=lambda: None)
    monkeypatch.setattr("asyncio.create_task", close_task)
    first, second = _event({"id": "one", "receipt": "a"}), _event({"id": "two", "receipt": "b"})
    adapter._enqueue_text_event(first); adapter._enqueue_text_event(second)
    assert adapter._pending_text_batches["s"].delivery_receipts == [first.delivery_receipts[0], second.delivery_receipts[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "/stop"])
async def test_active_session_commands_ack_receipt_once(monkeypatch, command):
    adapter = _branch_adapter(monkeypatch)
    event = _routed_event(command)
    _mark_active(adapter, event)
    if command == "/stop":
        adapter._dispatch_active_session_command = AsyncMock(return_value=None)

    await adapter.handle_message(event)

    if command == "/stop":
        adapter._dispatch_active_session_command.assert_awaited_once()
    else:
        adapter._message_handler.assert_awaited_once_with(event)
    assert adapter._http_session.post.call_count == 1
    assert adapter._http_session.post.call_args.args[0].endswith("/ack")
    assert adapter._pending_delivery_terminal_ops == {}
    assert adapter._inflight_deliveries == {}


@pytest.mark.asyncio
async def test_active_session_clarify_reply_acks_receipt_once(monkeypatch):
    adapter = _branch_adapter(monkeypatch)
    event = _routed_event("the second choice")
    _mark_active(adapter, event)

    with patch(
        "tools.clarify_gateway.get_pending_for_session",
        return_value=object(),
    ):
        await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    assert adapter._http_session.post.call_count == 1
    assert adapter._http_session.post.call_args.args[0].endswith("/ack")
    assert adapter._inflight_deliveries == {}


@pytest.mark.asyncio
async def test_active_session_busy_consumed_event_acks_receipt_once(monkeypatch):
    adapter = _branch_adapter(monkeypatch)
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _routed_event("follow up")
    session_key = _mark_active(adapter, event)

    with patch("tools.clarify_gateway.get_pending_for_session", return_value=None):
        await adapter.handle_message(event)

    adapter._busy_session_handler.assert_awaited_once_with(event, session_key)
    adapter._message_handler.assert_not_awaited()
    assert adapter._http_session.post.call_count == 1
    assert adapter._http_session.post.call_args.args[0].endswith("/ack")
    assert adapter._inflight_deliveries == {}


@pytest.mark.asyncio
async def test_active_session_queued_event_has_no_premature_terminal_operation(monkeypatch):
    adapter = _branch_adapter(monkeypatch)
    event = _routed_event("queued follow up")
    session_key = _mark_active(adapter, event)

    with patch("tools.clarify_gateway.get_pending_for_session", return_value=None):
        await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._http_session.post.call_count == 0
    assert adapter._pending_messages[session_key] is event
    assert adapter._pending_delivery_terminal_ops == {}
    assert "one" in adapter._inflight_deliveries


@pytest.mark.asyncio
async def test_policy_rejected_poll_delivery_is_never_dispatched_and_retries_ack(
    monkeypatch,
):
    adapter = _branch_adapter(monkeypatch)
    raw_receipt = {"id": "rejected", "receipt": "lease-token"}
    rejected = {
        "messageId": "forbidden-group-message",
        "chatId": "120363999999999999@g.us",
        "isGroup": True,
        "text": "not admitted",
        "_hermesDelivery": raw_receipt,
    }
    poll_response = MagicMock(status=200)
    poll_response.json = AsyncMock(return_value=[rejected])
    adapter._http_session.post = MagicMock(
        side_effect=[_AsyncCM(poll_response), OSError("ack unavailable")]
    )
    adapter._build_message_event = AsyncMock(return_value=None)
    adapter.handle_message = AsyncMock()

    async def stop_after_cycle(_delay):
        adapter._running = False

    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter.asyncio.sleep", stop_after_cycle
    )

    await adapter._poll_messages()

    adapter._build_message_event.assert_awaited_once_with(rejected)
    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_delivery_terminal_ops["rejected"] == (
        "ack",
        raw_receipt,
    )
    assert adapter._inflight_deliveries["rejected"] == raw_receipt

    adapter._http_session.post = MagicMock(
        return_value=_AsyncCM(MagicMock(status=200))
    )
    await adapter._flush_delivery_terminal_ops()
    assert adapter._pending_delivery_terminal_ops == {}
    assert adapter._inflight_deliveries == {}
