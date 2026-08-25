import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import (
    MessageEvent, MessageType, ProcessingOutcome, merge_pending_message_event,
)
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


def _event(receipt, kind=MessageType.TEXT):
    return MessageEvent(text="x", message_type=kind, delivery_receipts=[receipt])


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
