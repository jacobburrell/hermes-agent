"""Behavior contracts for the fail-closed WhatsApp outbound boundary."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.whatsapp_outbound import (
    OUTPUT_KIND_KEY,
    classify_whatsapp_outbound,
)
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


@pytest.mark.parametrize(
    ("content", "metadata"),
    [
        ("Here is the answer you asked for.", {"notify": True}),
        ("Which option should I use?", {OUTPUT_KIND_KEY: "clarify"}),
        ("Approve running this command?", {OUTPUT_KIND_KEY: "approval"}),
        ("/status: connected", {OUTPUT_KIND_KEY: "command"}),
    ],
)
def test_user_visible_whatsapp_output_is_explicitly_classified(content, metadata):
    assert classify_whatsapp_outbound(content, metadata) is not None


@pytest.mark.parametrize(
    ("content", "metadata"),
    [
        ("⚕ Hermes Agent\nGateway restart complete", {"notify": True}),
        ("Traceback (most recent call last):\n  File \"gateway.py\", line 8", {"notify": True}),
        ("Goal persistence resumed for session abc", {"notify": True}),
        ("Internal diagnostic: " + ("x" * 12_000), {"notify": True}),
        ("A normal-looking but unclassified send", {}),
        ("Tool progress: searching", {OUTPUT_KIND_KEY: "final"}),
        ("Actual answer", {"notify": True, "_interim_send": True}),
    ],
)
def test_internal_or_unknown_whatsapp_output_fails_closed(content, metadata):
    assert classify_whatsapp_outbound(content, metadata) is None


@pytest.mark.asyncio
async def test_internal_text_never_reaches_whatsapp_bridge():
    adapter = _make_adapter()
    adapter._outbound_policy = "user_visible_only"
    adapter._trusted_outbound_ids = {}
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._http_session.post = MagicMock(side_effect=AssertionError("must not send"))

    result = await adapter.send(
        "15551234567",
        "Gateway restart complete\nTraceback (most recent call last):",
        metadata={"notify": True},
    )

    assert result.success
    assert result.message_id is None
    adapter._http_session.post.assert_not_called()


@pytest.mark.asyncio
async def test_unclassified_media_never_reaches_whatsapp_bridge(tmp_path):
    adapter = _make_adapter()
    adapter._outbound_policy = "user_visible_only"
    adapter._trusted_outbound_ids = {}
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._http_session.post = MagicMock(side_effect=AssertionError("must not send"))
    image = tmp_path / "image.png"
    image.write_bytes(b"not-a-real-image")

    result = await adapter._send_media_to_bridge("15551234567", str(image), "image")

    assert result.success
    adapter._http_session.post.assert_not_called()


@pytest.mark.asyncio
async def test_final_answer_reaches_whatsapp_bridge_once():
    adapter = _make_adapter()
    adapter._outbound_policy = "user_visible_only"
    adapter._trusted_outbound_ids = {}
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"messageId": "answer-1"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(response))

    result = await adapter.send(
        "15551234567", "Here is the requested answer.", metadata={"notify": True}
    )

    assert result.success
    assert result.message_id == "answer-1"
    assert adapter._http_session.post.call_count == 1
    assert adapter._http_session.post.call_args.kwargs["json"]["outputKind"] == "final"
