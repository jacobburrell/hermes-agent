"""Behavior contracts for the fail-closed WhatsApp outbound boundary."""

import sys
from types import SimpleNamespace
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


INTERNAL_EVENTS = [
    "[SILENT]",
    "No reply: agent is still working",
    "Memory updated: saved preference",
    "Self-improvement review completed",
    "Context compression started",
    "Conversation limit reached",
    "Session reset after interruption",
    "Session restored from history",
    "History cleared",
    "Background process started",
    "Provider error: upstream unavailable",
    "Model failure: retrying",
    "API error: invalid response",
    "Token exhausted",
    "Credential depleted",
    "HTTP 503 provider failure",
    "Internal reasoning fallback engaged",
    "⚕ Hermes Agent\nGateway restart complete",
    "Goal persistence resumed for session abc",
    "Traceback (most recent call last):\n  File \"gateway.py\", line 8",
    "Internal diagnostic: " + ("x" * 12_000),
]


@pytest.mark.parametrize("content", INTERNAL_EVENTS)
def test_internal_events_fail_closed_even_with_forged_final_kind(content):
    metadata = {OUTPUT_KIND_KEY: "final"}
    assert classify_whatsapp_outbound(content, metadata) is None
    assert classify_whatsapp_outbound(content, metadata) is None


@pytest.mark.parametrize("content", ["", "   ", "A normal-looking but unclassified send"])
def test_blank_or_unknown_text_fails_closed(content):
    assert classify_whatsapp_outbound(content, {OUTPUT_KIND_KEY: "final"}) is None


def test_ordinary_error_explanation_remains_user_visible():
    assert classify_whatsapp_outbound(
        "HTTP 429 means the provider asked us to retry later.", {OUTPUT_KIND_KEY: "final"}
    ) == "final"


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
async def test_final_answer_reaches_whatsapp_bridge_once(monkeypatch):
    adapter = _make_adapter()
    adapter._outbound_policy = "user_visible_only"
    adapter._trusted_outbound_ids = {}
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientTimeout=lambda **_kwargs: None))
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
