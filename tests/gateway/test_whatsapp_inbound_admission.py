"""Adapter-edge contracts for WhatsApp group observation and album admission."""

import asyncio
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType


def _adapter():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        "group_policy": "allowlist",
        "group_allow_from": ["group@g.us"],
        "observed_group_context_limit": 2,
    }))
    adapter._observed_group_context = defaultdict(deque)
    adapter._observed_group_context_limit = 2
    adapter._pending_album_batches = {}
    adapter._pending_album_tasks = {}
    adapter._pending_album_started = {}
    adapter._message_handler = AsyncMock()
    return adapter


def _data(body="ambient", **extra):
    value = {
        "isGroup": True, "chatId": "group@g.us", "chatName": "Group",
        "senderId": "member@s.whatsapp.net", "senderName": "Member",
        "body": body, "botIds": ["bot@s.whatsapp.net"], "mentionedIds": [],
        "quotedParticipant": "", "mediaUrls": [], "hasMedia": False,
    }
    value.update(extra)
    return value


def test_access_drop_precedes_group_observation_and_operation():
    adapter = _adapter()
    assert adapter._classify_inbound_message(_data(chatId="other@g.us")) == "drop"
    assert adapter._classify_inbound_message(_data()) == "observe"
    assert adapter._classify_inbound_message(_data(mentionedIds=["bot@s.whatsapp.net"])) == "operate"
    assert adapter._classify_inbound_message(_data(quotedParticipant="bot@s.whatsapp.net")) == "operate"
    assert adapter._classify_inbound_message(_data("/new")) == "operate"


@pytest.mark.asyncio
async def test_observed_chatter_is_adapter_only_and_attaches_to_next_addressed_turn():
    adapter = _adapter()
    ambient = await adapter._build_message_event(_data("first"), admission="observe")
    adapter._observe_group_event(ambient)
    ambient = await adapter._build_message_event(_data("second"), admission="observe")
    adapter._observe_group_event(ambient)
    addressed = await adapter._build_message_event(
        _data("please help", mentionedIds=["bot@s.whatsapp.net"]), admission="operate"
    )
    adapter._attach_observed_group_context(addressed)
    assert "first" in addressed.metadata["observed_group_context"] and "second" in addressed.metadata["observed_group_context"]
    assert addressed.text == "please help"
    assert adapter._observed_group_context == {}


@pytest.mark.asyncio
async def test_real_adapter_admission_never_sends_observe_to_base_handler():
    adapter = _adapter()
    adapter._text_batch_delay_seconds = 0
    adapter._send_read_receipt = AsyncMock()
    handled = AsyncMock()
    adapter.handle_message = handled
    await adapter._process_inbound_data(_data("ambient"))
    await asyncio.sleep(0)
    handled.assert_not_awaited()
    await adapter._process_inbound_data(_data("help", mentionedIds=["bot@s.whatsapp.net"]))
    await asyncio.sleep(0.02)
    handled.assert_awaited_once()
    assert handled.await_args.args[0].metadata["observed_group_context"].endswith("[Member] ambient")


@pytest.mark.asyncio
async def test_media_album_promotes_all_items_when_any_item_mentions_bot():
    adapter = _adapter()
    handled = AsyncMock()
    adapter.handle_message = handled
    first = await adapter._build_message_event(
        _data("", hasMedia=True, mediaType="image", mediaUrls=["/tmp/a.jpg"], albumId="a"), admission="observe"
    )
    second = await adapter._build_message_event(
        _data("@bot review these", hasMedia=True, mediaType="image", mediaUrls=["/tmp/b.jpg"], albumId="a",
              mentionedIds=["bot@s.whatsapp.net"]), admission="operate"
    )
    first.media_urls = ["/tmp/a.jpg"]
    second.media_urls = ["/tmp/b.jpg"]
    adapter._enqueue_group_media_event(first, "observe")
    adapter._enqueue_group_media_event(second, "operate")
    await asyncio.sleep(0.42)
    handled.assert_awaited_once()
    event = handled.await_args.args[0]
    assert event.media_urls == ["/tmp/a.jpg", "/tmp/b.jpg"]
    assert event.text == "review these"


@pytest.mark.asyncio
async def test_media_without_album_id_only_coalesces_when_reply_anchor_matches():
    adapter = _adapter()
    handled = AsyncMock()
    adapter.handle_message = handled
    source = adapter.build_source("group@g.us", "Group", "group", "member", "Member")
    raw = {"isGroup": True, "chatId": "group@g.us", "mediaUrls": [], "botIds": ["bot"], "mentionedIds": ["bot"]}
    first = MessageEvent("one", MessageType.PHOTO, source=source, reply_to_message_id="anchor", raw_message=raw)
    second = MessageEvent("two", MessageType.PHOTO, source=source, reply_to_message_id="anchor", raw_message=raw)
    adapter._enqueue_group_media_event(first, "operate")
    adapter._enqueue_group_media_event(second, "operate")
    await asyncio.sleep(0.42)
    handled.assert_awaited_once()
    assert handled.await_args.args[0].text == "one\ntwo"


@pytest.mark.asyncio
async def test_unanchored_photo_burst_is_one_operational_envelope():
    adapter = _adapter()
    handled = AsyncMock()
    adapter.handle_message = handled
    source = adapter.build_source("group@g.us", "Group", "group", "member", "Member")
    for number in range(7):
        adapter._enqueue_group_media_event(
            MessageEvent(
                str(number), MessageType.PHOTO, source=source,
                raw_message={"isGroup": True, "chatId": "group@g.us", "botIds": ["bot"], "mentionedIds": ["bot"]},
            ),
            "operate",
        )
    await asyncio.sleep(0.42)
    handled.assert_awaited_once()
    assert handled.await_args.args[0].text == "\n".join(map(str, range(7)))
