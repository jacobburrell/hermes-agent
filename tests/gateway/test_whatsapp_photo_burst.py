"""WhatsApp photo burst admission at the real bridge polling seam.

The Baileys bridge emits one event per inbound image and carries no album ID.
This test uses its exact event shape: per-image media URL/type/caption, quote
metadata, and a read-receipt key.  A permitted group photo burst must become
one ordered turn while every physical inbound image remains receipted.
"""

import asyncio
import json
import sys
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


_BURST_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "whatsapp_bridge_image_burst.json"
)


class _Response:
    status = 200

    def __init__(self, messages):
        self._messages = messages

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False

    async def json(self):
        return self._messages


class _StopPolling:
    async def __aenter__(self):
        raise asyncio.CancelledError

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False


class _OnePollSession:
    """Serve one /messages batch, then end polling without network I/O."""

    def __init__(self, messages):
        self._messages = messages
        self._calls = 0

    def get(self, _url, **_kwargs):
        self._calls += 1
        if self._calls == 1:
            return _Response(self._messages)
        return _StopPolling()


def _adapter(**extra):
    config_extra = {
        "session_name": "photo-burst-test",
        "group_policy": "open",
        "require_mention": False,
    }
    config_extra.update(extra)
    return WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra=config_extra,
        )
    )


def _message_at(index=0, **updates):
    messages = json.loads(_BURST_FIXTURE.read_text(encoding="utf-8"))
    message = messages[index]
    message.update(updates)
    return message


async def _stop_after_one_poll(adapter):
    poll_task = asyncio.create_task(adapter._poll_messages())
    while adapter._http_session._calls == 0:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    poll_task.cancel()
    with suppress(asyncio.CancelledError):
        await poll_task
    await asyncio.sleep(0)


def test_photo_burst_window_defaults_match_the_adapter_contract():
    assert WhatsAppAdapter._PHOTO_BURST_QUIET_SECONDS == 0.35
    assert WhatsAppAdapter._PHOTO_BURST_MAX_SECONDS == 1.0


@pytest.mark.asyncio
async def test_poll_coalesces_real_shaped_admitted_group_photo_burst(monkeypatch):
    """Seven accepted bridge images must wait, then make one ordered turn."""
    messages = json.loads(_BURST_FIXTURE.read_text(encoding="utf-8"))
    assert len(messages) == 7
    assert all("albumId" not in message for message in messages)

    adapter = _adapter()
    adapter._running = True
    adapter._http_session = _OnePollSession(messages)
    adapter.handle_message = AsyncMock()
    adapter._send_read_receipt = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientTimeout=lambda **_kwargs: object()),
    )
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda _path: True,
    )

    await _stop_after_one_poll(adapter)

    assert adapter._send_read_receipt.await_count == 7
    assert adapter.handle_message.await_count == 0

    await asyncio.sleep(0.40)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == [f"/tmp/wa-photo-{index}.jpg" for index in range(1, 8)]
    assert event.media_types == ["image/jpeg"] * 7
    assert event.text == "\n\n".join(
        f"site photo {word}"
        for word in ("one", "two", "three", "four", "five", "six", "seven")
    )
    assert event.reply_to_message_id == "bot-message-1"
    assert event.reply_to_text == "please review these photos"


@pytest.mark.asyncio
async def test_require_mention_keeps_ambient_group_photos_out_of_receipts_and_turns(
    monkeypatch,
):
    """Admission stays before cache/receipt/burst scheduling for ambient chat."""
    ambient = _message_at(
        hasQuotedMessage=False,
        quotedMessageId=None,
        quotedParticipant=None,
        quotedText="",
    )
    adapter = _adapter(require_mention=True)
    adapter._running = True
    adapter._http_session = _OnePollSession([ambient])
    adapter.handle_message = AsyncMock()
    adapter._send_read_receipt = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientTimeout=lambda **_kwargs: object()),
    )
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda _path: (_ for _ in ()).throw(AssertionError("ambient media must not cache")),
    )

    await _stop_after_one_poll(adapter)
    await asyncio.sleep(WhatsAppAdapter._PHOTO_BURST_QUIET_SECONDS + 0.05)

    adapter._send_read_receipt.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_photo_bursts == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ("mention", "reply", "command", "free_response"))
async def test_existing_group_triggers_still_admit_one_photo_turn(monkeypatch, trigger):
    """The burst buffer is downstream of the unchanged WhatsApp admission gate."""
    updates = {"hasQuotedMessage": False, "quotedMessageId": None, "quotedParticipant": None, "quotedText": ""}
    extra = {"require_mention": True}
    if trigger == "mention":
        updates["mentionedIds"] = ["15559999999@s.whatsapp.net"]
    elif trigger == "reply":
        updates = {}
    elif trigger == "command":
        updates["body"] = "/status"
    else:
        extra["free_response_chats"] = ["120363000000000001@g.us"]

    adapter = _adapter(**extra)
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda _path: True
    )
    event = await adapter._build_message_event(_message_at(**updates))

    assert event is not None
    adapter._enqueue_group_photo_burst(event)
    await asyncio.sleep(WhatsAppAdapter._PHOTO_BURST_QUIET_SECONDS + 0.05)

    adapter.handle_message.assert_awaited_once_with(event)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra", "updates"),
    (
        ({"group_policy": "disabled"}, {}),
        ({}, {"chatId": "status@broadcast"}),
    ),
)
async def test_disallowed_or_broadcast_group_photo_never_enters_burst_buffer(
    extra, updates, monkeypatch
):
    """Existing group access and broadcast exclusions remain before batching."""
    adapter = _adapter(**extra)
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda _path: (_ for _ in ()).throw(AssertionError("rejected media must not cache")),
    )

    assert await adapter._build_message_event(_message_at(**updates)) is None
    assert adapter._pending_photo_bursts == {}


@pytest.mark.asyncio
async def test_photo_burst_hard_cap_and_reply_anchors_keep_dispatches_separate(monkeypatch):
    """One timer is bounded; conflicting quote anchors never merge."""
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(WhatsAppAdapter, "_PHOTO_BURST_QUIET_SECONDS", 0.04)
    monkeypatch.setattr(WhatsAppAdapter, "_PHOTO_BURST_MAX_SECONDS", 0.06)
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda _path: True
    )

    first = await adapter._build_message_event(_message_at(0))
    second = await adapter._build_message_event(_message_at(1, quotedMessageId="bot-message-2"))
    third = await adapter._build_message_event(_message_at(2))
    assert first is not None and second is not None and third is not None

    adapter._enqueue_group_photo_burst(first)
    await asyncio.sleep(0.025)
    adapter._enqueue_group_photo_burst(third)
    adapter._enqueue_group_photo_burst(second)
    await asyncio.sleep(0.08)

    assert adapter.handle_message.await_count == 2
    dispatched = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert [event.reply_to_message_id for event in dispatched] == ["bot-message-1", "bot-message-2"]
    assert dispatched[0].media_urls == ["/tmp/wa-photo-1.jpg", "/tmp/wa-photo-3.jpg"]
    assert dispatched[1].media_urls == ["/tmp/wa-photo-2.jpg"]


@pytest.mark.asyncio
async def test_duplicate_photo_captions_retain_every_physical_occurrence(monkeypatch):
    """Repeated captions describe separate images and must not be de-duplicated."""
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(WhatsAppAdapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda _path: True
    )
    first = await adapter._build_message_event(_message_at(0, body="same caption"))
    second = await adapter._build_message_event(_message_at(1, body="same caption"))
    assert first is not None and second is not None

    adapter._enqueue_group_photo_burst(first)
    adapter._enqueue_group_photo_burst(second)
    await asyncio.sleep(0.03)

    event = adapter.handle_message.await_args.args[0]
    assert event.text == "same caption\n\nsame caption"
    assert event.media_urls == ["/tmp/wa-photo-1.jpg", "/tmp/wa-photo-2.jpg"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "second_updates", "second_profile"),
    (
        (
            "sender",
            {"senderId": "15550000002@s.whatsapp.net"},
            None,
        ),
        ("owner profile", {}, "other-owner"),
    ),
)
async def test_photo_burst_keys_do_not_merge_across_sender_or_owner_profile(
    monkeypatch, boundary, second_updates, second_profile
):
    """The adapter-local burst key keeps independent ingress lanes apart."""
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(WhatsAppAdapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda _path: True
    )
    first = await adapter._build_message_event(_message_at(0))
    second = await adapter._build_message_event(_message_at(1, **second_updates))
    assert first is not None and second is not None
    if second_profile:
        first.source.profile = "photo-burst-test"
        second.source.profile = second_profile

    adapter._enqueue_group_photo_burst(first)
    adapter._enqueue_group_photo_burst(second)
    await asyncio.sleep(0.03)

    assert adapter.handle_message.await_count == 2, boundary
    # These are independent timers, so their callback order is deliberately
    # unspecified.  Assert each singleton by the same identity fields the
    # burst key isolates: profile + sender.
    dispatched = {
        (event.source.profile, event.user_id): event.media_urls
        for event in (call.args[0] for call in adapter.handle_message.await_args_list)
    }
    assert dispatched == {
        (first.source.profile, first.user_id): first.media_urls,
        (second.source.profile, second.user_id): second.media_urls,
    }


@pytest.mark.asyncio
async def test_photo_arriving_during_dispatch_gets_its_own_timer(monkeypatch):
    """A finished timer cannot strand the next physical photo in its map."""
    adapter = _adapter()
    monkeypatch.setattr(WhatsAppAdapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(WhatsAppAdapter, "_PHOTO_BURST_MAX_SECONDS", 0.02)
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda _path: True
    )
    first = await adapter._build_message_event(_message_at(0))
    later = await adapter._build_message_event(_message_at(1))
    assert first is not None and later is not None
    dispatched = []

    async def _dispatch(event):
        dispatched.append(event)
        if len(dispatched) == 1:
            adapter._enqueue_group_photo_burst(later)
            await asyncio.sleep(0.03)

    adapter.handle_message = _dispatch
    adapter._enqueue_group_photo_burst(first)
    await asyncio.sleep(0.08)

    assert [event.media_urls for event in dispatched] == [
        ["/tmp/wa-photo-1.jpg"],
        ["/tmp/wa-photo-2.jpg"],
    ]
    assert adapter._pending_photo_bursts == {}


@pytest.mark.asyncio
async def test_disconnect_cancels_photo_burst_before_bridge_shutdown(monkeypatch):
    """Teardown is explicit: a buffered accepted photo never dispatches late."""
    adapter = _adapter()
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda _path: True
    )
    event = await adapter._build_message_event(_message_at())
    assert event is not None

    adapter._enqueue_group_photo_burst(event)
    adapter._bridge_process = None
    adapter._poll_task = None
    adapter._http_session = SimpleNamespace(closed=True)
    adapter._session_lock_identity = None
    adapter._running = True
    await adapter.disconnect()
    await asyncio.sleep(WhatsAppAdapter._PHOTO_BURST_QUIET_SECONDS + 0.05)

    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_photo_bursts == {}
    assert adapter._pending_photo_burst_tasks == {}
