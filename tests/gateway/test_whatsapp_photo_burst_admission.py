"""Admission-first photo burst handling for the WhatsApp bridge adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


_CHAT_ID = "120363001234567890@g.us"
_BOT_ID = "15551230000@s.whatsapp.net"


def _adapter() -> WhatsAppAdapter:
    """Construct only the adapter state used by the ingress boundary."""
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(
        enabled=True,
        extra={
            "group_policy": "allowlist",
            "group_allow_from": [_CHAT_ID],
            "require_mention": True,
            "send_read_receipts": True,
        },
    )
    adapter._group_policy = "allowlist"
    adapter._group_allow_from = {_CHAT_ID}
    adapter._dm_policy = "pairing"
    adapter._allow_from = set()
    adapter._mention_patterns = []
    adapter._owner_profile = "owner-a"
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_photo_bursts = {}
    adapter._pending_photo_burst_tasks = {}
    adapter._pending_photo_burst_hard_tasks = {}
    adapter._pending_photo_burst_started = {}
    adapter._shutting_down = False
    adapter._bridge_process = None
    adapter._poll_task = None
    adapter._send_read_receipts = True
    adapter._http_session = object()
    adapter._bridge_port = 3000
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = MagicMock()
    adapter._send_read_receipt = AsyncMock()
    return adapter


def _raw_group_photo(
    *,
    body: str = "photo",
    message_id: str = "message-1",
    sender_id: str = "sender-a@s.whatsapp.net",
    album_id: str | None = None,
    mentioned: bool = False,
    reply_anchor: str | None = None,
) -> dict:
    """Bridge-shaped photo event; albumId is deliberately optional."""
    payload = {
        "isGroup": True,
        "chatId": _CHAT_ID,
        "senderId": sender_id,
        "senderName": "Member",
        "body": body,
        "messageId": message_id,
        "hasMedia": True,
        "mediaType": "image",
        "mediaUrls": [f"/tmp/{message_id}.jpg"],
        "readReceiptKey": {"id": message_id},
        "botIds": [_BOT_ID],
        "mentionedIds": [_BOT_ID] if mentioned else [],
        "hasQuotedMessage": reply_anchor is not None,
        "quotedMessageId": reply_anchor,
        "quotedParticipant": _BOT_ID if reply_anchor else "",
    }
    if album_id is not None:
        payload["albumId"] = album_id
    return payload


def _photo_event(
    raw: dict,
    *,
    profile: str = "owner-a",
    reply_anchor: str | None = None,
) -> MessageEvent:
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=raw["chatId"],
        chat_type="group",
        user_id=raw["senderId"],
        user_name=raw["senderName"],
        profile=profile,
    )
    return MessageEvent(
        text=raw["body"],
        message_type=MessageType.PHOTO,
        source=source,
        raw_message=raw,
        message_id=raw["messageId"],
        media_urls=[raw["mediaUrls"][0]],
        media_types=["image/jpeg"],
        reply_to_message_id=reply_anchor,
        reply_to_text="bot context" if reply_anchor else None,
        reply_to_author_id=_BOT_ID if reply_anchor else None,
        reply_to_is_own_message=reply_anchor is not None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("has_media", "media_type"),
    [
        (False, ""),
        (True, "image"),
        (True, "video"),
        (True, "audio"),
        (True, "document"),
    ],
)
async def test_ambient_group_events_do_no_ingress_work(has_media, media_type):
    """Unaddressed group traffic exits before build/cache/receipt/timers/Base."""
    adapter = _adapter()
    raw = _raw_group_photo(mentioned=False)
    raw["hasMedia"] = has_media
    raw["mediaType"] = media_type
    raw["body"] = "ambient traffic"
    adapter._build_message_event = AsyncMock()

    await adapter._process_inbound_data(raw)
    await asyncio.sleep(0)

    adapter._build_message_event.assert_not_awaited()
    adapter._send_read_receipt.assert_not_awaited()
    adapter._enqueue_text_event.assert_not_called()
    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_photo_bursts == {}
    assert adapter._pending_photo_burst_tasks == {}
    assert adapter._pending_photo_burst_hard_tasks == {}


@pytest.mark.asyncio
async def test_admitted_bridge_photo_burst_without_album_id_dispatches_once(monkeypatch):
    """An addressed raw bridge burst retains ordered duplicate captions once."""
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(adapter, "_PHOTO_BURST_HARD_CAP_SECONDS", 0.04)
    raw_events = [
        _raw_group_photo(body="same", message_id="photo-1", mentioned=True, reply_anchor="bot-1"),
        _raw_group_photo(body="same", message_id="photo-2", mentioned=True, reply_anchor="bot-1"),
        _raw_group_photo(body="third", message_id="photo-3", mentioned=True, reply_anchor="bot-1"),
    ]
    built_events = [
        _photo_event(raw, reply_anchor="bot-1") for raw in raw_events
    ]
    adapter._build_message_event = AsyncMock(side_effect=built_events)

    for raw in raw_events:
        # No albumId: this exercises the raw bridge burst fallback.
        assert "albumId" not in raw
        await adapter._process_inbound_data(raw)

    await asyncio.sleep(0.08)

    adapter.handle_message.assert_awaited_once()
    merged = adapter.handle_message.await_args.args[0]
    assert merged.media_urls == [
        "/tmp/photo-1.jpg",
        "/tmp/photo-2.jpg",
        "/tmp/photo-3.jpg",
    ]
    assert merged.text == "same\nsame\nthird"
    assert merged.reply_to_message_id == "bot-1"
    assert merged.reply_to_text == "bot context"
    assert merged.reply_to_is_own_message is True
    assert adapter._send_read_receipt.await_count == 3
    assert adapter._pending_photo_bursts == {}


@pytest.mark.asyncio
async def test_ambient_photo_never_extends_an_admitted_burst(monkeypatch):
    """An unmentioned follow-up cannot join an already-admitted album lane."""
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(adapter, "_PHOTO_BURST_HARD_CAP_SECONDS", 0.04)
    addressed = _raw_group_photo(message_id="addressed", mentioned=True)
    ambient = _raw_group_photo(message_id="ambient", mentioned=False)
    adapter._build_message_event = AsyncMock(
        return_value=_photo_event(addressed)
    )

    await adapter._process_inbound_data(addressed)
    await adapter._process_inbound_data(ambient)
    await asyncio.sleep(0.08)

    adapter._build_message_event.assert_awaited_once_with(addressed)
    adapter._send_read_receipt.assert_awaited_once_with(addressed)
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].media_urls == [
        "/tmp/addressed.jpg"
    ]


@pytest.mark.asyncio
async def test_partial_new_adapter_without_shutdown_flag_admits_normally(monkeypatch):
    """Ingress remains compatible with narrow ``__new__`` adapter fixtures."""
    adapter = _adapter()
    del adapter._shutting_down
    monkeypatch.setattr(adapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(adapter, "_PHOTO_BURST_HARD_CAP_SECONDS", 0.04)
    raw = _raw_group_photo(message_id="partial", mentioned=True)
    adapter._build_message_event = AsyncMock(return_value=_photo_event(raw))

    await adapter._process_inbound_data(raw)
    await asyncio.sleep(0.08)

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].message_id == "partial"


@pytest.mark.asyncio
async def test_continuous_admitted_arrivals_flush_at_hard_cap(monkeypatch):
    """New admitted photos cannot perpetually reset the one-second cap."""
    adapter = _adapter()
    # Keep the behavioral ratio while making the concurrency test fast.
    monkeypatch.setattr(adapter, "_PHOTO_BURST_QUIET_SECONDS", 0.02)
    monkeypatch.setattr(adapter, "_PHOTO_BURST_HARD_CAP_SECONDS", 0.05)

    for index in range(5):
        adapter._enqueue_admitted_photo_burst(
            _photo_event(
                _raw_group_photo(message_id=f"cap-{index}", mentioned=True)
            )
        )
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.06)

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].media_urls == [
        f"/tmp/cap-{index}.jpg" for index in range(5)
    ]


@pytest.mark.asyncio
async def test_simultaneous_quiet_and_hard_flush_dispatches_once(monkeypatch):
    """The quiet and hard timers race safely through one pop-and-dispatch."""
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(adapter, "_PHOTO_BURST_HARD_CAP_SECONDS", 0.01)
    adapter._enqueue_admitted_photo_burst(
        _photo_event(_raw_group_photo(message_id="race", mentioned=True))
    )

    await asyncio.sleep(0.05)

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].message_id == "race"


@pytest.mark.asyncio
async def test_photo_burst_keys_do_not_merge_owner_sender_reply_or_album(monkeypatch):
    """Independent source lanes dispatch independently; only same keys merge."""
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_PHOTO_BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(adapter, "_PHOTO_BURST_HARD_CAP_SECONDS", 0.04)

    first = _photo_event(_raw_group_photo(message_id="first", mentioned=True), profile="owner-a")
    other_sender = _photo_event(
        _raw_group_photo(message_id="sender", sender_id="sender-b@s.whatsapp.net", mentioned=True),
        profile="owner-a",
    )
    other_profile = _photo_event(
        _raw_group_photo(message_id="profile", mentioned=True), profile="owner-b"
    )
    other_reply = _photo_event(
        _raw_group_photo(message_id="reply", mentioned=True, reply_anchor="bot-2"),
        profile="owner-a",
        reply_anchor="bot-2",
    )
    other_album = _photo_event(
        _raw_group_photo(message_id="album", mentioned=True, album_id="album-2"),
        profile="owner-a",
    )

    for event in (first, other_sender, other_profile, other_reply, other_album):
        adapter._enqueue_admitted_photo_burst(event)
    await asyncio.sleep(0.08)

    assert adapter.handle_message.await_count == 5
    dispatched = {
        event.message_id: event
        for (event,), _kwargs in adapter.handle_message.await_args_list
    }
    assert set(dispatched) == {"first", "sender", "profile", "reply", "album"}
    assert all(len(event.media_urls) == 1 for event in dispatched.values())


@pytest.mark.asyncio
async def test_teardown_cancels_admitted_photo_burst(monkeypatch):
    """Deferred admitted media never dispatches after adapter teardown."""
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_PHOTO_BURST_QUIET_SECONDS", 1.0)
    monkeypatch.setattr(adapter, "_PHOTO_BURST_HARD_CAP_SECONDS", 1.0)
    adapter._enqueue_admitted_photo_burst(
        _photo_event(_raw_group_photo(message_id="late", mentioned=True))
    )

    adapter._cancel_pending_photo_bursts()
    await asyncio.sleep(0)

    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_photo_bursts == {}
    assert adapter._pending_photo_burst_tasks == {}
    assert adapter._pending_photo_burst_hard_tasks == {}


@pytest.mark.asyncio
async def test_disconnect_blocks_inflight_builder_before_receipt_or_dispatch(tmp_path):
    """Disconnect is an ingress barrier even if a builder absorbs cancellation."""
    adapter = _adapter()
    adapter._session_path = tmp_path / "session"
    adapter._session_path.mkdir()
    adapter._http_session = None
    adapter._release_platform_lock = MagicMock()
    adapter._release_bridge_port_lock = MagicMock()
    adapter._mark_disconnected = MagicMock()
    adapter._close_bridge_log = MagicMock()
    raw = _raw_group_photo(message_id="inflight", mentioned=True)
    built = _photo_event(raw)
    build_started = asyncio.Event()
    release_build = asyncio.Event()

    async def cancellation_resistant_build(_raw):
        build_started.set()
        try:
            await release_build.wait()
        except asyncio.CancelledError:
            # Simulate a lower-level operation that completes after its caller
            # is cancelled; the post-build shutdown check must still stop it.
            await release_build.wait()
        return built

    adapter._build_message_event = cancellation_resistant_build
    adapter._poll_task = asyncio.create_task(adapter._process_inbound_data(raw))
    await build_started.wait()

    disconnect = asyncio.create_task(adapter.disconnect())
    await asyncio.sleep(0)
    assert not disconnect.done()
    release_build.set()
    await disconnect

    adapter._send_read_receipt.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_photo_bursts == {}
    assert adapter._pending_photo_burst_tasks == {}
    assert adapter._pending_photo_burst_hard_tasks == {}
