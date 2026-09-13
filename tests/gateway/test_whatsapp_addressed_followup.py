"""Durable, opt-in WhatsApp addressed-followup admission."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from plugins.platforms.whatsapp.inbound_archive import WhatsAppInboundArchive


CHAT = "120363001234567890@g.us"
SENDER = "15550001111@s.whatsapp.net"
BOT = "15551230000@s.whatsapp.net"


def _message(message_id: str, body: str, *, sender: str = SENDER, mentioned=False):
    return {
        "messageId": message_id,
        "chatId": CHAT,
        "senderId": sender,
        "from": sender,
        "isGroup": True,
        "body": body,
        "timestamp": 1,
        "hasMedia": False,
        "botIds": [BOT],
        "mentionedIds": [BOT] if mentioned else [],
    }


def _adapter(home: Path, *, window=30):
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={
        "require_mention": True,
        "group_policy": "open",
        "addressed_followup_window_seconds": window,
    })
    adapter._dm_policy = "pairing"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []
    adapter._inbound_archive_home = home
    adapter._inbound_archive = WhatsAppInboundArchive(
        home / "whatsapp" / "inbound-archive-v1", home, home / "cache",
    )
    return adapter


def _record(adapter, data):
    """Mirror the adapter's poll-path archive handoff without a transport mock."""
    admitted = adapter._should_process_message(data)
    followup = data.get("_addressed_followup_context")
    explicit = adapter._is_explicit_group_trigger(data)
    free_response = adapter._normalize_whatsapp_id(data["chatId"]) in adapter._whatsapp_free_response_chats()
    anchor = bool(
        admitted and (explicit or free_response)
    )
    preserve = bool(admitted and not anchor and isinstance(followup, dict))
    adapter._inbound_archive_instance().record(
        data, "operate" if admitted else "observe",
        followup_anchor=anchor,
        preserve_followup_anchor=preserve,
        followup_chat_id=adapter._normalize_whatsapp_id(data["chatId"]),
        followup_sender_id=adapter._normalize_whatsapp_id(data["senderId"]),
    )
    return admitted


class _OnePoll:
    """One real adapter poll response, then stop before the next interval."""
    def __init__(self, adapter, message):
        self.adapter, self.message = adapter, message

    def get(self, *_args, **_kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    @property
    def status(self):
        return 200

    async def json(self):
        self.adapter._running = False
        return [self.message]


async def _poll_once(adapter, message, captured):
    adapter._running = True
    adapter._http_session = _OnePoll(adapter, message)
    adapter._bridge_port = 1
    adapter._bridge_req = lambda *_args, **_kwargs: adapter._http_session.get()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = lambda _data: True
    adapter._send_read_receipts = False
    adapter._send_read_receipt = AsyncMock()
    adapter._archive_manifest_capability = object()
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    adapter._enqueue_text_event = captured.append
    adapter.handle_message = AsyncMock()
    await adapter._poll_messages()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_same_sender_addressed_followup_is_durable_and_turn_local(tmp_path):
    home = tmp_path / "jackwhatsapp"; (home / "cache").mkdir(parents=True)
    adapter = _adapter(home)
    first = _message("addressed", "@Jack review item A", mentioned=True)
    captured = []
    await _poll_once(adapter, first, captured)
    assert len(captured) == 1

    # Reopening the archive models a gateway restart.  There is no in-memory
    # conversation flag to accidentally broaden admission after a restart.
    restarted = _adapter(home)
    followup = _message("correction", "Actually, use item B instead.")
    restarted_events = []
    await _poll_once(restarted, followup, restarted_events)
    assert len(restarted_events) == 1
    context = followup["_addressed_followup_context"]
    assert context == {"message_id": "addressed", "text": "@Jack review item A"}

    event = restarted_events[0]
    prompt = GatewayRunner._prepend_inbound_reply_context(event, event.source, event.text)
    assert prompt.startswith('[Continuing your just-addressed message: "@Jack review item A"]')
    assert prompt.endswith("Actually, use item B instead.")


@pytest.mark.asyncio
async def test_continuation_chain_cannot_extend_original_explicit_deadline(tmp_path):
    home = tmp_path / "jackwhatsapp"; (home / "cache").mkdir(parents=True)
    adapter = _adapter(home, window=30)
    first_events = []
    await _poll_once(adapter, _message("explicit", "@Jack review item A", mentioned=True), first_events)
    with adapter._inbound_archive_instance()._connect() as db:
        original = db.execute("SELECT created_at FROM archive_addressed_followup_anchor").fetchone()[0]

    early_events = []
    await _poll_once(adapter, _message("early", "Correction: use item B."), early_events)
    assert len(early_events) == 1
    with adapter._inbound_archive_instance()._connect() as db:
        assert db.execute("SELECT created_at FROM archive_addressed_followup_anchor").fetchone()[0] == original
        db.execute("UPDATE archive_addressed_followup_anchor SET created_at=?", (time.time() - 31,))

    late_events = []
    await _poll_once(adapter, _message("late", "And item C."), late_events)
    assert late_events == []
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambient_lookalikes_other_senders_and_intervening_messages_do_not_admit(tmp_path):
    home = tmp_path / "jackwhatsapp"; (home / "cache").mkdir(parents=True)
    adapter = _adapter(home)
    addressed_events = []
    await _poll_once(adapter, _message("addressed", "@Jack review item A", mentioned=True), addressed_events)
    assert len(addressed_events) == 1

    # Same prose from another person cannot borrow the anchor. It is archived
    # but creates neither a model queue item nor a native outbound send.
    ambient_events = []
    await _poll_once(adapter, _message("intervening", "ordinary group chatter", sender="15550002222@s.whatsapp.net"), ambient_events)
    assert ambient_events == []
    adapter.handle_message.assert_not_awaited()
    adapter._send_read_receipt.assert_not_awaited()
    # The authorized ambient event also durably closes the first sender's
    # continuation window.
    assert adapter._should_process_message(_message("late", "use item B")) is False


def test_followup_window_profile_isolation_and_expiry_fail_closed(tmp_path):
    first_home = tmp_path / "jackwhatsapp"; second_home = tmp_path / "other"
    for home in (first_home, second_home):
        (home / "cache").mkdir(parents=True)
    first = _adapter(first_home, window=1)
    assert _record(first, _message("addressed", "@Jack review item A", mentioned=True)) is True
    assert _adapter(second_home, window=30)._should_process_message(_message("wrong-profile", "use item B")) is False

    with first._inbound_archive_instance()._connect() as db:
        db.execute("UPDATE archive_addressed_followup_anchor SET created_at=?", (time.time() - 2,))
    assert first._should_process_message(_message("expired", "use item B")) is False


def test_disabled_or_invalid_window_keeps_groups_explicit_only(tmp_path):
    home = tmp_path / "jackwhatsapp"; (home / "cache").mkdir(parents=True)
    for window in (0, False, "not-a-number", 121):
        profile_home = home / str(window)
        (profile_home / "cache").mkdir(parents=True)
        adapter = _adapter(profile_home, window=window)
        assert _record(adapter, _message(f"addressed-{window}", "@Jack review item A", mentioned=True)) is True
        assert adapter._should_process_message(_message(f"plain-{window}", "use item B")) is False
