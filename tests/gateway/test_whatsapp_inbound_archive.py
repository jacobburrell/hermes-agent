"""Private WhatsApp archive foundation: no model/provider projection."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.platforms.whatsapp.inbound_archive import ArchiveRejected, WhatsAppInboundArchive
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from gateway.platforms.base import MessageType


def _raw(mid="m1", **extra):
    return {"messageId": mid, "chatId": "chat@g.us", "senderId": "1555000@s.whatsapp.net", "body": "ambient", "timestamp": 1, "hasMedia": False, **extra}


def test_observe_operate_duplicate_collision_and_restart(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, accepted = archive.record(_raw(), "observe")
    assert accepted
    assert archive.record(_raw(), "operate") == (event_id, True)  # replay does not consume dispatch semantics
    assert archive.record(_raw(body="conflict"), "operate") == (event_id, False)
    reopened = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    assert reopened.record(_raw(), "observe") == (event_id, True)
    with reopened._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM archive_collision").fetchone()[0] == 1


def test_owned_media_album_order_profile_isolation_and_rejection(tmp_path):
    home = tmp_path / "one"; cache = home / "cache"; cache.mkdir(parents=True)
    owned = cache / "photo.jpg"; owned.write_bytes(b"media")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(_raw(hasMedia=True, mediaType="image", mime="image/jpeg", nativeMetadata={"album": {"groupId": "a", "role": "child", "messageIndex": 2}}), "operate")
    archive.materialize(event_id, _raw(hasMedia=True, mediaType="image", mime="image/jpeg"), [str(owned)])
    with archive._connect() as db:
        row = db.execute("SELECT owned_path,sha256,size,download_status,album_ordinal FROM archive_attachment").fetchone()
        assert row["owned_path"] == str(owned.resolve()) and row["size"] == 5 and row["download_status"] == "owned"
    two = tmp_path / "two"; (two / "cache").mkdir(parents=True)
    assert WhatsAppInboundArchive(two / "whatsapp" / "inbound-archive-v1", two, two / "cache").record(_raw(), "observe")[0] == 1
    with pytest.raises(ArchiveRejected): archive.record(_raw(mid=""), "drop")
    archive.materialize(event_id, _raw(hasMedia=True), [str(cache / "gone.jpg")])
    with archive._connect() as db:
        assert db.execute("SELECT download_status FROM archive_attachment").fetchone()[0] == "deleted_or_replaced"


class _Response:
    status = 200
    def __init__(self, adapter, payload): self.adapter, self.payload = adapter, payload
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return False
    async def json(self): self.adapter._running = False; return self.payload


class _Session:
    def __init__(self, adapter, payload): self.adapter, self.payload = adapter, payload
    def get(self, *args, **kwargs): return _Response(self.adapter, self.payload)


@pytest.mark.asyncio
async def test_adapter_observe_operate_failure_collision_and_retry(monkeypatch, tmp_path):
    """Real poll boundary archives before the one legacy handoff, never after it."""
    async def run(raw, *, admitted, archive_result=(1, True), archive_error=False, retention=True):
        adapter = object.__new__(WhatsAppAdapter)
        adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
        adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
        adapter._is_archive_authorized = Mock(return_value=retention)
        adapter._should_process_message = Mock(return_value=admitted)
        archive = SimpleNamespace(record=Mock(side_effect=OSError("disk") if archive_error else lambda *_: archive_result), materialize=Mock())
        adapter._inbound_archive_instance = Mock(return_value=archive)
        built = SimpleNamespace(message_type=MessageType.DOCUMENT, media_urls=[], media_types=[])
        adapter._build_message_event = AsyncMock(return_value=built)
        adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()
        await adapter._poll_messages()
        return adapter, archive
    raw = _raw()
    observed, archive = await run(raw, admitted=False)
    observed.handle_message.assert_not_awaited(); observed._build_message_event.assert_not_awaited()
    assert observed._should_process_message.call_count == 1 and archive.record.call_count == 1
    operated, archive = await run(raw, admitted=True)
    operated.handle_message.assert_awaited_once(); operated._build_message_event.assert_awaited_once_with(raw, already_admitted=True)
    assert archive.record.call_count == 1
    failed, _ = await run(raw, admitted=True, archive_error=True)
    failed.handle_message.assert_not_awaited()
    collided, _ = await run(raw, admitted=True, archive_result=(1, False))
    collided.handle_message.assert_not_awaited()
    retried, _ = await run(raw, admitted=True, archive_result=(1, True))
    retried.handle_message.assert_awaited_once()
    pairing, archive = await run(raw, admitted=True, retention=False)
    pairing.handle_message.assert_awaited_once(); archive.record.assert_not_called()
    await __import__("asyncio").sleep(0)
    pairing._send_read_receipt.assert_awaited_once_with(raw)


def test_symlink_escape_is_rejected(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    outside = tmp_path / "outside"; outside.write_bytes(b"x")
    link = cache / "escape"; link.symlink_to(outside)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(_raw(hasMedia=True), "operate")
    archive.materialize(event_id, _raw(hasMedia=True), [str(link)])
    with archive._connect() as db:
        assert db.execute("SELECT download_status FROM archive_attachment").fetchone()[0] == "deleted_or_replaced"


def test_root_symlink_message_identity_and_payload_scrub(tmp_path):
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    target = tmp_path / "target"; target.mkdir()
    (home / "whatsapp").mkdir(); (home / "whatsapp" / "inbound-archive-v1").symlink_to(target, target_is_directory=True)
    with pytest.raises(ArchiveRejected): WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    root = home / "safe"; archive = WhatsAppInboundArchive(root, home, home / "cache")
    event_id, _ = archive.record(_raw(mid="MiD:7", senderId="1555:7@s.whatsapp.net", mediaUrls=["https://secret"], _inboundLease={"token":"secret"}), "observe")
    with archive._connect() as db:
        row = db.execute("SELECT message_id,sender_id,payload_json FROM archive_event WHERE id=?", (event_id,)).fetchone()
        assert row["message_id"] == "MiD:7" and row["sender_id"] == "1555:7@s.whatsapp.net"
        assert "secret" not in row["payload_json"] and "mediaUrls" not in row["payload_json"]


def test_ancestor_and_database_replacement_are_refused_before_reopen(tmp_path):
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    archive.path.unlink(); archive.path.symlink_to(tmp_path / "outside-db")
    with pytest.raises(ArchiveRejected): archive.record(_raw(), "observe")
    # A profile-relative ancestor is checked before a fresh archive is created.
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "whatsapp").symlink_to(tmp_path / "target", target_is_directory=True)
    with pytest.raises(ArchiveRejected): WhatsAppInboundArchive(bad / "whatsapp" / "inbound-archive-v1", bad, bad / "cache")
