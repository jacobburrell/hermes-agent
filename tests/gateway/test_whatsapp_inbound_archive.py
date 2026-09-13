"""Private WhatsApp archive foundation: no model/provider projection."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.platforms.whatsapp.inbound_archive import (
    ArchiveRejected, MaterializationResult, WhatsAppInboundArchive, bridge_event_digest,
)
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from gateway.platforms.base import MessageType
from gateway.platforms.event import MessageEvent


@pytest.fixture(autouse=True)
def _aiohttp_timeout_for_injected_bridge_sessions(monkeypatch):
    """Keep fake bridge transport tests independent of optional aiohttp.

    The fixtures inject a context-manager-only session; ``_bridge_req`` still
    imports aiohttp only to build its timeout object.  Without this tiny fake,
    the poll loop retries an import failure instead of consuming the injected
    response.  Real aiohttp connection behavior remains outside this fixture.
    """
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "aiohttp",
            SimpleNamespace(ClientTimeout=lambda *, total: SimpleNamespace(total=total)),
        )


def _raw(mid="m1", **extra):
    return {"messageId": mid, "chatId": "chat@g.us", "senderId": "1555000@s.whatsapp.net", "body": "ambient", "timestamp": 1, "hasMedia": False, **extra}


def _lease(*, delivery="a" * 64, digest="b" * 64, consumer="test-consumer", token="test-token", epoch=1):
    return {
        "consumerId": consumer, "deliveryId": delivery, "eventDigest": digest,
        "token": token, "epoch": epoch, "expiresAt": int(time.time() * 1000) + 60_000,
    }


def _leased_raw(*, lease_kwargs=None, **raw_kwargs):
    raw = _raw(**raw_kwargs)
    raw["_inboundLease"] = _lease(**(lease_kwargs or {}))
    raw["_inboundLease"]["eventDigest"] = bridge_event_digest(raw)
    return raw


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


def test_bridge_receipt_is_profile_scoped_fenced_and_restart_safe(tmp_path):
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    event_id, accepted = archive.record(_raw(mid="leased"), "operate")
    assert accepted
    first = {
        "deliveryId": "a" * 64, "eventDigest": "b" * 64,
        "consumerId": "python-owner", "epoch": 1, "token": "first-token",
    }
    receipt = archive.bind_bridge_receipt(event_id, first)
    assert not receipt.ready and not receipt.acknowledged and not receipt.recovery_pending
    receipt = archive.mark_bridge_receipt_ready(receipt)
    assert receipt.ready and archive.pending_bridge_receipts() == [receipt]

    renewed = {**first, "token": "renewed-token"}
    renewed_receipt = archive.renew_bridge_receipt(receipt, renewed)
    assert renewed_receipt.request["token"] == "renewed-token"
    assert not archive.mark_bridge_receipt_acked(receipt)
    restarted = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    [pending] = restarted.pending_bridge_receipts()
    assert pending == renewed_receipt
    pending = restarted.prepare_bridge_handoff(pending)
    assert pending.recovery_pending
    assert restarted.mark_bridge_receipt_acked(pending)
    assert restarted.pending_bridge_receipts() == []
    [recovery] = restarted.pending_bridge_recoveries()
    assert recovery.delivery_id == pending.delivery_id and recovery.recovery_pending

    # A same opaque delivery ID may never be reused with a different bridge
    # digest, including after a process restart and before any ACK.
    next_event, _ = restarted.record(_raw(mid="leased-next"), "operate")
    with pytest.raises(ArchiveRejected, match="digest collision"):
        restarted.bind_bridge_receipt(next_event, {**first, "eventDigest": "c" * 64})


def test_bridge_receipt_rejects_delayed_epoch_and_same_epoch_binding_replays(tmp_path):
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    event_id, _ = archive.record(_raw(mid="epoch"), "operate")
    original = {
        "deliveryId": "a" * 64, "eventDigest": "b" * 64,
        "consumerId": "consumer-a", "epoch": 1, "token": "token-a",
    }
    current = {**original, "consumerId": "consumer-b", "epoch": 2, "token": "token-b"}
    archive.bind_bridge_receipt(event_id, original)
    updated = archive.bind_bridge_receipt(event_id, current)
    assert updated.request["consumerId"] == "consumer-b" and updated.request["epoch"] == 2

    # A delayed poll from the former owner must not overwrite the new holder.
    with pytest.raises(ArchiveRejected, match="stale.*epoch"):
        archive.bind_bridge_receipt(event_id, original)
    # Nor may a same-generation replay substitute a token or consumer.
    with pytest.raises(ArchiveRejected, match="incompatible.*binding"):
        archive.bind_bridge_receipt(event_id, {**current, "token": "different-token"})
    with pytest.raises(ArchiveRejected, match="incompatible.*binding"):
        archive.bind_bridge_receipt(event_id, {**current, "consumerId": "consumer-c"})


@pytest.mark.asyncio
async def test_acked_handoff_remains_recoverable_but_duplicate_poll_never_dispatches(tmp_path):
    """An ACKed bridge delivery is recoverable metadata, never a re-run turn."""
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    raw = _leased_raw(mid="handoff")
    event_id, accepted = archive.record(raw, "operate")
    assert accepted
    receipt = archive.bind_bridge_receipt(event_id, raw["_inboundLease"])
    receipt = archive.mark_bridge_receipt_ready(receipt)
    receipt = archive.prepare_bridge_handoff(receipt)
    assert archive.mark_bridge_receipt_acked(receipt)
    [recovery] = archive.pending_bridge_recoveries()
    assert recovery.delivery_id == receipt.delivery_id and recovery.acknowledged

    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._inbound_consumer_id = "test-consumer"
    adapter._http_session = _Session(adapter, [raw])
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = Mock(return_value=True)
    adapter._should_process_message = Mock(return_value=True)
    adapter._inbound_archive_instance = Mock(return_value=archive)
    adapter._build_message_event = AsyncMock()
    adapter.handle_message = AsyncMock()

    await adapter._poll_messages()

    adapter._build_message_event.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


def test_bridge_recovery_reservation_is_direct_only_and_generation_fenced(tmp_path):
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    root = home / "whatsapp" / "inbound-archive-v1"

    def prepared(archive, raw, delivery, digest):
        event_id, _ = archive.record(raw, "operate")
        receipt = archive.bind_bridge_receipt(event_id, _lease(delivery=delivery, digest=digest))
        receipt = archive.mark_bridge_receipt_ready(receipt)
        receipt = archive.prepare_bridge_handoff(receipt)
        assert archive.mark_bridge_receipt_acked(receipt)

    direct = WhatsAppInboundArchive(root, home, home / "cache")
    prepared(direct, _raw(mid="direct", chatId="1555@s.whatsapp.net", isGroup=False), "1" * 64, "2" * 64)
    # Separate SQLite connections emulate concurrent startup workers.  Only
    # one can reserve; its live pid/start fence keeps the other from rearming.
    other = WhatsAppInboundArchive(root, home, home / "cache")
    with ThreadPoolExecutor(max_workers=2) as pool:
        reservations = list(pool.map(lambda archive: archive.reserve_bridge_recovery("1" * 64), (direct, other)))
    [recovery] = [item for item in reservations if item is not None]
    assert recovery.chat_id == "1555@s.whatsapp.net" and not recovery.is_group and recovery.generation == 1
    assert sum(item is not None for item in reservations) == 1
    assert direct.register_bridge_recovery_delivery(recovery, "ledger-row")
    assert not other.register_bridge_recovery_delivery(recovery, "ledger-row-two")
    assert direct.settle_bridge_recovery_delivery(recovery, "ledger-row")
    assert direct.pending_bridge_recoveries() == []

    grouped = WhatsAppInboundArchive(home / "whatsapp" / "group-archive", home, home / "cache")
    prepared(grouped, _raw(mid="group", isGroup=True), "3" * 64, "4" * 64)
    assert grouped.reserve_bridge_recovery("3" * 64) is None
    assert grouped.pending_bridge_recoveries() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lease",
    [
        _lease(consumer="another-consumer"),
        {**_lease(), "eventDigest": "not-a-digest"},
        {**_lease(), "expiresAt": 1},
    ],
    ids=("consumer-mismatch", "bad-digest", "expired"),
)
async def test_adapter_rejects_stale_or_mismatched_lease_before_archive_ack_or_dispatch(lease):
    raw = _leased_raw(mid="invalid-lease")
    raw["_inboundLease"].update(lease)
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._inbound_consumer_id = "test-consumer"
    adapter._http_session = _Session(adapter, [raw])
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._inbound_archive_instance = Mock()
    adapter._ack_inbound_receipt = AsyncMock(return_value=True)
    adapter._build_message_event = AsyncMock()
    adapter.handle_message = AsyncMock()

    await adapter._poll_messages()

    adapter._inbound_archive_instance.assert_not_called()
    adapter._ack_inbound_receipt.assert_not_awaited()
    adapter._build_message_event.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_leased_media_is_owned_and_handoff_is_durable_before_bridge_ack(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    source = cache / "photo.jpg"; source.write_bytes(b"owned-before-ack")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raw = _leased_raw(
        mid="leased-media", hasMedia=True, mediaType="image", mime="image/jpeg",
        mediaUrls=[str(source)], lease_kwargs={"delivery": "c" * 64},
    )
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._inbound_consumer_id = "test-consumer"
    session = _LeaseSession(adapter, [raw], archive)
    adapter._http_session = session
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = Mock(return_value=True)
    adapter._should_process_message = Mock(return_value=False)
    adapter._is_allowed_profile_bridge_path = lambda path: path == str(source)
    adapter._inbound_archive_instance = Mock(return_value=archive)
    adapter._build_message_event = AsyncMock()
    adapter.handle_message = AsyncMock()

    await adapter._poll_messages()

    assert len(session.posts) == 1
    assert session.posts[0][0].endswith("/messages/ack")
    assert session.posts[0][1] == {
        "consumerId": "test-consumer", "deliveryId": "c" * 64,
        "epoch": 1, "token": "test-token",
    }
    [recovery] = archive.pending_bridge_recoveries()
    assert recovery.acknowledged and recovery.recovery_pending
    adapter._build_message_event.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_renew_persists_rotated_token_before_later_ack(tmp_path):
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    event_id, _ = archive.record(_raw(mid="renew"), "operate")
    original_lease = _lease(delivery="e" * 64, digest="f" * 64)
    receipt = archive.bind_bridge_receipt(event_id, original_lease)

    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1
    adapter.platform = SimpleNamespace(value="whatsapp")
    renewed_lease = {**original_lease, "token": "rotated-token", "expiresAt": int(time.time() * 1000) + 120_000}

    class _RenewSession:
        def post(self, *args, **kwargs):
            return _Response(adapter, {"delivery": renewed_lease})

    adapter._http_session = _RenewSession()
    renewed = await adapter._renew_inbound_receipt(archive, receipt)
    assert renewed is not None
    updated, expiry = renewed
    assert expiry == renewed_lease["expiresAt"] and updated.request["token"] == "rotated-token"
    # The old request cannot ACK after token rotation, even before a restart.
    assert not archive.mark_bridge_receipt_acked(receipt)
    updated = archive.mark_bridge_receipt_ready(updated)
    updated = archive.prepare_bridge_handoff(updated)
    assert archive.mark_bridge_receipt_acked(updated)


def test_owned_media_album_order_profile_isolation_and_rejection(tmp_path):
    home = tmp_path / "one"; cache = home / "cache"; cache.mkdir(parents=True)
    owned = cache / "photo.jpg"; owned.write_bytes(b"media")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(_raw(hasMedia=True, mediaType="image", mime="image/jpeg", nativeMetadata={"album": {"groupId": "a", "role": "child", "messageIndex": 2}}), "operate")
    archive.materialize(event_id, _raw(hasMedia=True, mediaType="image", mime="image/jpeg"), [str(owned)])
    with archive._connect() as db:
        row = db.execute("SELECT owned_path,sha256,size,download_status,album_ordinal FROM archive_attachment").fetchone()
        assert Path(row["owned_path"]).parent == archive.media_root and Path(row["owned_path"]).read_bytes() == b"media" and row["size"] == 5 and row["download_status"] == "owned"
    two = tmp_path / "two"; (two / "cache").mkdir(parents=True)
    assert WhatsAppInboundArchive(two / "whatsapp" / "inbound-archive-v1", two, two / "cache").record(_raw(), "observe")[0] == 1
    with pytest.raises(ArchiveRejected): archive.record(_raw(mid=""), "drop")
    with pytest.raises(ArchiveRejected):
        archive.materialize(event_id, _raw(hasMedia=True), [str(cache / "gone.jpg")])
    with archive._connect() as db:
        assert db.execute("SELECT download_status FROM archive_attachment").fetchone()[0] == "owned"


class _Response:
    status = 200
    def __init__(self, adapter, payload): self.adapter, self.payload = adapter, payload
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return False
    async def json(self): self.adapter._running = False; return self.payload


class _Session:
    def __init__(self, adapter, payload): self.adapter, self.payload = adapter, payload
    def get(self, *args, **kwargs): return _Response(self.adapter, self.payload)


class _LeaseSession(_Session):
    def __init__(self, adapter, payload, archive):
        super().__init__(adapter, payload)
        self.archive = archive
        self.posts = []

    def post(self, url, *, json, **kwargs):
        self.posts.append((url, json))
        # /ack is allowed only after the media object is owned by the profile
        # archive, never merely after a bridge download.
        with self.archive._connect() as db:
            attachment = db.execute("SELECT owned_path FROM archive_attachment").fetchone()
            assert attachment is not None and Path(attachment["owned_path"]).is_file()
        return _Response(self.adapter, {"status": "acknowledged"})


def _actual_node_spool_digest(event):
    bridge = Path(__file__).parents[2] / "scripts" / "whatsapp-bridge" / "inbound_spool.js"
    script = (
        f"import {{ inboundEventDigest }} from {json.dumps(str(bridge))};"
        "let source='';process.stdin.setEncoding('utf8');"
        "process.stdin.on('data', chunk => { source += chunk; });"
        "process.stdin.on('end', () => process.stdout.write(inboundEventDigest(JSON.parse(source)).eventDigest));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], input=json.dumps(event),
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    "native_metadata",
    (
        {"latitude": 0.000001},
        {"latitude": 0.0000001},
        {"latitude": -0.0},
        {"location": {"latitude": 0.000001, "longitude": 0.0000001, "altitude": -0.0}},
    ),
    ids=("fixed-boundary", "exponent-boundary", "negative-zero", "nested-metadata"),
)
def test_bridge_digest_matches_node_number_serialization_boundaries(native_metadata):
    event = _raw(mid="numeric", nativeMetadata=native_metadata)
    assert bridge_event_digest(event) == _actual_node_spool_digest(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("body", "quote", "media-descriptor"))
async def test_actual_node_digest_contract_rejects_mutated_leased_event(mutation):
    raw = _leased_raw(
        mid="node-contract", body="original", quotedText="quoted original", hasMedia=True,
        mediaType="image", mime="image/jpeg", fileName="photo.jpg", mediaUrls=["/bridge/photo.jpg"],
        mediaMetadata=[{"mediaType": "image", "mime": "image/jpeg", "fileName": "photo.jpg", "size": 5}],
    )
    node_digest = _actual_node_spool_digest(raw)
    assert node_digest == bridge_event_digest(raw)
    raw["_inboundLease"]["eventDigest"] = node_digest
    mutated = json.loads(json.dumps(raw))
    if mutation == "body":
        mutated["body"] = "changed"
    elif mutation == "quote":
        mutated["quotedText"] = "changed quote"
    else:
        mutated["mediaMetadata"][0]["mime"] = "image/png"

    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._inbound_consumer_id = "test-consumer"
    adapter._http_session = _Session(adapter, [mutated])
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._inbound_archive_instance = Mock()
    adapter._ack_inbound_receipt = AsyncMock(return_value=True)
    adapter._build_message_event = AsyncMock()
    adapter.handle_message = AsyncMock()

    await adapter._poll_messages()

    adapter._inbound_archive_instance.assert_not_called()
    adapter._ack_inbound_receipt.assert_not_awaited()
    adapter._build_message_event.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


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
    operated.handle_message.assert_awaited_once(); operated._build_message_event.assert_awaited_once_with(raw, already_admitted=True, archive_manifest=None)
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


@pytest.mark.asyncio
async def test_adapter_media_materialization_gates_observe_receipts_and_dispatch(monkeypatch):
    async def run(*, admitted, complete, raw_urls=None):
        raw = _raw(hasMedia=True, isGroup=True, mediaType="image", mediaUrls=raw_urls or ["/profile/cache/photo.jpg"])
        adapter = object.__new__(WhatsAppAdapter)
        adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
        adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
        adapter._is_allowed_profile_bridge_path = lambda path: str(path).startswith("/profile/cache/")
        adapter._is_archive_authorized = Mock(return_value=True)
        adapter._should_process_message = Mock(return_value=admitted)
        archive = SimpleNamespace(
            record=Mock(return_value=(1, True)),
            materialize=Mock(return_value=SimpleNamespace(
                complete=complete, owned_paths=("/archive/owned.jpg",) if complete else (),
                owned_descriptors=({"kind": "image", "mime": "image/jpeg", "file_name": "photo.jpg"},) if complete else (),
            )),
        )
        adapter._inbound_archive_instance = Mock(return_value=archive)
        adapter._agent_visible_archive_manifest = Mock(
            side_effect=lambda materialized: adapter._trusted_archive_manifest(materialized)
        )
        built = SimpleNamespace(message_type=MessageType.PHOTO, media_urls=["/profile/cache/photo.jpg"], media_types=["image/jpeg"])
        adapter._build_message_event = AsyncMock(return_value=built)
        adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()
        await adapter._poll_messages()
        await __import__("asyncio").sleep(0)
        return adapter, archive, raw

    observed, observed_archive, raw = await run(admitted=False, complete=True)
    observed_archive.materialize.assert_called_once_with(1, raw, ["/profile/cache/photo.jpg"], expected_attachment_count=1)
    observed.handle_message.assert_not_awaited(); observed._send_read_receipt.assert_not_awaited()

    incomplete, incomplete_archive, raw = await run(admitted=True, complete=False)
    incomplete_archive.materialize.assert_called_once_with(1, raw, ["/profile/cache/photo.jpg"], expected_attachment_count=1)
    incomplete.handle_message.assert_not_awaited(); incomplete._send_read_receipt.assert_not_awaited()

    operated, complete_archive, raw = await run(admitted=True, complete=True)
    complete_archive.materialize.assert_called_once_with(1, raw, ["/profile/cache/photo.jpg"], expected_attachment_count=1)
    operated.handle_message.assert_awaited_once(); operated._send_read_receipt.assert_awaited_once_with(raw)

    partial, partial_archive, raw = await run(
        admitted=True, complete=False,
        raw_urls=["/profile/cache/photo.jpg", "/outside/rejected.jpg"],
    )
    partial_archive.materialize.assert_called_once_with(
        1, raw, ["/profile/cache/photo.jpg", None], expected_attachment_count=2,
    )
    partial.handle_message.assert_not_awaited(); partial._send_read_receipt.assert_not_awaited()


@pytest.mark.asyncio
async def test_observe_document_uses_raw_slots_without_building_event(monkeypatch):
    allowed = "/profile/cache/kept.pdf"; rejected = "/outside/rejected.pdf"
    raw = _raw(hasMedia=True, isGroup=True, mediaType="document", mediaUrls=[allowed, rejected])
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_allowed_profile_bridge_path = lambda path: path == allowed
    adapter._is_archive_authorized = Mock(return_value=True); adapter._should_process_message = Mock(return_value=False)
    archive = SimpleNamespace(record=Mock(return_value=(1, True)), materialize=Mock(return_value=SimpleNamespace(complete=False)))
    adapter._inbound_archive_instance = Mock(return_value=archive)
    adapter._build_message_event = AsyncMock(); adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()

    await adapter._poll_messages()

    adapter._build_message_event.assert_not_awaited(); adapter.handle_message.assert_not_awaited(); adapter._send_read_receipt.assert_not_awaited()
    archive.materialize.assert_called_once_with(1, raw, [allowed, None], expected_attachment_count=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type,helper_name", [("image", "cache_image_from_url"), ("audio", "cache_audio_from_url")])
async def test_observe_http_media_never_caches_or_builds(media_type, helper_name, monkeypatch):
    raw = _raw(hasMedia=True, isGroup=True, mediaType=media_type, mediaUrls=["https://bridge.test/media"])
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = Mock(return_value=True); adapter._should_process_message = Mock(return_value=False)
    archive = SimpleNamespace(record=Mock(return_value=(1, True)), materialize=Mock(return_value=SimpleNamespace(complete=False)))
    adapter._inbound_archive_instance = Mock(return_value=archive)
    cache_helper = AsyncMock(); monkeypatch.setattr(f"plugins.platforms.whatsapp.adapter.{helper_name}", cache_helper)
    adapter._build_message_event = AsyncMock(); adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()

    await adapter._poll_messages()

    cache_helper.assert_not_awaited(); adapter._build_message_event.assert_not_awaited()
    archive.materialize.assert_called_once_with(1, raw, [None], expected_attachment_count=1)
    assert not archive.materialize.return_value.complete
    adapter.handle_message.assert_not_awaited(); adapter._send_read_receipt.assert_not_awaited()


@pytest.mark.asyncio
async def test_operate_builder_receives_only_archive_owned_media_after_materialize(monkeypatch, tmp_path):
    image = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0dIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05"
        b"\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "cache" / "bridge-cache.png"; source.write_bytes(image)
    owned = home / "whatsapp" / "inbound-archive-v1" / "media" / "digest"; owned.parent.mkdir(parents=True); owned.write_bytes(image)
    raw = _raw(hasMedia=True, mediaType="image", mime="image/png", fileName="photo.png", mediaUrls=[str(source)])
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = Mock(return_value=True); adapter._should_process_message = Mock(return_value=True)
    adapter._is_allowed_profile_bridge_path = lambda path: path == str(source)
    cache_image = AsyncMock(); monkeypatch.setattr("plugins.platforms.whatsapp.adapter.cache_image_from_url", cache_image)

    def materialize(*_args, **_kwargs):
        source.write_bytes(b"replaced")
        return MaterializationResult(True, 1, 1, ("owned",), (str(owned),), ({"kind": "image", "mime": "image/png", "file_name": "photo.png"},))

    archive = SimpleNamespace(record=Mock(return_value=(1, True)), materialize=Mock(side_effect=materialize))
    adapter._inbound_archive_instance = Mock(return_value=archive)
    built = []
    async def build(data, *, already_admitted, archive_manifest):
        built.append(data)
        return SimpleNamespace(message_type=MessageType.PHOTO, media_urls=list(data["mediaUrls"]), media_types=["image/jpeg"])
    adapter._build_message_event = AsyncMock(side_effect=build)
    adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()

    await adapter._poll_messages()

    exposed = Path(built[0]["mediaUrls"][0])
    assert source.read_bytes() == b"replaced" and exposed.read_bytes() == image
    assert exposed.parent == home / "cache" / "images"
    assert raw["mediaUrls"] == [str(source)]
    cache_image.assert_not_awaited()
    assert adapter.handle_message.await_args.args[0].media_urls == [str(exposed)]


@pytest.mark.asyncio
async def test_leased_seven_photo_burst_archives_every_member_then_dispatches_one_ordered_turn(tmp_path):
    """The real poll/receipt boundary admits an album once, never seven times.

    The transport fixture is intentionally lease-shaped: the production bridge
    supplies exactly this fenced event after its native spool has written it.
    Existing Node digest-contract tests cover the shared JS digest; this test
    exercises the Python receipt/archive/dispatch half with seven members.
    """
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raws = []
    for index in reversed(range(7)):
        path = cache / f"bridge-{index}.jpg"; path.write_bytes(f"photo-{index}".encode())
        raws.append(_leased_raw(
            mid=f"album-{index}", body=f"caption-{index}", hasMedia=True, mediaType="image",
            mime="image/jpeg", mediaUrls=[str(path)],
            nativeMetadata={"album": {"groupId": "native-parent", "role": "child", "messageIndex": index}},
            lease_kwargs={"delivery": f"{index:x}" * 64},
        ))

    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._inbound_consumer_id = "test-consumer"; adapter._http_session = _LeaseSession(adapter, raws, archive)
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = Mock(return_value=True)
    adapter._should_process_message = Mock(side_effect=lambda raw: raw["messageId"] == "album-6")
    adapter._is_allowed_profile_bridge_path = lambda path: str(path).startswith(str(cache))
    adapter._inbound_archive_instance = Mock(return_value=archive)
    adapter._archive_manifest_capability = object()
    adapter._agent_visible_archive_manifest = Mock(side_effect=lambda materialized: adapter._trusted_archive_manifest(materialized))
    adapter._inbound_album_quiet_seconds = 0.05; adapter._inbound_album_hard_cap_seconds = 1.0
    adapter._native_inbound_album_quiet_seconds = 0.05; adapter._native_inbound_album_hard_cap_seconds = 1.0
    built = []

    async def build(data, *, already_admitted, archive_manifest):
        built.append((data, archive_manifest))
        return MessageEvent(
            text=data["body"], message_type=MessageType.PHOTO, source=SimpleNamespace(),
            message_id=data["messageId"], media_urls=list(data["mediaUrls"]), media_types=["image/jpeg"],
        )

    adapter._build_message_event = AsyncMock(side_effect=build)
    adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()
    await adapter._poll_messages()

    assert len(adapter._http_session.posts) == 7  # receipt/ACK is still per durable member
    with archive._connect() as db:
        admissions = db.execute("SELECT admission FROM archive_event ORDER BY message_id").fetchall()
    assert [row["admission"] for row in admissions].count("operate") == 1
    assert [row["admission"] for row in admissions].count("observe") == 6
    adapter.handle_message.assert_awaited_once()
    album = adapter.handle_message.await_args.args[0]
    assert [Path(path).read_bytes() for path in album.media_urls] == [f"photo-{i}".encode() for i in range(7)]
    assert album.text == "\n\n".join(f"caption-{i}" for i in range(7))
    assert [member["message_id"] for member in album.metadata["whatsapp_album_members"]] == [f"album-{i}" for i in range(7)]
    assert [member["caption"] for member in album.metadata["whatsapp_album_members"]] == [f"caption-{i}" for i in range(7)]
    assert [member["media_count"] for member in album.metadata["whatsapp_album_members"]] == [1] * 7
    assert adapter._send_read_receipt.await_count == 1


@pytest.mark.asyncio
async def test_ambient_photo_burst_stays_archived_and_never_starts_a_turn(tmp_path):
    """A fallback-timer burst with no addressed member is observe-only."""
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raws = []
    for index in range(3):
        path = cache / f"ambient-{index}.jpg"; path.write_bytes(b"ambient")
        raws.append(_leased_raw(
            mid=f"ambient-{index}", body=f"ambient-{index}", hasMedia=True, mediaType="image",
            mime="image/jpeg", mediaUrls=[str(path)], lease_kwargs={"delivery": f"{index + 7:x}" * 64},
        ))
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._inbound_consumer_id = "test-consumer"; adapter._http_session = _LeaseSession(adapter, raws, archive)
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = Mock(return_value=True); adapter._should_process_message = Mock(return_value=False)
    adapter._is_allowed_profile_bridge_path = lambda path: str(path).startswith(str(cache))
    adapter._inbound_archive_instance = Mock(return_value=archive)
    adapter._inbound_album_quiet_seconds = 0.01; adapter._inbound_album_hard_cap_seconds = 0.05
    adapter._build_message_event = AsyncMock(); adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()
    await adapter._poll_messages()

    adapter._build_message_event.assert_not_awaited(); adapter.handle_message.assert_not_awaited()
    adapter._send_read_receipt.assert_not_awaited()
    with archive._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM archive_event WHERE admission='observe'").fetchone()[0] == 3


@pytest.mark.asyncio
async def test_fallback_photo_burst_hard_cap_prevents_an_indefinite_quiet_wait():
    """No native album association still has a finite bounded admission delay."""
    adapter = object.__new__(WhatsAppAdapter)
    adapter._inbound_album_quiet_seconds = 0.20; adapter._inbound_album_hard_cap_seconds = 0.03
    adapter._build_message_event = AsyncMock(side_effect=lambda data, **_: MessageEvent(
        text=data["body"], message_type=MessageType.PHOTO, source=SimpleNamespace(),
        message_id=data["messageId"], media_urls=[], media_types=[],
    ))
    adapter.handle_message = AsyncMock()
    first = _raw(mid="fallback-one", hasMedia=True, mediaType="image", body="one")
    second = _raw(mid="fallback-two", hasMedia=True, mediaType="image", body="two")
    assert adapter._enqueue_inbound_album_member(first, None, admitted=True)
    await __import__("asyncio").sleep(0.015)
    assert adapter._enqueue_inbound_album_member(second, None, admitted=True)
    await __import__("asyncio").sleep(0.05)
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "one\n\ntwo"


@pytest.mark.asyncio
async def test_native_album_settles_beyond_fallback_quiet_window_without_splitting():
    """Association metadata gets its own settlement policy, not fallback timing."""
    adapter = object.__new__(WhatsAppAdapter)
    adapter._inbound_album_quiet_seconds = 0.01; adapter._inbound_album_hard_cap_seconds = 0.02
    adapter._native_inbound_album_quiet_seconds = 0.08; adapter._native_inbound_album_hard_cap_seconds = 0.25
    adapter._build_message_event = AsyncMock(side_effect=lambda data, **_: MessageEvent(
        text=data["body"], message_type=MessageType.PHOTO, source=SimpleNamespace(),
        message_id=data["messageId"], media_urls=[], media_types=[],
    ))
    adapter.handle_message = AsyncMock()
    first = _raw(mid="native-stagger-one", hasMedia=True, mediaType="image", body="one",
                 nativeMetadata={"album": {"groupId": "stable-parent", "messageIndex": 0}})
    second = _raw(mid="native-stagger-two", hasMedia=True, mediaType="image", body="two",
                  nativeMetadata={"album": {"groupId": "stable-parent", "messageIndex": 1}})
    assert adapter._enqueue_inbound_album_member(first, None, admitted=True)
    # This exceeds fallback's entire hard cap.  It must still join the native
    # association instead of manufacturing a second operational turn.
    await asyncio.sleep(0.04)
    assert adapter._enqueue_inbound_album_member(second, None, admitted=True)
    await asyncio.sleep(0.12)
    adapter.handle_message.assert_awaited_once()
    album = adapter.handle_message.await_args.args[0]
    assert album.text == "one\n\ntwo"
    assert [member["caption"] for member in album.metadata["whatsapp_album_members"]] == ["one", "two"]


def test_album_receipts_survive_restart_as_one_recovery_obligation(tmp_path):
    """Crash after member ACK but before flush cannot recover one turn per photo."""
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    root = home / "whatsapp" / "inbound-archive-v1"
    archive = WhatsAppInboundArchive(root, home, cache)
    album_key = "a" * 64
    receipts = []
    for index in range(2):
        raw = _leased_raw(mid=f"restart-album-{index}", hasMedia=True, mediaType="image",
                          chatId="15551230000@s.whatsapp.net", senderId="15551230000@s.whatsapp.net",
                          isGroup=False,
                          nativeMetadata={"album": {"groupId": "restart-parent", "messageIndex": index}},
                          lease_kwargs={"delivery": f"{index + 1:x}" * 64})
        event_id, accepted = archive.record(raw, "operate" if index else "observe")
        assert accepted
        receipt = archive.bind_bridge_receipt(event_id, raw["_inboundLease"])
        receipt = archive.mark_bridge_receipt_ready(receipt)
        receipt = archive.prepare_bridge_handoff(receipt, album_key=album_key)
        assert archive.mark_bridge_receipt_acked(receipt)
        receipts.append(receipt)

    # A fresh process sees only the operational representative; no in-memory
    # enqueue state is needed to avoid duplicated recovery clarifications.
    restarted = WhatsAppInboundArchive(root, home, cache)
    pending = restarted.pending_bridge_recoveries()
    assert [item.delivery_id for item in pending] == [receipts[1].delivery_id]
    recovery = restarted.reserve_bridge_recovery(receipts[1].delivery_id)
    assert recovery is not None
    assert restarted.register_bridge_recovery_delivery(recovery, "one-album-final")
    assert restarted.settle_bridge_recovery_delivery(recovery, "one-album-final")
    assert restarted.pending_bridge_recoveries() == []


@pytest.mark.asyncio
async def test_multiplexed_ambient_archive_and_agent_copy_stay_with_adapter_profile(tmp_path, monkeypatch):
    """A receiving adapter never follows a different active profile's cache root."""
    image = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0dIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05"
        b"\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    profile_a, profile_b = tmp_path / "profiles" / "a", tmp_path / "profiles" / "b"
    source = profile_a / "cache" / "images" / "bridge.png"
    source.parent.mkdir(parents=True); source.write_bytes(image)
    (profile_b / "cache").mkdir(parents=True)
    profile_b_cache_files_before = {
        path.relative_to(profile_b / "cache")
        for path in (profile_b / "cache").rglob("*") if path.is_file()
    }
    # Simulate an ambient callback running with B active while A owns the
    # WhatsApp transport and its bridge cache.
    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    raw = _raw("profile-bound", hasMedia=True, isGroup=True, mediaType="image", mime="image/png", fileName="bridge.png", mediaUrls=[str(source)])
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._inbound_archive = None; adapter._inbound_archive_home = profile_a.resolve(); adapter._archive_manifest_capability = object()
    adapter._is_archive_authorized = Mock(return_value=True); adapter._should_process_message = Mock(return_value=False)
    adapter._build_message_event = AsyncMock(); adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()

    await adapter._poll_messages()

    archive = adapter._inbound_archive_instance()
    with archive._connect() as db:
        owned = Path(db.execute("SELECT owned_path FROM archive_attachment").fetchone()[0])
    assert owned.parent == archive.media_root and owned.read_bytes() == image
    assert profile_a in owned.parents
    assert {
        path.relative_to(profile_b / "cache")
        for path in (profile_b / "cache").rglob("*") if path.is_file()
    } == profile_b_cache_files_before
    adapter._build_message_event.assert_not_awaited(); adapter.handle_message.assert_not_awaited()

    materialized = type("Materialized", (), {
        "owned_paths": (str(owned),),
        "owned_descriptors": ({"kind": "image", "mime": "image/png", "file_name": "bridge.png"},),
    })()
    visible = adapter._agent_visible_archive_manifest(materialized).paths[0]
    assert Path(visible).read_bytes() == image
    assert Path(visible).is_relative_to(profile_a / "cache" / "images")
    assert os.environ["HERMES_HOME"] == str(profile_b)  # owner binding never rewrites global state
    assert {
        path.relative_to(profile_b / "cache")
        for path in (profile_b / "cache").rglob("*") if path.is_file()
    } == profile_b_cache_files_before


def test_profile_cache_symlink_is_rejected_before_inbound_or_agent_visible_write(tmp_path):
    """A profile-relative cache symlink must never route owned bytes to another profile."""
    profile_a, profile_b = tmp_path / "profiles" / "a", tmp_path / "profiles" / "b"
    profile_a.mkdir(parents=True); profile_b.mkdir(parents=True)
    (profile_a / "cache").symlink_to(profile_b, target_is_directory=True)
    owned = tmp_path / "owned.png"
    owned.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0dIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05"
        b"\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    adapter = object.__new__(WhatsAppAdapter)
    adapter._inbound_archive = None
    adapter._inbound_archive_home = profile_a.resolve()
    adapter._archive_manifest_capability = object()

    # The incoming bridge path is not trusted once the owner's cache root is
    # redirected.  Neither archive materialization nor an admitted handoff
    # may create any file in profile B.
    assert not adapter._is_allowed_profile_bridge_path(str(profile_b / "incoming.png"))
    with pytest.raises(ValueError, match="symlink"):
        adapter._inbound_archive_instance()
    materialized = type("Materialized", (), {
        "owned_paths": (str(owned),),
        "owned_descriptors": ({"kind": "image", "mime": "image/png", "file_name": "owned.png"},),
    })()
    with pytest.raises(ValueError, match="symlink"):
        adapter._agent_visible_archive_manifest(materialized)
    assert not any(path.is_file() for path in profile_b.rglob("*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,mime,file_name,expected_type,content", [
    ("image", "image/jpeg", "photo.jpg", MessageType.PHOTO, b"image"),
    ("audio", "audio/mpeg", "sound.mp3", MessageType.AUDIO, b"audio"),
    ("video", "video/mp4", "movie.mp4", MessageType.VIDEO, b"video"),
    ("document", "text/plain", "report.txt", MessageType.DOCUMENT, b"report body"),
])
async def test_builder_accepts_only_capability_bound_owned_manifest(
    tmp_path, kind, mime, file_name, expected_type, content,
):
    owned = tmp_path / "archive" / "media" / "digest"; owned.parent.mkdir(parents=True); owned.write_bytes(content)
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp"); adapter._archive_manifest_capability = object()
    adapter.build_source = lambda **_kwargs: SimpleNamespace()
    raw = _raw(hasMedia=True, mediaType=kind, mime=mime, fileName=file_name, mediaUrls=[str(owned)])
    materialized = SimpleNamespace(
        owned_paths=(str(owned),),
        owned_descriptors=({"kind": kind, "mime": mime, "file_name": file_name},),
    )
    manifest = adapter._trusted_archive_manifest(materialized)

    event = await adapter._build_message_event(raw, already_admitted=True, archive_manifest=manifest)
    untrusted = await adapter._build_message_event(raw, already_admitted=True)

    assert event.message_type == expected_type and event.media_urls == [str(owned)] and event.media_types == [mime]
    assert untrusted.media_urls == []


def test_symlink_escape_is_rejected(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    outside = tmp_path / "outside"; outside.write_bytes(b"x")
    link = cache / "escape"; link.symlink_to(outside)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(_raw(hasMedia=True), "operate")
    archive.materialize(event_id, _raw(hasMedia=True), [str(link)])
    with archive._connect() as db:
        assert db.execute("SELECT download_status FROM archive_attachment").fetchone()[0] == "deleted_or_replaced"

def test_failed_owned_copy_leaves_no_staging_parts(tmp_path, monkeypatch):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    source = cache / "photo.jpg"; source.write_bytes(b"media")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(_raw(hasMedia=True), "operate")
    monkeypatch.setattr("plugins.platforms.whatsapp.inbound_archive.os.fsync", lambda _fd: (_ for _ in ()).throw(OSError("disk")))
    archive.materialize(event_id, _raw(hasMedia=True), [str(source)])
    with archive._connect() as db: assert db.execute("SELECT download_status FROM archive_attachment").fetchone()[0] == "deleted_or_replaced"
    assert not list(archive.media_root.glob("*.part"))

def test_symlinked_media_root_is_rejected(tmp_path):
    home = tmp_path / "profile"; (home / "cache").mkdir(parents=True); root = home / "whatsapp" / "inbound-archive-v1"; root.mkdir(parents=True)
    outside = tmp_path / "outside"; outside.mkdir(); (root / "media").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArchiveRejected): WhatsAppInboundArchive(root, home, home / "cache")
    assert not list(outside.iterdir())


def test_digest_named_symlink_is_never_accepted_as_owned(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    source = cache / "photo.jpg"; source.write_bytes(b"same-content")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    digest = __import__("hashlib").sha256(b"same-content").hexdigest()
    outside = tmp_path / "unrelated"; outside.write_bytes(b"same-content")
    target = archive.media_root / digest; target.symlink_to(outside)
    event_id, _ = archive.record(_raw(hasMedia=True, mediaType="image"), "operate")

    archive.materialize(event_id, _raw(hasMedia=True, mediaType="image"), [str(source)])

    with archive._connect() as db:
        row = db.execute("SELECT owned_path,download_status FROM archive_attachment").fetchone()
    assert row["owned_path"] is None and row["download_status"] != "owned"
    assert target.is_symlink() and outside.read_bytes() == b"same-content"


def test_digest_symlink_swap_at_open_boundary_is_rejected(tmp_path, monkeypatch):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    source = cache / "photo.jpg"; source.write_bytes(b"same-content")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raw = _raw(hasMedia=True, mediaType="image")
    first_id, _ = archive.record(raw, "operate")
    assert archive.materialize(first_id, raw, [str(source)]).complete
    digest = hashlib.sha256(b"same-content").hexdigest(); target = archive.media_root / digest
    outside = tmp_path / "outside"; outside.write_bytes(b"same-content")
    second_id, _ = archive.record(_raw(mid="second", hasMedia=True, mediaType="image"), "operate")
    real_open = __import__("plugins.platforms.whatsapp.inbound_archive", fromlist=["os"]).os.open
    swapped = False

    def swap_at_target_open(path, flags, *args):
        nonlocal swapped
        if not swapped and Path(path) == target and not flags & (os.O_WRONLY | os.O_RDWR):
            swapped = True
            target.unlink(); target.symlink_to(outside)
        return real_open(path, flags, *args)

    import os
    monkeypatch.setattr("plugins.platforms.whatsapp.inbound_archive.os.open", swap_at_target_open)
    result = archive.materialize(second_id, _raw(mid="second", hasMedia=True, mediaType="image"), [str(source)])

    with archive._connect() as db:
        row = db.execute("SELECT owned_path,download_status FROM archive_attachment WHERE event_id=?", (second_id,)).fetchone()
    assert not result.complete and row["owned_path"] is None and row["download_status"] != "owned"
    assert target.is_symlink() and outside.read_bytes() == b"same-content" and not list(archive.media_root.glob("*.part"))


def test_concurrent_digest_publication_never_overwrites_winner(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    first_source = cache / "first.jpg"; second_source = cache / "second.jpg"
    first_source.write_bytes(b"same-content"); second_source.write_bytes(b"same-content")
    root = home / "whatsapp" / "inbound-archive-v1"
    first = WhatsAppInboundArchive(root, home, cache)
    second = WhatsAppInboundArchive(root, home, cache)
    raw_one = _raw(mid="one", hasMedia=True, mediaType="image")
    raw_two = _raw(mid="two", hasMedia=True, mediaType="image")
    first_id, _ = first.record(raw_one, "operate")
    second_id, _ = second.record(raw_two, "operate")
    barrier = threading.Barrier(2)

    def materialize(archive, event_id, raw, source):
        barrier.wait()
        archive.materialize(event_id, raw, [str(source)])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(materialize, first, first_id, raw_one, first_source),
            pool.submit(materialize, second, second_id, raw_two, second_source),
        ]
        for future in futures:
            future.result()

    with first._connect() as db:
        rows = db.execute("SELECT owned_path,download_status FROM archive_attachment ORDER BY event_id").fetchall()
    digest = hashlib.sha256(b"same-content").hexdigest()
    target = first.media_root / digest
    assert len(rows) == 2 and {row["owned_path"] for row in rows} == {str(target)}
    assert {row["download_status"] for row in rows} == {"owned"}
    assert target.read_bytes() == b"same-content" and target.stat().st_nlink == 1
    assert not list(first.media_root.glob("*.part"))


def test_dedup_loser_fsyncs_media_directory_before_manifest_write(tmp_path, monkeypatch):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    source = cache / "photo.jpg"; source.write_bytes(b"same-content")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    digest = hashlib.sha256(b"same-content").hexdigest(); target = archive.media_root / digest
    target.write_bytes(b"same-content"); os.chmod(target, 0o600)
    event_id, _ = archive.record(_raw(hasMedia=True, mediaType="image"), "operate")
    order = []; real_fsync = os.fsync; real_connect = archive._connect

    def record_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            order.append("media_fsync")
        return real_fsync(fd)

    class GuardedConnection:
        def __enter__(self):
            self.connection = real_connect()
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def execute(self, query, *args):
            if "INSERT INTO archive_attachment" in query:
                order.append("manifest_write")
                assert "media_fsync" in order
            return self.connection.execute(query, *args)

    import stat
    monkeypatch.setattr("plugins.platforms.whatsapp.inbound_archive.os.fsync", record_fsync)
    monkeypatch.setattr(archive, "_connect", GuardedConnection)
    result = archive.materialize(event_id, _raw(hasMedia=True, mediaType="image"), [str(source)])

    assert result.complete and order.index("media_fsync") < order.index("manifest_write")

@pytest.mark.parametrize("kind,mime,name", [("image","image/jpeg","a.jpg"),("video","video/mp4","b.mp4"),("audio","audio/ogg","c.ogg"),("document","application/pdf","d.pdf")])
def test_media_kinds_are_owned_with_descriptors(tmp_path, kind, mime, name):
    home=tmp_path / kind; cache=home / "cache"; cache.mkdir(parents=True); source=cache / name; source.write_bytes(kind.encode())
    archive=WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raw=_raw(hasMedia=True, mediaType=kind, mime=mime, fileName=name); event_id,_=archive.record(raw,"operate"); archive.materialize(event_id,raw,[str(source)])
    with archive._connect() as db: row=db.execute("SELECT descriptor_json,owned_path,download_status FROM archive_attachment").fetchone()
    assert row["download_status"] == "owned" and Path(row["owned_path"]).read_bytes() == kind.encode()
    assert {"kind":kind,"mime":mime,"file_name":name} == __import__("json").loads(row["descriptor_json"])

def test_owned_object_survives_source_deletion(tmp_path):
    home=tmp_path / "profile"; cache=home / "cache"; cache.mkdir(parents=True); source=cache / "a.jpg"; source.write_bytes(b"durable")
    archive=WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache); raw=_raw(hasMedia=True, mediaType="image", mime="image/jpeg", fileName="a.jpg"); event_id,_=archive.record(raw,"operate"); archive.materialize(event_id,raw,[str(source)]); source.unlink()
    with archive._connect() as db: row=db.execute("SELECT owned_path,sha256,download_status FROM archive_attachment").fetchone()
    assert row["download_status"] == "owned" and Path(row["owned_path"]).read_bytes() == b"durable" and __import__("hashlib").sha256(b"durable").hexdigest() == row["sha256"]


def test_failed_retry_preserves_verified_owned_attachment(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    source = cache / "a.jpg"; source.write_bytes(b"durable")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raw = _raw(hasMedia=True, mediaType="image", mime="image/jpeg", fileName="a.jpg")
    event_id, _ = archive.record(raw, "operate")
    assert archive.materialize(event_id, raw, [str(source)]).complete
    with archive._connect() as db:
        before = db.execute("SELECT owned_path,sha256,size,download_status FROM archive_attachment").fetchone()
    source.unlink()

    retry = archive.materialize(event_id, raw, [str(source)])

    with archive._connect() as db:
        after = db.execute("SELECT owned_path,sha256,size,download_status FROM archive_attachment").fetchone()
    assert retry.complete and retry.statuses == ("owned",) and dict(after) == dict(before)


def test_missing_media_returns_incomplete_manifest(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raw = _raw(hasMedia=True, mediaType="image", fileName="missing.jpg")
    event_id, _ = archive.record(raw, "operate")

    result = archive.materialize(event_id, raw, [str(cache / "missing.jpg")])

    assert not result.complete and result.owned_count == 0 and result.attachment_count == 1
    assert result.statuses == ("deleted_or_replaced",)


def test_partial_attachment_slots_are_retained_in_order(tmp_path):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    source = cache / "first.jpg"; source.write_bytes(b"owned")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    raw = _raw(hasMedia=True, mediaType="image", fileName="album.jpg")
    event_id, _ = archive.record(raw, "operate")

    result = archive.materialize(
        event_id, raw, [str(source), None], expected_attachment_count=2,
    )

    with archive._connect() as db:
        rows = db.execute("SELECT ordinal,owned_path,download_status FROM archive_attachment ORDER BY ordinal").fetchall()
    assert not result.complete and result.statuses == ("owned", "missing_or_rejected")
    assert [(row["ordinal"], row["download_status"]) for row in rows] == [(0, "owned"), (1, "missing_or_rejected")]
    assert Path(rows[0]["owned_path"]).read_bytes() == b"owned" and rows[1]["owned_path"] is None

def test_same_bytes_dedup_per_profile_but_not_across_profiles(tmp_path):
    def materialize(home, message):
        cache=home / "cache"; cache.mkdir(parents=True, exist_ok=True); source=cache / f"{message}.jpg"; source.write_bytes(b"same")
        archive=WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache); raw=_raw(messageId=message,hasMedia=True,mediaType="image"); event_id,_=archive.record(raw,"operate"); archive.materialize(event_id,raw,[str(source)])
        with archive._connect() as db: return db.execute("SELECT event_id,ordinal,owned_path,sha256 FROM archive_attachment WHERE event_id=?", (event_id,)).fetchone()
    first=materialize(tmp_path / "one","a"); second=materialize(tmp_path / "one","b"); third=materialize(tmp_path / "two","a")
    assert first[0] != second[0] and first[1] == second[1] == 0 and first[2] == second[2] and first[3] == second[3]
    assert third[2] != first[2] and third[3] == first[3]


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
