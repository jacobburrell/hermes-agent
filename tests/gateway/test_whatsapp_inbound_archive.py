"""Private WhatsApp archive foundation: no model/provider projection."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.platforms.whatsapp.inbound_archive import ArchiveRejected, MaterializationResult, WhatsAppInboundArchive
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
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda path: str(path).startswith("/profile/cache/"),
    )
    async def run(*, admitted, complete, raw_urls=None):
        raw = _raw(hasMedia=True, isGroup=True, mediaType="image", mediaUrls=raw_urls or ["/profile/cache/photo.jpg"])
        adapter = object.__new__(WhatsAppAdapter)
        adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
        adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
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
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda path: path == allowed,
    )
    raw = _raw(hasMedia=True, isGroup=True, mediaType="document", mediaUrls=[allowed, rejected])
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
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
    source = tmp_path / "bridge-cache.jpg"; source.write_bytes(b"before")
    owned = tmp_path / "archive" / "media" / "digest"; owned.parent.mkdir(parents=True); owned.write_bytes(b"owned")
    raw = _raw(hasMedia=True, mediaType="image", mediaUrls=[str(source)])
    adapter = object.__new__(WhatsAppAdapter)
    adapter._running = True; adapter._bridge_port = 1; adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._http_session = _Session(adapter, [raw]); adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = Mock(return_value=True); adapter._should_process_message = Mock(return_value=True)
    monkeypatch.setattr("plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda path: path == str(source))
    cache_image = AsyncMock(); monkeypatch.setattr("plugins.platforms.whatsapp.adapter.cache_image_from_url", cache_image)

    def materialize(*_args, **_kwargs):
        source.write_bytes(b"replaced")
        return MaterializationResult(True, 1, 1, ("owned",), (str(owned),), ({"kind": "image", "mime": "image/jpeg", "file_name": "photo.jpg"},))

    archive = SimpleNamespace(record=Mock(return_value=(1, True)), materialize=Mock(side_effect=materialize))
    adapter._inbound_archive_instance = Mock(return_value=archive)
    built = []
    async def build(data, *, already_admitted, archive_manifest):
        built.append(data)
        return SimpleNamespace(message_type=MessageType.PHOTO, media_urls=list(data["mediaUrls"]), media_types=["image/jpeg"])
    adapter._build_message_event = AsyncMock(side_effect=build)
    adapter.handle_message = AsyncMock(); adapter._send_read_receipt = AsyncMock()

    await adapter._poll_messages()

    assert source.read_bytes() == b"replaced" and built[0]["mediaUrls"] == [str(owned)]
    assert raw["mediaUrls"] == [str(source)]
    cache_image.assert_not_awaited()
    assert adapter.handle_message.await_args.args[0].media_urls == [str(owned)]


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
    if kind == "document":
        assert "[Content of report.txt]:" in event.text
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
