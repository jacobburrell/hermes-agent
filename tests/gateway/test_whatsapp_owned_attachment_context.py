"""Real model-facing paths for owned WhatsApp operating attachments."""
import asyncio
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.image_routing import build_native_content_parts
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from plugins.platforms.whatsapp.inbound_archive import WhatsAppInboundArchive, bridge_event_digest


_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0dIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05"
    b"\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _raw(message_id: str, **extra):
    return {
        "messageId": message_id,
        "chatId": "chat@g.us",
        "senderId": "1555000@s.whatsapp.net",
        "body": "caption kept",
        "timestamp": 1,
        "hasMedia": True,
        **extra,
    }


def _adapter() -> WhatsAppAdapter:
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._archive_manifest_capability = object()
    adapter.build_source = lambda **_kwargs: SimpleNamespace()
    return adapter


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")})
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_: "native"
    return runner


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="owned-media", chat_type="dm", user_id="42")


class _OneMessageSession:
    """Minimal bridge transport which stops the poller after one real payload."""

    def __init__(self, adapter, message):
        self.adapter = adapter
        self.message = message

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


def _native_document_event(cache_dir: Path) -> dict:
    """Run the pure native bridge extractor, including its real local write."""
    helpers = Path(__file__).parents[2] / "scripts" / "whatsapp-bridge" / "bridge_helpers.js"
    script = f"""
        import {{ extractBridgeEvent }} from {json.dumps(str(helpers))};
        import {{ mkdirSync, writeFileSync }} from 'node:fs';
        import path from 'node:path';
        const cacheDir = process.argv[1];
        const event = await extractBridgeEvent({{
          msg: {{
            key: {{
              id: 'native-document',
              remoteJid: 'chat@g.us',
              participant: '1555000@s.whatsapp.net',
              fromMe: false,
            }},
            pushName: 'Tester',
            messageTimestamp: 1,
            message: {{
              documentMessage: {{
                caption: 'Please review this contract',
                fileName: 'contract.pdf',
                mimetype: 'application/pdf',
                contextInfo: {{
                  stanzaId: 'quoted-request',
                  participant: '1555999@s.whatsapp.net',
                  remoteJid: 'chat@g.us',
                  quotedMessage: {{ conversation: 'Archive the signed contract.' }},
                }},
              }},
            }},
          }},
          chatId: 'chat@g.us',
          senderId: '1555000@s.whatsapp.net',
          senderNumber: '1555000',
          botIds: ['1555999@s.whatsapp.net'],
          cacheDirs: {{ document: cacheDir }},
          downloadMedia: async () => Buffer.from('%PDF-1.7 native bridge bytes'),
          writeMediaFile: async ({{ buffer, dir, prefix, fileName }}) => {{
            mkdirSync(dir, {{ recursive: true }});
            const target = path.join(dir, `${{prefix}}_native_${{fileName}}`);
            writeFileSync(target, buffer);
            return target;
          }},
        }});
        process.stdout.write(JSON.stringify(event));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(cache_dir)],
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


@pytest.mark.asyncio
async def test_owned_whatsapp_image_becomes_native_model_attachment_not_bridge_path(tmp_path, monkeypatch):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    bridge_path = cache / "bridge-image.png"; bridge_path.write_bytes(_PNG)
    raw = _raw("image", mediaType="image", mime="image/png", fileName="photo.png")
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(raw, "operate")
    result = archive.materialize(event_id, raw, [str(bridge_path)])
    adapter = _adapter(); manifest = adapter._agent_visible_archive_manifest(result)
    event_data = dict(raw); event_data["mediaUrls"] = list(manifest.paths)
    event = await adapter._build_message_event(event_data, already_admitted=True, archive_manifest=manifest)

    exposed = Path(event.media_urls[0])
    assert result.complete and event.text == "caption kept"
    assert event.message_type == MessageType.PHOTO and exposed.read_bytes() == _PNG
    assert exposed.parent == home / "cache" / "images"
    assert str(bridge_path) not in event.media_urls and str(result.owned_paths[0]) not in event.media_urls
    runner = _runner(); source = _source()
    await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    parts, skipped = build_native_content_parts(
        "caption kept", runner._consume_pending_native_image_paths(build_session_key(source)),
    )
    assert not skipped and any(part.get("type") == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,") for part in parts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "mime", "filename", "content", "cache_subdir"),
    [
        ("audio", "audio/mpeg", "call.mp3", b"ID3\x04owned audio", "audio"),
        ("video", "video/mp4", "walk.mp4", b"\x00\x00\x00\x18ftypmp42owned video", "videos"),
    ],
)
async def test_owned_whatsapp_media_uses_agent_visible_copy_not_bridge_or_archive(
    tmp_path, monkeypatch, kind, mime, filename, content, cache_subdir,
):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    bridge_path = cache / f"bridge-{filename}"; bridge_path.write_bytes(content)
    raw = _raw("media", mediaType=kind, mime=mime, fileName=filename)
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(raw, "operate")
    result = archive.materialize(event_id, raw, [str(bridge_path)])
    adapter = _adapter(); manifest = adapter._agent_visible_archive_manifest(result)
    event_data = dict(raw); event_data["mediaUrls"] = list(manifest.paths)
    event = await adapter._build_message_event(event_data, already_admitted=True, archive_manifest=manifest)

    exposed = Path(event.media_urls[0])
    assert result.complete and exposed.read_bytes() == content
    assert exposed.parent == home / "cache" / cache_subdir
    assert str(bridge_path) not in event.media_urls and str(result.owned_paths[0]) not in event.media_urls
    runner = _runner(); source = _source()
    prompt = await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    assert str(exposed) in prompt and str(bridge_path) not in prompt and str(result.owned_paths[0]) not in prompt


@pytest.mark.asyncio
async def test_owned_whatsapp_document_uses_agent_cache_copy_not_bridge_path(tmp_path, monkeypatch):
    home = tmp_path / "profile"; cache = home / "cache"; cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    bridge_path = cache / "bridge-contract.pdf"; bridge_path.write_bytes(b"%PDF-1.4 owned")
    raw = _raw(
        "document", mediaType="document", mime="application/pdf", fileName="contract.pdf",
        nativeMetadata={"album": {"groupId": "album", "role": "child", "messageIndex": 2}},
    )
    archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache)
    event_id, _ = archive.record(raw, "operate")
    result = archive.materialize(event_id, raw, [str(bridge_path)])
    adapter = _adapter(); manifest = adapter._agent_visible_archive_manifest(result)
    event_data = dict(raw); event_data["mediaUrls"] = list(manifest.paths)
    event = await adapter._build_message_event(event_data, already_admitted=True, archive_manifest=manifest)

    exposed = Path(event.media_urls[0])
    assert result.complete and event.text == "caption kept" and event.media_types == ["application/pdf"]
    assert exposed.read_bytes() == b"%PDF-1.4 owned" and exposed.parent == home / "cache" / "documents"
    assert str(bridge_path) not in event.media_urls and str(result.owned_paths[0]) not in event.media_urls
    runner = _runner(); source = _source()
    prompt = await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    assert str(exposed) in prompt and "contract.pdf" in prompt and str(bridge_path) not in prompt


@pytest.mark.asyncio
async def test_native_bridge_document_is_archived_then_exposed_to_model_with_caption_and_quote(tmp_path, monkeypatch):
    """A bridge-native attachment cannot degrade to an URL/text-only agent turn."""
    home = tmp_path / "profile"
    bridge_cache = home / "cache" / "documents"
    bridge_cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    raw = _native_document_event(bridge_cache)
    bridge_path = Path(raw["mediaUrls"][0])
    assert bridge_path.read_bytes() == b"%PDF-1.7 native bridge bytes"

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._running = True
    adapter._bridge_port = 1
    adapter._http_session = _OneMessageSession(adapter, raw)
    # The focused test environment intentionally omits optional aiohttp.
    # Keep the actual poll/archive/event path, but substitute the transport
    # context manager exactly as the adapter's HTTP client would provide it.
    adapter._bridge_req = (
        lambda _method, _path, _timeout, **_kwargs: adapter._http_session.get()
    )
    adapter._inbound_archive = None
    adapter._inbound_archive_home = home.resolve()
    adapter._archive_manifest_capability = object()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = lambda _data: True
    adapter._should_process_message = lambda _data: True
    adapter._message_is_reply_to_bot = lambda _data: True
    adapter._send_read_receipt = AsyncMock()
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    received = []

    async def capture(event):
        received.append(event)

    adapter.handle_message = capture
    await asyncio.wait_for(adapter._poll_messages(), timeout=2)
    await asyncio.sleep(0)

    assert len(received) == 1
    event = received[0]
    exposed = Path(event.media_urls[0])
    assert event.text == "Please review this contract"
    assert event.reply_to_text == "Archive the signed contract."
    assert event.reply_to_is_own_message
    assert event.media_types == ["application/pdf"]
    assert exposed.read_bytes() == b"%PDF-1.7 native bridge bytes"
    assert exposed.parent == home / "cache" / "documents"
    assert str(bridge_path) not in event.media_urls

    archive = adapter._inbound_archive_instance()
    with archive._connect() as db:
        owned = Path(db.execute("SELECT owned_path FROM archive_attachment").fetchone()[0])
    assert owned.read_bytes() == b"%PDF-1.7 native bridge bytes"
    assert owned.parent == archive.media_root

    prompt = await _runner()._prepare_inbound_message_text(event=event, source=_source(), history=[])
    assert str(exposed) in prompt
    assert "contract.pdf" in prompt
    assert 'Replying to your previous message: "Archive the signed contract."' in prompt
    assert str(bridge_path) not in prompt and str(owned) not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("quote_cache_state", ["present", "gone"])
async def test_native_reply_uses_prior_owned_attachment_not_bridge_quote_path(
    tmp_path, monkeypatch, quote_cache_state,
):
    """A reply/forward quote reuses the durable original bytes, not a cache URL."""
    home = tmp_path / "profile"
    image_cache = home / "cache" / "images"
    image_cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    bridge_original = image_cache / "bridge-original.png"
    bridge_original.write_bytes(_PNG)
    archive = WhatsAppInboundArchive(
        home / "whatsapp" / "inbound-archive-v1", home, image_cache,
    )
    original = _raw(
        "quoted-original", mediaType="image", mime="image/png", fileName="original.png",
        nativeMetadata={"album": {"groupId": "album-a", "role": "child", "messageIndex": 1}},
    )
    original_id, _ = archive.record(original, "observe")
    original_materialized = archive.materialize(original_id, original, [str(bridge_original)])
    assert original_materialized.complete

    reply = _raw(
        "reply-to-original", hasMedia=False, mediaType="", mime="", fileName="",
        body="Please review the image I replied to.",
        hasQuotedMessage=True, quotedMessageId="quoted-original",
        quotedRemoteJid="chat@g.us", quotedParticipant="1555000@s.whatsapp.net",
        quotedText="Original photo caption", quotedForwarded=True,
        quotedMediaUrls=([str(bridge_original)] if quote_cache_state == "present" else []),
        quotedMediaType="image",
    )
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._running = True
    adapter._bridge_port = 1
    adapter._http_session = _OneMessageSession(adapter, reply)
    adapter._bridge_req = lambda _method, _path, _timeout, **_kwargs: adapter._http_session.get()
    adapter._inbound_archive = archive
    adapter._inbound_archive_home = home.resolve()
    adapter._archive_manifest_capability = object()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = lambda _data: True
    adapter._should_process_message = lambda _data: True
    adapter._message_is_reply_to_bot = lambda _data: True
    adapter._send_read_receipt = AsyncMock()
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    received = []

    async def capture(event):
        received.append(event)

    adapter.handle_message = capture
    # The production text batching queue needs adapter configuration that this
    # focused native transport fixture intentionally does not construct. Keep
    # the poll/archive/event path and collect the just-built text event here.
    adapter._enqueue_text_event = received.append
    await asyncio.wait_for(adapter._poll_messages(), timeout=2)
    await asyncio.sleep(0)

    assert len(received) == 1
    event = received[0]
    exposed = Path(event.media_urls[0])
    assert exposed.read_bytes() == _PNG
    assert exposed.parent == home / "cache" / "images"
    assert str(bridge_original) not in event.media_urls
    assert str(original_materialized.owned_paths[0]) not in event.media_urls
    assert event.raw_message["quotedMediaUrls"] == [str(exposed)]
    assert event.reply_to_text == "Original photo caption"
    assert event.raw_message["quotedForwarded"] is True

    runner = _runner(); source = _source()
    await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    parts, skipped = build_native_content_parts(
        event.text, runner._consume_pending_native_image_paths(build_session_key(source)),
    )
    assert not skipped
    assert any(
        part.get("type") == "image_url"
        and part["image_url"]["url"].startswith("data:image/png;base64,")
        for part in parts
    )


@pytest.mark.asyncio
async def test_native_reply_to_owned_document_keeps_descriptor_quote_and_local_file(
    tmp_path, monkeypatch,
):
    """A reply to a native document reaches the turn as a real owned file, not a URL.

    The document's original native bridge write is intentionally used here: this
    covers the boundary where a later reply must recover the prior attachment
    from the archive even though its transient bridge-cache reference is no
    longer an authority for the model-facing event.
    """
    home = tmp_path / "profile"
    document_cache = home / "cache" / "documents"
    document_cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    original = _native_document_event(document_cache)
    bridge_original = Path(original["mediaUrls"][0])
    archive = WhatsAppInboundArchive(
        home / "whatsapp" / "inbound-archive-v1", home, document_cache,
    )
    original_id, _ = archive.record(original, "observe")
    original_materialized = archive.materialize(
        original_id, original, [str(bridge_original)],
    )
    assert original_materialized.complete

    reply = _raw(
        "reply-to-native-document", hasMedia=False, mediaType="", mime="", fileName="",
        body="Can you review the document I replied to?",
        hasQuotedMessage=True, quotedMessageId="native-document",
        quotedRemoteJid="chat@g.us", quotedParticipant="1555999@s.whatsapp.net",
        quotedText="Please review this contract", quotedForwarded=False,
        quotedMediaUrls=[str(bridge_original)], quotedMediaType="document",
    )
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._running = True
    adapter._bridge_port = 1
    adapter._http_session = _OneMessageSession(adapter, reply)
    adapter._bridge_req = lambda _method, _path, _timeout, **_kwargs: adapter._http_session.get()
    adapter._inbound_archive = archive
    adapter._inbound_archive_home = home.resolve()
    adapter._archive_manifest_capability = object()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = lambda _data: True
    adapter._should_process_message = lambda _data: True
    adapter._message_is_reply_to_bot = lambda _data: True
    adapter._send_read_receipt = AsyncMock()
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    received = []
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = received.append

    await asyncio.wait_for(adapter._poll_messages(), timeout=2)

    assert len(received) == 1
    event = received[0]
    exposed = Path(event.media_urls[0])
    assert event.media_types == ["application/pdf"]
    assert exposed.read_bytes() == b"%PDF-1.7 native bridge bytes"
    assert exposed.parent == home / "cache" / "documents"
    assert str(bridge_original) not in event.media_urls
    assert str(original_materialized.owned_paths[0]) not in event.media_urls
    assert event.reply_to_text == "Please review this contract"

    prompt = await _runner()._prepare_inbound_message_text(
        event=event, source=_source(), history=[],
    )
    assert "[The user sent a document: 'contract.pdf'. It is saved at:" in prompt
    assert str(exposed) in prompt
    assert 'Replying to your previous message: "Please review this contract"' in prompt
    assert str(bridge_original) not in prompt
    assert str(original_materialized.owned_paths[0]) not in prompt
    assert "blob:" not in prompt


def test_unarchived_quoted_attachment_is_not_exposed_from_bridge_cache(tmp_path, monkeypatch):
    """A quote without an owned original keeps its text but not a temporary path."""
    home = tmp_path / "profile"
    image_cache = home / "cache" / "images"
    image_cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    bridge_path = image_cache / "bridge-only.png"
    bridge_path.write_bytes(_PNG)
    adapter = object.__new__(WhatsAppAdapter)
    adapter._inbound_archive = WhatsAppInboundArchive(
        home / "whatsapp" / "inbound-archive-v1", home, image_cache,
    )
    adapter._inbound_archive_home = home.resolve()
    adapter._archive_manifest_capability = object()

    assert adapter._agent_visible_quoted_media_paths({
        "chatId": "chat@g.us",
        "quotedRemoteJid": "chat@g.us",
        "quotedMessageId": "missing-original",
        "quotedMediaUrls": [str(bridge_path)],
    }) == ()


@pytest.mark.asyncio
async def test_cross_chat_quoted_media_cannot_escape_its_archived_chat(tmp_path, monkeypatch):
    """An admitted reply cannot use its quote metadata to disclose another chat's file."""
    home = tmp_path / "profile"
    image_cache = home / "cache" / "images"
    image_cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    bridge_path = image_cache / "other-chat-bridge.png"
    bridge_path.write_bytes(_PNG)
    archive = WhatsAppInboundArchive(
        home / "whatsapp" / "inbound-archive-v1", home, image_cache,
    )
    other_chat_original = _raw(
        "cross-chat-original", chatId="other-chat@g.us", mediaType="image",
        mime="image/png", fileName="other-chat.png",
    )
    original_id, _ = archive.record(other_chat_original, "observe")
    assert archive.materialize(original_id, other_chat_original, [str(bridge_path)]).complete

    reply = _raw(
        "cross-chat-reply", hasMedia=False, mediaType="", mime="", fileName="",
        body="Please inspect the quoted image.", hasQuotedMessage=True,
        quotedMessageId="cross-chat-original", quotedRemoteJid="other-chat@g.us",
        quotedText="Other chat caption", quotedMediaUrls=[str(bridge_path)],
        quotedMediaType="image",
    )
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._running = True
    adapter._bridge_port = 1
    adapter._http_session = _OneMessageSession(adapter, reply)
    adapter._bridge_req = lambda _method, _path, _timeout, **_kwargs: adapter._http_session.get()
    adapter._inbound_archive = archive
    adapter._inbound_archive_home = home.resolve()
    adapter._archive_manifest_capability = object()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = lambda _data: True
    adapter._should_process_message = lambda _data: True
    adapter._message_is_reply_to_bot = lambda _data: True
    adapter._send_read_receipt = AsyncMock()
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    received = []
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = received.append

    await asyncio.wait_for(adapter._poll_messages(), timeout=2)

    assert len(received) == 1
    event = received[0]
    assert event.media_urls == []
    assert event.raw_message["quotedMediaUrls"] == []
    assert str(bridge_path) not in event.text
    runner = _runner(); source = _source()
    await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    parts, skipped = build_native_content_parts(
        event.text, runner._consume_pending_native_image_paths(build_session_key(source)),
    )
    assert not skipped
    assert not any(part.get("type") == "image_url" for part in parts)


@pytest.mark.asyncio
async def test_malformed_quote_media_is_cleared_before_model_content(tmp_path, monkeypatch):
    """A quote path cannot bypass archival by omitting its required identity."""
    home = tmp_path / "profile"
    image_cache = home / "cache" / "images"
    image_cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    bridge_path = image_cache / "malformed-quote.png"
    bridge_path.write_bytes(_PNG)
    malformed = _raw(
        "malformed-quote", hasMedia=False, mediaType="", mime="", fileName="",
        body="Please inspect the quote.", hasQuotedMessage=True,
        # A native reply must identify its quoted message. This malformed event
        # previously skipped the poll guard yet reached `_quoted_media`.
        quotedMessageId="", quotedMediaUrls=[str(bridge_path)], quotedMediaType="image",
    )
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._running = True
    adapter._bridge_port = 1
    adapter._http_session = _OneMessageSession(adapter, malformed)
    adapter._bridge_req = lambda _method, _path, _timeout, **_kwargs: adapter._http_session.get()
    adapter._inbound_archive = WhatsAppInboundArchive(
        home / "whatsapp" / "inbound-archive-v1", home, image_cache,
    )
    adapter._inbound_archive_home = home.resolve()
    adapter._archive_manifest_capability = object()
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = lambda _data: True
    adapter._should_process_message = lambda _data: True
    adapter._message_is_reply_to_bot = lambda _data: True
    adapter._send_read_receipt = AsyncMock()
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    received = []
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = received.append

    await asyncio.wait_for(adapter._poll_messages(), timeout=2)

    assert len(received) == 1
    event = received[0]
    assert event.media_urls == []
    assert event.raw_message["quotedMediaUrls"] == []
    runner = _runner(); source = _source()
    await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    parts, skipped = build_native_content_parts(
        event.text, runner._consume_pending_native_image_paths(build_session_key(source)),
    )
    assert not skipped
    assert not any(part.get("type") == "image_url" for part in parts)


@pytest.mark.asyncio
async def test_temporary_blob_reference_is_archived_as_missing_and_never_becomes_an_agent_attachment(tmp_path, monkeypatch):
    """The real poll path fails closed instead of handing a browser-only URL to Jack.

    Native bridge extraction normally downloads media to a profile-private
    cache path before it is spooled.  A ``blob:`` value is neither durable nor
    bridge-owned; it must leave a local download-status record and suppress
    the operating turn rather than being treated as a usable attachment.
    """
    home = tmp_path / "profile"
    cache = home / "cache" / "documents"
    cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    raw = _native_document_event(cache)
    raw["mediaUrls"] = ["blob:browser-temporary-contract"]
    # Exercise the initialized bridge path rather than the legacy no-lease
    # embedding seam.  The digest is the exact Node-spool digest over the
    # unleased event; a real bridge adds this delivery fence after staging.
    raw["_inboundLease"] = {
        "consumerId": "test-consumer",
        "deliveryId": "a" * 64,
        "eventDigest": bridge_event_digest(raw),
        "token": "leased-token",
        "epoch": 1,
        "expiresAt": int(time.time() * 1000) + 30_000,
    }

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = SimpleNamespace(value="whatsapp")
    adapter._running = True
    adapter._bridge_port = 1
    adapter._http_session = _OneMessageSession(adapter, raw)
    adapter._bridge_req = lambda _method, _path, _timeout, **_kwargs: adapter._http_session.get()
    adapter._inbound_archive = None
    adapter._inbound_archive_home = home.resolve()
    adapter._archive_manifest_capability = object()
    adapter._inbound_consumer_id = "test-consumer"
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._is_archive_authorized = lambda _data: True
    adapter._should_process_message = lambda _data: True
    adapter._send_read_receipt = AsyncMock()
    adapter.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
    received = []

    async def capture(event):
        received.append(event)

    adapter.handle_message = capture
    bridge_calls = []
    adapter._bridge_req = lambda method, path, _timeout, **kwargs: (
        bridge_calls.append((method, path, kwargs)) or adapter._http_session.get()
    )
    await asyncio.wait_for(adapter._poll_messages(), timeout=2)
    await asyncio.sleep(0)

    assert received == []
    archive = adapter._inbound_archive_instance()
    with archive._connect() as db:
        row = db.execute(
            "SELECT download_status, owned_path FROM archive_attachment"
        ).fetchone()
    assert row["download_status"] == "missing_or_rejected"
    assert row["owned_path"] is None
    with archive._connect() as db:
        receipt = db.execute("SELECT acknowledged FROM archive_bridge_receipt").fetchone()
    assert receipt["acknowledged"] == 0
    assert [call[:2] for call in bridge_calls] == [("get", "messages")]
