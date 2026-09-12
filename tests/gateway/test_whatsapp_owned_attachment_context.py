"""Real model-facing paths for owned WhatsApp operating attachments."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.image_routing import build_native_content_parts
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from plugins.platforms.whatsapp.inbound_archive import WhatsAppInboundArchive


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

    assert result.complete and event.text == "caption kept"
    assert event.message_type == MessageType.PHOTO and event.media_urls == list(result.owned_paths)
    assert str(bridge_path) not in event.media_urls
    runner = _runner(); source = _source()
    await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    parts, skipped = build_native_content_parts(
        "caption kept", runner._consume_pending_native_image_paths(build_session_key(source)),
    )
    assert not skipped and any(part.get("type") == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,") for part in parts)


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
