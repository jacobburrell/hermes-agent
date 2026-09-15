"""Opt-in local WhatsApp projection: no provider selection or history rewrite."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.turn_context import build_turn_context
from gateway.config import Platform
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from plugins.platforms.whatsapp.context_projection import (
    ProjectionItem,
    ProjectionSettings,
    WhatsAppContextProjection,
)
from plugins.platforms.whatsapp.inbound_archive import WhatsAppInboundArchive


def _raw(mid: str, *, chat: str = "chat-a@g.us", body: str = "ordinary text", media: bool = False):
    return {
        "messageId": mid, "chatId": chat, "senderId": "1555000@s.whatsapp.net",
        "body": body, "timestamp": 1, "hasMedia": media,
        **({"mediaType": "image", "mime": "image/jpeg", "fileName": "private.jpg"} if media else {}),
    }


def _archive(tmp_path: Path, name: str = "profile") -> tuple[WhatsAppInboundArchive, Path]:
    home = (tmp_path / name).resolve()
    cache = home / "cache"
    cache.mkdir(parents=True)
    return WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, cache), cache


def _extra(**overrides):
    value = {
        "enabled": True, "cross_chat": False, "authorized_chats": [],
        "retention_days": 30, "deletion": "tombstone", "per_chat_limit": 8,
        "profile_limit": 16, "attachment_limit": 4,
    }
    value.update(overrides)
    return {"context_projection": value}


def _processor(request):
    assert request.text == "ordinary text"
    assert all("path" not in attachment for attachment in request.attachments)
    return (ProjectionItem("fact", "The approved local fact."),)


def test_disabled_or_malformed_config_never_calls_processor(tmp_path):
    archive, _cache = _archive(tmp_path)
    event_id, accepted = archive.record(_raw("m1"), "operate")
    assert accepted
    calls = []

    def processor(_request):
        calls.append(True)
        return ()

    assert ProjectionSettings.from_extra({"context_projection": {"enabled": False}}) is None
    assert ProjectionSettings.from_extra(_extra(retention_days="30")) is None
    # Adapter construction is the runtime boundary: no valid opt-in means it
    # does not construct a projection or invoke an injected callback.
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = SimpleNamespace(extra={"context_projection": {"enabled": False}})
    adapter._context_projection_processor = processor
    adapter._inbound_archive = archive
    assert adapter._capture_context_projection_note(event_id, _raw("m1")) is None
    assert calls == []


def test_projection_is_rollout_forward_profile_scoped_and_attachment_redacted(tmp_path):
    archive, cache = _archive(tmp_path)
    media = cache / "private.jpg"
    media.write_bytes(b"private-media")
    first, accepted = archive.record(_raw("m1", media=True), "operate")
    assert accepted and archive.materialize(first, _raw("m1", media=True), [str(media)]).complete
    settings = ProjectionSettings.from_extra(_extra())
    projection = WhatsAppContextProjection(archive, settings)
    calls = []
    def processor(request):
        calls.append(request)
        return _processor(request)
    note = projection.capture_and_render(event_id=first, raw=_raw("m1", media=True), processor=processor, now=1000)
    assert note and "The approved local fact." in note and "image: owned" in note
    encoded = note.lower()
    for forbidden in ("ordinary text", "private.jpg", str(media).lower(), "private-media", hashlib.sha256(b"private-media").hexdigest()):
        assert forbidden not in encoded

    # Reopening keeps only rollout-forward data and profile-private scope.
    reopened = WhatsAppContextProjection(
        WhatsAppInboundArchive(archive.root, archive.profile_home, cache), settings,
    )
    assert "The approved local fact." in reopened.render(event_id=first, chat_id="chat-a@g.us", now=1001)
    assert "The approved local fact." in reopened.capture_and_render(
        event_id=first, raw=_raw("m1", media=True), processor=processor, now=1002,
    )
    assert len(calls) == 1  # restart/replay never reprocesses old source text
    other, _ = _archive(tmp_path, "other-profile")
    other_id, accepted = other.record(_raw("m1"), "operate")
    assert accepted
    assert WhatsAppContextProjection(other, settings).render(event_id=other_id, chat_id="chat-a@g.us") is None


def test_cross_chat_requires_explicit_allowlist_and_retention_preserves_tombstone(tmp_path):
    archive, _cache = _archive(tmp_path)
    a, accepted = archive.record(_raw("a", chat="chat-a@g.us"), "operate")
    assert accepted
    b, accepted = archive.record(_raw("b", chat="chat-b@g.us"), "operate")
    assert accepted
    settings = ProjectionSettings.from_extra(_extra(
        cross_chat=True, authorized_chats=["chat-a@g.us", "chat-b@g.us"], retention_days=1,
    ))
    projection = WhatsAppContextProjection(archive, settings)
    assert projection.capture_and_render(event_id=a, raw=_raw("a", chat="chat-a@g.us"), processor=_processor, now=1000)
    note = projection.capture_and_render(event_id=b, raw=_raw("b", chat="chat-b@g.us"), processor=_processor, now=1001)
    assert note and note.count("The approved local fact.") == 2
    assert ProjectionSettings.from_extra(_extra(cross_chat=True, authorized_chats=[])) is None

    assert projection.render(event_id=b, chat_id="chat-b@g.us", now=1000 + 86402) is None
    with archive._connect() as db:
        rows = db.execute("SELECT text,tombstoned_at FROM archive_context_projection_item ORDER BY id").fetchall()
    assert rows and all(row["text"] == "" and row["tombstoned_at"] is not None for row in rows)


def test_conflict_evidence_is_proven_and_sidecar_keeps_transcript_content_stable(tmp_path):
    archive, _cache = _archive(tmp_path)
    event_id, accepted = archive.record(_raw("same"), "operate")
    assert accepted
    assert archive.record(_raw("same", body="different body"), "operate") == (event_id, False)
    settings = ProjectionSettings.from_extra(_extra())
    note = WhatsAppContextProjection(archive, settings).capture_and_render(
        event_id=event_id, raw=_raw("same"), processor=_processor,
    )
    assert note and "identity conflict retained" in note.lower()

    # The actual model-boundary helper receives the sidecar only in api_content;
    # stored conversation text stays byte-identical and no history is rewritten.
    class Agent:
        session_id = "s"; model = "m"; provider = "p"; base_url = ""; api_key = ""
        api_mode = "chat_completions"; platform = "whatsapp"; quiet_mode = True; max_iterations = 1
        tools = []; valid_tool_names = set(); _skip_mcp_refresh = True; compression_enabled = False
        _cached_system_prompt = "SYSTEM"; _memory_store = _memory_manager = None
        _memory_nudge_interval = _turns_since_memory = _user_turn_count = 0
        _todo_store = SimpleNamespace(has_items=lambda: False)
        _tool_guardrails = SimpleNamespace(reset_for_turn=lambda: None)
        _compression_warning = None; _interrupt_requested = False; _memory_write_origin = "assistant_tool"
        _stream_context_scrubber = _stream_think_scrubber = None
        context_compressor = SimpleNamespace(protect_first_n=2, protect_last_n=2)
        def _ensure_db_session(self): pass
        def _restore_primary_runtime(self): pass
        def _cleanup_dead_connections(self): return False
        def _emit_status(self, _message): pass
        def _replay_compression_warning(self): pass
        def _hydrate_todo_store(self, *_args): pass
        def _safe_print(self, *_args): pass
        def _persist_session(self, *_args): pass

    agent = Agent()
    agent._gateway_turn_context_notes = note
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]), patch(
        "agent.auxiliary_client.set_runtime_main", lambda *_args: None,
    ):
        ctx = build_turn_context(
            agent=agent, user_message="current message", system_message=None, conversation_history=None,
            task_id=None, stream_callback=None, persist_user_message=None,
            restore_or_build_system_prompt=lambda *_a, **_k: None, install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda value: value, summarize_user_message_for_log=str,
            set_session_context=lambda _value: None, set_current_write_origin=lambda _value: None,
            ra=lambda: SimpleNamespace(_set_interrupt=lambda *_a: None),
        )
    message = ctx.messages[ctx.current_turn_user_idx]
    assert message["content"] == "current message"
    assert message["api_content"] == "current message\n\n" + note
