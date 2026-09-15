"""Opt-in local WhatsApp projection: no provider selection or history rewrite."""
from __future__ import annotations

import hashlib
import json
import asyncio
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.turn_context import build_turn_context
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
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
        "processor_timeout_seconds": 5, "processor_item_limit": 8,
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
    settings = ProjectionSettings.from_extra(_extra())
    projection = WhatsAppContextProjection(archive, settings)
    assert projection.activate(now=999) == 0
    media = cache / "private.jpg"
    media.write_bytes(b"private-media")
    first, accepted = archive.record(_raw("m1", media=True), "operate")
    assert accepted and archive.materialize(first, _raw("m1", media=True), [str(media)]).complete
    calls = []
    def processor(request):
        calls.append(request)
        return _processor(request)
    note = projection.capture_and_render(event_id=first, raw=_raw("m1", media=True), processor=processor, now=1000)
    assert note and "The approved local fact." in note and "image: owned" in note
    # Provenance is sufficient to audit the projection without disclosing raw
    # WhatsApp identities, message IDs, file names, or archive paths.
    assert re.search(r"event:[a-f0-9]{64}", note)
    assert re.search(r"chat:[a-f0-9]{64}", note)
    assert re.search(r"sender:[a-f0-9]{64}", note)
    assert re.search(r"message:[a-f0-9]{64}", note)
    assert re.search(r"attachment:[a-f0-9]{64}", note)
    assert "source_ms:1000" in note
    encoded = note.lower()
    for forbidden in ("ordinary text", "private.jpg", str(media).lower(), "private-media", "chat-a@g.us", "1555000@s.whatsapp.net", hashlib.sha256(b"private-media").hexdigest()):
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
    settings = ProjectionSettings.from_extra(_extra(
        cross_chat=True, authorized_chats=["chat-a@g.us", "chat-b@g.us"], retention_days=1,
    ))
    projection = WhatsAppContextProjection(archive, settings)
    assert projection.activate(now=999) == 0
    a, accepted = archive.record(_raw("a", chat="chat-a@g.us"), "operate")
    assert accepted
    b, accepted = archive.record(_raw("b", chat="chat-b@g.us"), "operate")
    assert accepted
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
    settings = ProjectionSettings.from_extra(_extra())
    projection = WhatsAppContextProjection(archive, settings)
    assert projection.activate(now=999) == 0
    event_id, accepted = archive.record(_raw("same"), "operate")
    assert accepted
    assert archive.record(_raw("same", body="different body"), "operate") == (event_id, False)
    note = projection.capture_and_render(
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


def test_activation_watermark_excludes_preexisting_rows_and_bounds_processor_output(tmp_path):
    archive, _cache = _archive(tmp_path)
    old_id, accepted = archive.record(_raw("old"), "operate")
    assert accepted
    settings = ProjectionSettings.from_extra(_extra(processor_item_limit=2))
    projection = WhatsAppContextProjection(archive, settings)
    assert projection.activate(now=1000) == old_id
    fresh_id, accepted = archive.record(_raw("fresh"), "operate")
    assert accepted
    calls = []
    def processor(_request):
        calls.append(True)
        return [ProjectionItem("fact", f"safe fact {index}") for index in range(10)]
    assert projection.capture_and_render(event_id=old_id, raw=_raw("old"), processor=processor) is None
    note = projection.capture_and_render(event_id=fresh_id, raw=_raw("fresh"), processor=processor)
    assert note and note.count("safe fact") == 2 and len(calls) == 1
    with archive._connect() as db:
        assert db.execute(
            "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? AND event_id=?",
            (archive.scope, fresh_id),
        ).fetchone()["state"] == "completed"


def test_render_revalidates_destination_scope_and_stale_provenance(tmp_path):
    archive, _cache = _archive(tmp_path)
    settings = ProjectionSettings.from_extra(_extra(cross_chat=True, authorized_chats=["chat-a@g.us"]))
    projection = WhatsAppContextProjection(archive, settings)
    projection.activate(now=1000)
    event_id, accepted = archive.record(_raw("m", chat="chat-a@g.us"), "operate")
    assert accepted
    assert projection.capture_and_render(event_id=event_id, raw=_raw("m", chat="chat-a@g.us"), processor=_processor)

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._inbound_archive = archive
    adapter.config = SimpleNamespace(extra=_extra(cross_chat=True, authorized_chats=["chat-a@g.us"]))
    event = SimpleNamespace(
        _whatsapp_context_projection_event_ids=(event_id,),
        source=SimpleNamespace(chat_id="chat-a@g.us"),
    )
    assert "untrusted" in adapter._render_context_projection_notes(event)
    # A live scope/config revocation is checked at render time, even for an
    # event that was already accepted and batched.
    adapter.config.extra = _extra(cross_chat=True, authorized_chats=["other@g.us"])
    assert adapter._render_context_projection_notes(event) is None
    adapter.config.extra = _extra(cross_chat=True, authorized_chats=["chat-a@g.us"])
    with archive._connect() as db:
        db.execute("UPDATE archive_event SET event_digest='0' WHERE id=?", (event_id,))
    assert adapter._render_context_projection_notes(event) is None
    with archive._connect() as db:
        row = db.execute(
            "SELECT text,tombstoned_at FROM archive_context_projection_item WHERE profile_scope=? AND event_id=?",
            (archive.scope, event_id),
        ).fetchone()
    assert row["text"] == "" and row["tombstoned_at"] is not None


def test_processor_failure_is_durably_uncertain_and_not_retried(tmp_path):
    archive, _cache = _archive(tmp_path)
    settings = ProjectionSettings.from_extra(_extra())
    projection = WhatsAppContextProjection(archive, settings)
    projection.activate(now=1000)
    event_id, accepted = archive.record(_raw("m"), "operate")
    assert accepted
    calls = []
    def fail(_request):
        calls.append(True)
        raise RuntimeError("processor unavailable")
    assert projection.capture_and_render(event_id=event_id, raw=_raw("m"), processor=fail) is None
    assert projection.capture_and_render(event_id=event_id, raw=_raw("m"), processor=fail) is None
    assert calls == [True]
    with archive._connect() as db:
        assert db.execute(
            "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? AND event_id=?",
            (archive.scope, event_id),
        ).fetchone()["state"] == "uncertain"


def test_processor_deadline_marks_uncertain_without_a_projection_row(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import plugins.platforms.whatsapp.context_projection as projection_module

    archive, _cache = _archive(tmp_path)
    settings = ProjectionSettings.from_extra(_extra(processor_timeout_seconds=1))
    projection = WhatsAppContextProjection(archive, settings)
    projection.activate(now=1000)
    event_id, accepted = archive.record(_raw("m"), "operate")
    assert accepted
    def slow(_request):
        time.sleep(1.05)
        return [ProjectionItem("fact", "too late")]
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(projection_module, "_PROCESSOR_EXECUTOR", executor)
    monkeypatch.setattr(projection_module, "_PROCESSOR_SLOT", threading.BoundedSemaphore(1))
    try:
        assert projection.capture_and_render(event_id=event_id, raw=_raw("m"), processor=slow) is None
    finally:
        # A running thread cannot be killed; waiting here prevents it from
        # occupying a later test's bounded admission slot.
        executor.shutdown(wait=True)
    with archive._connect() as db:
        claim = db.execute(
            "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? AND event_id=?",
            (archive.scope, event_id),
        ).fetchone()
        event = db.execute(
            "SELECT 1 FROM archive_context_projection_event WHERE profile_scope=? AND event_id=?",
            (archive.scope, event_id),
        ).fetchone()
    assert claim["state"] == "uncertain" and event is None


def test_processor_consumes_only_bounded_items_inside_worker(tmp_path):
    """An endless invalid iterator cannot outlive the configured examination cap."""
    archive, _cache = _archive(tmp_path)
    projection = WhatsAppContextProjection(
        archive, ProjectionSettings.from_extra(_extra(processor_item_limit=3)),
    )
    projection.activate(now=1000)
    event_id, accepted = archive.record(_raw("bounded"), "operate")
    assert accepted
    examined = []

    def endless_invalid(_request):
        while True:
            examined.append(True)
            yield {"kind": "not-an-item", "text": "discard"}

    assert projection.capture_and_render(event_id=event_id, raw=_raw("bounded"), processor=endless_invalid) is None
    assert len(examined) == 3
    with archive._connect() as db:
        assert db.execute(
            "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? AND event_id=?",
            (archive.scope, event_id),
        ).fetchone()["state"] == "completed"


def test_saturated_processor_admission_never_queues_and_releases_after_worker_exit(tmp_path, monkeypatch):
    """A timed-out running worker holds the only slot until it has actually exited."""
    from concurrent.futures import ThreadPoolExecutor
    import plugins.platforms.whatsapp.context_projection as projection_module

    archive, _cache = _archive(tmp_path)
    projection = WhatsAppContextProjection(
        archive, ProjectionSettings.from_extra(_extra(processor_timeout_seconds=1)),
    )
    projection.activate(now=1000)
    slow_id, accepted = archive.record(_raw("slow"), "operate"); assert accepted
    queued_id, accepted = archive.record(_raw("queued"), "operate"); assert accepted
    recovered_id, accepted = archive.record(_raw("recovered"), "operate"); assert accepted
    started, release = threading.Event(), threading.Event()
    queued_calls = []
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(projection_module, "_PROCESSOR_EXECUTOR", executor)
    monkeypatch.setattr(projection_module, "_PROCESSOR_SLOT", threading.BoundedSemaphore(1))

    def slow(_request):
        started.set()
        assert release.wait(timeout=5)
        return (ProjectionItem("fact", "too late"),)

    def queued(_request):
        queued_calls.append(True)
        return (ProjectionItem("fact", "must not run"),)

    try:
        assert projection.capture_and_render(event_id=slow_id, raw=_raw("slow"), processor=slow) is None
        assert started.is_set()
        began = time.monotonic()
        assert projection.capture_and_render(event_id=queued_id, raw=_raw("queued"), processor=queued) is None
        assert time.monotonic() - began < 0.25  # no internal executor queue wait
        assert queued_calls == []
        release.set()
        # A barrier proves the timed-out worker has returned and its future
        # completion callback released the slot before another admission.
        executor.submit(lambda: None).result(timeout=2)
        assert projection.capture_and_render(
            event_id=recovered_id, raw=_raw("recovered"),
            processor=lambda _request: (ProjectionItem("fact", "after exit"),),
        )
    finally:
        release.set()
        executor.shutdown(wait=True)
    assert queued_calls == []
    with archive._connect() as db:
        states = [row["state"] for row in db.execute(
            "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? ORDER BY event_id", (archive.scope,),
        )]
    assert states == ["uncertain", "uncertain", "completed"]


def test_timeout_before_worker_start_cancels_without_invocation(tmp_path, monkeypatch):
    """An executor-start race receives an absolute deadline and no late callback runs."""
    from concurrent.futures import ThreadPoolExecutor
    import plugins.platforms.whatsapp.context_projection as projection_module

    archive, _cache = _archive(tmp_path)
    projection = WhatsAppContextProjection(
        archive, ProjectionSettings.from_extra(_extra(processor_timeout_seconds=1)),
    )
    projection.activate(now=1000)
    event_id, accepted = archive.record(_raw("start-race"), "operate"); assert accepted
    started, release = threading.Event(), threading.Event()
    calls = []
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(projection_module, "_PROCESSOR_EXECUTOR", executor)
    monkeypatch.setattr(projection_module, "_PROCESSOR_SLOT", threading.BoundedSemaphore(1))

    def occupy():
        started.set()
        assert release.wait(timeout=5)
    blocker = executor.submit(occupy)
    assert started.wait(timeout=2)
    try:
        assert projection.capture_and_render(
            event_id=event_id, raw=_raw("start-race"),
            processor=lambda _request: calls.append(True) or (),
        ) is None
    finally:
        release.set()
        blocker.result(timeout=2)
        executor.shutdown(wait=True)
    assert calls == []


def test_worker_refuses_invocation_when_started_after_absolute_deadline():
    """The worker itself checks the deadline, closing submit/start races."""
    from plugins.platforms.whatsapp.context_projection import (
        ProjectionInput, _bounded_processor_items,
    )

    calls = []
    request = ProjectionInput("scope", "event:0", "chat:0", "sender:0", "body", ())
    assert _bounded_processor_items(
        lambda _request: calls.append(True) or (ProjectionItem("fact", "late"),),
        request, 1, time.monotonic() - 0.001,
    ) == ()
    assert calls == []


def test_late_attempt_cannot_publish_after_concurrent_uncertain_claim(tmp_path, monkeypatch):
    """A late worker result is fenced by its attempt ID after a concurrent recovery sees it."""
    from concurrent.futures import ThreadPoolExecutor
    import plugins.platforms.whatsapp.context_projection as projection_module

    archive, _cache = _archive(tmp_path)
    projection = WhatsAppContextProjection(archive, ProjectionSettings.from_extra(_extra()))
    projection.activate(now=1000)
    event_id, accepted = archive.record(_raw("race"), "operate")
    assert accepted
    started, release = threading.Event(), threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(projection_module, "_PROCESSOR_EXECUTOR", executor)
    monkeypatch.setattr(projection_module, "_PROCESSOR_SLOT", threading.BoundedSemaphore(1))

    def blocked(_request):
        started.set()
        assert release.wait(timeout=5)
        return (ProjectionItem("fact", "late result"),)

    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            projection.capture_and_render(event_id=event_id, raw=_raw("race"), processor=blocked)
        ),
    )
    worker.start()
    assert started.wait(timeout=2)
    try:
        assert projection.capture_and_render(event_id=event_id, raw=_raw("race"), processor=blocked) is None
    finally:
        release.set()
        worker.join(timeout=5)
        executor.shutdown(wait=True)
    assert not worker.is_alive() and result == [None]
    with archive._connect() as db:
        claim = db.execute(
            "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? AND event_id=?",
            (archive.scope, event_id),
        ).fetchone()
        event = db.execute(
            "SELECT 1 FROM archive_context_projection_event WHERE profile_scope=? AND event_id=?",
            (archive.scope, event_id),
        ).fetchone()
    assert claim["state"] == "uncertain" and event is None


def test_render_resanitizes_hostile_persisted_projection_rows(tmp_path):
    archive, _cache = _archive(tmp_path)
    projection = WhatsAppContextProjection(archive, ProjectionSettings.from_extra(_extra()))
    projection.activate(now=1000)
    event_id, accepted = archive.record(_raw("hostile", media=True), "operate")
    assert accepted
    assert projection.capture_and_render(event_id=event_id, raw=_raw("hostile", media=True), processor=_processor)
    hostile = "https://attacker.invalid/private /tmp/private.jpg"
    with archive._connect() as db:
        db.execute(
            "INSERT INTO archive_context_projection_item(profile_scope,event_id,chat_id,kind,text,created_at) VALUES(?,?,?,?,?,?)",
            (archive.scope, event_id, "chat-a@g.us", "attacker", hostile, 1001),
        )
        db.execute(
            "UPDATE archive_context_projection_event SET metadata_json=? WHERE profile_scope=? AND event_id=?",
            (json.dumps({"attachments": [
                {"kind": "../../file", "status": {"leak": True}, "attachment_ref": "file:/tmp/private"},
                {"kind": ["image"], "status": ["owned"], "attachment_ref": ["attachment:bad"]},
            ]}), archive.scope, event_id),
        )
    note = projection.render(event_id=event_id, chat_id="chat-a@g.us")
    assert note and "The approved local fact." in note and "media: unknown_historical" in note
    assert hostile not in note and "attacker" not in note and "file:/tmp" not in note


@pytest.mark.asyncio
async def test_adapter_batch_to_runner_preserves_history_and_cached_prefix(tmp_path, monkeypatch):
    """Real adapter batching reaches the real runner before the mocked model edge."""
    from gateway.run import GatewayRunner

    home = (tmp_path / "profile").resolve(); (home / "cache").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    extra = _extra()
    extra.update({"dm_policy": "open", "text_batch_delay_seconds": 0.0, "text_batch_split_delay_seconds": 0.0})
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = WhatsAppAdapter(config)
    adapter._inbound_archive_home = home
    adapter._inbound_archive = WhatsAppInboundArchive(home / "whatsapp" / "inbound-archive-v1", home, home / "cache")
    adapter.set_context_projection_processor(
        lambda _request: (ProjectionItem("fact", "The bounded test fact."),)
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="final"))
    runner = GatewayRunner(GatewayConfig(platforms={Platform.WHATSAPP: config}))
    runner.adapters = {Platform.WHATSAPP: adapter}; runner._profile_adapters = {}
    runner._authorization_home_for_source = lambda _source: None
    runner._is_user_authorized = lambda _source, **_kwargs: True
    runner._admit_bot_message = lambda _source: True
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    calls = []

    async def provider(**kwargs):
        calls.append({
            "message": kwargs["message"], "context_prompt": kwargs["context_prompt"],
            "history": list(kwargs["history"]),
            "sidecars": list(runner._peek_session_state(kwargs["session_key"]).conversation.sidecar_notes),
        })
        final = f"final-{len(calls)}"
        return {"final_response": final, "messages": [
            {"role": "user", "content": kwargs["message"]}, {"role": "assistant", "content": final},
        ], "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1,
            "agent_persisted": False, "failed": False}

    runner._run_agent = provider
    source = adapter.build_source(chat_id="1555000@s.whatsapp.net", chat_type="dm", user_id="1555000@s.whatsapp.net", message_id="prior")
    await adapter.handle_message(MessageEvent(text="prior history", message_type=MessageType.TEXT, source=source, message_id="prior"))
    for _ in range(100):
        if len(calls) == 1:
            break
        await asyncio.sleep(0.01)
    assert len(calls) == 1
    projection = adapter._context_projection_for_activation()
    raw_one = _raw("one", chat="1555000@s.whatsapp.net", body="first merged")
    raw_two = _raw("two", chat="1555000@s.whatsapp.net", body="second merged")
    one_id, accepted = adapter._inbound_archive.record(raw_one, "operate"); assert accepted
    two_id, accepted = adapter._inbound_archive.record(raw_two, "operate"); assert accepted
    await asyncio.to_thread(adapter._capture_context_projection_note, one_id, raw_one, projection)
    await asyncio.to_thread(adapter._capture_context_projection_note, two_id, raw_two, projection)
    first = MessageEvent(text="first merged", message_type=MessageType.TEXT, source=source, message_id="one")
    second = MessageEvent(text="second merged", message_type=MessageType.TEXT, source=source, message_id="two")
    first._whatsapp_context_projection_event_ids = (one_id,)
    second._whatsapp_context_projection_event_ids = (two_id,)
    adapter._enqueue_text_event(first); adapter._enqueue_text_event(second)
    await asyncio.gather(*tuple(adapter._pending_text_batch_tasks.values()))
    for _ in range(100):
        if len(calls) == 2:
            break
        await asyncio.sleep(0.01)
    for _ in range(100):
        if adapter.send.await_count == 2:
            break
        await asyncio.sleep(0.01)

    assert len(calls) == 2
    merged = calls[1]
    assert merged["message"] == "first merged\nsecond merged"
    assert merged["context_prompt"] == calls[0]["context_prompt"]
    assert any(item.get("content") == "prior history" for item in merged["history"])
    assert "Local WhatsApp context projection" in "\n".join(merged["sidecars"])
    assert all("Local WhatsApp context projection" not in str(item.get("content")) for item in merged["history"])
    assert adapter.send.await_count == 2
