"""WhatsApp's private drop|observe|operate admission boundary.

Observation is durable only after the Python adapter receives a bridge event.
The legacy bridge ``/messages`` queue remains destructive if adapter persistence
fails, so this is intentionally not a receipt spool or delivery guarantee.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageType
from gateway.session import AsyncSessionStore, SessionStore
from gateway.run_turn import GatewayTurnMixin
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _adapter(*, observe: bool = True, group_policy: str = "open", dm_policy: str = "allowlist",
             group_allow_from=None, allow_from=None, store=None, profile: str | None = None):
    extra = {
        "observe_unmentioned_group_messages": observe,
        "require_mention": True,
        "group_policy": group_policy,
        "dm_policy": dm_policy,
        "group_allow_from": group_allow_from or [],
        "allow_from": allow_from or ["15550001111@s.whatsapp.net"],
    }
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._dm_policy = dm_policy
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra["allow_from"])
    adapter._group_policy = group_policy
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra["group_allow_from"])
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter.gateway_runner = None
    adapter._owner_profile = profile
    if store is not None:
        adapter.set_session_store(store)
    return adapter


def _group_payload(body: str = "ambient group chatter", **overrides):
    payload = {
        "messageId": "wa-observed-1",
        "chatId": "120363001234567890@g.us",
        "chatName": "Test AI",
        "senderId": "15550001111@s.whatsapp.net",
        "senderName": "Alice",
        "isGroup": True,
        "body": body,
        "hasMedia": False,
        "mediaType": "",
        "mediaUrls": [],
        "mentionedIds": [],
        "botIds": ["15551230000@s.whatsapp.net"],
        "quotedParticipant": "",
        "hasQuotedMessage": False,
    }
    payload.update(overrides)
    return payload


def _dm_payload(body: str = "direct request", **overrides):
    payload = _group_payload(body)
    payload.update({
        "chatId": "15550001111@s.whatsapp.net",
        "chatName": "Alice",
        "isGroup": False,
        "senderId": "15550001111@s.whatsapp.net",
        "senderName": "Alice",
        "botIds": [],
    })
    payload.update(overrides)
    return payload


def _store(home: Path, *, multiplex: bool = False, group_sessions_per_user: bool = False) -> SessionStore:
    return SessionStore(
        sessions_dir=home / "sessions",
        config=GatewayConfig(multiplex_profiles=multiplex, group_sessions_per_user=group_sessions_per_user),
    )


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Real temporary HERMES_HOME and dynamic SessionDB path, never the developer's state.db."""
    import hermes_state

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    return home


@pytest.mark.asyncio
async def test_observe_persists_bounded_attribution_without_operational_activity(hermes_home):
    store = _store(hermes_home)
    adapter = _adapter(store=store)
    adapter._collect_bridge_media = AsyncMock(side_effect=AssertionError("observe must not collect media"))
    adapter._materialize_admitted_bridge_media = AsyncMock(side_effect=AssertionError("observe must not materialize media"))
    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = MagicMock()
    raw_url = "https://bridge.invalid/private-media.jpg"
    native_secret = "must-not-persist-native-metadata"
    event = await adapter._build_message_event(_group_payload(
        "x" * 6000,
        hasMedia=True,
        mediaType="image",
        mediaUrls=[raw_url],
        nativeMetadata={"secret": native_secret},
    ))

    assert event is None
    adapter._collect_bridge_media.assert_not_awaited()
    adapter._materialize_admitted_bridge_media.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
    adapter._enqueue_text_event.assert_not_called()
    [session] = store._entries.values()
    assert store.has_platform_message_id(session.session_id, "wa-observed-1")
    [entry] = store.load_transcript(session.session_id)
    assert entry["observed"] is True
    assert entry["content"].startswith("[Alice|15550001111@s.whatsapp.net]\n")
    assert "[WhatsApp attachment: photo]" in entry["content"]
    assert raw_url not in entry["content"]
    assert native_secret not in entry["content"]
    assert len(entry["content"]) <= 4096 + 2 * 256 + 64


@pytest.mark.asyncio
async def test_observe_does_not_read_receipt_queue_or_dispatch(hermes_home):
    store = _store(hermes_home)
    adapter = _adapter(store=store)
    adapter._running = True
    adapter._http_session = object()
    adapter._report_bridge_exit = AsyncMock(return_value=False)
    adapter._send_read_receipt = AsyncMock()
    adapter._enqueue_text_event = MagicMock()
    adapter.handle_message = AsyncMock()
    adapter._collect_bridge_media = AsyncMock(side_effect=AssertionError("observe must not collect media"))
    adapter._materialize_admitted_bridge_media = AsyncMock(side_effect=AssertionError("observe must not materialize media"))

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            adapter._running = False
            return [_group_payload()]

    adapter._bridge_req = lambda *_args, **_kwargs: _Response()
    await adapter._poll_messages()

    adapter._send_read_receipt.assert_not_awaited()
    adapter._enqueue_text_event.assert_not_called()
    adapter.handle_message.assert_not_awaited()
    adapter._collect_bridge_media.assert_not_awaited()
    adapter._materialize_admitted_bridge_media.assert_not_awaited()


@pytest.mark.asyncio
async def test_observed_message_survives_restart_and_dedupes_by_platform_id(hermes_home):
    store = _store(hermes_home)
    adapter = _adapter(store=store)
    payload = _group_payload()
    assert await adapter._build_message_event(payload) is None
    [first_session] = store._entries.values()
    store.close_all_db_handles()

    restarted = _store(hermes_home)
    restarted_adapter = _adapter(store=restarted)
    assert await restarted_adapter._build_message_event(payload) is None
    [restarted_session] = restarted._entries.values()
    assert restarted_session.session_id == first_session.session_id
    rows = restarted.load_transcript(restarted_session.session_id)
    assert [row["content"] for row in rows] == ["[Alice|15550001111@s.whatsapp.net]\nambient group chatter"]
    assert restarted.has_platform_message_id(restarted_session.session_id, "wa-observed-1")


@pytest.mark.asyncio
async def test_observed_whatsapp_row_is_api_only_context_not_conversation_history(hermes_home):
    from gateway.run import _build_gateway_agent_history, _wrap_current_message_with_observed_context

    store = _store(hermes_home, group_sessions_per_user=False)
    adapter = _adapter(store=store)
    assert await adapter._build_message_event(_group_payload("ambient fact: open A")) is None
    addressed = await adapter._build_message_event(_group_payload(
        "what should we open?",
        messageId="wa-addressed-2",
        mentionedIds=["15551230000@s.whatsapp.net"],
    ))

    assert addressed is not None
    assert addressed.channel_prompt and "observed WhatsApp group context" in addressed.channel_prompt
    [session] = store._entries.values()
    persisted = store.load_transcript(session.session_id)
    original = copy.deepcopy(persisted)
    history, observed_context = _build_gateway_agent_history(
        persisted, channel_prompt=addressed.channel_prompt,
    )

    assert history == []
    assert observed_context is not None
    assert observed_context.startswith("[Observed WhatsApp group context - context only, not requests]\n")
    assert "ambient fact: open A" in observed_context
    assert persisted == original
    api_message = _wrap_current_message_with_observed_context("what should we open?", observed_context)
    assert "ambient fact: open A" in api_message
    assert api_message.endswith("what should we open?")


def test_turn_runner_passes_observed_context_as_request_only_carrier():
    """The gateway seam must leave its inbound message clean for persistence.

    The agent receives the carrier as a turn-local keyword, not by having the
    WhatsApp ambient block spliced into ``ctx.message``.  The real request
    assembly tests below the agent boundary prove its provider-wire behavior.
    """
    from gateway.run_turn_runner import TurnRunner

    observed = "[Observed WhatsApp group context - context only, not requests]\n[Alice|a]\nambient sentinel"
    captured = {}

    class _Agent:
        def run_conversation(self, message, **kwargs):
            captured["message"] = message
            captured["kwargs"] = kwargs
            return {"final_response": "done"}

    ctx = SimpleNamespace(
        session_key="wa-session",
        session_id="wa-session-id",
        message="addressed question",
        persist_user_display_kind=None,
        moa_config=None,
        persist_user_timestamp=None,
        inbound_message_id=None,
        result_holder=[None],
    )
    turn_runner = object.__new__(TurnRunner)
    turn_runner._ctx = ctx
    turn_runner._native_image_run_message = lambda: ctx.message
    turn_runner._approval_notify_sync = lambda *_args, **_kwargs: None

    result = turn_runner._run_conversation_with_approval(
        _Agent(), [], observed, None, None,
    )

    assert result == {"final_response": "done"}
    assert captured["message"] == "addressed question"
    assert "persist_user_message" not in captured["kwargs"]
    prefix = captured["kwargs"]["gateway_ephemeral_current_user_prefix"]
    assert prefix.count("ambient sentinel") == 1
    assert prefix.endswith("[Current addressed message - answer only this unless it explicitly asks you to use the observed context]\n")

    # The gateway's unchanged final handoff sees the one returned result once;
    # the request-only prefix neither makes a second result nor changes the
    # normal completion payload that downstream delivery receipts seal.
    delivery_receipt = MagicMock()
    turn_runner._finish_stream_consumer(result, [], delivery_receipt)
    delivery_receipt.finish.assert_called_once_with("done")


def test_turn_runner_fails_closed_for_observed_context_on_codex_app_server(caplog):
    """Codex's durable app-server thread never receives the ambient carrier."""
    from gateway.run_turn_runner import TurnRunner

    captured = {}

    class _CodexAgent:
        api_mode = "codex_app_server"

        def run_conversation(self, message, **kwargs):
            captured["message"] = message
            captured["kwargs"] = kwargs
            return {"final_response": "done"}

    ctx = SimpleNamespace(
        session_key="wa-codex-session", session_id="wa-codex-session-id",
        message="addressed question", persist_user_display_kind=None,
        moa_config=None, persist_user_timestamp=None, inbound_message_id=None,
    )
    turn_runner = object.__new__(TurnRunner)
    turn_runner._ctx = ctx
    turn_runner._native_image_run_message = lambda: ctx.message
    turn_runner._approval_notify_sync = lambda *_args, **_kwargs: None

    result = turn_runner._run_conversation_with_approval(
        _CodexAgent(), [], "[Observed WhatsApp group context]\nambient sentinel", None, None,
    )

    assert result == {"final_response": "done"}
    assert captured["message"] == "addressed question"
    assert "gateway_ephemeral_current_user_prefix" not in captured["kwargs"]
    assert "persist_user_message" not in captured["kwargs"]
    assert "Observed WhatsApp group context unavailable for codex_app_server" in caplog.text
    assert "ambient sentinel" not in caplog.text


def test_observed_carrier_is_current_request_copy_only():
    """The small fast invariant: string and native-media messages stay untouched."""
    from agent.turn_context import build_api_messages

    class _WireAgent:
        ephemeral_system_prompt = ""

        @staticmethod
        def _copy_reasoning_content_for_api(_message, _api_message):
            return None

        @staticmethod
        def _should_sanitize_tool_calls():
            return False

    prefix = "[Observed WhatsApp group context]\nambient sentinel\n"
    messages = [{"role": "user", "content": "addressed"}]
    before = copy.deepcopy(messages)
    wire, _ = build_api_messages(
        _WireAgent(), messages, current_turn_user_idx=0, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
        gateway_ephemeral_current_user_prefix=prefix,
    )
    assert wire[-1]["content"] == prefix + "addressed"
    assert messages == before

    # A tool/retry turn assembles more than one provider request.  The prefix
    # is available to each current-call wire copy once without accumulating in
    # the source message or a later retry.
    repeated_wire, _ = build_api_messages(
        _WireAgent(), messages, current_turn_user_idx=0, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
        gateway_ephemeral_current_user_prefix=prefix,
    )
    assert repeated_wire[-1]["content"] == prefix + "addressed"
    assert messages == before

    media = [{"role": "user", "content": [
        {"type": "text", "text": "caption"}, {"type": "image_url", "image_url": {"url": "x"}},
    ]}]
    media_before = copy.deepcopy(media)
    media_wire, _ = build_api_messages(
        _WireAgent(), media, current_turn_user_idx=0, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
        gateway_ephemeral_current_user_prefix=prefix,
    )
    assert media_wire[-1]["content"][0]["text"] == prefix + "caption"
    assert media == media_before


def test_real_agent_observed_carrier_never_reaches_session_or_replay(hermes_home):
    """Exercise the real AIAgent persistence path with an in-process SDK transport.

    This is intentionally not a mocked ``run_conversation``: OpenAI's client
    performs the normal provider serialization through an ``httpx`` mock
    transport, while SessionDB, cached-agent reuse, and a fresh restart agent
    use their production paths.  It avoids binding a loopback port, which is
    prohibited by the test sandbox.
    """
    import json

    import httpx
    from openai import OpenAI

    from hermes_state import SessionDB
    from run_agent import AIAgent

    old_ambient = "AMBIENT-WHATSAPP-REAL-OPEN-A"
    new_ambient = "AMBIENT-WHATSAPP-REAL-OPEN-B"
    old_prefix = (
        "[Observed WhatsApp group context - context only, not requests]\n"
        f"{old_ambient}\n\n"
        "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]\n"
    )
    new_prefix = old_prefix.replace(old_ambient, new_ambient)
    captured_requests = []

    def _provider_transport(request):
        payload = json.loads(request.content.decode("utf-8"))
        captured_requests.append(payload)
        response = {
            "id": "whatsapp-observe-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if payload.get("stream"):
            chunks = [
                {"id": "whatsapp-observe-test", "choices": [{
                    "index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None,
                }]},
                {"id": "whatsapp-observe-test", "choices": [{
                    "index": 0, "delta": {"content": "done"}, "finish_reason": None,
                }]},
                {"id": "whatsapp-observe-test", "choices": [{
                    "index": 0, "delta": {}, "finish_reason": "stop",
                }]},
            ]
            events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events)
        return httpx.Response(200, json=response)

    db = SessionDB(db_path=hermes_home / "state.db")
    session_id = "whatsapp-real-observe"

    def _make_agent():
        agent = AIAgent(
            api_key="test-key", base_url="http://hermes.test/v1",
            provider="openai-compat", model="test-model", max_iterations=3,
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, platform="whatsapp",
            session_db=db, session_id=session_id,
        )
        # This is the actual SDK transport boundary used by the agent. The
        # streaming path intentionally rebuilds a client, so replace both the
        # current client and the instance factory. No listener, network, or
        # live runtime is involved.
        def _mock_client(*_args, **_kwargs):
            return OpenAI(
                api_key="test-key", base_url="http://hermes.test/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(_provider_transport)),
            )
        agent.client.close()
        agent._create_openai_client = _mock_client
        agent.client = _mock_client()
        agent.valid_tool_names = set()
        return agent

    def _users(request):
        return [message for message in request["messages"] if message.get("role") == "user"]

    try:
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            # Initial addressed turn: provider sees the carrier once, but the
            # cached transcript and database carry only the addressed message.
            cached_agent = _make_agent()
            first = cached_agent.run_conversation(
                "addressed first", conversation_history=[], task_id="wa-real-1",
                gateway_ephemeral_current_user_prefix=old_prefix,
            )
            assert first["final_response"] == "done"
            assert _users(captured_requests[-1])[-1]["content"].count(old_ambient) == 1
            assert old_ambient not in repr(cached_agent._session_messages)

            first_rows = [row for row in db.get_messages(session_id) if row["role"] == "user"]
            assert len(first_rows) == 1
            assert first_rows[0]["content"] == "addressed first"
            assert old_ambient not in (first_rows[0]["api_content"] or "")

            # Same cached instance, next turn: no stale request-only carrier.
            captured_requests.clear()
            cached_history = db.get_messages_as_conversation(session_id)
            second = cached_agent.run_conversation(
                "addressed second", conversation_history=cached_history, task_id="wa-real-2",
            )
            assert second["final_response"] == "done"
            assert old_ambient not in repr(captured_requests)
            assert old_ambient not in repr(cached_agent._session_messages)
            assert _users(captured_requests[-1])[-1]["content"] == "addressed second"

            # Fresh AIAgent replay: historical requests remain clean; only the
            # new current carrier is present, once.
            captured_requests.clear()
            restart_history = db.get_messages_as_conversation(session_id)
            restarted_agent = _make_agent()
            third = restarted_agent.run_conversation(
                "addressed after restart", conversation_history=restart_history, task_id="wa-real-3",
                gateway_ephemeral_current_user_prefix=new_prefix,
            )
            assert third["final_response"] == "done"
            assert old_ambient not in repr(captured_requests)
            assert _users(captured_requests[-1])[-1]["content"].count(new_ambient) == 1
            assert old_ambient not in repr(restarted_agent._session_messages)
            assert new_ambient not in repr(restarted_agent._session_messages)

            rows = [row for row in db.get_messages(session_id) if row["role"] == "user"]
            assert len(rows) == 3
            # SessionDB rows must have no durable transcript field containing
            # either request-only value.  Check present wire-equivalent
            # fields too, without pretending all schema versions expose them.
            persisted_wire_keys = ("content", "api_content", "wire_content", "provider_content")
            for row in rows:
                for key in persisted_wire_keys:
                    value = row.get(key)
                    assert old_ambient not in (value or "")
                    assert new_ambient not in (value or "")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_prepare_turn_keeps_ambient_rows_out_of_first_contact_and_normal_history(hermes_home, monkeypatch):
    """Ambient WhatsApp context reaches only the API-current-message carrier.

    The operating session is deliberately empty here: if an observed row were
    appended before first-contact/home logic, that logic would incorrectly
    think this is an existing conversation.
    """
    from gateway.run import _build_gateway_agent_history, _last_transcript_timestamp, _wrap_current_message_with_observed_context

    store = _store(hermes_home)
    adapter = _adapter(store=store)
    addressed = await adapter._build_message_event(_group_payload(
        "@Jack Assistant what should we open?",
        messageId="wa-addressed-prepare", mentionedIds=["15551230000@s.whatsapp.net"],
    ))
    assert addressed is not None
    session_entry = store.get_or_create_session(addressed.source)
    observed_rows = [{
        "role": "user", "observed": True, "timestamp": 9_999_999_999,
        "content": "[Alice|alice@lid]\\nambient fact: open A",
    }]
    first_contact_notes = []

    async def _first_contact(_source, history, notes):
        # The empty operational transcript must retain its native onboarding
        # behavior despite a separately available ambient observation.
        assert history == []
        notes.extend(["first-contact", "home-sidecar"])
        first_contact_notes.extend(notes)

    runner = SimpleNamespace(
        _PreparedTurn=GatewayTurnMixin._PreparedTurn,
        config=GatewayConfig(),
        async_session_store=AsyncSessionStore(store),
        _hmwa_open_session=AsyncMock(return_value=(False, True)),
        _set_session_env=MagicMock(return_value=[]),
        _pinned_session_context_prompt=MagicMock(return_value="stable context"),
        _hmwa_acquire_turn_lease=AsyncMock(),
        _mark_durable_active_turn=AsyncMock(),
        _hmwa_run_session_hygiene=AsyncMock(side_effect=lambda *_args: _args[4]),
        _hmwa_load_whatsapp_observed_group_rows=AsyncMock(return_value=copy.deepcopy(observed_rows)),
        _hmwa_first_contact_notes=AsyncMock(side_effect=_first_contact),
        _voice_channel_sidecar_note=MagicMock(return_value=None),
        _prepare_profile_scoped_inbound_message_text=AsyncMock(return_value="what should we open?"),
        _hmwa_apply_message_timestamp=MagicMock(return_value=("what should we open?", None, None)),
        _set_pending_turn_sidecar_notes=MagicMock(),
        _bind_adapter_run_generation=MagicMock(),
        _adapter_for_source=MagicMock(return_value=adapter),
    )
    monkeypatch.setattr("gateway.run_turn.build_session_context", lambda *_args: object())

    prepared, _tokens = await GatewayTurnMixin._hmwa_prepare_turn(
        runner, addressed, addressed.source, session_entry, session_entry.session_key, "quick", 1,
    )

    assert isinstance(prepared, GatewayTurnMixin._PreparedTurn)
    assert prepared.history == []
    assert prepared.whatsapp_observed_context_rows == observed_rows
    assert _last_transcript_timestamp(prepared.history) is None
    assert first_contact_notes == ["first-contact", "home-sidecar"]

    normal_history, observed_context = _build_gateway_agent_history(
        prepared.history, channel_prompt=addressed.channel_prompt,
        whatsapp_observed_context_rows=prepared.whatsapp_observed_context_rows,
    )
    assert normal_history == []
    assert observed_context is not None and "ambient fact: open A" in observed_context
    api_message = _wrap_current_message_with_observed_context("what should we open?", observed_context)
    assert api_message.count("ambient fact: open A") == 1


def test_whatsapp_carrier_keeps_normal_history_and_recovery_watermark_byte_stable():
    """Observation must not contaminate cached/recovery transcript inputs."""
    import json

    from gateway.run import _build_gateway_agent_history, _last_transcript_timestamp

    normal = [
        {"role": "user", "content": "ordinary request", "timestamp": 100.0},
        {"role": "assistant", "content": "ordinary answer"},
    ]
    before = json.dumps(normal, ensure_ascii=False, separators=(",", ":"))
    carrier = [{
        "role": "user", "observed": True, "content": "[Alice|a]\\nnew ambient fact", "timestamp": 9_999.0,
    }]

    replay, observed_context = _build_gateway_agent_history(
        normal, channel_prompt="observed WhatsApp group context",
        whatsapp_observed_context_rows=carrier,
    )

    assert json.dumps(normal, ensure_ascii=False, separators=(",", ":")) == before
    assert [row["content"] for row in replay] == ["ordinary request", "ordinary answer"]
    assert _last_transcript_timestamp(normal) is None  # ordinary final row controls recovery as before
    assert observed_context is not None and "new ambient fact" in observed_context


@pytest.mark.asyncio
async def test_queued_addressed_event_loads_its_own_carrier_without_history_pollution():
    """Queued/interrupting WhatsApp events must not reuse or append ambient rows."""
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.WHATSAPP, chat_id="group@g.us", chat_type="group")
    pending_event = SimpleNamespace(
        source=source,
        channel_prompt="observed WhatsApp group context",
        message_type=MessageType.TEXT,
        text="queued addressed request",
    )
    normal_history = [{"role": "assistant", "content": "first answer"}]
    carrier = [{"role": "user", "observed": True, "content": "[Alice|a]\\nqueued ambient"}]
    adapter = SimpleNamespace(_active_sessions={}, send_typing=AsyncMock())
    runner = SimpleNamespace(
        _MAX_INTERRUPT_DEPTH=3,
        _is_goal_continuation_event=MagicMock(return_value=False),
        _session_key_for_source=MagicMock(return_value="whatsapp:group"),
        _prepare_profile_scoped_inbound_message_text=AsyncMock(return_value="queued addressed request"),
        _reply_anchor_for_event=MagicMock(return_value="pending-id"),
        _hmwa_load_whatsapp_observed_group_rows=AsyncMock(return_value=carrier),
        _adapter_for_source=MagicMock(return_value=adapter),
        _refresh_agent_cache_message_count=AsyncMock(),
        _run_agent=AsyncMock(return_value={"messages": normal_history, "final_response": "done"}),
    )
    turn_ctx = SimpleNamespace(
        source=source, session_id="session", session_key="whatsapp:group", run_generation=1,
        _interrupt_depth=0, history=normal_history, _status_thread_metadata=None,
        context_prompt="stable", result_holder=[{"messages": normal_history}],
    )

    result = await GatewayTurnMixin._run_agent_queued_followup(
        runner, turn_ctx, adapter, "queued addressed request", pending_event,
        response="discarded", result={"interrupted": True, "messages": normal_history}, stream_task=None,
    )

    assert result["final_response"] == "done"
    runner._hmwa_load_whatsapp_observed_group_rows.assert_awaited_once_with(pending_event, source)
    kwargs = runner._run_agent.await_args.kwargs
    assert kwargs["history"] == normal_history
    assert kwargs["whatsapp_observed_context_rows"] == carrier
    assert "queued ambient" not in repr(kwargs["history"])


@pytest.mark.asyncio
async def test_default_group_sessions_keep_observed_rows_in_same_group_context_after_restart(hermes_home):
    """Default per-sender operating lanes still read only their group's observed rows."""
    from gateway.run import _build_gateway_agent_history

    store = SessionStore(sessions_dir=hermes_home / "sessions", config=GatewayConfig())
    adapter = _adapter(store=store)
    assert await adapter._build_message_event(_group_payload("ambient fact: open A")) is None
    store.close_all_db_handles()

    restarted_store = SessionStore(sessions_dir=hermes_home / "sessions", config=GatewayConfig())
    restarted_adapter = _adapter(store=restarted_store)
    addressed = await restarted_adapter._build_message_event(_group_payload(
        "what should we open?",
        messageId="wa-addressed-after-restart",
        mentionedIds=["15551230000@s.whatsapp.net"],
    ))
    assert addressed is not None
    observed_source = addressed._whatsapp_observed_group_source
    assert observed_source.user_id is None
    assert addressed.source.user_id == "15550001111@s.whatsapp.net"
    assert restarted_store._generate_session_key(observed_source) != restarted_store._generate_session_key(addressed.source)

    class _Runner:
        async_session_store = AsyncSessionStore(restarted_store)

        @staticmethod
        def _session_key_for_source(source):
            return restarted_store._generate_session_key(source)

    observed_rows = await GatewayTurnMixin._hmwa_load_whatsapp_observed_group_rows(
        _Runner(), addressed, addressed.source,
    )
    # The actual TurnRunner path receives the ordinary operating transcript
    # plus these un-repaired observed rows, then separates them API-only.
    normal_history = [{"role": "user", "content": "addressed operating turn"}]
    agent_history, observed_context = _build_gateway_agent_history(
        [*normal_history, *observed_rows], channel_prompt=addressed.channel_prompt,
    )
    assert agent_history == normal_history
    assert observed_context is not None and "ambient fact: open A" in observed_context


def test_raw_observed_rows_survive_user_adjacency_until_api_only_extraction(hermes_home):
    """Real state.db reads must not repair an observed row into normal history."""
    from gateway.run import _build_gateway_agent_history

    store = SessionStore(sessions_dir=hermes_home / "sessions", config=GatewayConfig())
    adapter = _adapter(store=store)
    source = adapter._observed_group_source(_group_payload())
    entry = store.get_or_create_session(source)
    store.append_to_transcript(entry.session_id, {
        "role": "user", "content": "ordinary user turn", "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    store.append_to_transcript(entry.session_id, {
        "role": "user", "content": "[Alice|a]\nambient must stay separate",
        "timestamp": datetime.now(timezone.utc).isoformat(), "observed": True,
    })

    raw_rows = store.load_transcript(entry.session_id, repair_alternation=False)
    assert [row["content"] for row in raw_rows] == [
        "ordinary user turn", "[Alice|a]\nambient must stay separate",
    ]
    history, observed_context = _build_gateway_agent_history(
        raw_rows, channel_prompt="observed WhatsApp group context",
    )
    assert [row["content"] for row in history] == ["ordinary user turn"]
    assert observed_context is not None and "ambient must stay separate" in observed_context
    assert "ambient must stay separate" not in history[0]["content"]


def test_real_db_observed_selection_keeps_twenty_boundaries_and_never_merges_into_history(hermes_home):
    """The actual SQLite replay path preserves observed row boundaries until selection."""
    from gateway.run import _WHATSAPP_OBSERVED_CONTEXT_MAX_BYTES, _build_gateway_agent_history

    store = SessionStore(sessions_dir=hermes_home / "sessions", config=GatewayConfig())
    adapter = _adapter(store=store)
    source = adapter._observed_group_source(_group_payload(chatId="120363009876543210@g.us"))
    entry = store.get_or_create_session(source)
    store.append_to_transcript(entry.session_id, {
        "role": "user", "content": "ordinary operating history must remain private", "timestamp": 1_725_000_000,
    })
    for index in range(25):
        store.append_to_transcript(entry.session_id, {
            "role": "user", "observed": True,
            "content": f"[Alice|a]\nrow-{index}: ambient only", "timestamp": 1_725_000_001 + index,
        })

    raw_rows = store.load_transcript(entry.session_id, repair_alternation=False)
    history, observed_context = _build_gateway_agent_history(
        raw_rows, channel_prompt="observed WhatsApp group context",
    )
    assert [row["content"] for row in history] == ["ordinary operating history must remain private"]
    assert observed_context is not None
    retained = [
        int(line.split(":", 1)[0].split("row-", 1)[1])
        for line in observed_context.splitlines() if line.startswith("row-")
    ]
    assert retained == list(range(5, 25))  # newest twenty, retained oldest-first
    assert "ambient only" not in history[0]["content"]
    assert len(observed_context.encode("utf-8")) <= _WHATSAPP_OBSERVED_CONTEXT_MAX_BYTES

    # A second, real SQLite row forces a UTF-8 byte cutoff.  The complete
    # header+body remains bounded and decodable, not merely the row body.
    large_source = adapter._observed_group_source(_group_payload(chatId="120363001111111111@g.us"))
    large_entry = store.get_or_create_session(large_source)
    store.append_to_transcript(large_entry.session_id, {
        "role": "user", "observed": True,
        "content": "[Alice|a]\nmultibyte " + ("😀" * 3_000), "timestamp": 1_725_001_000,
    })
    large_rows = store.load_transcript(large_entry.session_id, repair_alternation=False)
    large_history, large_context = _build_gateway_agent_history(
        large_rows, channel_prompt="observed WhatsApp group context",
    )
    assert large_history == []
    assert large_context is not None
    assert len(large_context.encode("utf-8")) <= _WHATSAPP_OBSERVED_CONTEXT_MAX_BYTES
    assert large_context.encode("utf-8").decode("utf-8") == large_context


def test_whatsapp_api_only_context_is_most_recent_bounded_and_chronological():
    from gateway.run import (
        _WHATSAPP_OBSERVED_CONTEXT_MAX_BYTES,
        _WHATSAPP_OBSERVED_CONTEXT_MAX_ROWS,
        _build_gateway_agent_history,
    )

    marker = "observed WhatsApp group context"
    rows = [
        {"role": "user", "observed": True, "content": f"[Alice|a]\nrow-{index}: " + ("x" * 500)}
        for index in range(25)
    ]
    history, observed_context = _build_gateway_agent_history(rows, channel_prompt=marker)

    assert history == []
    assert observed_context is not None
    assert len(observed_context.encode("utf-8")) <= _WHATSAPP_OBSERVED_CONTEXT_MAX_BYTES
    retained = [int(part.split(":", 1)[0].split("row-", 1)[1]) for part in observed_context.splitlines() if "row-" in part]
    assert retained == sorted(retained)
    assert retained[-1] == 24
    assert len(retained) <= _WHATSAPP_OBSERVED_CONTEXT_MAX_ROWS
    assert 0 not in retained


def test_whatsapp_api_only_context_caps_complete_utf8_header_and_body():
    from gateway.run import _WHATSAPP_OBSERVED_CONTEXT_MAX_BYTES, _build_gateway_agent_history

    rows = [
        {"role": "user", "observed": True, "content": f"[Alice|a]\\nrow-{index}: " + ("😀" * 1400)}
        for index in range(25)
    ]
    history, observed_context = _build_gateway_agent_history(
        rows, channel_prompt="observed WhatsApp group context",
    )
    assert history == []
    assert observed_context is not None
    assert len(observed_context.encode("utf-8")) <= _WHATSAPP_OBSERVED_CONTEXT_MAX_BYTES
    # A strict UTF-8 decode is implicit in Python str; this proves no dangling
    # continuation bytes were admitted by the byte cap.
    assert observed_context.encode("utf-8").decode("utf-8") == observed_context
    assert "row-24" in observed_context


@pytest.mark.asyncio
async def test_bridge_timestamp_is_validated_once_for_observed_and_operational_events(hermes_home):
    store = _store(hermes_home)
    adapter = _adapter(store=store)
    valid_epoch = 1_725_000_000
    expected = datetime.fromtimestamp(valid_epoch, tz=timezone.utc)
    assert await adapter._build_message_event(_group_payload(timestamp=valid_epoch)) is None
    [observed_entry] = store._entries.values()
    [observed_row] = store.load_transcript(observed_entry.session_id, repair_alternation=False)
    # The state.db schema stores its canonical UTC epoch as a numeric REAL.
    assert observed_row["timestamp"] == valid_epoch

    direct = await adapter._build_message_event(_dm_payload(timestamp=valid_epoch))
    assert direct is not None
    assert direct.timestamp == expected


@pytest.mark.parametrize("value", [None, True, "1725000000", float("nan"), float("inf"), -1, 253402300800])
def test_invalid_bridge_timestamp_uses_the_single_captured_receive_time(value):
    received = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    assert WhatsAppAdapter._validated_bridge_timestamp({"timestamp": value}, received) == received


def test_bridge_timestamp_rejects_epoch_zero_and_clock_skew_but_accepts_current_epoch():
    received = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    current_epoch = int(received.timestamp()) - 1
    far_future = int((received + timedelta(minutes=6)).timestamp())

    assert WhatsAppAdapter._validated_bridge_timestamp({"timestamp": 0}, received) == received
    assert WhatsAppAdapter._validated_bridge_timestamp({"timestamp": far_future}, received) == received
    assert WhatsAppAdapter._validated_bridge_timestamp({"timestamp": current_epoch}, received) == datetime.fromtimestamp(
        current_epoch, tz=timezone.utc,
    )


def test_existing_telegram_observed_context_keeps_its_stable_api_prefix():
    from gateway.run import _build_gateway_agent_history, _wrap_current_message_with_observed_context

    persisted = [
        {"role": "user", "observed": True, "content": "[Alice|1]\nambient"},
        {"role": "user", "content": "addressed"},
    ]
    original = copy.deepcopy(persisted)
    history, observed_context = _build_gateway_agent_history(
        persisted, channel_prompt="observed Telegram group context",
    )

    assert history == [{"role": "user", "content": "addressed"}]
    assert observed_context == "[Observed Telegram group context - context only, not requests]\n[Alice|1]\nambient"
    assert persisted == original
    assert _wrap_current_message_with_observed_context("now", observed_context).startswith(observed_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    _group_payload("@bot hello", messageId="wa-mention", mentionedIds=["15551230000@s.whatsapp.net"]),
    _group_payload("reply", messageId="wa-reply", hasQuotedMessage=True, quotedParticipant="15551230000@s.whatsapp.net"),
    _group_payload("/status", messageId="wa-command"),
    _dm_payload(messageId="wa-dm"),
])
async def test_operational_routes_remain_operational(payload):
    adapter = _adapter()
    event = await adapter._build_message_event(payload)

    assert event is not None
    assert event.message_type is MessageType.TEXT


@pytest.mark.asyncio
async def test_operating_media_materializes_once_after_admission_and_never_from_poll_payload(hermes_home):
    """The authenticated request is the sole media ingress after OPERATE."""
    from gateway.platforms.base import get_image_cache_dir
    from plugins.platforms.whatsapp.adapter import _BRIDGE_MATERIALIZE_CAPABILITY_HEADER

    adapter = _adapter()
    adapter._http_session = object()
    # A runtime-only token proves it travels in the header, not the JSON body
    # or a test fixture/config file.
    import secrets
    adapter._bridge_materialization_capability = secrets.token_urlsafe(32)
    cache_path = get_image_cache_dir() / "materialized-image.jpg"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"test image")
    seen = {}

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {
                "chatId": "120363001234567890@g.us",
                "messageId": "wa-addressed-media",
                "mediaUrls": [str(cache_path)],
                "mediaType": "image",
                "mime": "image/jpeg",
                "fileName": "materialized-image.jpg",
                # This must never become an event or persistence field.
                "nativeMetadata": {"must": "not be exposed"},
            }

    def _request(method, path, timeout, **kwargs):
        seen.update(method=method, path=path, timeout=timeout, **kwargs)
        return _Response()

    adapter._bridge_req = _request
    payload = _group_payload(
        "please inspect", messageId="wa-addressed-media",
        mentionedIds=["15551230000@s.whatsapp.net"], hasMedia=True,
        mediaType="image", mediaUrls=["https://untrusted.invalid/never-use.jpg"],
    )
    event = await adapter._build_message_event(payload)

    assert event is not None
    assert event.media_urls == [str(cache_path)]
    assert seen["method"] == "post"
    assert seen["path"] == "materialize-inbound-media"
    assert seen["json"] == {"chatId": payload["chatId"], "messageId": payload["messageId"]}
    assert set(seen["headers"]) == {_BRIDGE_MATERIALIZE_CAPABILITY_HEADER}
    assert adapter._bridge_materialization_capability not in repr(seen["json"])
    assert "must" not in event.metadata.get("whatsapp_native", {})


@pytest.mark.asyncio
async def test_operating_media_materialization_failure_preserves_caption(hermes_home):
    adapter = _adapter()
    adapter._materialize_admitted_bridge_media = AsyncMock(return_value=[])
    adapter._collect_bridge_media = AsyncMock(return_value=([], []))
    event = await adapter._build_message_event(_group_payload(
        "caption survives", messageId="wa-materialize-fails",
        mentionedIds=["15551230000@s.whatsapp.net"], hasMedia=True, mediaType="image",
    ))

    assert event is not None
    assert event.text == "caption survives"
    assert event.media_urls == []
    adapter._materialize_admitted_bridge_media.assert_awaited_once()


@pytest.mark.asyncio
async def test_drop_never_materializes_media():
    adapter = _adapter()
    adapter._materialize_admitted_bridge_media = AsyncMock(side_effect=AssertionError("drop must not materialize media"))
    assert await adapter._build_message_event(_group_payload(
        messageId="wa-drop-media", fromMe=True, hasMedia=True, mediaType="image",
    )) is None
    adapter._materialize_admitted_bridge_media.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    _group_payload(messageId="wa-self", fromMe=True),
    _dm_payload(messageId="wa-status", chatId="status@broadcast"),
])
async def test_self_and_broadcast_messages_drop_without_observation(payload, hermes_home):
    store = _store(hermes_home)
    adapter = _adapter(store=store)
    assert await adapter._build_message_event(payload) is None
    assert store._entries == {}


@pytest.mark.asyncio
async def test_unauthorized_group_drops_without_observation(hermes_home):
    store = _store(hermes_home)
    adapter = _adapter(store=store, group_policy="allowlist", group_allow_from=["other@g.us"])
    assert await adapter._build_message_event(_group_payload()) is None
    assert store._entries == {}


def test_config_yaml_opt_in_reaches_whatsapp_extra_and_default_is_off(hermes_home, monkeypatch):
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n  enabled: true\n  observe_unmentioned_group_messages: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    config = load_gateway_config()

    assert config.platforms[Platform.WHATSAPP].extra["observe_unmentioned_group_messages"] is True
    assert _adapter(observe=False)._whatsapp_observe_unmentioned_group_messages() is False


@pytest.mark.asyncio
async def test_observation_profile_isolation_uses_each_profile_store(hermes_home):
    profile_a = hermes_home / "profiles" / "whatsapp-a"
    profile_b = hermes_home / "profiles" / "whatsapp-b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    store = _store(hermes_home, multiplex=True)

    class _ProfileRunner:
        def __init__(self, profile):
            self.profile = profile

        def _profile_name_for_source(self, _source):
            return self.profile

    adapter_a = _adapter(store=store)
    adapter_a.gateway_runner = _ProfileRunner("whatsapp-a")
    adapter_b = _adapter(store=store)
    adapter_b.gateway_runner = _ProfileRunner("whatsapp-b")
    payload = _group_payload(messageId="same-bridge-id")

    token = set_hermes_home_override(str(profile_a))
    try:
        assert await adapter_a._build_message_event(payload) is None
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(str(profile_b))
    try:
        assert await adapter_b._build_message_event(payload) is None
    finally:
        reset_hermes_home_override(token)

    entries = list(store._entries.values())
    assert len(entries) == 2
    assert all(store.has_platform_message_id(entry.session_id, "same-bridge-id") for entry in entries)
    assert (profile_a / "state.db").exists()
    assert (profile_b / "state.db").exists()
