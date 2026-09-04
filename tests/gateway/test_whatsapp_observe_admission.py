"""WhatsApp's private drop|observe|operate admission boundary.

Observation is durable only after the Python adapter receives a bridge event.
The legacy bridge ``/messages`` queue remains destructive if adapter persistence
fails, so this is intentionally not a receipt spool or delivery guarantee.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageType
from gateway.session import SessionStore
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
