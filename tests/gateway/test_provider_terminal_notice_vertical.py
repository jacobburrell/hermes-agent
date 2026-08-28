"""End-to-end gateway contract for typed provider terminal failures."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent.runtime_notices import provider_terminal_notice
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.runtime_notice_delivery import canonical_human_terminal_content
from gateway.session import SessionEntry, SessionSource


SESSION_KEY = "agent:main:whatsapp:group:chat-1:user-1"


class TransportSpy:
    def __init__(self) -> None:
        self.calls = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="terminal-1")

    def calls_with_content(self, content: str):
        return [call for call in self.calls if call[1] == content]


def _source(platform: Platform = Platform.WHATSAPP) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_type="group",
        user_id="user-1",
    )


def _event(platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text="please continue",
        source=_source(platform),
        message_id="inbound-1",
    )


def _agent_result():
    raw = "misleading private body: billing auth timeout 429 secret-request"
    notice = provider_terminal_notice(
        reason="rate_limit",
        message=raw,
        retryable=True,
        provider="secret-provider",
        model="secret-model",
        diagnostic="secret-diagnostic",
    )
    return {
        "final_response": raw,
        "messages": [
            {"role": "user", "content": "please continue"},
            {"role": "assistant", "content": raw},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 2,
        "failed": True,
        "failure_reason": "rate_limit",
        "runtime_notice": notice,
    }


def _runner(monkeypatch, profile_home, adapter, platform=Platform.WHATSAPP):
    profile_home.mkdir(parents=True, exist_ok=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {platform: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._run_agent = AsyncMock(return_value=_agent_result())

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY,
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", profile_home)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


@pytest.mark.asyncio
async def test_human_provider_failure_uses_one_sanitized_transport_send(
    monkeypatch, tmp_path
) -> None:
    adapter = TransportSpy()
    runner = _runner(monkeypatch, tmp_path / "profile", adapter)

    response = await runner._handle_message_with_agent(
        _event(), _source(), SESSION_KEY, 1
    )

    expected = canonical_human_terminal_content(_agent_result()["runtime_notice"])
    assert response is None
    terminal_calls = adapter.calls_with_content(expected)
    assert len(terminal_calls) == 1
    assert "secret-request" not in terminal_calls[0][1]
    assert "billing" not in terminal_calls[0][1].lower()
    assert not adapter.calls_with_content(_agent_result()["final_response"])


@pytest.mark.asyncio
async def test_two_runners_share_durable_terminal_tombstone(
    monkeypatch, tmp_path
) -> None:
    home = tmp_path / "profile"
    first_adapter = TransportSpy()
    first = _runner(monkeypatch, home, first_adapter)
    assert await first._handle_message_with_agent(
        _event(), _source(), SESSION_KEY, 1
    ) is None

    second_adapter = TransportSpy()
    second = _runner(monkeypatch, home, second_adapter)
    assert await second._handle_message_with_agent(
        _event(), _source(), SESSION_KEY, 1
    ) is None

    expected = canonical_human_terminal_content(_agent_result()["runtime_notice"])
    assert len(first_adapter.calls_with_content(expected)) == 1
    assert second_adapter.calls_with_content(expected) == []


@pytest.mark.asyncio
async def test_programmatic_api_server_retains_raw_existing_result_rail(
    monkeypatch, tmp_path
) -> None:
    adapter = TransportSpy()
    runner = _runner(
        monkeypatch, tmp_path / "profile", adapter, platform=Platform.API_SERVER
    )

    response = await runner._handle_message_with_agent(
        _event(Platform.API_SERVER),
        _source(Platform.API_SERVER),
        SESSION_KEY,
        1,
    )

    assert response == _agent_result()["final_response"]
    assert not adapter.calls_with_content(response)


def test_routed_profile_home_never_falls_back_to_primary(
    monkeypatch, tmp_path
) -> None:
    served_home = tmp_path / "served"
    primary_home = tmp_path / "primary"
    served_home.mkdir()
    primary_home.mkdir()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    monkeypatch.setattr(gateway_run, "_hermes_home", primary_home)
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **_kwargs: [("served", served_home)],
    )

    served = _source()
    served.profile = "served"
    unknown = _source()
    unknown.profile = "stale-or-unserved"

    assert runner._runtime_notice_profile_home_for_source(served) == served_home
    assert runner._runtime_notice_profile_home_for_source(unknown) is None
