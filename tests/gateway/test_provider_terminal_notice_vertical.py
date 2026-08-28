"""End-to-end gateway contract for typed provider terminal failures."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from agent.runtime_notices import provider_terminal_notice
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.runtime_notice_delivery import canonical_human_terminal_content
from gateway.session import SessionEntry, SessionSource, SessionStore


SESSION_KEY = "agent:main:whatsapp:group:chat-1:user-1"


class TransportSpy:
    def __init__(self) -> None:
        self.calls = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="terminal-1")

    def calls_with_content(self, content: str):
        return [call for call in self.calls if call[1] == content]


class DefiniteNoDeliverySpy(TransportSpy):
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append((chat_id, content, reply_to, metadata))
        return SendResult(success=False, error="definite pre-delivery rejection")

    def runtime_notice_definitely_not_delivered(self, _result):
        return True


def _source(platform: Platform = Platform.WHATSAPP) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_type="group",
        user_id="user-1",
    )


def _event(
    platform: Platform = Platform.WHATSAPP,
    *,
    message_id: str | None = "inbound-1",
    timestamp: datetime | None = None,
) -> MessageEvent:
    kwargs = {"timestamp": timestamp} if timestamp is not None else {}
    return MessageEvent(
        text="please continue",
        source=_source(platform),
        message_id=message_id,
        **kwargs,
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


def _tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _real_retry_exhaustion_result(platform: Platform, status_callback):
    """Drive the real classifier/retry producer with a mocked provider."""

    from run_agent import AIAgent

    class ProviderUnavailable(Exception):
        status_code = 503

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_definitions()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            provider="custom",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform=platform.value,
            status_callback=status_callback,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = ProviderUnavailable(
        "private upstream failure"
    )
    agent._try_recover_primary_transport = MagicMock(return_value=False)
    agent._has_pending_fallback = MagicMock(return_value=False)
    agent._persist_session = MagicMock()
    agent._dump_api_request_debug = MagicMock()
    with patch("agent.conversation_loop.jittered_backoff", return_value=0):
        result = agent.run_conversation(
            "please continue", conversation_history=[], task_id="producer-turn"
        )
    assert agent.client.chat.completions.create.call_count == 3
    return result


def _real_zai_long_backoff_result(platform: Platform, status_callback):
    """Drive the real long-wait Z.AI status producer without sleeping."""

    from run_agent import AIAgent

    class ProviderOverloaded(Exception):
        status_code = 503

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_definitions()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.z.ai/api/coding/paas/v4",
            provider="zai-coding-plan",
            model="glm-5.2",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform=platform.value,
            status_callback=status_callback,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = ProviderOverloaded(
        "private overload body"
    )
    agent._try_recover_primary_transport = MagicMock(return_value=False)
    agent._has_pending_fallback = MagicMock(return_value=False)
    agent._persist_session = MagicMock()
    agent._dump_api_request_debug = MagicMock()
    with (
        patch("agent.conversation_loop.is_zai_coding_overload_error", return_value=True),
        patch("agent.conversation_loop.zai_coding_overload_retry_ceiling", return_value=3),
        patch(
            "agent.conversation_loop.adaptive_rate_limit_backoff",
            return_value=(0, "zai_coding_overload_long"),
        ),
    ):
        result = agent.run_conversation(
            "please continue", conversation_history=[], task_id="zai-long-turn"
        )
    assert agent.client.chat.completions.create.call_count == 3
    return result


def _turn_runner_status_callback(runner, source, adapter):
    context = SimpleNamespace(
        _status_adapter=adapter,
        _run_still_current=lambda: True,
        source=source,
        _cleanup_progress=False,
        _loop_for_step=asyncio.get_running_loop(),
        _status_chat_id=source.chat_id,
        _status_thread_metadata=None,
    )
    return gateway_run.TurnRunner(runner, context)._status_callback_sync


def _runner(monkeypatch, profile_home, adapter, platform=Platform.WHATSAPP):
    profile_home.mkdir(parents=True, exist_ok=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {platform: adapter} if adapter is not None else {}
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
async def test_two_runners_recover_no_message_id_turn_and_dedupe(
    monkeypatch, tmp_path
) -> None:
    home = tmp_path / "profile"
    sessions_dir = home / "sessions"
    first_adapter = TransportSpy()
    first = _runner(monkeypatch, home, first_adapter)
    first_store = SessionStore(sessions_dir, first.config)
    first_store._db = None
    first_store.get_or_create_session(_source())
    first.session_store = first_store
    first_event = _event(message_id=None)
    assert await first._handle_message_with_agent(
        first_event, first_event.source, SESSION_KEY, 1
    ) is None
    assert first_store.recover_interrupted_turns() == 1

    # A fresh runner receives the scheduler's synthetic replay for the same
    # no-message-id turn. Its default adapter-arrival timestamp is different;
    # the persisted session recovery id, not timestamp equality, suppresses a
    # second terminal send.
    second_adapter = TransportSpy()
    second = _runner(monkeypatch, home, second_adapter)
    second_store = SessionStore(sessions_dir, second.config)
    second_store._db = None
    second.session_store = second_store
    second._run_startup_resume_event = AsyncMock()
    with patch("gateway.restart_loop_guard.check_and_record", return_value=False):
        assert second._schedule_resume_pending_sessions() == 1
    await asyncio.sleep(0)
    reconstructed = second._run_startup_resume_event.await_args.args[1]
    assert reconstructed.internal is True
    assert reconstructed.text == ""
    assert reconstructed.metadata["gateway_resume_pending_replay"] is True
    second._release_running_agent_state(SESSION_KEY)
    assert reconstructed.timestamp != first_event.timestamp
    assert await second._handle_message_with_agent(
        reconstructed, reconstructed.source, SESSION_KEY, 1
    ) is None

    expected = canonical_human_terminal_content(_agent_result()["runtime_notice"])
    assert len(first_adapter.calls_with_content(expected)) == 1
    assert second_adapter.calls_with_content(expected) == []


@pytest.mark.asyncio
async def test_unavailable_resume_then_real_no_id_message_gets_fresh_terminal_run(
    monkeypatch, tmp_path
) -> None:
    home = tmp_path / "profile"
    sessions_dir = home / "sessions"
    first_adapter = TransportSpy()
    first = _runner(monkeypatch, home, first_adapter)
    first_store = SessionStore(sessions_dir, first.config)
    first_store._db = None
    first_store.get_or_create_session(_source())
    first.session_store = first_store
    interrupted = _event(message_id=None)
    assert await first._handle_message_with_agent(
        interrupted, interrupted.source, SESSION_KEY, 1
    ) is None
    interrupted_run_id = interrupted._runtime_notice_run_id
    assert first_store.recover_interrupted_turns() == 1

    # Startup cannot schedule the synthetic replay while the adapter is down.
    # A later real human event is a new logical turn even though the session is
    # still resume-pending, so it must not inherit the interrupted tombstone.
    resumed = _runner(monkeypatch, home, None)
    resumed_store = SessionStore(sessions_dir, resumed.config)
    resumed_store._db = None
    resumed.session_store = resumed_store
    with patch("gateway.restart_loop_guard.check_and_record", return_value=False):
        assert resumed._schedule_resume_pending_sessions() == 0

    resumed_adapter = TransportSpy()
    resumed.adapters = {Platform.WHATSAPP: resumed_adapter}
    real_event = _event(message_id=None)
    assert await resumed._handle_message_with_agent(
        real_event, real_event.source, SESSION_KEY, 1
    ) is None

    expected = canonical_human_terminal_content(_agent_result()["runtime_notice"])
    assert real_event._runtime_notice_run_id != interrupted_run_id
    assert len(resumed_adapter.calls_with_content(expected)) == 1


@pytest.mark.asyncio
async def test_gateway_missing_adapter_returns_canonical_existing_final_rail(
    monkeypatch, tmp_path
) -> None:
    home = tmp_path / "profile"
    runner = _runner(monkeypatch, home, None)

    response = await runner._handle_message_with_agent(
        _event(), _source(), SESSION_KEY, 1
    )

    assert response == canonical_human_terminal_content(
        _agent_result()["runtime_notice"]
    )
    assert not (home / "state.db").exists()


@pytest.mark.asyncio
async def test_gateway_definite_no_delivery_returns_canonical_existing_final_rail(
    monkeypatch, tmp_path
) -> None:
    adapter = DefiniteNoDeliverySpy()
    runner = _runner(monkeypatch, tmp_path / "profile", adapter)

    response = await runner._handle_message_with_agent(
        _event(), _source(), SESSION_KEY, 1
    )

    expected = canonical_human_terminal_content(_agent_result()["runtime_notice"])
    assert response == expected
    assert len(adapter.calls_with_content(expected)) == 1


@pytest.mark.asyncio
async def test_gateway_capacity_degradation_returns_canonical_existing_final_rail(
    monkeypatch, tmp_path
) -> None:
    adapter = TransportSpy()
    runner = _runner(monkeypatch, tmp_path / "profile", adapter)
    monkeypatch.setattr("gateway.runtime_notice_ledger._MAX_ROWS", 0)

    response = await runner._handle_message_with_agent(
        _event(), _source(), SESSION_KEY, 1
    )

    expected = canonical_human_terminal_content(_agent_result()["runtime_notice"])
    assert response == expected
    assert adapter.calls_with_content(expected) == []


@pytest.mark.asyncio
async def test_real_retry_exhaustion_producer_has_zero_human_status_bubbles(
    monkeypatch, tmp_path
) -> None:
    adapter = TransportSpy()
    source = _source()
    runner = _runner(monkeypatch, tmp_path / "profile", adapter)
    callback = _turn_runner_status_callback(runner, source, adapter)

    async def run_real_producer(*_args, **_kwargs):
        return await asyncio.to_thread(
            _real_retry_exhaustion_result, Platform.WHATSAPP, callback
        )

    runner._run_agent = AsyncMock(side_effect=run_real_producer)
    response = await runner._handle_message_with_agent(
        _event(), source, SESSION_KEY, 1
    )
    await asyncio.sleep(0)

    assert response is None
    terminal_calls = [
        call for call in adapter.calls if call[1].startswith("⚠️ The model provider")
    ]
    assert len(terminal_calls) == 1
    assert not any(
        "Retrying" in call[1] or "API call failed (attempt" in call[1]
        for call in adapter.calls
    )


@pytest.mark.asyncio
async def test_real_retry_exhaustion_raw_surface_keeps_status_and_result(
    monkeypatch, tmp_path
) -> None:
    adapter = TransportSpy()
    source = _source(Platform.API_SERVER)
    runner = _runner(
        monkeypatch, tmp_path / "profile", adapter, platform=Platform.API_SERVER
    )
    callback = _turn_runner_status_callback(runner, source, adapter)

    async def run_real_producer(*_args, **_kwargs):
        return await asyncio.to_thread(
            _real_retry_exhaustion_result, Platform.API_SERVER, callback
        )

    runner._run_agent = AsyncMock(side_effect=run_real_producer)
    response = await runner._handle_message_with_agent(
        _event(Platform.API_SERVER), source, SESSION_KEY, 1
    )
    await asyncio.sleep(0.05)

    assert response.startswith("API call failed after 3 retries:")
    assert any("Retrying in 0.0s" in call[1] for call in adapter.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "expect_status"),
    [(Platform.WHATSAPP, False), (Platform.API_SERVER, True)],
)
async def test_zai_long_backoff_status_is_human_silent_but_raw_visible(
    monkeypatch, tmp_path, platform: Platform, expect_status: bool
) -> None:
    adapter = TransportSpy()
    source = _source(platform)
    runner = _runner(monkeypatch, tmp_path / platform.value, adapter, platform)
    callback = _turn_runner_status_callback(runner, source, adapter)

    async def run_real_producer(*_args, **_kwargs):
        return await asyncio.to_thread(
            _real_zai_long_backoff_result, platform, callback
        )

    runner._run_agent = AsyncMock(side_effect=run_real_producer)
    response = await runner._handle_message_with_agent(
        _event(platform), source, SESSION_KEY, 1
    )
    await asyncio.sleep(0.05)

    long_wait_calls = [
        call for call in adapter.calls if "adaptive long backoff" in call[1]
    ]
    assert bool(long_wait_calls) is expect_status
    if platform is Platform.WHATSAPP:
        assert response is None
        assert len(
            [
                call
                for call in adapter.calls
                if call[1].startswith("⚠️ The model provider")
            ]
        ) == 1
    else:
        assert response.startswith("API call failed after 3 retries:")


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


@pytest.mark.asyncio
async def test_multiplex_handle_writes_only_served_profile_ledgers(
    monkeypatch, tmp_path
) -> None:
    primary = tmp_path / "primary"
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    for home in (primary, profile_a, profile_b):
        home.mkdir()
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **_kwargs: [("profile-a", profile_a), ("profile-b", profile_b)],
    )

    primary_adapter = TransportSpy()
    adapter_a = TransportSpy()
    adapter_b = TransportSpy()
    runner = _runner(monkeypatch, primary, primary_adapter)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._profile_adapters = {
        "profile-a": {Platform.WHATSAPP: adapter_a},
        "profile-b": {Platform.WHATSAPP: adapter_b},
    }

    def set_session(name: str) -> str:
        key = f"{SESSION_KEY}:{name}"
        runner.session_store.get_or_create_session.return_value = SessionEntry(
            session_key=key,
            session_id=f"session-{name}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.WHATSAPP,
            chat_type="group",
        )
        return key

    source_a = _source()
    source_a.profile = "profile-a"
    event_a = _event()
    event_a.source = source_a
    key_a = set_session("profile-a")
    assert await asyncio.wait_for(
        runner._handle_message_with_agent(event_a, source_a, key_a, 1),
        timeout=5,
    ) is None
    assert (profile_a / "state.db").exists()
    assert not (profile_b / "state.db").exists()
    assert not (primary / "state.db").exists()
    assert len(adapter_a.calls_with_content(
        canonical_human_terminal_content(_agent_result()["runtime_notice"])
    )) == 1
    assert adapter_b.calls == []

    source_b = _source()
    source_b.profile = "profile-b"
    event_b = _event()
    event_b.source = source_b
    key_b = set_session("profile-b")
    assert await asyncio.wait_for(
        runner._handle_message_with_agent(event_b, source_b, key_b, 1),
        timeout=5,
    ) is None
    assert (profile_b / "state.db").exists()
    assert not (primary / "state.db").exists()
    assert len(adapter_b.calls_with_content(
        canonical_human_terminal_content(_agent_result()["runtime_notice"])
    )) == 1

    before_primary = len(primary_adapter.calls)
    before_a = len(adapter_a.calls)
    before_b = len(adapter_b.calls)
    unknown = _source()
    unknown.profile = "stale-unserved"
    unknown_event = _event()
    unknown_event.source = unknown
    unknown_key = set_session("unknown")
    response = await asyncio.wait_for(
        runner._handle_message_with_agent(
            unknown_event, unknown, unknown_key, 1
        ),
        timeout=5,
    )

    assert response == canonical_human_terminal_content(
        _agent_result()["runtime_notice"]
    )
    assert len(primary_adapter.calls) == before_primary
    assert len(adapter_a.calls) == before_a
    assert len(adapter_b.calls) == before_b
    assert not (primary / "state.db").exists()
