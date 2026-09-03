"""Focused contract tests for WhatsApp lookup acknowledgements."""

import asyncio
import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


ACK_TEXT = "Hmm, let me check."
RAW_LOOKUP_ARGS = {"query": "private query must never be rendered"}


class CaptureAdapter(BasePlatformAdapter):
    """A normal adapter send surface with no progress/status/stream rails."""

    def __init__(self, platform=Platform.WHATSAPP):
        super().__init__(
            PlatformConfig(enabled=True, token="***", typing_indicator=False), platform
        )
        self.sent = []
        self.edits = []
        self.ack_sent = threading.Event()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        if content == ACK_TEXT:
            self.ack_sent.set()
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append((chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _source(platform=Platform.WHATSAPP, chat_type="dm", **kwargs):
    return SessionSource(
        platform=platform, chat_id="15551234567", chat_type=chat_type, **kwargs
    )


def _turn_runner(
    *, adapter, enabled=True, current=True, platform=Platform.WHATSAPP, source=None
):
    gateway_run = importlib.import_module("gateway.run")
    loop = asyncio.get_running_loop()
    ctx = TurnContext(
        source=source or _source(platform),
        _run_still_current=lambda: current,
        inbound_message_id="inbound-message-id",
        lookup_acknowledgement_enabled=enabled,
        _lookup_acknowledgement_loop=loop,
    )
    owner = SimpleNamespace(_adapter_for_source=lambda source: adapter)
    # Exercise the production routing metadata helper without a full gateway.
    owner._is_telegram_dm_topic_target = (
        gateway_run.GatewayRunner._is_telegram_dm_topic_target
    )
    owner._thread_metadata_for_target = (
        gateway_run.GatewayRunner._thread_metadata_for_target.__get__(owner)
    )
    owner._thread_metadata_for_source = (
        gateway_run.GatewayRunner._thread_metadata_for_source.__get__(owner)
    )
    return gateway_run.TurnRunner(owner, ctx), ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["web_search", "web_extract", "browser_navigate"])
async def test_qualifying_lookup_sends_static_interim_reply_once(monkeypatch, tool_name):
    """The first canonical lookup is sent on the captured gateway loop only."""
    gateway_run = importlib.import_module("gateway.run")
    adapter = CaptureAdapter()
    turn_runner, ctx = _turn_runner(adapter=adapter)
    scheduled = []

    def schedule(coro, loop, **kwargs):
        assert ctx._lookup_acknowledgement_fired[0] is True
        scheduled.append((coro, loop, kwargs))
        return object()

    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", schedule)

    turn_runner.combined_tool_start_callback("first", tool_name, RAW_LOOKUP_ARGS)
    turn_runner.combined_tool_start_callback("retry", tool_name, RAW_LOOKUP_ARGS)

    assert len(scheduled) == 1
    coroutine, loop, schedule_kwargs = scheduled[0]
    assert loop is ctx._lookup_acknowledgement_loop
    assert schedule_kwargs["log_message"] == "WhatsApp lookup acknowledgement scheduling error"
    await coroutine

    assert adapter.sent == [
        {
            "chat_id": "15551234567",
            "content": ACK_TEXT,
            "reply_to": "inbound-message-id",
            "metadata": {"_interim_send": True},
        }
    ]
    visible = repr(adapter.sent)
    assert RAW_LOOKUP_ARGS["query"] not in visible
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_acknowledgement_preserves_routed_profile_metadata(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    adapter = CaptureAdapter()
    source = _source(profile="routed-profile", thread_id="routed-thread")
    turn_runner, _ = _turn_runner(adapter=adapter, source=source)
    scheduled = []
    monkeypatch.setattr(
        gateway_run,
        "safe_schedule_threadsafe",
        lambda coro, loop, **kwargs: scheduled.append(coro),
    )

    turn_runner.combined_tool_start_callback("lookup", "web_search", RAW_LOOKUP_ARGS)
    await scheduled[0]

    assert adapter.sent[0]["metadata"] == {
        "thread_id": "routed-thread",
        "hermes_profile": "routed-profile",
        "_interim_send": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name",
    ["terminal", "clarify", "approval", "execute_code", "read_file", "browser", "mcp_browser_navigate", "web_search_custom"],
)
async def test_non_lookup_tools_never_schedule_an_acknowledgement(monkeypatch, tool_name):
    gateway_run = importlib.import_module("gateway.run")
    turn_runner, _ = _turn_runner(adapter=CaptureAdapter())
    calls = []
    monkeypatch.setattr(
        gateway_run, "safe_schedule_threadsafe", lambda *args, **kwargs: calls.append(args)
    )

    turn_runner.combined_tool_start_callback("call", tool_name, RAW_LOOKUP_ARGS)

    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "current", "platform"),
    [
        (False, True, Platform.WHATSAPP),
        (True, False, Platform.WHATSAPP),
        (True, True, Platform.TELEGRAM),
    ],
)
async def test_disabled_stale_or_non_whatsapp_turns_never_schedule(monkeypatch, enabled, current, platform):
    gateway_run = importlib.import_module("gateway.run")
    turn_runner, _ = _turn_runner(
        adapter=CaptureAdapter(platform=platform),
        enabled=enabled,
        current=current,
        platform=platform,
    )
    calls = []
    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", lambda *args, **kwargs: calls.append(args))

    turn_runner.combined_tool_start_callback("call", "web_search", RAW_LOOKUP_ARGS)

    assert calls == []


@pytest.mark.asyncio
async def test_scheduling_failure_keeps_the_latch_closed_for_retries(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")
    turn_runner, ctx = _turn_runner(adapter=CaptureAdapter())
    calls = []

    def fail_schedule(coro, loop, **kwargs):
        calls.append((coro, loop))
        assert ctx._lookup_acknowledgement_fired[0] is True
        raise RuntimeError("gateway loop is closed")

    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", fail_schedule)

    turn_runner.combined_tool_start_callback("first", "web_search", RAW_LOOKUP_ARGS)
    turn_runner.combined_tool_start_callback("retry", "web_search", RAW_LOOKUP_ARGS)

    assert len(calls) == 1
    assert ctx._lookup_acknowledgement_fired == [True]


class _LookupAgent:
    """Agent double that proves the ack precedes final normal delivery."""

    ack_sent: threading.Event | None = None

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        assert self.tool_start_callback is not None
        self.tool_start_callback("lookup-1", "web_search", RAW_LOOKUP_ARGS)
        assert type(self).ack_sent is not None
        assert type(self).ack_sent.wait(timeout=1.0)
        self.tool_start_callback("lookup-retry", "browser_navigate", RAW_LOOKUP_ARGS)
        return {"final_response": "Final answer", "messages": [], "api_calls": 1}


class _NoLookupAckAgent:
    """The opt-out/non-WhatsApp path must not install a start callback."""

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        assert self.tool_start_callback is None
        return {"final_response": "Final answer", "messages": [], "api_calls": 1}


def _gateway_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        multiplex_profiles=False,
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


@pytest.mark.asyncio
async def test_real_config_snapshot_acks_before_one_final_without_other_rails(monkeypatch, tmp_path):
    """A temp HERMES_HOME config drives the real turn and normal final send."""
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "display": {
                    "lookup_acknowledgement": True,
                    "tool_progress": "off",
                    "thinking_progress": False,
                    "interim_assistant_messages": False,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _LookupAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = CaptureAdapter()
    _LookupAgent.ack_sent = adapter.ack_sent
    runner = _gateway_runner(adapter)
    source = _source()
    event = MessageEvent(
        text="look this up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="inbound-message-id",
    )
    session_key = "agent:main:whatsapp:dm:15551234567"

    async def handler(inbound_event):
        result = await runner._run_agent(
            message=inbound_event.text,
            context_prompt="",
            history=[],
            source=inbound_event.source,
            session_id="lookup-session",
            session_key=session_key,
            event_message_id=inbound_event.message_id,
            inbound_message_id=inbound_event.message_id,
        )
        return result["final_response"]

    adapter.set_message_handler(handler)
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)

    assert [call["content"] for call in adapter.sent] == [ACK_TEXT, "Final answer"]
    assert adapter.sent[0]["reply_to"] == "inbound-message-id"
    assert adapter.sent[0]["metadata"] == {"_interim_send": True}
    assert adapter.sent[1]["metadata"].get("_interim_send") is None
    assert adapter.edits == []
    assert RAW_LOOKUP_ARGS["query"] not in repr(adapter.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ack_enabled", "platform"),
    [(False, Platform.WHATSAPP), (True, Platform.TELEGRAM)],
)
async def test_config_opt_out_or_other_platform_never_installs_lookup_callback(
    monkeypatch, tmp_path, ack_enabled, platform
):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"lookup_acknowledgement": ack_enabled}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _NoLookupAckAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = CaptureAdapter(platform=platform)
    runner = _gateway_runner(adapter)
    result = await runner._run_agent(
        message="look this up",
        context_prompt="",
        history=[],
        source=_source(platform),
        session_id="lookup-session",
        session_key=f"agent:main:{platform.value}:dm:15551234567",
        event_message_id="inbound-message-id",
        inbound_message_id="inbound-message-id",
    )

    assert result["final_response"] == "Final answer"
    assert adapter.sent == []
