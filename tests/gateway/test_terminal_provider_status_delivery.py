"""Provider terminal status callbacks must not duplicate human-chat finals.

The conversation loop reports a terminal provider failure through the agent's
status callback *and* returns a user-facing ``final_response``.  The first is
diagnostic progress; the second is the durable answer delivered by
``BasePlatformAdapter`` and its delivery ledger.  Human chat must receive only
the latter.  Programmatic/local surfaces keep both diagnostics and finals.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import TurnRunner, _prepare_gateway_status_message, _sanitize_gateway_final_response
from gateway.session import SessionSource
from run_agent import AIAgent


_LONG_PROVIDER_DETAIL = "provider-controlled refusal detail " * 24

# These final strings are the exact terminal envelopes produced by
# agent.conversation_loop, not hand-written generic error fragments.  The two
# content-policy envelopes deliberately exceed the generic provider-error
# length threshold: human chat still needs the fixed safe category, while
# local/API retain the diagnostic body unchanged.
TERMINAL_PROVIDER_CASES = (
    (
        "invalid_response",
        "❌ Max retries (3) exceeded for invalid responses. Giving up.",
        "Invalid API response after 3 retries: slow response (61s) — likely upstream timeout",
        "failed after retries",
    ),
    (
        "http_200_content_filter",
        "⚠️ The model declined to respond to this request (safety refusal).",
        "⚠️  The model declined to respond to this request "
        "(safety refusal — not a Hermes/gateway failure).\n\n"
        f"Model's explanation: {_LONG_PROVIDER_DETAIL}\n\n"
        "Try rephrasing the request, narrowing the context, or "
        "adding a fallback provider with `hermes fallback add`.",
        "provider rejected",
    ),
    (
        "nonretryable_content_policy",
        "❌ Provider safety filter blocked this request: provider detail",
        "⚠️  The model provider's safety filter blocked this request "
        "(not a Hermes/gateway failure).\n\n"
        f"Provider message: {_LONG_PROVIDER_DETAIL}\n\n"
        "Try rephrasing the request, narrowing the context, or "
        "adding a fallback provider with `hermes fallback add`.",
        "provider rejected",
    ),
    (
        "nonretryable_auth",
        "❌ Non-retryable error (HTTP 401): Incorrect API key provided: provider detail",
        "HTTP 401: Incorrect API key provided: provider detail",
        "authentication failed",
    ),
    (
        "nonretryable_tls",
        "❌ TLS certificate verification failed: provider detail",
        "SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] "
        "certificate verify failed: unable to get local issuer certificate",
        "failed after retries",
    ),
    (
        "billing",
        "❌ Billing or credits exhausted — provider detail",
        "Billing or credits exhausted: provider detail",
        "billing or usage",
    ),
    (
        "rate_limit",
        "❌ Rate limited after 3 retries — provider detail",
        "API call failed after 3 retries: rate limited after 3 retries: provider detail",
        "rate-limiting",
    ),
    (
        "timeout_transport",
        "❌ API failed after 3 retries — provider detail",
        "API call failed after 3 retries: httpx.ConnectError: connection reset",
        "not responding",
    ),
    (
        "max_retry_generic",
        "❌ API failed after 3 retries — provider detail",
        "API call failed after 3 retries: HTTP 500: provider detail",
        "failed after retries",
    ),
)


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")


class _WhatsAppAdapter(BasePlatformAdapter):
    """Real Base final-delivery path with an in-memory WhatsApp transport."""

    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, typing_indicator=False), Platform.WHATSAPP
        )
        self.sent = []

    async def connect(self, *, is_reconnect=False):  # pragma: no cover - offline test
        return True

    async def disconnect(self):  # pragma: no cover - offline test
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - offline test
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="final-1")


def _terminal_status_agent(adapter, monkeypatch):
    """Wire the real status callback to a fake scheduler without transport."""
    ctx = SimpleNamespace(
        _status_adapter=adapter,
        _run_still_current=lambda: True,
        source=SessionSource(platform=Platform.WHATSAPP, chat_id="test-chat"),
        _status_chat_id="test-chat",
        _status_thread_metadata=None,
        _loop_for_step=object(),
        _cleanup_progress=False,
    )
    turn = object.__new__(TurnRunner)
    turn._ctx = ctx
    scheduled = []

    def _capture_schedule(coro, *_args, **_kwargs):
        scheduled.append(coro)
        coro.close()
        return None

    monkeypatch.setattr("gateway.run.safe_schedule_threadsafe", _capture_schedule)
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent._vprint = lambda *_args, **_kwargs: None
    agent.status_callback = turn._status_callback_sync
    return agent, scheduled


def _event():
    return MessageEvent(
        text="please retry",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.WHATSAPP, chat_id="test-chat", chat_type="group"
        ),
        message_id="incoming-1",
    )


def _ledger_rows():
    with dl._connect() as conn:
        return conn.execute("SELECT state, content FROM delivery_obligations").fetchall()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("_kind", "terminal_status", "raw_final", "expected_fragment"),
    TERMINAL_PROVIDER_CASES,
)
async def test_terminal_provider_status_is_silent_while_base_delivers_one_final(
    _kind, terminal_status, raw_final, expected_fragment, monkeypatch
):
    """One terminal provider failure yields zero status sends and one final.

    This is the production shape: the agent emits a terminal diagnostic before
    the existing Base adapter final-delivery rail processes the returned final
    response and records its one delivery obligation.
    """
    adapter = _WhatsAppAdapter()
    agent, scheduled = _terminal_status_agent(adapter, monkeypatch)

    agent._emit_status(terminal_status, terminal_provider=True)

    assert scheduled == [], "terminal provider diagnostics must not send on WhatsApp"

    expected_final = _sanitize_gateway_final_response(Platform.WHATSAPP, raw_final)
    assert expected_fragment in expected_final.lower()
    assert expected_final != raw_final
    assert "provider-controlled refusal detail" not in expected_final
    if "content_filter" in _kind or "content_policy" in _kind:
        assert len(raw_final) > 400, "must exercise the long-envelope bypass"
    adapter._message_handler = AsyncMock(return_value=expected_final)
    session_key = "agent:main:whatsapp:group:test-chat"
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter._process_message_background(_event(), session_key)

    assert adapter.sent == [expected_final]
    assert _ledger_rows() == [("delivered", expected_final)]


@pytest.mark.parametrize(
    ("_kind", "terminal_status", "raw_final", "_expected_fragment"),
    TERMINAL_PROVIDER_CASES,
)
def test_terminal_provider_diagnostics_stay_available_on_raw_surfaces(
    _kind, terminal_status, raw_final, _expected_fragment
):
    """The dedicated event suppresses chat only; local/API remain diagnostic."""
    events = []
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent._vprint = lambda *_args, **_kwargs: None
    agent.status_callback = lambda kind, message: events.append((kind, message))

    agent._emit_status(terminal_status, terminal_provider=True)

    assert events == [("terminal_provider", terminal_status)]
    for platform in ("local", "api_server", "webhook"):
        assert (
            _prepare_gateway_status_message(platform, events[0][0], terminal_status)
            == terminal_status
        )
        assert _sanitize_gateway_final_response(platform, raw_final) == raw_final


def test_normal_short_prose_about_a_model_safety_filter_is_not_rewritten():
    """Only fixed terminal envelopes get the long-body bypass."""
    answer = "The model provider's safety filter blocked my draft yesterday."

    assert _sanitize_gateway_final_response(Platform.WHATSAPP, answer) == answer


def test_ordinary_lifecycle_status_keeps_its_existing_callback_kind():
    """The distinction is opt-in at terminal provider sites, not a global mute."""
    events = []
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent._vprint = lambda *_args, **_kwargs: None
    agent.status_callback = lambda kind, message: events.append((kind, message))

    agent._emit_status("ordinary lifecycle status")

    assert events == [("lifecycle", "ordinary lifecycle status")]
