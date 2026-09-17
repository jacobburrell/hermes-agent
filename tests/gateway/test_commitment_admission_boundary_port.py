"""Current split-gateway coverage for commitment admission at final delivery."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.commitment_admission_boundary import _SAFE_REFUSAL, guard_final_response
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from hermes_cli import kanban_db as kb


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _event():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user", profile="jack")
    event = MessageEvent(text="Please continue this and update me later.", source=source, message_id="m-1")
    event._gateway_accepted = True
    return event, source


def _ctx(source):
    return SimpleNamespace(
        source=source, session_id="origin-session",
        user_config={"goals": {"commitment_admission": {"enabled": True, "assignee": "jack"}}},
    )


class _Runner:
    def __init__(self, proposal, ready=True):
        self._proposal = proposal
        self._ready = ready

    def _commitment_proposer(self, request, response, source):
        return self._proposal

    def _owns_kanban_dispatcher_lock(self):
        return self._ready

    def _adapter_for_source(self, source):
        return SimpleNamespace(send=lambda *a, **kw: None)


def _continuing():
    return {"disposition": "continuing", "objective": "Prepare report", "completion_criteria": "Report is saved",
            "next_action": "Review supplied notes"}


def test_enabled_guard_persists_before_releasing_continuing_reply(temp_home):
    event, source = _event()
    response, receipt = guard_final_response(runner=_Runner(_continuing()), ctx=_ctx(source), event=event,
                                             response="I will keep working and update you.")
    assert response == "I will keep working and update you."
    assert receipt and receipt.may_promise_follow_up
    assert event.metadata["commitment_task_id"] == receipt.task_id


def test_unavailable_classifier_or_dispatcher_failure_cannot_release_promise(temp_home):
    event, source = _event()
    response, receipt = guard_final_response(runner=_Runner(None), ctx=_ctx(source), event=event,
                                             response="I will update you later.")
    assert response == _SAFE_REFUSAL and receipt is None
    event, source = _event()
    response, receipt = guard_final_response(runner=_Runner(_continuing(), ready=False), ctx=_ctx(source), event=event,
                                             response="I will update you later.")
    assert response == _SAFE_REFUSAL
    assert receipt is not None and not receipt.may_promise_follow_up


def test_completed_response_is_unchanged_and_enabled_turn_suppresses_streaming():
    event, source = _event()
    response, receipt = guard_final_response(
        runner=_Runner({"disposition": "completed"}), ctx=_ctx(source), event=event, response="It is done."
    )
    assert (response, receipt) == ("It is done.", None)
    context = TurnContext(source=source, user_config=_ctx(source).user_config)
    stream, delta, interim, enabled = TurnRunner(SimpleNamespace(config=None), context)._setup_stream_consumer("telegram")
    assert stream is None and delta is None and enabled is False
    assert interim("not emitted") is None
