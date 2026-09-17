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
from hermes_state import SessionDB


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

    def _is_user_authorized_for_source(self, source):
        return True

    def _adapter_for_source(self, source):
        return SimpleNamespace(send=lambda *a, **kw: None, supports_async_delivery=True)


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


def test_completed_response_is_unchanged_and_disabled_stream_fixture_is_safe():
    event, source = _event()
    response, receipt = guard_final_response(
        runner=_Runner({"disposition": "completed"}), ctx=_ctx(source), event=event, response="It is done."
    )
    assert (response, receipt) == ("It is done.", None)
    context = TurnContext(
        source=source, user_config=_ctx(source).user_config,
        resolve_display_setting=lambda *_args: False, _run_still_current=lambda: True,
    )
    stream, delta, interim, enabled = TurnRunner(SimpleNamespace(config=None), context)._setup_stream_consumer("telegram")
    assert stream is None and callable(delta) and enabled is False
    assert callable(interim)
    assert interim("not emitted") is None


def test_quote_is_context_unless_adapter_marks_it_untrusted_authority(temp_home):
    event, source = _event()
    event.reply_to_text = "A colleague said the course needs approval."
    response, receipt = guard_final_response(
        runner=_Runner(_continuing()), ctx=_ctx(source), event=event,
        response="I will continue the enrollment research.",
    )
    assert response.startswith("I will continue") and receipt and receipt.may_promise_follow_up

    event, source = _event()
    event.metadata["quote_is_untrusted_authority"] = True
    response, receipt = guard_final_response(
        runner=_Runner(_continuing()), ctx=_ctx(source), event=event,
        response="I will continue the enrollment research.",
    )
    assert response == _SAFE_REFUSAL and receipt is not None and not receipt.may_promise_follow_up


def test_commitment_stream_fence_discards_raw_deltas_after_safe_boundary():
    """Only the guarded terminal response may be emitted after an admission fence."""
    context = TurnContext(_run_still_current=lambda: True)
    turn = TurnRunner(SimpleNamespace(), context)
    delivered = []
    turn._commitment_stream_events = [
        ("delta", "ordinary answer"), ("interim", ("thinking", False)),
    ]
    turn._commitment_stream_flush = (
        lambda text: delivered.append(("delta", text)),
        lambda text, *, already_streamed=False: delivered.append(("interim", text, already_streamed)),
    )
    assert delivered == []
    turn._finish_commitment_stream_fence(release=False)
    assert delivered == []

    turn._commitment_stream_events = [("delta", "I will update you later")]
    turn._commitment_stream_flush = (lambda text: delivered.append(("delta", text)), None)
    turn._finish_commitment_stream_fence(release=False)
    assert all("update you later" not in str(item) for item in delivered)


def test_gateway_presentation_fence_keeps_raw_history_but_projects_guarded_turn(temp_home):
    """A raw agent-persisted promise cannot reappear after the common guard."""
    db = SessionDB()
    try:
        db.ensure_session("origin-session", source="gateway")
        event, source = _event()
        context = TurnContext(source=source, session_id="origin-session", user_config=_ctx(source).user_config)
        turn = TurnRunner(SimpleNamespace(_session_db=SimpleNamespace(_db=db)), context)
        fence, ready = turn._begin_gateway_commitment_presentation_fence()
        assert ready and fence
        db.append_message("origin-session", "assistant", "I will update you later.",
                          display_metadata={"_gateway_commitment_fence": fence["turn_id"]})
        assert db.resolve_api_presentation_fence(
            "origin-session", turn_id=fence["turn_id"], fallback=_SAFE_REFUSAL,
            assistant_row_ids=db.find_api_presentation_fence_rows("origin-session", turn_id=fence["turn_id"]),
            require_exact_rows=True, presented_assistant_row_id=1)
        # A later normal turn clears only its own fence; it never overwrites
        # the earlier refused projection.
        later, later_ready = turn._begin_gateway_commitment_presentation_fence()
        assert later_ready and later
        db.append_message("origin-session", "assistant", "It is done.",
                          display_metadata={"_gateway_commitment_fence": later["turn_id"]})
        assert db.resolve_api_presentation_fence(
            "origin-session", turn_id=later["turn_id"],
            assistant_row_ids=db.find_api_presentation_fence_rows("origin-session", turn_id=later["turn_id"]),
            require_exact_rows=True, presented_assistant_row_id=2)
        shown, pending = db.get_api_presentation_snapshot(
            "origin-session", limit=None, offset=0, latest=False)
        assert pending == []
        assert [m["content"] for m in shown] == [_SAFE_REFUSAL, "It is done."]
        assert [m["content"] for m in db.get_messages("origin-session")] == [
            "I will update you later.", "It is done."]
    finally:
        db.close()


def test_exact_gateway_fence_hides_every_owned_intermediate_not_later_turn(temp_home):
    db = SessionDB()
    try:
        db.ensure_session("origin-session", source="gateway")
        assert db.begin_api_presentation_fence(
            "origin-session", turn_id="guarded", source="gateway_commitment", after_row_id=0)
        for text in ("I will keep working", "I will update you later"):
            db.append_message("origin-session", "assistant", text,
                              display_metadata={"_gateway_commitment_fence": "guarded"})
        db.append_message("origin-session", "assistant", "A later independent answer")
        owned = db.find_api_presentation_fence_rows("origin-session", turn_id="guarded")
        assert len(owned) == 2
        assert db.resolve_api_presentation_fence(
            "origin-session", turn_id="guarded", fallback=_SAFE_REFUSAL,
            assistant_row_ids=owned, require_exact_rows=True, presented_assistant_row_id=owned[-1])
        shown, pending = db.get_api_presentation_snapshot(
            "origin-session", limit=None, offset=0, latest=False)
        assert pending == []
        assert [row["content"] for row in shown] == [_SAFE_REFUSAL, "A later independent answer"]
        assert [row["content"] for row in db.get_messages("origin-session")] == [
            "I will keep working", "I will update you later", "A later independent answer"]
    finally:
        db.close()


def test_unchanged_final_hides_owned_interim_rows(temp_home):
    db = SessionDB()
    try:
        db.ensure_session("origin-session", source="gateway")
        assert db.begin_api_presentation_fence(
            "origin-session", turn_id="normal", source="gateway_commitment", after_row_id=0)
        for text in ("draft interim", "Validated final"):
            db.append_message("origin-session", "assistant", text,
                              display_metadata={"_gateway_commitment_fence": "normal"})
        ids = db.find_api_presentation_fence_rows("origin-session", turn_id="normal")
        assert db.resolve_api_presentation_fence(
            "origin-session", turn_id="normal", assistant_row_ids=ids,
            require_exact_rows=True, presented_assistant_row_id=ids[-1])
        shown, _ = db.get_api_presentation_snapshot("origin-session", limit=None, offset=0, latest=False)
        assert [row["content"] for row in shown] == ["Validated final"]
        assert [row["content"] for row in db.get_messages("origin-session")] == ["draft interim", "Validated final"]
    finally:
        db.close()


def test_interrupted_gateway_fence_hides_only_its_marked_rows(temp_home):
    """A crashed guarded turn must not suppress a later independent assistant row."""
    from gateway.platforms.api_server import APIServerAdapter
    db = SessionDB()
    try:
        db.ensure_session("origin-session", source="gateway")
        assert db.begin_api_presentation_fence(
            "origin-session", turn_id="interrupted", source="gateway_commitment", after_row_id=0)
        db.append_message("origin-session", "assistant", "I will finish this later.",
                          display_metadata={"_gateway_commitment_fence": "interrupted"})
        db.append_message("origin-session", "assistant", "Independent later answer.")
        shown, pending = db.get_api_presentation_snapshot(
            "origin-session", limit=None, offset=0, latest=False)
        assert pending and pending[0]["row_ids"]
        projected = APIServerAdapter._project_commitment_presentation(SimpleNamespace(), shown, pending)
        assert [row["content"] for row in projected] == ["Independent later answer."]
        # Canonical transcript has not been rewritten by the projection.
        assert [row["content"] for row in db.get_messages("origin-session")] == [
            "I will finish this later.", "Independent later answer."]
    finally:
        db.close()


def test_dead_empty_gateway_fence_reconciles_without_hiding_later_turn(temp_home):
    """A crash before any assistant row cannot become an indefinite broad fence."""
    db = SessionDB()
    try:
        db.ensure_session("origin-session", source="gateway")
        assert db.begin_api_presentation_fence(
            "origin-session", turn_id="crashed-empty", source="gateway_commitment", after_row_id=0)
        # This PID is intentionally non-existent on the local host.  The
        # snapshot reconciliation may clear only because no marked row exists.
        db._write_sql("UPDATE api_presentation_fences SET owner_pid=? WHERE session_id=? AND turn_id=?",
                      (999_999_999, "origin-session", "crashed-empty"))
        db.append_message("origin-session", "assistant", "Later independent answer.")
        shown, pending = db.get_api_presentation_snapshot(
            "origin-session", limit=None, offset=0, latest=False)
        assert pending == []
        assert [row["content"] for row in shown] == ["Later independent answer."]
        assert [row["content"] for row in db.get_messages("origin-session")] == ["Later independent answer."]
    finally:
        db.close()


def test_gateway_fence_requires_session_db_only_when_commitment_mode_is_enabled():
    """No DB is a truthful refusal for guarded work, not a regression for ordinary turns."""
    event, source = _event()
    enabled_ctx = TurnContext(source=source, session_id="origin-session", user_config=_ctx(source).user_config)
    _, ready = TurnRunner(SimpleNamespace(_session_db=None), enabled_ctx)._begin_gateway_commitment_presentation_fence()
    assert ready is False

    disabled_ctx = TurnContext(source=source, session_id="origin-session", user_config={})
    fence, ready = TurnRunner(SimpleNamespace(_session_db=None), disabled_ctx)._begin_gateway_commitment_presentation_fence()
    assert (fence, ready) == (None, True)

    heartbeat_ctx = TurnContext(source=source, session_id="origin-session", scheduled_heartbeat=True,
                                user_config=_ctx(source).user_config)
    fence, ready = TurnRunner(SimpleNamespace(_session_db=None), heartbeat_ctx)._begin_gateway_commitment_presentation_fence()
    assert (fence, ready) == (None, True)
