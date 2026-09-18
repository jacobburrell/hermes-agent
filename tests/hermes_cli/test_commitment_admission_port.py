"""Focused current-main coverage for the transport-neutral commitment store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kb_connect
from hermes_cli import kanban_db_notify as kb_notify
from hermes_cli.commitment_admission import (
    CommitmentAttachment,
    CommitmentContext,
    CommitmentProposal,
    KanbanCommitmentStore,
    admit_commitment,
    idempotency_key,
)


@pytest.fixture
def isolated_kanban(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _context(**changes) -> CommitmentContext:
    values = dict(
        profile="jackwhatsapp", requester_id="user-1", session_id="session-1",
        platform="test", chat_id="chat-1", message_id="message-1",
        text="Please keep working and tell me when it is done.", authorized=True,
        operational=True, chat_type="dm", thread_id="thread-1",
        attachment_refs=("attachment:one",), return_capable=True,
        dispatcher_ready=True,
    )
    values.update(changes)
    return CommitmentContext(**values)


def _proposal(**changes) -> CommitmentProposal:
    values = dict(
        disposition="continuing", objective="Prepare the requested report",
        completion_criteria="A report is saved and the requester is notified.",
        next_action="Inspect the supplied attachment.", classification="available",
    )
    values.update(changes)
    return CommitmentProposal(**values)


def test_admission_requires_route_dispatcher_and_stable_source_identity():
    class Store:
        def save(self, context, proposal):  # pragma: no cover - assertion below proves it is not called
            raise AssertionError("invalid commitment must not persist")

    assert not admit_commitment(_context(return_capable=False), _proposal(), Store()).may_promise_follow_up
    assert not admit_commitment(_context(dispatcher_ready=False), _proposal(), Store()).may_promise_follow_up
    assert not admit_commitment(_context(message_id=""), _proposal(), Store()).may_promise_follow_up


def test_idempotency_is_bound_to_source_thread_and_message():
    original = idempotency_key(_context())
    assert original != idempotency_key(_context(message_id="message-2"))
    assert original != idempotency_key(_context(thread_id="thread-2"))
    assert original == idempotency_key(_context())


def test_current_split_kanban_store_stages_subscribes_promotes_and_replays(isolated_kanban):
    store = KanbanCommitmentStore(assignee="jack", created_by="jackwhatsapp")
    first = admit_commitment(_context(), _proposal(), store)
    again = admit_commitment(_context(), _proposal(), store)

    assert first.may_promise_follow_up
    assert again.may_promise_follow_up
    assert again.task_id == first.task_id

    conn = kb_connect.connect()
    try:
        task = conn.execute("SELECT status, goal_mode, body FROM tasks WHERE id = ?", (first.task_id,)).fetchone()
        assert task["status"] == "ready"
        assert task["goal_mode"] == 1
        assert "attachment:one" in task["body"]
        subscriptions = kb_notify.list_notify_subs(conn, first.task_id)
        assert [(s["platform"], s["chat_id"], s["thread_id"]) for s in subscriptions] == [
            ("test", "chat-1", "thread-1")
        ]
    finally:
        conn.close()


def test_replay_does_not_resume_an_operator_blocked_task_and_keeps_execution_session_separate(isolated_kanban):
    store = KanbanCommitmentStore(assignee="jack", created_by="jackwhatsapp")
    first = admit_commitment(_context(session_id="origin-session"), _proposal(), store)
    conn = kb_connect.connect()
    try:
        assert kb.block_task(conn, first.task_id, reason="need operator approval", kind="needs_input")
    finally:
        conn.close()

    replay = admit_commitment(_context(session_id="origin-session"), _proposal(), store)
    assert replay.task_id == first.task_id
    assert not replay.may_promise_follow_up
    conn = kb_connect.connect()
    try:
        row = conn.execute("SELECT status, session_id FROM tasks WHERE id = ?", (first.task_id,)).fetchone()
        assert row["status"] == "blocked"
        assert row["session_id"] != "origin-session"
        assert row["session_id"].startswith("commitment:")
    finally:
        conn.close()


def test_attachment_bytes_are_copied_and_missing_refs_are_manifested(isolated_kanban):
    attachment = CommitmentAttachment(
        reference="transport:file-1", filename="notes.txt", content_type="text/plain", data=b"durable bytes",
    )
    result = admit_commitment(_context(attachment_refs=(attachment, "transport:missing")), _proposal(),
                              KanbanCommitmentStore(assignee="jack", created_by="jackwhatsapp"))
    assert result.may_promise_follow_up
    conn = kb_connect.connect()
    try:
        attachments = kb.list_attachments(conn, result.task_id)
        assert len(attachments) == 1
        assert Path(attachments[0].stored_path).read_bytes() == b"durable bytes"
        body = conn.execute("SELECT body FROM tasks WHERE id = ?", (result.task_id,)).fetchone()["body"]
        assert '"state": "missing"' in body
    finally:
        conn.close()


def test_long_body_reconciliation_and_waiting_condition_are_durable(isolated_kanban):
    proposal = _proposal(
        disposition="waiting",
        objective="Enroll in the requested course " + ("with supporting detail " * 80),
        completion_criteria="Enrollment is confirmed with the registrar.",
        next_action="Resume after the registrar replies.",
        waiting_for="Explicit approval from Alice",
    )
    result = admit_commitment(_context(message_id="long-wait"), proposal,
                              KanbanCommitmentStore(assignee="jack", created_by="jackwhatsapp"))
    assert result.task_id and result.persisted and result.delivery_ready
    conn = kb_connect.connect()
    try:
        task = kb.get_task(conn, result.task_id)
        body = json.loads(task.body)
        assert body["waiting_for"] == "Explicit approval from Alice"
        assert body["objective"] == proposal.objective
    finally:
        conn.close()


def test_persistence_failure_never_authorizes_follow_up():
    class BrokenStore:
        def save(self, context, proposal):
            raise RuntimeError("disk unavailable")

    result = admit_commitment(_context(), _proposal(), BrokenStore())
    assert not result.may_promise_follow_up
    assert result.reason == "durable task persistence failed"
