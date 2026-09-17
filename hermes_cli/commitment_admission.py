"""Transport-neutral durable admission for ordinary-language commitments.

This deliberately has no gateway or WhatsApp dependency: an edge supplies a
trusted source context, and this module refuses to release continuing work
until the existing Kanban task and notification route are durable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class CommitmentProposal:
    disposition: str = "none"
    objective: str = ""
    completion_criteria: str = ""
    next_action: str = ""
    waiting_for: str = ""
    confidence: float = 0.0
    classification: str = "available"


@dataclass(frozen=True)
class CommitmentAttachment:
    """An attachment offered by an authenticated transport boundary.

    ``data`` is deliberately optional: a bridge that could not download media
    must say so in the durable origin manifest rather than pretending a URL is
    a durable worker attachment.
    """
    reference: str
    filename: str = "attachment"
    content_type: str = ""
    data: bytes | None = None
    state: str = "available"  # available | missing | download_failed


@dataclass(frozen=True)
class CommitmentContext:
    profile: str
    requester_id: str
    session_id: str
    platform: str
    chat_id: str
    message_id: str
    text: str
    authorized: bool
    operational: bool
    internal: bool = False
    quoted: bool = False
    chat_type: str = ""
    thread_id: str = ""
    attachment_refs: tuple[CommitmentAttachment | str, ...] = ()
    return_capable: bool = False
    dispatcher_ready: bool = False


@dataclass(frozen=True)
class CommitmentAdmission:
    disposition: str
    task_id: str = ""
    persisted: bool = False
    delivery_ready: bool = False
    reason: str = ""

    @property
    def may_promise_follow_up(self) -> bool:
        return self.disposition == "continuing" and self.persisted and self.delivery_ready


def normalize_proposal(raw: Any) -> CommitmentProposal:
    """Normalize an auxiliary assessment without making prose authoritative.

    The edge may obtain this mapping from an auxiliary model, but anything
    malformed or outside the small enum is fail-closed as ``unavailable``.
    Deterministic validation below still establishes authority and routing.
    """
    if not isinstance(raw, Mapping):
        return CommitmentProposal(classification="unavailable")
    disposition = _clean(raw.get("disposition"), 32).lower()
    if disposition not in {"none", "completed", "continuing", "waiting", "cancel", "correction"}:
        return CommitmentProposal(classification="unavailable")
    return CommitmentProposal(
        disposition=disposition,
        objective=_clean(raw.get("objective")),
        completion_criteria=_clean(raw.get("completion_criteria")),
        next_action=_clean(raw.get("next_action")),
        waiting_for=_clean(raw.get("waiting_for")),
        confidence=float(raw.get("confidence") or 0.0),
        classification="available" if raw.get("classification", "available") == "available" else "unavailable",
    )


def _clean(value: Any, limit: int = 1200) -> str:
    return str(value or "").replace("\0", "").strip()[:limit]


def deterministic_validation(context: CommitmentContext, proposal: CommitmentProposal) -> Optional[str]:
    if proposal.disposition not in {"continuing", "waiting"}: return "not a continuing commitment"
    if proposal.classification != "available": return "classifier result is not usable"
    if not context.authorized or not context.operational or context.internal or context.quoted:
        return "source is not an authorized operational request"
    if not all((_clean(context.requester_id), _clean(context.session_id), _clean(context.message_id))):
        return "missing stable source identity"
    if not all((_clean(context.platform), _clean(context.chat_id))) or not context.return_capable:
        return "no durable return route"
    if proposal.disposition == "continuing" and not context.dispatcher_ready: return "no ready continuation dispatcher"
    if not _clean(proposal.objective, 200) or not _clean(proposal.completion_criteria, 200):
        return "proposal lacks observable completion criteria"
    if proposal.disposition == "waiting" and not _clean(proposal.waiting_for, 200):
        return "waiting proposal lacks a waiting condition"
    if proposal.disposition == "continuing" and not _clean(proposal.next_action, 200): return "proposal lacks next action"
    return None


def idempotency_key(context: CommitmentContext) -> str:
    raw = "\x1f".join(("commitment-v1", context.profile, context.platform, context.chat_id,
                         context.thread_id, context.requester_id, context.message_id))
    return "commitment:" + hashlib.sha256(raw.encode()).hexdigest()


class KanbanCommitmentStore:
    """Compatibility adapter over current Kanban task/notification APIs."""
    def __init__(self, *, assignee: str, created_by: str, board: Optional[str] = None):
        self.assignee, self.created_by, self.board = _clean(assignee, 128), _clean(created_by, 128), board

    def _attachment_manifest(self, values: Sequence[CommitmentAttachment | str]) -> list[dict[str, str]]:
        manifest: list[dict[str, str]] = []
        for value in values:
            if isinstance(value, CommitmentAttachment):
                manifest.append({"reference": _clean(value.reference), "filename": _clean(value.filename),
                                 "state": _clean(value.state, 40) or "available"})
            else:
                manifest.append({"reference": _clean(value), "filename": "", "state": "missing"})
        return manifest

    def _execution_session_id(self, context: CommitmentContext) -> str:
        # Never borrow the originating chat session: concurrent commitments
        # need independent Goal ownership and restart recovery.
        return "commitment:" + hashlib.sha256(idempotency_key(context).encode()).hexdigest()[:24]

    @staticmethod
    def _matching_subscription(sub: Mapping[str, Any], context: CommitmentContext) -> bool:
        metadata = sub.get("delivery_metadata") or {}
        return (
            sub.get("platform") == context.platform and sub.get("chat_id") == context.chat_id
            and (sub.get("thread_id") or "") == (context.thread_id or "")
            and sub.get("user_id") == context.requester_id
            and sub.get("notifier_profile") == context.profile
            and isinstance(metadata, Mapping) and metadata.get("commitment_key") == idempotency_key(context)
        )

    def save(self, context: CommitmentContext, proposal: CommitmentProposal) -> tuple[str, bool]:
        if not self.assignee: raise RuntimeError("no commitment assignee")
        # Current main deliberately splits connection and notification concerns
        # out of ``kanban_db``.  Keep that boundary here instead of restoring
        # the old monolithic facade just for commitment admission.
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kb_connect
        from hermes_cli import kanban_db_notify as kb_notify
        manifest = self._attachment_manifest(context.attachment_refs)
        body = json.dumps({"kind": "user_commitment", "objective": proposal.objective,
            "completion_criteria": proposal.completion_criteria, "next_action": proposal.next_action,
            "origin": {"profile": context.profile, "requester_id": context.requester_id,
                       "platform": context.platform, "chat_id": context.chat_id,
                       "thread_id": context.thread_id, "message_id": context.message_id,
                       "attachments": manifest,
                       "commitment_key": idempotency_key(context)}} , sort_keys=True)
        conn = kb_connect.connect(board=self.board)
        try:
            task_id = kb.create_task(conn, title=proposal.objective[:240], body=body,
                assignee=self.assignee, created_by=self.created_by or context.profile,
                idempotency_key=idempotency_key(context), goal_mode=True,
                initial_status="blocked", session_id=self._execution_session_id(context))
            task = kb.get_task(conn, str(task_id))
            # An idempotency collision must be the same trusted obligation;
            # never borrow another requester/profile's task merely by key.
            if task is None or _clean(getattr(task, "body", "")) != body:
                raise RuntimeError("commitment task linkage does not match source")
            comments = {
                str(row[0]) for row in conn.execute(
                    "SELECT body FROM task_comments WHERE task_id = ?", (str(task_id),)
                ).fetchall()
            }
            staged = f"commitment_staged:{idempotency_key(context)}"
            admitted = f"commitment_admitted:{idempotency_key(context)}"
            if staged not in comments:
                kb.add_comment(conn, str(task_id), self.created_by or context.profile, staged)
            kb_notify.add_notify_sub(conn, task_id=str(task_id), platform=context.platform, chat_id=context.chat_id,
                user_id=context.requester_id, chat_type=context.chat_type or "dm", thread_id=context.thread_id or None,
                notifier_profile=context.profile, delivery_mode="notify", delivery_metadata={
                    "commitment_admission": True, "commitment_key": idempotency_key(context),
                    "suppress_internal_diagnostics": True, "dedupe_unchanged_blockers": True,
                    "origin_profile": context.profile, "origin_requester": context.requester_id,
                    "origin_thread": context.thread_id or "",
                })
            subs = kb_notify.list_notify_subs(conn, str(task_id))
            ready = any(self._matching_subscription(s, context) for s in subs)
            if not ready: return str(task_id), False
            # Copy owned bytes before executable promotion.  A URL/ref alone is
            # only a manifest entry and explicitly remains missing to workers.
            existing_files = {att.filename for att in kb.list_attachments(conn, str(task_id))}
            for attachment in context.attachment_refs:
                if not isinstance(attachment, CommitmentAttachment) or attachment.data is None:
                    continue
                if attachment.filename in existing_files:
                    continue
                kb.store_attachment_bytes(conn, str(task_id), attachment.filename, attachment.data,
                                          content_type=attachment.content_type or None,
                                          uploaded_by=context.requester_id, board=self.board)
            # ``create_task`` returns the existing card for an idempotent replay.
            # A previously promoted card is already ready, and attempting to
            # unblock it is correctly a no-op in the split lifecycle API.
            row = conn.execute("SELECT status FROM tasks WHERE id = ?", (str(task_id),)).fetchone()
            if not row: return str(task_id), False
            # A blocked card with an admission marker was subsequently parked
            # by an operator/worker.  Replays must never silently resume it.
            if row["status"] == "blocked" and admitted in comments:
                return str(task_id), False
            if proposal.disposition == "waiting":
                if admitted not in comments:
                    kb.add_comment(conn, str(task_id), self.created_by or context.profile, admitted)
                return str(task_id), True
            if row["status"] != "ready" and not kb.unblock_task(conn, str(task_id)):
                return str(task_id), False
            if admitted not in comments:
                kb.add_comment(conn, str(task_id), self.created_by or context.profile, admitted)
            return str(task_id), True
        finally:
            conn.close()


def admit_commitment(context: CommitmentContext, proposal: CommitmentProposal, store: Any) -> CommitmentAdmission:
    if reason := deterministic_validation(context, proposal):
        return CommitmentAdmission("none", reason=reason)
    try: task_id, ready = store.save(context, proposal)
    except Exception: return CommitmentAdmission("none", reason="durable task persistence failed")
    return CommitmentAdmission(proposal.disposition, str(task_id or ""), bool(task_id), bool(ready),
                               "" if task_id else "task store returned no task id")
