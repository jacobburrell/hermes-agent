"""Fail-closed presentation guard for local chat surfaces without push delivery."""
from __future__ import annotations

from typing import Any

from hermes_cli.config import cfg_get, load_config
from hermes_cli.commitment_admission import CommitmentContext, propose_with_auxiliary

_SAFE = ("I can help in this conversation, but this interface cannot save and deliver "
         "a reliable follow-up task, so I will not claim that work will continue unattended.")
_VERIFY = ("I cannot verify that this response is backed by a reliable follow-up task from this "
           "interface, so I will not claim that work will continue unattended.")


def local_commitment_state() -> bool | None:
    """False only for verified disabled config; malformed/unreadable is fail-closed."""
    try:
        settings = cfg_get(load_config(), "goals", "commitment_admission", default=None)
    except Exception:
        return None
    if settings is None:
        return False
    return bool(settings.get("enabled")) if isinstance(settings, dict) else None


def guard_local_result(result: Any, *, text: str, session_id: str, platform: str,
                       state: bool | None = None) -> Any:
    """Keep ordinary verified completed/none replies; never create a local task."""
    state = local_commitment_state() if state is None else state
    if state is False or not isinstance(result, dict):
        return result
    final = result.get("final_response")
    if not isinstance(final, str) or not final.strip():
        return result
    proposal = None
    if state is not False:
        try:
            proposal = propose_with_auxiliary(CommitmentContext(
                profile="default", requester_id="local-user", session_id=str(session_id),
                platform=platform, chat_id=str(session_id), message_id=f"{session_id}:local",
                text=str(text or ""), authorized=True, operational=True,
                return_capable=False, dispatcher_ready=False), final)
        except Exception:
            proposal = None
    if proposal is not None and proposal.classification == "available" and proposal.disposition in {"none", "completed"}:
        return result
    safe = dict(result)
    safe["final_response"] = (
        _SAFE if proposal is not None and proposal.classification == "available" else _VERIFY
    )
    # Raw ``messages`` deliberately remain untouched for prompt-cache parity.
    safe["_local_commitment_presentation_override"] = safe["final_response"]
    return safe


def persist_local_commitment_presentation(result: Any, *, session_db: Any, session_id: str) -> bool:
    """Record a display-only local refusal without changing model history.

    The existing transcript schema already separates ``display_kind`` and
    ``display_metadata`` from assistant content.  This deliberately stamps
    only the just-persisted terminal row; callers retain the original model
    ``messages`` for prompt-cache and replay invariants.
    """
    if not isinstance(result, dict) or session_db is None or not session_id:
        return False
    override = result.get("_local_commitment_presentation_override")
    raw_messages = result.get("messages")
    if not isinstance(override, str) or not override or not isinstance(raw_messages, list):
        return False
    raw = next((row.get("content") for row in reversed(raw_messages)
                if isinstance(row, dict) and row.get("role") == "assistant"
                and isinstance(row.get("content"), str) and row.get("content")), "")
    if not raw:
        return False
    try:
        return bool(session_db.set_latest_matching_message_display_kind(
            session_id, role="assistant", content=raw, display_kind="local_commitment_guard",
            display_metadata={"local_commitment_guard": {"text": override}}))
    except Exception:
        return False


def persist_local_commitment_exception_presentation(session_db: Any, *, session_id: str,
                                                    after_row_id: int) -> bool:
    """Hide a locally persisted partial reply after a guarded turn raises.

    TUI serializes turns for one session.  We capture its pre-turn watermark
    and apply a display-only verification fallback to assistant rows written
    by the failed turn; content remains unchanged in SQLite and in the model
    result/cache.  Failure to read or stamp is reported to the caller so it
    can retain its normal error path rather than fabricate a successful guard.
    """
    if session_db is None or not session_id:
        return False
    try:
        rows = session_db.get_messages(session_id)
        changed = False
        for row in rows:
            row_id = row.get("id") or row.get("_row_id")
            content = row.get("content")
            if (isinstance(row_id, int) and row_id > after_row_id and row.get("role") == "assistant"
                    and isinstance(content, str) and content):
                changed = bool(session_db.set_latest_matching_message_display_kind(
                    session_id, role="assistant", content=content,
                    display_kind="local_commitment_guard",
                    display_metadata={"local_commitment_guard": {"text": _VERIFY}})) or changed
        return changed
    except Exception:
        return False
