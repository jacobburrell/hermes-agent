"""Fail-closed presentation guard for local chat surfaces without push delivery."""
from __future__ import annotations

from typing import Any
import uuid

from hermes_cli.config import cfg_get, load_config
from hermes_cli.commitment_admission import CommitmentContext, propose_with_auxiliary

_SAFE = ("I can help in this conversation, but this interface cannot save and deliver "
         "a reliable follow-up task, so I will not claim that work will continue unattended.")
_VERIFY = ("I cannot verify that this response is backed by a reliable follow-up task from this "
           "interface, so I will not claim that work will continue unattended.")
_PRESENTATION_KEY = "local_commitment_guard"


def local_commitment_state() -> bool | None:
    """False only for verified disabled config; malformed/unreadable is fail-closed."""
    try:
        settings = cfg_get(load_config(), "goals", "commitment_admission", default=None)
    except Exception:
        return None
    if settings is None:
        return False
    return bool(settings.get("enabled")) if isinstance(settings, dict) else None


def local_commitment_refusal_result() -> dict[str, Any]:
    """A pre-execution refusal with no synthetic transcript/history payload."""
    return {"final_response": _VERIFY, "_local_commitment_presentation_override": _VERIFY,
            "_local_commitment_preflight_refusal": True}


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


def begin_local_commitment_fence(session_db: Any, session_id: str, *, source: str) -> dict[str, Any] | None:
    """Create a durable per-turn fence before local model execution."""
    if session_db is None or not session_id:
        return None
    try:
        rows = session_db.get_messages(session_id)
        watermark = max((int(row.get("id") or row.get("_row_id") or 0) for row in rows), default=0)
        turn_id = f"local-commitment:{uuid.uuid4()}"
        if not session_db.begin_api_presentation_fence(
                session_id, turn_id=turn_id, source=source, after_row_id=watermark):
            return None
        return {"session_id": session_id, "turn_id": turn_id, "after_row_id": watermark}
    except Exception:
        return None


def finish_local_commitment_fence(session_db: Any, fence: Any, result: Any = None, *, failed: bool = False) -> bool:
    """Resolve exactly the owned assistant rows with a safe display projection."""
    if session_db is None or not isinstance(fence, dict):
        return False
    session_id, turn_id = fence.get("session_id"), fence.get("turn_id")
    if not session_id or not turn_id:
        return False
    override = result.get("_local_commitment_presentation_override") if isinstance(result, dict) else None
    # A classifier failure/exception must never clear an unvalidated row.
    content = override if isinstance(override, str) and override else (_VERIFY if failed else None)
    try:
        row_ids = session_db.assistant_message_ids_after(session_id, int(fence.get("after_row_id") or 0))
        return bool(session_db.resolve_api_presentation_fence(
            session_id, turn_id=turn_id, presentation_key=_PRESENTATION_KEY,
            terminal_content=content, assistant_row_ids=row_ids))
    except Exception:
        return False
