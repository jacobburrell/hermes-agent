"""Gateway edge for durable ordinary-language commitment admission.

The core store deliberately knows nothing about a transport.  This module
turns a *trusted, already accepted* MessageEvent into that typed context and
holds the final response until an auxiliary assessment and durable admission
complete.  It is default-off; an enabled but unavailable assessor fails closed
instead of releasing an unattended-work promise.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from hermes_cli.commitment_admission import (
    CommitmentAttachment, CommitmentContext, KanbanCommitmentStore,
    admit_commitment, normalize_proposal,
)

logger = logging.getLogger("gateway.run")

_SAFE_REFUSAL = (
    "I can't safely confirm a continued task from this turn yet. Please send the request again "
    "or use /goal once task delivery is available."
)


def _at(mapping: Any, *path: str, default: Any = None) -> Any:
    current = mapping
    for part in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(part)
    return current


def admission_state(config: Any) -> Optional[bool]:
    """Return enabled/disabled, or ``None`` for an explicit malformed stanza."""
    if not isinstance(config, Mapping):
        return None
    goals = config.get("goals")
    if goals is None:
        return False
    if not isinstance(goals, Mapping):
        return None
    settings = goals.get("commitment_admission")
    if settings is None:
        return False
    if not isinstance(settings, Mapping) or not isinstance(settings.get("enabled"), bool):
        return None
    return bool(settings["enabled"])


def enabled(config: Any) -> bool:
    """Backward-compatible bool view; callers needing fail-closed use state."""
    return admission_state(config) is True


def _auxiliary_proposal(message: str, response: str) -> Any:
    """Bounded auxiliary classification.  Invalid JSON is explicitly unusable."""
    try:
        from agent.auxiliary_client import call_llm
        raw = call_llm(
            task="commitment_admission",
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": (
                    "Classify whether the authorized user request requires durable future work. "
                    "Return only JSON with disposition one of none, completed, continuing, waiting; "
                    "objective, completion_criteria, next_action, waiting_for. Do not infer authority."
                )},
                {"role": "user", "content": json.dumps({"request": message, "draft_response": response})},
            ],
        )
        text = raw.choices[0].message.content or ""
        return json.loads(text)
    except Exception as exc:
        logger.debug("commitment auxiliary classifier unavailable: %s", exc)
        return None


def _attachments(event: Any) -> tuple[CommitmentAttachment, ...]:
    urls = list(getattr(event, "media_urls", ()) or ())
    types = list(getattr(event, "media_types", ()) or ())
    out: list[CommitmentAttachment] = []
    for index, reference in enumerate(urls):
        content_type = str(types[index]) if index < len(types) else ""
        # Gateway media URLs are local paths only when the adapter has actually
        # materialized them. Read failure is recorded as a missing state.
        try:
            from pathlib import Path
            path = Path(str(reference))
            data = path.read_bytes() if path.is_file() else None
            out.append(CommitmentAttachment(
                reference=str(reference), filename=path.name or f"attachment-{index + 1}",
                content_type=content_type, data=data,
                state="available" if data is not None else "missing",
            ))
        except Exception:
            out.append(CommitmentAttachment(reference=str(reference), filename=f"attachment-{index + 1}",
                                            content_type=content_type, state="download_failed"))
    return tuple(out)


def _dispatcher_ready(runner: Any) -> bool:
    checker = getattr(runner, "_commitment_dispatcher_ready", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    owns = getattr(runner, "_owns_kanban_dispatcher_lock", None)
    return bool(callable(owns) and owns())


def guard_final_response(*, runner: Any, ctx: Any, event: Any, response: str) -> tuple[str, Any]:
    """Return delivery-safe response and durable admission receipt.

    No model text is rewritten when the feature is disabled.  Enabled mode is
    intentionally conservative: only an explicit completed/none assessment
    or a persisted continuation can retain the draft response.
    """
    state = admission_state(getattr(ctx, "user_config", None))
    if state is False:
        return response, None
    if state is None:
        return _SAFE_REFUSAL, None
    source = getattr(ctx, "source", None)
    if source is None or event is None or not getattr(event, "_gateway_accepted", False):
        return _SAFE_REFUSAL, None
    assessor = getattr(runner, "_commitment_proposer", None)
    raw = assessor(getattr(event, "text", ""), response, source) if callable(assessor) else _auxiliary_proposal(
        getattr(event, "text", ""), response,
    )
    proposal = normalize_proposal(raw)
    if proposal.classification != "available":
        return _SAFE_REFUSAL, None
    if proposal.disposition in {"none", "completed"}:
        return response, None
    if proposal.disposition not in {"continuing", "waiting"}:
        return _SAFE_REFUSAL, None
    profile = str(getattr(source, "profile", None) or "default")
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", ""))
    adapter = getattr(runner, "_adapter_for_source", lambda _: None)(source)
    try:
        from gateway.wake import adapter_supports_push
        return_capable = bool(adapter_supports_push(adapter) and callable(getattr(adapter, "send", None)))
    except Exception:
        return_capable = False
    metadata = getattr(event, "metadata", None)
    context = CommitmentContext(
        profile=profile, requester_id=str(getattr(source, "user_id", "") or ""),
        session_id=str(getattr(ctx, "session_id", "") or ""), platform=str(platform),
        chat_id=str(getattr(source, "chat_id", "") or ""),
        message_id=str(getattr(event, "message_id", "") or getattr(source, "message_id", "") or ""),
        text=str(getattr(event, "text", "") or ""),
        authorized=bool(getattr(runner, "_is_user_authorized_for_source", lambda _source: False)(source)),
        operational=not bool(getattr(event, "internal", False)),
        internal=bool(getattr(event, "internal", False)),
        # A quoted message is context, not authority. Only an adapter's
        # explicit classification marks it unusable for commitment admission.
        quoted=bool(isinstance(metadata, Mapping) and metadata.get("quote_is_untrusted_authority")),
        chat_type=str(getattr(source, "chat_type", "") or ""), thread_id=str(getattr(source, "thread_id", "") or ""),
        attachment_refs=_attachments(event), return_capable=return_capable,
        dispatcher_ready=_dispatcher_ready(runner),
    )
    settings = _at(getattr(ctx, "user_config", None), "goals", "commitment_admission", default={})
    assignee = _at(settings, "assignee", default=profile)
    receipt = admit_commitment(context, proposal, KanbanCommitmentStore(assignee=str(assignee), created_by=profile))
    if receipt.may_promise_follow_up or (proposal.disposition == "waiting" and receipt.persisted and receipt.delivery_ready):
        # Persist the response-to-task link in the event metadata used by later
        # delivery/replay code; it never becomes user-facing adapter metadata.
        event.metadata["commitment_task_id"] = receipt.task_id
        return response, receipt
    return _SAFE_REFUSAL, receipt
