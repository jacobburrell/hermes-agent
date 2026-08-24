"""Evidence-gated, privacy-preserving learning candidates for persistent goals.

This module intentionally does not write profile memory or skills.  A terminal
goal produces a quarantined retrospective; only repeated independently
verified outcomes make it eligible for review/promotion.  The curator may
deduplicate or archive candidates, but it never treats an agent claim as an
instruction by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable


_FILENAME = "goal_learning_candidates.json"
_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|password|secret|authorization)\b\s*[:=]\s*)([^\s,;]+)"
)
_RAW_PATH_RE = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\)[^\s'\"]+")
_MESSAGE_ID_RE = re.compile(r"(?i)\b(?:message|chat|thread)[_-]?id\s*[:=]\s*\S+")


def _home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _sanitize(value: Any, *, limit: int = 700) -> str:
    text = str(value or "")
    text = _SECRET_RE.sub(r"\1[REDACTED]", text)
    text = _RAW_PATH_RE.sub("[PATH]", text)
    text = _MESSAGE_ID_RE.sub("[MESSAGE_ID]", text)
    text = " ".join(text.split())
    return text[:limit]


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def _write(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _observed_provenance(ledger: Iterable[dict[str, Any]]) -> bool:
    return any(
        str(entry.get("provenance") or "") in {
            "deterministic_gate", "tool_observed", "user_confirmed",
        }
        for entry in ledger
        if isinstance(entry, dict)
    )


def record_terminal_retrospective(state: Any, outcome: str) -> dict[str, Any] | None:
    """Persist a sanitized candidate from an achieved/input/impossible outcome.

    Candidates from failed/speculative work are retained for context but are
    never promotion-eligible.  A candidate becomes ``review_eligible`` only
    after two distinct, evidence-backed achieved goals with the same tactic
    fingerprint, or after one deterministic proven quirk.  No skill is
    enabled here; that remains an explicit curator/reviewer action.
    """
    if outcome not in {"achieved", "needs_input", "not_achievable", "policy_stop"}:
        return None
    ledger = [entry for entry in (getattr(state, "progress_ledger", None) or []) if isinstance(entry, dict)]
    paths = [path for path in (getattr(state, "recovery_paths", None) or []) if isinstance(path, dict)]
    tried = [path for path in paths if path.get("state") in {"tried", "in_progress"}]
    strategy = [
        _sanitize(path.get("family"), limit=160)
        for path in tried
        if _sanitize(path.get("family"), limit=160)
    ]
    blocker = _sanitize(getattr(state, "last_reason", ""), limit=300)
    fingerprint_source = "|".join([
        _sanitize(getattr(state, "goal", ""), limit=320).lower(),
        "|".join(sorted(strategy)).lower(),
        str(outcome),
    ])
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8", "replace")).hexdigest()[:24]
    evidence_backed = _observed_provenance(ledger)
    candidate = {
        "id": f"goal-{fingerprint}",
        "fingerprint": fingerprint,
        "created_at": time.time(),
        "updated_at": time.time(),
        "outcome": outcome,
        "problem_fingerprint": _sanitize(getattr(state, "goal", ""), limit=420),
        "blocker": blocker,
        "strategies": strategy[:8],
        "prerequisites": [
            _sanitize(entry.get("remaining_hypotheses"), limit=240)
            for entry in ledger[-4:]
            if entry.get("remaining_hypotheses")
        ],
        "evidence": [
            {
                "provenance": _sanitize(entry.get("provenance"), limit=40),
                "summary": _sanitize(entry.get("evidence") or entry.get("reason"), limit=300),
            }
            for entry in ledger[-8:]
            if entry.get("evidence") or entry.get("reason")
        ],
        "scope": "profile-local candidate; shared only through reviewed skills",
        "confidence": "evidence_backed" if evidence_backed else "unverified",
        "verification_count": 1 if outcome == "achieved" and evidence_backed else 0,
        "status": "quarantined",
        "goal_ids": [str(getattr(state, "goal_id", ""))[:64]],
        "expires_at": time.time() + 90 * 24 * 60 * 60,
    }
    try:
        path = _home() / _FILENAME
        candidates = _read(path)
        existing = next((item for item in candidates if item.get("fingerprint") == fingerprint), None)
        if existing is None:
            candidates.append(candidate)
            result = candidate
        else:
            goal_id = str(getattr(state, "goal_id", ""))[:64]
            goal_ids = {str(value) for value in (existing.get("goal_ids") or []) if value}
            if goal_id and goal_id not in goal_ids:
                goal_ids.add(goal_id)
                if outcome == "achieved" and evidence_backed:
                    existing["verification_count"] = int(existing.get("verification_count", 0) or 0) + 1
            existing["goal_ids"] = sorted(goal_ids)[-8:]
            existing["updated_at"] = time.time()
            existing["confidence"] = "evidence_backed" if evidence_backed else existing.get("confidence", "unverified")
            if outcome == "achieved" and evidence_backed and int(existing.get("verification_count", 0) or 0) >= 2:
                existing["status"] = "review_eligible"
            # One strongly proven deterministic quirk is useful enough to
            # surface early, but still requires curator/reviewer promotion.
            if outcome == "achieved" and any(
                entry.get("provenance") == "deterministic_gate" for entry in ledger
            ):
                existing["status"] = "review_eligible"
            result = existing
        _write(path, candidates[-200:])
        return result
    except OSError:
        return None


__all__ = ["record_terminal_retrospective"]
