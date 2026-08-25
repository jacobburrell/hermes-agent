"""Fail-closed user-visible output policy for WhatsApp.

WhatsApp is an inbox, not an operator console.  This module is intentionally
adapter-local: it protects the personal-account bridge without changing stock
display policy for other platforms.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

logger = logging.getLogger(__name__)

OUTPUT_KIND_KEY = "_whatsapp_output_kind"
USER_VISIBLE_KINDS = frozenset({"final", "clarify", "approval", "command"})
STRICT_POLICY = "user_visible_only"

# These are structural diagnostics rather than ordinary user prose.  Keep the
# check narrow: an answer explaining a normal error remains possible, while a
# traceback, lifecycle notice, or raw internal payload cannot reach WhatsApp.
_INVISIBLE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF\u180E\u2061-\u2064]")
_SILENT = {"[SILENT]", "[[SILENT]]", "<SILENT>", "SILENCIO"}
_INTERNAL_PAYLOAD_PATTERNS = tuple(re.compile(p, re.I | re.S) for p in (
    r"^\s*(?:traceback|internal diagnostic|internal reasoning fallback|goal persistence|⚕\s*hermes agent)",
    r"^\s*(?:context compression|conversation limit|session (?:reset|restored)|history cleared|background process|model failure|api error|credential depleted|http \d{3} provider)",
    r"^\s*(?:◐\s*)?session automatically reset\b", r"^\s*✨\s*session reset\b",
    r"^\s*(?:⚠️?\s*)?(?:gateway|bridge) (?:is )?(?:starting|restarting|shutting down|draining|stopping|restarted|restart(?:ed)?|is back)\b",
    r"^\s*(?:⚠️?\s*)?no reply\s*:", r"^\s*(?:request payload too large|context length exceeded)\b[\s\S]*(?:cannot compress further|max compression attempts)",
    r"^\s*(?:💾\s*)?self-improvement review\s*:", r"^\s*(?:preflight |compressing |compacting )?context\b[\s\S]*\b(?:queued|compress(?:ing|ed)|cannot compress further|max compression attempts)\b",
    r"^\s*(?:session restored successfully|conversation history cleared|use /resume|adjust reset timing|operation interrupted|interrupted during api call|interrupting current task)\b",
    r"^\s*\[important:\s*background process", r"^\s*\[background process\s+proc_[a-z0-9_-]+\s+(?:finished|completed|is still running)\]",
    r"^\s*◆\s*(?:model|provider|context)\s*:", r"^\s*(?:provider|model)\s+(?:error|diagnostic|quota|authentication|rate limit|retry budget exhausted)\b",
    r"^\s*(?:api\s+)?(?:call|request)\s+failed\b", r"^\s*(?:unhandled|uncaught)\s+(?:gateway\s+)?(?:exception|error)\b",
    r"^\s*http\s+\d{3}:\s*provider\b", r"^\s*(?:memory|user profile)\s+(?:updated|saved)\b",
    r"^\s*self-improvement review completed\b", r"^\s*empty after tools\b",
    r"^\s*(?:all\s+)?(?:api\s+)?(?:provider\s+)?(?:credentials?|tokens?)\s+(?:are\s+)?exhausted\b", r"^\s*token exhaustion\b",
    r"^\s*(?:openai|anthropic|openrouter|google|xai)\b[\s\S]*(?:error|quota|rate limit|authentication|invalid api key)\b",
    r"^\s*⚠️\s*the model produced only internal reasoning and no final answer, despite retries(?: and fallback)?\.\s*its last reasoning",
))


def classify_whatsapp_outbound(
    content: Any,
    metadata: Mapping[str, Any] | None,
    *,
    media: bool = False,
) -> str | None:
    """Return the approved user-visible kind, else ``None``.

    Every WhatsApp wire send must carry an explicit user-visible category.
    ``notify=True`` is the existing gateway final-delivery marker and is
    treated as ``final``.  Unknown and interim/status sends fail closed.
    """
    metadata = metadata or {}
    if metadata.get("_interim_send") or metadata.get("non_conversational"):
        return None
    kind = metadata.get(OUTPUT_KIND_KEY)
    if kind is None and metadata.get("notify") is True:
        kind = "final"
    if kind not in USER_VISIBLE_KINDS:
        return None
    text = _INVISIBLE.sub("", str(content or "")).strip()
    if not text and not media:
        return None
    if text and (text.upper() in _SILENT or any(pattern.search(text) for pattern in _INTERNAL_PAYLOAD_PATTERNS)):
        return None
    return str(kind)


def suppress_whatsapp_outbound(content: Any, metadata: Mapping[str, Any] | None, *, reason: str) -> None:
    """Record a redacted local audit line for an intentionally suppressed send."""
    logger.warning(
        "[whatsapp] suppressed outbound %s (kind=%r, chars=%d)",
        reason,
        (metadata or {}).get(OUTPUT_KIND_KEY),
        len(str(content or "")),
    )
