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
_INTERNAL_PAYLOAD_RE = re.compile(
    r"(?:^|\n)\s*(?:traceback \(most recent call last\)|file \"[^\"]+\", line \d+|"
    r"stack trace:|gateway (?:restart|started|stopped|shutdown)|"
    r"goal (?:status|continuation|persistence)|internal (?:status|error|diagnostic)|"
    r"hermes(?: agent)? (?:status|diagnostic)|tool progress:)",
    re.IGNORECASE,
)


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
    text = str(content or "").strip()
    if not text and not media:
        return None
    if text and _INTERNAL_PAYLOAD_RE.search(text):
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
