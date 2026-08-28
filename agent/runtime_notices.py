"""Driver-neutral runtime notice descriptors.

The agent core owns the semantic classification of a notice.  Delivery
surfaces may decide *where* and *how* to render it, but must never infer a
kind or failure category from user-visible prose.

Keep this module dependency-free: it is imported by the conversation loop,
CLI/TUI consumers, and the messaging gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NoticeKind(str, Enum):
    """Stable semantic classes for non-answer runtime output."""

    BUSY_ACK = "busy_ack"
    RUNTIME_PROGRESS = "runtime_progress"
    RUNTIME_LIFECYCLE = "runtime_lifecycle"
    OPERATOR_NOTICE = "operator_notice"
    TERMINAL_FAILURE = "terminal_failure"


class FailureCategory(str, Enum):
    """Human-safe terminal failure categories.

    These categories intentionally contain no provider response text.  Human
    chat surfaces render a fixed actionable message from this value and the
    stable notice code; programmatic surfaces retain the original raw result.
    """

    AUTH = "auth"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    CONTENT_POLICY = "content_policy"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    PROVIDER = "provider"
    ENDPOINT = "endpoint"
    RUNTIME = "runtime"
    DISK = "disk"
    GATEWAY = "gateway"


_REASON_CATEGORY = {
    "auth": FailureCategory.AUTH,
    "auth_permanent": FailureCategory.AUTH,
    "billing": FailureCategory.BILLING,
    "billing_unverified": FailureCategory.BILLING,
    "rate_limit": FailureCategory.RATE_LIMIT,
    "upstream_rate_limit": FailureCategory.RATE_LIMIT,
    "content_policy_blocked": FailureCategory.CONTENT_POLICY,
    "provider_policy_blocked": FailureCategory.CONTENT_POLICY,
    "timeout": FailureCategory.TIMEOUT,
    "ssl_cert_verification": FailureCategory.TRANSPORT,
}

_STABLE_PROVIDER_REASONS = frozenset(
    {
        "auth",
        "auth_permanent",
        "billing",
        "billing_unverified",
        "rate_limit",
        "upstream_rate_limit",
        "overloaded",
        "server_error",
        "timeout",
        "ssl_cert_verification",
        "model_not_found",
        "provider_policy_blocked",
        "content_policy_blocked",
        "format_error",
        "invalid_response",
        "output_truncated",
        "unknown",
    }
)


@dataclass(frozen=True, slots=True)
class AgentRuntimeNotice:
    """Immutable semantic notice emitted by an agent/runtime producer.

    ``message`` and ``diagnostic`` are never classification inputs.  The
    gateway may pass them through only on the explicit raw programmatic
    surfaces; human terminal copy is selected from ``failure_category`` and
    ``code``.
    """

    kind: NoticeKind
    code: str
    message: str
    failure_category: Optional[FailureCategory] = None
    retryable: Optional[bool] = None
    provider: str = ""
    model: str = ""
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NoticeKind):
            raise TypeError("kind must be a NoticeKind")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("notice code must be a non-empty stable string")
        if self.kind is NoticeKind.TERMINAL_FAILURE and self.failure_category is None:
            raise ValueError("terminal_failure notices require failure_category")


def provider_terminal_notice(
    *,
    reason: str,
    message: str,
    retryable: bool,
    provider: str = "",
    model: str = "",
    diagnostic: str = "",
) -> AgentRuntimeNotice:
    """Build a structured provider terminal notice from classifier output.

    ``reason`` is the machine-readable ``FailoverReason.value`` (or one of the
    small synthetic producer reasons above), never text parsed from an error
    body.  Unknown future reasons collapse to the stable ``provider.unknown``
    contract instead of leaking an unbounded provider value into metrics.
    """

    normalized = str(reason or "unknown").strip().lower()
    if normalized not in _STABLE_PROVIDER_REASONS:
        normalized = "unknown"
    category = _REASON_CATEGORY.get(normalized, FailureCategory.PROVIDER)
    return AgentRuntimeNotice(
        kind=NoticeKind.TERMINAL_FAILURE,
        code=f"provider.{normalized}",
        message=str(message or ""),
        failure_category=category,
        retryable=bool(retryable),
        provider=str(provider or ""),
        model=str(model or ""),
        diagnostic=str(diagnostic or ""),
    )


__all__ = [
    "AgentRuntimeNotice",
    "FailureCategory",
    "NoticeKind",
    "provider_terminal_notice",
]
