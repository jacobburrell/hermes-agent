"""Contracts for driver-neutral runtime notice producer types."""

from dataclasses import FrozenInstanceError

import pytest

from agent.runtime_notices import (
    AgentRuntimeNotice,
    FailureCategory,
    NoticeKind,
    provider_terminal_notice,
)


def test_notice_kinds_are_stable_wire_values() -> None:
    assert {kind.value for kind in NoticeKind} == {
        "busy_ack",
        "runtime_progress",
        "runtime_lifecycle",
        "operator_notice",
        "terminal_failure",
    }


def test_runtime_notice_is_frozen() -> None:
    notice = provider_terminal_notice(
        reason="rate_limit",
        message="raw provider detail",
        retryable=True,
    )

    with pytest.raises(FrozenInstanceError):
        notice.message = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("reason", "expected_code", "expected_category"),
    [
        ("auth_permanent", "provider.auth_permanent", FailureCategory.AUTH),
        ("billing", "provider.billing", FailureCategory.BILLING),
        ("rate_limit", "provider.rate_limit", FailureCategory.RATE_LIMIT),
        (
            "content_policy_blocked",
            "provider.content_policy_blocked",
            FailureCategory.CONTENT_POLICY,
        ),
        ("timeout", "provider.timeout", FailureCategory.TIMEOUT),
        (
            "ssl_cert_verification",
            "provider.ssl_cert_verification",
            FailureCategory.TRANSPORT,
        ),
        ("model_not_found", "provider.model_not_found", FailureCategory.PROVIDER),
    ],
)
def test_provider_terminal_notice_uses_structured_reason_only(
    reason: str,
    expected_code: str,
    expected_category: FailureCategory,
) -> None:
    notice = provider_terminal_notice(
        reason=reason,
        message="misleading body: billing timeout auth 429",
        retryable=False,
        provider="example",
        model="model-a",
        diagnostic="private diagnostic",
    )

    assert notice == AgentRuntimeNotice(
        kind=NoticeKind.TERMINAL_FAILURE,
        code=expected_code,
        message="misleading body: billing timeout auth 429",
        failure_category=expected_category,
        retryable=False,
        provider="example",
        model="model-a",
        diagnostic="private diagnostic",
    )


def test_unknown_reason_collapses_to_bounded_code_and_category() -> None:
    notice = provider_terminal_notice(
        reason="vendor-secret-failure-9271",
        message="do not classify this billing-looking body",
        retryable=True,
    )

    assert notice.code == "provider.unknown"
    assert notice.failure_category is FailureCategory.PROVIDER


def test_terminal_failure_requires_category() -> None:
    with pytest.raises(ValueError, match="require failure_category"):
        AgentRuntimeNotice(
            kind=NoticeKind.TERMINAL_FAILURE,
            code="provider.unknown",
            message="failed",
        )


def test_notice_requires_enum_kind_and_nonempty_stable_code() -> None:
    with pytest.raises(TypeError, match="NoticeKind"):
        AgentRuntimeNotice(
            kind="terminal_failure",  # type: ignore[arg-type]
            code="provider.unknown",
            message="failed",
            failure_category=FailureCategory.PROVIDER,
        )
    with pytest.raises(ValueError, match="non-empty stable string"):
        AgentRuntimeNotice(kind=NoticeKind.OPERATOR_NOTICE, code=" ", message="x")
