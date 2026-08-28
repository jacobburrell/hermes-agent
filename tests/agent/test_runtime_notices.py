"""Contracts for driver-neutral runtime notice producer types."""

from dataclasses import FrozenInstanceError

import pytest

from agent.runtime_notices import (
    AgentRuntimeNotice,
    FailureCategory,
    NoticeKind,
    provider_terminal_notice,
)
from agent.error_classifier import ClassifiedError, FailoverReason


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


def test_output_truncation_is_not_claimed_by_provider_terminal_vertical() -> None:
    notice = provider_terminal_notice(
        reason="output_truncated",
        message="first response truncated",
        retryable=False,
    )

    assert notice.code == "provider.unknown"


def test_terminal_failure_requires_category() -> None:
    with pytest.raises(ValueError, match="require failure_category"):
        AgentRuntimeNotice(
            kind=NoticeKind.TERMINAL_FAILURE,
            code="provider.unknown",
            message="failed",
        )


def test_notice_requires_enum_kind_and_bounded_stable_code() -> None:
    with pytest.raises(TypeError, match="NoticeKind"):
        AgentRuntimeNotice(
            kind="terminal_failure",  # type: ignore[arg-type]
            code="provider.unknown",
            message="failed",
            failure_category=FailureCategory.PROVIDER,
        )
    with pytest.raises(ValueError, match="bounded stable identifier"):
        AgentRuntimeNotice(kind=NoticeKind.OPERATOR_NOTICE, code=" ", message="x")
    for code in ("provider.ERROR", "provider.rate-limit", "provider.x\nsecret"):
        with pytest.raises(ValueError, match="bounded stable identifier"):
            AgentRuntimeNotice(
                kind=NoticeKind.OPERATOR_NOTICE,
                code=code,
                message="x",
            )
    with pytest.raises(ValueError, match="bounded stable identifier"):
        AgentRuntimeNotice(
            kind=NoticeKind.OPERATOR_NOTICE,
            code=f"provider.{'x' * 80}",
            message="x",
        )


def test_billing_result_builder_emits_typed_terminal_notice() -> None:
    from agent.conversation_loop import _billing_failure_result

    classified = ClassifiedError(
        reason=FailoverReason.billing,
        retryable=False,
    )
    result = _billing_failure_result(
        classified=classified,
        summary="private provider balance body",
        messages=[],
        api_call_count=3,
        provider="example",
        base_url="https://example.invalid",
        model="model-a",
        guidance="add credits",
    )

    notice = result["runtime_notice"]
    assert notice.kind is NoticeKind.TERMINAL_FAILURE
    assert notice.code == "provider.billing"
    assert notice.failure_category is FailureCategory.BILLING
    assert notice.message == result["final_response"]


def test_content_policy_result_builder_emits_typed_terminal_notice() -> None:
    from agent.conversation_loop import _content_policy_blocked_result

    result = _content_policy_blocked_result(
        [],
        1,
        final_response="raw policy detail",
        error_detail="private refusal",
        provider="example",
        model="model-a",
    )

    notice = result["runtime_notice"]
    assert notice.code == "provider.content_policy_blocked"
    assert notice.failure_category is FailureCategory.CONTENT_POLICY
    assert notice.retryable is False
