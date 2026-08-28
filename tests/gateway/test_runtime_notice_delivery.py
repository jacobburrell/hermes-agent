"""Policy, durability, and transport contracts for typed runtime notices."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.runtime_notices import FailureCategory, NoticeKind, provider_terminal_notice
from gateway.runtime_notice_delivery import (
    DeliveryOutcome,
    GatewayNoticeEnvelope,
    RuntimeNoticeConfigStore,
    canonical_human_terminal_content,
    deliver_human_runtime_notice,
    resolve_runtime_notice_enabled,
)
from gateway.runtime_notice_ledger import (
    ReservationState,
    ReserveOutcome,
    TerminalNoticeLedger,
)


def _notice(reason: str = "rate_limit", message: str = "private provider body"):
    return provider_terminal_notice(
        reason=reason,
        message=message,
        retryable=False,
        provider="provider-secret",
        model="model-secret",
        diagnostic="request-secret",
    )


def _envelope(
    *,
    kind_notice=None,
    platform: str = "whatsapp",
    chat_type: str = "group",
    run_id: str = "run-1",
    legacy_alias: str | None = None,
) -> GatewayNoticeEnvelope:
    return GatewayNoticeEnvelope(
        notice=kind_notice or _notice(),
        session_key="session-1",
        run_id=run_id,
        platform=platform,
        chat_type=chat_type,
        legacy_alias=legacy_alias,
    )


@pytest.mark.parametrize(
    "config",
    [
        {"display": {"runtime_notice_kinds": {"terminal_failure": False}}},
        {"display": {"runtime_notices": False}},
        {
            "display": {
                "platforms": {
                    "whatsapp": {
                        "runtime_notice_kinds": {"terminal_failure": False}
                    }
                }
            }
        },
        {
            "display": {
                "platforms": {
                    "whatsapp": {
                        "runtime_notices": False,
                        "chat_types": {
                            "group": {
                                "runtime_notice_kinds": {"terminal_failure": False},
                                "runtime_notices": False,
                            }
                        },
                    }
                }
            }
        },
        {"display": {"legacy_provider_failures": False}},
    ],
)
def test_terminal_failure_cannot_be_disabled_at_any_policy_scope(config) -> None:
    envelope = _envelope(legacy_alias="legacy_provider_failures")

    assert resolve_runtime_notice_enabled(config, envelope, default=False) is True


def test_nonterminal_policy_uses_exact_canonical_precedence() -> None:
    progress = _notice().__class__(
        kind=NoticeKind.RUNTIME_PROGRESS,
        code="provider.retrying",
        message="retrying",
    )
    envelope = _envelope(kind_notice=progress)
    config = {
        "display": {
            "runtime_notices": False,
            "runtime_notice_kinds": {"runtime_progress": False},
            "platforms": {
                "whatsapp": {
                    "runtime_notices": False,
                    "runtime_notice_kinds": {"runtime_progress": False},
                    "chat_types": {
                        "group": {
                            "runtime_notices": False,
                            "runtime_notice_kinds": {"runtime_progress": True},
                        }
                    },
                }
            },
        }
    }

    assert resolve_runtime_notice_enabled(config, envelope, default=False) is True


def test_config_store_retains_last_good_after_user_yaml_breaks(tmp_path: Path) -> None:
    home = tmp_path / "profile"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text("display:\n  runtime_notices: false\n", encoding="utf-8")
    store = RuntimeNoticeConfigStore()

    first = store.load(home)
    config_path.write_text("display: [\n", encoding="utf-8")
    second = store.load(home)

    assert first.valid and not first.from_last_known_good
    assert second.valid and second.from_last_known_good
    assert second.config == first.config


def test_managed_overlay_wins_and_malformed_update_retains_last_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "profile"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        "display:\n  runtime_notices: false\n", encoding="utf-8"
    )
    managed_path = managed / "config.yaml"
    managed_path.write_text(
        "display:\n  runtime_notices: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    store = RuntimeNoticeConfigStore()

    first = store.load(home)
    managed_path.write_text("display: [\n", encoding="utf-8")
    second = store.load(home)

    assert first.config["display"]["runtime_notices"] is True
    assert second.from_last_known_good
    assert second.config == first.config


def test_missing_and_non_mapping_config_are_invalid_before_first_good(
    tmp_path: Path,
) -> None:
    home = tmp_path / "profile"
    home.mkdir()
    store = RuntimeNoticeConfigStore()

    assert not store.load(home).valid
    (home / "config.yaml").write_text("- not\n- a mapping\n", encoding="utf-8")
    assert not store.load(home).valid
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    assert store.load(home).valid


def test_ledger_concurrent_reservation_and_new_run(tmp_path: Path) -> None:
    ledger_a = TerminalNoticeLedger(tmp_path)
    ledger_b = TerminalNoticeLedger(tmp_path)

    def reserve(ledger: TerminalNoticeLedger, run_id: str):
        return ledger.reserve(
            session_key="session",
            run_id=run_id,
            code="provider.rate_limit",
            content="canonical",
            content_hash="hash",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda ledger: reserve(ledger, "run-1"), [ledger_a, ledger_b]))

    assert sorted(result.outcome.value for result in results) == [
        ReserveOutcome.ACQUIRED.value,
        ReserveOutcome.ALREADY_DELIVERED.value,
    ]
    assert reserve(ledger_b, "run-2").outcome is ReserveOutcome.ACQUIRED


def test_ledger_sent_and_ambiguous_are_durable_tombstones(tmp_path: Path) -> None:
    ledger = TerminalNoticeLedger(tmp_path)
    reservation = ledger.reserve(
        session_key="session",
        run_id="run-sent",
        code="provider.timeout",
        content="canonical",
        content_hash="hash",
    )
    assert ledger.mark_state_applied(
        "session", "run-sent", "provider.timeout", reservation.owner_token
    )
    assert ledger.mark_in_flight(
        "session", "run-sent", "provider.timeout", reservation.owner_token
    )
    assert ledger.mark_sent(
        "session", "run-sent", "provider.timeout", reservation.owner_token
    )
    assert (
        TerminalNoticeLedger(tmp_path)
        .reserve(
            session_key="session",
            run_id="run-sent",
            code="provider.timeout",
            content="canonical",
            content_hash="hash",
        )
        .outcome
        is ReserveOutcome.ALREADY_DELIVERED
    )

    ambiguous = ledger.reserve(
        session_key="session",
        run_id="run-ambiguous",
        code="provider.timeout",
        content="canonical",
        content_hash="hash",
    )
    assert ledger.mark_in_flight(
        "session", "run-ambiguous", "provider.timeout", ambiguous.owner_token
    )
    assert ledger.mark_ambiguous(
        "session", "run-ambiguous", "provider.timeout", ambiguous.owner_token
    )
    assert ledger.get_state("session", "run-ambiguous", "provider.timeout") == (
        ReservationState.AMBIGUOUS.value
    )


class _Adapter:
    def __init__(self, result=None, exc: BaseException | None = None):
        self.result = result or SimpleNamespace(success=True)
        self.exc = exc
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append((chat_id, content, metadata))
        if self.exc:
            raise self.exc
        return self.result


class _Runner:
    def __init__(self, adapter):
        self.adapter = adapter

    def _adapter_for_source(self, source):
        return self.adapter

    def _thread_metadata_for_source(self, source):
        return {"thread_id": source.thread_id}


@pytest.mark.asyncio
async def test_delivery_sends_sanitized_content_once_and_tombstones(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    adapter = _Adapter()
    runner = _Runner(adapter)
    source = SimpleNamespace(chat_id="chat-secret", thread_id="thread-1")
    envelope = _envelope()
    store = RuntimeNoticeConfigStore()

    first = await deliver_human_runtime_notice(
        runner=runner,
        source=source,
        envelope=envelope,
        profile_home=tmp_path,
        config_store=store,
    )
    second = await deliver_human_runtime_notice(
        runner=runner,
        source=source,
        envelope=envelope,
        profile_home=tmp_path,
        config_store=store,
    )

    assert first is DeliveryOutcome.SENT
    assert second is DeliveryOutcome.ALREADY_DELIVERED
    assert len(adapter.calls) == 1
    assert adapter.calls[0][1] == canonical_human_terminal_content(envelope.notice)
    assert "private provider body" not in adapter.calls[0][1]


@pytest.mark.asyncio
async def test_post_call_failure_is_ambiguous_and_never_retried(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    adapter = _Adapter(result=SimpleNamespace(success=False, error="private failure"))
    runner = _Runner(adapter)
    source = SimpleNamespace(chat_id="chat", thread_id=None)
    envelope = _envelope(run_id="ambiguous")
    store = RuntimeNoticeConfigStore()

    assert (
        await deliver_human_runtime_notice(
            runner=runner,
            source=source,
            envelope=envelope,
            profile_home=tmp_path,
            config_store=store,
        )
        is DeliveryOutcome.AMBIGUOUS
    )
    assert (
        await deliver_human_runtime_notice(
            runner=runner,
            source=source,
            envelope=envelope,
            profile_home=tmp_path,
            config_store=store,
        )
        is DeliveryOutcome.ALREADY_DELIVERED
    )
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_runtime_notice_logs_exclude_bodies_and_chat_ids(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    adapter = _Adapter()
    source = SimpleNamespace(chat_id="private-chat-id", thread_id=None)

    with caplog.at_level("INFO", logger="gateway.runtime_notice_delivery"):
        await deliver_human_runtime_notice(
            runner=_Runner(adapter),
            source=source,
            envelope=_envelope(),
            profile_home=tmp_path,
            config_store=RuntimeNoticeConfigStore(),
        )

    log_text = caplog.text
    assert "private-chat-id" not in log_text
    assert "private provider body" not in log_text
    assert "request-secret" not in log_text
    assert "provider.rate_limit" in log_text


@pytest.mark.asyncio
@pytest.mark.parametrize("config_text", [None, "display: [\n", "- not\n- mapping\n"])
async def test_terminal_delivery_is_enabled_before_any_valid_config_read(
    tmp_path: Path, config_text: str | None
) -> None:
    if config_text is not None:
        (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    adapter = _Adapter()

    outcome = await deliver_human_runtime_notice(
        runner=_Runner(adapter),
        source=SimpleNamespace(chat_id="chat", thread_id=None),
        envelope=_envelope(run_id=f"invalid-{config_text!r}"),
        profile_home=tmp_path,
        config_store=RuntimeNoticeConfigStore(),
    )

    assert outcome is DeliveryOutcome.SENT
    assert len(adapter.calls) == 1


def test_profile_config_last_good_isolated_by_profile_path(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.yaml").write_text(
        "display:\n  runtime_notices: false\n", encoding="utf-8"
    )
    (second / "config.yaml").write_text(
        "display:\n  runtime_notices: true\n", encoding="utf-8"
    )
    store = RuntimeNoticeConfigStore()

    assert store.load(first).config["display"]["runtime_notices"] is False
    assert store.load(second).config["display"]["runtime_notices"] is True
    (first / "config.yaml").write_text("display: [\n", encoding="utf-8")

    assert store.load(first).config["display"]["runtime_notices"] is False
    assert store.load(second).config["display"]["runtime_notices"] is True


def test_dead_inflight_owner_becomes_ambiguous_without_retry(tmp_path: Path) -> None:
    ledger = TerminalNoticeLedger(tmp_path)
    reservation = ledger.reserve(
        session_key="session",
        run_id="crashed-run",
        code="provider.timeout",
        content="canonical",
        content_hash="hash",
    )
    assert ledger.mark_in_flight(
        "session", "crashed-run", "provider.timeout", reservation.owner_token
    )
    conn = ledger._connect()
    try:
        conn.execute(
            """UPDATE terminal_notice_deliveries
               SET owner_pid=-1, owner_started_at=-1
               WHERE session_key='session' AND run_id='crashed-run'"""
        )
    finally:
        conn.close()

    after_restart = TerminalNoticeLedger(tmp_path).reserve(
        session_key="session",
        run_id="crashed-run",
        code="provider.timeout",
        content="canonical",
        content_hash="hash",
    )

    assert after_restart.outcome is ReserveOutcome.ALREADY_DELIVERED
    assert ledger.get_state("session", "crashed-run", "provider.timeout") == (
        ReservationState.AMBIGUOUS.value
    )
