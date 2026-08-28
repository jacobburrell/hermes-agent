"""Typed runtime-notice policy and human-chat delivery.

Producers provide :class:`agent.runtime_notices.AgentRuntimeNotice`; this
module owns gateway routing, policy, canonical human copy, and durable
at-most-once delivery.  User-visible text is never used to infer a notice kind
or failure category.

Raw/programmatic platforms keep their existing result rails.  Callers must not
route LOCAL, API_SERVER, WEBHOOK, or MSGRAPH_WEBHOOK responses through this
helper.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agent.runtime_notices import AgentRuntimeNotice, FailureCategory, NoticeKind
from gateway.runtime_notice_ledger import (
    ReserveOutcome,
    TerminalNoticeLedger,
)

logger = logging.getLogger(__name__)

RAW_RUNTIME_NOTICE_PLATFORMS = frozenset(
    {"local", "api_server", "webhook", "msgraph_webhook"}
)

_TRANSIENT_WHATSAPP_KINDS = frozenset(
    {
        NoticeKind.BUSY_ACK,
        NoticeKind.RUNTIME_PROGRESS,
        NoticeKind.RUNTIME_LIFECYCLE,
        NoticeKind.OPERATOR_NOTICE,
    }
)


class DeliveryOutcome(str, Enum):
    SENT = "sent"
    SUPPRESSED = "suppressed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    ALREADY_DELIVERED = "already_delivered"


@dataclass(frozen=True, slots=True)
class GatewayNoticeEnvelope:
    notice: AgentRuntimeNotice
    session_key: str
    run_id: str
    platform: str
    chat_type: str
    legacy_alias: Optional[str] = None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1", "enabled"}:
            return True
        if normalized in {"false", "no", "off", "0", "disabled"}:
            return False
    return None


def _dict_at(value: Any, *parts: str) -> dict:
    current = value
    for part in parts:
        if not isinstance(current, dict):
            return {}
        current = current.get(part)
    return current if isinstance(current, dict) else {}


def resolve_runtime_notice_enabled(
    config: dict,
    envelope: GatewayNoticeEnvelope,
    *,
    default: bool,
) -> bool:
    """Resolve one notice policy using the canonical presence-sensitive order.

    Order: chat exact, chat umbrella, platform exact, platform umbrella,
    global exact, global umbrella, this envelope's legacy alias, built-in.
    ``terminal_failure`` is a safety rail and always resolves true. Explicit
    false values are logged as invalid and ignored at every scope.
    """

    display = config.get("display") if isinstance(config, dict) else {}
    if not isinstance(display, dict):
        display = {}
    platform = str(envelope.platform or "").strip().lower()
    chat_type = str(envelope.chat_type or "").strip().lower()
    kind = envelope.notice.kind.value
    platform_cfg = _dict_at(display, "platforms", platform)
    chat_cfg = _dict_at(platform_cfg, "chat_types", chat_type)

    candidates: list[tuple[str, Any]] = [
        ("chat_exact", _dict_at(chat_cfg, "runtime_notice_kinds").get(kind)),
        ("chat_umbrella", chat_cfg.get("runtime_notices")),
        ("platform_exact", _dict_at(platform_cfg, "runtime_notice_kinds").get(kind)),
        ("platform_umbrella", platform_cfg.get("runtime_notices")),
        ("global_exact", _dict_at(display, "runtime_notice_kinds").get(kind)),
        ("global_umbrella", display.get("runtime_notices")),
    ]
    if envelope.legacy_alias:
        candidates.append(("legacy_alias", display.get(envelope.legacy_alias)))

    terminal = envelope.notice.kind is NoticeKind.TERMINAL_FAILURE
    for scope, raw in candidates:
        if raw is None:
            continue
        parsed = _as_bool(raw)
        if parsed is None:
            logger.warning(
                "runtime notice policy ignored invalid boolean: kind=%s scope=%s",
                kind,
                scope,
            )
            continue
        if terminal and not parsed:
            logger.warning(
                "runtime notice policy cannot disable terminal failure: scope=%s",
                scope,
            )
            continue
        return parsed
    if terminal:
        return True
    return bool(default)


@dataclass(frozen=True, slots=True)
class LoadedNoticeConfig:
    config: dict
    valid: bool
    from_last_known_good: bool = False


class RuntimeNoticeConfigStore:
    """Per-path live raw-config reader with last-known-good retention."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_good: dict[str, dict] = {}

    def load(self, profile_home: Path) -> LoadedNoticeConfig:
        path = Path(profile_home) / "config.yaml"
        user, user_valid, user_present = self._read_mapping(path)
        managed, managed_valid, managed_present = self._read_managed_mapping()

        key = str(path.resolve(strict=False))
        # Missing user config is a valid absent layer when an administrator
        # supplies a managed mapping. A present malformed/non-mapping user
        # layer, or malformed managed layer, must retain the prior COMBINED
        # policy rather than rebuilding from the other valid layer alone.
        invalid_user_layer = user_present and not user_valid
        invalid_managed_layer = managed_present and not managed_valid
        current_valid = (
            not invalid_user_layer
            and not invalid_managed_layer
            and (user_valid or (managed_present and managed_valid))
        )
        if not current_valid:
            with self._lock:
                previous = copy.deepcopy(self._last_good.get(key))
            if previous is not None:
                return LoadedNoticeConfig(previous, True, True)
            return LoadedNoticeConfig({}, False)

        merged = user if user_valid else {}
        if managed_valid:
            try:
                from hermes_cli.config import _deep_merge, _expand_env_vars

                merged = _expand_env_vars(merged)
                merged = _deep_merge(merged, _expand_env_vars(managed))
            except Exception as exc:
                logger.warning(
                    "runtime notice config overlay failed: source=managed error_class=%s",
                    type(exc).__name__[:80],
                )
                with self._lock:
                    previous = copy.deepcopy(self._last_good.get(key))
                if previous is not None:
                    return LoadedNoticeConfig(previous, True, True)
                return LoadedNoticeConfig({}, False)
        else:
            try:
                from hermes_cli.config import _expand_env_vars

                merged = _expand_env_vars(merged)
            except Exception:
                pass
        with self._lock:
            self._last_good[key] = copy.deepcopy(merged)
        return LoadedNoticeConfig(merged, True)

    @staticmethod
    def _read_mapping(path: Path) -> tuple[dict, bool, bool]:
        try:
            from hermes_cli.config import read_user_config_raw

            present = path.is_file()
            parsed, valid = read_user_config_raw(path, return_validity=True)
            return parsed, valid, present
        except Exception as exc:
            logger.warning(
                "runtime notice config read failed: source=user error_class=%s",
                type(exc).__name__[:80],
            )
            return {}, False, path.is_file()

    @staticmethod
    def _read_managed_mapping() -> tuple[dict, bool, bool]:
        try:
            from hermes_cli.managed_scope import get_managed_dir

            managed_dir = get_managed_dir()
            if managed_dir is None:
                return {}, True, False
            path = Path(managed_dir) / "config.yaml"
            if not path.is_file():
                return {}, True, False
            from hermes_cli.config import read_user_config_raw

            parsed, valid = read_user_config_raw(path, return_validity=True)
            return parsed, valid, True
        except Exception as exc:
            logger.warning(
                "runtime notice config read failed: source=managed error_class=%s",
                type(exc).__name__[:80],
            )
            return {}, False, True


_HUMAN_TERMINAL_COPY = {
    FailureCategory.AUTH: (
        "⚠️ The model provider rejected authentication. Check this profile's "
        "API key, then try again."
    ),
    FailureCategory.BILLING: (
        "⚠️ The model provider reported a billing or credit limit. Add credits "
        "or switch providers with /model, then try again."
    ),
    FailureCategory.RATE_LIMIT: (
        "⚠️ The model provider is rate-limiting requests. Try again shortly or "
        "switch providers with /model."
    ),
    FailureCategory.CONTENT_POLICY: (
        "⚠️ The model provider declined this request under its content policy. "
        "Revise the request and try again."
    ),
    FailureCategory.TIMEOUT: (
        "⚠️ The model provider timed out before completing the response. Try "
        "again shortly."
    ),
    FailureCategory.TRANSPORT: (
        "⚠️ Hermes could not complete the connection to the model provider. "
        "Check the provider endpoint and try again."
    ),
    FailureCategory.ENDPOINT: (
        "⚠️ Hermes could not reach the configured model endpoint. Check the "
        "endpoint and try again."
    ),
    FailureCategory.DISK: (
        "⚠️ Hermes could not complete the turn because local storage is full. "
        "Free disk space, then try again."
    ),
    FailureCategory.RUNTIME: (
        "⚠️ Hermes could not initialize the model runtime. Check the local "
        "runtime configuration, then try again."
    ),
    FailureCategory.GATEWAY: (
        "⚠️ Hermes encountered a gateway error before completing the turn. Try "
        "again or use /reset to start a fresh session."
    ),
    FailureCategory.PROVIDER: (
        "⚠️ The model provider could not complete the response. Try again "
        "shortly or switch providers with /model."
    ),
}


def canonical_human_terminal_content(notice: AgentRuntimeNotice) -> str:
    if notice.kind is not NoticeKind.TERMINAL_FAILURE:
        raise ValueError("canonical terminal copy requires terminal_failure")
    return _HUMAN_TERMINAL_COPY.get(
        notice.failure_category or FailureCategory.PROVIDER,
        _HUMAN_TERMINAL_COPY[FailureCategory.PROVIDER],
    )


def notice_run_id(event: Any, session_key: str) -> str:
    """Return a genuinely durable identity for one logical inbound turn.

    Provider message ids are the strongest identity. Events without one must
    already carry the persisted admission/session turn id installed by the
    runner. ``MessageEvent.timestamp`` is deliberately ignored: its default is
    local arrival time and changes when an adapter reconstructs the event.
    """

    message_id = str(getattr(event, "message_id", "") or "").strip()
    if message_id:
        return hashlib.sha256(
            f"{session_key}|{message_id}".encode("utf-8", "replace")
        ).hexdigest()[:32]

    existing = str(getattr(event, "_runtime_notice_run_id", "") or "").strip()
    if existing:
        return existing
    metadata = getattr(event, "metadata", None)
    durable_hint = (
        str(metadata.get("runtime_notice_turn_id") or "").strip()
        if isinstance(metadata, dict)
        else ""
    )
    if not durable_hint:
        raise ValueError("no durable identity for message-id-free runtime notice")
    return hashlib.sha256(
        f"{session_key}|durable|{durable_hint}".encode("utf-8", "replace")
    ).hexdigest()[:32]


def transient_notice_default(platform: str, kind: NoticeKind) -> bool:
    if platform in {"whatsapp", "whatsapp_cloud"} and kind in _TRANSIENT_WHATSAPP_KINDS:
        return False
    return True


def _adapter_guarantees_no_delivery(adapter: Any, result: Any) -> bool:
    """True only under an explicit adapter-owned definite-failure contract."""

    checker = getattr(adapter, "runtime_notice_definitely_not_delivered", None)
    if not callable(checker):
        return False
    try:
        return checker(result) is True
    except Exception:
        return False


async def _best_effort_post_call_transition(method: Any, *args: Any) -> None:
    """Never turn a completed/ambiguous transport call into a resend path."""

    try:
        await asyncio.to_thread(method, *args)
    except Exception as exc:
        logger.error(
            "runtime notice ledger post-call transition failed: error_class=%s",
            type(exc).__name__[:80],
        )


async def deliver_human_runtime_notice(
    *,
    runner: Any,
    source: Any,
    envelope: GatewayNoticeEnvelope,
    profile_home: Path,
    config_store: RuntimeNoticeConfigStore,
) -> DeliveryOutcome:
    """Deliver one notice through the runner's live adapter without retries.

    A failed/raised/cancelled adapter call is ``AMBIGUOUS`` by default.  It is
    never retried by this helper or after restart.  Only an adapter's explicit
    ``runtime_notice_definitely_not_delivered(result)`` guarantee permits the
    reservation to be released after a returned failure.

    If the bounded ledger contains only protected tombstones, delivery fails
    before the adapter call and the caller uses its canonical final rail. This
    preserves terminal visibility but temporarily degrades durable dedupe; the
    structured ``runtime_notice_dedupe_degraded`` metric log records that state.
    """

    platform = str(envelope.platform or "").strip().lower()
    if platform in RAW_RUNTIME_NOTICE_PLATFORMS:
        raise ValueError("programmatic platforms retain their existing result rails")

    loaded = await asyncio.to_thread(config_store.load, profile_home)
    enabled = resolve_runtime_notice_enabled(
        loaded.config,
        envelope,
        default=transient_notice_default(platform, envelope.notice.kind),
    )
    # Terminal failures remain enabled even when source config is absent,
    # malformed, or unserved. Non-terminal WhatsApp notices fail closed before
    # the first valid read; future verticals consume this same policy contract.
    if envelope.notice.kind is not NoticeKind.TERMINAL_FAILURE:
        if not loaded.valid and platform in {"whatsapp", "whatsapp_cloud"}:
            enabled = False
        if not enabled:
            return DeliveryOutcome.SUPPRESSED

    content = canonical_human_terminal_content(envelope.notice)
    # Resolve the transport before creating durable suppression state. A
    # missing adapter is a definite pre-wire failure and must fall back through
    # the caller's existing final-response rail rather than tombstone silence.
    adapter = runner._adapter_for_source(source)
    if adapter is None:
        logger.warning(
            "runtime notice delivery: kind=%s code=%s platform=%s outcome=%s",
            envelope.notice.kind.value,
            envelope.notice.code,
            platform,
            DeliveryOutcome.FAILED.value,
        )
        return DeliveryOutcome.FAILED

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    ledger = TerminalNoticeLedger(profile_home)
    reservation = await asyncio.to_thread(
        ledger.reserve,
        session_key=envelope.session_key,
        run_id=envelope.run_id,
        code=envelope.notice.code,
        content=content,
        content_hash=content_hash,
    )
    if reservation.outcome is ReserveOutcome.ALREADY_DELIVERED:
        logger.info(
            "runtime notice delivery: kind=%s code=%s platform=%s outcome=%s",
            envelope.notice.kind.value,
            envelope.notice.code,
            platform,
            DeliveryOutcome.ALREADY_DELIVERED.value,
        )
        return DeliveryOutcome.ALREADY_DELIVERED
    if reservation.outcome is ReserveOutcome.CAPACITY_EXHAUSTED:
        logger.warning(
            "runtime notice delivery: kind=%s code=%s platform=%s outcome=%s "
            "reason=ledger_capacity metric=runtime_notice_dedupe_degraded "
            "fallback=canonical_final",
            envelope.notice.kind.value,
            envelope.notice.code,
            platform,
            DeliveryOutcome.FAILED.value,
        )
        return DeliveryOutcome.FAILED

    owner = reservation.owner_token
    applied = await asyncio.to_thread(
        ledger.mark_state_applied,
        envelope.session_key,
        envelope.run_id,
        envelope.notice.code,
        owner,
    )
    if not applied:
        await asyncio.to_thread(
            ledger.release_reserved,
            envelope.session_key,
            envelope.run_id,
            envelope.notice.code,
            owner,
        )
        return DeliveryOutcome.FAILED
    started = await asyncio.to_thread(
        ledger.mark_in_flight,
        envelope.session_key,
        envelope.run_id,
        envelope.notice.code,
        owner,
    )
    if not started:
        await asyncio.to_thread(
            ledger.release_reserved,
            envelope.session_key,
            envelope.run_id,
            envelope.notice.code,
            owner,
        )
        return DeliveryOutcome.FAILED

    metadata = runner._thread_metadata_for_source(source)
    try:
        result = await adapter.send(source.chat_id, content, metadata=metadata)
    except BaseException:
        await _best_effort_post_call_transition(
            ledger.mark_ambiguous,
            envelope.session_key,
            envelope.run_id,
            envelope.notice.code,
            owner,
        )
        logger.warning(
            "runtime notice delivery: kind=%s code=%s platform=%s outcome=%s",
            envelope.notice.kind.value,
            envelope.notice.code,
            platform,
            DeliveryOutcome.AMBIGUOUS.value,
        )
        return DeliveryOutcome.AMBIGUOUS

    if bool(getattr(result, "success", False)):
        await _best_effort_post_call_transition(
            ledger.mark_sent,
            envelope.session_key,
            envelope.run_id,
            envelope.notice.code,
            owner,
        )
        outcome = DeliveryOutcome.SENT
    elif _adapter_guarantees_no_delivery(adapter, result):
        await _best_effort_post_call_transition(
            ledger.release_definite_failure,
            envelope.session_key,
            envelope.run_id,
            envelope.notice.code,
            owner,
        )
        outcome = DeliveryOutcome.FAILED
    else:
        await _best_effort_post_call_transition(
            ledger.mark_ambiguous,
            envelope.session_key,
            envelope.run_id,
            envelope.notice.code,
            owner,
        )
        outcome = DeliveryOutcome.AMBIGUOUS
    logger.info(
        "runtime notice delivery: kind=%s code=%s platform=%s outcome=%s",
        envelope.notice.kind.value,
        envelope.notice.code,
        platform,
        outcome.value,
    )
    return outcome


__all__ = [
    "DeliveryOutcome",
    "GatewayNoticeEnvelope",
    "LoadedNoticeConfig",
    "RAW_RUNTIME_NOTICE_PLATFORMS",
    "RuntimeNoticeConfigStore",
    "canonical_human_terminal_content",
    "deliver_human_runtime_notice",
    "notice_run_id",
    "resolve_runtime_notice_enabled",
]
