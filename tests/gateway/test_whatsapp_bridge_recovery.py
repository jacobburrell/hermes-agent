"""Restart recovery for ACKed WhatsApp bridge handoffs.

The bridge spool is gone after ACK.  These tests exercise the narrow
archive -> ordinary final delivery ledger recovery route and, critically,
prove it never calls the model-facing inbound handler again.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.whatsapp.inbound_archive import WhatsAppInboundArchive
from tests.gateway.restart_test_helpers import make_restart_runner


_CLARIFICATION = "I was interrupted before I could finish that request. Please send it again."


class _RecoveryAdapter(BasePlatformAdapter):
    def __init__(self, archive):
        super().__init__(PlatformConfig(enabled=True), Platform.WHATSAPP)
        self._archive = archive
        self._owner_profile = "jackwhatsapp"
        self.sent = []
        self.handle_message = AsyncMock()

    def _inbound_archive_instance(self):
        return self._archive

    async def connect(self, *, is_reconnect=False):  # pragma: no cover - unused
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="wa-final")


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _raw(*, message_id="inbound-1", group=False):
    chat_id = "120363000000000001@g.us" if group else "15551230000@s.whatsapp.net"
    return {
        "messageId": message_id, "chatId": chat_id,
        "senderId": "15551230000@s.whatsapp.net", "body": "original request",
        "timestamp": 1, "hasMedia": False, "isGroup": group,
    }


def _acked_archive(tmp_path, *, delivery_id="a" * 64, group=False, admission="operate"):
    home = tmp_path / "profile"
    (home / "cache").mkdir(parents=True)
    archive = WhatsAppInboundArchive(
        home / "whatsapp" / "inbound-archive-v1", home, home / "cache",
    )
    event_id, accepted = archive.record(_raw(group=group), admission)
    assert accepted
    lease = {
        "deliveryId": delivery_id, "eventDigest": "b" * 64,
        "consumerId": "consumer", "epoch": 1, "token": "token",
    }
    receipt = archive.bind_bridge_receipt(event_id, lease)
    receipt = archive.mark_bridge_receipt_ready(receipt)
    receipt = archive.prepare_bridge_handoff(receipt)
    assert archive.mark_bridge_receipt_acked(receipt)
    return archive


def _runner(archive):
    adapter = _RecoveryAdapter(archive)
    runner, _ = make_restart_runner(adapter)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner._profile_adapters = {"jackwhatsapp": {Platform.WHATSAPP: adapter}}
    adapter.gateway_runner = runner
    return runner, adapter


def _recovery_and_obligation(runner, archive, delivery_id="a" * 64):
    recovery = archive.reserve_bridge_recovery(delivery_id)
    assert recovery is not None and not recovery.is_group
    source = runner._bridge_recovery_source(runner.adapters[Platform.WHATSAPP], recovery)
    from gateway.session import build_session_key

    key = build_session_key(source, profile=source.profile)
    obligation_id = dl.compute_obligation_id(key, recovery.delivery_id, _CLARIFICATION)
    assert archive.register_bridge_recovery_delivery(recovery, obligation_id)
    return recovery, obligation_id, key


@pytest.mark.asyncio
async def test_direct_acked_handoff_sends_one_plain_final_without_inbound_handler(tmp_path):
    archive = _acked_archive(tmp_path)
    runner, adapter = _runner(archive)

    await runner._recover_pending_whatsapp_bridge_handoffs()

    assert [item[1] for item in adapter.sent] == [_CLARIFICATION]
    adapter.handle_message.assert_not_awaited()
    assert archive.pending_bridge_recoveries() == []
    with dl._connect() as db:
        assert db.execute("SELECT state FROM delivery_obligations").fetchone()[0] == "delivered"


@pytest.mark.asyncio
async def test_group_handoff_is_silent_and_settled(tmp_path):
    archive = _acked_archive(tmp_path, group=True)
    runner, adapter = _runner(archive)

    await runner._recover_pending_whatsapp_bridge_handoffs()

    assert adapter.sent == []
    adapter.handle_message.assert_not_awaited()
    assert archive.pending_bridge_recoveries() == []


@pytest.mark.asyncio
async def test_observed_handoff_stays_durable_at_ingress_then_settles_silently(tmp_path):
    archive = _acked_archive(tmp_path, admission="observe")
    # The ingress layer must retain its post-ACK fence even for ambient
    # traffic.  The startup layer—not a model turn—owns silent disposition.
    assert len(archive.pending_bridge_recoveries()) == 1
    runner, adapter = _runner(archive)

    await runner._recover_pending_whatsapp_bridge_handoffs()

    assert adapter.sent == []
    adapter.handle_message.assert_not_awaited()
    assert archive.pending_bridge_recoveries() == []


@pytest.mark.asyncio
async def test_delivered_ledger_row_settles_archive_without_second_send(tmp_path):
    archive = _acked_archive(tmp_path)
    runner, adapter = _runner(archive)
    recovery, obligation_id, key = _recovery_and_obligation(runner, archive)
    dl.record_obligation(
        obligation_id=obligation_id, session_key=key, platform="whatsapp",
        chat_id=recovery.chat_id, thread_id=None, content=_CLARIFICATION,
        adapter_profile="jackwhatsapp", bridge_recovery_delivery_id=recovery.delivery_id,
        bridge_recovery_generation=recovery.generation,
    )
    dl.mark_delivered(obligation_id)

    await runner._recover_pending_whatsapp_bridge_handoffs()

    assert adapter.sent == []
    assert archive.pending_bridge_recoveries() == []


@pytest.mark.asyncio
async def test_pending_ledger_row_sends_its_exact_existing_clarification_once(tmp_path):
    archive = _acked_archive(tmp_path)
    runner, adapter = _runner(archive)
    recovery, obligation_id, key = _recovery_and_obligation(runner, archive)
    dl.record_obligation(
        obligation_id=obligation_id, session_key=key, platform="whatsapp",
        chat_id=recovery.chat_id, thread_id=None, content=_CLARIFICATION,
        adapter_profile="jackwhatsapp", bridge_recovery_delivery_id=recovery.delivery_id,
        bridge_recovery_generation=recovery.generation,
    )

    await runner._recover_pending_whatsapp_bridge_handoffs()
    await runner._recover_pending_whatsapp_bridge_handoffs()

    assert [item[1] for item in adapter.sent] == [_CLARIFICATION]
    assert archive.pending_bridge_recoveries() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ("attempting", "failed"))
async def test_ambiguous_recovery_ledger_row_is_held_without_send(tmp_path, state):
    archive = _acked_archive(tmp_path)
    runner, adapter = _runner(archive)
    recovery, obligation_id, key = _recovery_and_obligation(runner, archive)
    dl.record_obligation(
        obligation_id=obligation_id, session_key=key, platform="whatsapp",
        chat_id=recovery.chat_id, thread_id=None, content=_CLARIFICATION,
        adapter_profile="jackwhatsapp", bridge_recovery_delivery_id=recovery.delivery_id,
        bridge_recovery_generation=recovery.generation,
    )
    if state == "attempting":
        dl.mark_attempting(obligation_id)
    else:
        dl.mark_failed(obligation_id, "offline")

    await runner._recover_pending_whatsapp_bridge_handoffs()

    assert adapter.sent == []
    assert dl.bridge_recovery_obligation(recovery.delivery_id, recovery.generation)["state"] == "held"
    assert len(archive.pending_bridge_recoveries()) == 1


@pytest.mark.asyncio
async def test_two_startup_workers_do_not_duplicate_direct_recovery(tmp_path):
    archive = _acked_archive(tmp_path)
    first, adapter = _runner(archive)
    second, _ = _runner(archive)
    second.adapters = {Platform.WHATSAPP: adapter}
    second._profile_adapters = {"jackwhatsapp": {Platform.WHATSAPP: adapter}}

    await asyncio.gather(
        first._recover_pending_whatsapp_bridge_handoffs(),
        second._recover_pending_whatsapp_bridge_handoffs(),
    )

    assert [item[1] for item in adapter.sent] == [_CLARIFICATION]
    assert archive.pending_bridge_recoveries() == []


@pytest.mark.asyncio
async def test_archive_registration_failure_is_quiet(tmp_path, monkeypatch):
    archive = _acked_archive(tmp_path)
    runner, adapter = _runner(archive)
    monkeypatch.setattr(archive, "register_bridge_recovery_delivery", lambda *_args: False)

    await runner._recover_pending_whatsapp_bridge_handoffs()

    assert adapter.sent == []
    adapter.handle_message.assert_not_awaited()
