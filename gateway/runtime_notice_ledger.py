"""Durable at-most-once ledger for human-chat terminal notices.

Terminal failures are intentionally different from ordinary final-response
delivery.  Retrying a send whose acknowledgement was lost can create the exact
duplicate error bubble this rail exists to prevent.  A row therefore becomes a
permanent tombstone once a transport call starts unless the adapter explicitly
guarantees that its failed result means no delivery occurred.

The database is profile-local (``<HERMES_HOME>/state.db``).  Reservations use
``BEGIN IMMEDIATE`` so two gateway runners cannot both deliver the same
``(session, run, code)`` notice.  A dead owner's pre-wire reservation can be
reclaimed; a dead owner's in-flight row is tombstoned as ambiguous and is never
blindly retried.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 1_000


class ReservationState(str, Enum):
    RESERVED = "reserved"
    IN_FLIGHT = "in_flight"
    SENT = "sent"
    AMBIGUOUS = "ambiguous"


class ReserveOutcome(str, Enum):
    ACQUIRED = "acquired"
    ALREADY_DELIVERED = "already_delivered"


@dataclass(frozen=True, slots=True)
class Reservation:
    outcome: ReserveOutcome
    owner_token: str = ""


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: object, started_at: object) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import _pid_exists, get_process_start_time

        current = get_process_start_time(pid_int)
        if current is None:
            return bool(_pid_exists(pid_int))
    except Exception:
        return False
    if started_at is None:
        return True
    try:
        return int(current) == int(started_at)
    except (TypeError, ValueError):
        return True


class TerminalNoticeLedger:
    """Profile-scoped durable reservation ledger."""

    def __init__(self, profile_home: Path):
        self.profile_home = Path(profile_home)
        self.path = self.profile_home / "state.db"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="state.db (runtime notices)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS terminal_notice_deliveries (
                    session_key TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    state TEXT NOT NULL,
                    state_applied INTEGER NOT NULL DEFAULT 0,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    owner_pid INTEGER,
                    owner_started_at INTEGER,
                    owner_token TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_key, run_id, code)
                )"""
            )
            return conn
        except Exception:
            conn.close()
            raise

    def reserve(
        self,
        *,
        session_key: str,
        run_id: str,
        code: str,
        content: str,
        content_hash: str,
    ) -> Reservation:
        """Atomically reserve one logical terminal notice.

        ``IN_FLIGHT`` is never reclaimed.  If its owner died, the row is first
        converted to ``AMBIGUOUS`` and remains a tombstone because the platform
        may have accepted the message before the process disappeared.
        """

        now = time.time()
        pid, started = _owner_stamp()
        token = uuid.uuid4().hex
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT state, owner_pid, owner_started_at
                   FROM terminal_notice_deliveries
                   WHERE session_key=? AND run_id=? AND code=?""",
                (session_key, run_id, code),
            ).fetchone()
            if row is not None:
                state, owner_pid, owner_started = row
                if state in (
                    ReservationState.SENT.value,
                    ReservationState.AMBIGUOUS.value,
                ):
                    conn.commit()
                    return Reservation(ReserveOutcome.ALREADY_DELIVERED)
                if state == ReservationState.IN_FLIGHT.value:
                    if not _owner_alive(owner_pid, owner_started):
                        conn.execute(
                            """UPDATE terminal_notice_deliveries
                               SET state=?, updated_at=?
                               WHERE session_key=? AND run_id=? AND code=?""",
                            (
                                ReservationState.AMBIGUOUS.value,
                                now,
                                session_key,
                                run_id,
                                code,
                            ),
                        )
                    conn.commit()
                    return Reservation(ReserveOutcome.ALREADY_DELIVERED)
                if state == ReservationState.RESERVED.value and _owner_alive(
                    owner_pid, owner_started
                ):
                    conn.commit()
                    return Reservation(ReserveOutcome.ALREADY_DELIVERED)

                conn.execute(
                    """UPDATE terminal_notice_deliveries
                       SET state=?, state_applied=0, content=?, content_hash=?,
                           owner_pid=?, owner_started_at=?, owner_token=?,
                           updated_at=?
                       WHERE session_key=? AND run_id=? AND code=?""",
                    (
                        ReservationState.RESERVED.value,
                        content,
                        content_hash,
                        pid,
                        started,
                        token,
                        now,
                        session_key,
                        run_id,
                        code,
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO terminal_notice_deliveries
                       (session_key, run_id, code, state, state_applied, content,
                        content_hash, owner_pid, owner_started_at, owner_token,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_key,
                        run_id,
                        code,
                        ReservationState.RESERVED.value,
                        content,
                        content_hash,
                        pid,
                        started,
                        token,
                        now,
                        now,
                    ),
                )
            self._prune_locked(conn, now)
            conn.commit()
            return Reservation(ReserveOutcome.ACQUIRED, token)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def mark_state_applied(
        self, session_key: str, run_id: str, code: str, owner_token: str
    ) -> bool:
        """Apply the caller-owned response suppression exactly once."""

        return self._update_owned(
            session_key,
            run_id,
            code,
            owner_token,
            "state_applied=1",
            extra_where=" AND state_applied=0",
        )

    def mark_in_flight(
        self, session_key: str, run_id: str, code: str, owner_token: str
    ) -> bool:
        return self._update_owned(
            session_key,
            run_id,
            code,
            owner_token,
            "state=?",
            (ReservationState.IN_FLIGHT.value,),
            extra_where=" AND state=?",
            extra_params=(ReservationState.RESERVED.value,),
        )

    def mark_sent(
        self, session_key: str, run_id: str, code: str, owner_token: str
    ) -> bool:
        return self._update_owned(
            session_key,
            run_id,
            code,
            owner_token,
            "state=?",
            (ReservationState.SENT.value,),
            extra_where=" AND state=?",
            extra_params=(ReservationState.IN_FLIGHT.value,),
        )

    def mark_ambiguous(
        self, session_key: str, run_id: str, code: str, owner_token: str
    ) -> bool:
        return self._update_owned(
            session_key,
            run_id,
            code,
            owner_token,
            "state=?",
            (ReservationState.AMBIGUOUS.value,),
            extra_where=" AND state=?",
            extra_params=(ReservationState.IN_FLIGHT.value,),
        )

    def release_reserved(
        self, session_key: str, run_id: str, code: str, owner_token: str
    ) -> bool:
        """Release only a definite pre-wire failure."""

        conn = self._connect()
        try:
            cursor = conn.execute(
                """DELETE FROM terminal_notice_deliveries
                   WHERE session_key=? AND run_id=? AND code=?
                     AND owner_token=? AND state=?""",
                (
                    session_key,
                    run_id,
                    code,
                    owner_token,
                    ReservationState.RESERVED.value,
                ),
            )
            return cursor.rowcount == 1
        finally:
            conn.close()

    def release_definite_failure(
        self, session_key: str, run_id: str, code: str, owner_token: str
    ) -> bool:
        """Release after an adapter explicitly proves no delivery occurred."""

        conn = self._connect()
        try:
            cursor = conn.execute(
                """DELETE FROM terminal_notice_deliveries
                   WHERE session_key=? AND run_id=? AND code=?
                     AND owner_token=? AND state=?""",
                (
                    session_key,
                    run_id,
                    code,
                    owner_token,
                    ReservationState.IN_FLIGHT.value,
                ),
            )
            return cursor.rowcount == 1
        finally:
            conn.close()

    def get_state(self, session_key: str, run_id: str, code: str) -> Optional[str]:
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT state FROM terminal_notice_deliveries
                   WHERE session_key=? AND run_id=? AND code=?""",
                (session_key, run_id, code),
            ).fetchone()
            return str(row[0]) if row else None
        finally:
            conn.close()

    def _update_owned(
        self,
        session_key: str,
        run_id: str,
        code: str,
        owner_token: str,
        assignment: str,
        assignment_params: tuple = (),
        *,
        extra_where: str = "",
        extra_params: tuple = (),
    ) -> bool:
        conn = self._connect()
        try:
            cursor = conn.execute(
                f"""UPDATE terminal_notice_deliveries
                    SET {assignment}, updated_at=?
                    WHERE session_key=? AND run_id=? AND code=?
                      AND owner_token=?{extra_where}""",
                (
                    *assignment_params,
                    time.time(),
                    session_key,
                    run_id,
                    code,
                    owner_token,
                    *extra_params,
                ),
            )
            return cursor.rowcount == 1
        finally:
            conn.close()

    @staticmethod
    def _prune_locked(conn: sqlite3.Connection, now: float) -> None:
        conn.execute(
            """DELETE FROM terminal_notice_deliveries
               WHERE updated_at < ? AND state IN (?, ?)""",
            (
                now - _RETENTION_SECONDS,
                ReservationState.SENT.value,
                ReservationState.AMBIGUOUS.value,
            ),
        )
        row_count = conn.execute(
            "SELECT COUNT(*) FROM terminal_notice_deliveries"
        ).fetchone()[0]
        excess = max(int(row_count) - _MAX_ROWS, 0)
        if excess:
            conn.execute(
                """DELETE FROM terminal_notice_deliveries WHERE rowid IN (
                       SELECT rowid FROM terminal_notice_deliveries
                       WHERE state IN (?, ?)
                       ORDER BY updated_at ASC LIMIT ?
                   )""",
                (
                    ReservationState.SENT.value,
                    ReservationState.AMBIGUOUS.value,
                    int(excess),
                ),
            )


__all__ = [
    "Reservation",
    "ReservationState",
    "ReserveOutcome",
    "TerminalNoticeLedger",
]
