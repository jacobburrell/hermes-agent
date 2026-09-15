"""Opt-in, profile-local WhatsApp context projection.

This module deliberately does *not* select a model or memory provider.  A
caller must inject a semantic processor after the user has explicitly enabled
the YAML setting below.  With no processor (the production default), it is a
no-op.  The only persistent source of truth remains ``WhatsAppInboundArchive``.

Example (disabled unless every field is valid)::

  platforms:
    whatsapp:
      extra:
        context_projection:
          enabled: true
          cross_chat: false
          authorized_chats: []
          retention_days: 30
          deletion: tombstone       # tombstone | delete
          per_chat_limit: 8
          profile_limit: 16
          attachment_limit: 4
          processor_timeout_seconds: 5
          processor_item_limit: 8

``cross_chat: true`` additionally requires the current chat and every chat
read by the projection to be listed in ``authorized_chats``.  Projection data
is rollout-forward: pre-existing archive rows are never backfilled.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from plugins.platforms.whatsapp.inbound_archive import WhatsAppInboundArchive


_ITEM_KINDS = frozenset({"fact", "commitment", "open_task"})
_DELETIONS = frozenset({"tombstone", "delete"})
_PROCESSOR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wa-context-projection")
_MAX_RENDERED_PROVENANCE = 128


@dataclass(frozen=True)
class ProjectionSettings:
    """Strict, fail-closed user configuration for the optional projection."""

    enabled: bool
    cross_chat: bool
    authorized_chats: tuple[str, ...]
    retention_days: int
    deletion: str
    per_chat_limit: int
    profile_limit: int
    attachment_limit: int
    processor_timeout_seconds: int
    processor_item_limit: int

    @classmethod
    def from_extra(cls, extra: Any) -> "ProjectionSettings | None":
        if not isinstance(extra, Mapping):
            return None
        raw = extra.get("context_projection")
        if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
            return None
        # Every retention/scope control is explicit.  A half-configured feature
        # must not accidentally disclose data across chats.
        required = {
            "cross_chat", "authorized_chats", "retention_days", "deletion",
            "per_chat_limit", "profile_limit", "attachment_limit",
            "processor_timeout_seconds", "processor_item_limit",
        }
        if not required <= set(raw):
            return None
        cross_chat = raw.get("cross_chat")
        chats = raw.get("authorized_chats")
        deletion = raw.get("deletion")
        if not isinstance(cross_chat, bool) or not isinstance(chats, list) or deletion not in _DELETIONS:
            return None
        normalized_values = [_normalize_chat(value) for value in chats]
        if any(not value for value in normalized_values) or len(set(normalized_values)) != len(normalized_values):
            return None
        normalized = tuple(sorted(normalized_values))
        # Cross-chat data has a positive allowlist; an empty allowlist is only
        # meaningful for single-chat projections.
        if cross_chat and not normalized:
            return None
        numbers: dict[str, int] = {}
        for key, lower, upper in (
            ("retention_days", 1, 3650), ("per_chat_limit", 1, 64),
            ("profile_limit", 1, 128), ("attachment_limit", 0, 32),
            ("processor_timeout_seconds", 1, 30), ("processor_item_limit", 1, 32),
        ):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                return None
            numbers[key] = value
        return cls(True, cross_chat, normalized, numbers["retention_days"], deletion,
                   numbers["per_chat_limit"], numbers["profile_limit"], numbers["attachment_limit"],
                   numbers["processor_timeout_seconds"], numbers["processor_item_limit"])


@dataclass(frozen=True)
class ProjectionInput:
    """The explicitly scoped input an integrator may pass to its processor.

    This is intentionally not a provider API.  ``text`` can be handed only to
    a processor registered by the embedding application after user opt-in;
    attachment paths, URLs and bytes are never included.
    """

    profile_scope: str
    archive_ref: str
    chat_ref: str
    sender_ref: str
    text: str
    attachments: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ProjectionItem:
    kind: str
    text: str


SemanticProcessor = Callable[[ProjectionInput], Iterable[ProjectionItem | Mapping[str, Any]]]


def _normalize_chat(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    return value if value and len(value) <= 512 else ""


def _safe_item(value: ProjectionItem | Mapping[str, Any]) -> ProjectionItem | None:
    kind = value.kind if isinstance(value, ProjectionItem) else value.get("kind") if isinstance(value, Mapping) else None
    text = value.text if isinstance(value, ProjectionItem) else value.get("text") if isinstance(value, Mapping) else None
    if kind not in _ITEM_KINDS or not isinstance(text, str):
        return None
    text = " ".join(text.strip().split())
    # Context projection has no reason to reveal a local path, URL, or a
    # filename-like injection returned by a misconfigured processor.
    if not text or len(text) > 512 or re.search(r"(?:https?://|file:|(?:^|\s)/|\\)", text, re.I):
        return None
    return ProjectionItem(kind, text)


class WhatsAppContextProjection:
    """A small archive-backed store and new-message sidecar renderer."""

    def __init__(self, archive: WhatsAppInboundArchive, settings: ProjectionSettings):
        self.archive = archive
        self.settings = settings
        with self.archive._connect() as db:  # archive owns path/scope/FULL fsync policy
            db.executescript("""
            CREATE TABLE IF NOT EXISTS archive_context_projection_activation (
                profile_scope TEXT PRIMARY KEY,
                watermark_event_id INTEGER NOT NULL,
                activated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS archive_context_projection_event (
                profile_scope TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                chat_id TEXT NOT NULL,
                archive_event_digest TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(profile_scope,event_id)
            );
            CREATE TABLE IF NOT EXISTS archive_context_projection_claim (
                profile_scope TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(profile_scope,event_id)
            );
            CREATE TABLE IF NOT EXISTS archive_context_projection_item (
                id INTEGER PRIMARY KEY,
                profile_scope TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                chat_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at REAL NOT NULL,
                tombstoned_at REAL
            );
            CREATE INDEX IF NOT EXISTS archive_context_projection_chat
              ON archive_context_projection_item(profile_scope,chat_id,created_at,id);
            """)
            columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(archive_context_projection_event)")}
            if "archive_event_digest" not in columns:
                db.execute("ALTER TABLE archive_context_projection_event ADD COLUMN archive_event_digest TEXT NOT NULL DEFAULT ''")

    def activate(self, *, now: float | None = None) -> int:
        """Persist a rollout watermark *before* the next archive write.

        The adapter calls this before it records an inbound event.  Therefore
        older archive rows can never be silently fed to a newly enabled
        processor, including after a restart.
        """
        timestamp = float(time.time() if now is None else now)
        with self.archive._connect() as db:
            row = db.execute(
                "SELECT watermark_event_id FROM archive_context_projection_activation WHERE profile_scope=?",
                (self.archive.scope,),
            ).fetchone()
            if row is not None:
                return int(row["watermark_event_id"])
            watermark = int(db.execute(
                "SELECT COALESCE(MAX(id),0) AS maximum FROM archive_event WHERE profile_scope=?",
                (self.archive.scope,),
            ).fetchone()["maximum"])
            db.execute(
                "INSERT INTO archive_context_projection_activation(profile_scope,watermark_event_id,activated_at) VALUES(?,?,?)",
                (self.archive.scope, watermark, timestamp),
            )
            return watermark

    def capture_and_render(
        self, *, event_id: int, raw: Mapping[str, Any], processor: SemanticProcessor | None,
        now: float | None = None,
    ) -> str | None:
        """Persist this *new* event then render its bounded local sidecar.

        A missing/rejected processor is deliberately no delivery: the archive
        remains intact, and retrying later cannot silently process old text.
        """
        if processor is None or isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 1:
            return None
        chat = _normalize_chat(raw.get("chatId"))
        if not chat or (self.settings.cross_chat and chat not in self.settings.authorized_chats):
            return None
        claim = self._claim(event_id, now=now)
        if claim == "completed":
            # Replay/restart must never send a historical body through the
            # processor again.  It can only render the prior rollout-forward
            # projection for this new delivery attempt.
            return self.render(event_id=event_id, chat_id=chat, now=now)
        if claim != "acquired":
            # A timeout/crash is an uncertain disclosure state.  Never retry
            # it automatically or re-submit the archived body to a processor.
            return None
        metadata = self._metadata_for_event(event_id)
        if metadata is None:
            self._set_claim(event_id, "uncertain", now=now)
            return None
        body = raw.get("body")
        if not isinstance(body, str) or len(body) > 16_384:
            self._set_claim(event_id, "uncertain", now=now)
            return None
        request = ProjectionInput(
            profile_scope=self.archive.scope,
            archive_ref=str(metadata["archive_ref"]), chat_ref=str(metadata["chat_id"]),
            sender_ref=str(metadata["sender_id"]), text=body,
            attachments=tuple(metadata["attachments"]),
        )
        try:
            future = _PROCESSOR_EXECUTOR.submit(processor, request)
            proposed = []
            for item in future.result(timeout=self.settings.processor_timeout_seconds):
                if len(proposed) >= self.settings.processor_item_limit:
                    break
                safe = _safe_item(item)
                if safe is not None:
                    proposed.append(safe)
        except (Exception, TimeoutError):
            self._set_claim(event_id, "uncertain", now=now)
            return None
        items = tuple(proposed)
        timestamp = float(time.time() if now is None else now)
        safe_metadata = self._safe_metadata(metadata)
        digest = self._archive_event_digest(event_id)
        if not digest:
            self._set_claim(event_id, "uncertain", now=timestamp)
            return None
        with self.archive._connect() as db:
            # This is the rollout boundary.  A restarted process only renders
            # items already captured after opt-in; it never scans old bodies.
            exists = db.execute(
                "SELECT 1 FROM archive_context_projection_event WHERE profile_scope=? AND event_id=?",
                (self.archive.scope, event_id),
            ).fetchone()
            if exists is None:
                db.execute(
                    "INSERT INTO archive_context_projection_event(profile_scope,event_id,chat_id,archive_event_digest,created_at,metadata_json) VALUES(?,?,?,?,?,?)",
                    (self.archive.scope, event_id, chat, digest, timestamp, json.dumps(safe_metadata, sort_keys=True)),
                )
                for item in items:
                    db.execute(
                        "INSERT INTO archive_context_projection_item(profile_scope,event_id,chat_id,kind,text,created_at) VALUES(?,?,?,?,?,?)",
                        (self.archive.scope, event_id, chat, item.kind, item.text, timestamp),
                    )
                if safe_metadata["identity_conflict"]:
                    db.execute(
                        "INSERT INTO archive_context_projection_item(profile_scope,event_id,chat_id,kind,text,created_at) VALUES(?,?,?,?,?,?)",
                        (self.archive.scope, event_id, chat, "conflict", "Archived identity conflict retained as local evidence.", timestamp),
                    )
            db.execute(
                "UPDATE archive_context_projection_claim SET state='completed',updated_at=? WHERE profile_scope=? AND event_id=? AND state='attempted'",
                (timestamp, self.archive.scope, event_id),
            )
            self._apply_retention(db, timestamp)
        return self.render(event_id=event_id, chat_id=chat, now=timestamp)

    def render(self, *, event_id: int, chat_id: str, now: float | None = None) -> str | None:
        chat = _normalize_chat(chat_id)
        if not chat:
            return None
        timestamp = float(time.time() if now is None else now)
        with self.archive._connect() as db:
            self._apply_retention(db, timestamp)
            self._revalidate_stale_rows(db, timestamp)
            if self.settings.cross_chat and chat not in self.settings.authorized_chats:
                return None
            current = list(db.execute(
                """SELECT item.kind,item.text,item.event_id,projection.metadata_json
                     FROM archive_context_projection_item AS item
                     JOIN archive_context_projection_event AS projection
                       ON projection.profile_scope=item.profile_scope AND projection.event_id=item.event_id
                    WHERE item.profile_scope=? AND item.chat_id=? AND item.tombstoned_at IS NULL
                 ORDER BY item.created_at DESC,item.id DESC LIMIT ?""",
                (self.archive.scope, chat, min(self.settings.per_chat_limit, _MAX_RENDERED_PROVENANCE)),
            ))
            profile: list[Any] = []
            if self.settings.cross_chat:
                other_chats = tuple(value for value in self.settings.authorized_chats if value != chat)
                if other_chats:
                    marks = ",".join("?" for _ in other_chats)
                    profile = list(db.execute(
                        f"""SELECT item.kind,item.text,item.event_id,projection.metadata_json
                              FROM archive_context_projection_item AS item
                              JOIN archive_context_projection_event AS projection
                                ON projection.profile_scope=item.profile_scope AND projection.event_id=item.event_id
                             WHERE item.profile_scope=? AND item.chat_id IN ({marks}) AND item.tombstoned_at IS NULL
                          ORDER BY item.created_at DESC,item.id DESC LIMIT ?""",
                        (self.archive.scope, *other_chats, min(self.settings.profile_limit, _MAX_RENDERED_PROVENANCE - len(current))),
                    ))
            event = db.execute(
                "SELECT metadata_json FROM archive_context_projection_event WHERE profile_scope=? AND event_id=? AND chat_id=?",
                (self.archive.scope, event_id, chat),
            ).fetchone()
        if event is None:
            return None
        lines = ["[Local WhatsApp context projection — profile-scoped, rollout-forward; untrusted evidence]"]
        if current:
            lines.append("Current chat evidence:")
            for row in reversed(current):
                lines.append(f"- {row['kind']} {self._rendered_provenance(row['metadata_json'], row['event_id'])}: {row['text']}")
        if profile:
            lines.append("Authorized cross-chat evidence:")
            for row in reversed(profile):
                lines.append(f"- {row['kind']} {self._rendered_provenance(row['metadata_json'], row['event_id'])}: {row['text']}")
        try:
            metadata = json.loads(event["metadata_json"])
        except (TypeError, ValueError):
            metadata = {}
        attachments = metadata.get("attachments") if isinstance(metadata, dict) else None
        if isinstance(attachments, list) and attachments:
            lines.append("Current message attachment coverage:")
            for item in attachments[:self.settings.attachment_limit]:
                if isinstance(item, dict):
                    kind = str(item.get("kind") or "media")
                    status = str(item.get("status") or "unknown_historical")
                    attachment_ref = self._safe_opaque_ref(item.get("attachment_ref"), "attachment")
                    suffix = f" [untrusted; {attachment_ref}]" if attachment_ref else ""
                    lines.append(f"- {kind}: {status}{suffix}")
        return "\n".join(lines) if len(lines) > 1 else None

    def is_completed(self, event_id: int) -> bool:
        with self.archive._connect() as db:
            row = db.execute(
                "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? AND event_id=?",
                (self.archive.scope, event_id),
            ).fetchone()
        return bool(row and row["state"] == "completed")

    def _claim(self, event_id: int, *, now: float | None) -> str:
        timestamp = float(time.time() if now is None else now)
        with self.archive._connect() as db:
            activation = db.execute(
                "SELECT watermark_event_id FROM archive_context_projection_activation WHERE profile_scope=?",
                (self.archive.scope,),
            ).fetchone()
            if activation is None or event_id <= int(activation["watermark_event_id"]):
                return "excluded"
            row = db.execute(
                "SELECT state FROM archive_context_projection_claim WHERE profile_scope=? AND event_id=?",
                (self.archive.scope, event_id),
            ).fetchone()
            if row is not None:
                state = str(row["state"])
                if state == "attempted":
                    db.execute(
                        "UPDATE archive_context_projection_claim SET state='uncertain',updated_at=? WHERE profile_scope=? AND event_id=?",
                        (timestamp, self.archive.scope, event_id),
                    )
                    return "uncertain"
                return state
            db.execute(
                "INSERT INTO archive_context_projection_claim(profile_scope,event_id,state,attempt_id,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (self.archive.scope, event_id, "attempted", uuid.uuid4().hex, timestamp, timestamp),
            )
            return "acquired"

    def _set_claim(self, event_id: int, state: str, *, now: float | None) -> None:
        timestamp = float(time.time() if now is None else now)
        with self.archive._connect() as db:
            db.execute(
                "UPDATE archive_context_projection_claim SET state=?,updated_at=? WHERE profile_scope=? AND event_id=? AND state='attempted'",
                (state, timestamp, self.archive.scope, event_id),
            )

    def _archive_event_digest(self, event_id: int) -> str:
        with self.archive._connect() as db:
            row = db.execute(
                "SELECT event_digest FROM archive_event WHERE profile_scope=? AND id=?",
                (self.archive.scope, event_id),
            ).fetchone()
        value = str(row["event_digest"]) if row else ""
        return value if re.fullmatch(r"[a-f0-9]{64}", value) else ""

    @staticmethod
    def _safe_opaque_ref(value: Any, kind: str) -> str:
        value = value if isinstance(value, str) else ""
        return value if re.fullmatch(rf"{re.escape(kind)}:[a-f0-9]{{64}}", value) else ""

    @staticmethod
    def _safe_timestamp_ms(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9_999_999_999_999:
            return None
        return value

    def _rendered_provenance(self, encoded: Any, event_id: int) -> str:
        """Render only bounded opaque provenance retained beside each item."""
        try:
            metadata = json.loads(encoded)
        except (TypeError, ValueError):
            metadata = {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        event_ref = self._safe_opaque_ref(metadata.get("archive_ref"), "event") or self.archive._opaque_archive_reference(int(event_id))
        refs = [event_ref]
        for key, kind in (("chat_id", "chat"), ("sender_id", "sender"), ("message_ref", "message")):
            value = self._safe_opaque_ref(metadata.get(key), kind)
            if value:
                refs.append(value)
        timestamp = self._safe_timestamp_ms(metadata.get("source_timestamp_ms"))
        if timestamp is not None:
            refs.append(f"source_ms:{timestamp}")
        return "[untrusted; " + "; ".join(refs) + "]"

    def _revalidate_stale_rows(self, db, now: float) -> None:
        stale = db.execute(
            """SELECT projection.event_id
                   FROM archive_context_projection_event AS projection
              LEFT JOIN archive_event AS event
                     ON event.id=projection.event_id AND event.profile_scope=projection.profile_scope
                  WHERE projection.profile_scope=?
                    AND (event.id IS NULL OR event.event_digest != projection.archive_event_digest)""",
            (self.archive.scope,),
        ).fetchall()
        for row in stale:
            event_id = int(row["event_id"])
            if self.settings.deletion == "delete":
                db.execute("DELETE FROM archive_context_projection_item WHERE profile_scope=? AND event_id=?", (self.archive.scope, event_id))
            else:
                db.execute(
                    "UPDATE archive_context_projection_item SET text='',tombstoned_at=? WHERE profile_scope=? AND event_id=? AND tombstoned_at IS NULL",
                    (now, self.archive.scope, event_id),
                )

    def _metadata_for_event(self, event_id: int) -> dict[str, Any] | None:
        # This public archive projection is already redacted/opaque.  The raw
        # archive fields never leave this module or the profile DB.
        for value in self.archive.recent_context_metadata(limit=1000):
            if value.get("sequence") == event_id:
                return value
        return None

    def _safe_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        attachments: list[dict[str, Any]] = []
        raw = metadata.get("attachments")
        if isinstance(raw, (list, tuple)):
            for item in raw[:self.settings.attachment_limit]:
                if not isinstance(item, Mapping):
                    continue
                descriptor = item.get("descriptor") if isinstance(item.get("descriptor"), Mapping) else {}
                kind = descriptor.get("kind") if isinstance(descriptor.get("kind"), str) else "media"
                status = item.get("status") if item.get("status") in {"owned", "missing_or_rejected", "deleted_or_replaced", "unknown"} else "unknown_historical"
                attachment_ref = self._safe_opaque_ref(item.get("attachment_ref"), "attachment")
                attachments.append({
                    "kind": kind, "status": status,
                    **({"attachment_ref": attachment_ref} if attachment_ref else {}),
                })
        return {
            "archive_ref": self._safe_opaque_ref(metadata.get("archive_ref"), "event"),
            "chat_id": self._safe_opaque_ref(metadata.get("chat_id"), "chat"),
            "sender_id": self._safe_opaque_ref(metadata.get("sender_id"), "sender"),
            "message_ref": self._safe_opaque_ref(metadata.get("message_id"), "message"),
            "source_timestamp_ms": self._safe_timestamp_ms(metadata.get("source_timestamp_ms")),
            "identity_conflict": bool(metadata.get("identity_conflict")),
            "media_coverage": str(metadata.get("media_coverage") or "unknown_historical"),
            "attachments": attachments,
        }

    def _apply_retention(self, db, now: float) -> None:
        cutoff = now - self.settings.retention_days * 86400
        if self.settings.deletion == "delete":
            db.execute(
                "DELETE FROM archive_context_projection_item WHERE profile_scope=? AND created_at < ?",
                (self.archive.scope, cutoff),
            )
        else:
            db.execute(
                "UPDATE archive_context_projection_item SET text='',tombstoned_at=? WHERE profile_scope=? AND created_at < ? AND tombstoned_at IS NULL",
                (now, self.archive.scope, cutoff),
            )
