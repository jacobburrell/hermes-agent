"""Private, profile-scoped WhatsApp inbound archive; never a prompt projection."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ArchiveRejected(ValueError):
    pass


@dataclass(frozen=True)
class MaterializationResult:
    """The durable attachment-manifest outcome for one archived event."""
    complete: bool
    owned_count: int
    attachment_count: int
    statuses: tuple[str, ...]
    owned_paths: tuple[str, ...]
    owned_descriptors: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class BridgeReceipt:
    """Exact fenced bridge acknowledgement stored beside an archived event."""
    delivery_id: str
    event_digest: str
    request: dict[str, object]
    ready: bool
    acknowledged: bool
    recovery_pending: bool
    # Album members share one recovery fence.  This is an opaque, profile
    # private digest; it is never exposed through metadata or diagnostics.
    album_key: str | None = None
    # One dispatch generation of an album.  This intentionally differs from
    # ``album_key`` so a late continuation cannot be settled by its original.
    recovery_key: str | None = None


@dataclass(frozen=True)
class BridgeRecovery:
    """One private, generation-fenced recovery clarification candidate."""
    delivery_id: str
    event_digest: str
    event_id: int
    generation: int
    chat_id: str
    is_group: bool
    state: str
    obligation_id: str | None


@dataclass(frozen=True)
class AddressedFollowupAnchor:
    """One local, sender-scoped continuation anchor for a WhatsApp group."""
    message_id: str
    text: str


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _node_digest_json(value: Any) -> str:
    """Canonical JSON compatible with the bridge's ``canonicalDigestJson``.

    This is intentionally separate from the archive's historical payload
    digest.  It authenticates exactly the leased bridge event, including the
    bridge's stable media descriptor hash, before Python ever archives or ACKs
    it.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if abs(value) > 9_007_199_254_740_991:
            raise ArchiveRejected("unsafe bridge integer")
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArchiveRejected("non-finite bridge number")
        # JSON.stringify(-0) is "0", unlike Python's json module.
        if value == 0:
            return "0"
        if value.is_integer():
            return _node_digest_json(int(value))
        # ECMAScript switches to fixed notation for [1e-6, 1e21), while
        # Python's shortest representation switches much earlier and pads
        # exponents (1e-07 vs 1e-7).  Start with Python's shortest round-trip
        # digits, then render them using JavaScript's boundary rules.
        rendered = repr(value).lower()
        if "e" not in rendered:
            return rendered
        mantissa, exponent_text = rendered.split("e", 1)
        exponent = int(exponent_text)
        absolute = abs(value)
        if absolute < 1e-6 or absolute >= 1e21:
            sign = "+" if exponent >= 0 else "-"
            return f"{mantissa}e{sign}{abs(exponent)}"
        negative = mantissa.startswith("-")
        digits = mantissa.removeprefix("-").replace(".", "")
        before_decimal = len(mantissa.removeprefix("-").split(".", 1)[0])
        position = before_decimal + exponent
        if position <= 0:
            fixed = "0." + "0" * (-position) + digits
        elif position >= len(digits):
            fixed = digits + "0" * (position - len(digits))
        else:
            fixed = digits[:position] + "." + digits[position:]
        return ("-" if negative else "") + fixed
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_node_digest_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        fields = []
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise ArchiveRejected("non-string bridge object key")
            fields.append(f"{json.dumps(key, ensure_ascii=False)}:{_node_digest_json(value[key])}")
        return "{" + ",".join(fields) + "}"
    raise ArchiveRejected("non-JSON bridge value")


def _node_jid(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    local, marker, domain = raw.rpartition("@")
    if not marker:
        return raw.split(":", 1)[0]
    return f"{local.split(':', 1)[0]}@{domain}"


def _node_string(value: Any) -> str:
    """Subset of JavaScript ``String`` used by the JSON bridge contract."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def bridge_event_digest(raw: Mapping[str, Any]) -> str:
    """Return the Node spool digest for a flat leased event, fail closed."""
    if not isinstance(raw, Mapping):
        raise ArchiveRejected("invalid bridge event")
    payload = dict(raw)
    for key in ("mediaUrls", "mediaMetadata", "_inboundLease"):
        payload.pop(key, None)
    if "chatId" in payload:
        payload["chatId"] = _node_jid(payload["chatId"])
    if "senderId" in payload:
        payload["senderId"] = _node_jid(payload["senderId"])
    if isinstance(payload.get("readReceiptKey"), Mapping):
        receipt_key = dict(payload["readReceiptKey"])
        receipt_key["remoteJid"] = _node_jid(receipt_key.get("remoteJid"))
        receipt_key["participant"] = _node_jid(receipt_key.get("participant"))
        payload["readReceiptKey"] = receipt_key
    sources = raw.get("mediaUrls") if isinstance(raw.get("mediaUrls"), list) else []
    declared = raw.get("mediaMetadata") if isinstance(raw.get("mediaMetadata"), list) else []
    entry_count = max(len(sources), len(declared), 1 if raw.get("hasMedia") else 0)
    stable_entries = []
    for index in range(entry_count):
        metadata = declared[index] if index < len(declared) and isinstance(declared[index], Mapping) else {}
        declared_sha = str(metadata.get("sha256") or "").lower()
        if len(declared_sha) != 64 or any(char not in "0123456789abcdef" for char in declared_sha):
            declared_sha = ""
        stable_entries.append({
            "index": index,
            "mediaType": _node_string(metadata.get("mediaType") or raw.get("mediaType") or ""),
            "mime": _node_string(metadata.get("mime") or raw.get("mime") or ""),
            "fileName": _node_string(metadata.get("fileName") or raw.get("fileName") or ""),
            "declaredSha256": declared_sha,
            "declaredSize": _node_string(metadata.get("size")),
        })
    ordered_metadata_digest = hashlib.sha256(_node_digest_json(stable_entries).encode()).hexdigest()
    return hashlib.sha256(_node_digest_json({
        "event": payload,
        "orderedUnownedMediaMetadataDigest": ordered_metadata_digest,
    }).encode()).hexdigest()


def _normal(value: Any) -> str:
    """Normalize JID case only; device qualifiers are identity-bearing."""
    return str(value or "").strip().lower()


class WhatsAppInboundArchive:
    """Durable archive with idempotent identity and collision fail-closed behavior."""
    def __init__(self, root: Path, profile_home: Path, cache_root):
        self.root, self.profile_home = Path(root), Path(profile_home).resolve()
        self.cache_roots = tuple(Path(item).resolve() for item in (cache_root if isinstance(cache_root, (list, tuple)) else (cache_root,)))
        self._validate_paths()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.path = self.root / "archive.sqlite3"
        self.media_root = self.root / "media"
        self.media_root.mkdir(exist_ok=True, mode=0o700); os.chmod(self.media_root, 0o700)
        self.scope = hashlib.sha256(str(self.profile_home).encode()).hexdigest()
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS archive_event (id INTEGER PRIMARY KEY, profile_scope TEXT NOT NULL, chat_id TEXT NOT NULL, sender_id TEXT NOT NULL, message_id TEXT NOT NULL, admission TEXT NOT NULL, event_digest TEXT NOT NULL, payload_json TEXT NOT NULL, album_group TEXT, album_role TEXT, album_index INTEGER, created_at REAL NOT NULL, UNIQUE(profile_scope,chat_id,message_id));
            CREATE TABLE IF NOT EXISTS archive_attachment (event_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, descriptor_json TEXT NOT NULL, owned_path TEXT, sha256 TEXT, size INTEGER, download_status TEXT NOT NULL, album_ordinal INTEGER, PRIMARY KEY(event_id,ordinal));
            CREATE TABLE IF NOT EXISTS archive_collision (id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, candidate_digest TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS archive_bridge_receipt (
                profile_scope TEXT NOT NULL,
                delivery_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                bridge_event_digest TEXT NOT NULL,
                consumer_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                token TEXT NOT NULL,
                ready INTEGER NOT NULL DEFAULT 0,
                recovery_pending INTEGER NOT NULL DEFAULT 0,
                album_key TEXT,
                recovery_key TEXT,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(profile_scope, delivery_id)
            );
            CREATE TABLE IF NOT EXISTS archive_bridge_recovery (
                profile_scope TEXT NOT NULL,
                delivery_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL,
                obligation_id TEXT,
                owner_pid INTEGER,
                owner_started_at INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(profile_scope, delivery_id)
            );
            CREATE TABLE IF NOT EXISTS archive_album_handoff (
                profile_scope TEXT NOT NULL,
                album_key TEXT NOT NULL,
                operational INTEGER NOT NULL,
                state TEXT NOT NULL,
                primary_event_id INTEGER,
                closed_at REAL NOT NULL,
                PRIMARY KEY(profile_scope, album_key)
            );
            CREATE TABLE IF NOT EXISTS archive_album_late_member (
                profile_scope TEXT NOT NULL,
                album_key TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                linked_at REAL NOT NULL,
                PRIMARY KEY(profile_scope, album_key, event_id)
            );
            CREATE TABLE IF NOT EXISTS archive_addressed_followup_anchor (
                profile_scope TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY(profile_scope, chat_id)
            );
            CREATE INDEX IF NOT EXISTS archive_event_profile_sequence ON archive_event(profile_scope, id DESC);
            CREATE INDEX IF NOT EXISTS archive_event_profile_chat_sequence ON archive_event(profile_scope, chat_id, id DESC);
            """)
            # Existing profile archives predate the receipt table.  The sole
            # additive column is deliberately default-false: an old row can
            # never be mistaken for an acknowledged runtime handoff.
            columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(archive_bridge_receipt)")}
            if "recovery_pending" not in columns:
                db.execute(
                    "ALTER TABLE archive_bridge_receipt "
                    "ADD COLUMN recovery_pending INTEGER NOT NULL DEFAULT 0"
                )
            if "album_key" not in columns:
                db.execute("ALTER TABLE archive_bridge_receipt ADD COLUMN album_key TEXT")
            if "recovery_key" not in columns:
                db.execute("ALTER TABLE archive_bridge_receipt ADD COLUMN recovery_key TEXT")
            db.execute(
                "CREATE INDEX IF NOT EXISTS archive_bridge_receipt_recovery_generation "
                "ON archive_bridge_receipt(profile_scope,recovery_key,recovery_pending,acknowledged)"
            )
            event_columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(archive_event)")}
            if "source_timestamp_ms" not in event_columns:
                db.execute("ALTER TABLE archive_event ADD COLUMN source_timestamp_ms INTEGER")
            album_columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(archive_album_handoff)")}
            if "primary_event_id" not in album_columns:
                db.execute("ALTER TABLE archive_album_handoff ADD COLUMN primary_event_id INTEGER")
        self._secure()

    def _validate_paths(self):
        """Validate every profile-relative ancestor before SQLite can mutate it."""
        try:
            relative = self.root.relative_to(self.profile_home)
        except ValueError as exc:
            raise ArchiveRejected("archive root escaped profile") from exc
        chain = [self.profile_home]
        current = self.profile_home
        for part in relative.parts:
            current /= part; chain.append(current)
        directories = [*chain, self.root / "media"]
        for path in directories:
            if os.path.lexists(path):
                mode = os.lstat(path).st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    raise ArchiveRejected("unsafe archive directory")
        for path in [self.root / "archive.sqlite3", self.root / "archive.sqlite3-wal", self.root / "archive.sqlite3-shm"]:
            if os.path.lexists(path) and stat.S_ISLNK(os.lstat(path).st_mode):
                raise ArchiveRejected("unsafe archive symlink")

    def _secure(self):
        for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            if path.exists():
                if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode): raise ArchiveRejected("unsafe archive database path")
                os.chmod(path, 0o600)

    def _connect(self):
        self._validate_paths()
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
        self._secure()
        return db

    @staticmethod
    def _payload(raw: Mapping[str, Any]) -> dict:
        # Deliberately exclude media URLs, key material, bridge leases, and unknown native blobs.
        keys = ("messageId", "chatId", "senderId", "senderName", "chatName", "isGroup", "body", "timestamp", "mediaType", "mime", "fileName", "hasMedia", "quotedMessageId", "quotedParticipant", "quotedRemoteJid", "quotedText", "quotedOutboundByJack", "quotedForwarded")
        out = {key: raw.get(key) for key in keys if key in raw}
        album = ((raw.get("nativeMetadata") or {}).get("album") if isinstance(raw.get("nativeMetadata"), dict) else None)
        if isinstance(album, dict): out["album"] = {k: album.get(k) for k in ("groupId", "role", "messageIndex") if k in album}
        return out

    @staticmethod
    def _source_timestamp_ms(value: Any) -> int | None:
        """Convert an integral bridge Unix-seconds timestamp to milliseconds."""
        sqlite_max = (1 << 63) - 1
        max_seconds = sqlite_max // 1000
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            seconds = value
        elif isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            seconds = int(value)
        elif isinstance(value, str):
            decimal = value.strip()
            if not re.fullmatch(r"\d+", decimal) or len(decimal) > len(str(max_seconds)):
                return None
            seconds = int(decimal)
        else:
            return None
        return seconds * 1000 if 0 <= seconds <= max_seconds else None

    @staticmethod
    def _bounded_int(value: Any, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int | None:
        if isinstance(value, bool):
            return None
        return value if isinstance(value, int) and minimum <= value <= maximum else None

    @staticmethod
    def _safe_metadata_identifier(value: Any, *, maximum: int = 512) -> str:
        if not isinstance(value, str):
            return ""
        value = value.strip()
        if (not value or len(value) > maximum or "/" in value or "\\" in value
                or re.match(r"^[a-z][a-z0-9+.-]{0,31}:", value, re.IGNORECASE)):
            return ""
        return value

    @staticmethod
    def _opaque_metadata_reference(value: Any, *, maximum: int = 512) -> str:
        """Return a stable non-content reference for an unsafe persisted ID."""
        if not isinstance(value, str):
            return ""
        value = value.strip()
        if not value or len(value) > maximum:
            return ""
        return "opaque:" + hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _safe_event_identity(cls, value: Any) -> str:
        return cls._safe_metadata_identifier(value) or cls._opaque_metadata_reference(value)

    @staticmethod
    def _safe_archived_at_ms(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or not 0 <= seconds <= ((1 << 63) - 1) / 1000:
            return None
        return int(seconds * 1000)

    @staticmethod
    def _safe_media_type(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        value = value.lower()
        return value if re.fullmatch(r"[a-z0-9_-]{1,64}", value) else ""

    @staticmethod
    def _safe_attachment_descriptor(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        kind = WhatsAppInboundArchive._safe_media_type(value.get("kind"))
        if kind:
            result["kind"] = kind
        mime = value.get("mime")
        if isinstance(mime, str) and re.fullmatch(
            r"[a-z0-9][a-z0-9.+-]{0,63}/[a-z0-9][a-z0-9.+-]{0,63}", mime.lower()
        ):
            result["mime"] = mime.lower()
        file_name = value.get("file_name")
        if isinstance(file_name, str) and file_name and len(file_name) <= 512:
            result["file_name_sha256"] = hashlib.sha256(file_name.encode("utf-8")).hexdigest()
        return result

    @staticmethod
    def _safe_album_metadata(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        result: dict[str, Any] = {}
        group_id = value.get("groupId")
        if isinstance(group_id, str) and group_id and len(group_id) <= 512:
            result["group_ref"] = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
        role = value.get("role")
        if isinstance(role, str) and role in {"parent", "child"}:
            result["role"] = role
        message_index = WhatsAppInboundArchive._bounded_int(value.get("messageIndex"), maximum=1_000_000)
        if message_index is not None:
            result["message_index"] = message_index
        return result or None

    @staticmethod
    def _metadata_payload(payload_json: str) -> dict[str, Any]:
        """Project only local provenance metadata, never message or quote text."""
        empty = {"quote": None, "album": None, "has_media": False, "media_type": ""}
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            return empty
        if not isinstance(payload, dict):
            return empty
        quote = {
            "message_id": WhatsAppInboundArchive._safe_metadata_identifier(payload.get("quotedMessageId")),
            "participant_id": _normal(WhatsAppInboundArchive._safe_metadata_identifier(payload.get("quotedParticipant"))),
            "remote_chat_id": _normal(WhatsAppInboundArchive._safe_metadata_identifier(payload.get("quotedRemoteJid"))),
            "outbound_by_jack": bool(payload.get("quotedOutboundByJack")),
        }
        return {
            "quote": quote if any(quote.values()) else None,
            "album": WhatsAppInboundArchive._safe_album_metadata(payload.get("album")),
            "has_media": bool(payload.get("hasMedia")),
            "media_type": WhatsAppInboundArchive._safe_media_type(payload.get("mediaType")),
        }

    def record(
        self, raw: Mapping[str, Any], admission: str, *,
        followup_anchor: bool = False,
        followup_chat_id: str | None = None,
        followup_sender_id: str | None = None,
    ) -> tuple[int, bool]:
        """Persist one inbound event and update its local continuation fence.

        Every newly archived group event clears that chat's anchor. Only an
        explicit/proven continuation can install the next one, making an
        intervening message a durable invalidation rather than an in-memory
        best effort.
        """
        if admission not in {"drop", "observe", "operate"}: raise ArchiveRejected("invalid admission")
        data = self._payload(raw); chat, sender = map(_normal, (data.get("chatId"), data.get("senderId"))); mid = str(data.get("messageId") or "").strip()
        if not all((chat, sender, mid)): raise ArchiveRejected("missing stable archive identity")
        anchor_chat = _normal(followup_chat_id)
        anchor_sender = _normal(followup_sender_id)
        if bool(followup_anchor) and (admission != "operate" or not anchor_chat or not anchor_sender):
            raise ArchiveRejected("invalid addressed followup anchor")
        digest = _digest(data); album = data.get("album") if isinstance(data.get("album"), dict) else {}
        with self._connect() as db:
            row = db.execute("SELECT id,event_digest FROM archive_event WHERE profile_scope=? AND chat_id=? AND message_id=?", (self.scope,chat,mid)).fetchone()
            if row:
                if row["event_digest"] != digest:
                    db.execute("INSERT INTO archive_collision(event_id,candidate_digest,created_at) VALUES(?,?,?)", (row["id"],digest,time.time()))
                    return int(row["id"]), False
                return int(row["id"]), True
            created_at = time.time()
            cur = db.execute("INSERT INTO archive_event(profile_scope,chat_id,sender_id,message_id,admission,event_digest,payload_json,source_timestamp_ms,album_group,album_role,album_index,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (self.scope,chat,sender,mid,admission,digest,_canonical(data),self._source_timestamp_ms(data.get("timestamp")),album.get("groupId"),album.get("role"),album.get("messageIndex"),created_at))
            event_id = int(cur.lastrowid)
            if anchor_chat:
                db.execute(
                    "DELETE FROM archive_addressed_followup_anchor WHERE profile_scope=? AND chat_id=?",
                    (self.scope, anchor_chat),
                )
                if followup_anchor:
                    db.execute(
                        "INSERT INTO archive_addressed_followup_anchor(profile_scope,chat_id,sender_id,event_id,created_at) VALUES(?,?,?,?,?)",
                        (self.scope, anchor_chat, anchor_sender, event_id, created_at),
                    )
            return event_id, True

    def addressed_followup_anchor(
        self, chat_id: str, sender_id: str, window_seconds: float,
    ) -> AddressedFollowupAnchor | None:
        """Return a valid same-chat/sender anchor, otherwise fail closed.

        The TTL is local archive time rather than an untrusted bridge
        timestamp, so stale or forged payloads cannot extend it and it remains
        valid across restart.
        """
        if isinstance(window_seconds, bool):
            return None
        try:
            window = float(window_seconds)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(window) or not 0 < window <= 120:
            return None
        chat, sender = _normal(chat_id), _normal(sender_id)
        if not chat or not sender or not self.path.exists():
            return None
        with self._connect() as db:
            row = db.execute(
                """SELECT anchor.created_at,event.message_id,event.payload_json
                   FROM archive_addressed_followup_anchor AS anchor
                   JOIN archive_event AS event ON event.id=anchor.event_id
                   WHERE anchor.profile_scope=? AND anchor.chat_id=? AND anchor.sender_id=?""",
                (self.scope, chat, sender),
            ).fetchone()
            if row is None:
                return None
            if time.time() - float(row["created_at"]) > window:
                db.execute(
                    "DELETE FROM archive_addressed_followup_anchor WHERE profile_scope=? AND chat_id=?",
                    (self.scope, chat),
                )
                return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            return None
        text = payload.get("body") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            return None
        return AddressedFollowupAnchor(str(row["message_id"]), text.strip())

    def recent_context_metadata(self, *, chat_id: str | None = None, limit: int = 100) -> tuple[dict[str, Any], ...]:
        """Return bounded, profile-private provenance only; never prompt content."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise ValueError("context metadata limit must be an integer from 1 through 1000")
        normalized_chat = None if chat_id is None else _normal(chat_id)
        if chat_id is not None and not normalized_chat:
            raise ValueError("context metadata chat identity must be nonempty")
        if not self.path.exists():
            return ()
        where, params = "event.profile_scope = ?", [self.scope]
        if normalized_chat is not None:
            where += " AND event.chat_id = ?"
            params.append(normalized_chat)
        params.append(limit)
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT event.id, event.chat_id, event.sender_id, event.message_id,
                           event.admission, event.payload_json, event.source_timestamp_ms,
                           event.created_at,
                           (SELECT COUNT(*) FROM archive_collision collision
                              WHERE collision.event_id = event.id) AS collision_count
                      FROM archive_event event WHERE {where}
                  ORDER BY event.id DESC LIMIT ?""", params,
            ).fetchall()
            if not rows:
                return ()
            event_ids = [int(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in event_ids)
            attachments: dict[int, list[dict[str, Any]]] = {event_id: [] for event_id in event_ids}
            for attachment in db.execute(
                f"""SELECT event_id, ordinal, descriptor_json, sha256, size, download_status, album_ordinal
                      FROM archive_attachment WHERE event_id IN ({placeholders})
                  ORDER BY event_id DESC, ordinal ASC""", event_ids,
            ):
                try:
                    descriptor = json.loads(attachment["descriptor_json"])
                except (TypeError, ValueError):
                    descriptor = {}
                attachments[int(attachment["event_id"])].append({
                    "ordinal": int(attachment["ordinal"]),
                    "descriptor": self._safe_attachment_descriptor(descriptor),
                    "content_sha256": (
                        str(attachment["sha256"])
                        if isinstance(attachment["sha256"], str)
                        and re.fullmatch(r"[a-f0-9]{64}", attachment["sha256"])
                        else ""
                    ),
                    "size": self._bounded_int(attachment["size"]),
                    "status": str(attachment["download_status"])
                    if attachment["download_status"] in {"owned", "missing_or_rejected", "deleted_or_replaced"}
                    else "unknown",
                    "album_ordinal": self._bounded_int(attachment["album_ordinal"], maximum=1_000_000),
                })
        result = []
        for row in reversed(rows):
            metadata = self._metadata_payload(row["payload_json"])
            result.append({
                "archive_ref": f"archive_event:{int(row['id'])}",
                "sequence": int(row["id"]),
                "chat_id": self._safe_event_identity(row["chat_id"]),
                "sender_id": self._safe_event_identity(row["sender_id"]),
                "message_id": self._safe_event_identity(row["message_id"]),
                "admission": str(row["admission"])
                if row["admission"] in {"drop", "observe", "operate"} else "unknown",
                "source_timestamp_ms": self._bounded_int(row["source_timestamp_ms"]),
                "archived_at_ms": self._safe_archived_at_ms(row["created_at"]),
                "collision_count": int(row["collision_count"]),
                "quote": metadata["quote"],
                "album": metadata["album"],
                "has_media": metadata["has_media"],
                "media_type": metadata["media_type"],
                "attachments": tuple(attachments[int(row["id"])]),
            })
        return tuple(result)

    @staticmethod
    def _receipt_identity(lease: Mapping[str, Any]) -> tuple[str, str, dict[str, object]]:
        """Validate the bridge's fenced lease without retaining arbitrary payload."""
        if not isinstance(lease, Mapping):
            raise ArchiveRejected("missing inbound bridge lease")
        delivery_id = str(lease.get("deliveryId") or "").lower()
        event_digest = str(lease.get("eventDigest") or "").lower()
        consumer_id = str(lease.get("consumerId") or "").strip()
        token = str(lease.get("token") or "")
        try:
            epoch = int(lease.get("epoch"))
        except (TypeError, ValueError) as exc:
            raise ArchiveRejected("invalid inbound bridge lease epoch") from exc
        if (
            len(delivery_id) != 64 or any(char not in "0123456789abcdef" for char in delivery_id)
            or len(event_digest) != 64 or any(char not in "0123456789abcdef" for char in event_digest)
            or not consumer_id or len(consumer_id) > 128 or not token or len(token) > 256 or epoch < 1
        ):
            raise ArchiveRejected("invalid inbound bridge lease")
        return delivery_id, event_digest, {
            "consumerId": consumer_id,
            "deliveryId": delivery_id,
            "epoch": epoch,
            "token": token,
        }

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> BridgeReceipt:
        request: dict[str, object] = {
            "consumerId": str(row["consumer_id"]),
            "deliveryId": str(row["delivery_id"]),
            "epoch": int(row["epoch"]),
            "token": str(row["token"]),
        }
        return BridgeReceipt(
            delivery_id=str(row["delivery_id"]), event_digest=str(row["bridge_event_digest"]),
            request=request, ready=bool(row["ready"]), acknowledged=bool(row["acknowledged"]),
            recovery_pending=bool(row["recovery_pending"]),
            album_key=(str(row["album_key"]) if row["album_key"] else None),
            recovery_key=(str(row["recovery_key"]) if row["recovery_key"] else None),
        )

    def bind_bridge_receipt(self, event_id: int, lease: Mapping[str, Any], *, ready: bool = False) -> BridgeReceipt:
        """Persist/refresh one exact bridge lease after archive commit, before ACK."""
        delivery_id, event_digest, request = self._receipt_identity(lease)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event = db.execute(
                "SELECT id FROM archive_event WHERE id=? AND profile_scope=?", (event_id, self.scope)
            ).fetchone()
            if event is None:
                raise ArchiveRejected("bridge receipt event escaped profile")
            row = db.execute(
                "SELECT * FROM archive_bridge_receipt WHERE profile_scope=? AND delivery_id=?",
                (self.scope, delivery_id),
            ).fetchone()
            if row is not None:
                if str(row["bridge_event_digest"]) != event_digest or int(row["event_id"]) != event_id:
                    db.execute(
                        "INSERT INTO archive_collision(event_id,candidate_digest,created_at) VALUES(?,?,?)",
                        (event_id, event_digest, now),
                    )
                    raise ArchiveRejected("inbound bridge delivery digest collision")
                if not bool(row["acknowledged"]):
                    previous_epoch = int(row["epoch"])
                    previous_consumer = str(row["consumer_id"])
                    previous_token = str(row["token"])
                    if int(request["epoch"]) < previous_epoch:
                        raise ArchiveRejected("stale inbound bridge lease epoch")
                    if int(request["epoch"]) == previous_epoch and (
                        str(request["consumerId"]) != previous_consumer
                        or str(request["token"]) != previous_token
                    ):
                        raise ArchiveRejected("incompatible inbound bridge lease binding")
                    db.execute(
                        """UPDATE archive_bridge_receipt
                           SET consumer_id=?,epoch=?,token=?,ready=MAX(ready,?),updated_at=?
                           WHERE profile_scope=? AND delivery_id=? AND bridge_event_digest=?""",
                        (request["consumerId"], request["epoch"], request["token"], int(ready), now,
                         self.scope, delivery_id, event_digest),
                    )
                    row = db.execute(
                        "SELECT * FROM archive_bridge_receipt WHERE profile_scope=? AND delivery_id=?",
                        (self.scope, delivery_id),
                    ).fetchone()
                return self._receipt_from_row(row)
            db.execute(
                """INSERT INTO archive_bridge_receipt(
                       profile_scope,delivery_id,event_id,bridge_event_digest,consumer_id,epoch,token,ready,acknowledged,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,0,?,?)""",
                (self.scope, delivery_id, event_id, event_digest, request["consumerId"], request["epoch"],
                 request["token"], int(ready), now, now),
            )
            row = db.execute(
                "SELECT * FROM archive_bridge_receipt WHERE profile_scope=? AND delivery_id=?",
                (self.scope, delivery_id),
            ).fetchone()
            return self._receipt_from_row(row)

    def mark_bridge_receipt_ready(self, receipt: BridgeReceipt) -> BridgeReceipt:
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE archive_bridge_receipt SET ready=1,updated_at=?
                   WHERE profile_scope=? AND delivery_id=? AND bridge_event_digest=?
                     AND consumer_id=? AND epoch=? AND token=? AND acknowledged=0""",
                (time.time(), self.scope, receipt.delivery_id, receipt.event_digest,
                 receipt.request["consumerId"], receipt.request["epoch"], receipt.request["token"]),
            )
            if cursor.rowcount != 1:
                raise ArchiveRejected("inbound bridge receipt changed before ready")
            row = db.execute(
                "SELECT * FROM archive_bridge_receipt WHERE profile_scope=? AND delivery_id=?",
                (self.scope, receipt.delivery_id),
            ).fetchone()
            return self._receipt_from_row(row)

    def prepare_bridge_handoff(
        self, receipt: BridgeReceipt, *, album_key: str | None = None, recovery_key: str | None = None,
    ) -> BridgeReceipt:
        """Durably record the post-ACK recovery obligation before bridge ACK.

        A bridge ACK removes the spool record.  This row is consequently the
        handoff fence which a later runtime-recovery slice must inspect: it
        may send one ordinary recovery clarification, but must never replay
        the archived message as a new model turn.
        """
        if album_key is not None:
            album_key = self._validated_album_key(album_key)
        if recovery_key is not None:
            recovery_key = self._validated_album_key(recovery_key)
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE archive_bridge_receipt SET recovery_pending=1,
                       album_key=COALESCE(?,album_key),recovery_key=COALESCE(?,recovery_key),updated_at=?
                   WHERE profile_scope=? AND delivery_id=? AND bridge_event_digest=?
                     AND consumer_id=? AND epoch=? AND token=?
                     AND ready=1 AND acknowledged=0""",
                (album_key, recovery_key, time.time(), self.scope, receipt.delivery_id, receipt.event_digest,
                 receipt.request["consumerId"], receipt.request["epoch"], receipt.request["token"]),
            )
            if cursor.rowcount != 1:
                raise ArchiveRejected("inbound bridge receipt changed before handoff")
            row = db.execute(
                "SELECT * FROM archive_bridge_receipt WHERE profile_scope=? AND delivery_id=?",
                (self.scope, receipt.delivery_id),
            ).fetchone()
            return self._receipt_from_row(row)

    @staticmethod
    def _validated_album_key(album_key: str) -> str:
        album_key = str(album_key or "").lower()
        if len(album_key) != 64 or any(char not in "0123456789abcdef" for char in album_key):
            raise ArchiveRejected("invalid inbound album recovery key")
        return album_key

    def close_album_handoff(self, album_key: str, *, primary_event_id: int) -> bool:
        """Persist a completed native association before ordinary turn dispatch.

        Association metadata has no reliable cardinality on all Baileys paths.
        A later sibling is therefore retained as a linked continuation of this
        completed association rather than reclassified as a new user turn.
        """
        album_key = self._validated_album_key(album_key)
        with self._connect() as db:
            cursor = db.execute(
                """INSERT INTO archive_album_handoff(
                       profile_scope,album_key,operational,state,primary_event_id,closed_at
                   ) VALUES(?,?,1,'closed',?,?)
                   ON CONFLICT(profile_scope,album_key) DO UPDATE SET
                       operational=1,state='closed',
                       primary_event_id=COALESCE(archive_album_handoff.primary_event_id,excluded.primary_event_id),
                       closed_at=excluded.closed_at""",
                (self.scope, album_key, int(primary_event_id), time.time()),
            )
            return cursor.rowcount > 0

    def is_closed_operational_album(self, album_key: str) -> bool:
        album_key = self._validated_album_key(album_key)
        with self._connect() as db:
            row = db.execute(
                """SELECT 1 FROM archive_album_handoff
                   WHERE profile_scope=? AND album_key=? AND state='closed' AND operational=1""",
                (self.scope, album_key),
            ).fetchone()
            return row is not None

    def link_late_album_member(self, album_key: str, event_id: int) -> bool:
        """Associate a post-close native sibling with its original album."""
        album_key = self._validated_album_key(album_key)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            closed = db.execute(
                """SELECT 1 FROM archive_album_handoff
                   WHERE profile_scope=? AND album_key=? AND state='closed' AND operational=1""",
                (self.scope, album_key),
            ).fetchone()
            event = db.execute(
                "SELECT 1 FROM archive_event WHERE profile_scope=? AND id=?",
                (self.scope, int(event_id)),
            ).fetchone()
            if closed is None or event is None:
                return False
            # An already-authorized sibling of an operational association is
            # itself operational, but only as the explicit continuation below;
            # it never re-enters generic ambient admission.
            promoted = db.execute(
                "UPDATE archive_event SET admission='operate' WHERE profile_scope=? AND id=?",
                (self.scope, int(event_id)),
            )
            if promoted.rowcount != 1:
                return False
            cursor = db.execute(
                """INSERT INTO archive_album_late_member(profile_scope,album_key,event_id,linked_at)
                   VALUES(?,?,?,?) ON CONFLICT(profile_scope,album_key,event_id) DO NOTHING""",
                (self.scope, album_key, int(event_id), time.time()),
            )
            if cursor.rowcount == 1:
                return True
            return db.execute(
                """SELECT 1 FROM archive_album_late_member
                   WHERE profile_scope=? AND album_key=? AND event_id=?""",
                (self.scope, album_key, int(event_id)),
            ).fetchone() is not None

    def album_handoff_reply_context(self, album_key: str) -> dict[str, object] | None:
        """Return only the original quote pointer needed by a continuation."""
        album_key = self._validated_album_key(album_key)
        with self._connect() as db:
            row = db.execute(
                """SELECT event.payload_json FROM archive_album_handoff AS handoff
                   JOIN archive_event AS event ON event.id=handoff.primary_event_id
                   WHERE handoff.profile_scope=? AND handoff.album_key=?
                     AND handoff.state='closed' AND handoff.operational=1""",
                (self.scope, album_key),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        message_id = str(payload.get("quotedMessageId") or "").strip()
        text = str(payload.get("quotedText") or "").strip()
        if not message_id or not text:
            return None
        return {
            "message_id": message_id,
            "text": text,
            "is_own": bool(payload.get("quotedOutboundByJack")),
        }

    def settle_bridge_observation(self, receipt: BridgeReceipt, *, exact_only: bool = False) -> bool:
        """Close a silent observation handoff without generating a recovery.

        Whole ambient albums are intentionally settled together.  A late
        sibling of an already-dispatched native album passes ``exact_only`` so
        it cannot clear the original turn's still-live recovery fence.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT recovery_key FROM archive_bridge_receipt
                   WHERE profile_scope=? AND delivery_id=? AND bridge_event_digest=?
                     AND consumer_id=? AND epoch=? AND token=?
                     AND recovery_pending=1 AND acknowledged=1""",
                (self.scope, receipt.delivery_id, receipt.event_digest,
                 receipt.request["consumerId"], receipt.request["epoch"], receipt.request["token"]),
            ).fetchone()
            if row is None:
                return False
            recovery_key = str(row["recovery_key"] or "")
            if recovery_key and not exact_only:
                cursor = db.execute(
                    """UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=?
                       WHERE profile_scope=? AND recovery_key=? AND recovery_pending=1""",
                    (time.time(), self.scope, recovery_key),
                )
            else:
                cursor = db.execute(
                    """UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=?
                       WHERE profile_scope=? AND delivery_id=? AND recovery_pending=1""",
                    (time.time(), self.scope, receipt.delivery_id),
                )
            # A grouped update deliberately clears more than one member; the
            # expected success condition is at least one protected receipt.
            return cursor.rowcount > 0

    def renew_bridge_receipt(self, receipt: BridgeReceipt, lease: Mapping[str, Any]) -> BridgeReceipt:
        delivery_id, event_digest, request = self._receipt_identity(lease)
        if (
            delivery_id != receipt.delivery_id or event_digest != receipt.event_digest
            or request["consumerId"] != receipt.request["consumerId"]
            or request["epoch"] != receipt.request["epoch"] or request["token"] == receipt.request["token"]
        ):
            raise ArchiveRejected("inbound bridge renewal changed receipt identity")
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE archive_bridge_receipt SET token=?,updated_at=?
                   WHERE profile_scope=? AND delivery_id=? AND bridge_event_digest=?
                     AND consumer_id=? AND epoch=? AND token=? AND acknowledged=0""",
                (request["token"], time.time(), self.scope, delivery_id, event_digest,
                 request["consumerId"], request["epoch"], receipt.request["token"]),
            )
            if cursor.rowcount != 1:
                raise ArchiveRejected("inbound bridge receipt changed during renewal")
            row = db.execute(
                "SELECT * FROM archive_bridge_receipt WHERE profile_scope=? AND delivery_id=?",
                (self.scope, delivery_id),
            ).fetchone()
            return self._receipt_from_row(row)

    def mark_bridge_receipt_acked(self, receipt: BridgeReceipt) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE archive_bridge_receipt SET acknowledged=1,updated_at=?
                   WHERE profile_scope=? AND delivery_id=? AND bridge_event_digest=?
                     AND consumer_id=? AND epoch=? AND token=?
                     AND ready=1 AND recovery_pending=1 AND acknowledged=0""",
                (time.time(), self.scope, receipt.delivery_id, receipt.event_digest,
                 receipt.request["consumerId"], receipt.request["epoch"], receipt.request["token"]),
            )
            return cursor.rowcount > 0

    def pending_bridge_receipts(self) -> list[BridgeReceipt]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT * FROM archive_bridge_receipt
                   WHERE profile_scope=? AND ready=1 AND acknowledged=0
                   ORDER BY created_at,delivery_id""", (self.scope,)
            ).fetchall()
            return [self._receipt_from_row(row) for row in rows]

    def pending_bridge_recoveries(self) -> list[BridgeReceipt]:
        """ACKed deliveries that still need the later runtime recovery fence.

        This intentionally returns durable metadata only.  The ingress slice
        does not run a model or emit a recovery message during startup.
        """
        with self._connect() as db:
            # An ACK can outlive this process.  Album membership is written
            # before that ACK, so a restart gets one representative recovery
            # obligation rather than one clarification per photo.  Prefer an
            # operational member when a group has mixed observe/operate
            # admission; an all-observe album remains a single silent item.
            rows = db.execute(
                """SELECT receipt.* FROM archive_bridge_receipt AS receipt
                   JOIN archive_event AS event ON event.id=receipt.event_id
                   WHERE receipt.profile_scope=?
                     AND receipt.recovery_pending=1 AND receipt.acknowledged=1
                     AND (
                       receipt.recovery_key IS NULL OR receipt.delivery_id=(
                         SELECT candidate.delivery_id
                         FROM archive_bridge_receipt AS candidate
                         JOIN archive_event AS candidate_event ON candidate_event.id=candidate.event_id
                         WHERE candidate.profile_scope=receipt.profile_scope
                           AND candidate.recovery_key=receipt.recovery_key
                           AND candidate.recovery_pending=1 AND candidate.acknowledged=1
                         ORDER BY CASE candidate_event.admission WHEN 'operate' THEN 0 ELSE 1 END,
                                  candidate.updated_at,candidate.delivery_id
                         LIMIT 1
                       )
                     )
                   ORDER BY receipt.updated_at,receipt.delivery_id""", (self.scope,)
            ).fetchall()
            return [self._receipt_from_row(row) for row in rows]

    def bridge_recovery(self, delivery_id: str) -> BridgeRecovery | None:
        """Return a pending recovery's durable state without claiming it.

        Startup uses this after a crash between archive registration and the
        ordinary final ledger write.  It is intentionally metadata-only: no
        inbound text, media, or model execution can be recovered through this
        method.
        """
        delivery_id = str(delivery_id or "").lower()
        if len(delivery_id) != 64 or any(char not in "0123456789abcdef" for char in delivery_id):
            raise ArchiveRejected("invalid bridge recovery delivery")
        with self._connect() as db:
            row = db.execute(
                """SELECT receipt.delivery_id, receipt.bridge_event_digest, receipt.event_id,
                          event.payload_json, event.chat_id, recovery.generation,
                          recovery.state, recovery.obligation_id
                   FROM archive_bridge_receipt AS receipt
                   JOIN archive_event AS event ON event.id=receipt.event_id
                   LEFT JOIN archive_bridge_recovery AS recovery
                     ON recovery.profile_scope=receipt.profile_scope
                    AND recovery.delivery_id=receipt.delivery_id
                   WHERE receipt.profile_scope=? AND receipt.delivery_id=?
                     AND receipt.recovery_pending=1 AND receipt.acknowledged=1""",
                (self.scope, delivery_id),
            ).fetchone()
            if row is None or row["state"] is None:
                return None
            payload = json.loads(str(row["payload_json"]))
            chat_id = str(row["chat_id"] or "").strip()
            if not chat_id:
                raise ArchiveRejected("missing bridge recovery chat")
            return BridgeRecovery(
                delivery_id=str(row["delivery_id"]), event_digest=str(row["bridge_event_digest"]),
                event_id=int(row["event_id"]), generation=int(row["generation"]), chat_id=chat_id,
                is_group=bool(payload.get("isGroup")) or chat_id.lower().endswith("@g.us"),
                state=str(row["state"]),
                obligation_id=(str(row["obligation_id"]) if row["obligation_id"] else None),
            )

    def bridge_handoff_admission(self, delivery_id: str) -> str | None:
        """Admission stored with an ACKed handoff, without exposing its body.

        The adapter preserves the ingress handoff for every authorised record.
        Startup consumes ``observe`` records silently, while only an
        ``operate`` record may enter the no-model direct recovery path.
        """
        delivery_id = str(delivery_id or "").lower()
        if len(delivery_id) != 64 or any(char not in "0123456789abcdef" for char in delivery_id):
            raise ArchiveRejected("invalid bridge recovery delivery")
        with self._connect() as db:
            row = db.execute(
                """SELECT event.admission
                   FROM archive_bridge_receipt AS receipt
                   JOIN archive_event AS event ON event.id=receipt.event_id
                   WHERE receipt.profile_scope=? AND receipt.delivery_id=?
                     AND receipt.recovery_pending=1 AND receipt.acknowledged=1""",
                (self.scope, delivery_id),
            ).fetchone()
        if row is None:
            return None
        admission = str(row["admission"] or "")
        if admission not in {"observe", "operate"}:
            raise ArchiveRejected("invalid bridge recovery admission")
        return admission

    def discard_bridge_recovery(self, delivery_id: str, *, state: str = "held_silent") -> bool:
        """Close a no-turn bridge handoff without generating outbound text.

        Observe-only and invalid events have no agent work to recover.  Keeping
        their receipt pending would later manufacture a direct-message
        clarification even though the inbound path intentionally stayed silent.
        """
        delivery_id = str(delivery_id or "").lower()
        if len(delivery_id) != 64 or any(char not in "0123456789abcdef" for char in delivery_id):
            raise ArchiveRejected("invalid bridge recovery delivery")
        if state not in {"held_silent", "held_group"}:
            raise ArchiveRejected("invalid bridge recovery disposition")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute(
                """SELECT recovery_key FROM archive_bridge_receipt
                   WHERE profile_scope=? AND delivery_id=?
                     AND recovery_pending=1 AND acknowledged=1""",
                (self.scope, delivery_id),
            ).fetchone()
            if receipt is None:
                return False
            now = time.time()
            disposition = db.execute(
                """INSERT INTO archive_bridge_recovery(
                       profile_scope,delivery_id,generation,state,created_at,updated_at)
                   VALUES(?,?,1,?,?,?)
                   ON CONFLICT(profile_scope,delivery_id) DO UPDATE SET
                       state=excluded.state,updated_at=excluded.updated_at
                   WHERE archive_bridge_recovery.state IN ('reserved','held_silent','held_group')
                     AND archive_bridge_recovery.obligation_id IS NULL""",
                (self.scope, delivery_id, state, now, now),
            )
            if disposition.rowcount != 1:
                return False
            recovery_key = str(receipt["recovery_key"] or "")
            if recovery_key:
                cursor = db.execute(
                    """UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=?
                       WHERE profile_scope=? AND recovery_key=? AND recovery_pending=1""",
                    (now, self.scope, recovery_key),
                )
            else:
                cursor = db.execute(
                    """UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=?
                       WHERE profile_scope=? AND delivery_id=? AND recovery_pending=1""",
                    (now, self.scope, delivery_id),
                )
            return cursor.rowcount > 0

    @staticmethod
    def _recovery_owner_stamp() -> tuple[int, int | None]:
        """Use pid + start time so a recycled pid cannot steal a live handoff."""
        pid = os.getpid()
        try:
            from gateway.status import get_process_start_time
            return pid, get_process_start_time(pid)
        except Exception:
            return pid, None

    @staticmethod
    def _recovery_owner_alive(pid: Any, started_at: Any) -> bool:
        try:
            from gateway.delivery_ledger import _owner_alive
            return bool(_owner_alive(pid, started_at))
        except Exception:
            return False

    def _recovery_from_row(self, row: sqlite3.Row) -> BridgeRecovery:
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArchiveRejected("invalid archived bridge recovery payload") from exc
        if not isinstance(payload, Mapping):
            raise ArchiveRejected("invalid archived bridge recovery payload")
        chat_id = str(row["chat_id"] or "").strip()
        if not chat_id:
            raise ArchiveRejected("missing bridge recovery chat")
        is_group = bool(payload.get("isGroup")) or chat_id.lower().endswith("@g.us")
        return BridgeRecovery(
            delivery_id=str(row["delivery_id"]),
            event_digest=str(row["bridge_event_digest"]), event_id=int(row["event_id"]),
            generation=int(row["generation"]), chat_id=chat_id, is_group=is_group,
            state=str(row["state"]), obligation_id=(str(row["obligation_id"]) if row["obligation_id"] else None),
        )

    def reserve_bridge_recovery(self, delivery_id: str) -> BridgeRecovery | None:
        """Claim one ACKed handoff for a no-model recovery clarification.

        Only ACKed ``recovery_pending`` records are eligible.  Group records
        are settled silently; a direct record can be owned by one live startup
        worker, and a dead worker is generation-rearmed under ``BEGIN
        IMMEDIATE``.  This method never sends or invokes a model.
        """
        delivery_id = str(delivery_id or "").lower()
        if len(delivery_id) != 64 or any(char not in "0123456789abcdef" for char in delivery_id):
            raise ArchiveRejected("invalid bridge recovery delivery")
        owner_pid, owner_started = self._recovery_owner_stamp()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT receipt.*, event.payload_json, event.chat_id,
                          recovery.generation, recovery.state, recovery.obligation_id,
                          recovery.owner_pid, recovery.owner_started_at
                   FROM archive_bridge_receipt AS receipt
                   JOIN archive_event AS event ON event.id=receipt.event_id
                   LEFT JOIN archive_bridge_recovery AS recovery
                     ON recovery.profile_scope=receipt.profile_scope
                    AND recovery.delivery_id=receipt.delivery_id
                   WHERE receipt.profile_scope=? AND receipt.delivery_id=?
                     AND receipt.recovery_pending=1 AND receipt.acknowledged=1""",
                (self.scope, delivery_id),
            ).fetchone()
            if row is None:
                return None
            # SQLite aliases duplicate receipt columns, so construct a narrow
            # source row for the public recovery value instead of relying on
            # ambiguous mapping keys.
            payload = json.loads(str(row["payload_json"]))
            chat_id = str(row["chat_id"] or "").strip()
            is_group = bool(payload.get("isGroup")) or chat_id.lower().endswith("@g.us")
            now = time.time()
            if row["state"] is None:
                if is_group:
                    db.execute(
                        "INSERT INTO archive_bridge_recovery(profile_scope,delivery_id,generation,state,created_at,updated_at) VALUES(?,?,1,'held_group',?,?)",
                        (self.scope, delivery_id, now, now),
                    )
                    recovery_key = str(row["recovery_key"] or "")
                    if recovery_key:
                        db.execute(
                            "UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=? "
                            "WHERE profile_scope=? AND recovery_key=? AND recovery_pending=1",
                            (now, self.scope, recovery_key),
                        )
                    else:
                        db.execute(
                            "UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=? "
                            "WHERE profile_scope=? AND delivery_id=?",
                            (now, self.scope, delivery_id),
                        )
                    return None
                db.execute(
                    "INSERT INTO archive_bridge_recovery(profile_scope,delivery_id,generation,state,owner_pid,owner_started_at,created_at,updated_at) VALUES(?,?,1,'reserved',?,?,?,?)",
                    (self.scope, delivery_id, owner_pid, owner_started, now, now),
                )
                return BridgeRecovery(delivery_id, str(row["bridge_event_digest"]), int(row["event_id"]), 1, chat_id, False, "reserved", None)
            state = str(row["state"])
            if state != "reserved":
                return None
            if self._recovery_owner_alive(row["owner_pid"], row["owner_started_at"]):
                return None
            generation = int(row["generation"]) + 1
            cursor = db.execute(
                """UPDATE archive_bridge_recovery
                   SET generation=?,owner_pid=?,owner_started_at=?,updated_at=?
                   WHERE profile_scope=? AND delivery_id=? AND generation=?
                     AND state='reserved' AND obligation_id IS NULL""",
                (generation, owner_pid, owner_started, now, self.scope, delivery_id, int(row["generation"])),
            )
            if cursor.rowcount != 1:
                return None
            return BridgeRecovery(delivery_id, str(row["bridge_event_digest"]), int(row["event_id"]), generation, chat_id, False, "reserved", None)

    def register_bridge_recovery_delivery(self, recovery: BridgeRecovery, obligation_id: str) -> bool:
        """Atomically bind the ordinary delivery-ledger row to this recovery."""
        if recovery.is_group or not obligation_id:
            return False
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE archive_bridge_recovery SET state='delivery_registered',obligation_id=?,updated_at=?
                   WHERE profile_scope=? AND delivery_id=? AND generation=?
                     AND state='reserved' AND obligation_id IS NULL""",
                (str(obligation_id), time.time(), self.scope, recovery.delivery_id, recovery.generation),
            )
            return cursor.rowcount == 1

    def settle_bridge_recovery_delivery(self, recovery: BridgeRecovery, obligation_id: str) -> bool:
        """Close the archived handoff only after its exact final is accounted for."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                """UPDATE archive_bridge_recovery SET state='delivered',updated_at=?
                   WHERE profile_scope=? AND delivery_id=? AND generation=?
                     AND state='delivery_registered' AND obligation_id=?""",
                (time.time(), self.scope, recovery.delivery_id, recovery.generation, str(obligation_id)),
            )
            if cursor.rowcount != 1:
                return False
            receipt = db.execute(
                "SELECT recovery_key FROM archive_bridge_receipt WHERE profile_scope=? AND delivery_id=?",
                (self.scope, recovery.delivery_id),
            ).fetchone()
            recovery_key = str(receipt["recovery_key"] or "") if receipt is not None else ""
            if recovery_key:
                db.execute(
                    "UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=? "
                    "WHERE profile_scope=? AND recovery_key=? AND recovery_pending=1",
                    (time.time(), self.scope, recovery_key),
                )
            else:
                db.execute(
                    "UPDATE archive_bridge_receipt SET recovery_pending=0,updated_at=? WHERE profile_scope=? AND delivery_id=? AND recovery_pending=1",
                    (time.time(), self.scope, recovery.delivery_id),
                )
            return True

    def _verify_owned_object(self, owned_path: str | None, digest: str | None, size: int | None) -> bool:
        if not owned_path or not digest or size is None:
            return False
        target = Path(owned_path)
        if target.parent != self.media_root or target.name != digest:
            return False
        self._validate_paths()
        if not os.path.lexists(target):
            return False
        try:
            fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError:
            return False
        target_hash = hashlib.sha256(); target_size = 0
        with os.fdopen(fd, "rb") as existing:
            before = os.fstat(existing.fileno())
            if (not stat.S_ISREG(before.st_mode)
                    or stat.S_IMODE(before.st_mode) != 0o600):
                return False
            while chunk := existing.read(1024 * 1024):
                target_hash.update(chunk); target_size += len(chunk)
            after = os.fstat(existing.fileno())
        # The path must still name the exact verified descriptor: a replacement
        # after open (including a symlink swap) cannot become an owned manifest.
        try:
            named = os.stat(target, follow_symlinks=False)
        except OSError:
            return False
        return (
            stat.S_ISREG(named.st_mode)
            and stat.S_IMODE(named.st_mode) == 0o600
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                == (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns)
            and target_hash.hexdigest() == digest
            and target_size == size
        )

    def materialize(
        self,
        event_id: int,
        raw: Mapping[str, Any],
        paths: list[str | None],
        *,
        expected_attachment_count: int | None = None,
    ) -> MaterializationResult:
        """Persist one ordered manifest row per inbound attachment slot.

        ``paths`` may contain ``None`` for a bridge-rejected slot.  The slot
        remains durable evidence of incomplete media rather than disappearing
        when the adapter filters an unsafe source path.
        """
        slot_count = (
            expected_attachment_count
            if expected_attachment_count is not None
            else (len(paths) or (1 if raw.get("hasMedia") else 0))
        )
        if not isinstance(slot_count, int) or slot_count < 0 or len(paths) > slot_count:
            raise ArchiveRejected("invalid attachment slot count")
        descriptors = [{"kind": raw.get("mediaType") or "", "mime": raw.get("mime") or "", "file_name": raw.get("fileName") or ""} for _ in range(slot_count)]
        statuses: list[str] = []
        manifest_paths: list[str | None] = []
        with self._connect() as db:
            for ordinal, descriptor in enumerate(descriptors):
                descriptor_json = _canonical(descriptor)
                existing = db.execute("SELECT descriptor_json,owned_path,sha256,size,download_status FROM archive_attachment WHERE event_id=? AND ordinal=?", (event_id, ordinal)).fetchone()
                if existing and existing["download_status"] == "owned":
                    if existing["descriptor_json"] != descriptor_json:
                        raise ArchiveRejected("owned attachment identity changed")
                    if not self._verify_owned_object(existing["owned_path"], existing["sha256"], existing["size"]):
                        raise ArchiveRejected("owned attachment is no longer verified")
                    statuses.append("owned")
                    manifest_paths.append(existing["owned_path"])
                    continue
                candidate = paths[ordinal] if ordinal < len(paths) else None
                path = Path(candidate) if isinstance(candidate, str) and os.path.isabs(candidate) else None
                status, owned, digest, size = "missing_or_rejected", None, None, None
                if path:
                    try:
                        resolved = path.resolve(strict=True)
                        if path.is_symlink() or not any(root == resolved.parent or root in resolved.parents for root in self.cache_roots) or not resolved.is_file(): raise ArchiveRejected("cache path outside profile root")
                        source_fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)); before = os.fstat(source_fd)
                        if not stat.S_ISREG(before.st_mode): os.close(source_fd); raise ArchiveRejected("nonregular source")
                        hasher = hashlib.sha256(); copied = 0; staging = self.media_root / f".{time.time_ns()}.part"
                        try:
                            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                            with os.fdopen(source_fd, "rb") as src, os.fdopen(fd, "wb") as dst:
                                while chunk := src.read(1024 * 1024): hasher.update(chunk); copied += len(chunk); dst.write(chunk)
                                dst.flush(); os.fsync(dst.fileno())
                                after = os.fstat(src.fileno())
                            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or copied != before.st_size: raise ArchiveRejected("source changed")
                            digest, size = hasher.hexdigest(), copied; target = self.media_root / digest
                            self._validate_paths()
                            try:
                                # Link publication cannot replace a competing winner.  The
                                # staging link is removed below, leaving one owned object.
                                os.link(staging, target)
                            except FileExistsError:
                                pass
                            if not os.path.lexists(target):
                                raise ArchiveRejected("missing archive object")
                            if not self._verify_owned_object(str(target), digest, copied):
                                raise ArchiveRejected("corrupt archive object")
                            # Both the publishing winner and a deduplicating loser need
                            # a durable directory entry before SQLite can reference it.
                            fd = os.open(self.media_root, os.O_RDONLY)
                            try:
                                os.fsync(fd)
                            finally:
                                os.close(fd)
                            owned, status = str(target), "owned"
                        finally:
                            staging.unlink(missing_ok=True)
                    except (OSError, ArchiveRejected): status = "deleted_or_replaced"
                db.execute("""INSERT INTO archive_attachment(event_id,ordinal,descriptor_json,owned_path,sha256,size,download_status,album_ordinal)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(event_id,ordinal) DO UPDATE SET descriptor_json=excluded.descriptor_json,
                    owned_path=excluded.owned_path,sha256=excluded.sha256,size=excluded.size,
                    download_status=excluded.download_status,album_ordinal=excluded.album_ordinal
                    WHERE archive_attachment.download_status != 'owned'""",
                    (event_id,ordinal,descriptor_json,owned,digest,size,status,ordinal))
                statuses.append(status)
                manifest_paths.append(owned)
        complete = all(status == "owned" for status in statuses)
        return MaterializationResult(
            complete=complete,
            owned_count=sum(status == "owned" for status in statuses),
            attachment_count=len(statuses),
            statuses=tuple(statuses),
            owned_paths=tuple(str(path) for path in manifest_paths) if complete else (),
            owned_descriptors=tuple(dict(descriptor) for descriptor in descriptors) if complete else (),
        )

    def materialized_message_attachments(
        self, chat_id: Any, message_id: Any,
    ) -> MaterializationResult | None:
        """Return a verified, profile-scoped manifest for an earlier inbound message.

        WhatsApp quote stubs do not contain original attachment bytes.  A
        reply may reuse only the already-owned archive object for its exact
        chat/message identity; bridge-cache paths are neither durable nor an
        agent-visible authority.
        """
        chat, message = _normal(chat_id), str(message_id or "").strip()
        if not chat or not message:
            return None
        with self._connect() as db:
            event = db.execute(
                """SELECT id FROM archive_event
                   WHERE profile_scope=? AND chat_id=? AND message_id=?""",
                (self.scope, chat, message),
            ).fetchone()
            if event is None:
                return None
            rows = db.execute(
                """SELECT ordinal,descriptor_json,owned_path,sha256,size,download_status
                   FROM archive_attachment WHERE event_id=? ORDER BY ordinal""",
                (int(event["id"]),),
            ).fetchall()
        if not rows:
            return None
        descriptors: list[dict[str, str]] = []
        paths: list[str] = []
        for row in rows:
            if str(row["download_status"]) != "owned":
                return None
            try:
                descriptor = json.loads(str(row["descriptor_json"]))
            except (TypeError, ValueError):
                return None
            if not isinstance(descriptor, dict) or not self._verify_owned_object(
                row["owned_path"], row["sha256"], row["size"],
            ):
                return None
            paths.append(str(row["owned_path"]))
            descriptors.append(dict(descriptor))
        return MaterializationResult(
            complete=True,
            owned_count=len(paths),
            attachment_count=len(paths),
            statuses=("owned",) * len(paths),
            owned_paths=tuple(paths),
            owned_descriptors=tuple(descriptors),
        )
