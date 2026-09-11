"""Private, profile-scoped WhatsApp inbound archive; never a prompt projection."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from pathlib import Path
from typing import Any, Mapping


class ArchiveRejected(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


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
        self.scope = hashlib.sha256(str(self.profile_home).encode()).hexdigest()
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS archive_event (id INTEGER PRIMARY KEY, profile_scope TEXT NOT NULL, chat_id TEXT NOT NULL, sender_id TEXT NOT NULL, message_id TEXT NOT NULL, admission TEXT NOT NULL, event_digest TEXT NOT NULL, payload_json TEXT NOT NULL, album_group TEXT, album_role TEXT, album_index INTEGER, created_at REAL NOT NULL, UNIQUE(profile_scope,chat_id,message_id));
            CREATE TABLE IF NOT EXISTS archive_attachment (event_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, descriptor_json TEXT NOT NULL, owned_path TEXT, sha256 TEXT, size INTEGER, download_status TEXT NOT NULL, album_ordinal INTEGER, PRIMARY KEY(event_id,ordinal));
            CREATE TABLE IF NOT EXISTS archive_collision (id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, candidate_digest TEXT NOT NULL, created_at REAL NOT NULL);
            """)
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
        for path in [*chain, self.root / "archive.sqlite3", self.root / "archive.sqlite3-wal", self.root / "archive.sqlite3-shm"]:
            if os.path.lexists(path) and path.is_symlink(): raise ArchiveRejected("unsafe archive symlink")

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

    def record(self, raw: Mapping[str, Any], admission: str) -> tuple[int, bool]:
        if admission not in {"drop", "observe", "operate"}: raise ArchiveRejected("invalid admission")
        data = self._payload(raw); chat, sender = map(_normal, (data.get("chatId"), data.get("senderId"))); mid = str(data.get("messageId") or "").strip()
        if not all((chat, sender, mid)): raise ArchiveRejected("missing stable archive identity")
        digest = _digest(data); album = data.get("album") if isinstance(data.get("album"), dict) else {}
        with self._connect() as db:
            row = db.execute("SELECT id,event_digest FROM archive_event WHERE profile_scope=? AND chat_id=? AND message_id=?", (self.scope,chat,mid)).fetchone()
            if row:
                if row["event_digest"] != digest:
                    db.execute("INSERT INTO archive_collision(event_id,candidate_digest,created_at) VALUES(?,?,?)", (row["id"],digest,time.time()))
                    return int(row["id"]), False
                return int(row["id"]), True
            cur = db.execute("INSERT INTO archive_event(profile_scope,chat_id,sender_id,message_id,admission,event_digest,payload_json,album_group,album_role,album_index,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (self.scope,chat,sender,mid,admission,digest,_canonical(data),album.get("groupId"),album.get("role"),album.get("messageIndex"),time.time()))
            return int(cur.lastrowid), True

    def materialize(self, event_id: int, raw: Mapping[str, Any], paths: list[str]) -> None:
        descriptors = [{"kind": raw.get("mediaType") or "", "mime": raw.get("mime") or "", "file_name": raw.get("fileName") or ""} for _ in paths] or ([{"kind": raw.get("mediaType") or "", "mime": raw.get("mime") or "", "file_name": raw.get("fileName") or ""}] if raw.get("hasMedia") else [])
        with self._connect() as db:
            for ordinal, descriptor in enumerate(descriptors):
                path = Path(paths[ordinal]) if ordinal < len(paths) and os.path.isabs(paths[ordinal]) else None
                status, owned, digest, size = "not_requested", None, None, None
                if path:
                    try:
                        resolved = path.resolve(strict=True)
                        if path.is_symlink() or not any(root == resolved.parent or root in resolved.parents for root in self.cache_roots) or not resolved.is_file(): raise ArchiveRejected("cache path outside profile root")
                        content = resolved.read_bytes(); digest, size, owned, status = hashlib.sha256(content).hexdigest(), len(content), str(resolved), "owned"
                    except (OSError, ArchiveRejected): status = "deleted_or_replaced"
                db.execute("INSERT OR REPLACE INTO archive_attachment(event_id,ordinal,descriptor_json,owned_path,sha256,size,download_status,album_ordinal) VALUES(?,?,?,?,?,?,?,?)", (event_id,ordinal,_canonical(descriptor),owned,digest,size,status,ordinal))
