"""Private, profile-scoped WhatsApp inbound archive; never a prompt projection."""
from __future__ import annotations

import hashlib
import json
import os
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
        self.media_root = self.root / "media"
        self.media_root.mkdir(exist_ok=True, mode=0o700); os.chmod(self.media_root, 0o700)
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
