import path from 'path';
import {
  chmodSync,
  closeSync,
  constants,
  existsSync,
  fchmodSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'fs';
import { createHash, randomBytes, timingSafeEqual } from 'crypto';

const SCHEMA_VERSION = 1;
const DEFAULT_LEASE_MS = 30_000;
const MIN_LEASE_MS = 1_000;
const MAX_LEASE_MS = 5 * 60_000;
const DEFAULT_LIMIT = 100;
const DEFAULT_TOMBSTONE_LIMIT = 10_000;
const DEFAULT_TOMBSTONE_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

function canonicalValue(value) {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === 'object') {
    const result = {};
    for (const key of Object.keys(value).sort()) {
      if (value[key] !== undefined) result[key] = canonicalValue(value[key]);
    }
    return result;
  }
  if (typeof value === 'bigint') return value.toString();
  return value;
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalValue(value));
}

function canonicalDigestNumber(value) {
  if (!Number.isFinite(value)) throw new Error('Inbound event contains a non-finite number');
  if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
    throw new Error('Inbound event contains an unsafe integer');
  }
  return JSON.stringify(value);
}

export function canonicalDigestJson(value) {
  if (value === null) return 'null';
  if (typeof value === 'number') return canonicalDigestNumber(value);
  if (typeof value === 'bigint') return JSON.stringify(value.toString());
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map(item => (
      item === undefined || typeof item === 'function' || typeof item === 'symbol'
        ? 'null'
        : canonicalDigestJson(item)
    )).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    const fields = Object.keys(value).sort().flatMap(key => {
      const item = value[key];
      if (item === undefined || typeof item === 'function' || typeof item === 'symbol') return [];
      return [`${JSON.stringify(key)}:${canonicalDigestJson(item)}`];
    });
    return `{${fields.join(',')}}`;
  }
  throw new Error('Inbound event contains a non-JSON value');
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

export function opaqueInboundProfileNamespace(sessionDir) {
  const source = String(sessionDir || '').trim();
  return source ? sha256(path.resolve(source)) : null;
}

export function opaqueInboundAccountNamespace(accountIds) {
  const known = Array.from(new Set(
    (accountIds || []).map(value => String(value || '').trim().toLowerCase()).filter(Boolean),
  )).sort();
  return known.length ? sha256(known.join('|')) : null;
}

export function stageInboundEventSafely({
  spool,
  profileNamespace,
  accountNamespace,
  event,
  onFailure = () => {},
}) {
  if (!accountNamespace) {
    onFailure('account_not_ready');
    return null;
  }
  try {
    return spool.append({ profileNamespace, accountNamespace, event });
  } catch {
    onFailure('persistence_error');
    return null;
  }
}

function orderedUnownedMediaManifest(event) {
  const sourcePaths = Array.isArray(event?.mediaUrls) ? event.mediaUrls : [];
  const declared = Array.isArray(event?.mediaMetadata) ? event.mediaMetadata : [];
  const entryCount = Math.max(sourcePaths.length, declared.length, event?.hasMedia ? 1 : 0);
  const stableEntries = Array.from({ length: entryCount }, (_, index) => {
    const metadata = declared[index] && typeof declared[index] === 'object' ? declared[index] : {};
    const declaredSha256 = /^[a-f0-9]{64}$/i.test(String(metadata.sha256 || ''))
      ? String(metadata.sha256).toLowerCase()
      : '';
    return {
      index,
      mediaType: String(metadata.mediaType || event?.mediaType || ''),
      mime: String(metadata.mime || event?.mime || ''),
      fileName: String(metadata.fileName || event?.fileName || ''),
      declaredSha256,
      declaredSize: String(metadata.size ?? ''),
    };
  });
  const entries = stableEntries.map((entry, index) => ({
    ...entry,
    state: 'unowned',
    sourcePath: String(sourcePaths[index] || ''),
  }));
  const integrity = entries.length && entries.every(entry => entry.declaredSha256)
    ? 'declared_sha256'
    : 'unverified';
  return {
    state: entries.length ? 'unowned' : 'no_media',
    owner: 'none',
    ownedRoot: null,
    integrity,
    orderedMetadataDigest: sha256(canonicalDigestJson(stableEntries)),
    entries,
  };
}

export function inboundEventDigest(event) {
  const payload = { ...(event || {}) };
  // These are transient bridge-cache references. Node durably records them
  // for a later OPERATE transaction but never reads, hashes, copies, or owns
  // their contents during pre-admission staging.
  delete payload.mediaUrls;
  delete payload.mediaMetadata;
  delete payload._inboundLease;
  if (payload.chatId !== undefined) payload.chatId = normalizeJid(payload.chatId);
  if (payload.senderId !== undefined) payload.senderId = normalizeJid(payload.senderId);
  if (payload.readReceiptKey && typeof payload.readReceiptKey === 'object') {
    payload.readReceiptKey = {
      ...payload.readReceiptKey,
      remoteJid: normalizeJid(payload.readReceiptKey.remoteJid),
      participant: normalizeJid(payload.readReceiptKey.participant),
    };
  }
  const mediaManifest = orderedUnownedMediaManifest(event);
  return {
    eventDigest: sha256(canonicalDigestJson({
      event: payload,
      orderedUnownedMediaMetadataDigest: mediaManifest.orderedMetadataDigest,
    })),
    mediaManifest,
  };
}

function normalizeJid(value) {
  const raw = String(value || '').trim().toLowerCase();
  if (!raw) return '';
  const at = raw.lastIndexOf('@');
  if (at < 0) return raw.split(':', 1)[0];
  const local = raw.slice(0, at).split(':', 1)[0];
  const domain = raw.slice(at + 1);
  return `${local}@${domain}`;
}

function safeTokenEqual(left, right) {
  const a = Buffer.from(String(left || ''), 'utf8');
  const b = Buffer.from(String(right || ''), 'utf8');
  return a.length === b.length && timingSafeEqual(a, b);
}

function fsyncDirectory(dir) {
  let fd;
  try {
    assertNoSymlinkComponents(dir);
    fd = openSync(
      dir,
      constants.O_RDONLY | (constants.O_DIRECTORY || 0) | (constants.O_NOFOLLOW || 0),
    );
    if (!fstatSync(fd).isDirectory()) throw new Error(`Cannot fsync non-directory: ${dir}`);
    fsyncSync(fd);
  } catch (err) {
    // Some filesystems do not implement directory fsync. Permission and I/O
    // failures are not equivalent and must stop durable staging.
    if (!['EINVAL', 'ENOTSUP', 'EOPNOTSUPP'].includes(err?.code)) throw err;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

function assertNoSymlinkComponents(target) {
  const absolute = path.resolve(target);
  const parsed = path.parse(absolute);
  let current = parsed.root;
  for (const component of absolute.slice(parsed.root.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, component);
    if (existsSync(current)) {
      const info = lstatSync(current);
      if (info.isSymbolicLink() && info.uid !== 0) {
        throw new Error(`Inbound spool path contains an untrusted symbolic-link component: ${current}`);
      }
    }
  }
}

function ensurePrivateDirectory(dir) {
  assertNoSymlinkComponents(dir);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  assertNoSymlinkComponents(dir);
  const flags = constants.O_RDONLY | (constants.O_DIRECTORY || 0) | (constants.O_NOFOLLOW || 0);
  const fd = openSync(dir, flags);
  try {
    const before = fstatSync(fd);
    if (!before.isDirectory()) throw new Error(`Inbound spool path is not a directory: ${dir}`);
    if (typeof process.geteuid === 'function' && before.uid !== process.geteuid()) {
      throw new Error(`Inbound spool directory is not owned by the current user: ${dir}`);
    }
    fchmodSync(fd, 0o700);
    if ((fstatSync(fd).mode & 0o777) !== 0o700) {
      throw new Error(`Inbound spool directory is not private: ${dir}`);
    }
  } finally {
    closeSync(fd);
  }
}

function atomicWriteJson(filePath, value) {
  const dir = path.dirname(filePath);
  ensurePrivateDirectory(dir);
  const tempPath = path.join(
    dir,
    `.${path.basename(filePath)}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`,
  );
  const fd = openSync(tempPath, 'wx', 0o600);
  try {
    writeFileSync(fd, `${canonicalJson(value)}\n`, { encoding: 'utf8' });
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  renameSync(tempPath, filePath);
  chmodSync(filePath, 0o600);
  const info = lstatSync(filePath);
  if (info.isSymbolicLink() || !info.isFile() || (info.mode & 0o777) !== 0o600) {
    throw new Error(`Inbound spool durable file is not a private regular file: ${filePath}`);
  }
  fsyncDirectory(dir);
}

function readJson(filePath) {
  const info = lstatSync(filePath);
  if (info.isSymbolicLink() || !info.isFile()) {
    throw new Error(`Inbound spool data path is not a regular file: ${filePath}`);
  }
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

function jsonFiles(dir) {
  if (!existsSync(dir)) return [];
  ensurePrivateDirectory(dir);
  return readdirSync(dir).filter(name => name.endsWith('.json')).sort();
}

function requireOpaqueNamespace(name, value) {
  const text = String(value || '').trim();
  if (!/^[a-f0-9]{64}$/i.test(text)) {
    throw new Error(`${name} must be a 64-character opaque hexadecimal namespace`);
  }
  return text.toLowerCase();
}

function normalizeDeliveryId(value) {
  const text = String(value || '').trim().toLowerCase();
  return /^[a-f0-9]{64}$/.test(text) ? text : null;
}

function requireConsumerId(value) {
  const text = String(value || '').trim();
  if (!/^[a-zA-Z0-9._:-]{1,128}$/.test(text)) {
    throw new Error('consumerId is required and must be an opaque transport identifier');
  }
  return text;
}

function boundedLeaseMs(value, fallback) {
  const parsed = Number(value);
  const selected = Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  return Math.max(MIN_LEASE_MS, Math.min(selected, MAX_LEASE_MS));
}

export function inboundIdentity({ profileNamespace, accountNamespace, event }) {
  const profile = requireOpaqueNamespace('profileNamespace', profileNamespace);
  const account = requireOpaqueNamespace('accountNamespace', accountNamespace);
  const readKey = event?.readReceiptKey || {};
  const identity = {
    profile,
    account,
    chatId: normalizeJid(readKey.remoteJid || event?.chatId),
    messageId: String(readKey.id || event?.messageId || '').trim(),
    participant: normalizeJid(readKey.participant || event?.senderId || readKey.remoteJid),
    fromMe: Boolean(readKey.fromMe ?? event?.fromMe),
  };
  if (!identity.chatId || !identity.messageId || !identity.participant) {
    throw new Error('Inbound WhatsApp event requires chatId, messageId, and participant');
  }
  return identity;
}

function publicLeaseMetadata(record) {
  return {
    schemaVersion: SCHEMA_VERSION,
    consumerId: record.lease.consumerId,
    deliveryId: record.deliveryId,
    eventDigest: record.eventDigest,
    mediaManifest: record.mediaManifest,
    sequence: record.sequence,
    epoch: record.lease.epoch,
    token: record.lease.token,
    expiresAt: record.lease.expiresAt,
    attempt: record.lease.attempt,
  };
}

function publicLease(record) {
  // Flat event fields preserve the historical GET /messages list contract.
  // Lease metadata is deliberately namespaced for the Python durability/ACK
  // transaction and overwrites any untrusted inbound field of the same name.
  return { ...record.event, _inboundLease: publicLeaseMetadata(record) };
}

export class InboundSpool {
  constructor(rootDir, {
    now = () => Date.now(),
    leaseMs = DEFAULT_LEASE_MS,
    tombstoneLimit = DEFAULT_TOMBSTONE_LIMIT,
    tombstoneMaxAgeMs = DEFAULT_TOMBSTONE_MAX_AGE_MS,
  } = {}) {
    this.rootDir = path.resolve(rootDir);
    this.recordsDir = path.join(this.rootDir, 'records');
    this.tombstonesDir = path.join(this.rootDir, 'tombstones');
    this.quarantineDir = path.join(this.rootDir, 'quarantine');
    this.sequencePath = path.join(this.rootDir, 'sequence.json');
    this.leaseCursorPath = path.join(this.rootDir, 'lease-cursor.json');
    this.accountNamespacePath = path.join(this.rootDir, 'account-namespace.json');
    this.now = now;
    this.leaseMs = boundedLeaseMs(leaseMs, DEFAULT_LEASE_MS);
    this.tombstoneLimit = tombstoneLimit;
    this.tombstoneMaxAgeMs = tombstoneMaxAgeMs;
    for (const dir of [this.rootDir, this.recordsDir, this.tombstonesDir, this.quarantineDir]) {
      ensurePrivateDirectory(dir);
    }
    this.sequence = this.#loadSequence();
    this.leaseCursor = this.#loadLeaseCursor();
    this.#reconcileSettledRecords();
    this.pruneTombstones();
  }

  #recordPath(deliveryId) {
    return path.join(this.recordsDir, `${deliveryId}.json`);
  }

  #tombstonePath(deliveryId) {
    return path.join(this.tombstonesDir, `${deliveryId}.json`);
  }

  #loadSequence() {
    let persisted = 0;
    try { persisted = Number(readJson(this.sequencePath).sequence) || 0; } catch {}
    let recordMax = 0;
    for (const name of jsonFiles(this.recordsDir)) {
      try {
        recordMax = Math.max(
          recordMax,
          Number(readJson(path.join(this.recordsDir, name)).sequence) || 0,
        );
      } catch {}
    }
    const sequence = Math.max(persisted, recordMax);
    atomicWriteJson(this.sequencePath, { schemaVersion: SCHEMA_VERSION, sequence });
    return sequence;
  }

  #nextSequence() {
    this.sequence += 1;
    atomicWriteJson(this.sequencePath, { schemaVersion: SCHEMA_VERSION, sequence: this.sequence });
    return this.sequence;
  }

  #loadLeaseCursor() {
    let sequence = 0;
    try { sequence = Math.max(0, Number(readJson(this.leaseCursorPath).sequence) || 0); } catch {}
    atomicWriteJson(this.leaseCursorPath, { schemaVersion: SCHEMA_VERSION, sequence });
    return sequence;
  }

  #advanceLeaseCursor(sequence) {
    this.leaseCursor = Math.max(0, Number(sequence) || 0);
    atomicWriteJson(this.leaseCursorPath, {
      schemaVersion: SCHEMA_VERSION,
      sequence: this.leaseCursor,
    });
  }

  #quarantine(reason, deliveryId, evidence) {
    const opaqueId = sha256(String(deliveryId || '')).slice(0, 24);
    const stem = `${String(this.now()).padStart(16, '0')}-${opaqueId}-${randomBytes(4).toString('hex')}`;
    atomicWriteJson(path.join(this.quarantineDir, `${stem}.json`), {
      schemaVersion: SCHEMA_VERSION,
      reason,
      deliveryId,
      observedAt: this.now(),
      ...evidence,
    });
    return stem;
  }

  #retainQuarantinedFile(filePath, stem) {
    if (!existsSync(filePath)) return;
    renameSync(filePath, path.join(this.quarantineDir, `${stem}.retained`));
    fsyncDirectory(path.dirname(filePath));
    fsyncDirectory(this.quarantineDir);
  }

  #reconcileSettledRecords() {
    for (const name of jsonFiles(this.recordsDir)) {
      const recordPath = path.join(this.recordsDir, name);
      const deliveryId = name.slice(0, -5);
      const tombstonePath = this.#tombstonePath(deliveryId);
      if (!existsSync(tombstonePath)) continue;
      try {
        const record = readJson(recordPath);
        const tombstone = readJson(tombstonePath);
        if (record.eventDigest === tombstone.eventDigest) {
          unlinkSync(recordPath);
          fsyncDirectory(this.recordsDir);
        } else {
          const stem = this.#quarantine('settled_digest_collision', deliveryId, {
            storedDigest: record.eventDigest,
            observedDigest: tombstone.eventDigest,
          });
          this.#retainQuarantinedFile(recordPath, stem);
        }
      } catch {
        const stem = this.#quarantine('settled_record_corrupt', deliveryId, {});
        this.#retainQuarantinedFile(recordPath, stem);
      }
    }
  }

  resolveAccountNamespace(accountIds) {
    const current = Array.from(new Set(
      (accountIds || []).map(value => normalizeJid(value)).filter(Boolean),
    )).sort();
    if (!current.length) return null;

    let stored = null;
    if (existsSync(this.accountNamespacePath)) {
      try {
        const candidate = readJson(this.accountNamespacePath);
        if (
          /^[a-f0-9]{64}$/.test(String(candidate.namespace || ''))
          && Array.isArray(candidate.identities)
        ) {
          stored = candidate;
        } else {
          throw new Error('invalid account namespace binding');
        }
      } catch {
        const stem = this.#quarantine('account_namespace_corrupt', '', {});
        this.#retainQuarantinedFile(this.accountNamespacePath, stem);
      }
    }

    const previous = new Set((stored?.identities || []).map(value => normalizeJid(value)).filter(Boolean));
    const overlaps = current.some(value => previous.has(value));
    if (stored && overlaps) {
      const identities = Array.from(new Set([...previous, ...current])).sort();
      if (canonicalJson(identities) !== canonicalJson(stored.identities)) {
        atomicWriteJson(this.accountNamespacePath, {
          schemaVersion: SCHEMA_VERSION,
          namespace: stored.namespace,
          identities,
        });
      }
      return stored.namespace;
    }

    const namespace = opaqueInboundAccountNamespace(current);
    atomicWriteJson(this.accountNamespacePath, {
      schemaVersion: SCHEMA_VERSION,
      namespace,
      identities: current,
    });
    return namespace;
  }

  append({ profileNamespace, accountNamespace, event }) {
    const identity = inboundIdentity({ profileNamespace, accountNamespace, event });
    const deliveryId = sha256(canonicalJson(identity));
    const { eventDigest, mediaManifest } = inboundEventDigest(event);
    const recordPath = this.#recordPath(deliveryId);
    const tombstonePath = this.#tombstonePath(deliveryId);

    for (const [kind, filePath] of [['record', recordPath], ['tombstone', tombstonePath]]) {
      if (!existsSync(filePath)) continue;
      let stored;
      try {
        stored = readJson(filePath);
      } catch {
        const stem = this.#quarantine(`${kind}_corrupt`, deliveryId, {
          observedDigest: eventDigest,
        });
        this.#retainQuarantinedFile(filePath, stem);
        return { status: 'quarantined', deliveryId, eventDigest };
      }
      if (stored.eventDigest === eventDigest) {
        return {
          status: kind === 'record' ? 'duplicate_pending' : 'duplicate_settled',
          deliveryId,
          eventDigest,
        };
      }
      const stem = this.#quarantine(`${kind}_digest_collision`, deliveryId, {
        storedDigest: stored.eventDigest,
        observedDigest: eventDigest,
      });
      if (kind === 'record') this.#retainQuarantinedFile(filePath, stem);
      return { status: 'quarantined', deliveryId, eventDigest };
    }

    const timestamp = this.now();
    const record = {
      schemaVersion: SCHEMA_VERSION,
      deliveryId,
      eventDigest,
      identity,
      mediaManifest,
      sequence: this.#nextSequence(),
      state: 'ready',
      event: { ...(event || {}), _inboundLease: undefined },
      lease: {
        consumerId: '',
        epoch: 0,
        token: '',
        expiresAt: 0,
        attempt: 0,
      },
      createdAt: timestamp,
      updatedAt: timestamp,
    };
    atomicWriteJson(recordPath, record);
    return { status: 'appended', deliveryId, eventDigest, sequence: record.sequence };
  }

  lease({ consumerId, limit = DEFAULT_LIMIT, leaseMs = this.leaseMs } = {}) {
    const consumer = requireConsumerId(consumerId);
    const boundedLimit = Math.max(1, Math.min(Number(limit) || DEFAULT_LIMIT, DEFAULT_LIMIT));
    const duration = boundedLeaseMs(leaseMs, this.leaseMs);
    const now = this.now();
    const records = [];
    for (const name of jsonFiles(this.recordsDir)) {
      try {
        records.push(readJson(path.join(this.recordsDir, name)));
      } catch {
        const filePath = path.join(this.recordsDir, name);
        const stem = this.#quarantine('record_corrupt_during_lease', name.slice(0, -5), {});
        this.#retainQuarantinedFile(filePath, stem);
      }
    }
    records.sort((left, right) => left.sequence - right.sequence);
    const split = records.findIndex(record => Number(record.sequence) > this.leaseCursor);
    const scanOrder = split > 0
      ? [...records.slice(split), ...records.slice(0, split)]
      : records;

    const leased = [];
    for (const record of scanOrder) {
      if (leased.length >= boundedLimit) break;
      const live = record.state === 'leased' && Number(record.lease?.expiresAt) > now;
      if (live && record.lease.consumerId !== consumer) continue;
      if (live) {
        leased.push(publicLease(record));
        continue;
      }
      record.state = 'leased';
      record.lease = {
        consumerId: consumer,
        epoch: (Number(record.lease?.epoch) || 0) + 1,
        token: randomBytes(32).toString('hex'),
        expiresAt: now + duration,
        attempt: (Number(record.lease?.attempt) || 0) + 1,
      };
      record.updatedAt = now;
      atomicWriteJson(this.#recordPath(record.deliveryId), record);
      leased.push(publicLease(record));
    }
    if (leased.length) {
      this.#advanceLeaseCursor(leased[leased.length - 1]._inboundLease.sequence);
    }
    return leased;
  }

  renew({ consumerId, deliveryId, epoch, token, leaseMs = this.leaseMs } = {}) {
    const id = normalizeDeliveryId(deliveryId);
    if (!id) return { status: 'not_found' };
    const recordPath = this.#recordPath(id);
    if (!existsSync(recordPath)) return { status: 'not_found' };
    const record = readJson(recordPath);
    if (
      record.state !== 'leased'
      || record.lease.consumerId !== String(consumerId || '')
      || record.lease.epoch !== Number(epoch)
      || !safeTokenEqual(record.lease.token, token)
      || Number(record.lease.expiresAt) <= this.now()
    ) {
      return { status: 'stale_lease' };
    }
    const now = this.now();
    record.lease.token = randomBytes(32).toString('hex');
    record.lease.expiresAt = now + boundedLeaseMs(leaseMs, this.leaseMs);
    record.updatedAt = now;
    atomicWriteJson(recordPath, record);
    return { status: 'renewed', delivery: publicLeaseMetadata(record) };
  }

  acknowledge({ consumerId, deliveryId, epoch, token } = {}) {
    const id = normalizeDeliveryId(deliveryId);
    if (!id) return { status: 'not_found' };
    const ackFingerprint = sha256(canonicalJson({
      consumerId: String(consumerId || ''),
      deliveryId: id,
      epoch: Number(epoch),
      token: String(token || ''),
    }));
    const recordPath = this.#recordPath(id);
    if (!existsSync(recordPath)) {
      const tombstonePath = this.#tombstonePath(id);
      if (!existsSync(tombstonePath)) return { status: 'not_found' };
      try {
        const tombstone = readJson(tombstonePath);
        return safeTokenEqual(tombstone.ackFingerprint, ackFingerprint)
          ? { status: 'already_acknowledged', deliveryId: id }
          : { status: 'stale_lease', deliveryId: id };
      } catch {
        const stem = this.#quarantine('tombstone_corrupt_during_ack', id, {});
        this.#retainQuarantinedFile(tombstonePath, stem);
        return { status: 'not_found' };
      }
    }
    const record = readJson(recordPath);
    if (
      record.state !== 'leased'
      || record.lease.consumerId !== String(consumerId || '')
      || record.lease.epoch !== Number(epoch)
      || !safeTokenEqual(record.lease.token, token)
      || Number(record.lease.expiresAt) <= this.now()
    ) {
      return { status: 'stale_lease', deliveryId: id };
    }
    atomicWriteJson(this.#tombstonePath(id), {
      schemaVersion: SCHEMA_VERSION,
      deliveryId: id,
      eventDigest: record.eventDigest,
      terminalState: 'acknowledged',
      ackFingerprint,
      settledAt: this.now(),
    });
    unlinkSync(recordPath);
    fsyncDirectory(this.recordsDir);
    this.pruneTombstones();
    return { status: 'acknowledged', deliveryId: id, eventDigest: record.eventDigest };
  }

  pruneTombstones() {
    const now = this.now();
    const entries = [];
    for (const name of jsonFiles(this.tombstonesDir)) {
      const filePath = path.join(this.tombstonesDir, name);
      try {
        const value = readJson(filePath);
        entries.push({ filePath, settledAt: Number(value.settledAt) || 0 });
      } catch {
        const stem = this.#quarantine('tombstone_corrupt', name.slice(0, -5), {});
        this.#retainQuarantinedFile(filePath, stem);
      }
    }
    entries.sort((left, right) => right.settledAt - left.settledAt);
    let removed = false;
    for (const [index, entry] of entries.entries()) {
      if (index >= this.tombstoneLimit || now - entry.settledAt > this.tombstoneMaxAgeMs) {
        unlinkSync(entry.filePath);
        removed = true;
      }
    }
    if (removed) fsyncDirectory(this.tombstonesDir);
  }

  stats() {
    return {
      pending: jsonFiles(this.recordsDir).length,
      tombstones: jsonFiles(this.tombstonesDir).length,
      quarantined: jsonFiles(this.quarantineDir).filter(name => name.endsWith('.json')).length,
    };
  }
}

export function createInboundSpool(rootDir, options) {
  return new InboundSpool(rootDir, options);
}

function sendInboundSpoolResult(res, result) {
  const status = result?.status;
  if (status === 'not_found') return res.status(404).json(result);
  if (status === 'stale_lease') return res.status(409).json(result);
  return res.json(result);
}

/**
 * Register the narrow HTTP ownership boundary used by the Python adapter.
 *
 * The bridge intentionally leases flat event objects, rather than deleting
 * them on poll.  Only a matching fenced ACK can settle a record.  Keeping the
 * handlers here makes the protocol testable without starting Baileys.
 */
export function registerInboundSpoolRoutes(app, spool) {
  app.get('/messages', (req, res) => {
    try {
      return res.json(spool.lease({
        consumerId: req.query?.consumerId,
        limit: req.query?.limit,
      }));
    } catch {
      return res.status(400).json({ error: 'Invalid inbound lease request' });
    }
  });
  app.post('/messages/renew', (req, res) => {
    try { return sendInboundSpoolResult(res, spool.renew(req.body || {})); }
    catch { return res.status(400).json({ error: 'Invalid inbound lease renewal' }); }
  });
  app.post('/messages/ack', (req, res) => {
    try { return sendInboundSpoolResult(res, spool.acknowledge(req.body || {})); }
    catch { return res.status(400).json({ error: 'Invalid inbound acknowledgement' }); }
  });
}
