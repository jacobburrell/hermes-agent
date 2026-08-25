/**
 * Crash-safe inbound queue for the WhatsApp bridge.
 *
 * Each message is one JSON file published with a same-directory atomic rename.
 * Polling leases records but never deletes them; only an explicit ACK moves a
 * pending record to bounded completed-message retention. Failed deliveries can
 * be released for retry and move intact to dead-letter after bounded attempts.
 */

import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { createHash, randomBytes } from 'node:crypto';

const RECORD_VERSION = 1;

function positiveInteger(value, fallback) {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function fsyncDirectory(dir) {
  let fd;
  try {
    fd = openSync(dir, 'r');
    fsyncSync(fd);
  } catch {
    // Best-effort on platforms/filesystems that do not allow directory fsync.
  } finally {
    if (fd !== undefined) {
      try { closeSync(fd); } catch {}
    }
  }
}

function atomicWriteJson(filePath, payload) {
  const dir = path.dirname(filePath);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  const tmpPath = path.join(
    dir,
    `.${path.basename(filePath)}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`,
  );
  let fd;
  try {
    fd = openSync(tmpPath, 'wx', 0o600);
    writeFileSync(fd, `${JSON.stringify(payload)}\n`, { encoding: 'utf8' });
    fsyncSync(fd);
    closeSync(fd);
    fd = undefined;
    renameSync(tmpPath, filePath);
    fsyncDirectory(dir);
  } finally {
    if (fd !== undefined) {
      try { closeSync(fd); } catch {}
    }
    try { unlinkSync(tmpPath); } catch {}
  }
}

function stableDeliveryId(event) {
  const chatId = String(event?.chatId || '');
  const messageId = String(event?.messageId || '');
  const identity = messageId
    ? `${chatId}\0${messageId}`
    : `${JSON.stringify(event || {})}\0${randomBytes(16).toString('hex')}`;
  return createHash('sha256')
    .update(identity)
    .digest('hex');
}

function recordFileName(deliveryId) {
  if (!/^[a-f0-9]{64}$/.test(String(deliveryId || ''))) return null;
  return `${deliveryId}.json`;
}

function normalizeReceipt(value) {
  const id = String(value?.id || '');
  const receipt = String(value?.receipt || '');
  if (!recordFileName(id) || !/^[a-f0-9]{32}$/.test(receipt)) return null;
  return { id, receipt };
}

export class DurableInboundSpool {
  constructor({
    rootDir,
    maxAttempts = 5,
    maxPending = 100,
    leaseMs = 30 * 1000,
    retryBaseMs = 1000,
    ackRetentionMs = 7 * 24 * 60 * 60 * 1000,
    maxAcked = 10000,
    now = () => Date.now(),
    log = console,
  }) {
    if (!rootDir) throw new Error('DurableInboundSpool requires rootDir');
    this.rootDir = rootDir;
    this.pendingDir = path.join(rootDir, 'pending');
    this.ackedDir = path.join(rootDir, 'acked');
    this.deadLetterDir = path.join(rootDir, 'dead-letter');
    this.maxAttempts = positiveInteger(maxAttempts, 5);
    this.maxPending = positiveInteger(maxPending, 100);
    this.leaseMs = positiveInteger(leaseMs, 30 * 1000);
    this.retryBaseMs = positiveInteger(retryBaseMs, 1000);
    this.ackRetentionMs = positiveInteger(
      ackRetentionMs, 7 * 24 * 60 * 60 * 1000,
    );
    this.maxAcked = positiveInteger(maxAcked, 10000);
    this.now = now;
    this.log = log;
    this.lastError = null;
    mkdirSync(this.pendingDir, { recursive: true, mode: 0o700 });
    mkdirSync(this.ackedDir, { recursive: true, mode: 0o700 });
    mkdirSync(this.deadLetterDir, { recursive: true, mode: 0o700 });
    this._removeOrphanTemps();
    this._repairRequeueTransitions();
    this._pruneAcknowledged();
  }

  _removeOrphanTemps() {
    for (const dir of [this.pendingDir, this.ackedDir, this.deadLetterDir]) {
      for (const name of readdirSync(dir)) {
        if (!name.endsWith('.tmp')) continue;
        try { unlinkSync(path.join(dir, name)); } catch {}
      }
    }
  }

  _repairRequeueTransitions() {
    let repaired = false;
    for (const name of this._jsonFiles(this.pendingDir)) {
      const pendingPath = path.join(this.pendingDir, name);
      let record;
      try {
        record = this._readRecord(pendingPath);
      } catch {
        continue;
      }
      if (!record.requeuedFromDeadAt) continue;
      const deadPath = this._deadLetterPath(record.deliveryId);
      if (!deadPath || !existsSync(deadPath)) continue;
      try {
        unlinkSync(deadPath);
        repaired = true;
      } catch {}
    }
    if (repaired) fsyncDirectory(this.deadLetterDir);
  }

  _jsonFiles(dir) {
    return readdirSync(dir)
      .filter((name) => name.endsWith('.json'))
      .sort();
  }

  _pendingPath(deliveryId) {
    const name = recordFileName(deliveryId);
    return name ? path.join(this.pendingDir, name) : null;
  }

  _deadLetterPath(deliveryId) {
    const name = recordFileName(deliveryId);
    return name ? path.join(this.deadLetterDir, name) : null;
  }

  _ackedPath(deliveryId) {
    const name = recordFileName(deliveryId);
    return name ? path.join(this.ackedDir, name) : null;
  }

  _readRecord(filePath) {
    const value = JSON.parse(readFileSync(filePath, 'utf8'));
    if (
      !value
      || value.version !== RECORD_VERSION
      || !recordFileName(value.deliveryId)
      || !value.event
      || typeof value.event !== 'object'
    ) {
      throw new Error('invalid inbound spool record');
    }
    return value;
  }

  _deadLetter(record, pendingPath, reason) {
    const deadPath = this._deadLetterPath(record.deliveryId);
    if (!deadPath) throw new Error('invalid dead-letter delivery ID');
    const deadRecord = {
      ...record,
      deadLetteredAt: this.now(),
      deadLetterReason: reason,
    };
    // Publish the terminal metadata first, then atomically move the complete
    // record. A crash between those operations leaves a recoverable pending
    // record that the next poll will move again; it never loses the payload.
    atomicWriteJson(pendingPath, deadRecord);
    renameSync(pendingPath, deadPath);
    fsyncDirectory(this.pendingDir);
    fsyncDirectory(this.deadLetterDir);
    this.log?.error?.(
      `[bridge] inbound message ${record.deliveryId} moved to dead-letter: ${reason}`,
    );
  }

  _deadLetterCorrupt(filePath, error) {
    const stamp = `${this.now()}-${randomBytes(4).toString('hex')}`;
    const target = path.join(
      this.deadLetterDir,
      `${path.basename(filePath, '.json')}.corrupt-${stamp}.json`,
    );
    renameSync(filePath, target);
    fsyncDirectory(this.pendingDir);
    fsyncDirectory(this.deadLetterDir);
    this.log?.error?.(
      `[bridge] corrupt inbound spool record moved to dead-letter: ${error?.message || error}`,
    );
  }

  _records() {
    const records = [];
    for (const name of this._jsonFiles(this.pendingDir)) {
      const filePath = path.join(this.pendingDir, name);
      try {
        const record = this._readRecord(filePath);
        if (record.acknowledgedAt) {
          renameSync(filePath, this._ackedPath(record.deliveryId));
          fsyncDirectory(this.pendingDir);
          fsyncDirectory(this.ackedDir);
          continue;
        }
        if (record.deadLetteredAt) {
          renameSync(filePath, this._deadLetterPath(record.deliveryId));
          fsyncDirectory(this.pendingDir);
          fsyncDirectory(this.deadLetterDir);
          continue;
        }
        records.push({ record, filePath });
      } catch (error) {
        this._deadLetterCorrupt(filePath, error);
      }
    }
    records.sort((left, right) => (
      Number(left.record.createdAt || 0) - Number(right.record.createdAt || 0)
      || left.record.deliveryId.localeCompare(right.record.deliveryId)
    ));
    return records;
  }

  enqueue(event) {
    const deliveryId = stableDeliveryId(event);
    const pendingPath = this._pendingPath(deliveryId);
    const deadPath = this._deadLetterPath(deliveryId);
    const ackedPath = this._ackedPath(deliveryId);
    if (!pendingPath || !deadPath || !ackedPath) {
      throw new Error('could not derive inbound delivery ID');
    }
    if (
      existsSync(pendingPath)
      || existsSync(ackedPath)
      || existsSync(deadPath)
    ) return deliveryId;

    const record = {
      version: RECORD_VERSION,
      deliveryId,
      event,
      attempts: 0,
      createdAt: this.now(),
      lastAttemptAt: null,
      leaseOwner: null,
      leaseUntil: 0,
      leaseToken: null,
      nextAttemptAt: 0,
    };
    try {
      atomicWriteJson(pendingPath, record);
      this._enforcePendingBound();
      return deliveryId;
    } catch (error) {
      this.lastError = `enqueue_failed:${error?.message || error}`;
      throw error;
    }
  }

  quarantine(event, reason = 'manual_quarantine') {
    const deliveryId = stableDeliveryId(event);
    const pendingPath = this._pendingPath(deliveryId);
    const deadPath = this._deadLetterPath(deliveryId);
    const ackedPath = this._ackedPath(deliveryId);
    if (!pendingPath || !deadPath || !ackedPath) {
      throw new Error('could not derive inbound delivery ID');
    }
    if (existsSync(pendingPath) || existsSync(ackedPath) || existsSync(deadPath)) {
      return deliveryId;
    }
    const now = this.now();
    atomicWriteJson(deadPath, {
      version: RECORD_VERSION,
      deliveryId,
      event,
      attempts: 0,
      createdAt: now,
      lastAttemptAt: null,
      leaseOwner: null,
      leaseUntil: 0,
      leaseToken: null,
      nextAttemptAt: 0,
      deadLetteredAt: now,
      deadLetterReason: String(reason || 'manual_quarantine'),
    });
    fsyncDirectory(this.deadLetterDir);
    this.log?.error?.(
      `[bridge] inbound message ${deliveryId} quarantined: ${reason}`,
    );
    return deliveryId;
  }

  _enforcePendingBound() {
    const records = this._records();
    const overflow = records.length - this.maxPending;
    if (overflow <= 0) return;
    // Never evict work actively owned by Python. Because enqueue publishes the
    // newest record before enforcing the bound, at least the new/unleased item
    // is available to dead-letter when every older record is in flight.
    const eligible = records.filter(
      ({ record }) => Number(record.leaseUntil || 0) <= this.now(),
    );
    for (let index = 0; index < Math.min(overflow, eligible.length); index += 1) {
      const { record, filePath } = eligible[index];
      this._deadLetter(record, filePath, 'pending_capacity_exceeded');
    }
  }

  poll({ consumerId, limit = 100, renewDeliveries = [] } = {}) {
    const consumer = String(consumerId || '').trim();
    if (!consumer) throw new Error('consumerId is required');
    const boundedLimit = positiveInteger(limit, 100);
    const now = this.now();
    const deliveries = [];
    const renewals = new Map();
    for (const rawDelivery of Array.isArray(renewDeliveries) ? renewDeliveries : []) {
      const delivery = normalizeReceipt(rawDelivery);
      if (delivery) renewals.set(delivery.id, delivery.receipt);
    }

    // Capacity dead-lettering is pressure relief, not a permanent discard.
    // As soon as Python drains space, restore those intact records before
    // claiming new work. Poison/corrupt records remain quarantined.
    this._recoverCapacityDeadLetters();

    for (const { record, filePath } of this._records()) {
      if (deliveries.length >= boundedLimit) break;
      const leaseUntil = Number(record.leaseUntil || 0);
      const exactActiveRenewal = (
        record.leaseOwner === consumer
        && renewals.get(record.deliveryId) === record.leaseToken
      );
      if (exactActiveRenewal) {
        // A short bridge/network outage can outlast the persisted lease even
        // though Python is still processing the exact claim. Honour its
        // receipt heartbeat after expiry as long as no replacement consumer
        // has claimed the record (which would have changed owner/token). This
        // prevents a single live worker from executing the same turn twice.
        if (
          leaseUntil <= now
          || leaseUntil - now <= Math.floor(this.leaseMs / 2)
        ) {
          atomicWriteJson(filePath, {
            ...record,
            leaseUntil: now + this.leaseMs,
          });
        }
        continue;
      }
      if (leaseUntil > now) {
        continue;
      }

      if (Number(record.nextAttemptAt || 0) > now) continue;

      if (Number(record.attempts || 0) >= this.maxAttempts) {
        this._deadLetter(record, filePath, 'retry_attempts_exhausted');
        continue;
      }

      const claimed = {
        ...record,
        attempts: Number(record.attempts || 0) + 1,
        lastAttemptAt: now,
        leaseOwner: consumer,
        leaseUntil: now + this.leaseMs,
        leaseToken: randomBytes(16).toString('hex'),
        nextAttemptAt: 0,
      };
      atomicWriteJson(filePath, claimed);
      deliveries.push({
        ...claimed.event,
        _hermesDelivery: {
          id: record.deliveryId,
          receipt: claimed.leaseToken,
          attempt: claimed.attempts,
          maxAttempts: this.maxAttempts,
        },
      });
    }
    return deliveries;
  }

  pollLegacy({ limit = 100 } = {}) {
    const consumerId = `legacy-${randomBytes(8).toString('hex')}`;
    const deliveries = this.poll({ consumerId, limit });
    this.acknowledge(
      deliveries.map((delivery) => delivery._hermesDelivery),
    );
    return deliveries.map(({ _hermesDelivery, ...event }) => event);
  }

  acknowledge(deliveries) {
    const acked = [];
    const alreadyAcked = [];
    const missing = [];
    const stale = [];
    for (const rawDelivery of Array.isArray(deliveries) ? deliveries : []) {
      const delivery = normalizeReceipt(rawDelivery);
      if (!delivery) {
        stale.push(String(rawDelivery?.id || ''));
        continue;
      }
      const pendingPath = this._pendingPath(delivery.id);
      if (!pendingPath || !existsSync(pendingPath)) {
        if (existsSync(this._ackedPath(delivery.id))) {
          alreadyAcked.push(delivery.id);
        } else {
          missing.push(delivery.id);
        }
        continue;
      }
      let record;
      try {
        record = this._readRecord(pendingPath);
      } catch (error) {
        this._deadLetterCorrupt(pendingPath, error);
        continue;
      }
      if (record.leaseToken !== delivery.receipt) {
        stale.push(delivery.id);
        continue;
      }
      const ackedPath = this._ackedPath(delivery.id);
      atomicWriteJson(pendingPath, {
        ...record,
        acknowledgedAt: this.now(),
      });
      renameSync(pendingPath, ackedPath);
      fsyncDirectory(this.pendingDir);
      fsyncDirectory(this.ackedDir);
      acked.push(delivery.id);
    }
    if (acked.length > 0) this._pruneAcknowledged();
    return { acked, alreadyAcked, missing, stale };
  }

  release(deliveries, consumerId) {
    const consumer = String(consumerId || '').trim();
    const released = [];
    const stale = [];
    for (const rawDelivery of Array.isArray(deliveries) ? deliveries : []) {
      const delivery = normalizeReceipt(rawDelivery);
      if (!delivery) {
        stale.push(String(rawDelivery?.id || ''));
        continue;
      }
      const pendingPath = this._pendingPath(delivery.id);
      if (!pendingPath || !existsSync(pendingPath)) continue;
      let record;
      try {
        record = this._readRecord(pendingPath);
      } catch (error) {
        this._deadLetterCorrupt(pendingPath, error);
        continue;
      }
      if (
        !consumer
        || record.leaseOwner !== consumer
        || record.leaseToken !== delivery.receipt
      ) {
        stale.push(delivery.id);
        continue;
      }
      const exponent = Math.max(0, Number(record.attempts || 1) - 1);
      const retryDelay = Math.min(60 * 1000, this.retryBaseMs * (2 ** exponent));
      atomicWriteJson(pendingPath, {
        ...record,
        leaseOwner: null,
        leaseUntil: 0,
        leaseToken: null,
        nextAttemptAt: this.now() + retryDelay,
      });
      released.push(delivery.id);
    }
    return { released, stale };
  }

  _recordWithoutTerminalMetadata(record) {
    const {
      acknowledgedAt: _acknowledgedAt,
      deadLetteredAt: _deadLetteredAt,
      deadLetterReason: _deadLetterReason,
      ...activeRecord
    } = record;
    return {
      ...activeRecord,
      attempts: 0,
      lastAttemptAt: null,
      leaseOwner: null,
      leaseUntil: 0,
      leaseToken: null,
      nextAttemptAt: 0,
    };
  }

  _requeueDeadPath(deadPath, record) {
    const pendingPath = this._pendingPath(record.deliveryId);
    const ackedPath = this._ackedPath(record.deliveryId);
    if (!pendingPath || !ackedPath || existsSync(pendingPath)) return false;
    // A prior requeue may have published the pending copy and then failed to
    // unlink its dead source. If Python subsequently ACKed that active copy,
    // the tombstone is authoritative: remove the stale dead duplicate instead
    // of resurrecting an already-completed message.
    if (existsSync(ackedPath)) {
      try {
        unlinkSync(deadPath);
        fsyncDirectory(this.deadLetterDir);
      } catch {}
      return false;
    }
    // Publish a complete active copy first. A crash before dead-source
    // removal leaves both copies, never zero; the transition marker lets the
    // constructor deterministically finish removing the stale dead copy.
    atomicWriteJson(pendingPath, {
      ...this._recordWithoutTerminalMetadata(record),
      requeuedFromDeadAt: this.now(),
    });
    try { unlinkSync(deadPath); } catch {}
    fsyncDirectory(this.pendingDir);
    fsyncDirectory(this.deadLetterDir);
    return true;
  }

  _recoverCapacityDeadLetters() {
    let available = this.maxPending - this.pendingCount();
    if (available <= 0) return [];
    const recovered = [];
    for (const name of this._jsonFiles(this.deadLetterDir)) {
      if (available <= 0) break;
      const deadPath = path.join(this.deadLetterDir, name);
      let record;
      try {
        record = this._readRecord(deadPath);
      } catch {
        continue;
      }
      if (record.deadLetterReason !== 'pending_capacity_exceeded') continue;
      if (this._requeueDeadPath(deadPath, record)) {
        recovered.push(record.deliveryId);
        available -= 1;
      }
    }
    return recovered;
  }

  listDeadLetters({ limit = 100 } = {}) {
    const boundedLimit = positiveInteger(limit, 100);
    const entries = [];
    for (const name of this._jsonFiles(this.deadLetterDir)) {
      if (entries.length >= boundedLimit) break;
      const deadPath = path.join(this.deadLetterDir, name);
      try {
        const record = this._readRecord(deadPath);
        entries.push({
          id: record.deliveryId,
          reason: String(record.deadLetterReason || 'unknown'),
          attempts: Number(record.attempts || 0),
          createdAt: Number(record.createdAt || 0),
          deadLetteredAt: Number(record.deadLetteredAt || 0),
          chatId: String(record.event?.chatId || ''),
          messageId: String(record.event?.messageId || ''),
        });
      } catch (error) {
        entries.push({
          id: path.basename(name, '.json'),
          reason: 'corrupt_record',
          error: String(error?.message || error),
        });
      }
    }
    return entries;
  }

  requeueDeadLetters(deliveryIds) {
    const requeued = [];
    const missing = [];
    const invalid = [];
    const full = [];
    for (const rawId of Array.isArray(deliveryIds) ? deliveryIds : []) {
      const id = String(rawId || '');
      const deadPath = this._deadLetterPath(id);
      if (!deadPath) {
        invalid.push(id);
        continue;
      }
      if (!existsSync(deadPath)) {
        missing.push(id);
        continue;
      }
      if (this.pendingCount() >= this.maxPending) {
        full.push(id);
        continue;
      }
      let record;
      try {
        record = this._readRecord(deadPath);
      } catch {
        invalid.push(id);
        continue;
      }
      if (this._requeueDeadPath(deadPath, record)) requeued.push(id);
    }
    return { requeued, missing, invalid, full };
  }

  pendingCount() {
    return this._jsonFiles(this.pendingDir).length;
  }

  acknowledgedCount() {
    return this._jsonFiles(this.ackedDir).length;
  }

  _pruneAcknowledged() {
    const now = this.now();
    const records = [];
    for (const name of this._jsonFiles(this.ackedDir)) {
      const filePath = path.join(this.ackedDir, name);
      try {
        const record = this._readRecord(filePath);
        records.push({
          filePath,
          acknowledgedAt: Number(record.acknowledgedAt || record.createdAt || 0),
        });
      } catch {
        // A malformed completed tombstone cannot affect pending delivery. It
        // is safe to age it by zero and prune it before valid recent entries.
        records.push({ filePath, acknowledgedAt: 0 });
      }
    }
    records.sort((left, right) => left.acknowledgedAt - right.acknowledgedAt);
    let removed = false;
    for (let index = 0; index < records.length; index += 1) {
      const entry = records[index];
      const expired = now - entry.acknowledgedAt > this.ackRetentionMs;
      const overLimit = records.length - index > this.maxAcked;
      if (!expired && !overLimit) continue;
      try {
        unlinkSync(entry.filePath);
        removed = true;
      } catch {}
    }
    if (removed) fsyncDirectory(this.ackedDir);
  }

  deadLetterCount() {
    return this._jsonFiles(this.deadLetterDir).length;
  }

  health() {
    return {
      healthy: this.lastError === null,
      lastError: this.lastError,
      pending: this.pendingCount(),
      acknowledged: this.acknowledgedCount(),
      deadLetter: this.deadLetterCount(),
      maxAttempts: this.maxAttempts,
    };
  }

  noteError(operation, error) {
    this.lastError = `${operation}_failed:${error?.message || error}`;
  }

  clearError(prefix = '') {
    if (!prefix || String(this.lastError || '').startsWith(prefix)) {
      this.lastError = null;
    }
  }
}

export function createDurableInboundSpool(options) {
  return new DurableInboundSpool(options);
}
