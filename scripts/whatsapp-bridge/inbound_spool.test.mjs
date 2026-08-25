/** Behavior tests for the durable WhatsApp inbound spool. */

import { strict as assert } from 'node:assert';
import {
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { DurableInboundSpool } from './inbound_spool.js';

function tempRoot() {
  return mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-inbound-spool-'));
}

function event(messageId, body = messageId) {
  return {
    messageId,
    chatId: '15550001111@s.whatsapp.net',
    senderId: '15550002222@s.whatsapp.net',
    body,
    timestamp: 123,
  };
}

function noTemps(root) {
  for (const dir of ['pending', 'acked', 'dead-letter']) {
    assert.equal(
      readdirSync(path.join(root, dir)).some((name) => name.endsWith('.tmp')),
      false,
    );
  }
}

{
  const root = tempRoot();
  let now = 1000;
  try {
    const first = new DurableInboundSpool({
      rootDir: root,
      leaseMs: 100,
      now: () => now,
      log: { error() {} },
    });
    const deliveryId = first.enqueue(event('message-1', 'durable'));
    assert.match(deliveryId, /^[a-f0-9]{64}$/);
    assert.equal(first.pendingCount(), 1);
    noTemps(root);

    // A fresh bridge object sees the atomically persisted pending record.
    const restarted = new DurableInboundSpool({
      rootDir: root,
      leaseMs: 100,
      now: () => now,
      log: { error() {} },
    });
    const [claimedByA] = restarted.poll({ consumerId: 'consumer-a', limit: 1 });
    assert.equal(claimedByA.messageId, 'message-1');
    assert.equal(claimedByA.body, 'durable');
    assert.equal(claimedByA._hermesDelivery.id, deliveryId);
    assert.equal(claimedByA._hermesDelivery.attempt, 1);
    assert.match(claimedByA._hermesDelivery.receipt, /^[a-f0-9]{32}$/);
    assert.equal(restarted.pendingCount(), 1); // GET is non-destructive.

    assert.deepEqual(restarted.poll({ consumerId: 'consumer-a' }), []);
    assert.deepEqual(restarted.poll({ consumerId: 'consumer-b' }), []);

    // Leases are persisted, not process-local: restarting the bridge cannot
    // let a replacement consumer steal work that the first worker still owns.
    const restartedWhileLeased = new DurableInboundSpool({
      rootDir: root,
      leaseMs: 100,
      now: () => now,
      log: { error() {} },
    });
    assert.deepEqual(
      restartedWhileLeased.poll({ consumerId: 'consumer-b' }),
      [],
    );

    now += 101;
    // A brief bridge/network outage may let a live worker's lease expire.
    // Its exact active receipt still renews the same claim instead of
    // producing a second delivery to that worker. A replacement consumer
    // cannot use the stale token and must wait for the renewed lease.
    assert.deepEqual(restarted.poll({
      consumerId: 'consumer-a',
      renewDeliveries: [claimedByA._hermesDelivery],
    }), []);
    now += 99;
    assert.deepEqual(restarted.poll({ consumerId: 'consumer-b' }), []);
    now += 2;
    const [claimedByB] = restarted.poll({ consumerId: 'consumer-b' });
    assert.equal(claimedByB._hermesDelivery.attempt, 2);
    assert.notEqual(
      claimedByB._hermesDelivery.receipt,
      claimedByA._hermesDelivery.receipt,
    );

    now += 51;
    // Only a receipt Python explicitly reports as active is renewed. A
    // consumer-wide poll can no longer pin lost/settled claims forever.
    assert.deepEqual(restarted.poll({
      consumerId: 'consumer-b',
      renewDeliveries: [claimedByB._hermesDelivery],
    }), []);
    now += 50;
    assert.deepEqual(restarted.poll({ consumerId: 'consumer-a' }), []);
    now += 51;
    const [claimedByC] = restarted.poll({ consumerId: 'consumer-c' });
    assert.equal(claimedByC._hermesDelivery.attempt, 3);

    // A late worker cannot ACK and delete the newer worker's claim.
    const staleAck = restarted.acknowledge([claimedByA._hermesDelivery]);
    assert.deepEqual(staleAck.acked, []);
    assert.deepEqual(staleAck.stale, [deliveryId]);
    assert.equal(restarted.pendingCount(), 1);
    assert.deepEqual(
      restarted.acknowledge([claimedByB._hermesDelivery]).stale,
      [deliveryId],
    );

    const ack = restarted.acknowledge([claimedByC._hermesDelivery]);
    assert.deepEqual(ack.acked, [deliveryId]);
    assert.equal(restarted.pendingCount(), 0);
    assert.equal(restarted.acknowledgedCount(), 1);
    assert.deepEqual(
      restarted.acknowledge([claimedByC._hermesDelivery]).alreadyAcked,
      [deliveryId],
    );

    // Stable IDs plus the retained ACK tombstone suppress Baileys replay.
    assert.equal(restarted.enqueue(event('message-1', 'durable')), deliveryId);
    assert.equal(restarted.pendingCount(), 0);
    noTemps(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

{
  const root = tempRoot();
  let now = 12000;
  try {
    const spool = new DurableInboundSpool({
      rootDir: root,
      maxAcked: 1,
      ackRetentionMs: 10,
      now: () => now,
      log: { error() {} },
    });
    spool.enqueue(event('acked-old'));
    const [oldClaim] = spool.poll({ consumerId: 'worker' });
    spool.acknowledge([oldClaim._hermesDelivery]);
    now += 1;
    spool.enqueue(event('acked-new'));
    const [newClaim] = spool.poll({ consumerId: 'worker' });
    spool.acknowledge([newClaim._hermesDelivery]);
    assert.equal(spool.acknowledgedCount(), 1);

    now += 11;
    // Construction prunes expired tombstones, bounding disk growth while
    // keeping pending/dead-letter payloads untouched.
    const restarted = new DurableInboundSpool({
      rootDir: root,
      maxAcked: 1,
      ackRetentionMs: 10,
      now: () => now,
      log: { error() {} },
    });
    assert.equal(restarted.acknowledgedCount(), 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

{
  const root = tempRoot();
  let now = 5000;
  try {
    const spool = new DurableInboundSpool({
      rootDir: root,
      maxAttempts: 2,
      leaseMs: 100,
      retryBaseMs: 10,
      now: () => now,
      log: { error() {} },
    });
    spool.enqueue(event('poison'));
    now += 1;
    spool.enqueue(event('healthy-later'));

    const [firstAttempt] = spool.poll({ consumerId: 'worker', limit: 1 });
    assert.equal(firstAttempt.messageId, 'poison');
    assert.deepEqual(
      spool.release([firstAttempt._hermesDelivery], 'other-worker').released,
      [],
    );
    assert.deepEqual(
      spool.release([firstAttempt._hermesDelivery], 'worker').released,
      [firstAttempt._hermesDelivery.id],
    );
    // The poison record's backoff does not head-of-line block later work.
    const [healthy] = spool.poll({ consumerId: 'worker' });
    assert.equal(healthy.messageId, 'healthy-later');
    spool.acknowledge([healthy._hermesDelivery]);
    assert.deepEqual(spool.poll({ consumerId: 'worker' }), []); // poison backoff

    now += 10;
    const [secondAttempt] = spool.poll({ consumerId: 'worker' });
    assert.equal(secondAttempt._hermesDelivery.attempt, 2);
    spool.release([secondAttempt._hermesDelivery], 'worker');
    now += 20;
    assert.deepEqual(spool.poll({ consumerId: 'worker' }), []);
    assert.equal(spool.pendingCount(), 0);
    assert.equal(spool.deadLetterCount(), 1);
    const deadName = readdirSync(path.join(root, 'dead-letter'))[0];
    const dead = JSON.parse(
      readFileSync(path.join(root, 'dead-letter', deadName), 'utf8'),
    );
    assert.equal(dead.event.messageId, 'poison');
    assert.equal(dead.deadLetterReason, 'retry_attempts_exhausted');
    assert.deepEqual(spool.listDeadLetters(), [{
      id: secondAttempt._hermesDelivery.id,
      reason: 'retry_attempts_exhausted',
      attempts: 2,
      createdAt: 5000,
      deadLetteredAt: now,
      chatId: '15550001111@s.whatsapp.net',
      messageId: 'poison',
    }]);

    const requeue = spool.requeueDeadLetters([
      secondAttempt._hermesDelivery.id,
    ]);
    assert.deepEqual(requeue.requeued, [secondAttempt._hermesDelivery.id]);
    assert.equal(spool.deadLetterCount(), 0);
    const [recoveredPoison] = spool.poll({ consumerId: 'operator-retry' });
    assert.equal(recoveredPoison.messageId, 'poison');
    assert.equal(recoveredPoison._hermesDelivery.attempt, 1);
    spool.acknowledge([recoveredPoison._hermesDelivery]);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

{
  const root = tempRoot();
  let now = 9000;
  try {
    const spool = new DurableInboundSpool({
      rootDir: root,
      maxPending: 1,
      now: () => now,
      log: { error() {} },
    });
    spool.enqueue(event('oldest'));
    now += 1;
    spool.enqueue(event('newest'));
    assert.equal(spool.pendingCount(), 1);
    assert.equal(spool.deadLetterCount(), 1);
    const deadName = readdirSync(path.join(root, 'dead-letter'))[0];
    const dead = JSON.parse(
      readFileSync(path.join(root, 'dead-letter', deadName), 'utf8'),
    );
    assert.equal(dead.event.messageId, 'oldest');
    assert.equal(dead.deadLetterReason, 'pending_capacity_exceeded');

    const [newest] = spool.poll({ consumerId: 'worker' });
    assert.equal(newest.messageId, 'newest');
    spool.acknowledge([newest._hermesDelivery]);
    // Capacity overflow self-heals once a pending slot opens.
    const [recoveredOldest] = spool.poll({ consumerId: 'worker' });
    assert.equal(recoveredOldest.messageId, 'oldest');
    spool.acknowledge([recoveredOldest._hermesDelivery]);
    assert.equal(spool.deadLetterCount(), 0);

    const corruptName = `${'a'.repeat(64)}.json`;
    writeFileSync(path.join(root, 'pending', corruptName), '{broken', 'utf8');
    spool.poll({ consumerId: 'worker' });
    assert.equal(spool.deadLetterCount(), 1);
    assert.equal(spool.pendingCount(), 0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

{
  const root = tempRoot();
  let now = 12000;
  try {
    const spool = new DurableInboundSpool({
      rootDir: root,
      maxPending: 1,
      now: () => now,
      log: { error() {} },
    });
    spool.enqueue(event('transition-old'));
    now += 1;
    spool.enqueue(event('transition-new'));
    const [newest] = spool.poll({ consumerId: 'worker' });
    spool.acknowledge([newest._hermesDelivery]);

    const deadName = readdirSync(path.join(root, 'dead-letter'))[0];
    const deadPath = path.join(root, 'dead-letter', deadName);
    const dead = JSON.parse(readFileSync(deadPath, 'utf8'));
    const {
      deadLetteredAt: _deadLetteredAt,
      deadLetterReason: _deadLetterReason,
      ...active
    } = dead;
    // Crash snapshot: active copy was published, dead source not yet removed.
    writeFileSync(
      path.join(root, 'pending', deadName),
      `${JSON.stringify({
        ...active,
        attempts: 0,
        leaseOwner: null,
        leaseUntil: 0,
        leaseToken: null,
        nextAttemptAt: 0,
        requeuedFromDeadAt: now,
      })}\n`,
      'utf8',
    );

    const restarted = new DurableInboundSpool({
      rootDir: root,
      maxPending: 1,
      now: () => now,
      log: { error() {} },
    });
    assert.equal(restarted.deadLetterCount(), 0);
    assert.equal(restarted.pendingCount(), 1);
    const [repaired] = restarted.poll({ consumerId: 'repair-worker' });
    assert.equal(repaired.messageId, 'transition-old');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

{
  const root = tempRoot();
  try {
    const spool = new DurableInboundSpool({ rootDir: root, log: { error() {} } });
    spool.noteError('enqueue', new Error('temporary I/O failure'));
    spool.enqueue(event('sticky-error'));
    assert.equal(spool.health().healthy, false);
    spool.clearError('enqueue_failed:');
    assert.equal(spool.health().healthy, true);
    const [sticky] = spool.pollLegacy({ limit: 1 });
    assert.equal(sticky.messageId, 'sticky-error');

    spool.enqueue(event('legacy'));
    const [legacy] = spool.pollLegacy({ limit: 1 });
    assert.equal(legacy.messageId, 'legacy');
    assert.equal(Object.hasOwn(legacy, '_hermesDelivery'), false);
    assert.equal(spool.pendingCount(), 0);
    assert.equal(spool.acknowledgedCount(), 2);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

{
  const root = tempRoot();
  try {
    const spool = new DurableInboundSpool({ rootDir: root, log: { error() {} } });
    const deliveryId = spool.quarantine(
      event('volatile-overflow'),
      'volatile_retry_capacity_exceeded',
    );
    assert.equal(spool.pendingCount(), 0);
    assert.equal(spool.deadLetterCount(), 1);
    assert.equal(
      spool.listDeadLetters()[0].reason,
      'volatile_retry_capacity_exceeded',
    );

    const deadName = readdirSync(path.join(root, 'dead-letter'))[0];
    const deadSnapshot = readFileSync(
      path.join(root, 'dead-letter', deadName),
      'utf8',
    );
    assert.deepEqual(
      spool.requeueDeadLetters([deliveryId]).requeued,
      [deliveryId],
    );
    const [claimed] = spool.poll({ consumerId: 'worker' });
    spool.acknowledge([claimed._hermesDelivery]);

    // Simulate the rare unlink-failure tail from a completed dead→pending
    // transition. The ACK tombstone is authoritative; capacity recovery must
    // remove the stale dead duplicate, never resurrect the message.
    writeFileSync(
      path.join(root, 'dead-letter', deadName),
      deadSnapshot,
      'utf8',
    );
    assert.deepEqual(
      spool.requeueDeadLetters([deliveryId]).requeued,
      [],
    );
    assert.deepEqual(spool.poll({ consumerId: 'worker' }), []);
    assert.equal(spool.deadLetterCount(), 0);
    assert.equal(spool.acknowledgedCount(), 1);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

console.log('inbound_spool.test.mjs: all assertions passed');
