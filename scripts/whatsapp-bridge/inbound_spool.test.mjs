import assert from 'node:assert/strict';
import {
  copyFileSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  createInboundSpool,
  canonicalDigestJson,
  inboundEventDigest,
  opaqueInboundAccountNamespace,
  opaqueInboundProfileNamespace,
  registerInboundSpoolRoutes,
  stageInboundEventSafely,
} from './inbound_spool.js';

const PROFILE = 'a'.repeat(64);
const ACCOUNT = 'b'.repeat(64);

function event(overrides = {}) {
  return {
    messageId: 'wamid.one',
    chatId: '120363000000000000@g.us',
    senderId: '15551234567:3@s.whatsapp.net',
    body: 'hello',
    hasMedia: false,
    mediaType: '',
    mime: '',
    fileName: '',
    mediaUrls: [],
    readReceiptKey: {
      remoteJid: '120363000000000000@g.us',
      id: 'wamid.one',
      participant: '15551234567:3@s.whatsapp.net',
      fromMe: false,
    },
    ...overrides,
  };
}

function withSpool(run) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-inbound-spool-'));
  let now = 10_000;
  const options = { now: () => now, leaseMs: 1_000 };
  try {
    return run({
      root,
      spool: createInboundSpool(root, options),
      restart: () => createInboundSpool(root, options),
      setNow: value => { now = value; },
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function fakeRouteApp() {
  const routes = { get: new Map(), post: new Map() };
  return {
    routes,
    get(route, handler) { routes.get.set(route, handler); },
    post(route, handler) { routes.post.set(route, handler); },
  };
}

function callRoute(handler, { query = {}, body = {} } = {}) {
  const result = { status: 200, body: undefined };
  const res = {
    status(code) { result.status = code; return this; },
    json(value) { result.body = value; return this; },
  };
  handler({ query, body }, res);
  return result;
}

test('profile and account namespaces are opaque and account readiness fails closed', () => {
  assert.equal(opaqueInboundProfileNamespace(''), null);
  assert.match(opaqueInboundProfileNamespace('/private/profile/session'), /^[a-f0-9]{64}$/);
  assert.equal(opaqueInboundAccountNamespace([]), null);
  assert.equal(opaqueInboundAccountNamespace(['', null, undefined]), null);
  assert.match(opaqueInboundAccountNamespace(['15551234567@s.whatsapp.net']), /^[a-f0-9]{64}$/);
});

test('leased bridge routes preserve staged media through restart and require fenced ACK', () => withSpool(({ root, spool, restart }) => {
  const eventWithMedia = event({
    hasMedia: true,
    mediaType: 'image',
    mime: 'image/jpeg',
    fileName: 'photo.jpg',
    mediaUrls: ['/profile/cache/images/photo.jpg'],
    mediaMetadata: [{ mediaType: 'image', mime: 'image/jpeg', fileName: 'photo.jpg', size: 4 }],
  });
  const noAccount = stageInboundEventSafely({
    spool,
    profileNamespace: PROFILE,
    accountNamespace: null,
    event: eventWithMedia,
  });
  assert.equal(noAccount, null);
  const staged = stageInboundEventSafely({
    spool,
    profileNamespace: PROFILE,
    accountNamespace: ACCOUNT,
    event: eventWithMedia,
  });
  assert.equal(staged.status, 'appended');

  const app = fakeRouteApp();
  registerInboundSpoolRoutes(app, spool);
  const first = callRoute(app.routes.get.get('/messages'), { query: { consumerId: 'python-a' } });
  assert.equal(first.status, 200);
  assert.equal(first.body.length, 1);
  assert.deepEqual(first.body[0].mediaUrls, eventWithMedia.mediaUrls);
  assert.equal(first.body[0]._inboundLease.deliveryId, staged.deliveryId);

  const restarted = restart();
  const afterRestart = fakeRouteApp();
  registerInboundSpoolRoutes(afterRestart, restarted);
  // A competing consumer cannot take a live lease after bridge restart.
  assert.deepEqual(
    callRoute(afterRestart.routes.get.get('/messages'), { query: { consumerId: 'python-b' } }).body,
    [],
  );
  const ack = callRoute(afterRestart.routes.post.get('/messages/ack'), {
    body: first.body[0]._inboundLease,
  });
  assert.equal(ack.status, 200);
  assert.equal(ack.body.status, 'acknowledged');
  assert.deepEqual(
    callRoute(afterRestart.routes.get.get('/messages'), { query: { consumerId: 'python-a' } }).body,
    [],
  );
  assert.equal(restarted.stats().tombstones, 1);
  assert.equal(readdirSync(path.join(root, 'records')).length, 0);
}));

test('canonical identity isolates profile, account, participant, and fromMe', () => withSpool(({ spool }) => {
  const variants = [
    [PROFILE, ACCOUNT, event()],
    ['c'.repeat(64), ACCOUNT, event()],
    [PROFILE, 'd'.repeat(64), event()],
    [PROFILE, ACCOUNT, event({
      senderId: '15550000000@s.whatsapp.net',
      readReceiptKey: {
        ...event().readReceiptKey,
        participant: '15550000000@s.whatsapp.net',
      },
    })],
    [PROFILE, ACCOUNT, event({ readReceiptKey: { ...event().readReceiptKey, fromMe: true } })],
  ];
  const ids = variants.map(([profileNamespace, accountNamespace, value]) => spool.append({
    profileNamespace,
    accountNamespace,
    event: value,
  }).deliveryId);
  assert.equal(new Set(ids).size, variants.length);
}));

test('device suffix and later alias-map enrichment do not change delivery identity', () => withSpool(({ spool }) => {
  const first = spool.append({
    profileNamespace: PROFILE,
    accountNamespace: ACCOUNT,
    event: event(),
    lidAliases: {},
  });
  const duplicate = spool.append({
    profileNamespace: PROFILE,
    accountNamespace: ACCOUNT,
    event: event({
      senderId: '15551234567:9@s.whatsapp.net',
      readReceiptKey: {
        ...event().readReceiptKey,
        participant: '15551234567:9@s.whatsapp.net',
      },
    }),
    lidAliases: { '15551234567': '19999999999' },
  });
  assert.equal(duplicate.status, 'duplicate_pending');
  assert.equal(duplicate.deliveryId, first.deliveryId);
  assert.equal(duplicate.eventDigest, first.eventDigest);
}));

test('persisted account namespace survives identity enrichment and isolates disjoint accounts', () => withSpool(({ spool, restart }) => {
  const first = spool.resolveAccountNamespace(['15551234567:3@s.whatsapp.net']);
  const enriched = restart().resolveAccountNamespace([
    '15551234567:8@s.whatsapp.net',
    '90000000000000:2@lid',
  ]);
  assert.equal(enriched, first);
  const unrelated = restart().resolveAccountNamespace([
    '16662345678:1@s.whatsapp.net',
    '80000000000000:1@lid',
  ]);
  assert.notEqual(unrelated, first);
}));

test('staging failure is isolated without logging private event data', () => {
  let appendCalls = 0;
  const failures = [];
  const result = stageInboundEventSafely({
    spool: { append() { appendCalls += 1; } },
    profileNamespace: PROFILE,
    accountNamespace: null,
    event: event({ body: 'private body' }),
    onFailure: reason => failures.push(reason),
  });
  assert.equal(result, null);
  assert.equal(appendCalls, 0);
  assert.deepEqual(failures, ['account_not_ready']);
});

test('event digest ignores transient cache paths without reading media', () => {
  const first = inboundEventDigest(event({
    hasMedia: true,
    mediaType: 'image',
    mime: 'image/jpeg',
    mediaUrls: ['/does/not/exist/first.jpg'],
  }));
  const second = inboundEventDigest(event({
    hasMedia: true,
    mediaType: 'image',
    mime: 'image/jpeg',
    mediaUrls: ['/also/missing/second.jpg'],
  }));
  assert.equal(first.eventDigest, second.eventDigest);
  assert.equal(first.mediaManifest.state, 'unowned');
  assert.equal(first.mediaManifest.owner, 'none');
  assert.equal(first.mediaManifest.ownedRoot, null);
  assert.equal(first.mediaManifest.integrity, 'unverified');
  assert.equal(first.mediaManifest.entries[0].sourcePath, '/does/not/exist/first.jpg');
  assert.equal(first.mediaManifest.entries[0].state, 'unowned');
});

test('digest canonical numbers cover fixed, exponent, integer, and negative zero forms', () => {
  assert.equal(
    canonicalDigestJson({ values: [1, 1.5, 0.000001, 1.25e-7, -0] }),
    '{"values":[1,1.5,0.000001,1.25e-7,0]}',
  );
  assert.throws(() => canonicalDigestJson({ value: Number.NaN }), /non-finite/);
  assert.throws(() => canonicalDigestJson({ value: Number.POSITIVE_INFINITY }), /non-finite/);
  assert.throws(() => canonicalDigestJson({ value: 9007199254740992 }), /unsafe integer/);
});

test('declared byte hashes distinguish media with identical visible metadata', () => withSpool(({ spool }) => {
  const common = {
    hasMedia: true,
    mediaType: 'image',
    mime: 'image/jpeg',
    fileName: 'photo.jpg',
    mediaUrls: ['/bridge/cache/photo.jpg'],
  };
  const first = spool.append({
    profileNamespace: PROFILE,
    accountNamespace: ACCOUNT,
    event: event({ ...common, mediaMetadata: [{ sha256: '1'.repeat(64), size: '42' }] }),
  });
  const distinct = spool.append({
    profileNamespace: PROFILE,
    accountNamespace: ACCOUNT,
    event: event({ ...common, mediaMetadata: [{ sha256: '2'.repeat(64), size: '42' }] }),
  });
  assert.equal(first.status, 'appended');
  assert.equal(distinct.status, 'quarantined');
  assert.notEqual(first.eventDigest, distinct.eventDigest);
  assert.deepEqual(spool.lease({ consumerId: 'python-a' }), []);
}));

test('flat lease preserves content, caption, reply, media references, and append order', () => withSpool(({ spool }) => {
  const firstEvent = event({
    body: 'caption',
    hasMedia: true,
    mediaType: 'document',
    mime: 'application/pdf',
    fileName: 'report.pdf',
    mediaUrls: ['/bridge/cache/report.pdf'],
    quotedMessageId: 'wamid.parent',
    quotedParticipant: '15557654321@s.whatsapp.net',
    quotedText: 'parent text',
    hasQuotedMessage: true,
  });
  const secondEvent = event({
    messageId: 'wamid.two',
    body: 'second',
    readReceiptKey: { ...event().readReceiptKey, id: 'wamid.two' },
  });
  spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: firstEvent });
  spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: secondEvent });
  const leased = spool.lease({ consumerId: 'python-a' });
  assert.equal(leased.length, 2);
  assert.equal(leased[0].body, firstEvent.body);
  assert.deepEqual(leased[0].mediaUrls, firstEvent.mediaUrls);
  assert.equal(leased[0].quotedMessageId, firstEvent.quotedMessageId);
  assert.equal(leased[0].quotedParticipant, firstEvent.quotedParticipant);
  assert.equal(leased[0].quotedText, firstEvent.quotedText);
  assert.equal(leased[1].body, 'second');
  assert.ok(leased[0]._inboundLease.sequence < leased[1]._inboundLease.sequence);
}));

test('lease remains a flat list-compatible event and survives restart', () => withSpool(({ spool, restart }) => {
  const appended = spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: event() });
  assert.equal(appended.status, 'appended');
  const [leased] = spool.lease({ consumerId: 'python-a' });
  assert.equal(leased.body, 'hello');
  assert.equal(leased._inboundLease.consumerId, 'python-a');
  assert.equal(leased._inboundLease.deliveryId, appended.deliveryId);
  assert.equal(spool.stats().pending, 1);

  const [afterRestart] = restart().lease({ consumerId: 'python-a' });
  assert.deepEqual(afterRestart._inboundLease, leased._inboundLease);
}));

test('live lease is stable for its consumer and fenced from another consumer', () => withSpool(({ spool }) => {
  spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: event() });
  const [first] = spool.lease({ consumerId: 'python-a' });
  const [sameConsumer] = spool.lease({ consumerId: 'python-a' });
  assert.deepEqual(sameConsumer._inboundLease, first._inboundLease);
  assert.deepEqual(spool.lease({ consumerId: 'python-b' }), []);
}));

test('lease cursor fairly exposes later healthy ingress past an unacked page', () => withSpool(({ spool, restart }) => {
  for (let index = 0; index < 7; index += 1) {
    const messageId = `wamid.fair.${index}`;
    spool.append({
      profileNamespace: PROFILE,
      accountNamespace: ACCOUNT,
      event: event({
        messageId,
        body: `message-${index}`,
        readReceiptKey: { ...event().readReceiptKey, id: messageId },
      }),
    });
  }

  const first = spool.lease({ consumerId: 'python-a', limit: 3 });
  assert.deepEqual(first.map(item => item.body), ['message-0', 'message-1', 'message-2']);
  const second = restart().lease({ consumerId: 'python-a', limit: 3 });
  assert.deepEqual(second.map(item => item.body), ['message-3', 'message-4', 'message-5']);
  const third = restart().lease({ consumerId: 'python-a', limit: 3 });
  assert.equal(third[0].body, 'message-6');
  assert.equal(spool.stats().pending, 7);
}));

test('spool initialization rejects an ancestor symbolic link', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-inbound-spool-root-'));
  const real = path.join(root, 'real');
  const alias = path.join(root, 'alias');
  try {
    mkdirSync(real);
    symlinkSync(real, alias, 'dir');
    assert.throws(
      () => createInboundSpool(path.join(alias, 'spool')),
      /untrusted symbolic-link component/,
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('expired lease is redelivered with fenced epoch and token', () => withSpool(({ spool, setNow }) => {
  spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: event() });
  const [first] = spool.lease({ consumerId: 'python-a' });
  setNow(first._inboundLease.expiresAt + 1);
  const [second] = spool.lease({ consumerId: 'python-b' });
  assert.equal(second._inboundLease.deliveryId, first._inboundLease.deliveryId);
  assert.equal(second._inboundLease.epoch, first._inboundLease.epoch + 1);
  assert.notEqual(second._inboundLease.token, first._inboundLease.token);
  assert.equal(spool.acknowledge(first._inboundLease).status, 'stale_lease');
}));

test('renew rotates the token and exact ACK creates an idempotent tombstone', () => withSpool(({ spool, root }) => {
  spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: event() });
  const [leased] = spool.lease({ consumerId: 'python-a' });
  const initial = leased._inboundLease;
  const renewed = spool.renew(initial);
  assert.equal(renewed.status, 'renewed');
  assert.equal(renewed.delivery.epoch, initial.epoch);
  assert.notEqual(renewed.delivery.token, initial.token);
  assert.equal(spool.acknowledge(initial).status, 'stale_lease');

  const ack = spool.acknowledge(renewed.delivery);
  assert.equal(ack.status, 'acknowledged');
  assert.equal(spool.acknowledge(renewed.delivery).status, 'already_acknowledged');
  assert.equal(spool.stats().pending, 0);
  assert.equal(spool.stats().tombstones, 1);
  const [name] = readdirSync(path.join(root, 'tombstones'));
  const tombstone = JSON.parse(readFileSync(path.join(root, 'tombstones', name), 'utf8'));
  assert.equal(tombstone.terminalState, 'acknowledged');
  assert.equal('token' in tombstone, false);
}));

test('startup reconciles the tombstone-first ACK crash window', () => withSpool(({ spool, root, restart }) => {
  spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: event() });
  const [recordName] = readdirSync(path.join(root, 'records'));
  const savedRecord = path.join(root, 'saved-record.json');
  copyFileSync(path.join(root, 'records', recordName), savedRecord);
  const [leased] = spool.lease({ consumerId: 'python-a' });
  assert.equal(spool.acknowledge(leased._inboundLease).status, 'acknowledged');
  copyFileSync(savedRecord, path.join(root, 'records', recordName));
  const recovered = restart();
  assert.equal(recovered.stats().pending, 0);
  assert.equal(recovered.stats().tombstones, 1);
}));

test('digest collisions and corrupt records are quarantined without exposure', () => withSpool(({ spool, root }) => {
  const first = spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: event() });
  assert.equal(spool.append({
    profileNamespace: PROFILE,
    accountNamespace: ACCOUNT,
    event: event({ body: 'mutated payload' }),
  }).status, 'quarantined');
  assert.deepEqual(spool.lease({ consumerId: 'python-a' }), []);
  assert.equal(spool.stats().quarantined, 1);

  const healthyId = 'wamid.healthy';
  const healthy = spool.append({
    profileNamespace: PROFILE,
    accountNamespace: ACCOUNT,
    event: event({ messageId: healthyId, readReceiptKey: { ...event().readReceiptKey, id: healthyId } }),
  });
  writeFileSync(path.join(root, 'records', `${healthy.deliveryId}.json`), '{broken');
  assert.deepEqual(spool.lease({ consumerId: 'python-a' }), []);
  assert.equal(spool.stats().quarantined, 2);
  assert.equal(readdirSync(path.join(root, 'quarantine')).filter(name => name.endsWith('.retained')).length, 2);
  assert.match(first.deliveryId, /^[a-f0-9]{64}$/);
}));

test('invalid delivery IDs cannot escape the private spool root', () => withSpool(({ spool }) => {
  assert.deepEqual(spool.renew({
    consumerId: 'python-a', deliveryId: '../../outside', epoch: 1, token: 'x',
  }), { status: 'not_found' });
  assert.deepEqual(spool.acknowledge({
    consumerId: 'python-a', deliveryId: '../../outside', epoch: 1, token: 'x',
  }), { status: 'not_found' });
}));

test('spool directories and durable files are profile-private', () => withSpool(({ spool, root }) => {
  spool.append({ profileNamespace: PROFILE, accountNamespace: ACCOUNT, event: event() });
  assert.equal(statSync(root).mode & 0o777, 0o700);
  for (const dir of ['records', 'tombstones', 'quarantine']) {
    assert.equal(statSync(path.join(root, dir)).mode & 0o777, 0o700);
  }
  const [record] = readdirSync(path.join(root, 'records'));
  assert.equal(statSync(path.join(root, 'records', record)).mode & 0o777, 0o600);
  assert.equal(statSync(path.join(root, 'sequence.json')).mode & 0o777, 0o600);
  assert.equal(statSync(path.join(root, 'lease-cursor.json')).mode & 0o777, 0o600);
}));
