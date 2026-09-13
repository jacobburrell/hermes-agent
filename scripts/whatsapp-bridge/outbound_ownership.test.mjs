import { strict as assert } from 'node:assert';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { createOutboundOwnershipLedger, isVerifiedOutboundQuote } from './outbound_ownership.js';

const chatA = '120363001234567890-1234567890@g.us';
const chatB = '120363009999999999-1234567890@g.us';

{
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-wa-owned-'));
  const filePath = path.join(dir, 'outbound-owned-v1.json');
  const ledger = createOutboundOwnershipLedger({ filePath, maxEntries: 2 });
  assert.equal(ledger.remember({
    messageId: 'jack-1', chatId: chatA, accountNamespace: 'account-a',
  }), true);
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'jack-1', chatId: chatA, quotedRemoteJid: chatA,
    accountNamespace: 'account-a', ownershipLedger: ledger,
  }), true);
  // Message IDs alone are never enough: cross-chat, supplied conflict, and
  // account rotation each deny the quote.
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'jack-1', chatId: chatB, quotedRemoteJid: chatB,
    accountNamespace: 'account-a', ownershipLedger: ledger,
  }), false);
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'jack-1', chatId: chatA, quotedRemoteJid: chatB,
    accountNamespace: 'account-a', ownershipLedger: ledger,
  }), false);
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'jack-1', chatId: chatA, quotedRemoteJid: chatA,
    accountNamespace: 'account-rotated', ownershipLedger: ledger,
  }), false);
  // Baileys may omit remoteJid for a same-chat native reply.  Only genuine
  // absence uses the authenticated enclosing chat; explicit bad values deny.
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'jack-1', chatId: chatA, accountNamespace: 'account-a', ownershipLedger: ledger,
  }), true);
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'jack-1', chatId: chatA, quotedRemoteJid: '', quotedRemoteJidPresent: true,
    accountNamespace: 'account-a', ownershipLedger: ledger,
  }), false);
  console.log('  ✓ outbound quote proof is account+chat scoped and supports only absent same-chat remote');
}

{
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-wa-owned-device-'));
  const ledger = createOutboundOwnershipLedger({
    filePath: path.join(dir, 'outbound-owned-v1.json'),
  });
  assert.equal(ledger.remember({
    messageId: 'device-1', chatId: '15551234567:7@s.whatsapp.net', accountNamespace: 'account-a',
  }), true);
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'device-1', chatId: '15551234567:7@s.whatsapp.net',
    accountNamespace: 'account-a', ownershipLedger: ledger,
  }), true);
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'device-1', chatId: '15551234567@s.whatsapp.net',
    accountNamespace: 'account-a', ownershipLedger: ledger,
  }), false);
  console.log('  ✓ device JIDs retain their account/chat ownership scope');
}

{
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-wa-owned-fsync-'));
  const ledger = createOutboundOwnershipLedger({
    filePath: path.join(dir, 'outbound-owned-v1.json'),
    fsyncDirectory: () => { throw new Error('directory fsync failed'); },
  });
  assert.throws(() => ledger.remember({
    messageId: 'memory-only', chatId: chatA, accountNamespace: 'account-a',
  }), /directory fsync failed/);
  assert.equal(isVerifiedOutboundQuote({
    messageId: 'memory-only', chatId: chatA, quotedRemoteJid: chatA,
    accountNamespace: 'account-a', ownershipLedger: ledger,
  }), true);
  assert.equal(ledger.snapshot()[0].durable, false);
  console.log('  ✓ failed directory fsync retains only a non-durable scoped proof');
}
