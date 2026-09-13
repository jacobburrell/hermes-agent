/**
 * Bounded, durable proof of messages sent by this bridge.
 *
 * Native WhatsApp replies identify the quoted stanza by id, but an id alone is
 * not enough evidence: the same id must never cross an account or chat.  This
 * ledger intentionally keeps no message body or media, only that scoped proof.
 */

import {
  closeSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync,
  unlinkSync, writeFileSync,
} from 'fs';
import path from 'path';
import { randomBytes } from 'crypto';
import { normalizeWhatsAppId } from './bridge_helpers.js';

function normalizeId(value) {
  return normalizeWhatsAppId(value);
}

function defaultFsyncDirectory(dir) {
  const fd = openSync(dir, 'r');
  try {
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function readEntries(filePath, maxEntries) {
  try {
    const parsed = JSON.parse(readFileSync(filePath, 'utf8'));
    const raw = Array.isArray(parsed?.entries) ? parsed.entries : [];
    return raw
      .filter(entry => entry && typeof entry === 'object'
        && typeof entry.messageId === 'string' && entry.messageId
        && typeof entry.chatId === 'string' && entry.chatId
        && typeof entry.accountNamespace === 'string' && entry.accountNamespace)
      .slice(-maxEntries)
      .map(entry => ({
        messageId: entry.messageId,
        chatId: normalizeId(entry.chatId),
        accountNamespace: entry.accountNamespace,
        durable: true,
      }))
      .filter(entry => entry.chatId);
  } catch {
    return [];
  }
}

export function createOutboundOwnershipLedger({
  filePath,
  maxEntries = 512,
  fsyncDirectory = defaultFsyncDirectory,
} = {}) {
  if (!filePath) throw new TypeError('filePath is required');
  if (!Number.isInteger(maxEntries) || maxEntries < 1) {
    throw new RangeError('maxEntries must be a positive integer');
  }

  let entries = readEntries(filePath, maxEntries);

  function persist(nextEntries) {
    const dir = path.dirname(filePath);
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    const tmp = path.join(dir, `.${path.basename(filePath)}.${randomBytes(8).toString('hex')}.tmp`);
    try {
      const fd = openSync(tmp, 'w', 0o600);
      try {
        writeFileSync(fd, JSON.stringify({
          version: 1,
          entries: nextEntries.map(({ messageId, chatId, accountNamespace }) => ({
            messageId, chatId, accountNamespace,
          })),
        }));
        fsyncSync(fd);
      } finally {
        closeSync(fd);
      }
      renameSync(tmp, filePath);
      // The file fsync does not make the rename durable; the parent directory
      // must be synced before a restarted bridge may rely on this record.
      fsyncDirectory(dir);
    } catch (err) {
      try { unlinkSync(tmp); } catch {}
      throw err;
    }
  }

  function remember({ messageId, chatId, accountNamespace } = {}) {
    const id = String(messageId || '').trim();
    const chat = normalizeId(chatId);
    const account = String(accountNamespace || '').trim();
    if (!id || !chat || !account) return false;
    const nextEntries = entries.filter(entry => !(
      entry.messageId === id && entry.chatId === chat && entry.accountNamespace === account
    ));
    nextEntries.push({ messageId: id, chatId: chat, accountNamespace: account, durable: false });
    if (nextEntries.length > maxEntries) nextEntries.splice(0, nextEntries.length - maxEntries);
    try {
      persist(nextEntries);
      entries = nextEntries.map(entry => ({ ...entry, durable: true }));
    } catch (err) {
      // This process can retain a narrow scoped proof, but it must not claim
      // that a failed directory sync survived a restart.
      entries = nextEntries;
      throw err;
    }
    return true;
  }

  function ownsQuote({
    messageId,
    chatId,
    quotedRemoteJid,
    quotedRemoteJidPresent,
    accountNamespace,
  } = {}) {
    const id = String(messageId || '').trim();
    const chat = normalizeId(chatId);
    const account = String(accountNamespace || '').trim();
    if (!id || !chat || !account) return false;

    const rawQuotedRemote = String(quotedRemoteJid || '').trim();
    const remoteWasSupplied = quotedRemoteJidPresent === true || !!rawQuotedRemote;
    const quotedChat = remoteWasSupplied ? normalizeId(quotedRemoteJid) : chat;
    if (!quotedChat || quotedChat !== chat) return false;
    return entries.some(entry => entry.messageId === id
      && entry.chatId === chat && entry.accountNamespace === account);
  }

  function snapshot() {
    return entries.map(entry => ({ ...entry }));
  }

  return { remember, ownsQuote, snapshot };
}

export function isVerifiedOutboundQuote({
  messageId,
  chatId,
  quotedRemoteJid,
  quotedRemoteJidPresent,
  accountNamespace,
  ownershipLedger,
} = {}) {
  return Boolean(ownershipLedger?.ownsQuote({
    messageId,
    chatId,
    quotedRemoteJid,
    quotedRemoteJidPresent,
    accountNamespace,
  }));
}
