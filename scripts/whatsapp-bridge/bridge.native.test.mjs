/**
 * Unit tests for WhatsApp-native bridge payload helpers.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { createHash, randomBytes } from 'node:crypto';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { getAggregateVotesInPollMessage } from '@whiskeysockets/baileys';

import {
  buildPollPayload,
  buildTextSendPayload,
  createBoundedInboundMediaStore,
  createBoundedMessageStore,
  createInboundMediaMaterializer,
  appendMediaFailureNote,
  extractBridgeEvent,
  inboundReadReceiptKeys,
  mediaPayloadForFile,
  pollCreationMessageFromPayload,
  pollUpdateForAggregation,
} from './bridge_helpers.js';

// -- inbound read receipts ------------------------------------------------
{
  const groupKey = {
    id: 'incoming-group-1',
    remoteJid: '120363001234567890@g.us',
    participant: '15550001111@s.whatsapp.net',
    fromMe: false,
  };

  assert.deepEqual(inboundReadReceiptKeys({ key: groupKey, enabled: false }), []);
  assert.deepEqual(
    inboundReadReceiptKeys({ key: { ...groupKey, fromMe: true }, enabled: true }),
    [],
  );
  const receiptKeys = inboundReadReceiptKeys({ key: groupKey, enabled: true });
  assert.equal(receiptKeys.length, 1);
  assert.equal(receiptKeys[0], groupKey);
  assert.equal(receiptKeys[0].participant, groupKey.participant);
  console.log('  ✓ inbound read receipts preserve the original group message key');
}

// -- deferred inbound media materialization ------------------------------
{
  let downloads = 0;
  let cacheWrites = 0;
  let materializations = 0;
  let clock = 10_000;
  const rawMessage = {
    key: { id: 'deferred-image-1', remoteJid: '120363001234567890@g.us', fromMe: false },
    messageTimestamp: 123,
    message: { imageMessage: { caption: 'addressed image', mimetype: 'image/jpeg' } },
  };
  const eventArgs = {
    msg: rawMessage,
    chatId: '120363001234567890@g.us',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    isGroup: true,
    downloadMedia: async () => { downloads += 1; return Buffer.from('image'); },
    writeMediaFile: async () => { cacheWrites += 1; return '/private/cache/img.jpg'; },
  };

  // This is the /messages representation used by both DROP and OBSERVE
  // before Python classification: no downloader, writer, or raw path leaks.
  const metadataOnly = await extractBridgeEvent({ ...eventArgs, materializeMedia: false });
  assert.equal(metadataOnly.hasMedia, true);
  assert.deepEqual(metadataOnly.mediaUrls, []);
  assert.equal(metadataOnly.body, 'addressed image');
  assert.equal(downloads, 0);
  assert.equal(cacheWrites, 0);

  const inboundStore = createBoundedInboundMediaStore({
    limit: 1,
    ttlMs: 100,
    now: () => clock,
  });
  assert.equal(inboundStore.remember({
    chatId: eventArgs.chatId,
    messageId: rawMessage.key.id,
    msg: rawMessage,
    context: eventArgs,
  }), true);

  const capability = randomBytes(32).toString('base64url');
  const materializeEntry = async (entry) => {
    materializations += 1;
    const full = await extractBridgeEvent({ ...entry.context, msg: entry.msg, materializeMedia: true });
    return {
      mediaUrls: full.mediaUrls,
      mediaType: full.mediaType,
      mime: full.mime,
      fileName: full.fileName,
      // Deliberately ignored by the helper response shaping.
      nativeMetadata: full.nativeMetadata,
    };
  };
  const materializer = createInboundMediaMaterializer({ capability, inboundMediaStore: inboundStore, materialize: materializeEntry });

  const wrongCapability = await materializer({
    suppliedCapability: randomBytes(32).toString('base64url'),
    chatId: eventArgs.chatId,
    messageId: rawMessage.key.id,
  });
  assert.equal(wrongCapability.status, 403);
  assert.equal(downloads, 0);
  assert.equal(cacheWrites, 0);
  assert.equal(materializations, 0);

  // Correctly authenticated requests are still exact-origin-bound; a wrong
  // chat or message does not consume or disclose the retained raw media.
  const wrongChat = await materializer({
    suppliedCapability: capability,
    chatId: '120363009999999999@g.us',
    messageId: rawMessage.key.id,
  });
  assert.equal(wrongChat.status, 404);
  const wrongMessage = await materializer({
    suppliedCapability: capability,
    chatId: eventArgs.chatId,
    messageId: 'different-message',
  });
  assert.equal(wrongMessage.status, 404);
  assert.equal(downloads, 0);
  assert.equal(cacheWrites, 0);

  const materialized = await materializer({
    suppliedCapability: capability,
    chatId: eventArgs.chatId,
    messageId: rawMessage.key.id,
  });
  assert.equal(materialized.status, 200);
  assert.deepEqual(Object.keys(materialized.result).sort(), ['chatId', 'fileName', 'mediaType', 'mediaUrls', 'messageId', 'mime']);
  assert.deepEqual(materialized.result.mediaUrls, ['/private/cache/img.jpg']);
  assert.equal(downloads, 1);
  assert.equal(cacheWrites, 1);
  assert.equal(materializations, 1);
  assert.equal((await materializer({
    suppliedCapability: capability, chatId: eventArgs.chatId, messageId: rawMessage.key.id,
  })).status, 404);

  // Retention is bounded and expires without materializing either raw entry.
  assert.equal(inboundStore.remember({ chatId: eventArgs.chatId, messageId: 'old', msg: { key: { id: 'old', remoteJid: eventArgs.chatId } } }), true);
  assert.equal(inboundStore.remember({ chatId: eventArgs.chatId, messageId: 'new', msg: { key: { id: 'new', remoteJid: eventArgs.chatId } } }), true);
  assert.equal(inboundStore.take({ chatId: eventArgs.chatId, messageId: 'old' }), null);
  clock += 101;
  assert.equal(inboundStore.take({ chatId: eventArgs.chatId, messageId: 'new' }), null);
  assert.equal(downloads, 1);
  assert.equal(cacheWrites, 1);

  // Device JIDs and an older bridge's accidental double-@ display spelling
  // resolve to the same private origin key without exposing the raw message.
  const aliasStore = createBoundedInboundMediaStore();
  const deviceRaw = { key: { id: 'device-message', remoteJid: '120363001234567890:47@g.us' } };
  assert.equal(aliasStore.remember({
    chatId: '120363001234567890@47@g.us', messageId: 'device-message', msg: deviceRaw,
  }), true);
  const deviceEntry = aliasStore.take({ chatId: '120363001234567890:99@g.us', messageId: 'device-message' });
  assert.equal(deviceEntry?.chatId, '120363001234567890@g.us');
  console.log('  ✓ deferred inbound media is capability-bound, origin-bound, one-shot, and metadata-only before admission');
}

// A direct-message LID has the same deferred-materialization contract as a
// group.  In particular, its raw Baileys route—not a derived display label—is
// what establishes origin, and an unrelated phone JID must not download it.
{
  let downloads = 0;
  const rawDirectLid = {
    key: { id: 'deferred-direct-lid-1', remoteJid: '19876543210:42@lid', fromMe: false },
    message: { imageMessage: { caption: 'direct LID image', mimetype: 'image/jpeg' } },
  };
  const directMetadata = await extractBridgeEvent({
    msg: rawDirectLid,
    chatId: rawDirectLid.key.remoteJid,
    senderId: rawDirectLid.key.remoteJid,
    senderNumber: '19876543210',
    isGroup: false,
    materializeMedia: false,
    downloadMedia: async () => { downloads += 1; return Buffer.from('direct-image'); },
    writeMediaFile: async () => '/private/cache/direct-lid.jpg',
  });
  assert.equal(directMetadata.chatId, rawDirectLid.key.remoteJid);
  assert.deepEqual(directMetadata.mediaUrls, []);
  assert.equal(downloads, 0);

  const directStore = createBoundedInboundMediaStore();
  assert.equal(directStore.remember({
    chatId: directMetadata.chatId,
    messageId: directMetadata.messageId,
    msg: rawDirectLid,
    context: { chatId: directMetadata.chatId },
  }), true);
  const directCapability = randomBytes(32).toString('base64url');
  let internalCanonicalChatId = '';
  const directMaterializer = createInboundMediaMaterializer({
    capability: directCapability,
    inboundMediaStore: directStore,
    materialize: async (entry) => {
      downloads += 1;
      internalCanonicalChatId = entry.chatId;
      return {
        mediaUrls: ['/private/cache/direct-lid.jpg'],
        mediaType: 'image', mime: 'image/jpeg', fileName: '',
      };
    },
  });

  // A different direct-message identity cannot consume or download the raw
  // LID message, even with the capability and message ID.
  assert.equal((await directMaterializer({
    suppliedCapability: directCapability,
    chatId: '19876543210@s.whatsapp.net',
    messageId: directMetadata.messageId,
  })).status, 404);
  assert.equal(downloads, 0);

  // The API request uses the exact queued raw LID route.  The store may
  // canonicalize only internally for its bounded binding key; the response
  // must echo the original queued spelling that Python correlates.
  const directMaterialized = await directMaterializer({
    suppliedCapability: directCapability,
    chatId: directMetadata.chatId,
    messageId: directMetadata.messageId,
  });
  assert.equal(directMaterialized.status, 200);
  assert.equal(internalCanonicalChatId, '19876543210@lid');
  assert.equal(directMaterialized.result.chatId, directMetadata.chatId);
  assert.deepEqual(directMaterialized.result.mediaUrls, ['/private/cache/direct-lid.jpg']);
  assert.equal(downloads, 1);
  assert.equal((await directMaterializer({
    suppliedCapability: directCapability,
    chatId: directMetadata.chatId,
    messageId: directMetadata.messageId,
  })).status, 404);
  console.log('  ✓ direct-message LID media binds raw route, internal canonical key, and one-shot download');
}

// -- quoted outbound text -------------------------------------------------
{
  const store = createBoundedMessageStore(2);
  store.remember({
    key: {
      id: 'inbound-1',
      remoteJid: '15551234567@s.whatsapp.net',
      participant: '15550001111@s.whatsapp.net',
      fromMe: false,
    },
    message: { conversation: 'original text' },
  });

  const { content, options } = buildTextSendPayload('reply text', {
    chatId: '15551234567@s.whatsapp.net',
    replyTo: 'inbound-1',
    messageStore: store,
  });

  assert.deepEqual(content, { text: 'reply text' });
  assert.equal(options.quoted.key.id, 'inbound-1');
  assert.equal(options.quoted.message.conversation, 'original text');
  console.log('  ✓ text replies include Baileys quoted message when resolvable');
}

{
  const store = createBoundedMessageStore(2);
  const { content, options } = buildTextSendPayload('plain text', {
    chatId: '15551234567@s.whatsapp.net',
    replyTo: 'missing-id',
    messageStore: store,
  });

  assert.deepEqual(content, { text: 'plain text' });
  assert.deepEqual(options, {});
  console.log('  ✓ unresolved replyTo falls back to plain text');
}

// -- inbound quote/media/native metadata --------------------------------
{
  const event = await extractBridgeEvent({
    msg: {
      key: {
        id: 'incoming-1',
        remoteJid: '15551234567@s.whatsapp.net',
        participant: '15550001111@s.whatsapp.net',
        fromMe: false,
      },
      pushName: 'Tester',
      messageTimestamp: 123,
      message: {
        extendedTextMessage: {
          text: 'approved',
          contextInfo: {
            stanzaId: 'outbound-1',
            participant: '15559998888@s.whatsapp.net',
            remoteJid: '15551234567@s.whatsapp.net',
            quotedMessage: { conversation: 'approve deploy?' },
          },
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    botIds: ['15559998888@s.whatsapp.net'],
    downloadMedia: async () => Buffer.from(''),
  });

  assert.equal(event.quotedMessageId, 'outbound-1');
  assert.equal(event.quotedParticipant, '15559998888@s.whatsapp.net');
  assert.equal(event.quotedRemoteJid, '15551234567@s.whatsapp.net');
  assert.equal(event.quotedText, 'approve deploy?');
  assert.deepEqual(event.readReceiptKey, {
    id: 'incoming-1',
    remoteJid: '15551234567@s.whatsapp.net',
    participant: '15550001111@s.whatsapp.net',
    fromMe: false,
  });
  assert.equal(event.hasQuotedMessage, true);
  assert.equal(event.body, 'approved');
  console.log('  ✓ inbound quoted metadata includes quoted text');
}

{
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'doc-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        documentMessage: {
          caption: 'see attached',
          fileName: 'report.pdf',
          mimetype: 'application/pdf',
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    downloadMedia: async () => Buffer.from('pdf'),
    writeMediaFile: async () => '/tmp/report.pdf',
  });

  assert.equal(event.hasMedia, true);
  assert.equal(event.mediaType, 'document');
  assert.equal(event.mime, 'application/pdf');
  assert.equal(event.fileName, 'report.pdf');
  assert.equal(event.nativeType, 'documentMessage');
  assert.deepEqual(event.mediaUrls, ['/tmp/report.pdf']);
  console.log('  ✓ inbound document metadata preserves MIME and filename');
}

{
  const cacheDir = mkdtempSync(path.join(tmpdir(), 'hermes-wa-doc-'));
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'doc-2', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        documentMessage: {
          caption: 'see attached',
          fileName: 'report',
          mimetype: 'application/pdf',
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    downloadMedia: async () => Buffer.from('pdf'),
    cacheDirs: { document: cacheDir },
  });

  assert.equal(event.mediaUrls.length, 1);
  assert.ok(event.mediaUrls[0].endsWith('_report.pdf'), event.mediaUrls[0]);
  console.log('  ✓ MIME extension is preserved when document filename has none');
}

{
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'loc-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        locationMessage: {
          name: 'HQ',
          degreesLatitude: 41.015,
          degreesLongitude: 28.979,
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
  });

  assert.equal(event.mediaType, 'location');
  assert.equal(event.body, '[Location: HQ 41.015,28.979]');
  assert.deepEqual(event.nativeMetadata.location, {
    name: 'HQ',
    address: '',
    latitude: 41.015,
    longitude: 28.979,
    isLive: false,
  });
  console.log('  ✓ native location messages get text fallback and metadata');
}

{
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'poll-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        pollCreationMessage: {
          name: 'Approve deploy?',
          options: [{ optionName: 'Approve' }, { optionName: 'Deny' }],
          selectableOptionsCount: 1,
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
  });

  assert.equal(event.mediaType, 'poll');
  assert.equal(event.body, '[Poll: Approve deploy? Options: Approve, Deny]');
  assert.deepEqual(event.nativeMetadata.poll.options, ['Approve', 'Deny']);
  console.log('  ✓ poll creation messages get text fallback and metadata');
}

// -- outbound media/poll helpers -----------------------------------------
{
  const payload = mediaPayloadForFile({
    buffer: Buffer.from('gif89a'),
    filePath: '/tmp/loop.gif',
    mediaType: 'image',
    caption: 'loop',
  });

  assert.ok(payload.image, 'pure helper fallback keeps raw GIF as image bytes');
  assert.equal(payload.gifPlayback, undefined);
  assert.equal(payload.mimetype, 'image/gif');
  assert.equal(payload.caption, 'loop');
  console.log('  ✓ local GIF helper fallback stays truthful; live bridge converts to gifPlayback when possible');
}

{
  const payload = buildPollPayload({
    question: 'Proceed?',
    options: ['Approve', 'Deny'],
    selectableCount: 1,
  });

  assert.equal(payload.poll.name, 'Proceed?');
  assert.deepEqual(payload.poll.values, ['Approve', 'Deny']);
  assert.equal(payload.poll.selectableCount, 1);
  assert.equal(Buffer.isBuffer(payload.poll.messageSecret), true);
  assert.equal(payload.poll.messageSecret.length, 32);
  assert.deepEqual(pollCreationMessageFromPayload(payload), {
    messageContextInfo: {
      messageSecret: payload.poll.messageSecret,
    },
    pollCreationMessageV3: {
      name: 'Proceed?',
      options: [{ optionName: 'Approve' }, { optionName: 'Deny' }],
      selectableOptionsCount: 1,
    },
  });
  console.log('  ✓ poll payload primitive carries a cacheable vote secret');
}

{
  const pollCreation = {
    key: {
      id: 'poll-creation',
      remoteJid: '15551234567@s.whatsapp.net',
      fromMe: true,
    },
    message: {
      messageContextInfo: {
        messageSecret: Buffer.from('0123456789abcdef0123456789abcdef'),
      },
      pollCreationMessageV3: {
        name: 'Proceed?',
        options: [{ optionName: 'Approve' }, { optionName: 'Deny' }],
        selectableOptionsCount: 1,
      },
    },
  };
  const voteKey = {
    id: 'vote-message',
    remoteJid: '15551234567@s.whatsapp.net',
    participant: '15550001111@s.whatsapp.net',
    fromMe: false,
  };
  const encryptedVote = {
    encPayload: Buffer.from('payload'),
    encIv: Buffer.from('iv'),
  };

  const attempts = [];
  const pollUpdate = pollUpdateForAggregation({
    pollUpdateMessage: {
      pollCreationMessageKey: pollCreation.key,
      vote: encryptedVote,
      senderTimestampMs: 123,
    },
    pollUpdateMessageKey: voteKey,
    pollCreation,
    decryptPollVote: (vote, ctx) => {
      attempts.push({ pollCreatorJid: ctx.pollCreatorJid, voterJid: ctx.voterJid });
      assert.equal(vote, encryptedVote);
      assert.equal(ctx.pollMsgId, 'poll-creation');
      assert.equal(ctx.pollEncKey, pollCreation.message.messageContextInfo.messageSecret);
      if (ctx.pollCreatorJid !== 'creator-lid@lid') {
        throw new Error('wrong creator jid');
      }
      assert.equal(ctx.voterJid, '15550001111@s.whatsapp.net');
      return {
        selectedOptions: [createHash('sha256').update(Buffer.from('Approve')).digest()],
      };
    },
    getKeyAuthor: (key, meId = 'me') => (key?.fromMe ? meId : key?.participant || key?.remoteJid || ''),
    meId: 'classic-me@s.whatsapp.net',
    pollCreatorJids: ['classic-me@s.whatsapp.net', 'creator-lid@lid'],
  });

  assert.deepEqual(attempts.map(item => item.pollCreatorJid), ['classic-me@s.whatsapp.net', 'creator-lid@lid']);

  assert.equal(pollUpdate.pollUpdateMessageKey.id, 'vote-message');
  assert.equal(pollUpdate.senderTimestampMs, 123);
  const aggregation = getAggregateVotesInPollMessage({
    message: pollCreation.message,
    pollUpdates: [pollUpdate],
  });
  assert.deepEqual(
    aggregation.map(option => ({ name: option.name, voters: option.voters })),
    [
      { name: 'Approve', voters: ['15550001111@s.whatsapp.net'] },
      { name: 'Deny', voters: [] },
    ],
  );
  console.log('  ✓ encrypted poll upserts are wrapped into Baileys aggregation shape');
}

// -- media download failure containment (port of nanoclaw#2895) -----------
{
  assert.equal(appendMediaFailureNote('hello', []), 'hello');
  assert.equal(
    appendMediaFailureNote('check this out', ['image']),
    'check this out\n[image could not be downloaded]',
  );
  // Regression guard: an uncaptioned failed image must still produce a
  // non-empty body, or the empty-message guard drops the whole message.
  assert.equal(appendMediaFailureNote('', ['image']), '[image could not be downloaded]');
  assert.equal(
    appendMediaFailureNote('', ['image', 'document']),
    '[image could not be downloaded] [document could not be downloaded]',
  );
  console.log('  ✓ appendMediaFailureNote formats failure notes');
}

{
  // A throwing downloadMedia (expired CDN URL) must not reject out of
  // extractBridgeEvent — before this guard the whole upsert batch died and
  // the message was silently dropped.
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'img-fail-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: { imageMessage: { caption: '', mimetype: 'image/jpeg' } },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net',
    senderNumber: '15551234567',
    downloadMedia: async () => { throw new Error('Failed to fetch stream from https://mmg.whatsapp.net/x'); },
    cacheDirs: { image: mkdtempSync(path.join(tmpdir(), 'wa-media-')) },
  });
  assert.equal(event.hasMedia, true);
  assert.equal(event.mediaUrls.length, 0);
  assert.equal(event.body, '[image could not be downloaded]');
  console.log('  ✓ failed media download is contained and surfaced in body');
}

{
  // Captioned message keeps the caption and appends the failure note.
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'doc-fail-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: { documentMessage: { caption: 'see attached', fileName: 'q.pdf', mimetype: 'application/pdf' } },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net',
    senderNumber: '15551234567',
    downloadMedia: async () => { throw new Error('boom'); },
    cacheDirs: { document: mkdtempSync(path.join(tmpdir(), 'wa-media-')) },
  });
  assert.equal(event.body, 'see attached\n[document could not be downloaded]');
  assert.equal(event.mediaUrls.length, 0);
  console.log('  ✓ captioned failed download keeps caption and appends note');
}

console.log('\n✅ All WhatsApp native bridge helper tests passed.');
