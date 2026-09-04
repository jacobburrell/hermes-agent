/**
 * Unit tests for WhatsApp-native bridge payload helpers.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { getAggregateVotesInPollMessage } from '@whiskeysockets/baileys';

import {
  buildPollPayload,
  buildTextSendPayload,
  createBoundedMessageStore,
  drainMessageQueueForResponse,
  appendMediaFailureNote,
  extractBridgeEvent,
  inboundReadReceiptKeys,
  mediaPayloadForFile,
  normalizeWhatsAppId,
  pollCreationMessageFromPayload,
  pollUpdateForAggregation,
  rememberBridgeOutboundMessage,
  sendAndRememberBridgeOutbound,
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

// -- WhatsApp identifier normalization ------------------------------------
{
  const numericDeviceAliases = [
    ['123456789:1@lid', '123456789@lid'],
    ['15551234567:47@s.whatsapp.net', '15551234567@s.whatsapp.net'],
    ['bot.name-42:8@lid', 'bot.name-42@lid'],
  ];
  for (const [raw, canonical] of numericDeviceAliases) {
    assert.equal(normalizeWhatsAppId(raw), canonical);
    assert.equal(normalizeWhatsAppId(canonical), canonical);
    assert.equal(normalizeWhatsAppId(normalizeWhatsAppId(raw)), canonical);
  }
  assert.notEqual(
    normalizeWhatsAppId('123456789:1@lid'),
    normalizeWhatsAppId('987654321@lid'),
  );

  for (const malformedOrNonDeviceJid of [
    '123456789:worker@lid',
    'bot.name-42:worker@s.whatsapp.net',
    ':1@lid',
    '123456789:1@',
    '123456789:1@lid@extra',
    ' 123456789:1@lid',
  ]) {
    assert.equal(
      normalizeWhatsAppId(malformedOrNonDeviceJid),
      malformedOrNonDeviceJid,
    );
  }
  console.log('  ✓ only complete numeric device JIDs canonicalize idempotently');
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

// -- chat-bound outbound ownership ----------------------------------------
{
  const store = createBoundedMessageStore(2);
  const jackSent = {
    key: {
      id: 'jack-outbound-1',
      remoteJid: '15551234567:47@s.whatsapp.net',
      fromMe: true,
    },
    message: { conversation: 'Jack answer' },
  };
  assert.equal(
    rememberBridgeOutboundMessage(store, jackSent, {
      chatId: '15551234567@s.whatsapp.net',
      payload: { text: 'Jack answer' },
    }),
    true,
  );
  assert.equal(
    store.quotedOutboundByJack('jack-outbound-1', '15551234567@s.whatsapp.net'),
    true,
  );
  assert.equal(
    store.quotedOutboundByJack('jack-outbound-1', '15550009999@s.whatsapp.net'),
    false,
  );
  assert.equal(store.get('jack-outbound-1').message.conversation, 'Jack answer');

  store.remember({
    key: {
      id: 'owner-typed-1',
      remoteJid: '15551234567@s.whatsapp.net',
      fromMe: true,
    },
    message: { conversation: 'Owner typed this' },
  });
  assert.equal(
    store.quotedOutboundByJack('owner-typed-1', '15551234567@s.whatsapp.net'),
    false,
  );

  // A later inbound record with a colliding Baileys ID overwrites the
  // provenance instead of inheriting the successful-send proof.
  store.remember({
    key: {
      id: 'jack-outbound-1',
      remoteJid: '15551234567@s.whatsapp.net',
      fromMe: false,
    },
    message: { conversation: 'Inbound collision' },
  });
  assert.equal(
    store.quotedOutboundByJack('jack-outbound-1', '15551234567@s.whatsapp.net'),
    false,
  );
  assert.equal(store.quotedOutboundByJack('missing-after-restart', '15551234567@s.whatsapp.net'), null);

  // The same bound applies to a send result whose Baileys response omitted
  // message content; poll payloads retain a synthetic quoteable message.
  const pollPayload = buildPollPayload({
    question: 'Proceed?', options: ['Yes', 'No'], selectableCount: 1,
  });
  assert.equal(
    rememberBridgeOutboundMessage(store, {
      key: { id: 'jack-poll-1', remoteJid: '123456789:8@lid', fromMe: true },
    }, { chatId: '123456789@lid', payload: pollPayload }),
    true,
  );
  assert.equal(store.quotedOutboundByJack('jack-poll-1', '123456789@lid'), true);
  assert.ok(store.get('jack-poll-1').message.pollCreationMessageV3);

  // Capacity eviction is the intentional restart/old-message fallback state.
  assert.equal(store.quotedOutboundByJack('owner-typed-1', '15551234567@s.whatsapp.net'), null);
  assert.equal(rememberBridgeOutboundMessage(store, null, { chatId: 'x' }), false);
  console.log('  ✓ messageStore proves only chat-bound bridge outbound IDs');
}

{
  const store = createBoundedMessageStore(4);
  const sent = {
    key: { id: 'successful-send', remoteJid: '120363001234567890@g.us', fromMe: true },
    message: { imageMessage: { caption: 'photo reply' } },
  };
  const tracked = [];
  assert.equal(
    await sendAndRememberBridgeOutbound({
      send: async () => sent,
      messageStore: store,
      chatId: '120363001234567890@g.us',
      payload: { image: Buffer.from('image'), caption: 'photo reply' },
      afterRemember: result => tracked.push(result.key.id),
    }),
    sent,
  );
  assert.deepEqual(tracked, ['successful-send']);
  assert.equal(store.quotedOutboundByJack('successful-send', '120363001234567890@g.us'), true);
  assert.equal(store.get('successful-send').message.imageMessage.caption, 'photo reply');

  await assert.rejects(
    sendAndRememberBridgeOutbound({
      send: async () => { throw new Error('send failed'); },
      messageStore: store,
      chatId: '120363001234567890@g.us',
      payload: { text: 'never sent' },
      afterRemember: () => tracked.push('should-not-run'),
    }),
    /send failed/,
  );
  assert.equal(store.quotedOutboundByJack('failed-send', '120363001234567890@g.us'), null);
  assert.deepEqual(tracked, ['successful-send']);
  console.log('  ✓ only successful text/media send results create ownership records');
}

// -- inbound quote/media/native metadata --------------------------------
{
  const ownershipStore = createBoundedMessageStore(4);
  rememberBridgeOutboundMessage(ownershipStore, {
    key: {
      id: 'outbound-1',
      remoteJid: '15551234567:47@s.whatsapp.net',
      fromMe: true,
    },
    message: { conversation: 'approve deploy?' },
  }, { chatId: '15551234567@s.whatsapp.net' });
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
    messageStore: ownershipStore,
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
  assert.equal(event.quotedOutboundByJack, true);
  assert.equal(event.body, 'approved');
  console.log('  ✓ inbound quoted metadata carries same-chat outbound ownership proof');
}

{
  const ownershipStore = createBoundedMessageStore(4);
  rememberBridgeOutboundMessage(ownershipStore, {
    key: {
      id: 'outbound-before-photo',
      remoteJid: '120363001234567890@g.us',
      fromMe: true,
    },
    message: { conversation: 'send the photo' },
  }, { chatId: '120363001234567890@g.us' });
  const event = await extractBridgeEvent({
    msg: {
      key: {
        id: 'incoming-photo-reply',
        remoteJid: '120363001234567890@g.us',
        participant: '15550001111@s.whatsapp.net',
        fromMe: false,
      },
      messageTimestamp: 124,
      message: {
        imageMessage: {
          caption: 'here it is',
          mimetype: 'image/jpeg',
          contextInfo: {
            stanzaId: 'outbound-before-photo',
            participant: '15559998888:7@s.whatsapp.net',
            remoteJid: '120363001234567890@g.us',
            quotedMessage: { conversation: 'send the photo' },
          },
        },
      },
    },
    chatId: '120363001234567890@g.us',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    botIds: ['15559998888@s.whatsapp.net'],
    messageStore: ownershipStore,
    downloadMedia: async () => Buffer.from('jpeg'),
    writeMediaFile: async () => '/tmp/reply.jpg',
  });

  assert.equal(event.hasMedia, true);
  assert.equal(event.mediaType, 'image');
  assert.equal(event.quotedOutboundByJack, true);
  assert.deepEqual(event.mediaUrls, ['/tmp/reply.jpg']);
  console.log('  ✓ native media replies retain same-chat outbound ownership proof');
}

{
  const ownershipStore = createBoundedMessageStore(4);
  rememberBridgeOutboundMessage(ownershipStore, {
    key: { id: 'known-jack-id', remoteJid: '15551110000@s.whatsapp.net', fromMe: true },
    message: { conversation: 'Jack in another chat' },
  }, { chatId: '15551110000@s.whatsapp.net' });
  ownershipStore.remember({
    key: { id: 'known-foreign-id', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
    message: { conversation: 'Not Jack' },
  });
  ownershipStore.remember({
    key: { id: 'known-foreign-poll', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
    message: { pollCreationMessage: { name: 'Not Jack poll' } },
  });
  rememberBridgeOutboundMessage(ownershipStore, {
    key: { id: 'known-jack-here', remoteJid: '15551234567@s.whatsapp.net', fromMe: true },
    message: { conversation: 'Jack in this chat' },
  }, { chatId: '15551234567@s.whatsapp.net' });

  async function quotedEvent({ stanzaId, remoteJid = '15551234567@s.whatsapp.net', fromMe = false }) {
    return extractBridgeEvent({
      msg: {
        key: {
          id: `incoming-${stanzaId}`,
          remoteJid: '15551234567@s.whatsapp.net',
          participant: '15550001111@s.whatsapp.net',
          fromMe,
        },
        pushName: 'Tester',
        messageTimestamp: 123,
        message: {
          extendedTextMessage: {
            text: 'reply body',
            contextInfo: {
              stanzaId,
              participant: '15559998888:9@s.whatsapp.net',
              remoteJid,
              quotedMessage: { conversation: 'untrusted quoted text' },
            },
          },
        },
      },
      chatId: '15551234567@s.whatsapp.net',
      senderId: '15550001111@s.whatsapp.net',
      senderNumber: '15550001111',
      botIds: ['15559998888@s.whatsapp.net'],
      messageStore: ownershipStore,
    });
  }

  const proved = await quotedEvent({ stanzaId: 'known-jack-here' });
  const crossChat = await quotedEvent({ stanzaId: 'known-jack-id' });
  const foreign = await quotedEvent({ stanzaId: 'known-foreign-id' });
  const foreignPoll = await quotedEvent({ stanzaId: 'known-foreign-poll' });
  const missing = await quotedEvent({ stanzaId: 'missing-after-restart' });
  assert.equal(proved.quotedOutboundByJack, true);
  assert.equal(crossChat.quotedOutboundByJack, false);
  assert.equal(foreign.quotedOutboundByJack, false);
  assert.equal(foreignPoll.quotedOutboundByJack, false);
  assert.equal(missing.quotedOutboundByJack, null);
  assert.equal((await quotedEvent({
    stanzaId: 'missing-after-restart', remoteJid: '15551110000@s.whatsapp.net',
  })).quotedOutboundByJack, false);
  assert.equal((await quotedEvent({ stanzaId: 'known-jack-id', fromMe: true })).quotedOutboundByJack, false);
  const responseQueue = [proved, foreign, missing];
  const serializedMessagesResponse = JSON.parse(JSON.stringify(
    drainMessageQueueForResponse(responseQueue),
  ));
  assert.deepEqual(
    serializedMessagesResponse.map(event => event.quotedOutboundByJack),
    [true, false, null],
  );
  assert.equal(responseQueue.length, 0);
  console.log('  ✓ foreign, cross-chat, restart-missing, and self-echo quote verdicts fail closed');
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
  assert.equal(event.quotedOutboundByJack, null);
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
