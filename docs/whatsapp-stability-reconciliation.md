# WhatsApp stability reconciliation ledger

This ledger records how the pre-reconciliation production changes are handled
on `codex/whatsapp-stability`.  It is deliberately a mapping, not a second
implementation source: deploy only reviewed commits built on current upstream.

| Previous production change | Disposition | Destination |
| --- | --- | --- |
| `0dc9d89723` bridge outbound visibility/status guard | Keep generic behavior | `d962ab7ef7`, `f99780ab15`: profile-scoped strict policy at both adapter and authenticated Node send boundaries. Stock output remains unchanged. |
| `661aa03510`, `9e6e79f22a` internal failure and long-diagnostic suppression | Keep contract | `e0ffac146f`, `0c47c90041`, `89a5d4e561`: typed unknown-fails-closed is primary; structural diagnostic/banner detection is defense in depth. |
| `cb526569cd` group context/audio privacy plus named workflow routing | Split | `ec7bdd8f2d` keeps private group-audio and generic context invariants; named people/payment/tenant behavior remains profile skill/config, outside core. |
| `d1a2f59591` spool, album journal, reply-only delivery, reactions | Split | `49b4e51d4a` plus the receipt series through `c9badfac24` provide durable lease/ACK/release/dead-letter handling. Casa B/reaction/customer workflow behavior is excluded. |
| `5159766db5` group SOP binding | Replace | `4d7f524a5b`, `bd62957f7b` use the existing `channel_skill_bindings` schema for WhatsApp JIDs. |
| `4e3e384def`, `b6b73805a` direct group request follow-through | Keep generic invariant | `f71de36f27`, `91fe1a581a` partition ambient observation from addressed operational work while retaining the group chat ID for evidence. Goal-lifecycle changes remain on the Goal Reliability branch. |
| Dirty `cron/executions.py`, `cron/scheduler.py`, cron tests | Exclude | Goal Reliability branch. |
| Dirty `TEST_AI_QUEUE_ALBUM.md`, `test_ai_queue_album.*`, endpoint registration | Exclude | Separate Queue/Test AI work. |
| Dirty bridge session fingerprint/health handshake | Upstream replacement | `ce380926a7` imports `#82022`'s profile-scoped authenticated UDS/token control channel with contributor attribution. |
| Dirty global mention bypass | Retire | Use `group_policy: allowlist` plus `free_response_chats`; unauthorized groups stop before model/session intake. |
| `#84926` group observation/shared context | Not imported | Its broad observe/default-open behavior conflicts with the explicit allowlist contract. The narrow shared-context requirement is covered by `#84925` plus the session-lane commits. |
| `#84925` session ownership/isolation | Applied | All six commits (`4cebd9a070` through `2a24a67a23`) are imported with contributor attribution and preserve per-profile, per-platform isolation. |
| `#87885` authenticated no-user-id routing | Applied | `4e6930c96e` preserves authenticated allowlisted group routing when the bridge omits a user ID. |
| `#85081` JID/LID identity routing | Applied | `c3a946789e`, `23b5caa810` normalize and test JID/LID/number profile routes. |
| `#89322` device-suffixed mentions | Applied plus sibling fix | `3884256a52`, `c2ba9841c4` normalize linked-device suffixes consistently for mentions and replies. |
| `#87382` poll supervision | Applied | `2703d39a85` makes poll-task death fatal/recoverable instead of silently stopping inbound delivery. |
| `#85196` reconnect backoff | Applied | `16278dfd29`, `babf6920ff` add bounded monotonic backoff and jitter. |
| `#84180` atomic auth state | Applied | `54f56a12d8`, `ad5994b6e5` preserve the paired identity across partial/full-disk write failure. |
| `#89321` non-FULL history sync | Applied with boundary test | `56318b160c`, `bcc80284a4` restore group decryption metadata while FULL history remains disabled; Hermes has no `messaging-history.set` consumer, so downloaded history does not enter inbound delivery. |
| `#93377` restart port wait | Superseded | `#82022` replaces the managed TCP listener with authenticated UDS/pipe IPC; no port wait remains. |

## Deployment configuration contract

The stable profile, not the stock global defaults, owns the user-facing-only
policy. It must configure `outbound_policy: user_visible_only`,
`group_policy: allowlist`, the matching `group_allow_from` and
`free_response_chats` JIDs, `require_mention: true`, `reply_prefix: ""`,
`group_sessions_per_user: false`, and `group_session_lanes: true`. Its
`display.platforms.whatsapp` block disables tool progress, interim assistant
messages, long-running notifications, busy details, streaming, and restart
notifications.  Native typing remains allowed.

No behavior settings belong in `.env`; credentials and bridge-generated
secrets remain profile-scoped.

The strict boundary covers the deployed personal-account Baileys bridge
(`plugins/platforms/whatsapp`). WhatsApp Cloud retains its stock behavior and
is deliberately out of scope for this profile contract.

## Error email disposition

The existing standalone email sender can be selected by a profile-level hook
only after that profile has SMTP credentials in its secret `.env`. The audited
profiles do not currently have the required email credentials, so this branch
does not create a second core notification path or attempt delivery. A later
profile hook may route redacted local error notifications from
`jack@revoluciones.mx` to `error@jacob.mx`, with mocked tests and credentials
supplied out of band. Until then, errors remain in profile-scoped local logs;
the strict WhatsApp boundary drops them rather than exposing them to chats.
