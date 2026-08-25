# WhatsApp stability reconciliation ledger

This ledger records how the pre-reconciliation production changes are handled
on `codex/whatsapp-stability`.  It is deliberately a mapping, not a second
implementation source: deploy only reviewed commits built on current upstream.

| Previous production change | Disposition | Destination |
| --- | --- | --- |
| `0dc9d89723` bridge outbound visibility/status guard | Keep generic behavior | `46458ac268` (amended before review): profile-scoped strict policy at both adapter and authenticated Node send boundaries; no status-string one-offs. |
| `661aa03510`, `9e6e79f22a` internal failure and long-diagnostic suppression | Keep contract | Typed unknown-fails-closed is primary; structural diagnostic/banner detection is defense in depth. Stock defaults remain unchanged. |
| `cb526569cd` group context/audio privacy plus named workflow routing | Split | Preserve generic group context and private-audio invariants only; move named people/payment/tenant rules to profile skills or hooks. |
| `d1a2f59591` spool, album journal, reply-only delivery, reactions | Split / pending | Reconcile generic inbound acknowledge-after-durable-spool and generic album journal with authenticated IPC; exclude Casa B/reaction/SOP/customer workflow behavior. |
| `5159766db5` group SOP binding | Replace | Use the existing `channel_skill_bindings` schema, now bridged for WhatsApp JIDs. |
| `4e3e384def`, `b6b73805a` direct group request follow-through | Keep generic invariant | Test the shared group session so a direct request cannot be lost while a turn is active; combine only with Goal Reliability review. |
| Dirty `cron/executions.py`, `cron/scheduler.py`, cron tests | Exclude | Goal Reliability branch. |
| Dirty `TEST_AI_QUEUE_ALBUM.md`, `test_ai_queue_album.*`, endpoint registration | Exclude | Separate Queue/Test AI work. |
| Dirty bridge session fingerprint/health handshake | Upstream replacement | Cherry-pick `5d09a4b843` (`#80096`) with author attribution, then review its profile-scoped IPC/token migration. |
| Dirty global mention bypass | Retire | Use `group_policy: allowlist` plus `free_response_chats`; unauthorized groups stop before model/session intake. |
| `#84926` group observation/shared context | Partial / pending | Keep only the every-allowlisted-group shared-session invariant if current context handling lacks it; do not import observe-only/default-open behavior. |
| `#87885` authenticated no-user-id routing | Applied | `da474c5685` cherry-picked with `-x`. |
| `#85081` JID/LID identity routing | Pending | Test tip applied as `2e403d6e38`; reconcile its implementation parent before merge. |
| `#89322` device-suffixed mentions | Pending | Inspect against current mention normalization before cherry-pick. |
| `#87382`, `#85196`, `#84180`, `#89321` | Pending | Reconcile supervision, reconnect, atomic credentials, and history sync in small attributed commits. |
| `#93377` restart port wait | Superseded | `#82022` replaces the managed TCP listener with authenticated UDS/pipe IPC; no port wait remains. |

## Deployment configuration contract

The stable profile, not the stock global defaults, owns the user-facing-only
policy.  It must configure `outbound_policy: user_visible_only`, `group_policy: allowlist`, the matching
`group_allow_from` and `free_response_chats` JIDs, `require_mention: true`,
`reply_prefix: ""`, and `group_sessions_per_user: false`.  Its
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
does not create mail plumbing or attempt delivery. A later profile hook may
route redacted local error notifications from `jack@revoluciones.mx` to
`error@jacob.mx`, with mocked tests and credentials supplied out of band.
