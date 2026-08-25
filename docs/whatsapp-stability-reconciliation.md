# WhatsApp stability reconciliation ledger

This ledger records how the pre-reconciliation production changes are handled
on `codex/whatsapp-stability`.  It is deliberately a mapping, not a second
implementation source: deploy only reviewed commits built on current upstream.

| Previous production change | Disposition | Destination |
| --- | --- | --- |
| `0dc9d89723` bridge outbound visibility/status guard | Keep generic behavior | Separate delivery/output review after its tests are replayed; do not copy status-specific rules. |
| `661aa03510`, `9e6e79f22a` internal failure and long-diagnostic suppression | Keep contract | Per-profile WhatsApp display configuration plus final send-boundary tests; no global stock-default change. |
| `cb526569cd` group context/audio privacy plus named workflow routing | Split | Preserve generic group context and private-audio invariants only; move named people/payment/tenant rules to profile skills or hooks. |
| `d1a2f59591` spool, album journal, reply-only delivery, reactions | Split | Evaluate generic exactly-once delivery units separately from Test AI and customer workflow behavior. |
| `5159766db5` group SOP binding | Replace | Use the existing `channel_skill_bindings` schema, now bridged for WhatsApp JIDs. |
| `4e3e384def`, `b6b73805a` direct group request follow-through | Keep generic invariant | Test the shared group session so a direct request cannot be lost while a turn is active; combine only with Goal Reliability review. |
| Dirty `cron/executions.py`, `cron/scheduler.py`, cron tests | Exclude | Goal Reliability branch. |
| Dirty `TEST_AI_QUEUE_ALBUM.md`, `test_ai_queue_album.*`, endpoint registration | Exclude | Separate Queue/Test AI work. |
| Dirty bridge session fingerprint/health handshake | Upstream replacement | Cherry-pick `5d09a4b843` (`#80096`) with author attribution, then review its profile-scoped IPC/token migration. |
| Dirty global mention bypass | Retire | Use `group_policy: allowlist` plus `free_response_chats`; unauthorized groups stop before model/session intake. |

## Deployment configuration contract

The stable profile, not the stock global defaults, owns the user-facing-only
policy.  It must configure `group_policy: allowlist`, the matching
`group_allow_from` and `free_response_chats` JIDs, `require_mention: true`,
`reply_prefix: ""`, and `group_sessions_per_user: false`.  Its
`display.platforms.whatsapp` block disables tool progress, interim assistant
messages, long-running notifications, busy details, streaming, and restart
notifications.  Native typing remains allowed.

No behavior settings belong in `.env`; credentials and bridge-generated
secrets remain profile-scoped.
