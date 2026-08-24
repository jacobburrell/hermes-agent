# Goal-reliability maintenance runbook

This branch carries the goal lifecycle work until the focused upstream PRs
land. Treat it as an integration branch, not an installed runtime checkout.
The reliability invariants are more important than retaining any individual
patch unchanged:

- only `done` means the goal was achieved;
- a `waiting` goal has a persisted deadline/process wake and resumes without a
  new user message;
- `/goal --interval` is the canonical scheduler. New `/loop` commands are a
  compatibility spelling that creates that same scheduled goal; an existing
  persisted legacy loop remains controllable until its owner clears it;
- wake claims are lease-bounded and may be recovered after restart, but a live
  session gets only one injected wake at a time;
- owner approval is bound to the pending approval id and owner identity;
- scheduler changes preserve prompt caching and role alternation.

## Patch ledger

| Area | Local branch responsibility | Upstream work to compare on every update |
| --- | --- | --- |
| Truthful terminal verdicts | `hermes_cli/goals.py` | #91728 and any successor that changes judge semantics |
| WAIT wake-up | Goal durable claim + CLI/gateway/TUI drivers | #81432 and process-notification changes |
| Evidence / owner approval | bounded receipts and capability-bound approval | #81801 patterns only; do not import unrelated runtime-auth changes |
| UI / surface parity | CLI, gateway, TUI/Desktop status and controls | #93243 / UI goal work |
| Judge resilience | local retry-before-new-turn behavior | #93521 and provider-failure fixes |

Before retaining a local hunk, classify its upstream equivalent as **merged
and drop**, **adapt**, or **retain**. Never use a blanket `ours`/`theirs`
conflict choice: it can silently restore the false-completion bug or remove a
newer gateway safety fix.

## William: safe upstream update procedure

Do **not** run bare `hermes update` while the installed checkout is on an
unmerged goal-reliability branch. That updater follows the configured upstream
and can either refuse a dirty/custom branch or move it in a way that makes it
hard to audit what is local.

1. Record the known-good runtime **without changing its branch**:

   ```bash
   hermes --version
   hermes update --plan
   hermes update --check
   hermes gateway status
   hermes doctor
   ```

   This is the only safe point to ask what `main` would do. Do **not** deploy
   the feature branch yet, and do not let an updater implicitly merge main
   into it. If the updater reports a stale gateway, restart only after
   checking its supervisor/ownership; do not kill processes by hand.

2. Refresh `main` in a disposable **audit checkout**, never the runtime
   working tree or the integration branch:

   ```bash
   git fetch upstream main --prune
   git fetch origin goal-reliability --prune
   git worktree add ../Hermes-main-audit upstream/main
   cd ../Hermes-main-audit
   git rev-parse HEAD
   ```

   If `origin` is not configured, add William's writable fork first. Keep
   `upstream` pointed at `NousResearch/hermes-agent`. This checkout is the
   answer to “what changed on main?”; do not guess from PR titles.

3. Test clean upstream and classify each local concern before carrying the
   branch forward:

   ```bash
   scripts/run_tests.sh tests/hermes_cli/test_goals.py tests/gateway/test_goal_verdict_send.py tests/tui_gateway/test_goal_command.py -q
   ```

   Read the current implementation and the referenced PR heads. For each
   ledger row, record whether main already fixed it, changed its interface, or
   still needs this branch's behavior. Only then return to the integration
   checkout. Delete the audit worktree when done.

4. Put the integration branch back onto the newly audited `main` deliberately:

   ```bash
   cd ../Hermes
   git switch goal-reliability
   git log --oneline --left-right upstream/main...goal-reliability
   git rebase upstream/main
   scripts/run_tests.sh tests/hermes_cli/test_goals.py tests/hermes_cli/test_loops.py tests/gateway/test_goal_verdict_send.py tests/gateway/test_loop_command.py tests/tui_gateway/test_goal_command.py tests/tui_gateway/test_loop_command.py -q
   git diff upstream/main...HEAD -- hermes_cli/goals.py hermes_cli/loops.py cli.py gateway tui_gateway
   git push --force-with-lease origin goal-reliability
   ```

   Do not use `git reset --hard`, do not force-push without `--force-with-lease`,
   and do not overwrite an upstream improvement just to make a conflict easy.

5. Only after the integration branch is green, deploy that exact reviewed
   branch with `hermes update --backup --branch goal-reliability`, then verify
   the checkout SHA, `hermes doctor`, gateway version/status, and one
   `/goal --interval` smoke test. Once every retained ledger item is merged
   upstream, update the audit checkout to confirm it, then switch the runtime
   back to `main` and run the same backup/update verification.

## Required regression set

Use `scripts/run_tests.sh`, not raw `pytest`. At a minimum run the focused
goal, loop, gateway, and TUI files named in step 4. A release candidate also
needs these behavior checks:

1. A wait-on-deadline and wait-on-process goal wakes by itself after the
   dependency clears.
2. A blocked, unachievable, or needs-user verdict is never rendered as
   achieved.
3. A `done` candidate with a live background dependency waits instead.
4. An owner-required completion cannot be approved by unrelated text or a
   different identity.
5. A restart during a claimed wake recovers after the lease rather than
   silently stalling or double-injecting.
