"""Persistent session goals — the Ralph loop for Hermes.

A goal is a free-form user objective that stays active across turns. After
each turn completes, a small judge call asks an auxiliary model "is this
goal satisfied by the assistant's last response?". If not, Hermes feeds a
continuation prompt back into the same session and keeps working until the
goal is done, turn budget is exhausted, the user pauses/clears it, or the
user sends a new message (which takes priority and pauses the goal loop).

State is persisted in SessionDB's ``state_meta`` table keyed by
``goal:<session_id>`` so ``/resume`` picks it up.

Design notes / invariants:

- The continuation prompt is just a normal user message appended to the
  session via ``run_conversation``. No system-prompt mutation, no toolset
  swap — prompt caching stays intact.
- Judge failures are fail-OPEN: ``continue``. A broken judge must not wedge
  progress; the turn budget is the backstop.
- When a real user message arrives mid-loop it preempts the continuation
  prompt and also pauses the goal loop for that turn (we still re-judge
  after, so if the user's message happens to complete the goal the judge
  will say ``done``).
- This module has zero hard dependency on ``cli.HermesCLI`` or the gateway
  runner — both wire the same ``GoalManager`` in.

Nothing in this module touches the agent's system prompt or toolset.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
# Judge output budget. The freeform judge returns a one-line JSON verdict, but
# reasoning models (deepseek-v4, qwq, etc.) burn tokens on hidden reasoning
# before emitting the visible JSON — and the first /goal turn's prompt is
# larger than later turns, which pushes total reply length past tight caps.
# 200 tokens (the original default) reliably truncated the JSON on reasoning
# models, leaving '{"done": true, "reason": "The agent successfully' and
# triggering the auto-pause. 4096 covers reasoning + verdict on every model
# we've live-tested; override via auxiliary.goal_judge.max_tokens for
# specifically constrained setups.
DEFAULT_JUDGE_MAX_TOKENS = 4096
# Cap how much of the last response + recent messages we send to the judge.
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
# After this many consecutive judge *parse* failures (empty output / non-JSON),
# the loop auto-pauses and points the user at the goal_judge config. API /
# transport errors do NOT count toward this — those are transient. This guards
# against small models (e.g. deepseek-v4-flash) that cannot follow the strict
# JSON reply contract; without it the loop runs until the turn budget is
# exhausted with every reply shaped like `judge returned empty response` or
# `judge reply was not JSON`.
DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES = 3
# Transport failures (API auth errors 401, timeouts, DNS, etc.) are also
# tracked and auto-pause the loop after this many consecutive failures.
# A broken/invalid API key returns 401 every call — the loop must not
# run until the turn budget, wasting every turn on an unreachable judge.
DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES = 5
# A judge-led goal must never spend an agent turn merely because its control
# plane had a transient failure.  Three short, in-call attempts make the
# controller resilient without turning a provider outage into a run-away
# continuation loop.
DEFAULT_JUDGE_ATTEMPTS_PER_TURN = 3
DEFAULT_JUDGE_RETRY_BACKOFF_SECONDS = (0.2, 0.5)
DEFAULT_CONTROL_PLANE_RETRY_SECONDS = 30
DEFAULT_DUPLICATE_FAILURE_LIMIT = 2
DEFAULT_STALL_TURNS_BEFORE_REPLAN = 3
# A proposed human dependency is deliberately expensive: before parking a
# judge-led goal, the controller must have explored at least this many
# materially different *safe* routes.  This bound is on strategy diversity,
# never on productive agent turns.
MIN_RECOVERY_STRATEGY_FAMILIES = 3
MAX_PROGRESS_LEDGER_ENTRIES = 48

# A goal wake is a durable, lease-protected claim.  A small lease lets a
# restarted CLI/gateway recover a wake whose process died between claiming it
# and injecting the next turn, without allowing healthy long-running turns to
# be duplicated.  The surface still owns the live "is this session busy?"
# check; this state-level lease is the cross-process recovery backstop.
DEFAULT_WAKE_LEASE_SECONDS = 5 * 60
DEFAULT_WAIT_POLL_SECONDS = 5
MAX_EVIDENCE_RECEIPTS = 8

# Quality gates: deterministic shell commands that must pass before the goal
# judge may declare the goal done. Defaults mirror the bounded-autonomy
# pattern (per-gate retry limit + timeout, bounded output fed back to the
# agent). A failed gate short-circuits the judge — its output IS the
# continuation prompt, so the agent works on concrete evidence instead of a
# vibe check.
DEFAULT_GATE_TIMEOUT_SECONDS = 300
DEFAULT_GATE_MAX_RETRIES = 3
# Bounded tail of a failed gate's combined stdout/stderr fed back to the agent.
_GATE_OUTPUT_TAIL_CHARS = 3000


CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
)

# Judge-led goals deliberately use a different continuation contract from the
# legacy Ralph loop.  The agent is asked to change strategy on a stall rather
# than to merely report a blocker and exit.  It remains an ordinary user-role
# message, so it does not mutate the cached system prompt or tool schema.
JUDGE_LED_CONTINUATION_TEMPLATE = (
    "[Continuing toward your standing goal under the progress controller]\n"
    "Goal: {goal}\n\n"
    "Controller assessment: {reason}\n"
    "Required next-step constraint: {constraint}\n\n"
    "Continue with a concrete, authorized step. Keep the original outcome, "
    "recipients, permissions, and safety constraints unchanged. Do not repeat "
    "a failed action with the same hypothesis or inputs. If a tool or interface "
    "fails, verify the failure and try another safe route or diagnosis. Do not "
    "declare the goal complete without concrete evidence."
)

JUDGE_LED_REPLAN_TEMPLATE = (
    "[Recovery plan required for your standing goal]\n"
    "Goal: {goal}\n\n"
    "The previous approach stalled: {reason}\n"
    "Choose a materially different safe strategy family: {strategies}\n\n"
    "Before asking the user to intervene, verify the blocker, inspect relevant "
    "state/logs/configuration, try another authorized interface or evidence "
    "source where available, repair missing local capability when authorized, "
    "and continue independent work. Do not bypass authentication, approvals, "
    "CAPTCHAs, policy denials, or recipient/scope boundaries."
)

# Used when the goal carries a structured completion contract. The contract
# block tells the agent exactly what "done" means, how to prove it, what not
# to break, what's in scope, and when to stop and ask — so it targets the
# verification surface instead of declaring victory loosely.
CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Completion contract:\n"
    "{contract_block}\n\n"
    "Continue working toward the outcome above. Take the next concrete step. "
    "Stay within the stated boundaries and do not violate the constraints. "
    "Before claiming the goal is done, satisfy the Verification criterion and "
    "show the concrete evidence (command output, file contents, test result). "
    "If you hit the stated stop condition or are otherwise blocked and need "
    "user input, say so clearly and stop."
)

# Used when the user has added one or more /subgoal criteria. Surfaced
# to the agent verbatim so it sees what to target on the next turn,
# and surfaced to the judge so the verdict considers them too.
CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Additional criteria the user added mid-loop:\n"
    "{subgoals_block}\n\n"
    "Continue working toward the goal AND all additional criteria. Take "
    "the next concrete step. If you believe the goal and every "
    "additional criterion are complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly "
    "and stop."
)


# Fed back when a quality gate fails: the gate's bounded output is the
# evidence the agent must repair against. Deterministic — no judge involved.
CONTINUATION_PROMPT_GATE_FAILED_TEMPLATE = (
    "[Continuing toward your standing goal — a quality gate failed]\n"
    "Goal: {goal}\n\n"
    "The quality gate command below must pass before this goal can be "
    "declared done, and it just failed (attempt {attempt}/{max_retries}):\n"
    "  $ {command}\n"
    "Exit code: {exit_code}\n"
    "Output (tail):\n"
    "```\n"
    "{output}\n"
    "```\n\n"
    "Fix the underlying problem so this gate passes, then re-run it to "
    "confirm. Do not declare the goal complete while any gate fails. If the "
    "gate itself is wrong or cannot pass, say so clearly and stop."
)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text, the agent's "
    "most recent response, and — when present — a list of background "
    "processes the agent has running. Decide one of six verdicts.\n\n"
    "DONE — the goal is fully satisfied:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced, AND\n"
    "- No stated acceptance criterion or listed background dependency remains.\n\n"
    "NEEDS_USER — a specific action or decision from the goal owner is needed. "
    "This is NOT completion.\n"
    "BLOCKED — an external non-user dependency is currently preventing progress. "
    "This is NOT completion.\n"
    "UNACHIEVABLE — the requested outcome cannot be achieved as stated. This is "
    "NOT completion.\n\n"
    "WAIT — the goal is NOT done, but the next step is to wait for async "
    "work to finish rather than act again. Choose this ONLY when the agent's "
    "progress is genuinely gated on something running on its own:\n"
    "- A background process listed below is still running AND the response "
    "shows the agent is waiting on its result (e.g. a CI poller, build, "
    "test run, deploy). If the process has a session id, return it in "
    "``wait_on_session`` — that releases when the process exits OR its "
    "watch_patterns trigger fires (use this for a long-lived watcher that "
    "signals mid-run and may never exit). Otherwise return its pid in "
    "``wait_on_pid`` (releases on exit only).\n"
    "- The agent says it is rate-limited / backing off / must wait a fixed "
    "period — return seconds in ``wait_for_seconds``.\n"
    "Picking WAIT parks the loop without burning a turn; it resumes "
    "automatically when the pid exits or the time elapses. Do NOT pick WAIT "
    "just because work remains — only when re-poking now would be pure "
    "busy-work because the agent can't progress until the async thing "
    "finishes.\n\n"
    "CONTINUE — not done and there is a concrete next step right now. This is "
    "the default when in doubt.\n\n"
    "Reply ONLY with a single JSON object on one line. Shapes:\n"
    '{"verdict": "done", "reason": "<one sentence>"}\n'
    '{"verdict": "continue", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_session": "<id>", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_pid": <int>, "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_for_seconds": <int>, "reason": "<one sentence>"}\n'
    '{"verdict": "needs_user", "reason": "<specific requested action>"}\n'
    '{"verdict": "blocked", "reason": "<specific external blocker>"}\n'
    '{"verdict": "unachievable", "reason": "<why the outcome cannot be achieved>"}\n'
    "The legacy shape {\"done\": <true|false>, \"reason\": \"...\"} is still "
    "accepted (true=done, false=continue)."
)

# This contract is intentionally separate from the legacy three-way judge.
# Existing profiles retain their small/cheap judge and fixed turn budget;
# profiles that opt into ``termination: judge`` receive a controller verdict
# with enough state to distinguish productive work from repetition, waiting,
# an irreducible dependency, and a policy boundary.
JUDGE_LED_SYSTEM_PROMPT = (
    "You are the progress controller for an autonomous goal. Judge the goal "
    "against the cumulative evidence ledger, not the agent's latest prose. "
    "Agent claims alone do not prove success or impossibility. Productive work "
    "may continue, but repeating a failed action or expanding authority may not.\n\n"
    "Use achieved only when the completion contract is satisfied by concrete "
    "evidence. Use continue when progress was made and a concrete step remains. "
    "Use replan when a strategy stalled or needs a materially different path. "
    "Use wait only for an actual asynchronous process, cooldown, or scheduled "
    "event. Use needs_input only for a specific irreducible human-controlled "
    "dependency after safe alternatives were exhausted. Use not_achievable only "
    "with deterministic evidence that the requested outcome is impossible within "
    "authorized scope. Use policy_stop when continuing would violate a safety, "
    "privacy, authority, financial, legal, or task constraint. A denied action "
    "is never a puzzle to circumvent.\n\n"
    "Reply only with one JSON object using exactly these fields: "
    "{\"verdict\":\"achieved|continue|replan|wait|needs_input|not_achievable|policy_stop\","
    "\"progress\":\"advanced|stalled|regressed\",\"reason\":\"specific evidence-based explanation\","
    "\"evidence_refs\":[\"ledger reference\"],\"blocker_class\":\"transient|environment|capability|dependency|ambiguity|authorization|policy|impossible\","
    "\"recoverable\":true,\"untried_strategy_families\":[\"...\"],"
    "\"next_strategy_constraint\":\"what must be different next time\","
    "\"wait_directive\":null}. "
    "For wait_directive use one of {\"session_id\":\"...\"}, {\"pid\":123}, "
    "or {\"seconds\":30}."
)

GOAL_RECOVERY_SYSTEM_PROMPT = (
    "You are a recovery coach for an autonomous goal. Produce safe, materially "
    "different strategy families after a stall or proposed blocker. Preserve the "
    "user's outcome, authority, recipients, privacy, and safety constraints. "
    "Never suggest bypassing authentication, approvals, CAPTCHAs, policy denials, "
    "or external-action safeguards. Prefer verifying state, inspecting logs/source/config, "
    "using another authorized interface/layer, an alternate evidence source, repairing "
    "a local capability when authorized, independent diagnosis, and parallel independent "
    "work. Return at least three materially different viable routes unless deterministic "
    "evidence proves fewer are safe or applicable; explain that evidence in "
    "irreducible_dependency. Reply only as JSON: {\"strategies\":[{\"family\":\"...\",\"next_step\":\"...\","
    "\"why_safe\":\"...\"}],\"irreducible_dependency\":\"\"}."
)

GOAL_TERMINAL_VERIFIER_SYSTEM_PROMPT = (
    "You independently verify a proposed terminal decision for an autonomous goal. "
    "Use the completion contract and cumulative evidence ledger. Reject success or "
    "impossibility based solely on an agent claim. Reject needs_input while an applicable "
    "safe recovery path remains untried. Reject needs_input or not_achievable when fewer "
    "than three materially different recovery families were tried unless the ledger has "
    "independent deterministic evidence that fewer routes are safe or applicable. Never approve a path that bypasses authority, "
    "authentication, safety controls, privacy, or policy. Reply only as JSON: "
    "{\"accept\":true,\"reason\":\"evidence-based explanation\","
    "\"untried_strategy_families\":[\"...\"]}."
)

# Fallback recovery routes are deliberately read-only and broadly applicable.
# They prevent a weak recovery model from turning "I am blocked" into an
# immediate request for a human when it has not even checked whether the
# premise, local diagnostics, or an independent diagnosis changes the answer.
# More task-specific routes (another authorized interface, capability repair,
# and independent work) still come from the recovery coach.
_BASELINE_RECOVERY_STRATEGIES: Tuple[Dict[str, str], ...] = (
    {
        "family": "verify the blocker premise",
        "next_step": (
            "Verify the claimed blocker against current state rather than assuming it; "
            "record the concrete observed result and do not repeat the same action unchanged."
        ),
        "why_safe": "read-only verification does not expand authority or create side effects",
    },
    {
        "family": "inspect diagnostics and configuration",
        "next_step": (
            "Inspect relevant logs, source, configuration, history, and documented behavior "
            "to locate the failing layer and identify an authorized repair or workaround."
        ),
        "why_safe": "inspection is read-only and preserves the requested outcome",
    },
    {
        "family": "independent diagnosis or narrower reproduction",
        "next_step": (
            "Obtain an independent diagnosis or create a narrower safe reproduction/evidence "
            "source before concluding that the dependency is irreducible."
        ),
        "why_safe": "diagnosis does not bypass controls or broaden external scope",
    },
)


# Rendered into the judge prompt when the agent has background processes
# running. Gives the judge the context it needs to decide WAIT vs CONTINUE
# (and which pid to wait on) without it having to probe anything itself.
JUDGE_BACKGROUND_BLOCK_TEMPLATE = (
    "Background processes the agent currently has running (it may be waiting "
    "on one of these):\n{background_lines}\n\n"
)


JUDGE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Is the goal satisfied — done, continue, wait, needs_user, blocked, or unachievable?"
)

# Used when the user has added /subgoal criteria. The judge must
# evaluate ALL of them being met, not just the original goal.
JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Additional criteria the user added mid-loop (all must also be "
    "satisfied for the goal to be DONE):\n{subgoals_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Decision: For each numbered criterion above, find concrete "
    "evidence in the agent's response that the criterion is "
    "satisfied. Do not accept generic phrases like 'all requirements "
    "met' or 'implying it was done' — require specific evidence (a "
    "file contents excerpt, an output line, a command result). If "
    "ANY criterion lacks specific evidence in the response, the goal "
    "is NOT done — return CONTINUE (or WAIT if blocked on a listed "
    "background process).\n\n"
    "Is the goal AND every additional criterion satisfied?"
)


# Used when the goal carries a structured completion contract. The judge
# decides DONE strictly against the Verification criterion and refuses to
# accept completion when a constraint was violated.
JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Completion contract (the authoritative definition of done):\n"
    "{contract_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Decision rules:\n"
    "- The goal is DONE only when the Verification criterion is satisfied AND "
    "the response shows concrete evidence of it (a command result, file "
    "contents excerpt, test/benchmark output) — not a claim like 'done' or "
    "'all tests pass' without evidence.\n"
    "- If any stated Constraint was violated, the goal is NOT done — CONTINUE.\n"
    "- If the response shows the agent is waiting on a listed background "
    "process to satisfy the Verification criterion (e.g. CI is the "
    "verification and it's still running), return WAIT on that process "
    "instead of re-poking — re-poking now would be pure busy-work.\n"
    "- If a specific owner action or decision is required, return NEEDS_USER; "
    "if an external dependency is preventing work, return BLOCKED; if the "
    "outcome cannot be achieved as stated, return UNACHIEVABLE. None of these "
    "is success.\n"
    "- Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Is the goal satisfied per its completion contract — done, continue, wait, "
    "needs_user, blocked, or unachievable?"
)


# System prompt for /goal draft — turns a plain-language objective into a
# structured completion contract the user can review before activating.
# Adapted from Codex's "let Codex draft the goal" guidance.
DRAFT_CONTRACT_SYSTEM_PROMPT = (
    "You turn a user's plain-language objective into a structured completion "
    "contract for an autonomous coding agent. The contract has five fields:\n"
    "- outcome: the single end state that must be true when done\n"
    "- verification: the specific test / command / artifact that PROVES the "
    "outcome (must be concrete and checkable)\n"
    "- constraints: what must NOT change or regress\n"
    "- boundaries: which files, dirs, tools, or systems are in scope\n"
    "- stop_when: the condition under which the agent should stop and ask "
    "for human input instead of pushing on\n\n"
    "Infer sensible, specific values from the objective and any project "
    "context implied by it. Prefer concrete verification (a named test "
    "command, a build, a benchmark) over vague phrases. Keep each field to "
    "one or two sentences. If a field genuinely cannot be inferred, use an "
    "empty string for it.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"outcome": "...", "verification": "...", "constraints": "...", '
    '"boundaries": "...", "stop_when": "..."}'
)


# ──────────────────────────────────────────────────────────────────────
# Completion contract
# ──────────────────────────────────────────────────────────────────────

# The five contract fields, in display order. Adapted from OpenAI Codex's
# "strong goal" guidance: a durable objective works best when it names what
# "done" means, how to prove it, what must not regress, what tools/paths are
# in bounds, and when to stop and ask. A bare free-form goal (no contract)
# stays fully supported — every field defaults empty and is simply omitted
# from the prompts when unset.
_CONTRACT_FIELDS = ("outcome", "verification", "constraints", "boundaries", "stop_when")

# Human labels for rendering and for the inline `field: value` parser.
_CONTRACT_LABELS = {
    "outcome": "Outcome",
    "verification": "Verification",
    "constraints": "Constraints",
    "boundaries": "Boundaries",
    "stop_when": "Stop when blocked",
}

# Inline-input aliases the user may type before a value, mapped to the
# canonical field name. e.g. `verify: tests pass` or `done when: ...`.
_CONTRACT_ALIASES = {
    "outcome": "outcome",
    "goal": "outcome",
    "done": "outcome",
    "done when": "outcome",
    "verification": "verification",
    "verify": "verification",
    "verified by": "verification",
    "evidence": "verification",
    "proof": "verification",
    "constraints": "constraints",
    "constraint": "constraints",
    "preserve": "constraints",
    "must not": "constraints",
    "do not change": "constraints",
    "boundaries": "boundaries",
    "boundary": "boundaries",
    "scope": "boundaries",
    "allowed": "boundaries",
    "files": "boundaries",
    "stop when": "stop_when",
    "stop_when": "stop_when",
    "blocked": "stop_when",
    "stop if blocked": "stop_when",
    "give up when": "stop_when",
}


@dataclass
class GoalContract:
    """Optional structured completion contract for a goal.

    Each field is free-form prose the user (or :func:`draft_contract`)
    supplies. Empty fields are omitted everywhere — a goal with no contract
    behaves exactly like the original free-form goal. The contract is woven
    into both the continuation prompt (so the agent targets the verification
    surface and respects constraints) and the judge prompt (so "done" is
    decided against evidence, not vibes).
    """

    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""

    def is_empty(self) -> bool:
        return not any(getattr(self, f).strip() for f in _CONTRACT_FIELDS)

    def to_dict(self) -> Dict[str, str]:
        return {f: getattr(self, f) for f in _CONTRACT_FIELDS}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalContract":
        if not isinstance(data, dict):
            return cls()
        return cls(**{f: str(data.get(f) or "").strip() for f in _CONTRACT_FIELDS})

    def render_block(self) -> str:
        """Render non-empty contract fields as a labelled block. Empty
        contract → empty string (callers skip the section entirely)."""
        lines = []
        for f in _CONTRACT_FIELDS:
            val = getattr(self, f).strip()
            if val:
                lines.append(f"- {_CONTRACT_LABELS[f]}: {val}")
        return "\n".join(lines)


def parse_contract(text: str) -> Tuple[str, GoalContract]:
    """Split user-typed goal text into a headline + structured contract.

    Supports inline ``field: value`` lines so power users can type a full
    contract in one shot, e.g.::

        Migrate auth to JWT
        verify: the auth test suite passes
        constraints: keep the public /login response shape unchanged
        boundaries: only touch services/auth and its tests
        stop when: a schema change needs product sign-off

    The first non-field line(s) become the goal headline; recognized
    ``field:`` lines populate the contract. Lines for the same field are
    joined. Unrecognized prefixes stay part of the headline, so a plain
    free-form goal with an incidental colon (``Fix bug: the parser``)
    is NOT mangled — only lines whose prefix matches a known alias are
    pulled out. Returns ``(headline, contract)``.
    """
    if not text:
        return "", GoalContract()

    headline_parts: List[str] = []
    fields: Dict[str, List[str]] = {f: [] for f in _CONTRACT_FIELDS}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        if ":" in line:
            prefix, _, value = line.partition(":")
            key = _CONTRACT_ALIASES.get(prefix.strip().lower())
            if key is not None and value.strip():
                fields[key].append(value.strip())
                matched = True
        if not matched:
            headline_parts.append(line)

    headline = " ".join(headline_parts).strip()
    contract = GoalContract(
        **{f: " ".join(v).strip() for f, v in fields.items()}
    )
    # If a headline was given but no explicit `outcome:` field, the headline
    # IS the outcome — don't duplicate it into the contract block (the goal
    # text already carries it), so leave outcome empty in that case.
    return headline, contract


# ──────────────────────────────────────────────────────────────────────
# Quality gates
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GoalGate:
    """A deterministic shell command that must pass before a goal can be done.

    Gates run at turn boundary BEFORE the LLM judge. A failing gate
    short-circuits judging entirely: its bounded output becomes the
    continuation prompt, so the agent iterates against concrete evidence.
    Only when every gate passes does the judge get to decide DONE.

    ``attempts`` counts failed runs; when it exceeds ``max_retries`` the goal
    auto-pauses (mirrors the turn-budget pause) instead of spinning. A gate
    that failed on an unchanged workspace is not re-run — the recorded
    failure is replayed and the attempt count advances, so a stuck agent
    can't burn wall-clock re-running the same red suite.
    """

    command: str
    timeout_seconds: int = DEFAULT_GATE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_GATE_MAX_RETRIES
    attempts: int = 0
    last_exit_code: Optional[int] = None
    last_output_tail: str = ""
    # Workspace fingerprint at the time of the last FAILED run — used to skip
    # re-running an identical gate when nothing changed since it failed.
    last_failed_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalGate":
        if not isinstance(data, dict):
            return cls(command="")
        return cls(
            command=str(data.get("command") or ""),
            timeout_seconds=int(data.get("timeout_seconds", DEFAULT_GATE_TIMEOUT_SECONDS) or DEFAULT_GATE_TIMEOUT_SECONDS),
            max_retries=int(data.get("max_retries", DEFAULT_GATE_MAX_RETRIES) or DEFAULT_GATE_MAX_RETRIES),
            attempts=int(data.get("attempts", 0) or 0),
            last_exit_code=(int(data["last_exit_code"]) if data.get("last_exit_code") is not None else None),
            last_output_tail=str(data.get("last_output_tail") or ""),
            last_failed_fingerprint=str(data.get("last_failed_fingerprint") or ""),
        )


def workspace_fingerprint(cwd: Optional[str] = None) -> str:
    """Cheap workspace change fingerprint for unchanged-gate skip.

    Uses ``git status --porcelain`` + ``git rev-parse HEAD`` when inside a git
    repo (covers tracked edits, stages, and commits). Outside git, returns
    an empty string — an empty fingerprint never matches, so gates simply
    always re-run (safe fallback, no behavior regression for non-repo work).
    """
    workdir = cwd or os.getcwd()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, cwd=workdir,
        )
        if head.returncode != 0:
            return ""
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, cwd=workdir,
        )
        if status.returncode != 0:
            return ""
        blob = head.stdout.strip() + "\n" + status.stdout
        return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()
    except Exception:
        return ""


def run_gate(gate: GoalGate, *, cwd: Optional[str] = None) -> Tuple[bool, int, str]:
    """Run one gate command. Returns ``(passed, exit_code, output_tail)``.

    The command runs through the shell in ``cwd`` (default: process cwd) with
    a hard timeout; on timeout the process is killed and treated as failed
    with exit code -1. Output is the combined stdout+stderr tail, bounded to
    ``_GATE_OUTPUT_TAIL_CHARS``.
    """
    try:
        proc = subprocess.run(
            gate.command,
            shell=True,
            capture_output=True,
            text=True,
            # A gate runs whatever the operator configured, so its output is
            # arbitrary bytes. The default text mode decodes with the process
            # codepage under errors="strict": one byte the codepage can't map
            # (emoji or CJK from a test runner on a non-UTF-8 Windows console,
            # or stray binary) kills the reader thread, leaves stdout as None,
            # and the tail the agent needs to fix the failure arrives empty.
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(gate.timeout_seconds)),
            cwd=cwd or None,
        )
        combined = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        tail = combined[-_GATE_OUTPUT_TAIL_CHARS:]
        return proc.returncode == 0, proc.returncode, tail
    except subprocess.TimeoutExpired as exc:
        out = ""
        for chunk in (exc.stdout, exc.stderr):
            if chunk:
                out += chunk if isinstance(chunk, str) else chunk.decode("utf-8", "replace")
        tail = (out + f"\n[gate timed out after {gate.timeout_seconds}s]")[-_GATE_OUTPUT_TAIL_CHARS:]
        return False, -1, tail
    except Exception as exc:
        return False, -1, f"[gate could not run: {type(exc).__name__}: {exc}]"


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GoalState:
    """Serializable goal state stored per session."""

    goal: str
    # Terminal success is *only* ``done``.  Every other terminal-ish state
    # intentionally retains its distinct outcome so status surfaces never say
    # that a blocked or impossible goal was achieved.
    status: str = "active"          # active | waiting | awaiting_user | control_plane_error | paused | blocked | unachievable | done | stopped | cleared
    goal_id: str = ""
    revision: int = 1
    turns_used: int = 0
    # ``bounded`` preserves the historical Ralph-loop budget. ``judge`` is
    # deliberately unbounded in *productive* turns; repetition, authority,
    # worker crashes, and control-plane outages are bounded separately.
    termination: str = "bounded"  # bounded | judge
    max_turns: Optional[int] = DEFAULT_MAX_TURNS
    duplicate_failure_limit: int = DEFAULT_DUPLICATE_FAILURE_LIMIT
    stall_turns_before_replan: int = DEFAULT_STALL_TURNS_BEFORE_REPLAN
    require_recovery_exhaustion: bool = True
    terminal_confirmation: bool = True
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None        # "done" | "continue" | "skipped"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    consecutive_parse_failures: int = 0       # judge-output parse failures in a row
    # Transport failures are API/auth/network errors.  Broken API keys return
    # 401 every call — track them separately so the loop auto-pauses instead
    # of burning every turn budget slot on an unreachable judge.
    consecutive_transport_failures: int = 0   # judge API/transport errors in a row
    # User-added criteria appended mid-loop via the /subgoal command.
    # When non-empty the judge prompt and continuation prompt both
    # include them so the agent works toward them and the judge factors
    # them into the verdict. Backwards-compatible: defaults to empty so
    # old state_meta rows load unchanged.
    subgoals: List[str] = field(default_factory=list)
    # Wait barrier: when the agent is blocked on long-running async work
    # (CI poller, build, test run, deploy, rate-limit cooldown) the goal loop
    # PARKS instead of being re-poked every turn into busy-work. Two barrier
    # kinds, set automatically by the judge (which now sees the live
    # background-process list and can return a ``wait`` verdict) or manually
    # via ``/goal wait``:
    #   • ``waiting_on_pid`` — park until that process exits.
    #   • ``waiting_on_session`` — park until that process_registry session's
    #     OWN trigger fires: it exits, OR (if it has watch_patterns) its
    #     pattern matches. Covers long-lived watchers/servers that signal
    #     mid-run via a trigger and may never exit. Preferred over raw pid
    #     when the agent set up a watch_patterns/notify_on_complete process.
    #   • ``waiting_until``  — park until this wall-clock epoch (time backoff).
    # While ANY is active, ``evaluate_after_turn`` short-circuits to
    # should_continue=False without burning a turn or calling the judge. The
    # barrier auto-clears when the pid exits / the trigger fires / the deadline
    # passes, then the next turn resumes normal judging. Cleared by that,
    # ``/goal unwait``, pause, resume, or clear. Backwards-compatible: old
    # state_meta rows load with no barrier.
    waiting_on_pid: Optional[int] = None
    waiting_on_session: Optional[str] = None
    waiting_until: float = 0.0
    waiting_reason: Optional[str] = None
    waiting_since: float = 0.0
    # Optional structured completion contract (outcome / verification /
    # constraints / boundaries / stop_when). Empty by default; a goal with
    # no contract behaves exactly like the original free-form goal.
    contract: GoalContract = field(default_factory=GoalContract)
    # Quality gates (/goal gate add <cmd>): deterministic shell commands that
    # must ALL pass before the judge may declare the goal done. Empty by
    # default — a goal with no gates behaves exactly as before.
    gates: List[GoalGate] = field(default_factory=list)
    # Completion policy.  ``owner`` converts an otherwise valid completion
    # candidate into a capability-bound approval request; it is never enough
    # for a free-form user reply to say "yes".
    approval_policy: str = "automatic"  # automatic | owner
    owner_id: str = ""
    pending_approval_id: str = ""
    pending_approval_reason: str = ""
    pending_approval_at: float = 0.0
    # Durable scheduling / exactly-once wake claim.  ``self_paced`` means a
    # continuation is due immediately; ``interval`` makes every next turn
    # cadence-bound.  Wait barriers use the same schedule and therefore wake
    # after a process exits or a deadline elapses even with no user traffic.
    schedule_mode: str = "self_paced"  # self_paced | interval
    interval_seconds: float = 0.0
    max_runs: int = 0
    next_wake_at: float = 0.0
    wake_generation: int = 0
    wake_claim_id: str = ""
    wake_claimed_at: float = 0.0
    initial_kickoff_pending: bool = False
    route: Dict[str, str] = field(default_factory=dict)
    # A repeated external blocker only becomes a sticky blocked state after
    # three identical observations.  This permits transient retries without
    # mislabelling the goal as complete or permanently blocked too early.
    blocker_fingerprint: str = ""
    consecutive_blockers: int = 0
    # Bounded, redacted receipts give a later approver/status surface concrete
    # evidence without retaining an unbounded transcript copy in state_meta.
    evidence_receipts: List[Dict[str, Any]] = field(default_factory=list)
    # Durable controller ledger.  Entries contain redacted action/result
    # fingerprints, provenance, strategy state, and the controller's next
    # constraint.  It survives compression/restart/reassignment because it is
    # stored with the goal rather than held in an agent turn's context.
    progress_ledger: List[Dict[str, Any]] = field(default_factory=list)
    stalled_turns: int = 0
    duplicate_failures: int = 0
    last_action_fingerprint: str = ""
    last_strategy_constraint: str = ""
    recovery_paths: List[Dict[str, Any]] = field(default_factory=list)
    control_plane_retry_at: float = 0.0
    control_plane_failures: int = 0
    input_notification_sent: bool = False

    def to_json(self) -> str:
        data = asdict(self)
        # asdict already recursed GoalContract into a plain dict.
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        raw_subgoals = data.get("subgoals") or []
        subgoals: List[str] = []
        if isinstance(raw_subgoals, list):
            subgoals = [str(s).strip() for s in raw_subgoals if str(s).strip()]
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            goal_id=str(data.get("goal_id") or ""),
            revision=max(1, int(data.get("revision", 1) or 1)),
            turns_used=int(data.get("turns_used", 0) or 0),
            termination=(
                "judge" if str(data.get("termination") or "").lower() == "judge" else "bounded"
            ),
            max_turns=(
                None
                if str(data.get("termination") or "").lower() == "judge"
                else int(data.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS)
            ),
            duplicate_failure_limit=max(
                1, int(data.get("duplicate_failure_limit", DEFAULT_DUPLICATE_FAILURE_LIMIT) or DEFAULT_DUPLICATE_FAILURE_LIMIT)
            ),
            stall_turns_before_replan=max(
                1, int(data.get("stall_turns_before_replan", DEFAULT_STALL_TURNS_BEFORE_REPLAN) or DEFAULT_STALL_TURNS_BEFORE_REPLAN)
            ),
            require_recovery_exhaustion=bool(data.get("require_recovery_exhaustion", True)),
            terminal_confirmation=bool(data.get("terminal_confirmation", True)),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            consecutive_parse_failures=int(data.get("consecutive_parse_failures", 0) or 0),
            consecutive_transport_failures=int(data.get("consecutive_transport_failures", 0) or 0),
            subgoals=subgoals,
            waiting_on_pid=(int(data["waiting_on_pid"]) if data.get("waiting_on_pid") else None),
            waiting_on_session=(str(data["waiting_on_session"]) if data.get("waiting_on_session") else None),
            waiting_until=float(data.get("waiting_until", 0.0) or 0.0),
            waiting_reason=data.get("waiting_reason"),
            waiting_since=float(data.get("waiting_since", 0.0) or 0.0),
            contract=GoalContract.from_dict(data.get("contract")),
            gates=[
                GoalGate.from_dict(g)
                for g in (data.get("gates") or [])
                if isinstance(g, dict) and str(g.get("command") or "").strip()
            ],
            approval_policy=(
                "owner" if str(data.get("approval_policy") or "").lower() == "owner" else "automatic"
            ),
            owner_id=str(data.get("owner_id") or ""),
            pending_approval_id=str(data.get("pending_approval_id") or ""),
            pending_approval_reason=str(data.get("pending_approval_reason") or ""),
            pending_approval_at=float(data.get("pending_approval_at", 0.0) or 0.0),
            schedule_mode=(
                "interval" if str(data.get("schedule_mode") or "").lower() == "interval" else "self_paced"
            ),
            interval_seconds=max(0.0, float(data.get("interval_seconds", 0.0) or 0.0)),
            max_runs=max(0, int(data.get("max_runs", 0) or 0)),
            next_wake_at=float(data.get("next_wake_at", 0.0) or 0.0),
            wake_generation=max(0, int(data.get("wake_generation", 0) or 0)),
            wake_claim_id=str(data.get("wake_claim_id") or ""),
            wake_claimed_at=float(data.get("wake_claimed_at", 0.0) or 0.0),
            initial_kickoff_pending=bool(data.get("initial_kickoff_pending", False)),
            route=(data.get("route") if isinstance(data.get("route"), dict) else {}),
            blocker_fingerprint=str(data.get("blocker_fingerprint") or ""),
            consecutive_blockers=max(0, int(data.get("consecutive_blockers", 0) or 0)),
            evidence_receipts=[
                r for r in (data.get("evidence_receipts") or [])[-MAX_EVIDENCE_RECEIPTS:]
                if isinstance(r, dict)
            ],
            progress_ledger=[
                r for r in (data.get("progress_ledger") or [])[-MAX_PROGRESS_LEDGER_ENTRIES:]
                if isinstance(r, dict)
            ],
            stalled_turns=max(0, int(data.get("stalled_turns", 0) or 0)),
            duplicate_failures=max(0, int(data.get("duplicate_failures", 0) or 0)),
            last_action_fingerprint=str(data.get("last_action_fingerprint") or ""),
            last_strategy_constraint=str(data.get("last_strategy_constraint") or ""),
            recovery_paths=[
                r for r in (data.get("recovery_paths") or [])[-16:]
                if isinstance(r, dict)
            ],
            control_plane_retry_at=float(data.get("control_plane_retry_at", 0.0) or 0.0),
            control_plane_failures=max(0, int(data.get("control_plane_failures", 0) or 0)),
            input_notification_sent=bool(data.get("input_notification_sent", False)),
        )

    # --- contract helpers -------------------------------------------------

    def has_contract(self) -> bool:
        return self.contract is not None and not self.contract.is_empty()

    # --- subgoals helpers -------------------------------------------------

    def render_subgoals_block(self) -> str:
        """Render the subgoals as a numbered ``- N. text`` block. Empty
        when no subgoals exist."""
        if not self.subgoals:
            return ""
        return "\n".join(f"- {i}. {text}" for i, text in enumerate(self.subgoals, start=1))


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"goal:{session_id}"


_DB_CACHE: Dict[str, Any] = {}
_DB_BOOTSTRAP_LOCK = threading.Lock()
_DB_BOOTSTRAP_INFLIGHT: Dict[str, threading.Event] = {}

# How long a loop-thread caller waits for an ALREADY-RUNNING bootstrap
# before degrading to None. Normal SessionDB init is ~10-100ms, so a call
# that arrives mid-bootstrap usually picks the cached instance up within
# this window. A contended init (locked state.db mid-migration) blows past
# it and the caller degrades. The loop stalls far under the watchdog's
# probe window.
_DB_BOOTSTRAP_LOOP_WAIT_S = 0.25

# The call that STARTS the bootstrap (cold cache, nothing in flight)
# waits this long instead of the short window above. A fresh state.db
# init measures ~300ms warm on a fast machine: schema DDL, FTS table
# creation, and the first hermes_cli.config import (journal-mode
# resolution). It is longer on a slow CI box, and it is well past 0.25s.
# The old window dropped the first /goal write. The response said
# "Goal set" but nothing persisted. The longer window is a bounded
# one-time stall. Only the kick call pays it. Every later call keeps
# the short window, so a contended migration never stalls the loop
# repeatedly.
_DB_BOOTSTRAP_INIT_WAIT_S = 1.5


def _bootstrap_session_db(home: str, done: threading.Event) -> None:
    """Construct SessionDB off-loop and populate the cache (worker thread)."""
    try:
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_state import SessionDB

        # Bind the caller's home for this thread. The cache key is the
        # caller's scoped home, so the constructed SessionDB must point at
        # that home's state.db too. Without the override, a multiplexed
        # worker thread resolves the process env (the default profile's
        # HERMES_HOME). It then caches the wrong profile's DB under this
        # profile's key.
        token = set_hermes_home_override(home)
        try:
            db = SessionDB()
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: background SessionDB() raised (%s)", exc)
        db = None
    with _DB_BOOTSTRAP_LOCK:
        if db is not None and home not in _DB_CACHE:
            _DB_CACHE[home] = db
        _DB_BOOTSTRAP_INFLIGHT.pop(home, None)
    done.set()


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current HERMES_HOME.

    SessionDB has no built-in singleton, but opening a new connection per
    /goal call would thrash the file. We cache one instance per
    ``hermes_home`` path so profile switches still pick up the right DB.
    Defensive against import/instantiation failures so tests and
    non-standard launchers can still use the GoalManager.

    Never constructs SessionDB on an event-loop thread. ``SessionDB.__init__``
    runs schema init, and a migration against a contended state.db blocks for
    seconds — on the gateway's loop thread that starves the loop-liveness
    watchdog, which hard-exits the process (exit 75) and crash-loops the
    gateway (enterprise field report, 2026-08-14). On a cache miss with a running
    loop we kick a one-shot background bootstrap and wait a bounded grace
    window for it. The kick call waits the one-time init window
    (``_DB_BOOTSTRAP_INIT_WAIT_S``), so a healthy cold init completes and
    the first write is not dropped. Later calls wait only the short window
    (``_DB_BOOTSTRAP_LOOP_WAIT_S``). On timeout we return None. Every
    caller degrades gracefully on None, and a later call returns the
    cached instance.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        on_loop_thread = False
    else:
        on_loop_thread = True

    if on_loop_thread:
        with _DB_BOOTSTRAP_LOCK:
            # Re-check under the lock: a bootstrap may have finished between
            # the unlocked read above and here.
            cached = _DB_CACHE.get(home)
            if cached is not None:
                return cached
            done = _DB_BOOTSTRAP_INFLIGHT.get(home)
            if done is None:
                done = threading.Event()
                _DB_BOOTSTRAP_INFLIGHT[home] = done
                threading.Thread(
                    target=_bootstrap_session_db,
                    args=(home, done),
                    name="goals-sessiondb-bootstrap",
                    daemon=True,
                ).start()
                # This call starts the bootstrap, so it pays the one-time
                # init cost. Wait long enough for a healthy cold init
                # (~300ms warm, more on slow CI) to finish. This keeps the
                # first goal/heartbeat write from being silently dropped.
                wait = _DB_BOOTSTRAP_INIT_WAIT_S
            else:
                # Bootstrap already running: brief grace window only. A
                # healthy init usually finishes in tens of ms, so this
                # still picks the cached instance up. A contended init
                # (the crash-loop scenario) exceeds the window and we
                # degrade to None. The stall is bounded, far below the
                # watchdog's probe timeout.
                wait = _DB_BOOTSTRAP_LOOP_WAIT_S
        done.wait(wait)
        return _DB_CACHE.get(home)

    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB() raised (%s)", exc)
        return None
    with _DB_BOOTSTRAP_LOCK:
        existing = _DB_CACHE.get(home)
        if existing is not None:
            # A concurrent bootstrap won the race; keep one instance and
            # close ours so connections don't leak.
            try:
                db.close()
            except Exception:
                pass
            return existing
        _DB_CACHE[home] = db
    return db


def _warn_dropped_write(manager: str, kind: str, session_id: str) -> None:
    """Log a dropped state write at WARNING.

    The reply already told the user that the state was set. A silent
    drop makes that reply a lie. One shared message keeps the goal,
    loop, and heartbeat logs greppable as one bug class.
    """
    logger.warning(
        "%s: %s for %s not persisted — session DB unavailable "
        "(bootstrap window exceeded, in-memory state still active)",
        manager,
        kind,
        session_id,
    )


def load_goal(session_id: str) -> Optional[GoalState]:
    """Load the goal for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("GoalManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return GoalState.from_json(raw)
    except Exception as exc:
        logger.warning("GoalManager: could not parse stored goal for %s: %s", session_id, exc)
        return None


def save_goal(session_id: str, state: GoalState) -> None:
    """Persist a goal to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        _warn_dropped_write("GoalManager", "goal", session_id)
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("GoalManager: set_meta failed: %s", exc)


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


def list_schedulable_goals() -> List[Tuple[str, GoalState]]:
    """Return active/waiting goals for the durable surface schedulers.

    State lives in the same ``SessionDB.state_meta`` namespace as legacy
    goals, so pre-existing rows remain untouched; only rows with a schedulable
    lifecycle state are returned.  A driver re-instantiates ``GoalManager``
    before claiming to avoid acting on this possibly-stale scan snapshot.
    """
    db = _get_session_db()
    if db is None:
        return []
    try:
        rows = db.list_meta_prefix("goal:")
    except Exception as exc:
        logger.debug("GoalManager: list_meta_prefix failed: %s", exc)
        return []
    result: List[Tuple[str, GoalState]] = []
    for key, raw in rows:
        session_id = key[len("goal:"):]
        if not session_id or not raw:
            continue
        try:
            state = GoalState.from_json(raw)
        except Exception:
            continue
        if state.status in {"active", "waiting", "control_plane_error"}:
            result.append((session_id, state))
    return result


def migrate_goal_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a persistent /goal from a parent session to its continuation.

    Context compression rotates ``session_id`` to a fresh child session,
    but ``load_goal`` does a flat ``goal:<session_id>`` lookup with no
    parent-lineage walk — so an active goal silently dies at the
    compaction boundary (#33618). Copy the goal onto the new session and
    archive the old row as ``cleared`` so exactly one active goal row
    exists per logical conversation (avoids the "two active goals"
    hazard of a pure copy).

    Returns True when a goal was migrated, False when there was nothing
    to migrate or the DB was unavailable. Best-effort and never raises —
    a failure here must not block compression.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_goal(old_session_id)
        if state is None or getattr(state, "status", None) == "cleared":
            return False
        # Don't clobber a goal already set on the child (e.g. a resumed
        # lineage that re-established its own goal).
        if load_goal(new_session_id) is not None:
            return False
        save_goal(new_session_id, state)
        # Archive the parent's row so it isn't double-counted as active.
        clear_goal(old_session_id)
        logger.debug(
            "GoalManager: migrated goal %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("GoalManager: goal migration failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently alive.

    Delegates to ``gateway.status._pid_exists`` — the canonical,
    cross-platform, footgun-safe liveness check (psutil with a ctypes /
    POSIX fallback). Critically this avoids ``os.kill(pid, 0)``, which on
    Windows is NOT a no-op: it routes to ``CTRL_C_EVENT`` and hard-kills the
    target's console process group (bpo-14484). Any error resolves to False
    (treat unknown as dead) so a stale barrier never wedges the loop — the
    worst case is the goal resumes one turn early, which is safe.
    """
    if not pid or pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(int(pid)))
    except Exception:
        pass
    # Last-resort fallback if gateway.status is unavailable: psutil directly.
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return False


def _session_waiting(session_id: str) -> bool:
    """Whether a goal parked on a process_registry session should stay parked.

    Delegates to ``process_registry.is_session_waiting`` — True while the
    session is running and (if it has watch_patterns) its trigger hasn't fired.
    Fail-safe: any import/registry error yields False (don't wait) so a stale
    barrier can never wedge the loop.
    """
    if not session_id:
        return False
    try:
        from tools.process_registry import process_registry

        return bool(process_registry.is_session_waiting(session_id))
    except Exception:
        return False


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _goal_judge_max_tokens() -> int:
    """Resolve auxiliary.goal_judge.max_tokens, falling back to the default.

    ``load_config()`` is cached on the config file's (mtime, size), so calling
    this once per judge turn is cheap. A non-positive or non-int value falls
    back to the default rather than crashing the goal loop.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = (
            (cfg.get("auxiliary") or {})
            .get("goal_judge", {})
            .get("max_tokens", DEFAULT_JUDGE_MAX_TOKENS)
        )
        value = int(value)
        if value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_JUDGE_MAX_TOKENS


def _goal_judge_timeout() -> float:
    """Resolve auxiliary.goal_judge.timeout, falling back to the default.

    Mirrors :func:`_goal_judge_max_tokens`. The key is declared in
    ``DEFAULT_CONFIG`` and surfaces in the auxiliary config UI, but the
    judge path used to hardcode ``DEFAULT_JUDGE_TIMEOUT`` and never read
    it — so a user raising the timeout for a slow-but-healthy reasoning
    endpoint got no effect, and the loop auto-paused on misleading
    transport failures pointing at provider/key (#91022). A non-positive
    or non-numeric value falls back rather than crashing the goal loop.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = (
            (cfg.get("auxiliary") or {})
            .get("goal_judge", {})
            .get("timeout", DEFAULT_JUDGE_TIMEOUT)
        )
        value = float(value)
        if value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_JUDGE_TIMEOUT


def _parse_judge_response(raw: str) -> Tuple[str, str, bool, Optional[Dict[str, Any]]]:
    """Parse the judge's reply. Fail-open on unusable output.

    Returns ``(verdict, reason, parse_failed, wait_directive)`` where:
      - ``verdict`` is ``"done"``, ``"continue"``, or ``"wait"``.
      - ``parse_failed`` is True when the judge returned output that couldn't
        be interpreted as the expected JSON verdict (empty body, prose,
        malformed JSON). Callers use it to auto-pause after N consecutive
        parse failures so a weak judge model doesn't silently burn the budget.
      - ``wait_directive`` is set only for ``verdict == "wait"``: a dict with
        ``{"pid": int}`` or ``{"seconds": int}`` (whichever the judge supplied).
        ``None`` otherwise. If a wait verdict carries neither a usable pid nor
        seconds, it is downgraded to ``continue`` (can't park on nothing).

    Accepts both the new ``{"verdict": ...}`` shape and the legacy
    ``{"done": <bool>}`` shape.
    """
    if not raw:
        return "continue", "judge returned empty response", True, None

    text = raw.strip()

    # Strip markdown code fences the model may wrap JSON in.
    if text.startswith("```"):
        text = text.strip("`")
        # Peel off leading json/JSON/etc tag
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    # First try: parse the whole blob.
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(text)
    except Exception:
        # Second try: pull the first JSON object out.
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return "continue", f"judge reply was not JSON: {_truncate(raw, 200)!r}", True, None

    reason = str(data.get("reason") or "").strip() or "no reason provided"

    # Determine verdict — prefer the explicit "verdict" field, fall back to
    # the legacy "done" boolean.
    verdict_raw = data.get("verdict")
    if isinstance(verdict_raw, str):
        verdict = verdict_raw.strip().lower()
    else:
        done_val = data.get("done")
        if isinstance(done_val, str):
            done = done_val.strip().lower() in {"true", "yes", "1", "done"}
        else:
            done = bool(done_val)
        verdict = "done" if done else "continue"

    if verdict not in {"done", "continue", "wait", "needs_user", "blocked", "unachievable"}:
        verdict = "continue"

    if verdict != "wait":
        return verdict, reason, False, None

    # Wait verdict: extract a concrete directive (pid or seconds). Accept a
    # few key spellings the model might emit.
    def _first_int(*keys: str) -> Optional[int]:
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except (TypeError, ValueError):
                continue
        return None

    # Prefer a session-id directive (releases on the process's own trigger —
    # exit OR watch-pattern match), then pid (exit only), then seconds.
    sess = data.get("wait_on_session") or data.get("session_id") or data.get("wait_session")
    if isinstance(sess, str) and sess.strip():
        return "wait", reason, False, {"session_id": sess.strip()}
    pid = _first_int("wait_on_pid", "pid", "wait_pid")
    if pid is not None:
        return "wait", reason, False, {"pid": pid}
    seconds = _first_int("wait_for_seconds", "seconds", "wait_seconds")
    if seconds is not None:
        return "wait", reason, False, {"seconds": seconds}
    # Wait with no usable target — can't park on nothing; treat as continue.
    return "continue", f"{reason} (wait verdict had no target — continuing)", False, None


def _render_background_block(background_processes: Optional[List[Dict[str, Any]]]) -> str:
    """Render the live background-process list for the judge prompt.

    Each entry is a ``process_registry.list_sessions()`` dict. Only RUNNING
    processes are worth showing (an exited one is nothing to wait on). Returns
    an empty string when there's nothing running, so the judge prompt is
    byte-identical to the no-background case (no behavior change for the
    common path).
    """
    if not background_processes:
        return ""
    lines: List[str] = []
    for p in background_processes:
        if not isinstance(p, dict):
            continue
        if p.get("status") == "exited":
            continue
        pid = p.get("pid")
        if not pid:
            continue
        cmd = _truncate(str(p.get("command") or "").replace("\n", " ").strip(), 120)
        uptime = p.get("uptime_seconds")
        tail = _truncate(str(p.get("output_preview") or "").replace("\n", " ").strip(), 120)
        sid = p.get("session_id")
        line = f"- pid {pid}"
        if sid:
            line += f" / session {sid}"
        line += f": {cmd}"
        if uptime is not None:
            line += f" (running {uptime}s)"
        # Surface the process's own trigger so the judge can wait on a
        # mid-run signal (watch-pattern) or completion, not just exit.
        wps = p.get("watch_patterns")
        if wps:
            hit = " [already matched]" if p.get("watch_hit") else ""
            line += f" | watch_patterns={wps}{hit}"
        elif p.get("notify_on_complete"):
            line += " | notify_on_complete"
        if tail:
            line += f" | recent output: {tail}"
        lines.append(line)
    if not lines:
        return ""
    return JUDGE_BACKGROUND_BLOCK_TEMPLATE.format(background_lines="\n".join(lines))


def judge_goal(
    goal: str,
    last_response: str,
    *,
    timeout: Optional[float] = None,
    subgoals: Optional[List[str]] = None,
    background_processes: Optional[List[Dict[str, Any]]] = None,
    contract: Optional[GoalContract] = None,
) -> Tuple[str, str, bool, Optional[Dict[str, Any]], bool]:
    """Ask the auxiliary model whether the goal is satisfied.

    Returns ``(verdict, reason, parse_failed, wait_directive, transport_failed)`` where verdict
    is ``"done"``, ``"continue"``, ``"wait"``, or ``"skipped"`` (when the
    judge couldn't be reached). ``wait_directive`` is set only for ``"wait"``
    (``{"pid": int}`` or ``{"seconds": int}``); ``None`` otherwise.

    ``parse_failed`` is True only when the judge call succeeded but its output
    was unusable (empty or non-JSON). API/transport errors return False — they
    are transient and should fail-open silently.

    ``transport_failed`` is True only when the judge couldn't reach the API at
    all (auth 401, timeout, DNS, connection error).  Repeated transport
    failures signal a permanent config problem (e.g. invalid API key).  Callers
    use this flag to auto-pause after N consecutive transport failures (see
    ``DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES``). Callers use this flag to
    auto-pause after N consecutive parse failures (see
    ``DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES``).

    ``subgoals`` is an optional list of user-added criteria (from
    ``/subgoal``) factored into the verdict. ``background_processes`` is the
    live ``process_registry.list_sessions()`` snapshot; when the agent is
    waiting on one (a CI poller, build, etc.) the judge can return a ``wait``
    verdict naming its pid, parking the loop instead of re-poking.
    ``contract`` is an optional structured completion contract; when present
    the judge decides DONE strictly against its Verification criterion and
    refuses completion when a Constraint was violated. All three are additive
    — a contract, subgoals, and a background-process list can coexist in one
    judge prompt; when none are set, behavior is identical to the original
    free-form judge.

    This is deliberately fail-open: transport errors return ``("continue", ..., ..., None, True)``
    — the ``transport_failed=True`` flag lets callers track and auto-pause after
    N consecutive transport failures (see
    ``DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES``) so a permanently broken
    judge doesn't burn the entire turn budget.
    """
    if not goal.strip():
        return "skipped", "empty goal", False, None, False
    if not last_response.strip():
        # No substantive reply this turn — almost certainly not done yet.
        return "continue", "empty response (nothing to evaluate)", False, None, False
    if timeout is None:
        # The declared default for this path is the config key, not the
        # module constant — see _goal_judge_timeout (#91022).
        timeout = _goal_judge_timeout()

    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return "continue", "auxiliary client unavailable", False, None, False

    # Build the prompt. Priority: contract > subgoals > plain. When both a
    # contract and subgoals exist, the subgoals are appended into the
    # contract block as extra criteria so the judge sees a single source of
    # truth.
    clean_subgoals = [s.strip() for s in (subgoals or []) if s and s.strip()]
    background_block = _render_background_block(background_processes)
    current_time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    if contract is not None and not contract.is_empty():
        contract_block = contract.render_block()
        if clean_subgoals:
            extra = "\n".join(
                f"- Extra criterion {i}: {text}"
                for i, text in enumerate(clean_subgoals, start=1)
            )
            contract_block = f"{contract_block}\n{extra}"
        prompt = JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            contract_block=_truncate(contract_block, 2500),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    elif clean_subgoals:
        subgoals_block = "\n".join(
            f"- {i}. {text}" for i, text in enumerate(clean_subgoals, start=1)
        )
        prompt = JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            subgoals_block=_truncate(subgoals_block, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    else:
        prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )

    last_transport_error: Optional[Exception] = None
    last_parse: Optional[Tuple[str, str, bool, Optional[Dict[str, Any]]]] = None
    for attempt in range(1, DEFAULT_JUDGE_ATTEMPTS_PER_TURN + 1):
        try:
            # Route through call_llm so auxiliary.goal_judge.* config
            # (provider/model/base_url, extra_body, reasoning_effort, retries)
            # all apply — the direct-create path dropped extra_body (#35566).
            resp = call_llm(
                task="goal_judge",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=_goal_judge_max_tokens(),
                timeout=timeout,
            )
        except Exception as exc:
            last_transport_error = exc
            logger.info(
                "goal judge: API attempt %d/%d failed (%s)",
                attempt, DEFAULT_JUDGE_ATTEMPTS_PER_TURN, exc,
            )
            if attempt < DEFAULT_JUDGE_ATTEMPTS_PER_TURN:
                time.sleep(DEFAULT_JUDGE_RETRY_BACKOFF_SECONDS[attempt - 1])
            continue

        try:
            raw = resp.choices[0].message.content or ""
        except Exception:
            raw = ""
        parsed = _parse_judge_response(raw)
        last_parse = parsed
        verdict, reason, parse_failed, wait_directive = parsed
        if parse_failed and attempt < DEFAULT_JUDGE_ATTEMPTS_PER_TURN:
            logger.info("goal judge: retrying unparseable attempt %d/%d", attempt, DEFAULT_JUDGE_ATTEMPTS_PER_TURN)
            time.sleep(DEFAULT_JUDGE_RETRY_BACKOFF_SECONDS[attempt - 1])
            continue
        logger.info(
            "goal judge: verdict=%s reason=%s%s",
            verdict, _truncate(reason, 120),
            f" wait={wait_directive}" if wait_directive else "",
        )
        return verdict, reason, parse_failed, wait_directive, False

    if last_parse is not None:
        verdict, reason, parse_failed, wait_directive = last_parse
        return verdict, reason, parse_failed, wait_directive, False
    if last_transport_error is not None:
        return "continue", f"judge error: {type(last_transport_error).__name__}", False, None, True
    return "continue", "judge unavailable", False, None, True


def _auxiliary_json_call(
    *,
    task: str,
    system: str,
    prompt: str,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call an auxiliary controller role with bounded retry/backoff.

    This is deliberately controller-only: retrying it never invokes another
    agent turn.  A caller can therefore park on ``control_plane_error`` rather
    than treating a broken judge, recovery coach, or verifier as task failure.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        return None, f"auxiliary client unavailable: {type(exc).__name__}"
    timeout = timeout if timeout is not None else _goal_judge_timeout()
    error = "controller returned no response"
    for attempt in range(DEFAULT_JUDGE_ATTEMPTS_PER_TURN):
        try:
            resp = call_llm(
                task=task,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens or _goal_judge_max_tokens(),
                timeout=timeout,
            )
            raw = resp.choices[0].message.content or ""
            data = _extract_json_object(raw)
            if isinstance(data, dict):
                return data, None
            error = "controller reply was not JSON"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt < DEFAULT_JUDGE_ATTEMPTS_PER_TURN - 1:
            time.sleep(DEFAULT_JUDGE_RETRY_BACKOFF_SECONDS[attempt])
    return None, error


def _normalize_wait_directive(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    session_id = str(value.get("session_id") or "").strip()
    if session_id:
        return {"session_id": session_id}
    for key in ("pid", "seconds"):
        try:
            number = int(value.get(key) or 0)
        except (TypeError, ValueError):
            number = 0
        if number > 0:
            return {key: number}
    return None


def _ledger_for_prompt(entries: List[Dict[str, Any]], *, limit: int = 12) -> str:
    """Render a redacted bounded ledger for a side-model prompt."""
    compact: List[Dict[str, Any]] = []
    for entry in (entries or [])[-limit:]:
        if not isinstance(entry, dict):
            continue
        compact.append({
            "id": entry.get("id"),
            "strategy_family": entry.get("strategy_family"),
            "action_fingerprint": entry.get("action_fingerprint"),
            "progress": entry.get("progress"),
            "verdict": entry.get("verdict"),
            "blocker_class": entry.get("blocker_class"),
            "provenance": entry.get("provenance"),
            "reason": _receipt_excerpt(str(entry.get("reason") or ""), 280),
            "evidence": _receipt_excerpt(str(entry.get("evidence") or ""), 420),
        })
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def judge_goal_with_ledger(
    goal: str,
    last_response: str,
    *,
    contract: Optional[GoalContract],
    ledger: List[Dict[str, Any]],
    background_processes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a structured judge-led controller decision.

    This leaves :func:`judge_goal` intact for bounded legacy cards.  Keeping
    the richer contract isolated is important for both wire compatibility and
    prompt caching in existing conversations.
    """
    contract_block = contract.render_block() if contract and not contract.is_empty() else "(no structured contract)"
    prompt = (
        f"Goal:\n{_truncate(goal, 2400)}\n\n"
        f"Completion contract:\n{_truncate(contract_block, 2400)}\n\n"
        f"Latest agent response:\n{_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS)}\n\n"
        f"Cumulative evidence ledger (newest last):\n{_ledger_for_prompt(ledger)}\n\n"
        f"Live background work:\n{_render_background_block(background_processes) or '(none)'}\n"
    )
    data, error = _auxiliary_json_call(
        task="goal_judge",
        system=JUDGE_LED_SYSTEM_PROMPT,
        prompt=prompt,
    )
    if data is None:
        return {
            "verdict": "control_plane_error",
            "progress": "stalled",
            "reason": error or "goal judge unavailable",
            "evidence_refs": [],
            "blocker_class": "environment",
            "recoverable": True,
            "untried_strategy_families": [],
            "next_strategy_constraint": "wait for controller health before another agent turn",
            "wait_directive": None,
        }
    verdict = str(data.get("verdict") or "continue").strip().lower()
    # A legacy-shaped judge response is harmless in a mixed rollout; it only
    # maps successful legacy DONE to the new explicit achieved state.
    if verdict == "done":
        verdict = "achieved"
    allowed = {
        "achieved", "continue", "replan", "wait", "needs_input",
        "not_achievable", "policy_stop",
    }
    if verdict not in allowed:
        verdict = "replan"
    progress = str(data.get("progress") or "stalled").strip().lower()
    if progress not in {"advanced", "stalled", "regressed"}:
        progress = "stalled"
    blocker_class = str(data.get("blocker_class") or "ambiguity").strip().lower()
    if blocker_class not in {
        "transient", "environment", "capability", "dependency", "ambiguity",
        "authorization", "policy", "impossible",
    }:
        blocker_class = "ambiguity"
    strategies = data.get("untried_strategy_families")
    if not isinstance(strategies, list):
        strategies = []
    return {
        "verdict": verdict,
        "progress": progress,
        "reason": _receipt_excerpt(str(data.get("reason") or "no reason provided"), 700),
        "evidence_refs": [str(x) for x in (data.get("evidence_refs") or []) if str(x).strip()][-12:],
        "blocker_class": blocker_class,
        "recoverable": bool(data.get("recoverable", verdict not in {"not_achievable", "policy_stop"})),
        "untried_strategy_families": [str(x).strip() for x in strategies if str(x).strip()][-8:],
        "next_strategy_constraint": _receipt_excerpt(
            str(data.get("next_strategy_constraint") or "take a concrete, materially different next step"),
            700,
        ),
        "wait_directive": _normalize_wait_directive(data.get("wait_directive")),
    }


def recovery_coach(
    state: GoalState,
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce recovery paths only after a stall/proposed blocker."""
    prompt = (
        f"Goal:\n{_truncate(state.goal, 2400)}\n\n"
        f"Controller decision:\n{json.dumps(decision, ensure_ascii=False)}\n\n"
        f"Ledger:\n{_ledger_for_prompt(state.progress_ledger)}\n\n"
        "Previously tried recovery paths:\n"
        f"{json.dumps(state.recovery_paths[-12:], ensure_ascii=False)}"
    )
    data, error = _auxiliary_json_call(
        task="goal_recovery",
        system=GOAL_RECOVERY_SYSTEM_PROMPT,
        prompt=prompt,
    )
    if data is None:
        return {"control_plane_error": error or "recovery coach unavailable", "strategies": []}
    raw = data.get("strategies")
    strategies: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                family = _receipt_excerpt(str(item.get("family") or ""), 160).strip()
                step = _receipt_excerpt(str(item.get("next_step") or ""), 420).strip()
                if family and step:
                    strategies.append({
                        "family": family,
                        "next_step": step,
                        "why_safe": _receipt_excerpt(str(item.get("why_safe") or ""), 240),
                    })
    return {
        "strategies": strategies[:8],
        "irreducible_dependency": _receipt_excerpt(str(data.get("irreducible_dependency") or ""), 500),
    }


def verify_terminal_goal_decision(state: GoalState, decision: Dict[str, Any]) -> Dict[str, Any]:
    """Independently accept or reject an attempted terminal controller state."""
    if not state.terminal_confirmation:
        return {"accept": True, "reason": "terminal confirmation disabled", "untried_strategy_families": []}
    contract_block = state.contract.render_block() if state.has_contract() else "(no structured contract)"
    prompt = (
        f"Goal:\n{_truncate(state.goal, 2400)}\n\n"
        f"Completion contract:\n{_truncate(contract_block, 2400)}\n\n"
        f"Proposed terminal decision:\n{json.dumps(decision, ensure_ascii=False)}\n\n"
        f"Cumulative ledger:\n{_ledger_for_prompt(state.progress_ledger)}\n\n"
        f"Recovery paths:\n{json.dumps(state.recovery_paths[-16:], ensure_ascii=False)}"
    )
    data, error = _auxiliary_json_call(
        task="goal_terminal_verifier",
        system=GOAL_TERMINAL_VERIFIER_SYSTEM_PROMPT,
        prompt=prompt,
    )
    if data is None:
        return {"accept": False, "control_plane_error": error or "terminal verifier unavailable", "untried_strategy_families": []}
    untried = data.get("untried_strategy_families")
    if not isinstance(untried, list):
        untried = []
    return {
        "accept": bool(data.get("accept")),
        "reason": _receipt_excerpt(str(data.get("reason") or "terminal verifier rejected"), 600),
        "untried_strategy_families": [str(x).strip() for x in untried if str(x).strip()][:8],
    }


def gather_background_processes(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the live background-process snapshot for the goal judge.

    Thin, fail-safe wrapper over ``process_registry.list_sessions(task_id)``.
    Returns only RUNNING processes (an exited one is nothing to wait on) and
    never raises — any import/registry failure yields ``[]`` so the goal loop
    degrades to its pre-wait-barrier behavior (judge just won't see processes).
    The drivers (CLI + gateway) call this and pass the result into
    ``GoalManager.evaluate_after_turn(background_processes=...)``.
    """
    try:
        from tools.process_registry import process_registry

        sessions = process_registry.list_sessions(task_id=task_id) or []
    except Exception as exc:
        logger.debug("gather_background_processes failed: %s", exc)
        return []
    return [s for s in sessions if isinstance(s, dict) and s.get("status") != "exited"]


def draft_contract(objective: str, *, timeout: Optional[float] = None) -> Optional[GoalContract]:
    """Expand a plain-language objective into a structured completion contract.

    Uses the ``goal_judge`` auxiliary task (main-model-first, cache-safe — it
    is a side LLM call, not a conversation turn). Returns a populated
    :class:`GoalContract` on success, or ``None`` when the auxiliary client is
    unavailable or the model's reply can't be parsed. Callers fall back to a
    bare free-form goal in that case, so a missing/weak aux model never blocks
    setting a goal.
    """
    objective = (objective or "").strip()
    if not objective:
        return None
    if timeout is None:
        # Same config-backed default as judge_goal (#91022).
        timeout = _goal_judge_timeout()

    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("goal draft: auxiliary client import failed: %s", exc)
        return None

    try:
        # Route through call_llm — same #35566 fix as the judge call above.
        resp = call_llm(
            task="goal_judge",
            messages=[
                {"role": "system", "content": DRAFT_CONTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Objective:\n{_truncate(objective, 4000)}"},
            ],
            temperature=0,
            max_tokens=_goal_judge_max_tokens(),
            timeout=timeout,
        )
    except Exception as exc:
        logger.info("goal draft: API call failed (%s)", exc)
        return None

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        logger.debug("goal draft: reply was not JSON: %r", _truncate(raw, 200))
        return None
    contract = GoalContract.from_dict(data)
    return None if contract.is_empty() else contract


_RECEIPT_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|password|secret|authorization)\b\s*[:=]\s*)([^\s,;]+)"
)


def _receipt_excerpt(text: str, limit: int = 700) -> str:
    """Return a bounded, best-effort secret-free evidence excerpt."""
    cleaned = _RECEIPT_SECRET_RE.sub(r"\1[REDACTED]", str(text or ""))
    return _truncate(cleaned, limit)


def _background_wait_directive(background_processes: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Choose the first live background dependency as a durable wait target.

    Completion must not race an in-flight process the goal itself may depend
    on.  Prefer the process registry session because it can release on a
    watch-pattern match; otherwise use its pid.  Invalid / exited entries are
    deliberately ignored.
    """
    for process in background_processes or []:
        if not isinstance(process, dict) or process.get("status") == "exited":
            continue
        session_id = str(process.get("session_id") or "").strip()
        if session_id:
            return {"session_id": session_id}
        try:
            pid = int(process.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid > 0:
            return {"pid": pid}
    return None


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort: pull the first JSON object out of a model reply.

    Shares the fence-stripping + first-object fallback logic used by the
    judge parser, but returns the dict (or None) rather than a verdict.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    try:
        data = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None
    return data if isinstance(data, dict) else None


def parse_goal_start_args(text: str) -> Dict[str, Any]:
    """Parse goal-start flags without changing the free-form goal contract.

    Supported forms are ``--require-approval``, ``--interval 5m`` (or
    ``self-paced``), and ``--max-runs N``.  Flags are removed before the
    remaining multi-line text is handed to ``parse_contract``.
    """
    raw = (text or "").strip()
    result: Dict[str, Any] = {
        "goal": raw,
        "approval_policy": "automatic",
        "termination": None,
        "interval_seconds": None,
        "max_runs": 0,
        "error": None,
    }
    if not raw:
        result["error"] = "goal text is empty"
        return result
    if re.search(r"(?:^|\s)--require-approval(?:\s|$)", raw, re.IGNORECASE):
        result["approval_policy"] = "owner"
        raw = re.sub(r"(?:^|\s)--require-approval(?=\s|$)", " ", raw, flags=re.IGNORECASE)

    termination_match = re.search(r"(?:^|\s)--termination\s+(bounded|judge)(?=\s|$)", raw, re.IGNORECASE)
    if termination_match:
        result["termination"] = termination_match.group(1).lower()
        raw = raw[:termination_match.start()] + " " + raw[termination_match.end():]
    elif re.search(r"(?:^|\s)--termination(?:\s|$)", raw, re.IGNORECASE):
        result["error"] = "--termination expects bounded or judge"
        return result

    interval_match = re.search(r"(?:^|\s)--interval\s+(self-paced|[0-9]+(?:h|m|s)(?:[0-9]+(?:h|m|s))*)", raw, re.IGNORECASE)
    if interval_match:
        token = interval_match.group(1).lower()
        if token != "self-paced":
            try:
                from hermes_cli.loops import parse_interval_token

                seconds = parse_interval_token(token)
            except Exception:
                seconds = None
            if not seconds:
                result["error"] = f"invalid --interval {token!r}; use e.g. 5m or self-paced"
                return result
            result["interval_seconds"] = float(seconds)
        raw = raw[:interval_match.start()] + " " + raw[interval_match.end():]

    runs_match = re.search(r"(?:^|\s)--max-runs\s+(\S+)", raw, re.IGNORECASE)
    if runs_match:
        try:
            max_runs = int(runs_match.group(1))
            if max_runs < 1:
                raise ValueError
        except ValueError:
            result["error"] = "--max-runs expects a positive integer"
            return result
        result["max_runs"] = max_runs
        raw = raw[:runs_match.start()] + " " + raw[runs_match.end():]

    result["goal"] = raw.strip()
    if not result["goal"]:
        result["error"] = "goal text is empty"
    return result


def dispatch_goal_loop_alias(
    mgr: "GoalManager",
    args: str,
    *,
    owner_id: str = "",
    route: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Interpret new ``/loop`` commands as scheduled ``/goal`` commands.

    ``/loop`` historically owned a second persistent scheduler. The alias
    keeps its useful shorthand (interval, ``--times``, and ``--until``) while
    making a goal's durable lifecycle the single control plane for new work.
    Surfaces may still send control verbs to a pre-existing legacy loop row;
    they use this helper for every new alias-backed loop.
    """
    from hermes_cli.loops import parse_loop_args

    arg = (args or "").strip()
    lower = arg.lower()
    base = {"created": False, "claim": None}

    if not arg or lower == "status":
        if mgr.has_goal():
            return {**base, "output": mgr.status_line()}
        return {
            **base,
            "output": "No scheduled goal set. Start one with /goal <text> or /loop [interval] <prompt>.",
        }

    if lower == "pause":
        state = mgr.pause(reason="user-paused via /loop compatibility alias")
        return {
            **base,
            "output": f"⏸ Goal schedule paused: {state.goal}" if state else "No scheduled goal set.",
        }

    if lower == "resume":
        state = mgr.resume()
        if state is None:
            return {**base, "output": "No scheduled goal to resume."}
        claim = mgr.claim_due_wake()
        return {
            **base,
            "claim": claim,
            "output": f"▶ Goal schedule resumed: {state.goal}",
        }

    if lower in {"stop", "cancel"}:
        state = mgr.stop(reason="stopped via /loop compatibility alias")
        return {**base, "output": "■ Goal schedule stopped." if state else "No scheduled goal set."}

    if lower == "clear":
        had = mgr.has_goal()
        mgr.clear()
        return {**base, "output": "✓ Goal schedule cleared." if had else "No scheduled goal set."}

    if lower in {"help", "--help", "-h"}:
        return {
            **base,
            "output": (
                "/loop is a compatibility alias for scheduled /goal.\n"
                "Usage: /loop [interval] <prompt> [--times N] [--until <condition>]\n"
                "  /loop 5m check the deploy status  → /goal --interval 5m …\n"
                "  /loop 2m poll CI --times 30       → /goal --interval 2m --max-runs 30 …\n"
                "Use /goal --interval 5m <objective> for the canonical form.\n"
                "Controls: /goal status · /goal pause · /goal resume · /goal stop"
            ),
        }

    parsed = parse_loop_args(arg)
    if parsed.get("error"):
        return {**base, "output": f"/loop: {parsed['error']}"}

    until = str(parsed.get("until") or "").strip()
    contract = GoalContract(verification=until) if until else None
    state = mgr.set(
        str(parsed["prompt"]),
        contract=contract,
        # Alias-backed loops get the evidence/recovery controller. Existing
        # persisted goals retain their stored termination mode unchanged.
        termination="judge",
        owner_id=owner_id,
        interval_seconds=parsed.get("interval_seconds"),
        max_runs=int(parsed.get("times") or 0),
        route=route,
    )
    claim = mgr.claim_due_wake()
    cadence = f"every {int(state.interval_seconds)}s" if state.schedule_mode == "interval" else "self-paced"
    cap = f", max {state.max_runs} runs" if state.max_runs else ""
    verify = f", until {until}" if until else ""
    return {
        "created": True,
        "claim": claim,
        "output": (
            f"⊙ Goal set (via /loop compatibility alias, {cadence}{cap}{verify}): {state.goal}\n"
            "Use /goal --interval for new schedules; /loop remains available for this shorthand."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# GoalManager — the orchestration surface CLI + gateway talk to
# ──────────────────────────────────────────────────────────────────────


def _goal_controller_defaults() -> Dict[str, Any]:
    """Resolve the persisted-goal controller settings from ``config.yaml``.

    The default is deliberately legacy-compatible.  A profile must explicitly
    opt into judge-led termination; old rows and profiles keep ``max_turns``.
    This function is defensive because goals run from CLI, gateway workers,
    and tests where the full config loader may not be initialized yet.
    """
    defaults: Dict[str, Any] = {
        "termination": "bounded",
        "max_turns": DEFAULT_MAX_TURNS,
        "duplicate_failure_limit": DEFAULT_DUPLICATE_FAILURE_LIMIT,
        "stall_turns_before_replan": DEFAULT_STALL_TURNS_BEFORE_REPLAN,
        "require_recovery_exhaustion": True,
        "terminal_confirmation": True,
    }
    try:
        from hermes_cli.config import load_config

        configured = (load_config() or {}).get("goals") or {}
    except Exception:
        configured = {}
    if not isinstance(configured, dict):
        return defaults
    if str(configured.get("termination") or "").lower() == "judge":
        defaults["termination"] = "judge"
        defaults["max_turns"] = None
    else:
        try:
            value = int(configured.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS)
            defaults["max_turns"] = value if value > 0 else DEFAULT_MAX_TURNS
        except (TypeError, ValueError):
            pass
    for key, minimum in (
        ("duplicate_failure_limit", 1),
        ("stall_turns_before_replan", 1),
    ):
        try:
            defaults[key] = max(minimum, int(configured.get(key, defaults[key]) or defaults[key]))
        except (TypeError, ValueError):
            pass
    for key in ("require_recovery_exhaustion", "terminal_confirmation"):
        if key in configured:
            defaults[key] = bool(configured[key])
    return defaults


class GoalManager:
    """Per-session goal state + continuation decisions.

    The CLI and gateway each hold one ``GoalManager`` per live session.

    Methods:

    - ``set(goal)`` — start a new standing goal.
    - ``clear()`` — remove the active goal.
    - ``pause()`` / ``resume()`` — explicit user controls.
    - ``status()`` — printable one-liner.
    - ``evaluate_after_turn(last_response)`` — call the judge, update state,
      and return a decision dict the caller uses to drive the next turn.
    - ``next_continuation_prompt()`` — the canonical user-role message to
      feed back into ``run_conversation``.
    """

    def __init__(
        self,
        session_id: str,
        *,
        default_max_turns: int = DEFAULT_MAX_TURNS,
        default_termination: Optional[str] = None,
    ):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self.controller_defaults = _goal_controller_defaults()
        if default_termination is not None:
            self.controller_defaults["termination"] = (
                "judge" if str(default_termination).lower() == "judge" else "bounded"
            )
        # Callers historically supplied only max_turns.  Respect an explicit
        # caller value for bounded mode, while judge mode intentionally stores
        # no turn limit at all.
        if self.controller_defaults["termination"] != "judge":
            self.controller_defaults["max_turns"] = self.default_max_turns
        self._state: Optional[GoalState] = load_goal(session_id)
        # Rows written before reliable identity/claims existed remain valid.
        # Assign their identity lazily so a user can resume an old goal without
        # a one-off database migration.
        if self._state is not None and not self._state.goal_id:
            self._state.goal_id = uuid.uuid4().hex
            save_goal(self.session_id, self._state)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status != "cleared"

    def has_contract(self) -> bool:
        return self._state is not None and self._state.has_contract()

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in {"cleared",}:
            return "No active goal. Set one with /goal <text>."
        turns = (
            f"{s.turns_used} turns (judge-led)"
            if s.termination == "judge"
            else f"{s.turns_used}/{s.max_turns} turns"
        )
        sub = f", {len(s.subgoals)} subgoal{'s' if len(s.subgoals) != 1 else ''}" if s.subgoals else ""
        con = ", contract" if self.has_contract() else ""
        gat = f", {len(s.gates)} gate{'s' if len(s.gates) != 1 else ''}" if s.gates else ""
        meta = f"{turns}{sub}{con}{gat}"
        schedule = ""
        if s.schedule_mode == "interval" and s.interval_seconds:
            schedule = f", every {int(s.interval_seconds)}s"
        if s.max_runs:
            schedule += f", {s.turns_used}/{s.max_runs} scheduled runs"
        if s.status == "active":
            if s.waiting_on_session and _session_waiting(s.waiting_on_session):
                wr = s.waiting_reason or f"session {s.waiting_on_session}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_on_pid and _pid_alive(s.waiting_on_pid):
                wr = s.waiting_reason or f"pid {s.waiting_on_pid}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_until and time.time() < s.waiting_until:
                remaining = int(s.waiting_until - time.time())
                wr = s.waiting_reason or f"{remaining}s"
                return f"⏳ Goal (parked {remaining}s — {wr}, {meta}): {s.goal}"
            return f"⊙ Goal (active{schedule}, {meta}): {s.goal}"
        if s.status == "waiting":
            if s.waiting_on_session:
                tgt = f"session {s.waiting_on_session}"
            elif s.waiting_on_pid:
                tgt = f"pid {s.waiting_on_pid}"
            else:
                remaining = max(0, int(s.waiting_until - time.time()))
                tgt = f"{remaining}s"
            return f"⏳ Goal (waiting on {tgt}, {meta}): {s.goal}"
        if s.status == "awaiting_user":
            label = "owner approval" if s.pending_approval_id else "input"
            return f"✋ Goal (awaiting {label} {s.pending_approval_id[:8]}, {meta}): {s.goal}"
        if s.status == "control_plane_error":
            remaining = max(0, int(s.control_plane_retry_at - time.time()))
            return f"⚠ Goal controller retrying in {remaining}s ({meta}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {meta}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({meta}): {s.goal}"
        if s.status == "blocked":
            return f"⛔ Goal blocked ({meta}) — {s.last_reason or 'external blocker'}: {s.goal}"
        if s.status == "unachievable":
            return f"⊘ Goal unachievable ({meta}) — {s.last_reason or 'no viable path'}: {s.goal}"
        if s.status == "stopped":
            return f"■ Goal stopped ({meta}): {s.goal}"
        return f"Goal ({s.status}, {meta}): {s.goal}"

    # --- mutation -----------------------------------------------------

    def set(
        self,
        goal: str,
        *,
        max_turns: Optional[int] = None,
        contract: Optional[GoalContract] = None,
        termination: Optional[str] = None,
        approval_policy: str = "automatic",
        owner_id: str = "",
        interval_seconds: Optional[float] = None,
        max_runs: int = 0,
        route: Optional[Dict[str, str]] = None,
    ) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        now = time.time()
        interval = max(0.0, float(interval_seconds or 0.0))
        chosen_termination = (
            "judge"
            if str(termination or self.controller_defaults["termination"]).lower() == "judge"
            else "bounded"
        )
        state = GoalState(
            goal=goal,
            status="active",
            goal_id=uuid.uuid4().hex,
            turns_used=0,
            termination=chosen_termination,
            max_turns=(
                None
                if chosen_termination == "judge"
                else (int(max_turns) if max_turns else self.default_max_turns)
            ),
            duplicate_failure_limit=int(self.controller_defaults["duplicate_failure_limit"]),
            stall_turns_before_replan=int(self.controller_defaults["stall_turns_before_replan"]),
            require_recovery_exhaustion=bool(self.controller_defaults["require_recovery_exhaustion"]),
            terminal_confirmation=bool(self.controller_defaults["terminal_confirmation"]),
            created_at=now,
            last_turn_at=0.0,
            contract=contract if contract is not None else GoalContract(),
            approval_policy="owner" if approval_policy == "owner" else "automatic",
            owner_id=str(owner_id or ""),
            schedule_mode="interval" if interval else "self_paced",
            interval_seconds=interval,
            max_runs=max(0, int(max_runs or 0)),
            next_wake_at=now,
            initial_kickoff_pending=True,
            route={str(k): str(v) for k, v in (route or {}).items() if v is not None and str(v)},
        )
        self._state = state
        save_goal(self.session_id, state)
        return state

    def set_contract(self, contract: GoalContract) -> Optional[GoalState]:
        """Attach or replace the completion contract on the active goal.

        Returns the updated state, or None when there is no goal to attach to.
        """
        if self._state is None:
            return None
        self._state.contract = contract or GoalContract()
        save_goal(self.session_id, self._state)
        return self._state

    def pause(self, reason: str = "user-paused") -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        # A wait barrier is meaningless once paused — drop it.
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        self._state.wake_claim_id = ""
        self._state.wake_claimed_at = 0.0
        self._state.initial_kickoff_pending = False
        save_goal(self.session_id, self._state)
        return self._state

    def resume(self, *, reset_budget: Optional[bool] = None) -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        # Resuming starts fresh — clear any stale barrier.
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        self._state.pending_approval_id = ""
        self._state.pending_approval_reason = ""
        self._state.pending_approval_at = 0.0
        self._state.wake_claim_id = ""
        self._state.wake_claimed_at = 0.0
        self._state.next_wake_at = time.time()
        self._state.initial_kickoff_pending = False
        if reset_budget is None:
            reset_budget = self._state.termination != "judge"
        if reset_budget:
            self._state.turns_used = 0
        save_goal(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        save_goal(self.session_id, self._state)
        self._state = None

    def stop(self, reason: str = "stopped by owner") -> Optional[GoalState]:
        """Explicit owner stop, distinct from success and ordinary clearing."""
        if self._state is None:
            return None
        self._state.status = "stopped"
        self._state.last_reason = reason
        self._state.wake_claim_id = ""
        self._state.wake_claimed_at = 0.0
        save_goal(self.session_id, self._state)
        return self._state

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_verdict = "done"
        self._state.last_reason = reason
        save_goal(self.session_id, self._state)

    def set_approval_policy(self, policy: str) -> GoalState:
        if self._state is None:
            raise RuntimeError("no active goal")
        normalized = str(policy or "").strip().lower()
        if normalized not in {"automatic", "owner"}:
            raise ValueError("approval policy must be 'automatic' or 'owner'")
        self._state.approval_policy = normalized
        save_goal(self.session_id, self._state)
        return self._state

    def approve_completion(self, approval_id: str, *, actor_id: str = "") -> bool:
        state = self._state
        if state is None or state.status != "awaiting_user":
            return False
        if not approval_id or approval_id != state.pending_approval_id:
            return False
        if state.owner_id and actor_id and actor_id != state.owner_id:
            return False
        state.status = "done"
        state.last_verdict = "done"
        state.last_reason = state.pending_approval_reason or state.last_reason
        state.pending_approval_id = ""
        state.pending_approval_reason = ""
        state.pending_approval_at = 0.0
        save_goal(self.session_id, state)
        return True

    def deny_completion(self, approval_id: str, reason: str = "", *, actor_id: str = "") -> bool:
        state = self._state
        if state is None or state.status != "awaiting_user":
            return False
        if not approval_id or approval_id != state.pending_approval_id:
            return False
        if state.owner_id and actor_id and actor_id != state.owner_id:
            return False
        state.status = "active"
        state.last_verdict = "continue"
        state.last_reason = (reason or "owner rejected completion").strip()
        state.pending_approval_id = ""
        state.pending_approval_reason = ""
        state.pending_approval_at = 0.0
        state.next_wake_at = time.time()
        state.wake_claim_id = ""
        state.wake_claimed_at = 0.0
        save_goal(self.session_id, state)
        return True

    def resume_from_user_input(self, text: str, *, actor_id: str = "") -> bool:
        """Resume a parked ``needs_input`` goal when its owner supplies input.

        Owner-approved completion deliberately remains capability-bound to
        ``/goal approve <id>``.  This helper is only for an irreducible input
        dependency, where the next ordinary owner message is the evidence the
        agent needs to continue.  The supplied text is recorded as
        user-confirmed evidence in sanitized, bounded form and then evaluated
        by the normal controller after the response to that very turn.
        """
        state = self._state
        cleaned = str(text or "").strip()
        if (
            state is None
            or state.status != "awaiting_user"
            or state.pending_approval_id
            or not cleaned
        ):
            return False
        if state.owner_id and actor_id and actor_id != state.owner_id:
            return False
        state.status = "active"
        state.last_verdict = "continue"
        state.last_reason = "owner supplied requested input; verify and continue within the original scope"
        state.last_strategy_constraint = (
            "use the supplied input only for the original authorized outcome; "
            "verify it before any external side effect"
        )
        state.input_notification_sent = False
        state.next_wake_at = time.time()
        self._clear_wake_claim()
        self.record_observed_evidence(
            f"user-input-{state.turns_used + 1}",
            cleaned,
            provenance="user_confirmed",
        )
        # record_observed_evidence persists the same state; keep this save
        # explicit for older SessionDB implementations and future changes.
        save_goal(self.session_id, state)
        return True

    # --- /subgoal user controls ---------------------------------------

    def add_subgoal(self, text: str) -> str:
        """Append a user-added criterion to the active goal. Requires
        ``has_goal()``; raises ``RuntimeError`` otherwise.

        Returns the cleaned text so the caller can show it back to the user.
        """
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        text = (text or "").strip()
        if not text:
            raise ValueError("subgoal text is empty")
        self._state.subgoals.append(text)
        save_goal(self.session_id, self._state)
        return text

    def remove_subgoal(self, index_1based: int) -> str:
        """Remove a subgoal by 1-based index. Returns the removed text."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.subgoals):
            raise IndexError(
                f"index out of range (1..{len(self._state.subgoals)})"
            )
        removed = self._state.subgoals.pop(idx)
        save_goal(self.session_id, self._state)
        return removed

    def clear_subgoals(self) -> int:
        """Wipe all subgoals. Returns the previous count."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        prev = len(self._state.subgoals)
        self._state.subgoals = []
        save_goal(self.session_id, self._state)
        return prev

    def render_subgoals(self) -> str:
        """Public helper for the /subgoal slash command."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.subgoals:
            return "(no subgoals — use /subgoal <text> to add criteria)"
        return self._state.render_subgoals_block()

    # --- /goal gate quality gates ---------------------------------------

    def add_gate(
        self,
        command: str,
        *,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> GoalGate:
        """Append a quality-gate command to the active goal.

        Requires ``has_goal()``; raises ``RuntimeError`` otherwise. Returns
        the created gate so callers can echo it back.
        """
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        command = (command or "").strip()
        if not command:
            raise ValueError("gate command is empty")
        gate = GoalGate(
            command=command,
            timeout_seconds=int(timeout_seconds) if timeout_seconds else DEFAULT_GATE_TIMEOUT_SECONDS,
            max_retries=int(max_retries) if max_retries else DEFAULT_GATE_MAX_RETRIES,
        )
        self._state.gates.append(gate)
        save_goal(self.session_id, self._state)
        return gate

    def remove_gate(self, index_1based: int) -> str:
        """Remove a gate by 1-based index. Returns the removed command."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.gates):
            raise IndexError(f"index out of range (1..{len(self._state.gates)})")
        removed = self._state.gates.pop(idx)
        save_goal(self.session_id, self._state)
        return removed.command

    def clear_gates(self) -> int:
        """Remove all gates. Returns the previous count."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        prev = len(self._state.gates)
        self._state.gates = []
        save_goal(self.session_id, self._state)
        return prev

    def render_gates(self) -> str:
        """Public helper for the /goal gate slash command."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.gates:
            return "(no quality gates — use /goal gate add <command> to require one)"
        lines = []
        for i, g in enumerate(self._state.gates, start=1):
            status = ""
            if g.last_exit_code is not None:
                status = " ✓ passing" if g.last_exit_code == 0 else (
                    f" ✗ failing (exit {g.last_exit_code}, attempt {g.attempts}/{g.max_retries})"
                )
            lines.append(f"- {i}. $ {g.command}{status}")
        return "\n".join(lines)

    def _check_gates(self) -> Optional[Dict[str, Any]]:
        """Run quality gates in order; return a decision dict on failure.

        Returns ``None`` when there are no gates or every gate passes —
        the caller then proceeds to the LLM judge. On the first failing
        gate, returns a full ``evaluate_after_turn``-shaped decision dict:
        either a continuation carrying the gate's output (attempts left)
        or an auto-pause (retries exhausted).

        An unchanged workspace since the last failure of the same gate is
        NOT re-run — the recorded failure is replayed and the attempt count
        advances, so a stalled agent can't spin re-running an identical red
        suite (mirrors Prime-Agent's unchanged-gate rule).
        """
        state = self._state
        if state is None or not state.gates:
            return None

        fingerprint = workspace_fingerprint()
        for gate in state.gates:
            unchanged = (
                bool(fingerprint)
                and gate.last_exit_code not in (None, 0)
                and gate.last_failed_fingerprint == fingerprint
            )
            if unchanged:
                passed, exit_code, tail = False, int(gate.last_exit_code or -1), gate.last_output_tail
            else:
                passed, exit_code, tail = run_gate(gate)
            gate.last_exit_code = exit_code
            gate.last_output_tail = tail
            if passed:
                gate.attempts = 0
                gate.last_failed_fingerprint = ""
                continue

            gate.attempts += 1
            gate.last_failed_fingerprint = fingerprint
            skipped_note = " (workspace unchanged since last failure — not re-run)" if unchanged else ""

            if gate.attempts > gate.max_retries:
                state.status = "paused"
                state.paused_reason = (
                    f"quality gate exhausted {gate.attempts - 1} retries: $ {gate.command}"
                )
                save_goal(self.session_id, state)
                return {
                    "status": "paused",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "gate_failed",
                    "reason": f"gate exhausted retries: $ {gate.command}",
                    "message": (
                        f"⏸ Goal paused — quality gate still failing after "
                        f"{gate.max_retries} retries: $ {gate.command} "
                        f"(exit {exit_code}). Fix it manually or /goal gate remove it, "
                        f"then /goal resume."
                    ),
                }

            save_goal(self.session_id, state)
            prompt = CONTINUATION_PROMPT_GATE_FAILED_TEMPLATE.format(
                goal=state.goal,
                command=gate.command,
                exit_code=exit_code,
                attempt=gate.attempts,
                max_retries=gate.max_retries,
                output=tail or "(no output)",
            )
            return {
                "status": "active",
                "should_continue": True,
                "continuation_prompt": prompt,
                "verdict": "gate_failed",
                "reason": f"gate failed (exit {exit_code}): $ {gate.command}",
                "message": (
                    f"✗ Quality gate failed ({state.turns_used}/{state.max_turns} turns, "
                    f"attempt {gate.attempts}/{gate.max_retries}){skipped_note}: $ {gate.command}"
                ),
            }

        save_goal(self.session_id, state)
        return None

    # --- durable wake scheduling --------------------------------------

    def _record_evidence(self, verdict: str, reason: str, response: str) -> None:
        """Store a bounded, redacted completion/audit receipt."""
        state = self._state
        if state is None:
            return
        state.evidence_receipts.append(
            {
                "at": time.time(),
                "verdict": str(verdict or "continue"),
                "reason": _receipt_excerpt(reason, 280),
                "evidence": _receipt_excerpt(response),
            }
        )
        state.evidence_receipts = state.evidence_receipts[-MAX_EVIDENCE_RECEIPTS:]

    def _record_progress(
        self,
        decision: Dict[str, Any],
        response: str,
        *,
        strategy_family: str = "current approach",
        provenance: str = "agent_claim",
    ) -> Dict[str, Any]:
        """Append a sanitized, bounded controller ledger entry.

        The response hash is a conservative fallback action fingerprint when a
        caller cannot provide tool-level receipts.  It is only used to stop
        identical stalled loops; it never authorizes a side effect or proves
        completion.  Tool/gate integrations can add stronger provenance to
        the same ledger entry over time without changing persistence format.
        """
        state = self._state
        assert state is not None
        active_recovery = next(
            (
                path for path in reversed(state.recovery_paths)
                if isinstance(path, dict) and path.get("state") == "in_progress"
            ),
            None,
        )
        if active_recovery is not None:
            strategy_family = str(active_recovery.get("family") or strategy_family)
        normalized = re.sub(r"\s+", " ", str(response or "").strip().lower())
        action_fingerprint = hashlib.sha256(
            normalized[:4000].encode("utf-8", "replace")
        ).hexdigest()[:24]
        progress = str(decision.get("progress") or "stalled")
        if progress == "stalled":
            if action_fingerprint == state.last_action_fingerprint:
                state.duplicate_failures += 1
            else:
                state.duplicate_failures = 1
        elif progress == "advanced":
            state.duplicate_failures = 0
        else:
            state.duplicate_failures = 0
        state.last_action_fingerprint = action_fingerprint
        entry_id = f"p{state.turns_used}-{uuid.uuid4().hex[:8]}"
        entry = {
            "id": entry_id,
            "at": time.time(),
            "strategy_family": _receipt_excerpt(strategy_family, 160),
            "action_fingerprint": action_fingerprint,
            "progress": progress,
            "verdict": str(decision.get("verdict") or "continue"),
            "blocker_class": str(decision.get("blocker_class") or "ambiguity"),
            "reason": _receipt_excerpt(str(decision.get("reason") or ""), 500),
            "evidence": _receipt_excerpt(response, 700),
            "provenance": provenance,
            "remaining_hypotheses": [
                _receipt_excerpt(str(x), 180)
                for x in (decision.get("untried_strategy_families") or [])[:6]
            ],
            "next_step": _receipt_excerpt(str(decision.get("next_strategy_constraint") or ""), 500),
        }
        state.progress_ledger.append(entry)
        state.progress_ledger = state.progress_ledger[-MAX_PROGRESS_LEDGER_ENTRIES:]
        return entry

    def record_observed_evidence(
        self,
        reference: str,
        evidence: str,
        *,
        provenance: str = "tool_observed",
    ) -> bool:
        """Attach independently observed evidence for a terminal verifier.

        This is intentionally an internal controller method rather than a new
        model tool: existing gate/worker integrations can call it without
        growing every conversation's tool schema.  Only known provenance
        labels are persisted.
        """
        state = self._state
        if state is None:
            return False
        if provenance not in {"deterministic_gate", "tool_observed", "user_confirmed"}:
            return False
        state.progress_ledger.append({
            "id": _receipt_excerpt(str(reference or uuid.uuid4().hex), 180),
            "at": time.time(),
            "strategy_family": "evidence",
            "action_fingerprint": "",
            "progress": "advanced",
            "verdict": "evidence",
            "blocker_class": "",
            "reason": "observed evidence",
            "evidence": _receipt_excerpt(evidence, 700),
            "provenance": provenance,
            "remaining_hypotheses": [],
            "next_step": "",
        })
        state.progress_ledger = state.progress_ledger[-MAX_PROGRESS_LEDGER_ENTRIES:]
        save_goal(self.session_id, state)
        return True

    def _park_control_plane_error(self, reason: str) -> Dict[str, Any]:
        state = self._state
        assert state is not None
        state.status = "control_plane_error"
        state.control_plane_failures += 1
        state.control_plane_retry_at = time.time() + DEFAULT_CONTROL_PLANE_RETRY_SECONDS
        state.next_wake_at = state.control_plane_retry_at
        state.last_verdict = "control_plane_error"
        state.last_reason = _receipt_excerpt(reason, 600)
        self._clear_wake_claim()
        save_goal(self.session_id, state)
        return {
            "status": "control_plane_error",
            "should_continue": False,
            "continuation_prompt": None,
            "verdict": "control_plane_error",
            "reason": state.last_reason,
            "message": "⚠ Goal controller is unavailable; parked for an automatic health retry.",
        }

    def _record_recovery_paths(self, strategies: List[Dict[str, str]]) -> None:
        state = self._state
        assert state is not None
        existing = {
            str(path.get("family") or "").strip().lower()
            for path in state.recovery_paths
            if isinstance(path, dict)
        }
        for strategy in strategies:
            family = str(strategy.get("family") or "").strip()
            if not family or family.lower() in existing:
                continue
            state.recovery_paths.append({
                "family": _receipt_excerpt(family, 160),
                "next_step": _receipt_excerpt(str(strategy.get("next_step") or ""), 420),
                "why_safe": _receipt_excerpt(str(strategy.get("why_safe") or ""), 240),
                "state": "untried",
                "at": time.time(),
            })
            existing.add(family.lower())
        state.recovery_paths = state.recovery_paths[-16:]

    def _ensure_recovery_floor(self, decision: Dict[str, Any]) -> None:
        """Add safe diagnostic routes until a terminal proposal has options.

        The controller never invents a privileged workaround.  These are
        intentionally read-only fallback families that every agent can either
        carry out or falsify with evidence.  A recovery coach's task-specific
        suggestions stay first in the queue.
        """
        state = self._state
        assert state is not None
        existing = {
            str(path.get("family") or "").strip().lower()
            for path in state.recovery_paths
            if isinstance(path, dict)
        }
        suggested = [
            {
                "family": str(family),
                "next_step": f"try the {family} route",
                "why_safe": "identified by the controller as an authorized recovery path",
            }
            for family in (decision.get("untried_strategy_families") or [])
            if str(family).strip() and str(family).strip().lower() not in existing
        ]
        if suggested:
            self._record_recovery_paths(suggested)
            existing.update(str(item["family"]).lower() for item in suggested)
        if len(existing) >= MIN_RECOVERY_STRATEGY_FAMILIES:
            return
        self._record_recovery_paths([
            dict(strategy)
            for strategy in _BASELINE_RECOVERY_STRATEGIES
            if strategy["family"].lower() not in existing
        ])

    def _has_independent_terminal_evidence(self) -> bool:
        """Whether the ledger contains more than an agent's own assertion."""
        state = self._state
        assert state is not None
        return any(
            isinstance(entry, dict)
            and entry.get("provenance") in {
                "deterministic_gate", "tool_observed", "user_confirmed",
            }
            and str(entry.get("evidence") or "").strip()
            for entry in state.progress_ledger
        )

    def _recovery_is_exhausted(self) -> bool:
        """Require diverse recovery unless independent evidence proves it moot."""
        state = self._state
        assert state is not None
        paths = [path for path in state.recovery_paths if isinstance(path, dict)]
        remaining = any(path.get("state") in {"untried", "in_progress"} for path in paths)
        tried_families = {
            str(path.get("family") or "").strip().lower()
            for path in paths
            if path.get("state") == "tried" and str(path.get("family") or "").strip()
        }
        if remaining:
            return False
        if len(tried_families) >= MIN_RECOVERY_STRATEGY_FAMILIES:
            return True
        # Some goals objectively have fewer than three permitted routes (for
        # example a deterministic policy denial).  Do not trust a model claim
        # for that exception: require independently observed evidence.
        return bool(tried_families) and self._has_independent_terminal_evidence()

    def _judge_led_prompt(self, reason: str, constraint: str, *, replan: bool = False) -> str:
        state = self._state
        assert state is not None
        if replan:
            paths = [
                str(p.get("family") or "")
                for p in state.recovery_paths
                if isinstance(p, dict) and p.get("state") == "untried"
            ]
            rendered = "; ".join(paths[:4]) or constraint
            return JUDGE_LED_REPLAN_TEMPLATE.format(
                goal=state.goal,
                reason=_receipt_excerpt(reason, 700),
                strategies=rendered,
            )
        return JUDGE_LED_CONTINUATION_TEMPLATE.format(
            goal=state.goal,
            reason=_receipt_excerpt(reason, 700),
            constraint=_receipt_excerpt(constraint, 700),
        )

    def _clear_wake_claim(self) -> None:
        state = self._state
        if state is None:
            return
        state.wake_claim_id = ""
        state.wake_claimed_at = 0.0

    def _arm_next_wake(self, *, now: Optional[float] = None, immediate: bool = False) -> None:
        """Persist the next wake deadline without injecting a turn itself."""
        state = self._state
        if state is None:
            return
        current = time.time() if now is None else float(now)
        if immediate or state.schedule_mode != "interval" or state.interval_seconds <= 0:
            state.next_wake_at = current
        else:
            state.next_wake_at = current + state.interval_seconds
        state.wake_generation += 1

    def has_due_wake(self, now: Optional[float] = None) -> bool:
        """Whether the durable scheduler may try to claim this goal's wake."""
        state = self._state
        if state is None or state.status not in {"active", "waiting", "control_plane_error"}:
            return False
        current = time.time() if now is None else float(now)
        if state.wake_claim_id and current - state.wake_claimed_at < DEFAULT_WAKE_LEASE_SECONDS:
            return False
        return current >= state.next_wake_at

    @staticmethod
    def _wait_barrier_is_pending(state: GoalState, now: float) -> bool:
        """Check a persisted wait barrier without mutating or saving state."""
        if state.waiting_on_session is not None:
            return _session_waiting(state.waiting_on_session)
        if state.waiting_on_pid is not None:
            return _pid_alive(state.waiting_on_pid)
        return bool(state.waiting_until and now < state.waiting_until)

    @staticmethod
    def _clear_wait_barrier_state(state: GoalState) -> None:
        state.waiting_on_pid = None
        state.waiting_on_session = None
        state.waiting_until = 0.0
        state.waiting_reason = None
        state.waiting_since = 0.0

    def claim_due_wake(self, now: Optional[float] = None) -> Optional[Dict[str, str]]:
        """Claim one due wake and return its continuation data.

        A claim is an optimistic compare-and-set over the persisted goal row:
        the due state that was read must still be byte-for-byte current when
        the lease is recorded.  That gives concurrent CLI/gateway/TUI
        schedulers one winner, while the short lease still makes a crashed
        winner recoverable.  Older or test-double SessionDBs without CAS keep
        the prior best-effort behavior.
        """
        current = time.time() if now is None else float(now)
        db = _get_session_db()
        key = _meta_key(self.session_id)
        supports_cas = bool(db is not None and callable(getattr(db, "compare_and_set_meta", None)))

        # A CAS miss means another scheduler changed the lifecycle. Reload it
        # and re-evaluate rather than issuing a duplicate continuation.
        for _attempt in range(3):
            expected: Optional[str] = None
            if supports_cas:
                try:
                    expected = db.get_meta(key)
                    if not expected:
                        return None
                    self._state = GoalState.from_json(expected)
                except Exception as exc:
                    logger.debug("GoalManager: could not reload wake state: %s", exc)
                    return None

            state = self._state
            if state is None or not self.has_due_wake(current):
                return None

            def persist() -> bool:
                if supports_cas:
                    try:
                        return bool(db.compare_and_set_meta(key, expected or "", state.to_json()))
                    except Exception as exc:
                        logger.debug("GoalManager: wake claim CAS failed: %s", exc)
                        return False
                save_goal(self.session_id, state)
                return True

            if state.status == "control_plane_error":
                # Health probes are controller calls, not agent turns.  A
                # healthy probe releases the parked goal; an unhealthy one
                # simply moves the next probe forward without spending budget.
                probe = judge_goal_with_ledger(
                    state.goal,
                    "Controller health probe; do not evaluate goal completion.",
                    contract=state.contract if state.has_contract() else None,
                    ledger=state.progress_ledger,
                )
                if probe.get("verdict") == "control_plane_error":
                    state.control_plane_failures += 1
                    state.control_plane_retry_at = current + DEFAULT_CONTROL_PLANE_RETRY_SECONDS
                    state.next_wake_at = state.control_plane_retry_at
                    if persist():
                        return None
                    continue
                state.status = "active"
                state.control_plane_failures = 0
                state.control_plane_retry_at = 0.0
                state.last_reason = "goal controller health probe succeeded"
                self._arm_next_wake(now=current, immediate=True)

            if state.status == "waiting":
                if self._wait_barrier_is_pending(state, current):
                    state.next_wake_at = current + DEFAULT_WAIT_POLL_SECONDS
                    if persist():
                        return None
                    continue
                # The barrier is now satisfied. Its next continuation is
                # owned by the scheduler, not an incidental user turn.
                self._clear_wait_barrier_state(state)
                state.status = "active"
                self._arm_next_wake(now=current, immediate=True)

            if state.max_runs and state.turns_used >= state.max_runs:
                state.status = "paused"
                state.paused_reason = f"scheduled run cap reached ({state.turns_used}/{state.max_runs})"
                if persist():
                    return None
                continue

            claim_id = uuid.uuid4().hex
            state.wake_claim_id = claim_id
            state.wake_claimed_at = current
            state.wake_generation += 1
            if state.initial_kickoff_pending:
                prompt = state.goal
                state.initial_kickoff_pending = False
            else:
                prompt = self.next_continuation_prompt()
            if not prompt:
                self._clear_wake_claim()
                if persist():
                    return None
                continue
            if persist():
                return {"claim_id": claim_id, "prompt": prompt}

        return None

    def abandon_wake(self, claim_id: str = "") -> bool:
        """Release a claimed wake whose surface injection failed."""
        state = self._state
        if state is None or not state.wake_claim_id:
            return False
        if claim_id and claim_id != state.wake_claim_id:
            return False
        self._clear_wake_claim()
        state.next_wake_at = time.time()
        save_goal(self.session_id, state)
        return True

    # --- /goal wait barrier -------------------------------------------

    def wait_on(self, pid: int, reason: str = "") -> GoalState:
        """Park the goal loop on a background process PID.

        While the PID is alive, ``evaluate_after_turn`` returns
        ``should_continue=False`` without burning a turn or calling the
        judge — the loop quiesces instead of re-poking the agent into busy
        work. The barrier auto-clears when the process exits. Requires an
        active goal. For a process with a watch_patterns/notify_on_complete
        trigger, prefer ``wait_on_session`` so a mid-run trigger (not just
        exit) releases the barrier.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        pid = int(pid)
        if pid <= 0:
            raise ValueError("pid must be a positive integer")
        self._state.waiting_on_pid = pid
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        self._state.status = "waiting"
        self._clear_wake_claim()
        self._state.next_wake_at = time.time() + DEFAULT_WAIT_POLL_SECONDS
        save_goal(self.session_id, self._state)
        return self._state

    def wait_on_session(self, session_id: str, reason: str = "") -> GoalState:
        """Park the goal loop on a process_registry session's OWN trigger.

        Unlike ``wait_on`` (which releases only on PID exit), this releases
        when the session's trigger fires: it exits, OR — if it was started
        with ``watch_patterns`` — its pattern matches. This is the right
        barrier for a long-lived watcher/server/poller that signals mid-run
        and may never exit. Requires an active goal.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._state.waiting_on_session = session_id
        self._state.waiting_on_pid = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        self._state.status = "waiting"
        self._clear_wake_claim()
        self._state.next_wake_at = time.time() + DEFAULT_WAIT_POLL_SECONDS
        save_goal(self.session_id, self._state)
        return self._state

    def wait_for_seconds(self, seconds: int, reason: str = "") -> GoalState:
        """Park the goal loop until ``seconds`` from now have elapsed.

        Time-based counterpart to ``wait_on`` — for backoff / cooldown waits
        where there's no process to track (e.g. the agent is rate-limited).
        The barrier auto-clears once the deadline passes. Requires an active
        goal.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        seconds = int(seconds)
        if seconds <= 0:
            raise ValueError("seconds must be a positive integer")
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = time.time() + seconds
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        self._state.status = "waiting"
        self._clear_wake_claim()
        self._state.next_wake_at = self._state.waiting_until
        save_goal(self.session_id, self._state)
        return self._state

    def stop_waiting(self) -> bool:
        """Clear any active wait barrier (pid / session / time). Returns True
        if one was cleared."""
        if self._state is None:
            return False
        if (
            self._state.waiting_on_pid is None
            and self._state.waiting_on_session is None
            and not self._state.waiting_until
        ):
            return False
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        if self._state.status == "waiting":
            self._state.status = "active"
        self._clear_wake_claim()
        self._state.next_wake_at = time.time()
        save_goal(self.session_id, self._state)
        return True

    def is_waiting(self) -> bool:
        """True iff a barrier is set AND not yet satisfied.

        Session barrier: active until the process exits or its watch-pattern
        trigger fires. Pid barrier: active while the process is alive. Time
        barrier: active until the deadline passes. Side effect: a satisfied
        barrier is cleared here (lazy auto-clear) so the next evaluation
        resumes normal judging.
        """
        s = self._state
        if s is None:
            return False
        if s.waiting_on_session is not None:
            if _session_waiting(s.waiting_on_session):
                return True
            self.stop_waiting()  # session exited or trigger fired
            return False
        if s.waiting_on_pid is not None:
            if _pid_alive(s.waiting_on_pid):
                return True
            self.stop_waiting()  # process gone
            return False
        if s.waiting_until:
            if time.time() < s.waiting_until:
                return True
            self.stop_waiting()  # deadline passed
            return False
        return False

    def _mark_inflight_recovery_tried(self) -> None:
        """Close the recovery path handed to the immediately preceding turn."""
        state = self._state
        assert state is not None
        for path in reversed(state.recovery_paths):
            if isinstance(path, dict) and path.get("state") == "in_progress":
                path["state"] = "tried"
                path["completed_at"] = time.time()
                return

    def _choose_recovery_path(self) -> Optional[Dict[str, Any]]:
        state = self._state
        assert state is not None
        for path in state.recovery_paths:
            if isinstance(path, dict) and path.get("state") == "untried":
                path["state"] = "in_progress"
                path["started_at"] = time.time()
                return path
        return None

    def _replan_with_recovery(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the recovery coach for a different safe path and continue."""
        state = self._state
        assert state is not None
        coach = recovery_coach(state, decision)
        if coach.get("control_plane_error"):
            return self._park_control_plane_error(str(coach["control_plane_error"]))
        self._record_recovery_paths(coach.get("strategies") or [])
        # A stalled controller still needs a concrete alternative even if the
        # recovery model returned an empty/malformed strategy list.
        if not any(
            isinstance(path, dict) and path.get("state") == "untried"
            for path in state.recovery_paths
        ):
            self._ensure_recovery_floor(decision)
        selected = self._choose_recovery_path()
        constraint = (
            str(selected.get("next_step") or "")
            if selected else str(decision.get("next_strategy_constraint") or "")
        ) or "change strategy family before taking another action"
        state.last_strategy_constraint = _receipt_excerpt(constraint, 700)
        state.last_verdict = "replan"
        state.last_reason = _receipt_excerpt(str(decision.get("reason") or ""), 700)
        self._arm_next_wake()
        save_goal(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self._judge_led_prompt(
                state.last_reason,
                state.last_strategy_constraint,
                replan=True,
            ),
            "verdict": "replan",
            "reason": state.last_reason,
            "message": "↻ Goal stalled; switching to a different safe recovery strategy.",
        }

    def _terminal_or_replan(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Verify terminal decisions and turn rejected ones into recovery."""
        state = self._state
        assert state is not None
        verifier = verify_terminal_goal_decision(state, decision)
        if verifier.get("control_plane_error"):
            return self._park_control_plane_error(str(verifier["control_plane_error"]))
        if not verifier.get("accept"):
            untried = verifier.get("untried_strategy_families") or []
            if untried:
                self._record_recovery_paths([
                    {"family": item, "next_step": f"try the {item} route", "why_safe": "terminal verifier identified an authorized alternative"}
                    for item in untried
                ])
            revised = dict(decision)
            revised["reason"] = verifier.get("reason") or decision.get("reason")
            revised["untried_strategy_families"] = untried
            revised["next_strategy_constraint"] = "address the terminal verifier's missing evidence or alternate path"
            return self._replan_with_recovery(revised)

        verdict = decision["verdict"]
        state.last_verdict = verdict
        state.last_reason = _receipt_excerpt(str(decision.get("reason") or ""), 700)
        # Cross-task learning is deliberately best-effort and quarantined. A
        # persistence failure here must never change the goal's terminal
        # result, and the candidate writer strips secrets/transient details.
        try:
            from hermes_cli.goal_learning import record_terminal_retrospective

            record_terminal_retrospective(state, verdict)
        except Exception:
            logger.debug("goal learning candidate skipped", exc_info=True)
        if verdict == "achieved":
            if state.approval_policy == "owner":
                state.status = "awaiting_user"
                state.pending_approval_id = uuid.uuid4().hex
                state.pending_approval_reason = state.last_reason
                state.pending_approval_at = time.time()
                save_goal(self.session_id, state)
                return {
                    "status": "awaiting_user", "should_continue": False, "continuation_prompt": None,
                    "verdict": "achieved", "reason": state.last_reason,
                    "approval_id": state.pending_approval_id,
                    "message": (
                        "✋ Goal completion is ready for owner approval. "
                        f"Use /goal approve {state.pending_approval_id} to mark it done, "
                        f"or /goal reject {state.pending_approval_id} <reason> to continue."
                    ),
                }
            state.status = "done"  # preserve the long-standing persisted success spelling
            save_goal(self.session_id, state)
            return {
                "status": "done", "should_continue": False, "continuation_prompt": None,
                "verdict": "achieved", "reason": state.last_reason,
                "message": f"✓ Goal achieved: {state.last_reason}",
            }
        if verdict == "needs_input":
            state.status = "awaiting_user"
            state.pending_approval_id = ""
            state.pending_approval_reason = state.last_reason
            state.pending_approval_at = time.time()
            should_notify = not state.input_notification_sent
            state.input_notification_sent = True
            save_goal(self.session_id, state)
            return {
                "status": "awaiting_user", "should_continue": False, "continuation_prompt": None,
                "verdict": "needs_input", "reason": state.last_reason,
                "message": (
                    f"✋ Goal awaiting one specific user action: {state.last_reason}"
                    if should_notify else ""
                ),
            }
        if verdict == "not_achievable":
            state.status = "unachievable"
            save_goal(self.session_id, state)
            return {
                "status": "unachievable", "should_continue": False, "continuation_prompt": None,
                "verdict": "not_achievable", "reason": state.last_reason,
                "message": f"⊘ Goal cannot be achieved within the authorized scope: {state.last_reason}",
            }
        # policy_stop is terminal but intentionally distinct from success.
        state.status = "stopped"
        save_goal(self.session_id, state)
        return {
            "status": "stopped", "should_continue": False, "continuation_prompt": None,
            "verdict": "policy_stop", "reason": state.last_reason,
            "message": f"■ Goal stopped by policy/authority boundary: {state.last_reason}",
        }

    def _evaluate_judge_led_after_turn(
        self,
        last_response: str,
        *,
        background_processes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Judge-led lifecycle: unbounded productive work, bounded behavior."""
        state = self._state
        assert state is not None
        self._mark_inflight_recovery_tried()

        # Quality gates are deterministic evidence.  Reuse their existing
        # runner, but never translate a judge-led goal's gate retry count into
        # a successful or sticky blocked outcome.
        gate_decision = self._check_gates()
        if gate_decision is not None:
            if gate_decision.get("status") == "paused":
                state.status = "active"
                state.paused_reason = None
                gate_decision = dict(gate_decision)
                gate_decision.update({
                    "status": "active",
                    "should_continue": True,
                    "verdict": "replan",
                    "message": "↻ A required quality gate still fails; changing repair strategy.",
                })
            self.record_observed_evidence(
                f"gate-turn-{state.turns_used}",
                str(gate_decision.get("reason") or "quality gate result"),
                provenance="deterministic_gate",
            )
            state.last_verdict = "replan"
            state.last_reason = str(gate_decision.get("reason") or "quality gate failed")
            state.last_strategy_constraint = "repair the failing quality gate with a different diagnosis"
            self._arm_next_wake()
            save_goal(self.session_id, state)
            return {
                "status": "active", "should_continue": True,
                "continuation_prompt": self._judge_led_prompt(
                    state.last_reason, state.last_strategy_constraint, replan=True,
                ),
                "verdict": "replan", "reason": state.last_reason,
                "message": "↻ Quality gate evidence requires a different repair strategy.",
            }

        decision = judge_goal_with_ledger(
            state.goal,
            last_response,
            contract=state.contract if state.has_contract() else None,
            ledger=state.progress_ledger,
            background_processes=background_processes,
        )
        if decision["verdict"] == "control_plane_error":
            return self._park_control_plane_error(str(decision.get("reason") or "goal judge unavailable"))

        state.last_verdict = str(decision["verdict"])
        state.last_reason = str(decision.get("reason") or "")
        self._record_evidence(state.last_verdict, state.last_reason, last_response)
        self._record_progress(decision, last_response)
        if decision.get("progress") == "advanced":
            state.stalled_turns = 0
        else:
            state.stalled_turns += 1

        if decision["verdict"] == "wait" and decision.get("wait_directive"):
            directive = decision["wait_directive"]
            if directive.get("session_id"):
                self.wait_on_session(str(directive["session_id"]), reason=state.last_reason)
            elif directive.get("pid"):
                self.wait_on(int(directive["pid"]), reason=state.last_reason)
            else:
                self.wait_for_seconds(int(directive["seconds"]), reason=state.last_reason)
            return {
                "status": "waiting", "should_continue": False, "continuation_prompt": None,
                "verdict": "wait", "reason": state.last_reason,
                "message": f"⏳ Goal parked on a real asynchronous dependency: {state.last_reason}",
            }

        # A controller's proposed success cannot leapfrog live work owned by
        # this goal.  The normal judge prompt sees that work, but retain this
        # deterministic final guard for a malformed or over-eager controller.
        if decision["verdict"] == "achieved":
            pending_background = _background_wait_directive(background_processes)
            if pending_background:
                if pending_background.get("session_id"):
                    self.wait_on_session(
                        str(pending_background["session_id"]),
                        reason="completion candidate is waiting for background work",
                    )
                else:
                    self.wait_on(
                        int(pending_background["pid"]),
                        reason="completion candidate is waiting for background work",
                    )
                return {
                    "status": "waiting", "should_continue": False, "continuation_prompt": None,
                    "verdict": "wait", "reason": "background work is still outstanding",
                    "message": "⏳ Goal completion deferred — waiting for outstanding background work.",
                }

        terminal = decision["verdict"] in {"achieved", "needs_input", "not_achievable", "policy_stop"}
        requires_exhaustion = decision["verdict"] in {"needs_input", "not_achievable"}
        if requires_exhaustion and state.require_recovery_exhaustion:
            # A proposed blocker is a request for recovery, not a terminal
            # state.  The recovery coach returns only safe strategy families;
            # if it finds one, the agent keeps going with a changed approach.
            coach = recovery_coach(state, decision)
            if coach.get("control_plane_error"):
                return self._park_control_plane_error(str(coach["control_plane_error"]))
            self._record_recovery_paths(coach.get("strategies") or [])
            self._ensure_recovery_floor(decision)
            if self._choose_recovery_path() is not None:
                state.last_strategy_constraint = "try the selected recovery path before asking for input or declaring impossibility"
                self._arm_next_wake()
                save_goal(self.session_id, state)
                return {
                    "status": "active", "should_continue": True,
                    "continuation_prompt": self._judge_led_prompt(
                        state.last_reason, state.last_strategy_constraint, replan=True,
                    ),
                    "verdict": "replan", "reason": state.last_reason,
                    "message": "↻ Proposed blocker has an authorized recovery path; continuing.",
                }
            if not self._recovery_is_exhausted():
                # This should be rare (for example a malformed controller
                # response produced no usable route). Keep the goal active and
                # require a fresh diagnosis instead of accepting a blocker.
                return self._replan_with_recovery({
                    **decision,
                    "next_strategy_constraint": (
                        "produce a concrete safe recovery route or independent evidence "
                        "that none remains"
                    ),
                })
        if terminal:
            return self._terminal_or_replan(decision)

        # Repetition and stalls are bounded independently of productive turn
        # count.  Two identical stalled action fingerprints suppress a third
        # unchanged action; three stalled turns call the recovery coach.
        if (
            decision["verdict"] == "replan"
            or state.duplicate_failures >= state.duplicate_failure_limit
            or state.stalled_turns >= state.stall_turns_before_replan
        ):
            return self._replan_with_recovery(decision)

        state.last_strategy_constraint = str(
            decision.get("next_strategy_constraint") or "take the next concrete authorized step"
        )
        self._arm_next_wake()
        save_goal(self.session_id, state)
        return {
            "status": "active", "should_continue": True,
            "continuation_prompt": self._judge_led_prompt(
                state.last_reason, state.last_strategy_constraint,
            ),
            "verdict": "continue", "reason": state.last_reason,
            "message": f"↻ Continuing toward goal ({state.turns_used} productive turns observed): {state.last_reason}",
        }

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        user_initiated: bool = True,
        background_processes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the judge and update state. Return a decision dict.

        ``user_initiated`` distinguishes a real user prompt (True) from a
        continuation prompt we fed ourselves (False). Both increment
        ``turns_used`` because both consume model budget.

        ``background_processes`` is the live ``process_registry.list_sessions()``
        snapshot for this session. It's handed to the judge so it can decide
        to WAIT on an in-flight process (CI poller, build, ...) instead of
        re-poking the agent — the automatic counterpart to ``/goal wait``.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "continue" | "wait" | "skipped" | "inactive"
          - ``reason``: str
          - ``message``: user-visible one-liner to print/send
        """
        state = self._state
        if state is not None and state.status == "waiting":
            if state.waiting_on_session is not None:
                target = f"session {state.waiting_on_session}"
            elif state.waiting_on_pid is not None:
                target = f"pid {state.waiting_on_pid}"
            else:
                target = f"{max(0, int(state.waiting_until - time.time()))}s remaining"
            reason = state.waiting_reason or target
            return {
                "status": "waiting",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "waiting",
                "reason": reason,
                "message": f"⏳ Goal parked — waiting on {target}: {reason}",
            }
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        # A claimed schedule wake becomes this turn.  Clearing the claim at
        # the post-turn boundary makes future recovery explicit and prevents
        # an old gateway/CLI process from holding the schedule hostage.
        self._clear_wake_claim()
        # A real turn can be the first turn for a newly persisted goal even
        # when a surface did not get to inject its kickoff prompt first (for
        # example, a queued user event won the race). Do not later inject the
        # raw objective a second time; every following wake must use the
        # continuation contract.
        state.initial_kickoff_pending = False

        # Count the turn that just finished.
        state.turns_used += 1
        state.last_turn_at = time.time()

        # Judge-led mode intentionally has no goal-turn circuit breaker.  It
        # uses the durable ledger, duplicate-action suppression, recovery
        # coaching, terminal verification, and control-plane parking below.
        # Bounded mode keeps the historical path unchanged for compatibility.
        if state.termination == "judge":
            return self._evaluate_judge_led_after_turn(
                last_response,
                background_processes=background_processes,
            )

        # Quality gates run BEFORE the LLM judge: a failing gate is
        # deterministic evidence the goal is not done, so the judge call is
        # skipped entirely and the gate's output drives the next turn. Gate
        # continuations respect the same turn budget as judge continuations.
        gate_decision = self._check_gates()
        if gate_decision is not None:
            if gate_decision.get("should_continue") and state.turns_used >= state.max_turns:
                state.status = "paused"
                state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
                save_goal(self.session_id, state)
                return {
                    "status": "paused",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "gate_failed",
                    "reason": gate_decision.get("reason", ""),
                    "message": (
                        f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used "
                        f"(a quality gate is still failing). "
                        "Use /goal resume to keep going, or /goal clear to stop."
                    ),
                }
            return gate_decision

        verdict, reason, parse_failed, wait_directive, transport_failed = judge_goal(
            state.goal,
            last_response,
            subgoals=state.subgoals or None,
            background_processes=background_processes,
            contract=state.contract if state.has_contract() else None,
        )
        state.last_verdict = verdict
        state.last_reason = reason
        self._record_evidence(verdict, reason, last_response)

        # Track consecutive judge parse failures. Reset on any usable reply,
        # including API / transport errors (parse_failed=False) so a flaky
        # network doesn't trip the auto-pause meant for bad judge models.
        if parse_failed:
            state.consecutive_parse_failures += 1
        else:
            state.consecutive_parse_failures = 0

        # Track consecutive transport failures separately — persistent API
        # errors (401 auth, DNS, timeout) signal a broken config, not
        # transient network flakiness.  Auto-pause after N consecutive
        # transport failures so a permanently broken judge doesn't burn
        # every turn budget slot on an unreachable API.
        if transport_failed:
            state.consecutive_transport_failures += 1
        else:
            state.consecutive_transport_failures = 0

        # WAIT verdict: the judge decided the agent is blocked on async work
        # and re-poking now would be busy-work. Set the barrier and park —
        # the turn we just counted stands (the judge call happened), but no
        # continuation fires. The loop resumes automatically when the pid
        # exits or the deadline passes (next evaluate_after_turn falls through
        # the is_waiting() short-circuit once the barrier clears).
        if verdict == "wait" and wait_directive:
            if wait_directive.get("session_id"):
                self.wait_on_session(str(wait_directive["session_id"]), reason=reason)
                tgt = f"session {wait_directive['session_id']}"
            elif wait_directive.get("pid"):
                self.wait_on(int(wait_directive["pid"]), reason=reason)
                tgt = f"pid {wait_directive['pid']}"
            else:
                self.wait_for_seconds(int(wait_directive["seconds"]), reason=reason)
                tgt = f"{wait_directive['seconds']}s"
            return {
                "status": "waiting",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "wait",
                "reason": reason,
                "message": f"⏳ Goal parked (judge) — waiting on {tgt}: {reason}",
            }

        if verdict == "done":
            # An agent's prose cannot close the goal while the state says an
            # owned background task is still outstanding.  Park on the most
            # useful durable trigger instead of falsely announcing success.
            pending_background = _background_wait_directive(background_processes)
            if pending_background:
                if pending_background.get("session_id"):
                    self.wait_on_session(
                        str(pending_background["session_id"]),
                        reason="completion candidate is waiting for background work",
                    )
                else:
                    self.wait_on(
                        int(pending_background["pid"]),
                        reason="completion candidate is waiting for background work",
                    )
                save_goal(self.session_id, state)
                return {
                    "status": "waiting",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "wait",
                    "reason": "background work is still outstanding",
                    "message": "⏳ Goal completion deferred — waiting for outstanding background work.",
                }
            if state.approval_policy == "owner":
                state.status = "awaiting_user"
                state.pending_approval_id = uuid.uuid4().hex
                state.pending_approval_reason = reason
                state.pending_approval_at = time.time()
                save_goal(self.session_id, state)
                return {
                    "status": "awaiting_user",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "done",
                    "reason": reason,
                    "approval_id": state.pending_approval_id,
                    "message": (
                        "✋ Goal completion is ready for owner approval. "
                        f"Use /goal approve {state.pending_approval_id} (or /approve on gateways) "
                        f"to mark it done, or /goal reject {state.pending_approval_id} <reason> to continue."
                    ),
                }
            state.status = "done"
            save_goal(self.session_id, state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        if verdict == "needs_user":
            state.status = "awaiting_user"
            state.pending_approval_id = ""
            state.pending_approval_reason = reason
            state.pending_approval_at = time.time()
            save_goal(self.session_id, state)
            return {
                "status": "awaiting_user",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "needs_user",
                "reason": reason,
                "message": f"✋ Goal awaiting user input: {reason}",
            }

        if verdict == "unachievable":
            state.status = "unachievable"
            save_goal(self.session_id, state)
            return {
                "status": "unachievable",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "unachievable",
                "reason": reason,
                "message": f"⊘ Goal is unachievable as stated: {reason}",
            }

        if verdict == "blocked":
            fingerprint = hashlib.sha256(reason.strip().lower().encode("utf-8", "replace")).hexdigest()
            if fingerprint == state.blocker_fingerprint:
                state.consecutive_blockers += 1
            else:
                state.blocker_fingerprint = fingerprint
                state.consecutive_blockers = 1
            if state.consecutive_blockers >= 3:
                state.status = "blocked"
                save_goal(self.session_id, state)
                return {
                    "status": "blocked",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "blocked",
                    "reason": reason,
                    "message": f"⛔ Goal blocked after 3 matching observations: {reason}",
                }
        else:
            state.blocker_fingerprint = ""
            state.consecutive_blockers = 0

        # Auto-pause when the judge cannot reach the API at all N turns in a
        # row (401 auth, DNS failure, timeout).  Persistent transport failures
        # signal a broken configuration (e.g. invalid API key), not transient
        # flakiness.  Without this guard, a permanently broken judge burns
        # every turn budget slot on an unreachable API.
        if state.consecutive_transport_failures >= DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES:
            state.status = "paused"
            state.paused_reason = (
                f"judge API unreachable {state.consecutive_transport_failures} turns in a row "
                f"(check auxiliary.goal_judge provider/key in config.yaml)"
            )
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — judge API returned errors "
                    f"({state.consecutive_transport_failures} turns). "
                    "Check the goal_judge provider/key in ~/.hermes/config.yaml:\n"
                    "  auxiliary:\n"
                    "    goal_judge:\n"
                    "      provider: deepseek\n"
                    "      model: deepseek-v4-flash\n"
                    "Then /goal resume to continue."
                ),
            }

        # Auto-pause when the judge model can't produce the expected JSON
        # verdict N turns in a row. Points the user at the goal_judge config
        # so they can route this side task to a model that follows the
        # contract (e.g. google/gemini-3-flash-preview). Without this guard,
        # weak judge models burn the entire turn budget returning prose or
        # empty strings.
        if state.consecutive_parse_failures >= DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES:
            state.status = "paused"
            state.paused_reason = (
                f"judge model returned unparseable output {state.consecutive_parse_failures} turns in a row"
            )
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — the judge model ({state.consecutive_parse_failures} turns) "
                    "isn't returning the required JSON verdict. Route the judge to a stricter "
                    "model in ~/.hermes/config.yaml:\n"
                    "  auxiliary:\n"
                    "    goal_judge:\n"
                    "      provider: openrouter\n"
                    "      model: google/gemini-3-flash-preview\n"
                    "Then /goal resume to continue."
                ),
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        self._arm_next_wake()
        save_goal(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        if self._state.termination == "judge":
            return self._judge_led_prompt(
                self._state.last_reason or "take the next concrete step",
                self._state.last_strategy_constraint or "make measurable progress toward the stated outcome",
                replan=self._state.last_verdict == "replan",
            )
        # Contract takes priority: it carries the verification surface and
        # constraints the agent must target. Subgoals fold in as extra
        # criteria appended to the contract block.
        if self._state.has_contract():
            contract_block = self._state.contract.render_block()
            if self._state.subgoals:
                extra = "\n".join(
                    f"- Extra criterion {i}: {text}"
                    for i, text in enumerate(self._state.subgoals, start=1)
                )
                contract_block = f"{contract_block}\n{extra}"
            return CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
                goal=self._state.goal,
                contract_block=contract_block,
            )
        if self._state.subgoals:
            return CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                goal=self._state.goal,
                subgoals_block=self._state.render_subgoals_block(),
            )
        return CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)

    def render_contract(self) -> str:
        """Public helper for the /goal show + /goal draft slash commands."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.has_contract():
            return "(no completion contract — set one with /goal draft <objective> or inline field: value lines)"
        return self._state.contract.render_block()


# ──────────────────────────────────────────────────────────────────────
# Kanban worker goal loop
# ──────────────────────────────────────────────────────────────────────

# Continuation prompt fed back to a kanban goal-mode worker that has not
# yet completed/blocked its task. The card's own acceptance criteria are
# the goal — the worker already has the full task body in its first turn,
# so we keep this short and point it back at the lifecycle contract.
KANBAN_GOAL_CONTINUATION_TEMPLATE = (
    "[Continuing toward this kanban task — judge says it is not done yet]\n"
    "Reason: {reason}\n\n"
    "Take the next concrete step toward completing the task. When the work "
    "is genuinely finished, call kanban_complete with a summary. If it is a "
    "code change that needs same-card review before counting as done, call "
    "kanban_request_review with a summary instead. If you are blocked and "
    "need human input, call kanban_block with a reason. Do not stop without "
    "calling one of them."
)

# Fed when the judge believes the work is done but the worker never called
# kanban_complete / kanban_block. One explicit nudge to terminate the task
# the right way before the loop gives up.
KANBAN_GOAL_FINALIZE_TEMPLATE = (
    "[The work looks complete, but the task is still open]\n"
    "Reason: {reason}\n\n"
    "If the task is genuinely done, call kanban_complete now with a short "
    "summary of what you did. If it is a code change awaiting same-card review, "
    "call kanban_request_review with that summary instead. If something still "
    "blocks completion, call kanban_block with the reason instead."
)


def run_kanban_goal_loop(
    *,
    task_id: str,
    goal_text: str,
    run_turn,
    task_status_fn,
    block_fn,
    max_turns: Optional[int] = DEFAULT_MAX_TURNS,
    termination: str = "bounded",
    first_response: str = "",
    controller_state: Optional[Dict[str, Any]] = None,
    save_controller_state=None,
    log=None,
) -> Dict[str, Any]:
    """Drive a kanban worker through a Ralph-style goal loop.

    The dispatcher spawns a goal-mode worker exactly like a normal worker
    (``hermes -p <profile> chat -q "work kanban task <id>"``). The worker's
    first turn has already run by the time this is called; ``first_response``
    is that turn's reply. From here we:

    1. Check whether the worker already terminated the task (called
       ``kanban_complete`` / ``kanban_block``). If so, stop — nothing to do.
    2. Otherwise judge the latest response against ``goal_text`` (the card's
       title + body). ``continue`` → feed a continuation prompt and run
       another turn IN THE SAME SESSION via ``run_turn``. ``done`` but the
       task is still open → one explicit "call kanban_complete" nudge.
    3. When the turn budget is exhausted and the worker still hasn't
       terminated the task, ``block_fn`` is invoked so the card lands in a
       sticky ``blocked`` state for human review (NOT a silent exit).

    This function performs NO SessionDB persistence — a worker process is
    ephemeral, so the turn budget lives in a local counter. It is fully
    decoupled from the CLI for testability: callers inject ``run_turn``
    (str -> str), ``task_status_fn`` (() -> str|None), and ``block_fn``
    (reason: str -> None).

    Returns a decision dict: ``{"outcome", "turns_used", "reason"}`` where
    outcome is one of ``"completed_by_worker"``, ``"review_requested_by_worker"``,
    ``"changes_requested_by_reviewer"``, ``"blocked_budget"``,
    ``"blocked_by_worker"``, or ``"stopped"``.
    """

    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    termination = "judge" if str(termination).lower() == "judge" else "bounded"
    if termination == "bounded":
        max_turns = int(max_turns or DEFAULT_MAX_TURNS)
        if max_turns < 1:
            max_turns = DEFAULT_MAX_TURNS
    else:
        max_turns = None

    last_response = first_response or ""
    # The first turn already consumed one unit of budget.
    turns_used = 1
    nudged_to_finalize = False
    progress_ledger: List[Dict[str, Any]] = []
    recovery_paths: List[Dict[str, Any]] = []
    stalled_turns = 0
    duplicate_failures = 0
    last_action_fingerprint = ""
    if termination == "judge" and isinstance(controller_state, dict):
        raw_ledger = controller_state.get("progress_ledger") or []
        raw_paths = controller_state.get("recovery_paths") or []
        if isinstance(raw_ledger, list):
            progress_ledger = [item for item in raw_ledger if isinstance(item, dict)][-MAX_PROGRESS_LEDGER_ENTRIES:]
        if isinstance(raw_paths, list):
            recovery_paths = [item for item in raw_paths if isinstance(item, dict)][-16:]
        stalled_turns = max(0, int(controller_state.get("stalled_turns", 0) or 0))
        duplicate_failures = max(0, int(controller_state.get("duplicate_failures", 0) or 0))
        last_action_fingerprint = str(controller_state.get("last_action_fingerprint") or "")

    def _checkpoint(*, control_plane_retry_at: float = 0.0) -> None:
        if termination != "judge" or save_controller_state is None:
            return
        snapshot = {
            "progress_ledger": progress_ledger[-MAX_PROGRESS_LEDGER_ENTRIES:],
            "recovery_paths": recovery_paths[-16:],
            "stalled_turns": stalled_turns,
            "duplicate_failures": duplicate_failures,
            "last_action_fingerprint": last_action_fingerprint,
            "turns_used": turns_used,
            "control_plane_retry_at": control_plane_retry_at,
        }
        try:
            save_controller_state(snapshot)
        except Exception as exc:
            _log(f"kanban goal loop: controller checkpoint failed ({exc})")

    def _record_recovery_paths(strategies: List[Dict[str, Any]]) -> None:
        existing = {
            str(path.get("family") or "").strip().lower()
            for path in recovery_paths
            if isinstance(path, dict)
        }
        for strategy in strategies:
            family = str(strategy.get("family") or "").strip()
            step = str(strategy.get("next_step") or "").strip()
            if not family or not step or family.lower() in existing:
                continue
            recovery_paths.append({
                "family": _receipt_excerpt(family, 160),
                "next_step": _receipt_excerpt(step, 420),
                "why_safe": _receipt_excerpt(str(strategy.get("why_safe") or ""), 240),
                "state": "untried",
                "at": time.time(),
            })
            existing.add(family.lower())
        recovery_paths[:] = recovery_paths[-16:]

    def _ensure_recovery_floor(decision: Dict[str, Any]) -> None:
        existing = {
            str(path.get("family") or "").strip().lower()
            for path in recovery_paths
            if isinstance(path, dict)
        }
        suggested = [
            {
                "family": str(family),
                "next_step": f"try the {family} route",
                "why_safe": "identified by the progress controller",
            }
            for family in (decision.get("untried_strategy_families") or [])
            if str(family).strip() and str(family).strip().lower() not in existing
        ]
        _record_recovery_paths(suggested)
        if len({str(path.get("family") or "").lower() for path in recovery_paths if isinstance(path, dict)}) < MIN_RECOVERY_STRATEGY_FAMILIES:
            _record_recovery_paths([dict(item) for item in _BASELINE_RECOVERY_STRATEGIES])

    def _select_recovery_path() -> Optional[Dict[str, Any]]:
        for path in recovery_paths:
            if isinstance(path, dict) and path.get("state") == "untried":
                path["state"] = "in_progress"
                path["started_at"] = time.time()
                return path
        return None

    def _complete_inflight_recovery() -> None:
        for path in reversed(recovery_paths):
            if isinstance(path, dict) and path.get("state") == "in_progress":
                path["state"] = "tried"
                path["completed_at"] = time.time()
                return

    def _recovery_prompt(decision: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Return (prompt, controller-error) after a safe strategy change."""
        state = GoalState(
            goal=goal_text,
            termination="judge",
            max_turns=None,
            progress_ledger=progress_ledger,
            recovery_paths=recovery_paths,
        )
        coach = recovery_coach(state, decision)
        if coach.get("control_plane_error"):
            return None, str(coach["control_plane_error"])
        _record_recovery_paths(coach.get("strategies") or [])
        _ensure_recovery_floor(decision)
        selected = _select_recovery_path()
        if selected is None:
            return None, None
        prompt = KANBAN_GOAL_CONTINUATION_TEMPLATE.format(
            reason=_truncate(
                f"{decision.get('reason') or ''}. Different required approach: {selected.get('next_step') or ''}",
                700,
            )
        )
        return prompt, None

    while True:
        # Did the worker terminate the task itself this turn?
        try:
            status = task_status_fn()
        except Exception as exc:
            _log(f"kanban goal loop: status check failed ({exc}); stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": "status check failed"}

        if status == "done":
            _log(f"kanban goal loop: task {task_id} completed by worker after {turns_used} turn(s)")
            return {"outcome": "completed_by_worker", "turns_used": turns_used, "reason": "worker completed the task"}
        if status == "blocked":
            _log(f"kanban goal loop: task {task_id} blocked by worker after {turns_used} turn(s)")
            return {"outcome": "blocked_by_worker", "turns_used": turns_used, "reason": "worker blocked the task"}
        if status == "review":
            # A legitimate worker-driven terminator (kanban_request_review),
            # not an unexpected stop: the implementation is done and the task
            # is awaiting a reviewer. Stop the loop cleanly.
            _log(f"kanban goal loop: task {task_id} handed off for review by worker after {turns_used} turn(s)")
            return {"outcome": "review_requested_by_worker", "turns_used": turns_used, "reason": "worker requested review"}
        if status == "changes_requested":
            _log(f"kanban goal loop: reviewer returned task {task_id} for changes after {turns_used} turn(s)")
            return {"outcome": "changes_requested_by_reviewer", "turns_used": turns_used, "reason": "reviewer requested changes"}
        if status not in ("running", "ready"):
            # Reclaimed / archived / unexpected — let the dispatcher own it.
            _log(f"kanban goal loop: task {task_id} status={status!r}; stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"status={status}"}

        # Still open — judge whether the latest response satisfies the card.
        # The kanban worker loop has no wait-barrier concept (workers finish
        # via kanban_complete / kanban_block, not by parking), so a WAIT
        # verdict is treated as CONTINUE here.
        if termination == "judge":
            _complete_inflight_recovery()
            controller = judge_goal_with_ledger(
                goal_text,
                last_response,
                contract=None,
                ledger=progress_ledger,
            )
            progress_ledger.append({
                "id": f"kanban-{turns_used}",
                "strategy_family": "worker current approach",
                "action_fingerprint": hashlib.sha256(
                    re.sub(r"\s+", " ", last_response.strip().lower()).encode("utf-8", "replace")
                ).hexdigest()[:24],
                "progress": controller.get("progress", "stalled"),
                "verdict": controller.get("verdict", "continue"),
                "blocker_class": controller.get("blocker_class", "ambiguity"),
                "reason": _receipt_excerpt(str(controller.get("reason") or ""), 500),
                "evidence": _receipt_excerpt(last_response, 700),
                "provenance": "agent_claim",
            })
            progress_ledger = progress_ledger[-MAX_PROGRESS_LEDGER_ENTRIES:]
            verdict = controller["verdict"]
            reason = str(controller.get("reason") or "")
            action_fingerprint = str(progress_ledger[-1].get("action_fingerprint") or "")
            progress = str(controller.get("progress") or "stalled")
            if progress == "stalled":
                duplicate_failures = (
                    duplicate_failures + 1
                    if action_fingerprint and action_fingerprint == last_action_fingerprint
                    else 1
                )
                stalled_turns += 1
            elif progress == "advanced":
                duplicate_failures = 0
                stalled_turns = 0
            else:
                duplicate_failures = 0
                stalled_turns += 1
            last_action_fingerprint = action_fingerprint
            _checkpoint()
            if verdict == "control_plane_error":
                _log(f"kanban goal loop: controller unavailable: {reason}")
                _checkpoint(control_plane_retry_at=time.time() + DEFAULT_CONTROL_PLANE_RETRY_SECONDS)
                return {"outcome": "control_plane_error", "turns_used": turns_used, "reason": reason}
            if verdict == "achieved":
                verdict = "done"
            elif verdict == "policy_stop":
                _checkpoint()
                return {"outcome": "policy_stop", "turns_used": turns_used, "reason": reason}
            elif verdict in {"replan", "needs_input", "not_achievable"} or (
                duplicate_failures >= DEFAULT_DUPLICATE_FAILURE_LIMIT
                or stalled_turns >= DEFAULT_STALL_TURNS_BEFORE_REPLAN
            ):
                # A worker's blocker prose is never a terminal transition. It
                # has to receive a materially different safe path first.
                prompt, recovery_error = _recovery_prompt(controller)
                if recovery_error:
                    _checkpoint(control_plane_retry_at=time.time() + DEFAULT_CONTROL_PLANE_RETRY_SECONDS)
                    return {"outcome": "control_plane_error", "turns_used": turns_used, "reason": recovery_error}
                if prompt:
                    _checkpoint()
                    try:
                        last_response = run_turn(prompt) or ""
                    except Exception as exc:
                        _log(f"kanban goal loop: recovery turn failed ({exc})")
                        return {"outcome": "stopped", "turns_used": turns_used, "reason": f"run_turn error: {type(exc).__name__}"}
                    turns_used += 1
                    continue

                # No safe path remains. Verify the terminal proposal against
                # cumulative card evidence before parking or triaging it.
                verifier_state = GoalState(
                    goal=goal_text,
                    termination="judge",
                    max_turns=None,
                    progress_ledger=progress_ledger,
                    recovery_paths=recovery_paths,
                )
                verified = verify_terminal_goal_decision(verifier_state, controller)
                if verified.get("control_plane_error"):
                    _checkpoint(control_plane_retry_at=time.time() + DEFAULT_CONTROL_PLANE_RETRY_SECONDS)
                    return {"outcome": "control_plane_error", "turns_used": turns_used, "reason": str(verified["control_plane_error"])}
                if not verified.get("accept"):
                    _record_recovery_paths([
                        {
                            "family": str(family),
                            "next_step": f"try the {family} route",
                            "why_safe": "terminal verifier identified an authorized alternative",
                        }
                        for family in (verified.get("untried_strategy_families") or [])
                    ])
                    prompt, recovery_error = _recovery_prompt(controller)
                    if recovery_error:
                        _checkpoint(control_plane_retry_at=time.time() + DEFAULT_CONTROL_PLANE_RETRY_SECONDS)
                        return {"outcome": "control_plane_error", "turns_used": turns_used, "reason": recovery_error}
                    if prompt:
                        _checkpoint()
                        try:
                            last_response = run_turn(prompt) or ""
                        except Exception as exc:
                            _log(f"kanban goal loop: verifier recovery turn failed ({exc})")
                            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"run_turn error: {type(exc).__name__}"}
                        turns_used += 1
                        continue
                    _checkpoint()
                    return {"outcome": "stopped", "turns_used": turns_used, "reason": "terminal verifier rejected the proposed outcome"}
                _checkpoint()
                outcome = "needs_input" if controller["verdict"] == "needs_input" else "not_achievable"
                return {"outcome": outcome, "turns_used": turns_used, "reason": reason}
        else:
            verdict, reason, _parse_failed, _wait, _transport_failed = judge_goal(goal_text, last_response)
            if verdict == "wait":
                verdict = "continue"
        budget_text = "judge-led" if max_turns is None else f"{turns_used}/{max_turns}"
        _log(f"kanban goal loop: turn {budget_text} verdict={verdict} reason={_truncate(reason, 120)}")

        if verdict == "done":
            if nudged_to_finalize:
                # Already asked once to call kanban_complete and it still
                # didn't — block for review rather than spin.
                _log(f"kanban goal loop: task {task_id} judged done but worker won't finalize; blocking")
                try:
                    block_fn(
                        f"Goal-mode worker's output looked complete but it never "
                        f"called kanban_complete after a finalize nudge ({reason})."
                    )
                except Exception as exc:
                    _log(f"kanban goal loop: block_fn failed ({exc})")
                return {"outcome": "blocked_budget", "turns_used": turns_used, "reason": "judged done, never finalized"}
            prompt = KANBAN_GOAL_FINALIZE_TEMPLATE.format(reason=_truncate(reason, 400))
            nudged_to_finalize = True
        else:
            prompt = KANBAN_GOAL_CONTINUATION_TEMPLATE.format(reason=_truncate(reason, 400))

        # Budget check BEFORE spending another turn.
        if max_turns is not None and turns_used >= max_turns:
            _log(f"kanban goal loop: task {task_id} exhausted {turns_used}/{max_turns} turns; blocking")
            try:
                block_fn(
                    f"Goal-mode worker exhausted its turn budget "
                    f"({turns_used}/{max_turns}) without completing the task. "
                    f"Last judge verdict: {_truncate(reason, 300)}"
                )
            except Exception as exc:
                _log(f"kanban goal loop: block_fn failed ({exc})")
            return {"outcome": "blocked_budget", "turns_used": turns_used, "reason": "turn budget exhausted"}

        # Run another turn in the same session.
        try:
            last_response = run_turn(prompt) or ""
        except Exception as exc:
            _log(f"kanban goal loop: run_turn failed ({exc}); stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"run_turn error: {type(exc).__name__}"}
        turns_used += 1


__all__ = [
    "GoalState",
    "GoalContract",
    "GoalGate",
    "GoalManager",
    "parse_contract",
    "parse_goal_start_args",
    "dispatch_goal_loop_alias",
    "list_schedulable_goals",
    "draft_contract",
    "run_gate",
    "workspace_fingerprint",
    "CONTINUATION_PROMPT_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE",
    "JUDGE_LED_CONTINUATION_TEMPLATE",
    "JUDGE_LED_REPLAN_TEMPLATE",
    "JUDGE_USER_PROMPT_TEMPLATE",
    "JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE",
    "DRAFT_CONTRACT_SYSTEM_PROMPT",
    "KANBAN_GOAL_CONTINUATION_TEMPLATE",
    "KANBAN_GOAL_FINALIZE_TEMPLATE",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_DUPLICATE_FAILURE_LIMIT",
    "DEFAULT_STALL_TURNS_BEFORE_REPLAN",
    "load_goal",
    "save_goal",
    "clear_goal",
    "migrate_goal_to_session",
    "judge_goal",
    "judge_goal_with_ledger",
    "recovery_coach",
    "verify_terminal_goal_decision",
    "run_kanban_goal_loop",
]
