"""Behavior contracts for judge-led persistent goals.

These use injected controller roles; no live model or tool process is needed.
The tests deliberately exercise outcomes rather than prompt/source snapshots.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _decision(verdict="continue", progress="advanced", **overrides):
    result = {
        "verdict": verdict,
        "progress": progress,
        "reason": "controller evidence",
        "evidence_refs": [],
        "blocker_class": "ambiguity",
        "recoverable": True,
        "untried_strategy_families": [],
        "next_strategy_constraint": "take a concrete next step",
        "wait_directive": None,
    }
    result.update(overrides)
    return result


def _judge_defaults():
    return {
        "termination": "judge",
        "max_turns": None,
        "duplicate_failure_limit": 2,
        "stall_turns_before_replan": 3,
        "require_recovery_exhaustion": True,
        "terminal_confirmation": True,
    }


def test_judge_mode_allows_productive_work_beyond_legacy_turn_cap(
    hermes_home, monkeypatch
):
    from hermes_cli import goals

    monkeypatch.setattr(goals, "_goal_controller_defaults", _judge_defaults)
    monkeypatch.setattr(goals, "judge_goal_with_ledger", lambda *_a, **_kw: _decision())

    manager = goals.GoalManager("judge-over-20")
    state = manager.set("finish the migration")
    assert state.termination == "judge"
    assert state.max_turns is None

    for turn in range(25):
        result = manager.evaluate_after_turn(f"validated incremental step {turn}")
        assert result["should_continue"] is True
        assert result["status"] == "active"

    assert manager.state is not None
    assert manager.state.turns_used == 25
    assert manager.state.status == "active"


def test_identical_stalled_actions_force_replan_before_third_attempt(
    hermes_home, monkeypatch
):
    from hermes_cli import goals

    monkeypatch.setattr(goals, "_goal_controller_defaults", _judge_defaults)
    monkeypatch.setattr(
        goals,
        "judge_goal_with_ledger",
        lambda *_a, **_kw: _decision(progress="stalled"),
    )
    monkeypatch.setattr(
        goals,
        "recovery_coach",
        lambda *_a, **_kw: {
            "strategies": [{
                "family": "alternate interface",
                "next_step": "use the authorized alternate interface",
                "why_safe": "same authority and recipient scope",
            }]
        },
    )

    manager = goals.GoalManager("judge-duplicate")
    manager.set("send the authorized update")
    first = manager.evaluate_after_turn("bridge call failed with timeout")
    second = manager.evaluate_after_turn("bridge call failed with timeout")

    assert first["verdict"] == "continue"
    assert second["verdict"] == "replan"
    assert "alternate interface" in second["continuation_prompt"]
    assert manager.state.duplicate_failures >= 2


def test_proposed_needs_input_replans_when_safe_route_remains(
    hermes_home, monkeypatch
):
    from hermes_cli import goals

    monkeypatch.setattr(goals, "_goal_controller_defaults", _judge_defaults)
    monkeypatch.setattr(
        goals,
        "judge_goal_with_ledger",
        lambda *_a, **_kw: _decision(
            verdict="needs_input",
            progress="stalled",
            blocker_class="capability",
            reason="bridge is unavailable",
        ),
    )
    monkeypatch.setattr(
        goals,
        "recovery_coach",
        lambda *_a, **_kw: {
            "strategies": [{
                "family": "web interface",
                "next_step": "check the approved web interface and action ledger",
                "why_safe": "does not change recipient or message intent",
            }]
        },
    )

    manager = goals.GoalManager("judge-needs-input")
    manager.set("deliver the approved message")
    result = manager.evaluate_after_turn("The bridge is down; I need you to send it")

    assert result["verdict"] == "replan"
    assert manager.state is not None
    assert manager.state.status == "active"
    assert manager.state.pending_approval_reason == ""


def test_terminal_success_requires_independent_verifier(
    hermes_home, monkeypatch
):
    from hermes_cli import goals

    monkeypatch.setattr(goals, "_goal_controller_defaults", _judge_defaults)
    monkeypatch.setattr(
        goals,
        "judge_goal_with_ledger",
        lambda *_a, **_kw: _decision(verdict="achieved", progress="advanced"),
    )
    manager = goals.GoalManager("judge-terminal")
    manager.set("ship the verified change")

    monkeypatch.setattr(
        goals,
        "verify_terminal_goal_decision",
        lambda *_a, **_kw: {"accept": False, "reason": "missing test receipt", "untried_strategy_families": ["quality gate"]},
    )
    monkeypatch.setattr(
        goals,
        "recovery_coach",
        lambda *_a, **_kw: {"strategies": []},
    )
    rejected = manager.evaluate_after_turn("It is done")
    assert rejected["verdict"] == "replan"
    assert manager.state is not None and manager.state.status == "active"

    monkeypatch.setattr(
        goals,
        "verify_terminal_goal_decision",
        lambda *_a, **_kw: {"accept": True, "reason": "quality gate and artifact receipt verified", "untried_strategy_families": []},
    )
    completed = manager.evaluate_after_turn("The gate now passes with the artifact receipt")
    assert completed["verdict"] == "achieved"
    assert manager.state is not None and manager.state.status == "done"


def test_controller_outage_parks_without_a_followup_agent_turn(hermes_home, monkeypatch):
    from hermes_cli import goals

    monkeypatch.setattr(goals, "_goal_controller_defaults", _judge_defaults)
    monkeypatch.setattr(
        goals,
        "judge_goal_with_ledger",
        lambda *_a, **_kw: _decision(
            verdict="control_plane_error",
            progress="stalled",
            reason="judge transport unavailable",
        ),
    )
    manager = goals.GoalManager("judge-outage")
    manager.set("keep working once controller recovers")
    result = manager.evaluate_after_turn("first actual worker response")

    assert result["verdict"] == "control_plane_error"
    assert result["should_continue"] is False
    assert manager.state is not None
    assert manager.state.status == "control_plane_error"
    assert manager.state.turns_used == 1
    assert manager.has_due_wake(manager.state.control_plane_retry_at - 0.1) is False


def test_learning_candidates_are_sanitized_and_need_verified_reuse(hermes_home):
    from hermes_cli.goal_learning import record_terminal_retrospective
    from hermes_cli.goals import GoalState

    def state(goal_id):
        return GoalState(
            goal="repair the approved connector",
            goal_id=goal_id,
            last_reason="token=very-secret-value at /Users/person/private",
            recovery_paths=[{"family": "alternate interface", "state": "tried"}],
            progress_ledger=[{
                "provenance": "deterministic_gate",
                "evidence": "gate passed authorization=top-secret",
                "remaining_hypotheses": [],
            }],
        )

    first = record_terminal_retrospective(state("goal-1"), "achieved")
    second = record_terminal_retrospective(state("goal-2"), "achieved")

    assert first is not None and first["status"] == "quarantined"
    assert second is not None and second["status"] == "review_eligible"
    evidence = " ".join(item["summary"] for item in second["evidence"])
    assert "top-secret" not in evidence
    assert "[REDACTED]" in evidence
