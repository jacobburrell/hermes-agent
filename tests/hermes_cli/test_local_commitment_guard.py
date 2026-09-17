"""Local-surface commitment presentation is refusal-only and cache-safe."""
from __future__ import annotations

from hermes_cli.commitment_admission import CommitmentProposal
from hermes_cli import local_commitment_guard as guard


def _result(text: str = "I will keep working after this reply."):
    return {"final_response": text, "messages": [{"role": "assistant", "content": text}]}


def test_disabled_local_commitment_policy_leaves_immediate_answer_unchanged(monkeypatch):
    monkeypatch.setattr(guard, "local_commitment_state", lambda: False)
    result = _result("The answer is ready.")
    assert guard.guard_local_result(result, text="answer now", session_id="s", platform="cli") is result


def test_malformed_policy_and_classifier_outage_fail_closed_without_mutating_messages(monkeypatch):
    raw = _result()
    monkeypatch.setattr(guard, "local_commitment_state", lambda: None)
    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(disposition="none"))
    immediate = _result("The answer is ready.")
    assert guard.guard_local_result(immediate, text="answer", session_id="s", platform="tui") is immediate

    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(classification="unavailable"))
    refused = guard.guard_local_result(raw, text="please continue", session_id="s", platform="tui")
    assert raw["messages"][0]["content"] in refused["messages"][0]["content"]
    assert refused["final_response"] != raw["final_response"]
    assert "cannot verify" in refused["final_response"].lower()

    monkeypatch.setattr(guard, "local_commitment_state", lambda: True)
    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(classification="unavailable"))
    refused = guard.guard_local_result(raw, text="please continue", session_id="s", platform="tui")
    assert "cannot verify" in refused["final_response"].lower()


def test_verified_immediate_classifier_result_passes_and_continuing_is_refused(monkeypatch):
    monkeypatch.setattr(guard, "local_commitment_state", lambda: True)
    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(disposition="none"))
    immediate = _result("The answer is ready.")
    assert guard.guard_local_result(immediate, text="answer", session_id="s", platform="cli") is immediate

    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(
        disposition="continuing", objective="work", completion_criteria="done", next_action="start"))
    refused = guard.guard_local_result(_result(), text="continue", session_id="s", platform="cli")
    assert "cannot save and deliver" in refused["final_response"]


def test_tui_reload_projects_local_guard_without_rewriting_raw_content(monkeypatch):
    raw = "I will continue unattended."
    history = [{"role": "assistant", "content": raw,
                "display_metadata": {"local_commitment_guard": {"text": "Safe local reply."}}}]
    # The production server binds these helpers when it loads the mixin.  Bind
    # the sole compaction dependency here so this isolated test does not need
    # optional desktop-contract dependencies.
    import tui_gateway.session_history as history_mod
    monkeypatch.setattr(history_mod, "project_compaction_message_for_display", lambda row: row, raising=False)
    rendered = history_mod._history_to_messages(history)
    assert rendered == [{"role": "assistant", "text": "Safe local reply.",
                         "display_metadata": history[0]["display_metadata"]}]
    assert history[0]["content"] == raw


def test_actual_quiet_cli_entry_prints_safe_refusal_without_rewriting_history(monkeypatch, capsys):
    """Exercise the real ``-Q`` entry function with a no-egress classifier fake."""
    from types import SimpleNamespace
    import cli

    raw = _result()
    calls = []
    agent = SimpleNamespace(
        session_id="local-session",
        run_conversation=lambda **kwargs: calls.append(kwargs) or raw,
    )
    monkeypatch.setattr(guard, "local_commitment_state", lambda: True)
    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(
        disposition="continuing", objective="work", completion_criteria="done", next_action="start"))
    with __import__("pytest").raises(SystemExit) as exit_info:
        cli._run_quiet_single_query(
            SimpleNamespace(agent=agent, conversation_history=[{"role": "assistant", "content": "earlier"}],
                            session_id="local-session"), "continue")
    assert exit_info.value.code == 0
    assert raw["messages"][0]["content"] == "I will keep working after this reply."
    assert "will not claim" in capsys.readouterr().out
    assert calls[0]["conversation_history"] == [{"role": "assistant", "content": "earlier"}]


def test_actual_tui_invoke_holds_delta_and_interim_until_local_guard(monkeypatch):
    """The production TUI invoke seam must not emit a prospective promise early."""
    from types import SimpleNamespace
    import tui_gateway.prompt_turn as prompt_turn

    emitted = []

    class Stop:
        def set(self):
            pass

    class Thread:
        def join(self):
            pass

    raw = _result("I will keep working later.")

    def run_conversation(_message, **kwargs):
        kwargs["stream_callback"]("I will keep working later.")
        agent.interim_assistant_callback("I will keep working later.")
        return raw

    agent = SimpleNamespace(run_conversation=run_conversation, _mute_notification_reply=False,
                            _session_db=None, session_id="tui-session")
    st = SimpleNamespace(agent=agent, tts_queue=None, thinking_started=False, result=None, history=[])
    monkeypatch.setattr(guard, "local_commitment_state", lambda: True)
    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(
        disposition="continuing", objective="work", completion_criteria="done", next_action="start"))
    monkeypatch.setattr(prompt_turn, "_is_bot_mode_session", lambda _session: False, raising=False)
    monkeypatch.setattr(prompt_turn, "_load_interim_assistant_messages", lambda: True, raising=False)
    monkeypatch.setattr(prompt_turn, "_adopt_submit_user_row", lambda *args: None, raising=False)
    monkeypatch.setattr(prompt_turn, "_start_usage_ticker", lambda *_args: (Stop(), Thread()), raising=False)
    monkeypatch.setattr(prompt_turn, "_emit", lambda *args: emitted.append(args), raising=False)
    import inspect
    monkeypatch.setattr(prompt_turn, "inspect", inspect, raising=False)
    prompt_turn._invoke_agent(
        "ui", {"session_key": "tui-session", "history_lock": __import__("threading").RLock()}, st,
        "continue", "continue", None, [], None, None, text="continue")
    assert emitted == []
    assert "will not claim" in st.result["final_response"]
    assert st.result["messages"] == raw["messages"]


def test_tui_exception_projection_hides_partial_reply_without_rewriting_raw_history():
    raw = "I will continue after this process exits."

    class DB:
        def __init__(self):
            self.rows = [{"id": 2, "role": "assistant", "content": raw}]
            self.stamps = []

        def get_messages(self, _sid):
            return self.rows

        def set_latest_matching_message_display_kind(self, *args, **kwargs):
            self.stamps.append((args, kwargs))
            return True

    db = DB()
    assert guard.persist_local_commitment_exception_presentation(db, session_id="s", after_row_id=1)
    assert db.rows[0]["content"] == raw
    assert db.stamps[0][1]["display_metadata"]["local_commitment_guard"]["text"].startswith("I cannot verify")


def test_actual_tui_invoke_exception_projects_late_partial_reply(monkeypatch):
    """The production invoke seam stamps a persisted partial reply before re-raising."""
    from types import SimpleNamespace
    import pytest
    import tui_gateway.prompt_turn as prompt_turn

    class DB:
        def __init__(self):
            self.rows, self.stamps = [], []

        def get_messages(self, _sid):
            return list(self.rows)

        def set_latest_matching_message_display_kind(self, *args, **kwargs):
            self.stamps.append((args, kwargs))
            return True

    class Stop:
        def set(self): pass
    class Thread:
        def join(self): pass

    db = DB()
    def boom(_message, **_kwargs):
        db.rows.append({"id": 9, "role": "assistant", "content": "I will continue later."})
        raise RuntimeError("provider interrupted")

    agent = SimpleNamespace(run_conversation=boom, _mute_notification_reply=False, _session_db=db, session_id="s")
    st = SimpleNamespace(agent=agent, tts_queue=None, thinking_started=False, result=None, history=[])
    monkeypatch.setattr(guard, "local_commitment_state", lambda: True)
    monkeypatch.setattr(prompt_turn, "_is_bot_mode_session", lambda _s: False, raising=False)
    monkeypatch.setattr(prompt_turn, "_load_interim_assistant_messages", lambda: False, raising=False)
    monkeypatch.setattr(prompt_turn, "_adopt_submit_user_row", lambda *args: None, raising=False)
    monkeypatch.setattr(prompt_turn, "_start_usage_ticker", lambda *_args: (Stop(), Thread()), raising=False)
    import inspect
    monkeypatch.setattr(prompt_turn, "inspect", inspect, raising=False)
    with pytest.raises(RuntimeError, match="provider interrupted"):
        prompt_turn._invoke_agent("ui", {"session_key": "s", "history_lock": __import__("threading").RLock()}, st,
                                  "continue", "continue", None, [], None, None, text="continue")
    assert db.rows[0]["content"] == "I will continue later."
    assert db.stamps and "cannot verify" in db.stamps[0][1]["display_metadata"]["local_commitment_guard"]["text"].lower()


def test_actual_interactive_cli_agent_thread_holds_stream_and_refuses(monkeypatch):
    """Exercise the interactive CLI thread seam without a terminal/provider."""
    from types import SimpleNamespace
    from cli import HermesCLI, _ChatTurn

    raw = _result("I will keep working later.")
    streamed = []

    def run_conversation(**kwargs):
        callback = getattr(cli.agent, "stream_delta_callback", None)
        if callback:
            callback("I will keep working later.")
        return raw

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-session"
    cli.conversation_history = [{"role": "user", "content": "continue"}]
    cli.agent = SimpleNamespace(run_conversation=run_conversation, stream_delta_callback=lambda text: streamed.append(text),
                                interim_assistant_callback=None)
    for name in ("_sudo_password_callback", "_approval_callback", "_secret_capture_callback",
                 "_vault_unlock_callback", "_vault_save_login_callback", "_vault_code_callback"):
        setattr(cli, name, lambda *_args, **_kwargs: None)
    cli._flush_credit_notices = lambda: None
    cli._pending_moa_config = None
    cli._pending_moa_disable_after_turn = False
    monkeypatch.setattr(guard, "local_commitment_state", lambda: True)
    monkeypatch.setattr(guard, "propose_with_auxiliary", lambda *_args: CommitmentProposal(
        disposition="continuing", objective="work", completion_criteria="done", next_action="start"))
    turn = _ChatTurn()
    cli._chat_run_agent(turn, "continue")
    assert streamed == []
    assert "will not claim" in turn.result["final_response"]
    assert turn.result["messages"] == raw["messages"]
