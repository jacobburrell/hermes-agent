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


def test_tui_projection_hides_empty_guarded_interim_and_keeps_terminal(monkeypatch):
    import tui_gateway.session_history as history_mod
    monkeypatch.setattr(history_mod, "project_compaction_message_for_display", lambda row: row, raising=False)
    rendered = history_mod._history_to_messages([
        {"role": "assistant", "content": "promise interim",
         "display_metadata": {"local_commitment_guard": {"content": ""}}},
        {"role": "assistant", "content": "promise final",
         "display_metadata": {"local_commitment_guard": {"content": "Safe terminal."}}},
    ])
    assert rendered == [{"role": "assistant", "text": "Safe terminal.",
                         "display_metadata": {"local_commitment_guard": {"content": "Safe terminal."}}}]


def test_cli_resume_projects_fenced_content_without_mutating_raw_history():
    from hermes_cli.cli_agent_setup_mixin import _collect_resume_entries
    history = [{"role": "assistant", "content": "I will continue unattended.",
                "display_metadata": {"local_commitment_guard": {"content": "Safe local reply."}}}]
    entries, _idx, _full = _collect_resume_entries(history, {}, lambda text: text)
    assert entries == [("assistant", "Safe local reply.")]
    assert history[0]["content"] == "I will continue unattended."


def test_overlapping_fences_resolve_only_marker_owned_rows(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "overlap.db")
    try:
        db.create_session("s", source="tui")
        first = guard.begin_local_commitment_fence(db, "s", source="tui")
        db.append_message("s", role="assistant", content="first raw", display_metadata={
            "_local_commitment_turn_id": first["turn_id"],
            "local_commitment_guard": {"turn_id": first["turn_id"], "content": ""}})
        second = guard.begin_local_commitment_fence(db, "s", source="tui")
        db.append_message("s", role="assistant", content="second raw", display_metadata={
            "_local_commitment_turn_id": second["turn_id"],
            "local_commitment_guard": {"turn_id": second["turn_id"], "content": ""}})
        assert guard.finish_local_commitment_fence(
            db, first, {"_local_commitment_presentation_override": "safe first"})
        rows = db.get_messages("s")
        assert rows[0]["content"] == "first raw"
        assert rows[0]["display_metadata"]["local_commitment_guard"]["content"] == "safe first"
        assert rows[1]["display_metadata"]["local_commitment_guard"]["content"] == ""
    finally:
        db.close()


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
    assert "cannot verify" in capsys.readouterr().out.lower()
    assert calls == []  # no fence means no model execution or history rewrite


def test_actual_tui_invoke_holds_delta_and_interim_until_local_guard(monkeypatch, tmp_path):
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

    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    db.create_session("tui-session", source="tui")
    raw = _result("I will keep working later.")

    def run_conversation(_message, **kwargs):
        kwargs["stream_callback"]("I will keep working later.")
        agent.interim_assistant_callback("I will keep working later.")
        db.append_message("tui-session", role="assistant", content=raw["final_response"],
                          display_metadata={"_local_commitment_turn_id": agent._local_commitment_turn_id})
        return raw

    agent = SimpleNamespace(run_conversation=run_conversation, _mute_notification_reply=False,
                            _session_db=db, session_id="tui-session")
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
    try:
        prompt_turn._invoke_agent(
            "ui", {"session_key": "tui-session", "history_lock": __import__("threading").RLock()}, st,
            "continue", "continue", None, [], None, None, text="continue")
        assert emitted == []
        assert "will not claim" in st.result["final_response"]
        assert st.result["messages"] == raw["messages"]
    finally:
        db.close()


def test_actual_tui_invoke_exception_projects_late_partial_reply(monkeypatch, tmp_path):
    """The production invoke seam stamps a persisted partial reply before re-raising."""
    from types import SimpleNamespace
    import pytest
    import tui_gateway.prompt_turn as prompt_turn

    class Stop:
        def set(self): pass
    class Thread:
        def join(self): pass

    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "exception.db")
    db.create_session("s", source="tui")
    def boom(_message, **_kwargs):
        db.append_message("s", role="assistant", content="I will continue later.",
                          display_metadata={"_local_commitment_turn_id": agent._local_commitment_turn_id})
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
    try:
        with pytest.raises(RuntimeError, match="provider interrupted"):
            prompt_turn._invoke_agent("ui", {"session_key": "s", "history_lock": __import__("threading").RLock()}, st,
                                      "continue", "continue", None, [], None, None, text="continue")
        row = db.get_messages("s")[-1]
        assert row["content"] == "I will continue later."
        assert "cannot verify" in row["display_metadata"]["local_commitment_guard"]["content"].lower()
        import tui_gateway.session_history as history_mod
        monkeypatch.setattr(history_mod, "project_compaction_message_for_display", lambda item: item, raising=False)
        assert history_mod._history_to_messages(db.get_messages("s"))[-1]["text"].startswith("I cannot verify")
    finally:
        db.close()


def test_actual_interactive_cli_agent_thread_holds_stream_and_refuses(monkeypatch, tmp_path):
    """Exercise the interactive CLI thread seam without a terminal/provider."""
    from types import SimpleNamespace
    from cli import HermesCLI, _ChatTurn

    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "cli.db")
    db.create_session("cli-session", source="cli")
    raw = _result("I will keep working later.")
    streamed = []

    def run_conversation(**kwargs):
        callback = getattr(cli.agent, "stream_delta_callback", None)
        if callback:
            callback("I will keep working later.")
        db.append_message("cli-session", role="assistant", content=raw["final_response"],
                          display_metadata={"_local_commitment_turn_id": cli.agent._local_commitment_turn_id})
        return raw

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-session"
    cli.conversation_history = [{"role": "user", "content": "continue"}]
    cli.agent = SimpleNamespace(run_conversation=run_conversation, stream_delta_callback=lambda text: streamed.append(text),
                                interim_assistant_callback=None, _session_db=db)
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
    try:
        cli._chat_run_agent(turn, "continue")
        assert streamed == []
        assert "will not claim" in turn.result["final_response"]
        assert turn.result["messages"] == raw["messages"]
    finally:
        db.close()
