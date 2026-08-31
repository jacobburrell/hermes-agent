"""Focused contracts for the canonical live busy-ack policy reader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.display_config import resolve_busy_ack_enabled
from gateway.platforms.base import SessionSource
from gateway.run import GatewayRunner


def _write_config(home: Path, display: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"display": display}, sort_keys=False), encoding="utf-8"
    )


def _source(platform: Platform = Platform.WHATSAPP) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="busy-policy-test",
        chat_type="group",
        user_id="test-user",
    )


def _runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    return runner


def test_canonical_reader_retains_profile_last_known_good(tmp_path, monkeypatch):
    """Malformed YAML never converts a live WhatsApp mute into an env opt-in."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    _write_config(tmp_path, {"busy_ack_enabled": False})

    assert _runner()._busy_ack_enabled_for_source(_source()) is False
    (tmp_path / "config.yaml").write_text("display: [\n", encoding="utf-8")

    # The canonical status seam returns the same profile's LKG snapshot.
    assert _runner()._busy_ack_enabled_for_source(_source()) is False


def test_first_invalid_source_fails_closed_only_for_whatsapp(tmp_path, monkeypatch):
    """A fresh broken config silences transient mobile notices, not other rails."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text("display: [\n", encoding="utf-8")

    runner = _runner()
    assert runner._busy_ack_enabled_for_source(_source(Platform.WHATSAPP)) is False
    assert runner._busy_ack_enabled_for_source(_source(Platform.TELEGRAM)) is True
    assert runner._busy_ack_enabled_for_source(_source(Platform.DISCORD)) is True


def test_managed_scope_is_process_global_and_wins_at_the_live_boundary(
    tmp_path, monkeypatch
):
    """The routed user profile is scoped, then one managed overlay is applied."""
    import gateway.run as gateway_run

    managed = tmp_path / "managed"
    _write_config(tmp_path, {"busy_ack_enabled": True})
    _write_config(managed, {"busy_ack_enabled": False})
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    assert _runner()._busy_ack_enabled_for_source(_source()) is False


@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD])
def test_blank_managed_overlay_leaves_valid_user_busy_policy_intact(
    tmp_path, monkeypatch, platform
):
    """An empty managed overlay is no policy, not a malformed override."""
    import gateway.run as gateway_run
    from hermes_cli import managed_scope

    user_home = tmp_path / "user"
    managed_home = tmp_path / "managed"
    _write_config(user_home, {"busy_ack_enabled": True})
    managed_home.mkdir()
    (managed_home / "config.yaml").write_text("\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", user_home)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    managed_scope.invalidate_managed_cache()

    assert _runner()._busy_ack_enabled_for_source(_source(platform)) is True


def test_builtin_precedence_keeps_mobile_silent_and_other_platforms_unchanged():
    """No generated DEFAULT_CONFIG value may shadow this resolver chain."""
    assert resolve_busy_ack_enabled({}, "whatsapp") is False
    assert resolve_busy_ack_enabled({}, "whatsapp_cloud") is False
    assert resolve_busy_ack_enabled({}, "telegram") is True
    assert resolve_busy_ack_enabled({}, "discord") is True


def test_current_tip_documents_config_not_a_new_user_facing_env_switch():
    from hermes_cli.tips import TIPS

    busy_tips = [tip for tip in TIPS if "busy_ack" in tip]
    assert busy_tips == [
        "Set display.busy_ack_enabled: false in config.yaml to silence the ⚡/⏳/⏩ ack messages when a user messages a busy agent."
    ]


def test_documented_global_platform_and_chat_type_policy_paths_are_valid():
    from hermes_cli.config import _validate_config_key

    for key in (
        "display.busy_ack_enabled",
        "display.platforms.whatsapp.busy_ack_enabled",
        "display.platforms.whatsapp.chat_types.group.busy_ack_enabled",
    ):
        assert _validate_config_key(key) == (True, None)
