"""Real-path contracts for managed WhatsApp DM-policy resolution.

The gateway builds its own configuration dictionary instead of using the CLI
loader, so these tests exercise ``load_gateway_config`` under real profile-home
overrides.  They intentionally do not construct a bridge or send a message.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, load_gateway_config
from hermes_cli.managed_scope import invalidate_managed_cache
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_WA_ENV_KEYS = ("WHATSAPP_ENABLED", "WHATSAPP_DM_POLICY")


@pytest.fixture(autouse=True)
def _reset_policy_loader(monkeypatch):
    """Keep filesystem-backed policy tests independent within this file."""
    for key in _WA_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    invalidate_managed_cache()
    set_multiplex_active(False)
    yield
    set_multiplex_active(False)
    for key in _WA_ENV_KEYS:
        os.environ.pop(key, None)
    invalidate_managed_cache()


def _write_policy(home: Path, policy: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "platforms:\n"
        "  whatsapp:\n"
        "    enabled: true\n"
        f"    dm_policy: {policy}\n",
        encoding="utf-8",
    )


def _whatsapp_policy_for(home: Path) -> str:
    """Load through the same context-local profile-home seam as multiplexing."""
    token = set_hermes_home_override(str(home))
    try:
        config = load_gateway_config()
    finally:
        reset_hermes_home_override(token)
    return config.platforms[Platform.WHATSAPP].extra["dm_policy"]


def test_managed_only_profile_resolves_whatsapp_dm_policy(tmp_path, monkeypatch):
    """A fresh profile must not lose a managed WhatsApp intake restriction."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    managed = tmp_path / "managed"
    _write_policy(managed, "disabled")

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    assert _whatsapp_policy_for(profile_home) == "disabled"


def test_multiplex_profiles_keep_opposing_whatsapp_dm_policies_isolated(
    tmp_path, monkeypatch
):
    """Profile A's policy must never become profile B's configuration source."""
    default_home = tmp_path / "default"
    worker_home = tmp_path / "worker"
    _write_policy(default_home, "disabled")
    _write_policy(worker_home, "allowlist")

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    set_multiplex_active(True)

    assert _whatsapp_policy_for(default_home) == "disabled"
    assert _whatsapp_policy_for(worker_home) == "allowlist"


def test_process_managed_policy_wins_inside_each_profile_scope(tmp_path, monkeypatch):
    """One managed overlay is applied after, not instead of, each routed home."""
    default_home = tmp_path / "default"
    worker_home = tmp_path / "worker"
    managed = tmp_path / "managed"
    _write_policy(default_home, "open")
    _write_policy(worker_home, "allowlist")
    _write_policy(managed, "disabled")

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    set_multiplex_active(True)

    assert _whatsapp_policy_for(default_home) == "disabled"
    assert _whatsapp_policy_for(worker_home) == "disabled"


def test_blank_managed_overlay_is_neutral_to_profile_policy(tmp_path, monkeypatch):
    """An intentionally empty managed config must not erase a profile policy."""
    profile_home = tmp_path / "profile"
    managed = tmp_path / "managed"
    _write_policy(profile_home, "allowlist")
    managed.mkdir()
    (managed / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    assert _whatsapp_policy_for(profile_home) == "allowlist"


def test_valid_managed_policy_survives_malformed_profile_config(tmp_path, monkeypatch):
    """A torn user YAML file cannot discard the administrator's DM restriction."""
    profile_home = tmp_path / "profile"
    managed = tmp_path / "managed"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text("platforms: [\n", encoding="utf-8")
    _write_policy(managed, "disabled")

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("WHATSAPP_DM_POLICY", "open")

    # The managed config remains the valid policy source.  The legacy env is
    # only consulted by an adapter when config.extra has no policy at all.
    assert _whatsapp_policy_for(profile_home) == "disabled"


def test_platform_config_outranks_legacy_whatsapp_dm_policy_env(tmp_path, monkeypatch):
    """The loader keeps a policy in config.extra for the adapter's config-first chain."""
    profile_home = tmp_path / "profile"
    _write_policy(profile_home, "disabled")

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("WHATSAPP_DM_POLICY", "open")

    assert _whatsapp_policy_for(profile_home) == "disabled"
