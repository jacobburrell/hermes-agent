"""Real-path contracts for managed WhatsApp DM-policy resolution.

These tests exercise the filesystem-backed gateway loader under profile-home
overrides. They do not connect a bridge or send messages.
"""

from __future__ import annotations

import json
import os
import weakref
from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.session import SessionSource
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


def _load_for(home: Path):
    """Load through the same context-local profile-home seam as multiplexing."""
    token = set_hermes_home_override(str(home))
    try:
        return load_gateway_config()
    finally:
        reset_hermes_home_override(token)


def _policy_for(home: Path) -> str:
    return _load_for(home).platforms[Platform.WHATSAPP].extra["dm_policy"]


@pytest.mark.parametrize(
    "user_yaml",
    [
        pytest.param(None, id="missing"),
        pytest.param("platforms: [\n", id="malformed"),
        pytest.param("- not\n- a\n- mapping\n", id="nonmapping"),
    ],
)
def test_valid_managed_policy_survives_unusable_profile_config(
    tmp_path, monkeypatch, user_yaml
):
    """A missing or broken user layer cannot discard a managed DM restriction."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    if user_yaml is not None:
        (profile_home / "config.yaml").write_text(user_yaml, encoding="utf-8")
    managed = tmp_path / "managed"
    _write_policy(managed, "disabled")

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("WHATSAPP_DM_POLICY", "open")

    assert _policy_for(profile_home) == "disabled"


def test_empty_managed_config_is_neutral_to_valid_profile(tmp_path, monkeypatch):
    """An intentionally empty managed mapping does not erase profile policy."""
    profile_home = tmp_path / "profile"
    managed = tmp_path / "managed"
    _write_policy(profile_home, "allowlist")
    managed.mkdir()
    (managed / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    assert _policy_for(profile_home) == "allowlist"


def test_managed_policy_overrides_one_leaf_without_replacing_profile_mapping(
    tmp_path, monkeypatch
):
    """Managed leaves win while user siblings survive and managed-only leaves are added."""
    profile_home = tmp_path / "profile"
    managed = tmp_path / "managed"
    profile_home.mkdir()
    managed.mkdir()
    (profile_home / "config.yaml").write_text(
        "platforms:\n"
        "  whatsapp:\n"
        "    enabled: true\n"
        "    dm_policy: allowlist\n"
        "    send_read_receipts: true\n",
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(
        "platforms:\n"
        "  whatsapp:\n"
        "    dm_policy: disabled\n"
        "    require_mention: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    extra = _load_for(profile_home).platforms[Platform.WHATSAPP].extra

    assert extra["dm_policy"] == "disabled"
    assert extra["send_read_receipts"] is True
    assert extra["require_mention"] is True


def test_malformed_managed_config_is_neutral_to_valid_profile(tmp_path, monkeypatch):
    """Managed-scope parse failure keeps the valid profile policy (documented fail-open)."""
    profile_home = tmp_path / "profile"
    managed = tmp_path / "managed"
    _write_policy(profile_home, "allowlist")
    managed.mkdir()
    (managed / "config.yaml").write_text("platforms: [\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    assert _policy_for(profile_home) == "allowlist"


def test_config_policy_outranks_legacy_whatsapp_policy_env(tmp_path, monkeypatch):
    """The adapter consumes config.extra before the legacy environment fallback."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    profile_home = tmp_path / "profile"
    _write_policy(profile_home, "disabled")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("WHATSAPP_DM_POLICY", "open")

    config = _load_for(profile_home)
    adapter = WhatsAppAdapter(config.platforms[Platform.WHATSAPP])

    assert config.platforms[Platform.WHATSAPP].extra["dm_policy"] == "disabled"
    assert adapter._dm_policy == "disabled"


@pytest.mark.parametrize(
    "user_yaml",
    [
        pytest.param(None, id="missing"),
        pytest.param("platforms: [\n", id="malformed"),
        pytest.param("- not\n- a\n- mapping\n", id="nonmapping"),
    ],
)
def test_unusable_yaml_layer_preserves_legacy_gateway_json(
    tmp_path, monkeypatch, user_yaml
):
    """An unusable user layer with no managed mapping leaves the legacy base intact."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    if user_yaml is not None:
        (profile_home / "config.yaml").write_text(user_yaml, encoding="utf-8")
    (profile_home / "gateway.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "whatsapp": {
                        "enabled": True,
                        "extra": {"dm_policy": "allowlist"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert _policy_for(profile_home) == "allowlist"


def test_environment_policy_remains_adapter_fallback_without_config(tmp_path, monkeypatch):
    """With no config policy, the existing scoped/environment fallback is unchanged."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_DM_POLICY", "allowlist")

    config = _load_for(profile_home)
    adapter = WhatsAppAdapter(config.platforms[Platform.WHATSAPP])

    assert "dm_policy" not in config.platforms[Platform.WHATSAPP].extra
    assert adapter._dm_policy == "allowlist"


def test_absent_user_and_managed_layers_keep_default_gateway_shape(tmp_path, monkeypatch):
    """An ordinary empty home still has no platform entries or synthetic policy."""
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    config = _load_for(profile_home)

    assert config.platforms == {}


def test_routed_profile_cannot_replace_shared_whatsapp_transport_policy(monkeypatch):
    """Shared WhatsApp ingress remains authorized by its transport/default profile."""
    from gateway.run import GatewayRunner
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setenv("WHATSAPP_DM_POLICY", "open")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    transport = WhatsAppAdapter(
        PlatformConfig(enabled=True, extra={"dm_policy": "disabled"})
    )
    runner.adapters = {Platform.WHATSAPP: transport}
    runner._profile_adapters = {"worker": {}}
    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id="15551234567",
        chat_id="15551234567@s.whatsapp.net",
        chat_type="dm",
        profile="worker",
    )
    source._transport_adapter_ref = weakref.ref(transport)

    owner_profile = runner._adapter_profile_for_source(source)

    assert owner_profile is None
    assert runner._authorization_adapter(Platform.WHATSAPP, profile="worker") is None
    assert runner._adapter_policy(Platform.WHATSAPP, "dm", owner_profile) == "disabled"
