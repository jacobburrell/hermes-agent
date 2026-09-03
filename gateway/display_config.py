"""Per-platform display/verbosity configuration resolver.

Provides ``resolve_display_setting()`` — the single entry-point for reading
display settings with platform-specific overrides and sensible defaults.

Resolution order (first non-None wins):
    1. ``display.platforms.<platform>.<key>``  — explicit per-platform user override
    2. ``display.<key>``                       — global user setting
    3. ``_PLATFORM_DEFAULTS[<platform>][<key>]``  — built-in sensible default
    4. ``_GLOBAL_DEFAULTS[<key>]``              — built-in global default

Exception: ``display.streaming`` is CLI-only.  Gateway streaming follows the
top-level ``streaming`` config unless ``display.platforms.<platform>.streaming``
sets an explicit per-platform override.

Backward compatibility: ``display.tool_progress_overrides`` is still read as a
fallback for ``tool_progress`` when no ``display.platforms`` entry exists.  A
config migration (version bump) automatically moves the old format into the new
``display.platforms`` structure.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Overrideable display settings and their global defaults
# ---------------------------------------------------------------------------
# These are the settings that can be configured per-platform.
# Other display settings (compact, personality, skin, etc.) are CLI-only
# and don't participate in per-platform resolution.

_GLOBAL_DEFAULTS: dict[str, Any] = {
    "tool_progress": "all",
    "tool_progress_grouping": "accumulate",  # "accumulate" = edit one bubble; "separate" = one msg per tool
    "show_reasoning": False,
    # How a reasoning/thinking summary is rendered when show_reasoning is on.
    #   "code"      -> 💭 **Reasoning:** + fenced code block (legacy default)
    #   "blockquote"-> each line prefixed with "> "
    #   "subtext"   -> each line prefixed with "-# " (Discord small grey subtext)
    # Discord defaults to "subtext"; everywhere else defaults to "code".
    "reasoning_style": "code",
    "tool_preview_length": 0,
    "streaming": None,  # None = follow top-level streaming config
    # Gateway-only assistant/status chatter controls. These default on for
    # back-compat, but mobile platforms can opt down to final-answer-first.
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    # Whether a follow-up received while a session is busy gets a transient
    # queue/steer/interrupt acknowledgment.  This deliberately remains true
    # as the generic default for existing installations; WhatsApp transports
    # override it below because a status bubble is a permanent chat message.
    "busy_ack_enabled": True,
    "busy_ack_detail": True,
    # Whether busy_input_mode=steer sends a visible "Steered into current run"
    # acknowledgment after successfully injecting the user's mid-turn message.
    # Disable when the platform should steer silently (the text still lands in
    # the active run; only the confirmation echo is suppressed).
    "busy_steer_ack_enabled": True,
    # When true, delete tool-progress / "⏳ Working — N min" / status bubbles
    # after the final response lands on platforms that support message
    # deletion (e.g. Telegram). Off by default — progress is still shown
    # live, just cleaned up after success so the chat doesn't fill up with
    # stale breadcrumbs. Failed runs leave bubbles in place as breadcrumbs.
    "cleanup_progress": False,
    # Live working-state status on platforms whose typing indicator renders
    # text (Slack's assistant status line). Values:
    #   "full" / true  -> verb + argument preview ("is running pytest…")
    #   "verb"         -> verb only ("is running…") — keeps file paths and
    #                     commands out of shared channels
    #   "off" / false  -> static text (typing_status_text or "is thinking...")
    # Independent of tool_progress: works even when progress bubbles are off
    # (Slack's default), and costs no extra API calls — the existing typing
    # refresh cadence just renders different text.
    "live_status": "full",
}

# ---------------------------------------------------------------------------
# Sensible per-platform defaults — tiered by platform capability
# ---------------------------------------------------------------------------
# Tier 1 (high): Supports message editing, typically personal/team use
# Tier 2 (medium): Supports editing but often workspace/customer-facing
# Tier 3 (low): No edit support — each progress msg is permanent
# Tier 4 (minimal): Batch/non-interactive delivery

_TIER_HIGH = {
    "tool_progress": "all",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": None,  # follow global
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
}

_TIER_MEDIUM = {
    "tool_progress": "new",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": None,
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
}

_TIER_LOW = {
    "tool_progress": "off",
    "show_reasoning": False,
    "tool_preview_length": 40,
    "streaming": False,
    "interim_assistant_messages": False,
    "long_running_notifications": False,
    "busy_ack_detail": False,
}

_TIER_MINIMAL = {
    "tool_progress": "off",
    "show_reasoning": False,
    "tool_preview_length": 0,
    "streaming": False,
    "interim_assistant_messages": False,
    "long_running_notifications": False,
    "busy_ack_detail": False,
}

_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    # Tier 1 — full edit support, personal/team use
    # Telegram is usually a mobile inbox: keep tool_progress quiet and skip
    # the verbose busy-ack iteration counter, but DO surface real mid-turn
    # assistant commentary (interim_assistant_messages) and DO send periodic
    # heartbeats (long_running_notifications) so the user has signal between
    # turn start and final answer. Otherwise it looks like "typing..." for
    # 30 minutes with nothing happening. Opt in to verbose iteration detail
    # via display.platforms.telegram.busy_ack_detail / tool_progress.
    "telegram":    {
        **_TIER_HIGH,
        "tool_progress": "off",
        "busy_ack_detail": False,
    },
    # Discord has a native "subtext" primitive (-# small grey text) that reads
    # as metadata rather than content, so reasoning summaries default to it
    # here instead of the fenced code block used elsewhere.
    "discord":     {**_TIER_HIGH, "reasoning_style": "subtext"},

    # Tier 2 — edit support, often customer/workspace channels
    # Slack: tool_progress off by default — Bolt posts cannot be edited like CLI;
    # "new"/"all" spam permanent lines in channels (hermes-agent#14663).
    "slack":           {
        **_TIER_MEDIUM,
        "tool_progress": "off",
        "long_running_notifications": False,
        "busy_ack_detail": False,
    },
    "mattermost":      _TIER_MEDIUM,
    "matrix":          _TIER_MEDIUM,
    "feishu":          _TIER_MEDIUM,
    # Buzz (Nostr relay via buzz-cli): messages can be edited in place
    # (`buzz messages edit`), so grouped/accumulating progress works, but
    # channels are shared community spaces — keep the medium tier. Without
    # this entry Buzz inherited the verbose _GLOBAL_DEFAULTS and every
    # interim update became a separate permanent channel post (#95841).
    "buzz":            _TIER_MEDIUM,

    # Tier 3 — no edit support, progress messages are permanent
    "signal":          _TIER_LOW,
    # A busy acknowledgment is not a progress bubble and cannot be safely
    # edited away.  Make WhatsApp final-answer-first by default; users may
    # opt in with display.platforms.whatsapp.busy_ack_enabled: true.
    "whatsapp":        {**_TIER_MEDIUM, "busy_ack_enabled": False},
    # WhatsApp Cloud API: Meta added message editing in 2023 but the
    # Hermes Cloud adapter doesn't implement edit_message yet, so we
    # stay on TIER_LOW (tool_progress off) to avoid spamming each
    # status update as a separate message. Promote to TIER_MEDIUM once
    # Cloud's edit_message lands.
    "whatsapp_cloud":  {**_TIER_LOW, "busy_ack_enabled": False},
    # Photon (managed iMessage over the gRPC sidecar) and BlueBubbles are both
    # permanent-message iMessage inboxes with no message-edit support, so both
    # stay TIER_LOW. This keeps tool progress, interim scratch commentary,
    # "still working" heartbeats, and busy-ack iteration detail out of the
    # user's iMessage thread. Without this entry Photon inherited the noisy
    # global ("all") defaults and compacted/narrated on nearly every turn.
    "photon":          _TIER_LOW,
    "bluebubbles":     _TIER_LOW,
    "weixin":          _TIER_LOW,
    # WeCom is technically non-editable but exposes a native streaming
    # transport (msgtype: "stream" via aibot_respond_msg) that the gateway
    # consumer routes mid-stream content through. Enable streaming by default
    # so the WeCom client renders the typing animation and cumulative content
    # updates instead of a single one-shot markdown drop.
    "wecom":           {**_TIER_LOW, "streaming": True},
    "wecom_callback":  _TIER_LOW,
    "dingtalk":        _TIER_LOW,

    # Tier 4 — batch or non-interactive delivery
    "email":           _TIER_MINIMAL,
    "sms":             _TIER_MINIMAL,
    "webhook":         _TIER_MINIMAL,
    "homeassistant":   _TIER_MINIMAL,
    "api_server":      {**_TIER_HIGH, "tool_preview_length": 0},
}

# Canonical set of per-platform overrideable keys (for validation).
OVERRIDEABLE_KEYS = frozenset(_GLOBAL_DEFAULTS.keys())


def resolve_display_setting(
    user_config: dict,
    platform_key: str,
    setting: str,
    fallback: Any = None,
) -> Any:
    """Resolve a display setting with per-platform override support.

    Parameters
    ----------
    user_config : dict
        The full parsed config.yaml dict.
    platform_key : str
        Platform config key (e.g. ``"telegram"``, ``"slack"``).  Use
        ``_platform_config_key(source.platform)`` from gateway/run.py.
    setting : str
        Display setting name (e.g. ``"tool_progress"``, ``"show_reasoning"``).
    fallback : Any
        Fallback value when the setting isn't found anywhere.

    Returns
    -------
    The resolved value, or *fallback* if nothing is configured.
    """
    display_cfg = user_config.get("display") or {}

    # 1. Explicit per-platform override (display.platforms.<platform>.<key>)
    platforms = display_cfg.get("platforms") or {}
    plat_overrides = platforms.get(platform_key)
    if isinstance(plat_overrides, dict):
        val = plat_overrides.get(setting)
        if val is not None:
            return _normalise(setting, val)

    # 1b. Backward compat: display.tool_progress_overrides.<platform>
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict):
            val = legacy.get(platform_key)
            if val is not None:
                return _normalise(setting, val)

    # 2. Global user setting (display.<key>).  Skip display.streaming because
    # that key controls only CLI terminal streaming; gateway token streaming is
    # governed by the top-level streaming config plus per-platform overrides.
    if setting != "streaming":
        val = display_cfg.get(setting)
        if val is not None:
            return _normalise(setting, val)

    # 3. Built-in platform default
    plat_defaults = _PLATFORM_DEFAULTS.get(platform_key)
    if plat_defaults:
        val = plat_defaults.get(setting)
        if val is not None:
            return val

    # 4. Built-in global default
    val = _GLOBAL_DEFAULTS.get(setting)
    if val is not None:
        return val

    return fallback


def resolve_busy_ack_enabled(
    user_config: dict,
    platform_key: str,
    *,
    chat_type: str | None = None,
    legacy_env: str | None = None,
    first_invalid_config: bool = False,
) -> bool:
    """Resolve the busy-ack visibility policy for one inbound source.

    This is intentionally separate from :func:`resolve_display_setting`:
    busy acknowledgments are runtime notices, so their legacy process
    environment compatibility tier belongs *after* all YAML settings and
    before built-in defaults.  It also has a narrow chat-type override tier.

    Resolution order (first explicit value wins):

    1. ``display.chat_types.<chat_type>.busy_ack_enabled``
    2. ``display.platforms.<platform>.busy_ack_enabled``
    3. ``display.busy_ack_enabled``
    4. ``HERMES_GATEWAY_BUSY_ACK_ENABLED`` (legacy compatibility)
    5. built-in platform then global defaults

    A malformed *first* config read fails closed for WhatsApp transports.
    Once a caller has a last-known-good snapshot, it passes that snapshot with
    ``first_invalid_config=False`` so a torn live edit cannot change policy.
    """
    platform_key = str(platform_key or "").strip().lower()
    if first_invalid_config and platform_key in {"whatsapp", "whatsapp_cloud"}:
        return False

    display_cfg = user_config.get("display") if isinstance(user_config, dict) else None
    if not isinstance(display_cfg, dict):
        display_cfg = {}

    if chat_type:
        chat_types = display_cfg.get("chat_types")
        chat_cfg = chat_types.get(str(chat_type).strip().lower()) if isinstance(chat_types, dict) else None
        if (
            isinstance(chat_cfg, dict)
            and chat_cfg.get("busy_ack_enabled") is not None
        ):
            return bool(_normalise("busy_ack_enabled", chat_cfg["busy_ack_enabled"]))

    platforms = display_cfg.get("platforms")
    platform_cfg = platforms.get(platform_key) if isinstance(platforms, dict) else None
    if (
        isinstance(platform_cfg, dict)
        and platform_cfg.get("busy_ack_enabled") is not None
    ):
        return bool(_normalise("busy_ack_enabled", platform_cfg["busy_ack_enabled"]))

    if display_cfg.get("busy_ack_enabled") is not None:
        return bool(_normalise("busy_ack_enabled", display_cfg["busy_ack_enabled"]))

    if legacy_env is not None:
        return bool(_normalise("busy_ack_enabled", legacy_env))

    platform_defaults = _PLATFORM_DEFAULTS.get(platform_key)
    if platform_defaults and "busy_ack_enabled" in platform_defaults:
        return bool(platform_defaults["busy_ack_enabled"])
    return bool(_GLOBAL_DEFAULTS["busy_ack_enabled"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(setting: str, value: Any) -> Any:
    """Normalise YAML quirks (bare ``off`` → False in YAML 1.1)."""
    if setting == "tool_progress":
        if value is False:
            return "off"
        if value is True:
            return "all"
        val = str(value).strip().lower()
        if val in {"false", "0", "no"}:
            return "off"
        if val in {"true", "1", "yes", "on"}:
            return "all"
        return val if val in {"off", "new", "all", "verbose", "log"} else "all"
    if setting in {
        "show_reasoning",
        "streaming",
        "interim_assistant_messages",
        "long_running_notifications",
        "busy_ack_enabled",
        "busy_ack_detail",
        "busy_steer_ack_enabled",
        "thinking_progress",
    }:
        if isinstance(value, str):
            val = value.strip().lower()
            if val == "generic" and setting == "long_running_notifications":
                return "generic"
            return val in {"true", "1", "yes", "on", "raw", "verbose"}
        return bool(value)
    if setting == "cleanup_progress":
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        return bool(value)
    if setting == "live_status":
        # Tri-state: "full" (verb + preview), "verb" (verb only), "off".
        if value is True:
            return "full"
        if value is False:
            return "off"
        val = str(value).strip().lower()
        if val in {"true", "1", "yes", "on", "all"}:
            return "full"
        if val in {"false", "0", "no"}:
            return "off"
        return val if val in {"full", "verb", "off"} else "full"
    if setting == "tool_progress_grouping":
        val = str(value).lower()
        return val if val in ("accumulate", "separate") else "accumulate"
    if setting == "reasoning_style":
        val = str(value).lower()
        return val if val in ("code", "blockquote", "subtext") else "code"
    if setting == "tool_preview_length":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return value
