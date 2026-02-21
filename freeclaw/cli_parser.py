from __future__ import annotations

import argparse
from typing import Any


def _handler(handlers: dict[str, Any], key: str) -> Any:
    fn = handlers.get(key)
    if fn is None:
        raise KeyError(f"missing parser handler: {key}")
    return fn


def _add_common_runtime_flags(
    parser: argparse.ArgumentParser,
    *,
    verbose_tools_help: str = "Log tool calls/results.",
) -> None:
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--no-tools", action="store_true", help="Disable built-in tools.")

    g_shell = parser.add_mutually_exclusive_group()
    g_shell.add_argument(
        "--enable-shell",
        action="store_true",
        help="Enable sh_exec tool (command execution). (default: enabled)",
    )
    g_shell.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable sh_exec tool (command execution).",
    )

    g_custom = parser.add_mutually_exclusive_group()
    g_custom.add_argument(
        "--enable-custom-tools",
        action="store_true",
        help="Enable loading custom tools from workspace/skills/tools. (default: enabled)",
    )
    g_custom.add_argument(
        "--no-custom-tools",
        action="store_true",
        help="Disable loading custom tools (overrides FREECLAW_ENABLE_CUSTOM_TOOLS).",
    )

    parser.add_argument(
        "--custom-tools-dir",
        default=None,
        help="Override custom tools dir (must be within workspace). Default: workspace/skills/tools",
    )
    parser.add_argument("--no-skills", action="store_true", help="Disable injecting enabled skills into system prompt.")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory for persona.md/tools.md/custom tool specs (default from config/env).",
    )
    parser.add_argument(
        "--tool-root",
        default=None,
        help="Tool filesystem root (default from config/env; relative roots are resolved from cwd).",
    )
    parser.add_argument(
        "--max-tool-steps",
        type=int,
        default=None,
        help="Max tool loop steps (default from config; 50).",
    )
    parser.add_argument("--verbose-tools", action="store_true", help=verbose_tools_help)


def build_main_parser(*, handlers: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freeclaw")
    parser.add_argument(
        "--agent",
        default=None,
        help="Agent profile name (shorthand for --config config/agents/<name>/config.json).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json (default: ./config/config.json if present).",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Explicit env file to load (overrides auto-load). You can also set FREECLAW_ENV_FILE.",
    )
    parser.add_argument("--no-onboard", action="store_true", help="Skip first-run onboarding wizard.")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (debug, info, warning, error). You can also set FREECLAW_LOG_LEVEL.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: <config>/freeclaw.log). You can also set FREECLAW_LOG_FILE.",
    )
    parser.add_argument(
        "--log-format",
        default=None,
        choices=["text", "jsonl"],
        help="Log format (text or jsonl). You can also set FREECLAW_LOG_FORMAT.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_models = sub.add_parser("models", help="List available model ids for the configured provider.")
    p_models.add_argument(
        "--provider",
        choices=["nim", "openrouter", "groq"],
        default=None,
        help="Override provider (default from config).",
    )
    p_models.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Override base URL (default from config).",
    )
    p_models.add_argument(
        "--free-only",
        dest="free_only",
        action="store_true",
        help="(openrouter) Only show free models (best-effort).",
    )
    p_models.set_defaults(func=_handler(handlers, "cmd_models"))

    p_run = sub.add_parser("run", help="Run a single prompt.")
    p_run.add_argument("prompt")
    _add_common_runtime_flags(p_run)
    p_run.add_argument("--json", action="store_true", help="Print raw provider JSON response.")
    p_run.set_defaults(func=_handler(handlers, "cmd_run"))

    p_chat = sub.add_parser("chat", help="Interactive chat.")
    _add_common_runtime_flags(p_chat)
    p_chat.set_defaults(func=_handler(handlers, "cmd_chat"))

    p_tt = sub.add_parser("task-timer", help="Periodically review workspace/tasks.md and complete unchecked tasks.")
    p_tt.add_argument("--minutes", type=int, default=None, help="Interval minutes (default from config; 30). 0 disables.")
    p_tt.add_argument("--once", action="store_true", help="Run a single tick and exit.")
    _add_common_runtime_flags(p_tt)
    p_tt.set_defaults(func=_handler(handlers, "cmd_task_timer"))

    p_ta = sub.add_parser(
        "timer-api",
        help="Run an HTTP timer API server that auto-checks workspace/tasks.md and wakes the model when tasks are due.",
    )
    p_ta.add_argument("--minutes", type=int, default=None, help="Task timer interval minutes shown to the model (default from config; 30).")
    _add_common_runtime_flags(p_ta)
    p_ta.add_argument(
        "--all-agents",
        action="store_true",
        help="Authoritative mode: check due tasks for base config plus all agent profiles on each tick.",
    )
    g_ta_web = p_ta.add_mutually_exclusive_group()
    g_ta_web.add_argument(
        "--web-ui",
        dest="web_ui",
        action="store_true",
        default=None,
        help="Enable Web UI dashboard routes (/ and /api/system/metrics).",
    )
    g_ta_web.add_argument(
        "--no-web-ui",
        dest="web_ui",
        action="store_false",
        help="Disable Web UI dashboard routes.",
    )
    p_ta.add_argument("--host", default="0.0.0.0", help="Bind host for timer API server (default: public bind).")
    p_ta.add_argument(
        "--allow-remote-host",
        action="store_true",
        help="Deprecated: public bind is now default. Timer control endpoints remain localhost-only.",
    )
    p_ta.add_argument(
        "--localhost-only-host",
        action="store_true",
        help="Restrict timer-api bind host to localhost only.",
    )
    p_ta.add_argument("--port", type=int, default=None, help="Bind port for timer API server (default from config web_ui_port; 3000).")
    p_ta.add_argument(
        "--poll-seconds",
        type=float,
        default=300.0,
        help="Background scheduler poll interval in seconds (default: 300).",
    )
    p_ta.set_defaults(func=_handler(handlers, "cmd_timer_api"))

    p_cfg = sub.add_parser("config", help="Config utilities.")
    sub_cfg = p_cfg.add_subparsers(dest="cfg_cmd", required=True)
    p_init = sub_cfg.add_parser("init", help="Write default config to ./config/config.json.")
    p_init.add_argument(
        "--path",
        default=None,
        help="Override config path (default: ./config/config.json).",
    )
    p_init.set_defaults(func=_handler(handlers, "cmd_config_init"))

    p_env = sub_cfg.add_parser("env-init", help="Write a template .env file (default: ./config/.env).")
    p_env.add_argument("--path", default=None, help="Path to write (default: ./config/.env).")
    p_env.set_defaults(func=_handler(handlers, "cmd_config_env_init"))

    p_show = sub_cfg.add_parser("show", help="Show config (merged with env overrides by default).")
    p_show.add_argument("--raw", action="store_true", help="Show config.json contents only (no env overrides).")
    p_show.add_argument("--quiet-path", action="store_true", help="Do not print config path to stderr.")
    p_show.set_defaults(func=_handler(handlers, "cmd_config_show"))

    p_set = sub_cfg.add_parser("set", help="Set a config key.")
    p_set.add_argument("key", help="Config key (example: max_tool_steps).")
    p_set.add_argument("value", help="Value (string by default; use --json for JSON).")
    p_set.add_argument("--json", action="store_true", help="Parse value as JSON.")
    p_set.set_defaults(func=_handler(handlers, "cmd_config_set"))

    p_val = sub_cfg.add_parser("validate", help="Validate the effective config.")
    p_val.add_argument("--quiet-path", action="store_true", help="Do not print config path to stderr.")
    p_val.set_defaults(func=_handler(handlers, "cmd_config_validate"))

    p_reset = sub.add_parser("reset", help="Delete ./config and ./workspace (fresh install).")
    p_reset.add_argument("--yes", action="store_true", help="Do not prompt for confirmation.")
    p_reset.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print what would be deleted, but do not delete anything.",
    )
    p_reset.set_defaults(func=_handler(handlers, "cmd_reset"))

    p_discord = sub.add_parser("discord", help="Run a Discord bot that chats using the configured provider.")
    p_discord.add_argument("--token", default=None, help="Discord bot token (or DISCORD_BOT_TOKEN).")
    p_discord.add_argument(
        "--no-all-agents",
        action="store_true",
        help="If multiple agents exist, do not auto-launch them; run only this bot.",
    )
    p_discord.add_argument("--prefix", default=None, help="Command prefix trigger (default from config).")
    p_discord.add_argument(
        "--respond-to-all",
        action="store_true",
        default=None,
        help="Respond to every non-bot message in channels/DMs the bot can read (no prefix/mention required).",
    )
    _add_common_runtime_flags(p_discord, verbose_tools_help="Log tool calls/results to stderr.")
    p_discord.add_argument(
        "--history-messages",
        type=int,
        default=None,
        help="How many non-system messages to retain per conversation session (default from config).",
    )
    p_discord.add_argument(
        "--session-scope",
        choices=["channel", "user", "global"],
        default=None,
        help=(
            "Conversation state scope for Discord sessions: "
            "channel (default), user (shared across channels/DMs), or global."
        ),
    )
    p_discord.add_argument(
        "--web-ui-host",
        default=None,
        help=argparse.SUPPRESS,
    )
    p_discord.add_argument(
        "--web-ui-public",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p_discord.add_argument(
        "--web-ui-localonly",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p_discord.add_argument(
        "--web-ui-port",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p_discord.add_argument(
        "--no-web-ui",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p_discord.add_argument(
        "--no-web-ui-autostart",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p_discord.set_defaults(func=_handler(handlers, "cmd_discord"))

    p_onboard = sub.add_parser("onboard", help="Onboarding utilities.")
    p_onboard.add_argument(
        "--force",
        dest="onboard_force",
        action="store_true",
        help="Run the main onboarding wizard even if already onboarded.",
    )
    p_onboard.set_defaults(func=_handler(handlers, "cmd_onboard"))
    sub_onboard = p_onboard.add_subparsers(dest="onboard_cmd", required=False)
    p_create = sub_onboard.add_parser(
        "createagent",
        help="Create a named agent profile (config + env + workspace) under config/agents/<name>/.",
    )
    p_create.add_argument("name", nargs="?", help="Agent name (example: salesbot)")
    p_create.add_argument("--force", dest="create_force", action="store_true", help="Overwrite an existing agent.")
    p_create.add_argument(
        "--keep-provider",
        dest="keep_provider",
        action="store_true",
        help="Keep provider from the base/current config instead of prompting for a new provider.",
    )
    p_create.set_defaults(func=_handler(handlers, "cmd_onboard_createagent"))

    p_skill = sub.add_parser("skill", help="Skills: list/show/enable/disable")
    sub_skill = p_skill.add_subparsers(dest="skill_cmd", required=True)
    p_s_list = sub_skill.add_parser("list", help="List available skills (* = enabled).")
    p_s_list.set_defaults(func=_handler(handlers, "cmd_skill_list"))
    p_s_show = sub_skill.add_parser("show", help="Show a skill SKILL.md.")
    p_s_show.add_argument("name")
    p_s_show.set_defaults(func=_handler(handlers, "cmd_skill_show"))
    p_s_en = sub_skill.add_parser("enable", help="Enable a skill by name.")
    p_s_en.add_argument("name")
    p_s_en.set_defaults(func=_handler(handlers, "cmd_skill_enable"))
    p_s_dis = sub_skill.add_parser("disable", help="Disable a skill by name.")
    p_s_dis.add_argument("name")
    p_s_dis.set_defaults(func=_handler(handlers, "cmd_skill_disable"))

    return parser
