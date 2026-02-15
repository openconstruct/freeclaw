from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path
from typing import Any

from .config import ClawConfig, default_skills_dir, load_config_dict, save_config_dict
from .dotenv import default_config_env_path, set_env_var
from .http_client import get_json


def _prompt(msg: str, default: str | None = None) -> str:
    if default is None:
        return input(msg).strip()
    v = input(f"{msg} [{default}]: ").strip()
    return v if v else default


def _prompt_yes_no(msg: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{msg} ({d}): ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False


def _discord_invite_url(*, app_id: str, permissions: int) -> str:
    # scopes: bot + application commands (slash commands)
    return (
        "https://discord.com/oauth2/authorize"
        f"?client_id={app_id}&scope=bot%20applications.commands&permissions={permissions}"
    )


def _prompt_discord_app_id(current: str | None) -> str | None:
    raw = _prompt("Discord Application ID (Client ID) for invite link (optional)", default=(current or "")).strip()
    if not raw:
        return None
    # Discord app IDs are snowflakes; keep it simple and validate digits only.
    if not raw.isdigit():
        print("Application ID should be digits only; skipping.")
        return None
    return raw


def _fetch_nim_model_ids(*, base_url: str, api_key: str) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    resp = get_json(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "freeclaw/0.1.0",
        },
        timeout_s=30.0,
    ).json
    data = resp.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for m in data:
        mid = m.get("id") if isinstance(m, dict) else None
        if isinstance(mid, str) and mid.strip():
            ids.append(mid.strip())
    # Stable order for display; NIM doesn't guarantee ordering.
    return sorted(set(ids), key=str.lower)


def _prompt_select_model(*, base_url: str, api_key: str, current: str | None) -> str | None:
    want = _prompt_yes_no("Select a NIM model now? (recommended)", default=True)
    if not want:
        return current

    try:
        ids = _fetch_nim_model_ids(base_url=base_url, api_key=api_key)
    except Exception as e:
        print(f"Could not fetch {base_url.rstrip('/')}/models: {e}")
        manual = _prompt("Enter model id (or leave blank to auto-select at runtime)", default="").strip()
        return manual or None

    if not ids:
        manual = _prompt("No models returned. Enter model id (or leave blank to auto-select at runtime)", default="")
        return manual.strip() or None

    # Keep the display manageable; allow narrowing by typing a substring.
    candidates = ids
    while True:
        show = candidates[:40]
        print("\nAvailable models (showing up to 40):")
        for i, mid in enumerate(show, start=1):
            mark = " (current)" if current and mid == current else ""
            print(f"{i:2d}) {mid}{mark}")
        if len(candidates) > len(show):
            print(f"... and {len(candidates) - len(show)} more (type a filter substring to narrow)")

        raw = _prompt("Choose: number, filter text, or blank to keep current/auto", default="").strip()
        if not raw:
            return current
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(show):
                return show[idx - 1]
            print("Invalid selection number.")
            continue

        # Treat as filter substring.
        filt = raw.lower()
        narrowed = [m for m in ids if filt in m.lower()]
        if not narrowed:
            print("No matches for that filter.")
            continue
        candidates = narrowed


def _write_startup_md(
    *,
    path: Path,
    assistant_name: str,
    assistant_tone: str,
    discord_prefix: str,
    discord_respond_to_all: bool,
) -> None:
    txt = "\n".join(
        [
            "# startup",
            "",
            "This file documents the bot identity used by freeclaw prompts.",
            "",
            "## Name",
            assistant_name.strip() or "Freeclaw",
            "",
            "## Tone",
            assistant_tone.strip() or "(not set)",
            "",
            "## Discord Behavior",
            f"- Prefix: {discord_prefix}",
            f"- Respond to all messages: {str(bool(discord_respond_to_all)).lower()}",
            "- Reset session: `!claw reset` (or mention the bot + `reset` if using mentions)",
            "",
            "## Notes",
            "- If the bot should read normal messages, enable Message Content Intent in the Discord Developer Portal (Bot tab).",
            "- Tool docs: see tools.md in your workspace directory.",
            "",
        ]
    )
    path.write_text(txt + "\n", encoding="utf-8")


def _ensure_persona_md(*, path: Path, assistant_name: str, assistant_tone: str) -> None:
    if path.exists():
        return
    txt = "\n".join(
        [
            "# persona",
            "",
            "This is the bot's persistent persona file. freeclaw will include this file in the system prompt.",
            "",
            "## Name",
            assistant_name.strip() or "Freeclaw",
            "",
            "## Tone",
            assistant_tone.strip() or "(not set)",
            "",
            "## Persona",
            "- Mission:",
            "- Boundaries:",
            "- Style:",
            "- Defaults:",
            "",
        ]
    )
    path.write_text(txt + "\n", encoding="utf-8")


def _ensure_tools_md(*, path: Path) -> None:
    if path.exists():
        return
    try:
        from .tools import tool_schemas
    except Exception:
        tool_schemas = None  # type: ignore[assignment]

    tool_lines: list[str] = []
    if tool_schemas is not None:
        try:
            for t in tool_schemas(include_shell=True):
                fn = t.get("function") if isinstance(t, dict) else None
                if not isinstance(fn, dict):
                    continue
                nm = fn.get("name")
                desc = fn.get("description") or ""
                if isinstance(nm, str) and nm.strip():
                    tool_lines.append(f"- {nm}: {str(desc).strip()}")
        except Exception:
            tool_lines = []

    txt = "\n".join(
        [
            "# tools",
            "",
            "This file documents Freeclaw's built-in tools. The bot should consult this file when asked what tools it has.",
            "",
            "## Tool List",
            *(
                tool_lines
                if tool_lines
                else [
                    "- (tool list unavailable during onboarding; run freeclaw once to auto-generate)",
                ]
            ),
            "",
            "## Notes",
            "- Web search requires: pip install -e \".[web]\"",
            "- Shell command execution (sh_exec) is enabled by default; disable with --no-shell or FREECLAW_ENABLE_SHELL=false.",
            "",
        ]
    )
    path.write_text(txt + "\n", encoding="utf-8")


def _select_env_path(hint: Path | None) -> Path:
    cfg_env = default_config_env_path()
    if hint is not None:
        return hint
    # Keep secrets in ./config by default; override via --env-file or FREECLAW_ENV_FILE.
    return cfg_env


def run_onboarding(
    *,
    config_path: str | None,
    force: bool,
    env_path_hint: str | None = None,
) -> Path:
    if not sys.stdin.isatty():
        raise SystemExit(
            "First-run onboarding requires an interactive terminal. "
            "Create ./config/.env with NVIDIA_API_KEY=... and rerun."
        )

    cfg_path, cfg_dict = load_config_dict(config_path)
    cfg = ClawConfig.from_dict(cfg_dict)
    if cfg.onboarded and not force:
        return default_config_env_path()

    print("freeclaw first-run setup\n")
    print("This will:")
    print("- store your NVIDIA NIM API key in a .env file")
    print("- configure Discord defaults (token in .env, prefix in config)")
    print("- set a workspace directory for Freeclaw state (persona/tools/custom tools)")
    print("- set a disk access root for filesystem tools\n")

    hint = Path(env_path_hint).expanduser().resolve() if env_path_hint else None
    env_path = _select_env_path(hint)
    print(f"Using env file: {env_path}")

    existing_key = os.getenv("NVIDIA_API_KEY") or ""
    key_prompt = "NVIDIA_API_KEY (input hidden; leave blank to keep existing env value)"
    nim_key = getpass.getpass(f"{key_prompt}: ").strip()
    if not nim_key:
        nim_key = existing_key
    if not nim_key:
        raise SystemExit("NVIDIA_API_KEY is required.")
    set_env_var(env_path, "NVIDIA_API_KEY", nim_key)

    want_discord = _prompt_yes_no("Configure Discord bot now?", default=True)
    discord_token = ""
    discord_prefix = cfg.discord_prefix or "!claw"
    discord_history = cfg.discord_history_messages or 40
    discord_respond_to_all = bool(getattr(cfg, "discord_respond_to_all", False))
    discord_app_id = getattr(cfg, "discord_app_id", None)
    if want_discord:
        discord_token = getpass.getpass("DISCORD_BOT_TOKEN (input hidden; optional): ").strip()
        if discord_token:
            set_env_var(env_path, "DISCORD_BOT_TOKEN", discord_token)
        discord_app_id = _prompt_discord_app_id(discord_app_id)
        discord_prefix = _prompt("Discord command prefix", default=discord_prefix)
        discord_respond_to_all = _prompt_yes_no(
            "Discord: respond to every message (no prefix needed)?",
            default=discord_respond_to_all,
        )
        try:
            discord_history = int(_prompt("Discord history messages to retain", default=str(discord_history)))
        except ValueError:
            discord_history = 40

    base_url = (cfg.base_url or "https://integrate.api.nvidia.com/v1").strip()
    selected_model = _prompt_select_model(base_url=base_url, api_key=nim_key, current=cfg.model)

    assistant_name = _prompt("Bot name", default=(cfg.assistant_name or "Freeclaw")).strip() or "Freeclaw"
    assistant_tone = _prompt(
        "Bot tone (one sentence)",
        default=(cfg.assistant_tone or "Direct, pragmatic, concise. Ask clarifying questions when needed."),
    ).strip()

    # Keep everything local to the current directory by default.
    default_workspace = str((Path.cwd() / "workspace").resolve())
    workspace_dir = _prompt("Workspace directory (Freeclaw state lives here)", default=default_workspace).strip()
    workspace_p = Path(workspace_dir).expanduser()
    workspace_p.mkdir(parents=True, exist_ok=True)

    default_tool_root = str(workspace_p)
    tool_root = _prompt(
        "Disk access root for fs_* tools (enter / for full disk access)",
        default=default_tool_root,
    ).strip()
    tool_root_p = Path(tool_root).expanduser()
    # Don't create "/" accidentally; but ensure other paths exist.
    if str(tool_root_p) != "/":
        tool_root_p.mkdir(parents=True, exist_ok=True)
    _write_startup_md(
        path=(workspace_p / "startup.md"),
        assistant_name=assistant_name,
        assistant_tone=assistant_tone,
        discord_prefix=discord_prefix,
        discord_respond_to_all=discord_respond_to_all,
    )
    _ensure_persona_md(
        path=(workspace_p / "persona.md"),
        assistant_name=assistant_name,
        assistant_tone=assistant_tone,
    )
    _ensure_tools_md(path=(workspace_p / "tools.md"))

    # Skills: create a default global skills dir and add it to config.
    skills_dir = default_skills_dir()
    skills_dir.mkdir(parents=True, exist_ok=True)
    skills_dirs = list(cfg.skills_dirs or [])
    if str(skills_dir) not in skills_dirs:
        skills_dirs.append(str(skills_dir))

    # Minimal default skill scaffold.
    sample_skill = skills_dir / "scaffold" / "SKILL.md"
    if not sample_skill.exists():
        sample_skill.parent.mkdir(parents=True, exist_ok=True)
        sample_skill.write_text(
            "\n".join(
                [
                    "# scaffold",
                    "",
                    "When asked to create a new project/skill scaffold:",
                    "- create a directory with a clear name",
                    "- write a README.md with how to run/use it",
                    "- keep changes minimal and runnable",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    enabled_skills = list(cfg.enabled_skills or [])
    if "scaffold" not in enabled_skills:
        enabled_skills.append("scaffold")

    new_cfg = ClawConfig(
        onboarded=True,
        provider=cfg.provider or "nim",
        base_url=base_url,
        model=selected_model,
        assistant_name=assistant_name,
        assistant_tone=assistant_tone,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        workspace_dir=str(workspace_p),
        tool_root=str(tool_root_p),
        tool_max_read_bytes=cfg.tool_max_read_bytes,
        tool_max_write_bytes=cfg.tool_max_write_bytes,
        tool_max_list_entries=cfg.tool_max_list_entries,
        skills_dirs=skills_dirs,
        enabled_skills=enabled_skills,
        discord_prefix=discord_prefix,
        discord_history_messages=discord_history,
        discord_respond_to_all=discord_respond_to_all,
        discord_app_id=discord_app_id,
    )
    save_config_dict(cfg_path, new_cfg.to_dict())

    print("\nSetup complete.")
    print(f"- config: {cfg_path}")
    print(f"- env:    {env_path}")
    print(f"- workspace: {workspace_p}")
    print(f"- tool_root: {tool_root_p}")
    print(f"- startup {workspace_p / 'startup.md'}")
    print(f"- persona {workspace_p / 'persona.md'}")
    print(f"- tools   {workspace_p / 'tools.md'}")
    if want_discord and discord_app_id:
        # Minimal perms: View Channel (1024) + Send Messages (2048) + Read Message History (65536)
        # With extras: + Embed Links (16384) + Attach Files (32768)
        print("\nDiscord bot invite links:")
        print("- minimal (read/respond):")
        print(f"  {_discord_invite_url(app_id=discord_app_id, permissions=68608)}")
        print("- with embeds/files:")
        print(f"  {_discord_invite_url(app_id=discord_app_id, permissions=117760)}")
        print("\nNote: also enable Message Content Intent in the Discord Developer Portal (Bot tab).")
    print("\nNext:")
    print("- run:    python -m freeclaw run \"hello\"")
    print(
        "- discord python -m freeclaw discord --prefix \"{}\" --workspace \"{}\" --tool-root \"{}\"".format(
            discord_prefix, workspace_p, tool_root_p
        )
    )
    return env_path
