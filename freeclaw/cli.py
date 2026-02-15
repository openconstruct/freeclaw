import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from .agent import run_agent
from .config import (
    ClawConfig,
    load_config,
    load_config_dict,
    save_config_dict,
    write_default_config,
)
from .dotenv import autoload_dotenv, load_dotenv
from .integrations.discord_bot import run_discord_bot
from .onboarding import run_onboarding
from .providers.nim import NimChatClient
from .skills import find_skill, iter_skills, render_enabled_skills_system
from .tools import ToolContext, tool_schemas


DEFAULT_TOOL_SYSTEM = """You can use tools to read/write/list files within tool_root.

Tool rules:
- Only use fs_* tools when you need file contents or to create/update files.
- Keep reads small: prefer start_line/end_line, and read only what you need.
- When writing code, create/update files directly via fs_write and ensure paths are correct.
- Use persona.md (in workspace) as your persistent persona store. Keep it concise and update it when asked to change your identity/behavior.
- Use tools.md (in workspace) as the canonical human-readable tool list.

Other tools:
- web_search: search the web (DuckDuckGo) for sources.
- web_fetch: fetch a public http(s) URL and return its text (HTML is converted to plain text).
- http_request_json: call a public http(s) JSON API and return parsed JSON (blocks localhost/private IPs).
- memory_*: store and retrieve long-term notes in a local SQLite DB.
- sh_exec: execute a command (argv) in tool_root (enabled by default; disable via --no-shell or FREECLAW_ENABLE_SHELL=false).

Custom tools:
- By default, freeclaw loads additional tools from JSON specs under `.freeclaw/tools` (within workspace).
- Supported locations:
  - `.freeclaw/tools/<name>.json`
  - `.freeclaw/tools/<name>/tool.json`
- Spec fields (type=command):
  - `name`, `description`, `argv` (array of strings), `parameters` (JSON schema)
  - Optional: `workdir` ("tool_root" or "tool_dir"), `env` (map), `stdin` (string), `parse_json` (bool)
- Templates in `argv`/`env`/`stdin`: `{{arg_name}}` or `{{args_json}}`.
- After writing a new spec file, the tool becomes available on the next model step.
"""


def _should_skip_onboarding(cmd: str) -> bool:
    if os.getenv("FREECLAW_NO_ONBOARD", "").strip() not in {"", "0", "false", "False"}:
        return True
    return cmd in {"config", "onboard", "skill"}


def _build_system_prompt(base: str | None, skills_block: str | None) -> str | None:
    b = (base or "").strip()
    s = (skills_block or "").strip()
    if not b and not s:
        return None
    if b and s:
        return b + "\n\n" + s + "\n"
    if b:
        return b + "\n"
    return s + "\n"


def _identity_system(cfg: ClawConfig) -> str | None:
    name = (cfg.assistant_name or "").strip()
    tone = (cfg.assistant_tone or "").strip()
    if not name and not tone:
        return None
    lines: list[str] = []
    if name:
        lines.append(f"Your name is {name}.")
    if tone:
        lines.append(f"Tone: {tone}")
    return "\n".join(lines) + "\n"


def _resolve_tool_root(cfg: ClawConfig, args_tool_root: str | None) -> Path:
    root = Path(args_tool_root if args_tool_root is not None else (cfg.tool_root or ".")).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    return root


def _resolve_workspace_root(cfg: ClawConfig, args_workspace: str | None) -> Path:
    ws = Path(args_workspace if args_workspace is not None else (cfg.workspace_dir or "workspace")).expanduser()
    if not ws.is_absolute():
        ws = (Path.cwd() / ws).resolve()
    else:
        ws = ws.resolve()
    return ws


def _read_persona_md(workspace: Path) -> str | None:
    p = workspace / "persona.md"
    try:
        if not p.exists() or not p.is_file():
            return None
        data = p.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    # Hard cap to keep the system prompt bounded.
    if len(data) > 20_000:
        data = data[:20_000]
    return data.decode("utf-8", errors="replace").strip() or None


def _read_tools_md(workspace: Path) -> str | None:
    p = workspace / "tools.md"
    try:
        if not p.exists() or not p.is_file():
            return None
        data = p.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    if len(data) > 20_000:
        data = data[:20_000]
    return data.decode("utf-8", errors="replace").strip() or None


def _tool_list_system_for(*, tool_ctx: ToolContext | None, include_shell: bool) -> str:
    lines: list[str] = ["Available tools:"]
    include_custom = bool(tool_ctx is not None and tool_ctx.custom_tools_enabled)
    for t in tool_schemas(
        include_shell=include_shell,
        include_custom=include_custom,
        tool_ctx=tool_ctx,
    ):
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        desc = fn.get("description") or ""
        if isinstance(name, str) and name.strip():
            d = str(desc).strip()
            lines.append(f"- {name}: {d}")
    return "\n".join(lines).strip() + "\n"


def _ensure_persona_md(workspace: Path, cfg: ClawConfig) -> None:
    p = workspace / "persona.md"
    if p.exists():
        return
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "\n".join(
                [
                    "# persona",
                    "",
                    "This file is included in freeclaw's system prompt.",
                    "Edit it to define and evolve the bot's persona.",
                    "",
                    "## Name",
                    (cfg.assistant_name or "Freeclaw").strip() or "Freeclaw",
                    "",
                    "## Tone",
                    (cfg.assistant_tone or "").strip() or "(not set)",
                    "",
                    "## Persona",
                    "- Mission:",
                    "- Boundaries:",
                    "- Style:",
                    "- Defaults:",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        # Best-effort; if tool_root is read-only, just skip.
        return


def _ensure_tools_md(workspace: Path, *, tool_ctx: ToolContext | None, include_shell: bool) -> None:
    p = workspace / "tools.md"
    if p.exists():
        return
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "\n".join(
                [
                    "# tools",
                    "",
                    "This file is included in freeclaw's system prompt.",
                    "It lists the built-in tools currently available to the model.",
                    "",
                    _tool_list_system_for(tool_ctx=tool_ctx, include_shell=include_shell).strip(),
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _core_system_prelude(
    cfg: ClawConfig,
    *,
    tool_root: Path,
    workspace: Path,
    enable_tools: bool,
    tool_ctx: ToolContext | None,
    include_shell: bool,
) -> str:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    parts: list[str] = [f"Current date/time: {now}", f"Workspace: {workspace}", f"Tool root: {tool_root}"]

    ident = _identity_system(cfg)
    if ident:
        parts.append(ident.strip())

    # Make sure the docs ship even if onboarding wasn't run.
    _ensure_persona_md(workspace, cfg)
    if enable_tools:
        _ensure_tools_md(workspace, tool_ctx=tool_ctx, include_shell=include_shell)

    persona = _read_persona_md(workspace)
    if persona:
        parts.append("Persona (from persona.md):\n" + persona)
    else:
        parts.append("Persona: Use persona.md (in tool_root) to store your persona.")

    if enable_tools:
        # Always include a tool list that exactly matches the exposed tool schemas.
        parts.append(_tool_list_system_for(tool_ctx=tool_ctx, include_shell=include_shell).strip())

        tools_doc = _read_tools_md(workspace)
        if tools_doc:
            parts.append("Tools (from tools.md):\n" + tools_doc)
        else:
            parts.append("Tools: See tools.md (in tool_root) for a human-readable list of available tools.")

    return "\n\n".join(parts) + "\n\n"


def _client_from_config(cfg: ClawConfig) -> NimChatClient:
    # For now, the only supported provider is NIM; config still has provider knobs for future parity.
    if (cfg.provider or "nim") != "nim":
        raise SystemExit(f"Only provider='nim' is supported right now (got {cfg.provider!r}).")
    return NimChatClient.from_config(cfg)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    client = _client_from_config(cfg)

    temperature = args.temperature if args.temperature is not None else cfg.temperature
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.max_tokens

    tool_ctx = None
    tools_builder = None
    include_shell = False
    if not args.no_tools:
        tool_root = args.tool_root if args.tool_root is not None else cfg.tool_root
        workspace_dir = args.workspace if getattr(args, "workspace", None) is not None else cfg.workspace_dir
        enable_shell = None
        if getattr(args, "no_shell", False):
            enable_shell = False
        elif getattr(args, "enable_shell", False):
            enable_shell = True
        enable_custom_tools = None
        if getattr(args, "no_custom_tools", False):
            enable_custom_tools = False
        elif getattr(args, "enable_custom_tools", False):
            enable_custom_tools = True
        tool_ctx = ToolContext.from_config_values(
            tool_root=tool_root,
            workspace_dir=workspace_dir,
            max_read_bytes=cfg.tool_max_read_bytes,
            max_write_bytes=cfg.tool_max_write_bytes,
            max_list_entries=cfg.tool_max_list_entries,
            enable_shell=enable_shell,
            enable_custom_tools=enable_custom_tools,
            custom_tools_dir=getattr(args, "custom_tools_dir", None),
        )
        include_shell = bool(tool_ctx.shell_enabled)
        tools_builder = lambda: tool_schemas(
            include_shell=tool_ctx.shell_enabled,
            include_custom=tool_ctx.custom_tools_enabled,
            tool_ctx=tool_ctx,
        )

    base_system = args.system
    if base_system is None and not args.no_tools:
        base_system = DEFAULT_TOOL_SYSTEM
    skills_block = "" if args.no_skills else render_enabled_skills_system(cfg)
    tool_root_p = tool_ctx.root if tool_ctx is not None else _resolve_tool_root(cfg, args.tool_root)
    workspace_p = tool_ctx.workspace if tool_ctx is not None else _resolve_workspace_root(cfg, getattr(args, "workspace", None))
    core = _core_system_prelude(
        cfg,
        tool_root=tool_root_p,
        workspace=workspace_p,
        enable_tools=(not args.no_tools),
        tool_ctx=tool_ctx,
        include_shell=include_shell,
    )
    system_prompt = _build_system_prompt(core + (base_system or ""), skills_block)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    result = run_agent(
        client=client,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_tools=(not args.no_tools),
        tool_ctx=tool_ctx,
        max_tool_steps=args.max_tool_steps,
        verbose_tools=args.verbose_tools,
        tools_builder=tools_builder,
    )

    if args.json:
        sys.stdout.write(json.dumps(result.raw_last_response, indent=2))
        sys.stdout.write("\n")
        return 0

    sys.stdout.write(result.text)
    if not result.text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    client = _client_from_config(cfg)

    temperature = args.temperature if args.temperature is not None else cfg.temperature
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.max_tokens

    tool_ctx = None
    tools_builder = None
    include_shell = False
    if not args.no_tools:
        tool_root = args.tool_root if args.tool_root is not None else cfg.tool_root
        workspace_dir = args.workspace if getattr(args, "workspace", None) is not None else cfg.workspace_dir
        enable_shell = None
        if getattr(args, "no_shell", False):
            enable_shell = False
        elif getattr(args, "enable_shell", False):
            enable_shell = True
        enable_custom_tools = None
        if getattr(args, "no_custom_tools", False):
            enable_custom_tools = False
        elif getattr(args, "enable_custom_tools", False):
            enable_custom_tools = True
        tool_ctx = ToolContext.from_config_values(
            tool_root=tool_root,
            workspace_dir=workspace_dir,
            max_read_bytes=cfg.tool_max_read_bytes,
            max_write_bytes=cfg.tool_max_write_bytes,
            max_list_entries=cfg.tool_max_list_entries,
            enable_shell=enable_shell,
            enable_custom_tools=enable_custom_tools,
            custom_tools_dir=getattr(args, "custom_tools_dir", None),
        )
        include_shell = bool(tool_ctx.shell_enabled)
        tools_builder = lambda: tool_schemas(
            include_shell=tool_ctx.shell_enabled,
            include_custom=tool_ctx.custom_tools_enabled,
            tool_ctx=tool_ctx,
        )

    base_system = args.system
    if base_system is None and not args.no_tools:
        base_system = DEFAULT_TOOL_SYSTEM
    skills_block = "" if args.no_skills else render_enabled_skills_system(cfg)
    tool_root_p = tool_ctx.root if tool_ctx is not None else _resolve_tool_root(cfg, args.tool_root)
    workspace_p = tool_ctx.workspace if tool_ctx is not None else _resolve_workspace_root(cfg, getattr(args, "workspace", None))
    core = _core_system_prelude(
        cfg,
        tool_root=tool_root_p,
        workspace=workspace_p,
        enable_tools=(not args.no_tools),
        tool_ctx=tool_ctx,
        include_shell=include_shell,
    )
    system_prompt = _build_system_prompt(core + (base_system or ""), skills_block)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    sys.stdout.write("freeclaw chat (Ctrl-D to exit)\n")
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            sys.stdout.write("\n")
            return 0
        if not user:
            continue
        messages.append({"role": "user", "content": user})

        result = run_agent(
            client=client,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_tools=(not args.no_tools),
            tool_ctx=tool_ctx,
            max_tool_steps=args.max_tool_steps,
            verbose_tools=args.verbose_tools,
            tools_builder=tools_builder,
        )

        sys.stdout.write(result.text)
        if not result.text.endswith("\n"):
            sys.stdout.write("\n")


def cmd_config_init(args: argparse.Namespace) -> int:
    p = write_default_config(args.path)
    sys.stdout.write(f"Wrote {p}\n")
    return 0


def cmd_config_env_init(args: argparse.Namespace) -> int:
    from pathlib import Path

    p = Path(args.path).expanduser() if args.path else (Path.cwd() / "config" / ".env")
    if p.exists():
        raise SystemExit(f"Env file already exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(
            [
                "# freeclaw env",
                "# Put secrets here. Do not commit this file.",
                "",
                "NVIDIA_API_KEY=",
                "",
                "# Optional:",
                "# FREECLAW_MODEL=meta/llama-3.1-8b-instruct",
                "# FREECLAW_BASE_URL=https://integrate.api.nvidia.com/v1",
                "# FREECLAW_TOOL_ROOT=.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sys.stdout.write(f"Wrote {p}\n")
    return 0


def cmd_discord(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    client = _client_from_config(cfg)

    temperature = args.temperature if args.temperature is not None else cfg.temperature
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.max_tokens

    tool_ctx = None
    tools_builder = None
    include_shell = False
    if not args.no_tools:
        tool_root = args.tool_root if args.tool_root is not None else cfg.tool_root
        workspace_dir = args.workspace if getattr(args, "workspace", None) is not None else cfg.workspace_dir
        enable_shell = None
        if getattr(args, "no_shell", False):
            enable_shell = False
        elif getattr(args, "enable_shell", False):
            enable_shell = True
        enable_custom_tools = None
        if getattr(args, "no_custom_tools", False):
            enable_custom_tools = False
        elif getattr(args, "enable_custom_tools", False):
            enable_custom_tools = True
        tool_ctx = ToolContext.from_config_values(
            tool_root=tool_root,
            workspace_dir=workspace_dir,
            max_read_bytes=cfg.tool_max_read_bytes,
            max_write_bytes=cfg.tool_max_write_bytes,
            max_list_entries=cfg.tool_max_list_entries,
            enable_shell=enable_shell,
            enable_custom_tools=enable_custom_tools,
            custom_tools_dir=getattr(args, "custom_tools_dir", None),
        )
        include_shell = bool(tool_ctx.shell_enabled)
        tools_builder = lambda: tool_schemas(
            include_shell=tool_ctx.shell_enabled,
            include_custom=tool_ctx.custom_tools_enabled,
            tool_ctx=tool_ctx,
        )

    base_system = args.system
    if base_system is None and not args.no_tools:
        base_system = DEFAULT_TOOL_SYSTEM
    skills_block = "" if args.no_skills else render_enabled_skills_system(cfg)
    tool_root_p = tool_ctx.root if tool_ctx is not None else _resolve_tool_root(cfg, args.tool_root)
    workspace_p = tool_ctx.workspace if tool_ctx is not None else _resolve_workspace_root(cfg, getattr(args, "workspace", None))
    core = _core_system_prelude(
        cfg,
        tool_root=tool_root_p,
        workspace=workspace_p,
        enable_tools=(not args.no_tools),
        tool_ctx=tool_ctx,
        include_shell=include_shell,
    )
    system_prompt = _build_system_prompt(core + (base_system or ""), skills_block)

    asyncio.run(
        run_discord_bot(
            token=args.token,
            prefix=(args.prefix or cfg.discord_prefix),
            respond_to_all=(
                bool(args.respond_to_all)
                if args.respond_to_all is not None
                else bool(cfg.discord_respond_to_all)
            ),
            system_prompt=system_prompt,
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_ctx=tool_ctx,
            enable_tools=(not args.no_tools),
            max_tool_steps=args.max_tool_steps,
            verbose_tools=args.verbose_tools,
            tools_builder=tools_builder,
            history_messages=(
                args.history_messages
                if args.history_messages is not None
                else cfg.discord_history_messages
            ),
        )
    )
    return 0


def cmd_onboard(args: argparse.Namespace) -> int:
    env_path = run_onboarding(
        config_path=args.config,
        force=args.force,
        env_path_hint=args.env_file,
    )
    # Ensure this process sees the new key immediately.
    load_dotenv(env_path, override=False)
    return 0


def cmd_skill_list(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    skills = iter_skills(cfg)
    enabled = set(cfg.enabled_skills or [])
    for s in skills:
        mark = "*" if s.name in enabled else " "
        sys.stdout.write(f"{mark} {s.name}\n")
    return 0


def cmd_skill_show(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    sk = find_skill(cfg, args.name)
    if not sk:
        raise SystemExit(f"Skill not found: {args.name}")
    sys.stdout.write(sk.skill_md.read_text(encoding="utf-8", errors="replace"))
    return 0


def _update_enabled_skills(args: argparse.Namespace, add: bool) -> int:
    cfg_path, data = load_config_dict(args.config)
    cfg = ClawConfig.from_dict(data)
    enabled = list(cfg.enabled_skills or [])
    if add:
        if args.name not in enabled:
            enabled.append(args.name)
    else:
        enabled = [x for x in enabled if x != args.name]
    data = cfg.to_dict()
    data["enabled_skills"] = enabled
    save_config_dict(cfg_path, data)
    return 0


def cmd_skill_enable(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if not find_skill(cfg, args.name):
        raise SystemExit(f"Skill not found: {args.name}")
    return _update_enabled_skills(args, add=True)


def cmd_skill_disable(args: argparse.Namespace) -> int:
    return _update_enabled_skills(args, add=False)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Pre-parse so --env-file can override auto-load.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=None)
    pre.add_argument("--config", default=None)
    pre.add_argument("--no-onboard", action="store_true")
    pre_args, _ = pre.parse_known_args(argv)

    if pre_args.env_file:
        load_dotenv(Path(pre_args.env_file).expanduser(), override=False)
    else:
        autoload_dotenv()

    parser = argparse.ArgumentParser(prog="freeclaw")
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

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single prompt.")
    p_run.add_argument("prompt")
    p_run.add_argument("--system", default=None, help="Optional system prompt.")
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--max-tokens", type=int, default=None)
    p_run.add_argument("--no-tools", action="store_true", help="Disable built-in tools.")
    g_run_shell = p_run.add_mutually_exclusive_group()
    g_run_shell.add_argument(
        "--enable-shell",
        action="store_true",
        help="Enable sh_exec tool (command execution). (default: enabled)",
    )
    g_run_shell.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable sh_exec tool (command execution).",
    )
    g_run_custom = p_run.add_mutually_exclusive_group()
    g_run_custom.add_argument(
        "--enable-custom-tools",
        action="store_true",
        help="Enable loading custom tools from workspace/.freeclaw/tools. (default: enabled)",
    )
    g_run_custom.add_argument(
        "--no-custom-tools",
        action="store_true",
        help="Disable loading custom tools (overrides FREECLAW_ENABLE_CUSTOM_TOOLS).",
    )
    p_run.add_argument(
        "--custom-tools-dir",
        default=None,
        help="Override custom tools dir (must be within workspace). Default: workspace/.freeclaw/tools",
    )
    p_run.add_argument("--no-skills", action="store_true", help="Disable injecting enabled skills into system prompt.")
    p_run.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory for persona.md/tools.md/custom tool specs (default from config/env).",
    )
    p_run.add_argument(
        "--tool-root",
        default=None,
        help="Tool filesystem root (default from config/env; relative roots are resolved from cwd).",
    )
    p_run.add_argument("--max-tool-steps", type=int, default=12, help="Max tool loop steps.")
    p_run.add_argument("--verbose-tools", action="store_true", help="Log tool calls/results.")
    p_run.add_argument("--json", action="store_true", help="Print raw provider JSON response.")
    p_run.set_defaults(func=cmd_run)

    p_chat = sub.add_parser("chat", help="Interactive chat.")
    p_chat.add_argument("--system", default=None, help="Optional system prompt.")
    p_chat.add_argument("--temperature", type=float, default=None)
    p_chat.add_argument("--max-tokens", type=int, default=None)
    p_chat.add_argument("--no-tools", action="store_true", help="Disable built-in tools.")
    g_chat_shell = p_chat.add_mutually_exclusive_group()
    g_chat_shell.add_argument(
        "--enable-shell",
        action="store_true",
        help="Enable sh_exec tool (command execution). (default: enabled)",
    )
    g_chat_shell.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable sh_exec tool (command execution).",
    )
    g_chat_custom = p_chat.add_mutually_exclusive_group()
    g_chat_custom.add_argument(
        "--enable-custom-tools",
        action="store_true",
        help="Enable loading custom tools from workspace/.freeclaw/tools. (default: enabled)",
    )
    g_chat_custom.add_argument(
        "--no-custom-tools",
        action="store_true",
        help="Disable loading custom tools (overrides FREECLAW_ENABLE_CUSTOM_TOOLS).",
    )
    p_chat.add_argument(
        "--custom-tools-dir",
        default=None,
        help="Override custom tools dir (must be within workspace). Default: workspace/.freeclaw/tools",
    )
    p_chat.add_argument("--no-skills", action="store_true", help="Disable injecting enabled skills into system prompt.")
    p_chat.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory for persona.md/tools.md/custom tool specs (default from config/env).",
    )
    p_chat.add_argument(
        "--tool-root",
        default=None,
        help="Tool filesystem root (default from config/env; relative roots are resolved from cwd).",
    )
    p_chat.add_argument("--max-tool-steps", type=int, default=12, help="Max tool loop steps.")
    p_chat.add_argument("--verbose-tools", action="store_true", help="Log tool calls/results.")
    p_chat.set_defaults(func=cmd_chat)

    p_cfg = sub.add_parser("config", help="Config utilities.")
    sub_cfg = p_cfg.add_subparsers(dest="cfg_cmd", required=True)
    p_init = sub_cfg.add_parser("init", help="Write default config to ./config/config.json.")
    p_init.add_argument(
        "--path",
        default=None,
        help="Override config path (default: ./config/config.json).",
    )
    p_init.set_defaults(func=cmd_config_init)

    p_env = sub_cfg.add_parser("env-init", help="Write a template .env file (default: ./config/.env).")
    p_env.add_argument("--path", default=None, help="Path to write (default: ./config/.env).")
    p_env.set_defaults(func=cmd_config_env_init)

    p_discord = sub.add_parser("discord", help="Run a Discord bot that chats using the configured provider.")
    p_discord.add_argument("--token", default=None, help="Discord bot token (or DISCORD_BOT_TOKEN).")
    p_discord.add_argument("--prefix", default=None, help="Command prefix trigger (default from config).")
    p_discord.add_argument(
        "--respond-to-all",
        action="store_true",
        default=None,
        help="Respond to every non-bot message in channels/DMs the bot can read (no prefix/mention required).",
    )
    p_discord.add_argument("--system", default=None, help="Optional system prompt.")
    p_discord.add_argument("--temperature", type=float, default=None)
    p_discord.add_argument("--max-tokens", type=int, default=None)
    p_discord.add_argument("--no-tools", action="store_true", help="Disable built-in tools.")
    g_discord_shell = p_discord.add_mutually_exclusive_group()
    g_discord_shell.add_argument(
        "--enable-shell",
        action="store_true",
        help="Enable sh_exec tool (command execution). (default: enabled)",
    )
    g_discord_shell.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable sh_exec tool (command execution).",
    )
    g_discord_custom = p_discord.add_mutually_exclusive_group()
    g_discord_custom.add_argument(
        "--enable-custom-tools",
        action="store_true",
        help="Enable loading custom tools from workspace/.freeclaw/tools. (default: enabled)",
    )
    g_discord_custom.add_argument(
        "--no-custom-tools",
        action="store_true",
        help="Disable loading custom tools (overrides FREECLAW_ENABLE_CUSTOM_TOOLS).",
    )
    p_discord.add_argument(
        "--custom-tools-dir",
        default=None,
        help="Override custom tools dir (must be within workspace). Default: workspace/.freeclaw/tools",
    )
    p_discord.add_argument("--no-skills", action="store_true", help="Disable injecting enabled skills into system prompt.")
    p_discord.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory for persona.md/tools.md/custom tool specs (default from config/env).",
    )
    p_discord.add_argument(
        "--tool-root",
        default=None,
        help="Tool filesystem root (default from config/env; relative roots are resolved from cwd).",
    )
    p_discord.add_argument("--max-tool-steps", type=int, default=12, help="Max tool loop steps.")
    p_discord.add_argument("--verbose-tools", action="store_true", help="Log tool calls/results to stderr.")
    p_discord.add_argument(
        "--history-messages",
        type=int,
        default=None,
        help="How many non-system messages to retain per channel/DM session (default from config).",
    )
    p_discord.set_defaults(func=cmd_discord)

    p_onboard = sub.add_parser("onboard", help="Run first-run setup wizard.")
    p_onboard.add_argument("--force", action="store_true", help="Run even if already onboarded.")
    p_onboard.set_defaults(func=cmd_onboard)

    p_skill = sub.add_parser("skill", help="Skills: list/show/enable/disable")
    sub_skill = p_skill.add_subparsers(dest="skill_cmd", required=True)
    p_s_list = sub_skill.add_parser("list", help="List available skills (* = enabled).")
    p_s_list.set_defaults(func=cmd_skill_list)
    p_s_show = sub_skill.add_parser("show", help="Show a skill SKILL.md.")
    p_s_show.add_argument("name")
    p_s_show.set_defaults(func=cmd_skill_show)
    p_s_en = sub_skill.add_parser("enable", help="Enable a skill by name.")
    p_s_en.add_argument("name")
    p_s_en.set_defaults(func=cmd_skill_enable)
    p_s_dis = sub_skill.add_parser("disable", help="Disable a skill by name.")
    p_s_dis.add_argument("name")
    p_s_dis.set_defaults(func=cmd_skill_disable)

    args = parser.parse_args(argv)

    # If the full parser received --env-file, ensure it is loaded (in case user passes argv directly to main()).
    if args.env_file and (not pre_args.env_file or pre_args.env_file != args.env_file):
        load_dotenv(Path(args.env_file).expanduser(), override=False)

    if not args.no_onboard and not _should_skip_onboarding(args.cmd):
        cfg_path, data = load_config_dict(args.config)
        cfg = ClawConfig.from_dict(data)
        if not cfg.onboarded:
            env_path = run_onboarding(
                config_path=args.config,
                force=False,
                env_path_hint=args.env_file,
            )
            load_dotenv(env_path, override=False)
    return int(args.func(args))
