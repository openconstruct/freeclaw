# freeclaw

Minimal OpenClaw-like CLI that defaults to NVIDIA NIM (OpenAI-compatible `/v1/chat/completions`).

## Install

Editable install from this repo:

```bash
python -m pip install -e .
```

Optional features (extras):

- Web search (`web_search`): `python -m pip install -e '.[web]'` (installs `ddgs`)
- Discord bot: `python -m pip install -e '.[discord]'`
- Both: `python -m pip install -e '.[web,discord]'`

Note: `web` is just the name of an extra dependency group, not a Python package.

## Quickstart

1. Set your NVIDIA API key:
   - `export NVIDIA_API_KEY=...`
   - Or put it in `./config/.env` (recommended): `NVIDIA_API_KEY=...`

2. Run:
   - `python -m freeclaw run "Hello from NIM"`
   - `python -m freeclaw chat`
   - `python -m freeclaw task-timer`
   - `python -m freeclaw discord` (Discord bot)

On first run, `freeclaw` will launch an interactive onboarding menu to set up `./config/.env`, tool root, workspace, Discord defaults, and a starter skill.
Onboarding writes `startup.md`, `persona.md`, `tools.md`, and `tasks.md` into your workspace directory (default: `./workspace`).
freeclaw includes `persona.md` and `tools.md` in the system prompt (so the bot can persist its persona and accurately list available tools).

Reset everything (delete local config + workspaces; fresh install):

```bash
python -m freeclaw reset
```

## Multiple Agents

You can create multiple named agent profiles (each with its own workspace/model/persona/Discord settings):

- Create an agent: `python -m freeclaw onboard createagent <name>`
  - Writes `./config/agents/<name>/config.json`
  - Writes `./config/agents/<name>/.env` (includes a per-agent `FREECLAW_MEMORY_DB` so Discord sessions don't collide)
  - Default workspace: `./workspace/<name>/`
- Run an agent: `python -m freeclaw --agent <name> chat` (also works with `run`, `task-timer`, `discord`)

Discord default behavior:

- `python -m freeclaw discord` launches all Discord bots: base config + every `config/agents/<name>/` profile.
- Run only the base config bot: `python -m freeclaw discord --no-all-agents`
- Run only one agent: `python -m freeclaw --agent <name> discord`

## Configuration

Defaults are NIM-first. Override with env vars:

- `FREECLAW_PROVIDER` (default: `nim`)
- `FREECLAW_BASE_URL` (default for NIM: `https://integrate.api.nvidia.com/v1`; OpenRouter: `https://openrouter.ai/api/v1`)
- `FREECLAW_MODEL` (default: auto-detect from `/v1/models`, or error if not supported)
- `FREECLAW_TEMPERATURE` (default: `0.7`)
- `FREECLAW_MAX_TOKENS` (default: `1024`)
- `FREECLAW_MAX_TOOL_STEPS` (default: `50`)
- `FREECLAW_TOOL_ROOT` (default: `.`; tools are constrained to this root)
- `FREECLAW_TASK_TIMER_MINUTES` (default: `30`; 0 disables the task timer)
- `FREECLAW_TOOL_MAX_READ_BYTES` (default: `200000`)
- `FREECLAW_TOOL_MAX_WRITE_BYTES` (default: `2000000`)
- `FREECLAW_TOOL_MAX_LIST_ENTRIES` (default: `2000`)
- `FREECLAW_ASSISTANT_NAME` (default: `Freeclaw`)
- `FREECLAW_ASSISTANT_TONE` (default: `Direct, pragmatic, concise...`)
- `FREECLAW_WEB_MAX_BYTES` (default: `500000`)
- `FREECLAW_WEB_USER_AGENT` (default: `freeclaw/0.1.0 (+https://github.com/freeclaw/freeclaw)`)
- `FREECLAW_MEMORY_DB` (default: `./config/memory.sqlite3`; shared across CLI + Discord in the same project)
- `FREECLAW_ENABLE_SHELL` (default: enabled; set to `false` to disable `sh_exec`)
- `FREECLAW_SHELL_TIMEOUT_S` (default: `20.0`)
- `FREECLAW_SHELL_MAX_OUTPUT_BYTES` (default: `200000`)
- `FREECLAW_SHELL_BLOCK_NETWORK` (default: `false`; blocks `curl/wget/ssh/nc/...` in `sh_exec`)
- `FREECLAW_ENABLE_CUSTOM_TOOLS` (default: enabled; set to `false` to disable loading custom tools from disk)
- `FREECLAW_CUSTOM_TOOLS_DIR` (default: `<workspace>/.freeclaw/tools`; must be within workspace)
- `FREECLAW_CUSTOM_TOOLS_BLOCK_NETWORK` (default: `false`; if `true`, blocks `curl/wget/ssh/nc/...` for custom tools)
- `FREECLAW_LOG_LEVEL` (default: `warning`)
- `FREECLAW_LOG_FILE` (default: unset; logs go to stderr)

API key env vars (first found wins):

- `NVIDIA_API_KEY`
- `NIM_API_KEY`
- `NVIDIA_NIM_API_KEY`
- `OPENROUTER_API_KEY` (OpenRouter provider)
- `OPENAI_API_KEY` (fallback for OpenRouter provider)

You can also generate a config file:

- `python -m freeclaw config init`

This writes `./config/config.json` by default (override via `--path` or `FREECLAW_CONFIG_DIR`).

Config utilities:

- `python -m freeclaw config show` (effective config; includes env overrides)
- `python -m freeclaw config show --raw` (config.json only)
- `python -m freeclaw config set max_tool_steps 30`
- `python -m freeclaw config validate`

## Logging

CLI flags:

```bash
python -m freeclaw --log-level info --log-file ./config/freeclaw.log chat
```

Or env vars:

- `FREECLAW_LOG_LEVEL=debug|info|warning|error`
- `FREECLAW_LOG_FILE=./config/freeclaw.log`

## .env

`freeclaw` will auto-load env vars from (first match wins):

- `FREECLAW_ENV_FILE` if set
- `./config/.env`
- `./.env`

Create a template:

```bash
python -m freeclaw config env-init
```

Or re-run the onboarding menu any time:

```bash
python -m freeclaw onboard
```

## OpenRouter

To use OpenRouter:

- Set `FREECLAW_PROVIDER=openrouter`
- Set `OPENROUTER_API_KEY=...` (or `OPENAI_API_KEY=...`)
- Optionally set `FREECLAW_BASE_URL=https://openrouter.ai/api/v1`

List only free OpenRouter models (best-effort):

```bash
python -m freeclaw models --provider openrouter --free-only
```

## Tools

By default, `run` and `chat` enable basic filesystem tools (OpenAI function-calling compatible):

- `fs_read`: read a text file (optionally line-ranged)
- `fs_write`: write a text file (overwrite/append)
- `fs_list`: list a directory (optionally recursive, bounded)
- `fs_mkdir`: create a directory
- `fs_rm`: remove a file/dir within tool_root
- `fs_mv`: move/rename within tool_root
- `fs_cp`: copy within tool_root
- `text_search`: search for text within tool_root (bounded)

Disable tools with `--no-tools`.

Additional tools:

- Web: `web_search` (DuckDuckGo via `ddgs`) and `web_fetch` (public http(s) URL fetch; blocks localhost/private IPs).
- HTTP: `http_request_json` calls a public http(s) JSON API and returns parsed JSON (blocks localhost/private IPs).
- Memory: `memory_*` stores notes in a local SQLite DB. This memory is shared across CLI + Discord for the current project by default; override with `FREECLAW_MEMORY_DB`. Notes can be `pinned` or given a `ttl_seconds` expiry.
- Shell: `sh_exec` executes commands in `tool_root` (enabled by default; disable with `--no-shell` or `FREECLAW_ENABLE_SHELL=false`). Network tools are allowed by default; set `FREECLAW_SHELL_BLOCK_NETWORK=true` to block `curl/wget/ssh/nc/...`.
- Custom: additional tools are loaded from JSON specs under `<workspace>/.freeclaw/tools` (enabled by default; disable with `--no-custom-tools` or `FREECLAW_ENABLE_CUSTOM_TOOLS=false`).

### Custom Tools (Dynamic)

freeclaw loads tools from:

- `<workspace>/.freeclaw/tools/<name>.json`
- `<workspace>/.freeclaw/tools/<name>/tool.json`

Each file is a JSON object like:

```json
{
  "name": "hello_tool",
  "description": "Say hello using /bin/echo",
  "type": "command",
  "workdir": "tool_root",
  "argv": ["echo", "hello", "{{who}}"],
  "stdin": null,
  "env": { "EXAMPLE": "{{who}}" },
  "parameters": {
    "type": "object",
    "properties": { "who": { "type": "string" } },
    "required": ["who"]
  }
}
```

Template vars in `argv`:

- `{{arg_name}}` substitutes the function argument value (non-scalars are JSON-encoded).
- `{{args_json}}` substitutes the full arguments object as JSON.

## Task Timer

The task timer reads `<workspace>/tasks.md` and looks for unchecked checklist items (`- [ ] ...`).
It only calls the model when there are pending tasks.

Run it:

```bash
python -m freeclaw task-timer
```

Or run a single tick (useful for cron/systemd timers):

```bash
python -m freeclaw task-timer --once
```

Example `tasks.md`:

```md
# tasks

## Tasks
- [ ] Update README with install instructions
- [ ] Add a config validate command
```

### systemd (example)

```ini
[Unit]
Description=freeclaw task timer

[Service]
Type=simple
WorkingDirectory=/path/to/your/project
Environment=FREECLAW_ENV_FILE=/path/to/your/project/config/.env
ExecStart=/usr/bin/python3 -m freeclaw task-timer --minutes 10
Restart=on-failure
RestartSec=5
```

### cron (example)

```cron
*/5 * * * * cd /path/to/your/project && FREECLAW_ENV_FILE=./config/.env /usr/bin/python3 -m freeclaw task-timer --once
```

## Skills

Skills are folders containing a `SKILL.md`. `freeclaw` can inject enabled skills into the system prompt.

Commands:

- `python -m freeclaw skill list`
- `python -m freeclaw skill show <name>`
- `python -m freeclaw skill enable <name>`
- `python -m freeclaw skill disable <name>`

## Discord

Install dependencies:

```bash
pip install -e ".[discord]"
```

Web tools (DDG search + URL fetch) require:

```bash
pip install -e ".[web]"
```

Set your bot token:

- `export DISCORD_BOT_TOKEN=...`

Run the bot:

```bash
python -m freeclaw discord --prefix "!claw" --tool-root /home/jerr
```

In Discord, you must enable the **Message Content Intent** for the bot if you want it to read and respond to normal messages (including prefix-based commands).

If you want the bot to respond to every message (no prefix needed):

```bash
python -m freeclaw discord --respond-to-all --tool-root /home/jerr
```

Commands:

- `!claw <prompt>` chat in the current channel/DM
- `!claw new` start a new conversation (keeps settings)
- `!claw reset` clear the per-channel/DM session (messages + settings)
- `!claw help` show available commands

Slash commands (if the bot has `applications.commands` scope and sync succeeds):

- `/help` show available commands
- `/claw` chat
- `/reset` clear session (messages + settings)
- `/new` start a new conversation (keeps settings)
- `/tools` list tools
- `/model` show or set model override for this channel/DM
- `/temp` show or set temperature override for this channel/DM
- `/tokens` show or set max_tokens override for this channel/DM
- `/persona show` show persona
- `/persona set` set persona
- `/memory search` search saved memory

Discord conversation sessions persist across bot restarts in the same SQLite DB used for memory (default: `./config/memory.sqlite3`; override with `FREECLAW_MEMORY_DB`).
