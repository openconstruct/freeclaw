# freeclaw

Minimal OpenClaw-like CLI that defaults to NVIDIA NIM (OpenAI-compatible `/v1/chat/completions`).

## Quickstart

1. Set your NVIDIA API key:
   - `export NVIDIA_API_KEY=...`
   - Or put it in `.env` (recommended): `NVIDIA_API_KEY=...`

2. Run:
   - `python -m freeclaw run "Hello from NIM"`
   - `python -m freeclaw chat`
   - `python -m freeclaw discord` (Discord bot)

On first run, `freeclaw` will launch an interactive onboarding menu to set up `.env`, Discord defaults, tool root, and a starter skill.
Onboarding also writes a `startup.md` into your configured tool root containing the bot name/tone and basic Discord notes.
freeclaw will create `persona.md` and `tools.md` in the tool root if missing, and include them in the system prompt (so the bot can persist its persona and accurately list available tools).

## Configuration

Defaults are NIM-first. Override with env vars:

- `FREECLAW_PROVIDER` (default: `nim`)
- `FREECLAW_BASE_URL` (default for NIM: `https://integrate.api.nvidia.com/v1`)
- `FREECLAW_MODEL` (default: auto-detect from `/v1/models`, or error if not supported)
- `FREECLAW_TEMPERATURE` (default: `0.7`)
- `FREECLAW_MAX_TOKENS` (default: `1024`)
- `FREECLAW_TOOL_ROOT` (default: `.`; tools are constrained to this root)
- `FREECLAW_TOOL_MAX_READ_BYTES` (default: `200000`)
- `FREECLAW_TOOL_MAX_WRITE_BYTES` (default: `2000000`)
- `FREECLAW_TOOL_MAX_LIST_ENTRIES` (default: `2000`)
- `FREECLAW_ASSISTANT_NAME` (default: `Freeclaw`)
- `FREECLAW_ASSISTANT_TONE` (default: `Direct, pragmatic, concise...`)
- `FREECLAW_WEB_MAX_BYTES` (default: `500000`)
- `FREECLAW_WEB_USER_AGENT` (default: `freeclaw/0.1.0 (+https://github.com/freeclaw/freeclaw)`)
- `FREECLAW_MEMORY_DB` (default: `~/.config/freeclaw/memory.sqlite3`; global across CLI + Discord)
- `FREECLAW_ENABLE_SHELL` (default: enabled; set to `false` to disable `sh_exec`)
- `FREECLAW_SHELL_TIMEOUT_S` (default: `20.0`)
- `FREECLAW_SHELL_MAX_OUTPUT_BYTES` (default: `200000`)
- `FREECLAW_SHELL_BLOCK_NETWORK` (default: `false`; blocks `curl/wget/ssh/nc/...` in `sh_exec`)
- `FREECLAW_ENABLE_CUSTOM_TOOLS` (default: enabled; set to `false` to disable loading custom tools from disk)
- `FREECLAW_CUSTOM_TOOLS_DIR` (default: `<tool_root>/.freeclaw/tools`; must be within tool_root)
- `FREECLAW_CUSTOM_TOOLS_BLOCK_NETWORK` (default: `false`; if `true`, blocks `curl/wget/ssh/nc/...` for custom tools)

API key env vars (first found wins):

- `NVIDIA_API_KEY`
- `NIM_API_KEY`
- `NVIDIA_NIM_API_KEY`

You can also generate a config file:

- `python -m freeclaw config init`

This writes `~/.config/freeclaw/config.json`.

## .env

`freeclaw` will auto-load env vars from (first match wins):

- `FREECLAW_ENV_FILE` if set
- `./.env`
- `~/.config/freeclaw/.env` (or `$XDG_CONFIG_HOME/freeclaw/.env`)

Create a template:

```bash
python -m freeclaw config env-init
```

Or re-run the onboarding menu any time:

```bash
python -m freeclaw onboard
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
- Memory: `memory_*` stores notes in a local SQLite DB. This memory is global across runs (CLI + Discord) unless you point `FREECLAW_MEMORY_DB` elsewhere. Notes can be `pinned` or given a `ttl_seconds` expiry.
- Shell: `sh_exec` executes commands in `tool_root` (enabled by default; disable with `--no-shell` or `FREECLAW_ENABLE_SHELL=false`). Network tools are allowed by default; set `FREECLAW_SHELL_BLOCK_NETWORK=true` to block `curl/wget/ssh/nc/...`.
- Custom: additional tools are loaded from JSON specs under `<tool_root>/.freeclaw/tools` (enabled by default; disable with `--no-custom-tools` or `FREECLAW_ENABLE_CUSTOM_TOOLS=false`).

### Custom Tools (Dynamic)

freeclaw loads tools from:

- `<tool_root>/.freeclaw/tools/<name>.json`
- `<tool_root>/.freeclaw/tools/<name>/tool.json`

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
- `!claw reset` clear the per-channel/DM session

Slash commands (if the bot has `applications.commands` scope and sync succeeds):

- `/claw` chat
- `/reset` clear session
- `/tools` list tools
- `/model` show model
- `/persona show` show persona
- `/persona set` set persona
- `/memory search` search saved memory

Discord conversation sessions persist across bot restarts in the same SQLite DB used for memory (default: `~/.config/freeclaw/memory.sqlite3`).
