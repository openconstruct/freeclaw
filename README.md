# freeclaw
<img width="1024" height="1024" alt="feeeclaw" src="https://github.com/user-attachments/assets/8f0c4556-3272-40fe-8ed4-28d1ac12792a" />

Minimal OpenClaw-like CLI that supports NVIDIA NIM, OpenRouter, and Groq (OpenAI-compatible `/v1/chat/completions`).

## Install

INstall 

    git clone https://github.com/openconstruct/freeclaw
    cd freeclaw
    pip install -r requirements.txt
    python -m freeclaw onboard ( you will need an API key and a dicord bot key, I explain in detail here: https://medium.com/@jerryhowell/free-openclaw-alternative-freeclaw-ecf537abbcd0)
    python -m freeclaw discord ( or chat if you want to chat via SSH )



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
- Timer behavior with Discord:
  - If no local timer-api is running on the sidecar port (default `3000`), Discord auto-starts a timer sidecar.
  - If timer-api is already running on that port, Discord skips sidecar start.
- Workspace docs:
  - `google.md` is auto-created in each bot workspace with step-by-step Google Cloud OAuth setup instructions.

## Configuration

Defaults are NIM-first. Override with env vars:

- `FREECLAW_PROVIDER` (default: `nim`)
- `FREECLAW_BASE_URL` (default for NIM: `https://integrate.api.nvidia.com/v1`; OpenRouter: `https://openrouter.ai/api/v1`; Groq: `https://api.groq.com/openai/v1`)
- `FREECLAW_MODEL` (default: auto-detect from `/v1/models`, or error if not supported)
- `FREECLAW_TEMPERATURE` (default: `0.7`)
- `FREECLAW_MAX_TOKENS` (default: `1024`)
- `FREECLAW_MAX_TOOL_STEPS` (default: `50`)
- `FREECLAW_TOOL_ROOT` (default: `.`; tools are constrained to this root)
- `FREECLAW_WORKSPACE_DIR` (default: `workspace`; state files like `persona.md`, `tasks.md`, and `.freeclaw/*` live here)
- `FREECLAW_TASK_TIMER_MINUTES` (default: `30`; 0 disables the task timer)
- `FREECLAW_WEB_UI_ENABLED` (default: `true`; enables Web UI routes in `timer-api`)
- `FREECLAW_WEB_UI_PORT` (default: `3000`; default `timer-api` bind port)
- `FREECLAW_TIMER_DISCORD_NOTIFY` (default: `true`; when a timer run executes due tasks, send a Discord summary if destination is configured)
- `FREECLAW_TIMER_DISCORD_CHANNEL_ID` (optional; channel id for timer notifications via bot token; if unset, freeclaw tries to auto-detect the bot's most recent channel)
- `FREECLAW_TIMER_DISCORD_BOT_TOKEN` (optional; overrides `DISCORD_BOT_TOKEN`/`FREECLAW_DISCORD_TOKEN` for timer notifications)
- `FREECLAW_TIMER_DISCORD_WEBHOOK_URL` (optional; if set, webhook is used for timer notifications)
- `FREECLAW_TIMER_DISCORD_TIMEOUT_S` (default: `10.0`; timeout for timer notification HTTP calls)
- `FREECLAW_TIMER_DISCORD_CONTEXT` (default: `true`; include recent Discord session history in timer-run prompts)
- `FREECLAW_TIMER_DISCORD_CONTEXT_MAX_MESSAGES` (default: `24`; max recent user/assistant messages to include)
- `FREECLAW_TIMER_DISCORD_CONTEXT_MAX_CHARS` (default: `12000`; char budget for injected Discord context)
- `FREECLAW_DISCORD_SESSION_SCOPE` (default: `channel`; options: `channel`, `user`, `global`; controls how Discord conversation history is shared)
- `FREECLAW_TOOL_MAX_READ_BYTES` (default: `200000`)
- `FREECLAW_TOOL_MAX_WRITE_BYTES` (default: `2000000`)
- `FREECLAW_TOOL_MAX_LIST_ENTRIES` (default: `2000`)
- `FREECLAW_ASSISTANT_NAME` (default: `Freebot`)
- `FREECLAW_ASSISTANT_TONE` (default: `Direct, pragmatic, concise...`)
- `FREECLAW_WEB_MAX_BYTES` (default: `500000`)
- `FREECLAW_WEB_USER_AGENT` (default: `freeclaw/0.1.0 (+https://github.com/freeclaw/freeclaw)`)
- `FREECLAW_MEMORY_DB` (default: `./config/memory.sqlite3`; shared across CLI + Discord in the same project)
- `FREECLAW_GOOGLE_CLIENT_ID` (required for Google connect; OAuth web client id)
- `FREECLAW_GOOGLE_CLIENT_SECRET` (OAuth web client secret)
- `FREECLAW_GOOGLE_REDIRECT_URI` (required for Google connect; e.g. `http://<PUBLIC_IP>:3000/v1/oauth/callback`)
- `FREECLAW_GOOGLE_DEFAULT_SCOPES` (default: calendar+gmail readonly + `openid email`)
- `FREECLAW_GOOGLE_OAUTH_TIMEOUT_S` (default: `20.0`; timeout for Google OAuth HTTP calls)
- `FREECLAW_GOOGLE_CONNECT_EXPIRES_S` (default: `900`; pending connect flow expiry in seconds)
- `FREECLAW_ENABLE_SHELL` (default: enabled; set to `false` to disable `sh_exec`)
- `FREECLAW_SHELL_TIMEOUT_S` (default: `20.0`)
- `FREECLAW_SHELL_MAX_OUTPUT_BYTES` (default: `200000`)
- `FREECLAW_SHELL_BLOCK_NETWORK` (default: `false`; blocks `curl/wget/ssh/nc/...` in `sh_exec`)
- `FREECLAW_ENABLE_CUSTOM_TOOLS` (default: enabled; set to `false` to disable loading custom tools from disk)
- `FREECLAW_CUSTOM_TOOLS_DIR` (default: `<workspace>/skills/tools`; must be within workspace)
- `FREECLAW_CUSTOM_TOOLS_BLOCK_NETWORK` (default: `false`; if `true`, blocks `curl/wget/ssh/nc/...` for custom tools)
- `FREECLAW_LOG_LEVEL` (default: `info`)
- `FREECLAW_LOG_FILE` (default: `./config/freeclaw.log`; set to empty string to disable file logging)
- `FREECLAW_LOG_FORMAT` (default: `text`; options: `text`, `jsonl`)

Workspace safety:
- If `workspace_dir` resolves to filesystem root (`/`), freeclaw now falls back to `<cwd>/workspace` to prevent creating `/.freeclaw`.

API key env vars (first found wins):

- `NVIDIA_API_KEY`
- `NIM_API_KEY`
- `NVIDIA_NIM_API_KEY`
- `OPENROUTER_API_KEY` (OpenRouter provider)
- `OPENAI_API_KEY` (fallback for OpenRouter provider)
- `GROQ_API_KEY` (Groq provider)
- `GROQ_KEY` (fallback for Groq provider)

You can also generate a config file:

- `python -m freeclaw config init`

This writes `./config/config.json` by default (override via `--path` or `FREECLAW_CONFIG_DIR`).

Config utilities:

- `python -m freeclaw config show` (effective config; includes env overrides)
- `python -m freeclaw config show --raw` (config.json only)
- `python -m freeclaw config set max_tool_steps 30`
- `python -m freeclaw config validate`

## Logging

By default, logs are written to `./config/freeclaw.log` and stderr.
The logfile rotates at 100KB with up to 3 backups (`freeclaw.log.1`..`.3`), pruning older entries automatically.

CLI flags:

```bash
python -m freeclaw --log-level info --log-file ./config/freeclaw.log chat
```

JSON lines output:

```bash
python -m freeclaw --log-level info --log-format jsonl --log-file ./config/freeclaw.jsonl discord
```

Or env vars:

- `FREECLAW_LOG_LEVEL=debug|info|warning|error`
- `FREECLAW_LOG_FILE=./config/freeclaw.log`
- `FREECLAW_LOG_FORMAT=text|jsonl`

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

## Groq

To use Groq:

- Set `FREECLAW_PROVIDER=groq`
- Set `GROQ_API_KEY=...` (or `GROQ_KEY=...`)
- Optionally set `FREECLAW_BASE_URL=https://api.groq.com/openai/v1`

List Groq models:

```bash
python -m freeclaw models --provider groq
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
- Local timer API: `timer_api_get` queries the local `timer-api` server (`/api/system/metrics`, `/timer/status`, `/health`) and is localhost-only.
- Google:
  - Email: `google_email_list`, `google_email_get`, `google_email_send`
  - Calendar: `google_calendar_list`, `google_calendar_create`
  - These tools use the connected Google account for a specific `bot_id` + `discord_user_id` pair and require appropriate OAuth scopes.
- Memory: `memory_*` stores notes in a local SQLite DB. This memory is shared across CLI + Discord for the current project by default; override with `FREECLAW_MEMORY_DB`. Notes can be `pinned` or given a `ttl_seconds` expiry.
- Task scheduler: `task_*` manages recurring `tasks.md` entries (list/add/update/enable/disable/run-now) in structured form.
- Docs: `doc_ingest`/`doc_inject`, `doc_search`, `doc_get`, `doc_list`, `doc_delete` build and manage a persistent workspace document index (text + PDF via `pypdf`).
- Shell: `sh_exec` executes commands in `tool_root` (enabled by default; disable with `--no-shell` or `FREECLAW_ENABLE_SHELL=false`). Network tools are allowed by default; set `FREECLAW_SHELL_BLOCK_NETWORK=true` to block `curl/wget/ssh/nc/...`.
- Custom: additional tools are loaded from JSON specs under `<workspace>/skills/tools` (enabled by default; disable with `--no-custom-tools` or `FREECLAW_ENABLE_CUSTOM_TOOLS=false`).

### Custom Tools (Dynamic)

freeclaw loads tools from:

- `<workspace>/skills/tools/<name>.json`
- `<workspace>/skills/tools/<name>/tool.json`

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
By default, timer runs also inject bounded recent Discord session context (latest saved channel session) for continuity.

Run it:

```bash
python -m freeclaw task-timer
```

Or run a single tick (useful for cron/systemd timers):

```bash
python -m freeclaw task-timer --once
```

### Timer API (Server-Managed)

If you want freeclaw itself to handle scheduling (instead of systemd), run:

```bash
python -m freeclaw timer-api
```

For a single authoritative scheduler across base + all configured agents:

```bash
python -m freeclaw timer-api --all-agents
```

This starts an HTTP server with a background scheduler that checks `tasks.md` and wakes the model when tasks are due.
By default it binds to `0.0.0.0:3000`. You can disable Web UI routes in onboarding, or at runtime with `--no-web-ui`.
The same server also handles Google OAuth callback requests on `/v1/oauth/callback`.

Timer-to-Discord delivery:
- If `FREECLAW_TIMER_DISCORD_WEBHOOK_URL` is set, due-task run summaries are posted to that webhook.
- Otherwise, if a bot token (`FREECLAW_TIMER_DISCORD_BOT_TOKEN` or `DISCORD_BOT_TOKEN`) is set, freeclaw posts to `FREECLAW_TIMER_DISCORD_CHANNEL_ID` when provided, or auto-detects the latest channel used by that bot.

Useful endpoints:

- `GET /` (Web UI dashboard)
- `GET /health`
- `GET /v1/oauth/callback` (Google OAuth redirect target)
- `GET /timer/status`
- `GET /api/system/metrics` (CPU/RAM/temp/GPU/VRAM/uptime/bandwidth/storage/top processes + active bot token usage with 24h/7d history)
- `POST /timer/tick` (force a tick now)
- `POST /timer/enable`
- `POST /timer/disable`
- `POST /timer/config` with JSON body like `{"minutes":30}`

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

`.[discord]` now includes `pypdf`, so the bot can parse text from attached PDFs.

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

When users send message attachments, the bot will download and parse:

- PDF files (`.pdf`)
- Common text/code formats (`.txt`, `.md`, `.csv`, `.json`, `.html`, `.py`, etc.)

Parsed attachment text is appended to the model prompt for that message.

The bot can also send local files in its reply when the model emits a directive line:

- `[[send_file:relative/path/from/workspace/or/tool_root]]`
- Optional rename: `[[send_file:path as output-name.ext]]`

For safety, outbound file paths are restricted to the configured workspace/tool roots.

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
- `/google connect` start Google account link for this bot/user
- `/google poll` poll Google connect status
- `/google status` show linked Google account for this bot/user
- `/google disconnect` unlink Google account for this bot/user

Discord conversation sessions persist across bot restarts in the same SQLite DB used for memory (default: `./config/memory.sqlite3`; override with `FREECLAW_MEMORY_DB`).
