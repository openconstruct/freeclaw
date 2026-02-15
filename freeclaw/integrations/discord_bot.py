import asyncio
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import run_agent
from ..providers.nim import NimChatClient
from ..tools import ToolContext, tool_schemas
from ..tools.memory import memory_search
from ..paths import memory_db_path as _default_memory_db_path


DISCORD_MESSAGE_LIMIT = 2000


def _split_discord_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]

    lines = text.splitlines(True)  # keepends
    chunks: list[str] = []
    cur = ""
    in_code = False
    open_fence = "```"

    def close_fence() -> str:
        return "\n```\n" if not cur.endswith("\n") else "```\n"

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        if in_code:
            cf = close_fence()
            if len(cur) + len(cf) <= limit:
                cur += cf
        chunks.append(cur.rstrip("\n"))
        cur = ""

    def start_new_chunk() -> None:
        nonlocal cur
        if in_code:
            hdr = open_fence + "\n"
            cur = hdr

    for line in lines:
        # Track code fences (common markdown style, line-based).
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:
                open_fence = stripped if stripped else "```"
                in_code = True
            else:
                in_code = False
                open_fence = "```"

        # If a single line is too large, split it.
        while line and len(line) > limit:
            head = line[:limit]
            line = line[limit:]
            if cur:
                flush()
                start_new_chunk()
            chunks.append(head.rstrip("\n"))
            if line:
                start_new_chunk()

        reserve = len("\n```\n") if in_code else 0
        if cur and (len(cur) + len(line) > (limit - reserve)):
            flush()
            start_new_chunk()

        if len(line) > limit:
            # Safety; should be handled above.
            chunks.append(line[:limit].rstrip("\n"))
            continue

        cur += line

    if cur:
        flush()
    return [c for c in chunks if c != ""]


def _strip_bot_mention(content: str, bot_user_id: int) -> str:
    # Typical mention formats: <@123>, <@!123>
    content = content.strip()
    mention_re = re.compile(rf"^<@!?{bot_user_id}>\\s*")
    return mention_re.sub("", content).strip()


@dataclass
class DiscordSession:
    messages: list[dict[str, Any]]


async def run_discord_bot(
    *,
    token: str | None,
    prefix: str,
    respond_to_all: bool,
    system_prompt: str | None,
    client: NimChatClient,
    temperature: float,
    max_tokens: int,
    tool_ctx: ToolContext | None,
    enable_tools: bool,
    max_tool_steps: int,
    verbose_tools: bool,
    history_messages: int,
    tools_builder: Any = None,
) -> None:
    try:
        import discord  # type: ignore
        from discord import app_commands  # type: ignore
    except Exception as e:
        raise SystemExit(
            "discord.py is not installed. Install with:\n"
            "  pip install -e '.[discord]'\n"
            f"Import error: {e}"
        )

    bot_token = token or os.getenv("DISCORD_BOT_TOKEN") or os.getenv("FREECLAW_DISCORD_TOKEN")
    if not bot_token:
        raise SystemExit("Missing Discord bot token. Set DISCORD_BOT_TOKEN or pass --token.")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    intents.dm_messages = True

    db_path = None
    if tool_ctx is not None:
        db_path = tool_ctx.memory_db_path
    else:
        env_db = os.getenv("FREECLAW_MEMORY_DB")
        db_path = (
            Path(env_db).expanduser().resolve()
            if env_db and env_db.strip()
            else _default_memory_db_path().resolve()
        )

    def _db_connect() -> sqlite3.Connection:
        assert db_path is not None
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_session_schema(con: sqlite3.Connection) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS discord_sessions (
              channel_id INTEGER PRIMARY KEY,
              messages_json TEXT NOT NULL,
              updated_at INTEGER NOT NULL
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_discord_sessions_updated_at ON discord_sessions(updated_at);")
        con.commit()

    def _load_session_messages(channel_id: int) -> list[dict[str, Any]] | None:
        try:
            with _db_connect() as con:
                _init_session_schema(con)
                row = con.execute(
                    "SELECT messages_json FROM discord_sessions WHERE channel_id=?;",
                    (int(channel_id),),
                ).fetchone()
                if not row:
                    return None
                raw = row[0]
                data = json.loads(raw) if isinstance(raw, str) else None
                if not isinstance(data, list):
                    return None
                out: list[dict[str, Any]] = []
                for m in data:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role")
                    content = m.get("content")
                    if role not in {"system", "user", "assistant"}:
                        continue
                    if not isinstance(content, str):
                        continue
                    out.append({"role": str(role), "content": content})
                return out
        except Exception:
            return None

    def _save_session_messages(channel_id: int, msgs: list[dict[str, Any]]) -> None:
        try:
            payload = json.dumps(msgs, ensure_ascii=True)
            now = int(time.time())
            with _db_connect() as con:
                _init_session_schema(con)
                con.execute(
                    """
                    INSERT INTO discord_sessions(channel_id, messages_json, updated_at)
                    VALUES(?,?,?)
                    ON CONFLICT(channel_id) DO UPDATE SET
                      messages_json=excluded.messages_json,
                      updated_at=excluded.updated_at;
                    """,
                    (int(channel_id), payload, now),
                )
                con.commit()
        except Exception:
            return

    def _delete_session(channel_id: int) -> None:
        try:
            with _db_connect() as con:
                _init_session_schema(con)
                con.execute("DELETE FROM discord_sessions WHERE channel_id=?;", (int(channel_id),))
                con.commit()
        except Exception:
            return

    sessions: dict[int, DiscordSession] = {}
    locks: dict[int, asyncio.Lock] = {}

    def get_lock(channel_id: int) -> asyncio.Lock:
        lk = locks.get(channel_id)
        if lk is None:
            lk = asyncio.Lock()
            locks[channel_id] = lk
        return lk

    def get_session(channel_id: int) -> DiscordSession:
        sess = sessions.get(channel_id)
        if sess is None:
            msgs = _load_session_messages(channel_id) or []
            # Normalize system prompt to current run.
            if system_prompt:
                if msgs and msgs[0].get("role") == "system":
                    msgs[0] = {"role": "system", "content": system_prompt}
                else:
                    msgs = [{"role": "system", "content": system_prompt}] + [m for m in msgs if m.get("role") != "system"]
            else:
                msgs = [m for m in msgs if m.get("role") != "system"]
            sessions[channel_id] = DiscordSession(messages=msgs)
            return sessions[channel_id]
        return sess

    def truncate_in_place(msgs: list[dict[str, Any]]) -> None:
        if history_messages <= 0:
            return
        system_msgs = [m for m in msgs if m.get("role") == "system"][:1]
        rest = [m for m in msgs if m.get("role") != "system"]
        if len(rest) > history_messages:
            rest = rest[-history_messages:]
        msgs[:] = system_msgs + rest

    class ClawClient(discord.Client):
        def __init__(self, *, intents: "discord.Intents") -> None:  # type: ignore[name-defined]
            super().__init__(intents=intents)
            self.tree = app_commands.CommandTree(self)

        async def setup_hook(self) -> None:
            try:
                await self.tree.sync()
            except Exception as e:
                print(f"discord slash command sync failed: {e}")

        async def on_ready(self) -> None:
            assert self.user is not None
            print(f"freeclaw discord logged in as {self.user} (id={self.user.id})")
            print(f"prefix: {prefix!r}")
            print(f"respond_to_all: {bool(respond_to_all)}")

        async def _run_prompt(self, *, channel: Any, channel_id: int, prompt: str) -> list[str]:
            lk = get_lock(channel_id)
            async with lk:
                sess = get_session(channel_id)
                sess.messages.append({"role": "user", "content": prompt})
                truncate_in_place(sess.messages)

                # Pass a copy so run_agent can append tool messages without polluting the persisted chat history.
                agent_msgs = [dict(m) for m in sess.messages]
                result = await asyncio.to_thread(
                    run_agent,
                    client=client,
                    messages=agent_msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_tools=enable_tools,
                    tool_ctx=tool_ctx,
                    max_tool_steps=max_tool_steps,
                    verbose_tools=verbose_tools,
                    tools_builder=tools_builder,
                )

                text = (result.text or "").strip()
                sess.messages.append({"role": "assistant", "content": text})
                truncate_in_place(sess.messages)
                _save_session_messages(channel_id, sess.messages)

                return _split_discord_message(text or "")

        async def on_message(self, message: "discord.Message") -> None:  # type: ignore[name-defined]
            if message.author.bot:
                return
            if message.content is None:
                return

            assert self.user is not None
            content = message.content.strip()
            if not content:
                return
            channel_id = message.channel.id

            triggered = False
            prompt = ""
            explicit_cmd = False

            if content.startswith(prefix):
                triggered = True
                explicit_cmd = True
                prompt = content[len(prefix) :].strip()
            elif self.user.mentioned_in(message) and content.startswith("<@"):
                # Mention trigger: "@bot do X"
                triggered = True
                explicit_cmd = True
                prompt = _strip_bot_mention(content, self.user.id)
            elif respond_to_all:
                triggered = True
                prompt = content

            if not triggered:
                return

            if explicit_cmd and not prompt:
                await message.channel.send(f"Usage: `{prefix} <prompt>` or `{prefix} reset`")  # type: ignore[attr-defined]
                return

            # Only allow reset via an explicit command to avoid accidental clears
            # when running in "respond to all" mode.
            if explicit_cmd and prompt.lower() in {"reset", "restart", "clear"}:
                sessions.pop(channel_id, None)
                _delete_session(channel_id)
                await message.channel.send("Session cleared.")  # type: ignore[attr-defined]
                return

            async with message.channel.typing():  # type: ignore[attr-defined]
                chunks = await self._run_prompt(channel=message.channel, channel_id=channel_id, prompt=prompt)
            for ch in chunks:
                await message.channel.send(ch)  # type: ignore[attr-defined]

    client_app = ClawClient(intents=intents)

    @client_app.tree.command(name="claw", description="Chat with the bot.")  # type: ignore[attr-defined]
    async def claw_cmd(interaction: "discord.Interaction", prompt: str) -> None:  # type: ignore[name-defined]
        await interaction.response.defer(thinking=True)  # type: ignore[attr-defined]
        channel = interaction.channel
        if channel is None:
            await interaction.followup.send("No channel found for this interaction.")  # type: ignore[attr-defined]
            return
        chunks = await client_app._run_prompt(channel=channel, channel_id=channel.id, prompt=prompt.strip())
        for ch in chunks:
            await interaction.followup.send(ch)  # type: ignore[attr-defined]

    @client_app.tree.command(name="reset", description="Clear the conversation for this channel/DM.")  # type: ignore[attr-defined]
    async def reset_cmd(interaction: "discord.Interaction") -> None:  # type: ignore[name-defined]
        channel = interaction.channel
        if channel is None:
            await interaction.response.send_message("No channel found.")  # type: ignore[attr-defined]
            return
        sessions.pop(channel.id, None)
        _delete_session(channel.id)
        await interaction.response.send_message("Session cleared.")  # type: ignore[attr-defined]

    @client_app.tree.command(name="tools", description="List available tools.")  # type: ignore[attr-defined]
    async def tools_cmd(interaction: "discord.Interaction") -> None:  # type: ignore[name-defined]
        if not enable_tools or tool_ctx is None:
            await interaction.response.send_message("Tools are disabled for this bot run.")  # type: ignore[attr-defined]
            return
        items = []
        for t in tool_schemas(
            include_shell=tool_ctx.shell_enabled,
            include_custom=tool_ctx.custom_tools_enabled,
            tool_ctx=tool_ctx,
        ):
            fn = t.get("function") if isinstance(t, dict) else None
            if not isinstance(fn, dict):
                continue
            nm = fn.get("name")
            desc = fn.get("description") or ""
            if isinstance(nm, str) and nm.strip():
                items.append(f"- {nm}: {str(desc).strip()}")
        text = "Available tools:\n" + "\n".join(items)
        chunks = _split_discord_message(text)
        await interaction.response.send_message(chunks[0])  # type: ignore[attr-defined]
        for ch in chunks[1:]:
            await interaction.followup.send(ch)  # type: ignore[attr-defined]

    @client_app.tree.command(name="model", description="Show the configured model.")  # type: ignore[attr-defined]
    async def model_cmd(interaction: "discord.Interaction") -> None:  # type: ignore[name-defined]
        m = client.model or "auto"
        await interaction.response.send_message(f"Model: `{m}`")  # type: ignore[attr-defined]

    persona_group = app_commands.Group(name="persona", description="View/update persona.md in tool_root.")  # type: ignore[attr-defined]

    @persona_group.command(name="show", description="Show persona.md (truncated).")  # type: ignore[attr-defined]
    async def persona_show_cmd(interaction: "discord.Interaction") -> None:  # type: ignore[name-defined]
        if tool_ctx is None:
            await interaction.response.send_message("Tools are disabled; no tool_root available.")  # type: ignore[attr-defined]
            return
        p = tool_ctx.root / "persona.md"
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            txt = "(persona.md not found)"
        msg = "```md\n" + txt[:1800] + ("\n... (truncated)\n" if len(txt) > 1800 else "\n") + "```"
        await interaction.response.send_message(msg)  # type: ignore[attr-defined]

    @persona_group.command(name="set", description="Overwrite persona.md.")  # type: ignore[attr-defined]
    async def persona_set_cmd(interaction: "discord.Interaction", content: str) -> None:  # type: ignore[name-defined]
        if tool_ctx is None:
            await interaction.response.send_message("Tools are disabled; no tool_root available.")  # type: ignore[attr-defined]
            return
        b = (content or "").encode("utf-8")
        if len(b) > 20_000:
            await interaction.response.send_message("persona.md content too large (max 20000 bytes).")  # type: ignore[attr-defined]
            return
        p = tool_ctx.root / "persona.md"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            await interaction.response.send_message(f"Failed to write persona.md: {e}")  # type: ignore[attr-defined]
            return
        await interaction.response.send_message("Updated persona.md.")  # type: ignore[attr-defined]

    mem_group = app_commands.Group(name="memory", description="Search stored memory.")  # type: ignore[attr-defined]

    @mem_group.command(name="search", description="Search the global SQLite memory store.")  # type: ignore[attr-defined]
    async def memory_search_cmd(interaction: "discord.Interaction", query: str, limit: int = 5) -> None:  # type: ignore[name-defined]
        if tool_ctx is None:
            await interaction.response.send_message("Tools are disabled; memory is unavailable.")  # type: ignore[attr-defined]
            return
        res = memory_search(tool_ctx, query=query, limit=int(limit))
        items = res.get("results") if isinstance(res, dict) else None
        if not isinstance(items, list) or not items:
            await interaction.response.send_message("No matches.")  # type: ignore[attr-defined]
            return
        lines = []
        for it in items[:10]:
            if isinstance(it, dict):
                k = it.get("key") or it.get("id")
                c = str(it.get("content", "")).replace("\n", " ")
                if len(c) > 140:
                    c = c[:140] + "..."
                lines.append(f"- {k}: {c}")
        txt = "Memory results:\n" + "\n".join(lines)
        await interaction.response.send_message(txt)  # type: ignore[attr-defined]

    client_app.tree.add_command(mem_group)  # type: ignore[attr-defined]
    client_app.tree.add_command(persona_group)  # type: ignore[attr-defined]

    await client_app.start(bot_token)
