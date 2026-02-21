import json
from typing import Any, Callable

from .custom import custom_tool_schemas, dispatch_custom_tool_call
from .fs import (
    ToolContext,
    fs_cp,
    fs_diff,
    fs_glob,
    fs_list,
    fs_mkdir,
    fs_mv,
    fs_read,
    fs_rm,
    fs_stat,
    fs_write,
)
from .google_local import (
    google_calendar_create,
    google_calendar_list,
    google_email_get,
    google_email_list,
    google_email_send,
)
from .http import http_request_json
from .memory import memory_add, memory_delete, memory_get, memory_search
from .doc_ingest import doc_delete, doc_get, doc_ingest, doc_list, doc_search
from .shell import sh_exec
from .search import text_search
from .task_scheduler import task_add, task_disable, task_enable, task_list, task_run_now, task_update
from .timer_api import timer_api_get
from .web import web_fetch, web_search


def tool_schemas(
    *,
    include_shell: bool = True,
    include_custom: bool = False,
    tool_ctx: ToolContext | None = None,
    extra_tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # OpenAI-compatible function tools.
    reserved: set[str] = {
        "text_search",
        "fs_read",
        "fs_write",
        "fs_mkdir",
        "fs_list",
        "fs_stat",
        "fs_glob",
        "fs_diff",
        "fs_rm",
        "fs_mv",
        "fs_cp",
        "web_search",
        "web_fetch",
        "http_request_json",
        "timer_api_get",
        "google_email_list",
        "google_email_get",
        "google_email_send",
        "google_calendar_list",
        "google_calendar_create",
        "memory_add",
        "memory_get",
        "memory_search",
        "memory_delete",
        "task_list",
        "task_add",
        "task_update",
        "task_disable",
        "task_enable",
        "task_run_now",
        "doc_ingest",
        "doc_inject",
        "doc_search",
        "doc_get",
        "doc_list",
        "doc_delete",
        "sh_exec",
    }
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "text_search",
                "description": "Search for text within tool_root (bounded). Returns file/line/col and the matching line.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "regex": {"type": "boolean", "default": False},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "include_glob": {"type": ["string", "null"], "default": None},
                        "exclude_glob": {"type": ["string", "null"], "default": None},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                        "max_files": {"type": "integer", "minimum": 1, "default": 200},
                        "context_lines": {"type": "integer", "minimum": 0, "maximum": 10, "default": 0},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_read",
                "description": "Read a UTF-8 text file within tool_root. Use start_line/end_line to keep reads small.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to tool_root."},
                        "start_line": {"type": "integer", "minimum": 1, "default": 1},
                        "end_line": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                            "default": None,
                            "description": "Inclusive end line; null means to EOF.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_write",
                "description": "Write a UTF-8 text file within tool_root (overwrite or append). Creates parent dirs by default.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to tool_root."},
                        "content": {"type": "string"},
                        "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
                        "make_parents": {"type": "boolean", "default": True},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_mkdir",
                "description": "Create a directory within tool_root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path relative to tool_root."},
                        "parents": {"type": "boolean", "default": True},
                        "exist_ok": {"type": "boolean", "default": True},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_list",
                "description": "List files under a directory within tool_root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "recursive": {"type": "boolean", "default": False},
                        "max_depth": {"type": "integer", "minimum": 0, "default": 2},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_stat",
                "description": "Get file metadata (exists/type/size/mtime) within tool_root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to tool_root."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_glob",
                "description": "Glob for paths under tool_root (pattern relative to tool_root).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "max_results": {"type": ["integer", "null"], "minimum": 1, "default": None},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_diff",
                "description": "Show a unified diff between an existing file and proposed new content (within tool_root).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to tool_root."},
                        "content": {"type": "string", "description": "Proposed full file content."},
                        "context_lines": {"type": "integer", "minimum": 0, "default": 3},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_rm",
                "description": "Remove a file or directory within tool_root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to tool_root."},
                        "recursive": {"type": "boolean", "default": False},
                        "missing_ok": {"type": "boolean", "default": True},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_mv",
                "description": "Move/rename a file or directory within tool_root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "overwrite": {"type": "boolean", "default": False},
                        "make_parents": {"type": "boolean", "default": True},
                    },
                    "required": ["src", "dst"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_cp",
                "description": "Copy a file or directory within tool_root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "recursive": {"type": "boolean", "default": False},
                        "overwrite": {"type": "boolean", "default": False},
                        "make_parents": {"type": "boolean", "default": True},
                    },
                    "required": ["src", "dst"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using DuckDuckGo (ddgs). Returns a small list of results (title/url/snippet).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                        "safesearch": {"type": "string", "enum": ["off", "moderate", "strict"], "default": "moderate"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch a public http(s) URL and return its text (HTML is converted to plain text). Blocks localhost/private IPs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_bytes": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                            "default": None,
                            "description": "Optional override of response size limit; default from FREECLAW_WEB_MAX_BYTES.",
                        },
                        "timeout_s": {"type": "number", "minimum": 1, "default": 20.0},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "http_request_json",
                "description": "Make an HTTP request to a public http(s) URL and return parsed JSON (blocks localhost/private IPs).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "GET"},
                        "headers": {"type": ["object", "null"], "default": None},
                        "json_body": {"type": ["object", "array", "string", "number", "boolean", "null"], "default": None},
                        "timeout_s": {"type": "number", "minimum": 0.1, "default": 20.0},
                        "max_bytes": {"type": ["integer", "null"], "default": None},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "timer_api_get",
                "description": "Query the local timer-api (localhost-only) for health/status/system metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "enum": ["system_metrics", "status", "health"],
                            "default": "system_metrics",
                        },
                        "host": {
                            "type": "string",
                            "default": "127.0.0.1",
                            "description": "Localhost only: 127.0.0.1, localhost, or ::1",
                        },
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535, "default": 3000},
                        "timeout_s": {"type": "number", "minimum": 0.1, "default": 5.0},
                        "max_bytes": {"type": ["integer", "null"], "default": None},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "google_email_list",
                "description": "List recent Gmail messages for a linked bot/user account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bot_id": {"type": "integer"},
                        "discord_user_id": {"type": "integer"},
                        "query": {"type": ["string", "null"], "default": None},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
                    },
                    "required": ["bot_id", "discord_user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "google_email_get",
                "description": "Get a Gmail message by message_id for a linked bot/user account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bot_id": {"type": "integer"},
                        "discord_user_id": {"type": "integer"},
                        "message_id": {"type": "string"},
                    },
                    "required": ["bot_id", "discord_user_id", "message_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "google_email_send",
                "description": "Send a Gmail message for a linked bot/user account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bot_id": {"type": "integer"},
                        "discord_user_id": {"type": "integer"},
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body_text": {"type": "string"},
                        "cc": {"type": ["string", "null"], "default": None},
                        "bcc": {"type": ["string", "null"], "default": None},
                    },
                    "required": ["bot_id", "discord_user_id", "to", "subject", "body_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "google_calendar_list",
                "description": "List upcoming calendar events for a linked bot/user account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bot_id": {"type": "integer"},
                        "discord_user_id": {"type": "integer"},
                        "calendar_id": {"type": "string", "default": "primary"},
                        "time_min": {"type": ["string", "null"], "default": None},
                        "time_max": {"type": ["string", "null"], "default": None},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
                    },
                    "required": ["bot_id", "discord_user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "google_calendar_create",
                "description": "Create a calendar event for a linked bot/user account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bot_id": {"type": "integer"},
                        "discord_user_id": {"type": "integer"},
                        "summary": {"type": "string"},
                        "start_iso": {"type": "string"},
                        "end_iso": {"type": "string"},
                        "calendar_id": {"type": "string", "default": "primary"},
                        "description": {"type": ["string", "null"], "default": None},
                        "location": {"type": ["string", "null"], "default": None},
                    },
                    "required": ["bot_id", "discord_user_id", "summary", "start_iso", "end_iso"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_add",
                "description": "Store a memory note in a local SQLite DB (supports optional key/tags). If key exists, updates it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "key": {"type": ["string", "null"], "default": None},
                        "tags": {"type": ["array", "null"], "items": {"type": "string"}, "default": None},
                        "meta": {"type": ["object", "null"], "default": None},
                        "pinned": {"type": "boolean", "default": False},
                        "ttl_seconds": {"type": ["integer", "null"], "minimum": 1, "default": None},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_get",
                "description": "Get a memory item by id or key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["integer", "null"], "default": None},
                        "key": {"type": ["string", "null"], "default": None},
                        "include_expired": {"type": "boolean", "default": False},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search memory notes by query (uses SQLite FTS5 when available).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                        "include_expired": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_delete",
                "description": "Delete a memory item by id or key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["integer", "null"], "default": None},
                        "key": {"type": ["string", "null"], "default": None},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_list",
                "description": "List recurring tasks from workspace/tasks.md with ids for later updates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_disabled": {"type": "boolean", "default": True},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_add",
                "description": "Add a recurring task to workspace/tasks.md using <minutes>-<task> format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "minutes": {"type": "integer", "minimum": 1},
                        "task": {"type": "string"},
                        "enabled": {"type": "boolean", "default": True},
                        "allow_duplicate": {"type": "boolean", "default": False},
                    },
                    "required": ["minutes", "task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_update",
                "description": "Update an existing task by task_id or exact task text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": ["integer", "null"], "default": None},
                        "task": {"type": ["string", "null"], "default": None},
                        "new_minutes": {"type": ["integer", "null"], "minimum": 1, "default": None},
                        "new_task": {"type": ["string", "null"], "default": None},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_disable",
                "description": "Disable a recurring task in tasks.md by task_id or exact task text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": ["integer", "null"], "default": None},
                        "task": {"type": ["string", "null"], "default": None},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_enable",
                "description": "Enable a disabled recurring task in tasks.md by task_id or exact task text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": ["integer", "null"], "default": None},
                        "task": {"type": ["string", "null"], "default": None},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_run_now",
                "description": "Mark a task as due immediately for the next timer tick by clearing its last-run state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": ["integer", "null"], "default": None},
                        "task": {"type": ["string", "null"], "default": None},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doc_ingest",
                "description": "Ingest a local text/PDF document into a persistent workspace index for later retrieval/search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to tool_root."},
                        "key": {"type": ["string", "null"], "default": None},
                        "title": {"type": ["string", "null"], "default": None},
                        "replace": {"type": "boolean", "default": True},
                        "max_chars": {"type": "integer", "minimum": 1000, "maximum": 1000000, "default": 200000},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doc_inject",
                "description": "Alias of doc_ingest: ingest a local text/PDF document into persistent workspace index.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to tool_root."},
                        "key": {"type": ["string", "null"], "default": None},
                        "title": {"type": ["string", "null"], "default": None},
                        "replace": {"type": "boolean", "default": True},
                        "max_chars": {"type": "integer", "minimum": 1000, "maximum": 1000000, "default": 200000},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doc_search",
                "description": "Search previously ingested documents by keyword/full text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doc_get",
                "description": "Fetch an ingested document by id or key (optionally include content excerpt).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["integer", "null"], "default": None},
                        "key": {"type": ["string", "null"], "default": None},
                        "include_content": {"type": "boolean", "default": True},
                        "max_chars": {"type": "integer", "minimum": 500, "maximum": 200000, "default": 8000},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doc_list",
                "description": "List ingested documents with pagination and optional text filter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "query": {"type": ["string", "null"], "default": None},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "doc_delete",
                "description": "Delete an ingested document by id or key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["integer", "null"], "default": None},
                        "key": {"type": ["string", "null"], "default": None},
                    },
                },
            },
        },
    ]
    if include_shell:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "sh_exec",
                    "description": "Execute a command (argv array) in tool_root and return stdout/stderr. Enabled by default; disable via FREECLAW_ENABLE_SHELL=false or --no-shell.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "argv": {"type": "array", "items": {"type": "string"}},
                            "stdin": {"type": ["string", "null"], "default": None},
                            "env": {
                                "type": ["object", "null"],
                                "default": None,
                                "description": "Optional environment variables to set for the child process.",
                            },
                            "timeout_s": {"type": ["number", "null"], "default": None},
                            "max_output_bytes": {"type": ["integer", "null"], "default": None},
                        },
                        "required": ["argv"],
                    },
                },
            }
        )
    if include_custom and tool_ctx is not None:
        tools.extend(custom_tool_schemas(tool_ctx, reserved_names=reserved))
    if extra_tools:
        tools.extend(list(extra_tools))
    return tools


def _as_opt_int(args: dict[str, Any], key: str) -> int | None:
    return None if args.get(key) is None else int(args.get(key))


def _as_opt_str(args: dict[str, Any], key: str) -> str | None:
    return None if args.get(key) is None else str(args.get(key))


def _dispatch_http_request_json(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    mb = args.get("max_bytes")
    hdrs = args.get("headers")
    if hdrs is not None and not isinstance(hdrs, dict):
        raise ValueError("headers must be an object (map of string to string)")
    return http_request_json(
        ctx,
        url=str(args.get("url", "")),
        method=str(args.get("method", "GET")),
        headers=(None if hdrs is None else {str(k): str(v) for k, v in hdrs.items()}),
        json_body=args.get("json_body"),
        timeout_s=float(args.get("timeout_s", 20.0)),
        max_bytes=(None if mb is None else int(mb)),
    )


def _dispatch_sh_exec(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    argv = args.get("argv")
    if not isinstance(argv, list):
        raise ValueError("argv must be a list of strings")
    env = args.get("env")
    if env is not None and not isinstance(env, dict):
        raise ValueError("env must be an object (map of string to string)")
    return sh_exec(
        ctx,
        argv=[str(x) for x in argv],
        stdin=_as_opt_str(args, "stdin"),
        env=(None if env is None else {str(k): str(v) for k, v in env.items()}),
        timeout_s=(None if args.get("timeout_s") is None else float(args.get("timeout_s"))),
        max_output_bytes=(
            None if args.get("max_output_bytes") is None else int(args.get("max_output_bytes"))
        ),
    )


_TOOL_DISPATCHERS: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
    "text_search": lambda ctx, args: text_search(
        ctx,
        query=str(args.get("query", "")),
        path=str(args.get("path", ".")),
        regex=bool(args.get("regex", False)),
        case_sensitive=bool(args.get("case_sensitive", False)),
        include_glob=_as_opt_str(args, "include_glob"),
        exclude_glob=_as_opt_str(args, "exclude_glob"),
        max_results=int(args.get("max_results", 20)),
        max_files=int(args.get("max_files", 200)),
        context_lines=int(args.get("context_lines", 0)),
    ),
    "fs_read": lambda ctx, args: fs_read(
        ctx,
        path=str(args.get("path", "")),
        start_line=int(args.get("start_line", 1)),
        end_line=_as_opt_int(args, "end_line"),
    ),
    "fs_write": lambda ctx, args: fs_write(
        ctx,
        path=str(args.get("path", "")),
        content=str(args.get("content", "")),
        mode=str(args.get("mode", "overwrite")),
        make_parents=bool(args.get("make_parents", True)),
    ),
    "fs_mkdir": lambda ctx, args: fs_mkdir(
        ctx,
        path=str(args.get("path", "")),
        parents=bool(args.get("parents", True)),
        exist_ok=bool(args.get("exist_ok", True)),
    ),
    "fs_list": lambda ctx, args: fs_list(
        ctx,
        path=str(args.get("path", ".")),
        recursive=bool(args.get("recursive", False)),
        max_depth=int(args.get("max_depth", 2)),
    ),
    "fs_stat": lambda ctx, args: fs_stat(ctx, path=str(args.get("path", ""))),
    "fs_glob": lambda ctx, args: fs_glob(
        ctx,
        pattern=str(args.get("pattern", "")),
        max_results=_as_opt_int(args, "max_results"),
    ),
    "fs_diff": lambda ctx, args: fs_diff(
        ctx,
        path=str(args.get("path", "")),
        content=str(args.get("content", "")),
        context_lines=int(args.get("context_lines", 3)),
    ),
    "fs_rm": lambda ctx, args: fs_rm(
        ctx,
        path=str(args.get("path", "")),
        recursive=bool(args.get("recursive", False)),
        missing_ok=bool(args.get("missing_ok", True)),
    ),
    "fs_mv": lambda ctx, args: fs_mv(
        ctx,
        src=str(args.get("src", "")),
        dst=str(args.get("dst", "")),
        overwrite=bool(args.get("overwrite", False)),
        make_parents=bool(args.get("make_parents", True)),
    ),
    "fs_cp": lambda ctx, args: fs_cp(
        ctx,
        src=str(args.get("src", "")),
        dst=str(args.get("dst", "")),
        recursive=bool(args.get("recursive", False)),
        overwrite=bool(args.get("overwrite", False)),
        make_parents=bool(args.get("make_parents", True)),
    ),
    "web_search": lambda ctx, args: web_search(
        ctx,
        query=str(args.get("query", "")),
        max_results=int(args.get("max_results", 5)),
        safesearch=str(args.get("safesearch", "moderate")),
    ),
    "web_fetch": lambda ctx, args: web_fetch(
        ctx,
        url=str(args.get("url", "")),
        max_bytes=_as_opt_int(args, "max_bytes"),
        timeout_s=float(args.get("timeout_s", 20.0)),
    ),
    "http_request_json": _dispatch_http_request_json,
    "timer_api_get": lambda ctx, args: timer_api_get(
        ctx,
        endpoint=str(args.get("endpoint", "system_metrics")),
        host=str(args.get("host", "127.0.0.1")),
        port=int(args.get("port", 3000)),
        timeout_s=float(args.get("timeout_s", 5.0)),
        max_bytes=_as_opt_int(args, "max_bytes"),
    ),
    "google_email_list": lambda ctx, args: google_email_list(
        ctx,
        bot_id=int(args.get("bot_id", 0)),
        discord_user_id=int(args.get("discord_user_id", 0)),
        query=_as_opt_str(args, "query"),
        max_results=int(args.get("max_results", 10)),
    ),
    "google_email_get": lambda ctx, args: google_email_get(
        ctx,
        bot_id=int(args.get("bot_id", 0)),
        discord_user_id=int(args.get("discord_user_id", 0)),
        message_id=str(args.get("message_id", "")),
    ),
    "google_email_send": lambda ctx, args: google_email_send(
        ctx,
        bot_id=int(args.get("bot_id", 0)),
        discord_user_id=int(args.get("discord_user_id", 0)),
        to=str(args.get("to", "")),
        subject=str(args.get("subject", "")),
        body_text=str(args.get("body_text", "")),
        cc=_as_opt_str(args, "cc"),
        bcc=_as_opt_str(args, "bcc"),
    ),
    "google_calendar_list": lambda ctx, args: google_calendar_list(
        ctx,
        bot_id=int(args.get("bot_id", 0)),
        discord_user_id=int(args.get("discord_user_id", 0)),
        calendar_id=str(args.get("calendar_id", "primary")),
        time_min=_as_opt_str(args, "time_min"),
        time_max=_as_opt_str(args, "time_max"),
        max_results=int(args.get("max_results", 10)),
    ),
    "google_calendar_create": lambda ctx, args: google_calendar_create(
        ctx,
        bot_id=int(args.get("bot_id", 0)),
        discord_user_id=int(args.get("discord_user_id", 0)),
        summary=str(args.get("summary", "")),
        start_iso=str(args.get("start_iso", "")),
        end_iso=str(args.get("end_iso", "")),
        calendar_id=str(args.get("calendar_id", "primary")),
        description=_as_opt_str(args, "description"),
        location=_as_opt_str(args, "location"),
    ),
    "memory_add": lambda ctx, args: memory_add(
        ctx,
        content=str(args.get("content", "")),
        key=_as_opt_str(args, "key"),
        tags=(None if args.get("tags") is None else list(args.get("tags"))),
        meta=(None if args.get("meta") is None else dict(args.get("meta"))),
        pinned=bool(args.get("pinned", False)),
        ttl_seconds=_as_opt_int(args, "ttl_seconds"),
    ),
    "memory_get": lambda ctx, args: memory_get(
        ctx,
        id=_as_opt_int(args, "id"),
        key=_as_opt_str(args, "key"),
        include_expired=bool(args.get("include_expired", False)),
    ),
    "memory_search": lambda ctx, args: memory_search(
        ctx,
        query=str(args.get("query", "")),
        limit=int(args.get("limit", 5)),
        include_expired=bool(args.get("include_expired", False)),
    ),
    "memory_delete": lambda ctx, args: memory_delete(
        ctx,
        id=_as_opt_int(args, "id"),
        key=_as_opt_str(args, "key"),
    ),
    "task_list": lambda ctx, args: task_list(
        ctx,
        include_disabled=bool(args.get("include_disabled", True)),
    ),
    "task_add": lambda ctx, args: task_add(
        ctx,
        minutes=int(args.get("minutes", 0)),
        task=str(args.get("task", "")),
        enabled=bool(args.get("enabled", True)),
        allow_duplicate=bool(args.get("allow_duplicate", False)),
    ),
    "task_update": lambda ctx, args: task_update(
        ctx,
        task_id=_as_opt_int(args, "task_id"),
        task=_as_opt_str(args, "task"),
        new_minutes=_as_opt_int(args, "new_minutes"),
        new_task=_as_opt_str(args, "new_task"),
    ),
    "task_disable": lambda ctx, args: task_disable(
        ctx,
        task_id=_as_opt_int(args, "task_id"),
        task=_as_opt_str(args, "task"),
    ),
    "task_enable": lambda ctx, args: task_enable(
        ctx,
        task_id=_as_opt_int(args, "task_id"),
        task=_as_opt_str(args, "task"),
    ),
    "task_run_now": lambda ctx, args: task_run_now(
        ctx,
        task_id=_as_opt_int(args, "task_id"),
        task=_as_opt_str(args, "task"),
    ),
    "doc_ingest": lambda ctx, args: doc_ingest(
        ctx,
        path=str(args.get("path", "")),
        key=_as_opt_str(args, "key"),
        title=_as_opt_str(args, "title"),
        replace=bool(args.get("replace", True)),
        max_chars=int(args.get("max_chars", 200000)),
    ),
    "doc_inject": lambda ctx, args: doc_ingest(
        ctx,
        path=str(args.get("path", "")),
        key=_as_opt_str(args, "key"),
        title=_as_opt_str(args, "title"),
        replace=bool(args.get("replace", True)),
        max_chars=int(args.get("max_chars", 200000)),
    ),
    "doc_search": lambda ctx, args: doc_search(
        ctx,
        query=str(args.get("query", "")),
        limit=int(args.get("limit", 5)),
    ),
    "doc_get": lambda ctx, args: doc_get(
        ctx,
        id=_as_opt_int(args, "id"),
        key=_as_opt_str(args, "key"),
        include_content=bool(args.get("include_content", True)),
        max_chars=int(args.get("max_chars", 8000)),
    ),
    "doc_list": lambda ctx, args: doc_list(
        ctx,
        limit=int(args.get("limit", 20)),
        offset=int(args.get("offset", 0)),
        query=_as_opt_str(args, "query"),
    ),
    "doc_delete": lambda ctx, args: doc_delete(
        ctx,
        id=_as_opt_int(args, "id"),
        key=_as_opt_str(args, "key"),
    ),
    "sh_exec": _dispatch_sh_exec,
}


def dispatch_tool_call(ctx: ToolContext, name: str, arguments_json: str) -> dict[str, Any]:
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON arguments for {name}: {e}") from None
    if not isinstance(args, dict):
        raise ValueError(f"Tool arguments must be a JSON object for {name}")

    fn = _TOOL_DISPATCHERS.get(name)
    if fn is not None:
        return fn(ctx, args)

    # Custom tools (loaded from disk under tool_root).
    try:
        return dispatch_custom_tool_call(ctx, name, args)
    except KeyError:
        raise ValueError(f"Unknown tool: {name}") from None
