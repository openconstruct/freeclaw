from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .paths import config_dir


class _StderrToolFilter(logging.Filter):
    """Hide verbose tool lines from stderr while keeping them in file logs."""

    def __init__(self, *, show_tool_logs: bool) -> None:
        super().__init__()
        self._show_tool_logs = bool(show_tool_logs)

    def filter(self, record: logging.LogRecord) -> bool:
        if self._show_tool_logs:
            return True
        msg = record.getMessage()
        return not (msg.startswith("[tool] ") or msg.startswith("[tool-result] "))


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "pid": int(record.process),
            "thread": record.threadName,
            "module": record.module,
            "func": record.funcName,
            "line": int(record.lineno),
        }
        if record.exc_info:
            obj["exc"] = "".join(traceback.format_exception(*record.exc_info))
        if record.stack_info:
            obj["stack"] = str(record.stack_info)
        return json.dumps(obj, ensure_ascii=True)


def setup_logging(*, level: str | None = None, log_file: str | None = None, log_format: str | None = None) -> None:
    """
    Configure stdlib logging for both CLI and Discord.

    Defaults:
    - level: FREECLAW_LOG_LEVEL or "info"
    - log_file: FREECLAW_LOG_FILE or <config>/freeclaw.log
    - log_format: FREECLAW_LOG_FORMAT or "text" ("text" | "jsonl")
    """
    lvl = (level or os.getenv("FREECLAW_LOG_LEVEL") or "info").strip().upper()
    numeric = getattr(logging, lvl, None)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    if log_file is not None:
        lf = log_file.strip()
    else:
        env_log_file = os.getenv("FREECLAW_LOG_FILE")
        if env_log_file is not None:
            lf = env_log_file.strip()
        else:
            lf = str((config_dir() / "freeclaw.log").resolve())
    fmt = (log_format or os.getenv("FREECLAW_LOG_FORMAT") or "text").strip().lower()
    if fmt not in {"text", "jsonl", "json"}:
        fmt = "text"
    show_tool_stderr = (
        str(os.getenv("FREECLAW_LOG_TOOL_STDERR", "false")).strip().lower()
        in {"1", "true", "yes", "y", "on"}
    )
    handlers: list[logging.Handler] = []
    if lf:
        p = Path(lf).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                str(p),
                encoding="utf-8",
                maxBytes=100_000,
                backupCount=3,
            )
        )
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.addFilter(_StderrToolFilter(show_tool_logs=show_tool_stderr))
    handlers.append(stderr_handler)

    if fmt in {"jsonl", "json"}:
        formatter: logging.Formatter = _JsonLineFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s pid=%(process)d tid=%(threadName)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    for h in handlers:
        h.setFormatter(formatter)

    logging.basicConfig(level=numeric, handlers=handlers, force=True)
    logging.getLogger(__name__).info(
        "logging configured level=%s format=%s log_file=%s",
        logging.getLevelName(numeric),
        ("jsonl" if fmt in {"jsonl", "json"} else "text"),
        (str(Path(lf).expanduser().resolve()) if lf else "(stderr)"),
    )
