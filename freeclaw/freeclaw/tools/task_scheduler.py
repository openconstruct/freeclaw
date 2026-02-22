from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fs import ToolContext

_TASK_RE = re.compile(r"^(\s*)(#\s*)?(\d{1,6})\s*-\s*(.+?)\s*$")


@dataclass(frozen=True)
class TaskEntry:
    id: int
    line_idx: int
    minutes: int
    task: str
    enabled: bool
    raw_line: str


def _tasks_path(ctx: ToolContext) -> Path:
    return ctx.workspace / "tasks.md"


def _state_path(ctx: ToolContext) -> Path:
    # Keep timer state with other runtime data; avoid creating workspace/.freeclaw
    # just for last-run bookkeeping.
    return ctx.workspace / "mem" / "task_timer_state.json"


def _legacy_state_path(ctx: ToolContext) -> Path:
    return ctx.workspace / ".freeclaw" / "task_timer_state.json"


def _task_key(minutes: int, task: str) -> str:
    return f"dotime:{int(minutes)}:{str(task).strip().lower()}"


def _ensure_tasks_md(ctx: ToolContext) -> None:
    p = _tasks_path(ctx)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# tasks\n\n## Tasks\n<!-- Add tasks below -->\n", encoding="utf-8")


def _read_lines(ctx: ToolContext) -> list[str]:
    _ensure_tasks_md(ctx)
    p = _tasks_path(ctx)
    data = p.read_text(encoding="utf-8", errors="replace")
    return data.splitlines()


def _write_lines(ctx: ToolContext, lines: list[str]) -> None:
    p = _tasks_path(ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(lines).rstrip("\n") + "\n"
    p.write_text(txt, encoding="utf-8")


def _parse_entries(lines: list[str]) -> list[TaskEntry]:
    out: list[TaskEntry] = []
    n = 0
    for i, ln in enumerate(lines):
        m = _TASK_RE.match(ln)
        if not m:
            continue
        minutes = int(m.group(3))
        task = (m.group(4) or "").strip()
        if minutes <= 0 or not task:
            continue
        n += 1
        out.append(
            TaskEntry(
                id=n,
                line_idx=i,
                minutes=minutes,
                task=task,
                enabled=(m.group(2) is None),
                raw_line=ln,
            )
        )
    return out


def _entry_to_obj(e: TaskEntry) -> dict[str, Any]:
    return {
        "id": int(e.id),
        "minutes": int(e.minutes),
        "task": e.task,
        "enabled": bool(e.enabled),
        "line": int(e.line_idx + 1),
        "key": _task_key(e.minutes, e.task),
        "raw": e.raw_line,
    }


def _find_entry(entries: list[TaskEntry], *, task_id: int | None, task: str | None) -> TaskEntry:
    if task_id is not None:
        for e in entries:
            if int(e.id) == int(task_id):
                return e
        raise ValueError(f"task_id not found: {task_id}")

    q = (task or "").strip().lower()
    if not q:
        raise ValueError("Provide task_id or task")
    matches = [e for e in entries if e.task.strip().lower() == q]
    if not matches:
        raise ValueError(f"task not found: {task}")
    if len(matches) > 1:
        raise ValueError("task selector is ambiguous; use task_id")
    return matches[0]


def _read_last_run(path: Path) -> dict[str, int]:
    try:
        if not path.exists() or not path.is_file():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(obj, dict):
            return {}
        lr = obj.get("last_run")
        if not isinstance(lr, dict):
            return {}
        out: dict[str, int] = {}
        for k, v in lr.items():
            if not isinstance(k, str):
                continue
            try:
                out[k] = int(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _load_last_run(ctx: ToolContext) -> dict[str, int]:
    p = _state_path(ctx)
    out = _read_last_run(p)
    if out or (p.exists() and p.is_file()):
        return out
    return _read_last_run(_legacy_state_path(ctx))


def _save_last_run(ctx: ToolContext, last_run: dict[str, int]) -> None:
    p = _state_path(ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {"version": 1, "last_run": {str(k): int(v) for k, v in (last_run or {}).items()}}
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lp = _legacy_state_path(ctx)
    try:
        if lp.exists() and lp.is_file() and lp.resolve() != p.resolve():
            lp.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        if lp.parent.name == ".freeclaw":
            lp.parent.rmdir()
    except Exception:
        pass


def task_list(
    ctx: ToolContext,
    *,
    include_disabled: bool = True,
) -> dict[str, Any]:
    lines = _read_lines(ctx)
    entries = _parse_entries(lines)
    out = [_entry_to_obj(e) for e in entries if include_disabled or e.enabled]
    return {
        "ok": True,
        "tool": "task_list",
        "tasks_path": str(_tasks_path(ctx)),
        "count": len(out),
        "tasks": out,
    }


def task_add(
    ctx: ToolContext,
    *,
    minutes: int,
    task: str,
    enabled: bool = True,
    allow_duplicate: bool = False,
) -> dict[str, Any]:
    mins = int(minutes)
    if mins < 1:
        raise ValueError("minutes must be >= 1")
    t = (task or "").strip()
    if not t:
        raise ValueError("task is required")

    lines = _read_lines(ctx)
    entries = _parse_entries(lines)
    if not allow_duplicate:
        k = _task_key(mins, t)
        for e in entries:
            if _task_key(e.minutes, e.task) == k:
                return {
                    "ok": True,
                    "tool": "task_add",
                    "added": False,
                    "reason": "duplicate",
                    "existing": _entry_to_obj(e),
                    "tasks_path": str(_tasks_path(ctx)),
                }

    new_line = f"{mins}-{t}"
    if not enabled:
        new_line = "# " + new_line

    lines.append(new_line)
    _write_lines(ctx, lines)
    entries2 = _parse_entries(lines)
    added = entries2[-1] if entries2 else None
    return {
        "ok": True,
        "tool": "task_add",
        "added": True,
        "task": (_entry_to_obj(added) if added else None),
        "tasks_path": str(_tasks_path(ctx)),
    }


def task_update(
    ctx: ToolContext,
    *,
    task_id: int | None = None,
    task: str | None = None,
    new_minutes: int | None = None,
    new_task: str | None = None,
) -> dict[str, Any]:
    lines = _read_lines(ctx)
    entries = _parse_entries(lines)
    e = _find_entry(entries, task_id=(None if task_id is None else int(task_id)), task=task)

    mins = int(new_minutes) if new_minutes is not None else int(e.minutes)
    if mins < 1:
        raise ValueError("new_minutes must be >= 1")
    txt = (new_task if new_task is not None else e.task).strip()
    if not txt:
        raise ValueError("new_task must not be empty")

    prefix = "# " if not e.enabled else ""
    lines[e.line_idx] = f"{prefix}{mins}-{txt}"
    _write_lines(ctx, lines)
    updated = _find_entry(_parse_entries(lines), task_id=e.id, task=None)
    return {
        "ok": True,
        "tool": "task_update",
        "task": _entry_to_obj(updated),
        "tasks_path": str(_tasks_path(ctx)),
    }


def task_disable(
    ctx: ToolContext,
    *,
    task_id: int | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    lines = _read_lines(ctx)
    entries = _parse_entries(lines)
    e = _find_entry(entries, task_id=(None if task_id is None else int(task_id)), task=task)
    if not e.enabled:
        return {"ok": True, "tool": "task_disable", "changed": False, "task": _entry_to_obj(e)}
    lines[e.line_idx] = "# " + f"{e.minutes}-{e.task}"
    _write_lines(ctx, lines)
    e2 = _find_entry(_parse_entries(lines), task_id=e.id, task=None)
    return {"ok": True, "tool": "task_disable", "changed": True, "task": _entry_to_obj(e2)}


def task_enable(
    ctx: ToolContext,
    *,
    task_id: int | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    lines = _read_lines(ctx)
    entries = _parse_entries(lines)
    e = _find_entry(entries, task_id=(None if task_id is None else int(task_id)), task=task)
    if e.enabled:
        return {"ok": True, "tool": "task_enable", "changed": False, "task": _entry_to_obj(e)}
    lines[e.line_idx] = f"{e.minutes}-{e.task}"
    _write_lines(ctx, lines)
    e2 = _find_entry(_parse_entries(lines), task_id=e.id, task=None)
    return {"ok": True, "tool": "task_enable", "changed": True, "task": _entry_to_obj(e2)}


def task_run_now(
    ctx: ToolContext,
    *,
    task_id: int | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    lines = _read_lines(ctx)
    entries = _parse_entries(lines)
    e = _find_entry(entries, task_id=(None if task_id is None else int(task_id)), task=task)

    lr = _load_last_run(ctx)
    k = _task_key(e.minutes, e.task)
    # Removing last_run forces this task to be due on the next timer tick.
    lr.pop(k, None)
    _save_last_run(ctx, lr)
    return {
        "ok": True,
        "tool": "task_run_now",
        "armed": True,
        "task": _entry_to_obj(e),
        "state_path": str(_state_path(ctx)),
        "armed_at": int(time.time()),
    }
