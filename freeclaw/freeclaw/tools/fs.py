import os
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

from ..paths import memory_db_path as _default_memory_db_path

def _is_fs_root(p: Path) -> bool:
    try:
        return p == Path(p.anchor)
    except Exception:
        return False


@dataclass(frozen=True)
class ToolContext:
    root: Path
    workspace: Path
    max_read_bytes: int
    max_write_bytes: int
    max_list_entries: int
    max_web_bytes: int
    web_user_agent: str
    memory_db_path: Path
    shell_enabled: bool
    shell_timeout_s: float
    shell_max_output_bytes: int
    shell_block_network: bool
    custom_tools_enabled: bool
    custom_tools_dir: Path
    custom_tools_timeout_s: float
    custom_tools_max_output_bytes: int
    custom_tools_block_network: bool

    @staticmethod
    def from_config_values(
        *,
        tool_root: str,
        workspace_dir: str | None = None,
        max_read_bytes: int,
        max_write_bytes: int,
        max_list_entries: int,
        max_web_bytes: int | None = None,
        web_user_agent: str | None = None,
        memory_db_path: str | None = None,
        enable_shell: bool | None = None,
        shell_timeout_s: float | None = None,
        shell_max_output_bytes: int | None = None,
        shell_block_network: bool | None = None,
        enable_custom_tools: bool | None = None,
        custom_tools_dir: str | None = None,
        custom_tools_timeout_s: float | None = None,
        custom_tools_max_output_bytes: int | None = None,
        custom_tools_block_network: bool | None = None,
    ) -> "ToolContext":
        env_enable_shell = os.getenv("FREECLAW_ENABLE_SHELL")
        if enable_shell is not None:
            shell_enabled = bool(enable_shell)
        elif env_enable_shell is not None and str(env_enable_shell).strip() != "":
            shell_enabled = str(env_enable_shell).strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            # Default: enabled (disable via FREECLAW_ENABLE_SHELL=false or --no-shell).
            shell_enabled = True

        env_enable_custom = os.getenv("FREECLAW_ENABLE_CUSTOM_TOOLS")
        if enable_custom_tools is not None:
            custom_enabled = bool(enable_custom_tools)
        elif env_enable_custom is not None and str(env_enable_custom).strip() != "":
            custom_enabled = str(env_enable_custom).strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            # Default: enabled (disable via FREECLAW_ENABLE_CUSTOM_TOOLS=false or --no-custom-tools).
            custom_enabled = True

        root = Path(tool_root).expanduser()
        # Resolve relative roots against cwd, so default "." means "where you ran freeclaw".
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        else:
            root = root.resolve()

        ws = Path(workspace_dir or os.getenv("FREECLAW_WORKSPACE_DIR") or "workspace").expanduser()
        if not ws.is_absolute():
            ws = (Path.cwd() / ws).resolve()
        else:
            ws = ws.resolve()
        # Safety: never allow workspace to become filesystem root.
        # This prevents state dirs like /.freeclaw from being created.
        if _is_fs_root(ws):
            ws = (Path.cwd() / "workspace").resolve()

        # Default custom tools dir lives under workspace/skills/tools.
        # This keeps custom tool specs in the bot workspace without relying on .freeclaw.
        env_custom_tools_dir = os.getenv("FREECLAW_CUSTOM_TOOLS_DIR")
        if custom_tools_dir and str(custom_tools_dir).strip():
            ctd = Path(custom_tools_dir).expanduser()
        elif env_custom_tools_dir and str(env_custom_tools_dir).strip():
            ctd = Path(env_custom_tools_dir).expanduser()
        else:
            ctd = ws / "skills" / "tools"
        if not ctd.is_absolute():
            ctd = (ws / ctd).resolve()
        else:
            ctd = ctd.resolve()
        try:
            ctd.relative_to(ws)
        except Exception:
            raise ValueError("custom_tools_dir must be within workspace_dir") from None

        return ToolContext(
            root=root,
            workspace=ws,
            max_read_bytes=int(max_read_bytes),
            max_write_bytes=int(max_write_bytes),
            max_list_entries=int(max_list_entries),
            max_web_bytes=int(max_web_bytes if max_web_bytes is not None else int(os.getenv("FREECLAW_WEB_MAX_BYTES") or 500_000)),
            web_user_agent=str(
                web_user_agent
                or os.getenv("FREECLAW_WEB_USER_AGENT")
                or "freeclaw/0.1.0 (+https://github.com/freeclaw/freeclaw)"
            ),
            memory_db_path=Path(
                memory_db_path
                or os.getenv("FREECLAW_MEMORY_DB")
                or str(_default_memory_db_path())
            )
            .expanduser()
            .resolve(),
            shell_enabled=shell_enabled,
            shell_timeout_s=float(
                shell_timeout_s
                if shell_timeout_s is not None
                else float(os.getenv("FREECLAW_SHELL_TIMEOUT_S") or 20.0)
            ),
            shell_max_output_bytes=int(
                shell_max_output_bytes
                if shell_max_output_bytes is not None
                else int(os.getenv("FREECLAW_SHELL_MAX_OUTPUT_BYTES") or 200_000)
            ),
            shell_block_network=(
                bool(shell_block_network)
                if shell_block_network is not None
                else (str(os.getenv("FREECLAW_SHELL_BLOCK_NETWORK", "false")).strip().lower() not in {"0", "false", "no", "n", "off"})
            ),
            custom_tools_enabled=custom_enabled,
            custom_tools_dir=ctd,
            custom_tools_timeout_s=float(
                custom_tools_timeout_s
                if custom_tools_timeout_s is not None
                else float(os.getenv("FREECLAW_CUSTOM_TOOLS_TIMEOUT_S") or 20.0)
            ),
            custom_tools_max_output_bytes=int(
                custom_tools_max_output_bytes
                if custom_tools_max_output_bytes is not None
                else int(os.getenv("FREECLAW_CUSTOM_TOOLS_MAX_OUTPUT_BYTES") or 200_000)
            ),
            custom_tools_block_network=(
                bool(custom_tools_block_network)
                if custom_tools_block_network is not None
                else (
                    str(os.getenv("FREECLAW_CUSTOM_TOOLS_BLOCK_NETWORK", "false")).strip().lower()
                    not in {"0", "false", "no", "n", "off"}
                )
            ),
        )


def _resolve_in_root(ctx: ToolContext, user_path: str) -> Path:
    if user_path is None:
        raise ValueError("path is required")
    p = Path(str(user_path)).expanduser()
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (ctx.root / p).resolve()

    # Prevent escaping the tool root.
    try:
        resolved.relative_to(ctx.root)
    except Exception:
        raise ValueError(f"path escapes tool_root: {user_path!r}")
    return resolved


def fs_read(ctx: ToolContext, *, path: str, start_line: int = 1, end_line: int | None = None) -> dict[str, Any]:
    p = _resolve_in_root(ctx, path)
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line is not None and end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise ValueError("path is not a file")

    data = p.read_bytes()
    if len(data) > ctx.max_read_bytes:
        raise ValueError(f"file too large to read (bytes={len(data)} > max_read_bytes={ctx.max_read_bytes})")
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines(True)  # keepends

    s = start_line - 1
    e = len(lines) if end_line is None else end_line
    sliced = "".join(lines[s:e])
    return {
        "ok": True,
        "tool": "fs_read",
        "path": str(p.relative_to(ctx.root)),
        "abs_path": str(p),
        "start_line": start_line,
        "end_line": end_line,
        "content": sliced,
        "total_lines": len(lines),
        "bytes": len(data),
    }


def fs_write(
    ctx: ToolContext,
    *,
    path: str,
    content: str,
    mode: str = "overwrite",
    make_parents: bool = True,
) -> dict[str, Any]:
    p = _resolve_in_root(ctx, path)
    b = (content or "").encode("utf-8")
    if len(b) > ctx.max_write_bytes:
        raise ValueError(
            f"write too large (bytes={len(b)} > max_write_bytes={ctx.max_write_bytes})"
        )
    if make_parents:
        p.parent.mkdir(parents=True, exist_ok=True)

    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be 'overwrite' or 'append'")
    if mode == "append":
        with p.open("ab") as f:
            f.write(b)
    else:
        with p.open("wb") as f:
            f.write(b)

    return {
        "ok": True,
        "tool": "fs_write",
        "path": str(p.relative_to(ctx.root)),
        "abs_path": str(p),
        "mode": mode,
        "bytes_written": len(b),
    }


def fs_mkdir(
    ctx: ToolContext, *, path: str, parents: bool = True, exist_ok: bool = True
) -> dict[str, Any]:
    p = _resolve_in_root(ctx, path)
    p.mkdir(parents=parents, exist_ok=exist_ok)
    return {"ok": True, "tool": "fs_mkdir", "path": str(p.relative_to(ctx.root)), "abs_path": str(p)}


def fs_list(
    ctx: ToolContext,
    *,
    path: str = ".",
    recursive: bool = False,
    max_depth: int = 2,
) -> dict[str, Any]:
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    p = _resolve_in_root(ctx, path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_dir():
        raise ValueError("path is not a directory")

    entries: list[dict[str, Any]] = []

    def add_entry(child: Path) -> None:
        rel = child.relative_to(ctx.root)
        try:
            st = child.stat()
        except OSError:
            # Best-effort; still include path.
            entries.append({"path": str(rel), "type": "unknown"})
            return
        typ = "dir" if child.is_dir() else "file" if child.is_file() else "other"
        entries.append(
            {
                "path": str(rel),
                "type": typ,
                "size": int(getattr(st, "st_size", 0)),
                "mtime": int(getattr(st, "st_mtime", 0)),
            }
        )

    if not recursive:
        for child in sorted(p.iterdir(), key=lambda x: x.name)[: ctx.max_list_entries]:
            add_entry(child)
    else:
        root_depth = len(p.relative_to(ctx.root).parts)
        for dirpath, dirnames, filenames in os.walk(p):
            d = Path(dirpath)
            cur_depth = len(d.relative_to(ctx.root).parts) - root_depth
            if cur_depth > max_depth:
                dirnames[:] = []
                continue

            for name in sorted(dirnames):
                if len(entries) >= ctx.max_list_entries:
                    break
                add_entry(d / name)
            for name in sorted(filenames):
                if len(entries) >= ctx.max_list_entries:
                    break
                add_entry(d / name)
            if len(entries) >= ctx.max_list_entries:
                break

    return {
        "ok": True,
        "tool": "fs_list",
        "path": str(p.relative_to(ctx.root)),
        "abs_path": str(p),
        "recursive": recursive,
        "max_depth": max_depth,
        "entries": entries,
        "truncated": len(entries) >= ctx.max_list_entries,
    }


def fs_stat(ctx: ToolContext, *, path: str) -> dict[str, Any]:
    p = _resolve_in_root(ctx, path)
    exists = p.exists()
    typ = "missing"
    size = None
    mtime = None
    if exists:
        if p.is_dir():
            typ = "dir"
        elif p.is_file():
            typ = "file"
        else:
            typ = "other"
        try:
            st = p.stat()
            size = int(getattr(st, "st_size", 0))
            mtime = int(getattr(st, "st_mtime", 0))
        except OSError:
            size = None
            mtime = None
    return {
        "ok": True,
        "tool": "fs_stat",
        "path": str(p.relative_to(ctx.root)),
        "abs_path": str(p),
        "exists": bool(exists),
        "type": typ,
        "size": size,
        "mtime": mtime,
    }


def fs_glob(
    ctx: ToolContext,
    *,
    pattern: str,
    max_results: int | None = None,
) -> dict[str, Any]:
    pat = (pattern or "").strip()
    if not pat:
        raise ValueError("pattern is required")
    # Glob patterns must be relative to tool_root.
    if Path(pat).is_absolute() or pat.startswith("~"):
        raise ValueError("pattern must be relative to tool_root")

    lim = int(max_results) if max_results is not None else int(ctx.max_list_entries)
    if lim < 1:
        raise ValueError("max_results must be >= 1")
    lim = min(lim, int(ctx.max_list_entries))

    out: list[dict[str, Any]] = []
    for m in ctx.root.glob(pat):
        try:
            resolved = m.resolve()
            resolved.relative_to(ctx.root)
        except Exception:
            continue

        typ = "dir" if resolved.is_dir() else "file" if resolved.is_file() else "other"
        out.append({"path": str(resolved.relative_to(ctx.root)), "type": typ})
        if len(out) >= lim:
            break

    return {
        "ok": True,
        "tool": "fs_glob",
        "pattern": pat,
        "results": out,
        "truncated": len(out) >= lim,
        "max_results": lim,
    }


def fs_diff(
    ctx: ToolContext,
    *,
    path: str,
    content: str,
    context_lines: int = 3,
) -> dict[str, Any]:
    p = _resolve_in_root(ctx, path)
    new_text = (content or "")
    if context_lines < 0:
        raise ValueError("context_lines must be >= 0")

    old_text = ""
    existed = p.exists()
    if existed:
        if not p.is_file():
            raise ValueError("path is not a file")
        data = p.read_bytes()
        if len(data) > ctx.max_read_bytes:
            raise ValueError(
                f"file too large to diff (bytes={len(data)} > max_read_bytes={ctx.max_read_bytes})"
            )
        old_text = data.decode("utf-8", errors="replace")

    old_lines = old_text.splitlines(True)
    new_lines = new_text.splitlines(True)
    rel = str(p.relative_to(ctx.root))
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=int(context_lines),
        )
    )
    diff_text = "".join(diff_lines)
    changed = old_text != new_text
    return {
        "ok": True,
        "tool": "fs_diff",
        "path": rel,
        "abs_path": str(p),
        "existed": bool(existed),
        "changed": bool(changed),
        "diff": diff_text,
        "context_lines": int(context_lines),
    }


def fs_rm(
    ctx: ToolContext,
    *,
    path: str,
    recursive: bool = False,
    missing_ok: bool = True,
) -> dict[str, Any]:
    p = _resolve_in_root(ctx, path)
    if not p.exists():
        if missing_ok:
            return {"ok": True, "tool": "fs_rm", "path": str(p.relative_to(ctx.root)), "deleted": False}
        raise FileNotFoundError(str(p))

    deleted = False
    if p.is_dir():
        if recursive:
            shutil.rmtree(p)
            deleted = True
        else:
            p.rmdir()
            deleted = True
    else:
        p.unlink()
        deleted = True

    return {"ok": True, "tool": "fs_rm", "path": str(p.relative_to(ctx.root)), "deleted": bool(deleted)}


def fs_mv(
    ctx: ToolContext,
    *,
    src: str,
    dst: str,
    overwrite: bool = False,
    make_parents: bool = True,
) -> dict[str, Any]:
    s = _resolve_in_root(ctx, src)
    d = _resolve_in_root(ctx, dst)
    if not s.exists():
        raise FileNotFoundError(str(s))
    if d.exists() and not overwrite:
        raise ValueError("dst exists (set overwrite=true to replace)")
    if make_parents:
        d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists() and overwrite:
        if d.is_dir():
            shutil.rmtree(d)
        else:
            d.unlink()
    shutil.move(str(s), str(d))
    return {
        "ok": True,
        "tool": "fs_mv",
        "src": str(s.relative_to(ctx.root)),
        "dst": str(d.relative_to(ctx.root)),
    }


def fs_cp(
    ctx: ToolContext,
    *,
    src: str,
    dst: str,
    recursive: bool = False,
    overwrite: bool = False,
    make_parents: bool = True,
) -> dict[str, Any]:
    s = _resolve_in_root(ctx, src)
    d = _resolve_in_root(ctx, dst)
    if not s.exists():
        raise FileNotFoundError(str(s))
    if d.exists() and not overwrite:
        raise ValueError("dst exists (set overwrite=true to replace)")
    if make_parents:
        d.parent.mkdir(parents=True, exist_ok=True)

    if s.is_dir():
        if not recursive:
            raise ValueError("src is a directory (set recursive=true to copy directories)")
        if d.exists() and overwrite:
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        if d.exists() and overwrite and d.is_dir():
            shutil.rmtree(d)
        shutil.copy2(s, d)

    return {
        "ok": True,
        "tool": "fs_cp",
        "src": str(s.relative_to(ctx.root)),
        "dst": str(d.relative_to(ctx.root)),
        "recursive": bool(recursive),
    }
