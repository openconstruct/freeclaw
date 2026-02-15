from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fs import ToolContext
from .shell import exec_argv


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_TPL_RE = re.compile(r"{{\s*([A-Za-z0-9_]+)\s*}}")


@dataclass(frozen=True)
class CustomToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    typ: str
    argv: list[str]
    workdir: str  # "tool_root" | "tool_dir"
    timeout_s: float | None
    max_output_bytes: int | None
    block_network: bool | None
    parse_json: bool
    env: dict[str, str] | None
    stdin: str | None
    path: Path


def _read_json(path: Path, *, max_bytes: int = 200_000) -> Any:
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"tool spec too large: {path}")
    return json.loads(data.decode("utf-8", errors="replace"))


def _discover_spec_paths(base: Path) -> list[Path]:
    out: list[Path] = []
    if not base.exists() or not base.is_dir():
        return out
    # Two supported layouts:
    # - <dir>/<name>.json
    # - <dir>/<name>/tool.json
    for child in sorted(base.iterdir(), key=lambda p: p.name):
        if child.is_file() and child.suffix.lower() == ".json":
            out.append(child)
            continue
        if child.is_dir():
            tj = child / "tool.json"
            if tj.exists() and tj.is_file():
                out.append(tj)
    return out


def _validate_spec(obj: Any, *, path: Path) -> CustomToolSpec:
    if not isinstance(obj, dict):
        raise ValueError("tool spec must be a JSON object")
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool spec missing 'name'")
    name = name.strip()
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid tool name {name!r} (use [A-Za-z_][A-Za-z0-9_]{0,63})")

    desc = obj.get("description") or ""
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError("tool spec missing 'description'")
    desc = desc.strip()

    params = obj.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("tool spec missing 'parameters' (JSON schema object)")

    typ = obj.get("type") or "command"
    if not isinstance(typ, str):
        raise ValueError("tool spec 'type' must be a string")
    typ = typ.strip().lower()
    if typ != "command":
        raise ValueError("only type='command' is supported currently")

    argv = obj.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError("tool spec missing 'argv' (non-empty array)")
    argv_s: list[str] = []
    for a in argv:
        if not isinstance(a, str) or not a.strip():
            raise ValueError("tool spec 'argv' entries must be non-empty strings")
        argv_s.append(a)

    workdir = obj.get("workdir") or "tool_dir"
    if not isinstance(workdir, str):
        raise ValueError("tool spec 'workdir' must be a string")
    workdir = workdir.strip().lower()
    if workdir not in {"tool_root", "tool_dir"}:
        raise ValueError("tool spec 'workdir' must be 'tool_root' or 'tool_dir'")

    timeout_s = obj.get("timeout_s")
    if timeout_s is not None and not isinstance(timeout_s, (int, float)):
        raise ValueError("tool spec 'timeout_s' must be a number")
    max_output_bytes = obj.get("max_output_bytes")
    if max_output_bytes is not None and not isinstance(max_output_bytes, int):
        raise ValueError("tool spec 'max_output_bytes' must be an integer")
    block_network = obj.get("block_network")
    if block_network is not None and not isinstance(block_network, bool):
        raise ValueError("tool spec 'block_network' must be a boolean")
    parse_json = bool(obj.get("parse_json", False))

    env_obj = obj.get("env")
    env_out: dict[str, str] | None = None
    if env_obj is not None:
        if not isinstance(env_obj, dict):
            raise ValueError("tool spec 'env' must be an object")
        env_out = {}
        for k, v in env_obj.items():
            ks = str(k).strip()
            if not ks:
                continue
            if not isinstance(v, str):
                raise ValueError("tool spec 'env' values must be strings")
            env_out[ks] = v

    stdin_t = obj.get("stdin")
    if stdin_t is not None and not isinstance(stdin_t, str):
        raise ValueError("tool spec 'stdin' must be a string")

    return CustomToolSpec(
        name=name,
        description=desc,
        parameters=params,
        typ=typ,
        argv=argv_s,
        workdir=workdir,
        timeout_s=(float(timeout_s) if timeout_s is not None else None),
        max_output_bytes=(int(max_output_bytes) if max_output_bytes is not None else None),
        block_network=(bool(block_network) if block_network is not None else None),
        parse_json=parse_json,
        env=env_out,
        stdin=(stdin_t if isinstance(stdin_t, str) else None),
        path=path,
    )


def iter_custom_tools(ctx: ToolContext, *, reserved_names: set[str] | None = None) -> list[CustomToolSpec]:
    if not ctx.custom_tools_enabled:
        return []
    base = ctx.custom_tools_dir
    specs: dict[str, CustomToolSpec] = {}
    for p in _discover_spec_paths(base):
        try:
            obj = _read_json(p)
            spec = _validate_spec(obj, path=p)
        except Exception:
            # Best-effort: skip invalid specs.
            continue
        if reserved_names and spec.name in reserved_names:
            continue
        # First one wins (stable order).
        if spec.name not in specs:
            specs[spec.name] = spec
    return list(specs.values())


def custom_tool_schemas(ctx: ToolContext, *, reserved_names: set[str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in iter_custom_tools(ctx, reserved_names=reserved_names):
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
        )
    return out


def _render_template(s: str, args: dict[str, Any]) -> str:
    def repl(m: re.Match[str]) -> str:
        k = m.group(1)
        if k == "args_json":
            return json.dumps(args, ensure_ascii=True)
        if k not in args:
            raise ValueError(f"missing argument for template: {k}")
        v = args.get(k)
        if v is None:
            return ""
        if isinstance(v, (str, int, float, bool)):
            return str(v)
        return json.dumps(v, ensure_ascii=True)

    return _TPL_RE.sub(repl, s)


def _substitute_argv(argv: list[str], args: dict[str, Any]) -> list[str]:
    return [_render_template(a, args) for a in argv]


def dispatch_custom_tool_call(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if not ctx.custom_tools_enabled:
        raise KeyError(name)

    spec = None
    for s in iter_custom_tools(ctx):
        if s.name == name:
            spec = s
            break
    if spec is None:
        raise KeyError(name)

    # Resolve workdir.
    tool_dir = spec.path.parent
    if spec.path.name == "tool.json":
        tool_dir = spec.path.parent
    else:
        tool_dir = spec.path.parent

    cwd = ctx.root if spec.workdir == "tool_root" else tool_dir
    resolved_cwd = cwd.resolve()
    ok = False
    for base in (ctx.root, ctx.workspace):
        try:
            resolved_cwd.relative_to(base)
            ok = True
            break
        except Exception:
            pass
    if not ok:
        raise ValueError("custom tool workdir escapes allowed roots (tool_root/workspace)")

    argv = _substitute_argv(spec.argv, args)
    env = None
    if spec.env:
        env = {k: _render_template(v, args) for k, v in spec.env.items()}
    stdin = _render_template(spec.stdin, args) if spec.stdin else None
    res = exec_argv(
        ctx,
        argv=argv,
        stdin=stdin,
        env=env,
        timeout_s=(spec.timeout_s if spec.timeout_s is not None else ctx.custom_tools_timeout_s),
        max_output_bytes=(
            spec.max_output_bytes if spec.max_output_bytes is not None else ctx.custom_tools_max_output_bytes
        ),
        block_network=(
            spec.block_network if spec.block_network is not None else ctx.custom_tools_block_network
        ),
        cwd=resolved_cwd,
        tool_name=spec.name,
    )

    if spec.parse_json and isinstance(res, dict):
        out = res.get("stdout") if isinstance(res.get("stdout"), str) else ""
        try:
            parsed = json.loads(out) if out.strip() else None
            if isinstance(parsed, dict):
                parsed.setdefault("ok", True)
                parsed.setdefault("tool", spec.name)
                return parsed
        except Exception:
            pass
    return res
