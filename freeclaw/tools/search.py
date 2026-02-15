from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fs import ToolContext, _resolve_in_root


@dataclass(frozen=True)
class _Match:
    path: str
    line: int
    col: int
    text: str


def _iter_files(root: Path, *, include_glob: str | None, exclude_glob: str | None, max_files: int) -> list[Path]:
    inc = (include_glob or "**/*").strip() or "**/*"
    out: list[Path] = []
    for p in root.glob(inc):
        if len(out) >= max_files:
            break
        try:
            rp = p.resolve()
            rp.relative_to(root)
        except Exception:
            continue
        if not rp.exists() or not rp.is_file():
            continue
        if exclude_glob:
            # Exclude is matched against the path relative to root, using Path.match semantics.
            rel = rp.relative_to(root)
            if rel.match(exclude_glob):
                continue
        out.append(rp)
    return out


def text_search(
    ctx: ToolContext,
    *,
    query: str,
    path: str = ".",
    regex: bool = False,
    case_sensitive: bool = False,
    include_glob: str | None = None,
    exclude_glob: str | None = None,
    max_results: int = 20,
    max_files: int = 200,
    context_lines: int = 0,
) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    base = _resolve_in_root(ctx, path)
    if not base.exists() or not base.is_dir():
        raise ValueError("path must be an existing directory within tool_root")

    mr = int(max_results)
    if mr < 1:
        raise ValueError("max_results must be >= 1")
    mr = min(mr, 200)
    mf = int(max_files)
    if mf < 1:
        raise ValueError("max_files must be >= 1")
    mf = min(mf, int(ctx.max_list_entries))
    cl = int(context_lines)
    if cl < 0:
        raise ValueError("context_lines must be >= 0")
    cl = min(cl, 10)

    flags = 0 if case_sensitive else re.IGNORECASE
    pat = re.compile(q, flags=flags) if regex else None
    needle = q if case_sensitive else q.lower()

    files = _iter_files(base, include_glob=include_glob, exclude_glob=exclude_glob, max_files=mf)
    results: list[dict[str, Any]] = []
    scanned_files = 0

    for f in files:
        scanned_files += 1
        data = f.read_bytes()
        if len(data) > ctx.max_read_bytes:
            continue
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            if pat is not None:
                m = pat.search(line)
                if not m:
                    continue
                col = int(m.start()) + 1
            else:
                hay = line if case_sensitive else line.lower()
                idx = hay.find(needle)
                if idx < 0:
                    continue
                col = idx + 1

            before = lines[max(0, i - 1 - cl) : i - 1] if cl else []
            after = lines[i : i + cl] if cl else []
            results.append(
                {
                    "path": str(f.relative_to(ctx.root)),
                    "line": i,
                    "col": col,
                    "text": line,
                    "before": before,
                    "after": after,
                }
            )
            if len(results) >= mr:
                break
        if len(results) >= mr:
            break

    return {
        "ok": True,
        "tool": "text_search",
        "query": q,
        "path": str(base.relative_to(ctx.root)),
        "regex": bool(regex),
        "case_sensitive": bool(case_sensitive),
        "include_glob": include_glob,
        "exclude_glob": exclude_glob,
        "max_results": mr,
        "max_files": mf,
        "context_lines": cl,
        "scanned_files": scanned_files,
        "results": results,
        "truncated": len(results) >= mr,
    }

