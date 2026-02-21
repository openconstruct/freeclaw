from __future__ import annotations

import os
from typing import Any


def first_env(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


def extract_finish_reason(resp: dict[str, Any]) -> str | None:
    try:
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        c0 = choices[0] if isinstance(choices[0], dict) else None
        if not isinstance(c0, dict):
            return None
        fr = c0.get("finish_reason") or c0.get("stop_reason") or c0.get("finishReason")
        return fr.strip() if isinstance(fr, str) and fr.strip() else None
    except Exception:
        return None


def safe_label(s: str, *, fallback: str = "base", max_len: int = 80) -> str:
    out: list[str] = []
    for ch in (s or "").strip():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch.isspace():
            out.append("-")
    v = "".join(out).strip("-_")
    if not v:
        return fallback
    return v[: int(max_len)]
