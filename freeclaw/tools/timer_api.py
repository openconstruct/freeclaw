from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from .fs import ToolContext


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
_ENDPOINTS = {
    "system_metrics": "/api/system/metrics",
    "status": "/timer/status",
    "health": "/health",
}


def _normalize_local_host(host: str) -> str:
    h = (host or "").strip()
    if not h:
        return "127.0.0.1"
    if h.lower() not in _LOCAL_HOSTS:
        raise ValueError("timer_api_get host must be localhost-only (127.0.0.1, localhost, ::1)")
    return h


def timer_api_get(
    ctx: ToolContext,
    *,
    endpoint: str = "system_metrics",
    host: str = "127.0.0.1",
    port: int = 3000,
    timeout_s: float = 5.0,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    ep = (endpoint or "system_metrics").strip().lower()
    if ep not in _ENDPOINTS:
        raise ValueError("endpoint must be one of: system_metrics, status, health")

    h = _normalize_local_host(host)
    p = int(port)
    if p < 1 or p > 65535:
        raise ValueError("port must be between 1 and 65535")

    to_s = float(timeout_s)
    if to_s <= 0:
        raise ValueError("timeout_s must be > 0")

    lim = int(max_bytes) if max_bytes is not None else int(ctx.max_web_bytes)
    if lim <= 0:
        raise ValueError("max_bytes must be > 0")

    hostport = h
    if ":" in h and not h.startswith("["):
        hostport = f"[{h}]"
    path = _ENDPOINTS[ep]
    url = urllib.parse.urlunsplit(("http", f"{hostport}:{p}", path, "", ""))

    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": ctx.web_user_agent,
        },
    )
    with urllib.request.urlopen(req, timeout=to_s) as resp:
        status = int(getattr(resp, "status", 200))
        raw = resp.read(lim + 1)
        if len(raw) > lim:
            raise ValueError(f"response too large (bytes>{lim})")
        text = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(text) if text.strip() else None
        except Exception:
            raise ValueError("response is not valid JSON")

        return {
            "ok": True,
            "tool": "timer_api_get",
            "endpoint": ep,
            "url": url,
            "status": status,
            "bytes": len(raw),
            "json": data,
        }
