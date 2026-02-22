from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

from .fs import ToolContext
from .web import _validate_url


def http_request_json(
    ctx: ToolContext,
    *,
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    timeout_s: float = 20.0,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    u = _validate_url(url)
    m = (method or "GET").strip().upper()
    if m not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        raise ValueError("method must be one of: GET, POST, PUT, PATCH, DELETE")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    lim = int(max_bytes) if max_bytes is not None else int(ctx.max_web_bytes)
    if lim <= 0:
        raise ValueError("max_bytes must be > 0")

    hdrs: dict[str, str] = {"Accept": "application/json", "User-Agent": ctx.web_user_agent}
    if headers:
        for k, v in headers.items():
            ks = str(k).strip()
            if not ks:
                continue
            hdrs[ks] = str(v)

    body = None
    if m in {"POST", "PUT", "PATCH"} and json_body is not None:
        body = json.dumps(json_body, ensure_ascii=True).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")

    class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            target = urllib.parse.urljoin(req.full_url, str(newurl))
            _validate_url(target)
            return super().redirect_request(req, fp, code, msg, headers, target)

    req = urllib.request.Request(u, method=m, data=body, headers=hdrs)
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(req, timeout=float(timeout_s)) as resp:
        status = int(getattr(resp, "status", 200))
        content_type = str(resp.headers.get("Content-Type", "") or "")
        raw = resp.read(lim + 1)
        if len(raw) > lim:
            raise ValueError(f"response too large (bytes>{lim})")
        if "application/json" not in content_type.lower():
            # Some APIs reply with json but missing the header; attempt parse anyway.
            pass
        # Decode using charset if present.
        m2 = re.search(r"charset=([A-Za-z0-9_\\-]+)", content_type, flags=re.IGNORECASE)
        enc = (m2.group(1) if m2 else "utf-8").strip()
        try:
            text = raw.decode(enc, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(text) if text.strip() else None
        except Exception:
            raise ValueError("response is not valid JSON")

        return {
            "ok": True,
            "tool": "http_request_json",
            "url": u,
            "final_url": str(resp.geturl() or u),
            "status": status,
            "headers": {k: v for k, v in resp.headers.items()},
            "bytes": len(raw),
            "json": data,
        }

