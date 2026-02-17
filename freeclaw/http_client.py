import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit


log = logging.getLogger(__name__)


def _safe_url(url: str) -> str:
    try:
        u = urlsplit(url)
        # Avoid logging query strings that might include secrets.
        return urlunsplit((u.scheme, u.netloc, u.path, "", ""))
    except Exception:
        return url


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    json: Any


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> HttpResponse:
    t0 = time.time()
    safe_url = _safe_url(url)
    log.debug(
        "http request method=POST url=%s timeout_s=%.2f payload_bytes=%d",
        safe_url,
        float(timeout_s),
        len(json.dumps(payload)),
    )
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, method="POST", data=body)
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise RuntimeError(f"Expected JSON response, got Content-Type={content_type!r}")
            data = json.loads(raw.decode("utf-8"))
            log.debug(
                "http response method=POST url=%s status=%d elapsed_ms=%.1f bytes=%d",
                safe_url,
                int(resp.status),
                (time.time() - t0) * 1000.0,
                len(raw),
            )
            return HttpResponse(
                status=int(resp.status),
                headers={k: v for k, v in resp.headers.items()},
                json=data,
            )
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace")
        log.warning(
            "http error method=POST url=%s status=%d elapsed_ms=%.1f body_preview=%r",
            safe_url,
            int(e.code),
            (time.time() - t0) * 1000.0,
            msg[:300],
        )
        raise RuntimeError(f"HTTP {e.code} from {url}: {msg}") from None


def get_json(url: str, headers: dict[str, str], timeout_s: float) -> HttpResponse:
    t0 = time.time()
    safe_url = _safe_url(url)
    log.debug("http request method=GET url=%s timeout_s=%.2f", safe_url, float(timeout_s))
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise RuntimeError(f"Expected JSON response, got Content-Type={content_type!r}")
            data = json.loads(raw.decode("utf-8"))
            log.debug(
                "http response method=GET url=%s status=%d elapsed_ms=%.1f bytes=%d",
                safe_url,
                int(resp.status),
                (time.time() - t0) * 1000.0,
                len(raw),
            )
            return HttpResponse(
                status=int(resp.status),
                headers={k: v for k, v in resp.headers.items()},
                json=data,
            )
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace")
        log.warning(
            "http error method=GET url=%s status=%d elapsed_ms=%.1f body_preview=%r",
            safe_url,
            int(e.code),
            (time.time() - t0) * 1000.0,
            msg[:300],
        )
        raise RuntimeError(f"HTTP {e.code} from {url}: {msg}") from None
