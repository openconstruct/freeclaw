import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    json: Any


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> HttpResponse:
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
            return HttpResponse(
                status=int(resp.status),
                headers={k: v for k, v in resp.headers.items()},
                json=data,
            )
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {msg}") from None


def get_json(url: str, headers: dict[str, str], timeout_s: float) -> HttpResponse:
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
            return HttpResponse(
                status=int(resp.status),
                headers={k: v for k, v in resp.headers.items()},
                json=data,
            )
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {msg}") from None

