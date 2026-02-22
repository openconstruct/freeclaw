import html
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from .fs import ToolContext


def _is_private_host(host: str) -> bool:
    if not host:
        return True
    h = host.strip().lower()
    if h in {"localhost", "localhost.localdomain"}:
        return True
    if h.endswith(".local"):
        return True

    try:
        # Literal IP.
        ip = ipaddress.ip_address(h)
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        pass

    # Resolve DNS and reject anything that maps to a private/loopback/link-local/etc. address.
    try:
        infos = socket.getaddrinfo(h, None)
    except OSError:
        # If it can't be resolved, treat as unsafe.
        return True

    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_s = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_s)
        except ValueError:
            return True
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def _validate_url(url: str) -> str:
    if not url or not isinstance(url, str):
        raise ValueError("url is required")
    u = url.strip()
    parsed = urllib.parse.urlparse(u)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")
    if not parsed.netloc:
        raise ValueError("URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("URL must not include credentials")
    host = parsed.hostname or ""
    if _is_private_host(host):
        raise ValueError("Refusing to fetch from private/localhost host")
    return u


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._out: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if t in {"p", "div", "br", "li", "section", "header", "footer", "article"}:
            self._out.append("\n")
        if t in {"h1", "h2", "h3", "h4"}:
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if t in {"p", "div", "li"}:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if data:
            self._out.append(data)

    def text(self) -> str:
        raw = "".join(self._out)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


@dataclass(frozen=True)
class WebFetchResult:
    ok: bool
    tool: str
    url: str
    final_url: str
    status: int
    content_type: str
    bytes: int
    text: str


def web_fetch(
    ctx: ToolContext,
    *,
    url: str,
    max_bytes: int | None = None,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    u = _validate_url(url)
    lim = int(max_bytes) if max_bytes is not None else int(ctx.max_web_bytes)
    if lim <= 0:
        raise ValueError("max_bytes must be > 0")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            # urllib may supply relative redirect locations.
            target = urllib.parse.urljoin(req.full_url, str(newurl))
            _validate_url(target)
            return super().redirect_request(req, fp, code, msg, headers, target)

    req = urllib.request.Request(
        u,
        method="GET",
        headers={
            "Accept": "text/html,text/plain,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "User-Agent": ctx.web_user_agent,
        },
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(req, timeout=float(timeout_s)) as resp:
        status = int(getattr(resp, "status", 200))
        content_type = str(resp.headers.get("Content-Type", "") or "")
        raw = resp.read(lim + 1)
        if len(raw) > lim:
            raise ValueError(f"response too large (bytes>{lim})")

        # Try to decode using charset from headers, else utf-8.
        m = re.search(r"charset=([A-Za-z0-9_\\-]+)", content_type, flags=re.IGNORECASE)
        enc = (m.group(1) if m else "utf-8").strip()
        try:
            body = raw.decode(enc, errors="replace")
        except LookupError:
            body = raw.decode("utf-8", errors="replace")

        is_html = "html" in content_type.lower()
        if is_html:
            parser = _HTMLToText()
            parser.feed(body)
            text = parser.text()
        else:
            text = body.strip()

        out = WebFetchResult(
            ok=True,
            tool="web_fetch",
            url=u,
            final_url=str(resp.geturl() or u),
            status=status,
            content_type=content_type,
            bytes=len(raw),
            text=text,
        )
        return out.__dict__


def web_search(
    ctx: ToolContext,
    *,
    query: str,
    max_results: int = 5,
    safesearch: str = "moderate",
) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    mr = int(max_results)
    if mr < 1:
        raise ValueError("max_results must be >= 1")
    if mr > 10:
        mr = 10
    ss = (safesearch or "moderate").strip().lower()
    if ss not in {"off", "moderate", "strict"}:
        raise ValueError("safesearch must be one of: off, moderate, strict")

    try:
        from ddgs import DDGS  # type: ignore
    except Exception:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "DDG search backend not installed. Install with:\n"
                "  pip install -e '.[web]'\n"
                f"Import error: {e}"
            )

    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(q, safesearch=ss, max_results=mr):  # type: ignore[attr-defined]
            if not isinstance(r, dict):
                continue
            url = r.get("href") or r.get("url")
            title = r.get("title") or ""
            snippet = r.get("body") or r.get("snippet") or ""
            if isinstance(url, str) and url.strip():
                results.append(
                    {
                        "title": str(title),
                        "url": str(url),
                        "snippet": str(snippet),
                    }
                )
            if len(results) >= mr:
                break

    return {"ok": True, "tool": "web_search", "query": q, "max_results": mr, "results": results}
