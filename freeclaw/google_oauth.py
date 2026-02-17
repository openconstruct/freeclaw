from __future__ import annotations

import html
import json
import os
import secrets
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/calendar.readonly "
    "https://www.googleapis.com/auth/gmail.readonly "
    "openid email"
)


def _now_s() -> int:
    return int(time.time())


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    out: set[str] = set()
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    for r in rows:
        try:
            out.add(str(r[1]))
        except Exception:
            continue
    return out


def _ensure_oauth_flows_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS oauth_flows (
          connect_id TEXT PRIMARY KEY,
          state TEXT NOT NULL UNIQUE,
          bot_id TEXT NOT NULL,
          discord_user_id TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          expires_at INTEGER NOT NULL,
          scope TEXT,
          error TEXT,
          account_email TEXT,
          access_token TEXT,
          refresh_token TEXT,
          token_expires_at INTEGER,
          claimed_at INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_oauth_flows_lookup
          ON oauth_flows(bot_id, discord_user_id, status);
        """
    )
    cols = _table_columns(con, "oauth_flows")
    if cols:
        alter: list[str] = []
        if "error" not in cols:
            alter.append("ALTER TABLE oauth_flows ADD COLUMN error TEXT")
        if "account_email" not in cols:
            alter.append("ALTER TABLE oauth_flows ADD COLUMN account_email TEXT")
        if "access_token" not in cols:
            alter.append("ALTER TABLE oauth_flows ADD COLUMN access_token TEXT")
        if "refresh_token" not in cols:
            alter.append("ALTER TABLE oauth_flows ADD COLUMN refresh_token TEXT")
        if "token_expires_at" not in cols:
            alter.append("ALTER TABLE oauth_flows ADD COLUMN token_expires_at INTEGER")
        if "claimed_at" not in cols:
            alter.append("ALTER TABLE oauth_flows ADD COLUMN claimed_at INTEGER")
        for sql in alter:
            con.execute(sql)
    con.commit()


def _connect(db_path: str) -> sqlite3.Connection:
    p = Path(str(db_path)).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    _ensure_oauth_flows_schema(con)
    return con


def _google_client_id() -> str:
    client_id = str(
        os.getenv("FREECLAW_GOOGLE_CLIENT_ID")
        or os.getenv("GAUTH_GOOGLE_CLIENT_ID")
        or ""
    ).strip()
    if not client_id:
        raise ValueError("Missing FREECLAW_GOOGLE_CLIENT_ID.")
    if ".apps.googleusercontent.com" not in client_id:
        raise ValueError(
            "FREECLAW_GOOGLE_CLIENT_ID looks invalid. Use OAuth Client ID from Google Cloud Credentials."
        )
    return client_id


def _google_client_secret() -> str | None:
    raw = str(
        os.getenv("FREECLAW_GOOGLE_CLIENT_SECRET")
        or os.getenv("GAUTH_GOOGLE_CLIENT_SECRET")
        or ""
    ).strip()
    return raw or None


def google_redirect_uri_from_env() -> str | None:
    direct = str(
        os.getenv("FREECLAW_GOOGLE_REDIRECT_URI")
        or os.getenv("GAUTH_GOOGLE_REDIRECT_URI")
        or ""
    ).strip()
    if direct:
        return direct
    public_base = str(
        os.getenv("FREECLAW_PUBLIC_BASE_URL")
        or os.getenv("GAUTH_PUBLIC_BASE_URL")
        or ""
    ).strip().rstrip("/")
    if public_base:
        return f"{public_base}/v1/oauth/callback"
    return None


def _google_redirect_uri_required() -> str:
    redir = google_redirect_uri_from_env()
    if not redir:
        raise ValueError(
            "Missing FREECLAW_GOOGLE_REDIRECT_URI. "
            "Set it to your public callback URL, e.g. http://<PUBLIC_IP>:3000/v1/oauth/callback."
        )
    return redir


def _google_default_scopes() -> str:
    scopes = str(os.getenv("FREECLAW_GOOGLE_DEFAULT_SCOPES") or "").strip()
    return scopes or DEFAULT_SCOPES


def _google_connect_expires_s() -> int:
    raw = str(os.getenv("FREECLAW_GOOGLE_CONNECT_EXPIRES_S") or "900").strip() or "900"
    try:
        v = int(raw)
    except Exception:
        v = 900
    if v < 120:
        v = 120
    if v > 86400:
        v = 86400
    return int(v)


def _google_oauth_timeout_s() -> float:
    raw = str(os.getenv("FREECLAW_GOOGLE_OAUTH_TIMEOUT_S") or "20.0").strip() or "20.0"
    try:
        v = float(raw)
    except Exception:
        v = 20.0
    if v <= 0:
        v = 20.0
    return float(v)


def _as_str_id(v: Any, *, field: str) -> str:
    s = str(v or "").strip()
    if not s:
        raise ValueError(f"{field} is required")
    if len(s) > 200:
        raise ValueError(f"{field} is too long")
    return s


def _truncate(s: str, n: int) -> str:
    if len(s) <= int(n):
        return s
    return s[: int(n)] + "..."


def _html_page(*, title: str, body_lines: list[str]) -> str:
    t = html.escape(str(title or "Google OAuth"))
    body = "".join([f"<p>{html.escape(str(x))}</p>" for x in body_lines if str(x or "").strip()])
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>"
        + t
        + "</title><style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#111;line-height:1.5}"
        "h1{margin:0 0 10px;font-size:30px}p{margin:10px 0}code{background:#f5f5f5;padding:1px 6px;border-radius:4px}"
        "</style></head><body><h1>"
        + t
        + "</h1>"
        + body
        + "</body></html>"
    )


def oauth_state_exists(*, db_path: str, state: str) -> bool:
    st = str(state or "").strip()
    if not st:
        return False
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT 1 FROM oauth_flows WHERE state=? LIMIT 1",
            (st,),
        ).fetchone()
    return row is not None


def start_google_oauth_flow(
    *,
    db_path: str,
    bot_id: Any,
    discord_user_id: Any,
    scopes: str | None = None,
) -> dict[str, Any]:
    client_id = _google_client_id()
    redirect_uri = _google_redirect_uri_required()

    sid_bot = _as_str_id(bot_id, field="bot_id")
    sid_user = _as_str_id(discord_user_id, field="discord_user_id")
    scope = str(scopes or _google_default_scopes()).strip()
    if not scope:
        raise ValueError("Google OAuth scopes are empty.")

    now = _now_s()
    expires_at = now + int(_google_connect_expires_s())
    connect_id = secrets.token_urlsafe(18)
    state = secrets.token_urlsafe(28)

    with _connect(db_path) as con:
        con.execute(
            """
            INSERT INTO oauth_flows (
              connect_id, state, bot_id, discord_user_id, status,
              created_at, expires_at, scope
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (connect_id, state, sid_bot, sid_user, int(now), int(expires_at), scope),
        )
        con.commit()

    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
        }
    )
    auth_url = f"{GOOGLE_AUTH_URL}?{query}"
    return {
        "ok": True,
        "connect_id": connect_id,
        "authorization_url": auth_url,
        "expires_at": int(expires_at),
        "scope": scope,
        "redirect_uri": redirect_uri,
    }


def get_google_oauth_status(*, db_path: str, connect_id: str) -> dict[str, Any]:
    cid = str(connect_id or "").strip()
    if not cid:
        raise ValueError("connect_id is required")

    with _connect(db_path) as con:
        row = con.execute("SELECT * FROM oauth_flows WHERE connect_id=?", (cid,)).fetchone()
        if row is None:
            raise ValueError("connect_id not found")

        now = _now_s()
        status = str(row["status"] or "")
        if status == "pending" and now >= int(row["expires_at"] or 0):
            con.execute(
                "UPDATE oauth_flows SET status='expired', error=? WHERE connect_id=?",
                ("expired", cid),
            )
            con.commit()
            row = con.execute("SELECT * FROM oauth_flows WHERE connect_id=?", (cid,)).fetchone()
            assert row is not None

        return {
            "ok": True,
            "connect_id": str(row["connect_id"] or ""),
            "status": str(row["status"] or "unknown"),
            "bot_id": str(row["bot_id"] or ""),
            "discord_user_id": str(row["discord_user_id"] or ""),
            "expires_at": int(row["expires_at"] or 0),
            "scope": (None if row["scope"] is None else str(row["scope"])),
            "account_email": (None if row["account_email"] is None else str(row["account_email"])),
            "error": (None if row["error"] is None else str(row["error"])),
            "claimed": bool(row["claimed_at"]),
        }


def claim_google_oauth_tokens(
    *,
    db_path: str,
    connect_id: str,
    bot_id: Any,
    discord_user_id: Any,
) -> dict[str, Any]:
    cid = str(connect_id or "").strip()
    if not cid:
        raise ValueError("connect_id is required")
    sid_bot = _as_str_id(bot_id, field="bot_id")
    sid_user = _as_str_id(discord_user_id, field="discord_user_id")

    with _connect(db_path) as con:
        row = con.execute("SELECT * FROM oauth_flows WHERE connect_id=?", (cid,)).fetchone()
        if row is None:
            raise ValueError("connect_id not found")
        if str(row["bot_id"] or "") != sid_bot or str(row["discord_user_id"] or "") != sid_user:
            raise PermissionError("connect_id does not belong to this bot/user")

        status = str(row["status"] or "")
        if status != "authorized":
            raise RuntimeError(f"connect_id status is {status}, not authorized")
        if row["claimed_at"] is not None:
            raise RuntimeError("connect_id already claimed")

        access_token = str(row["access_token"] or "")
        if not access_token:
            raise RuntimeError("no access token available to claim")

        out = {
            "ok": True,
            "connect_id": str(row["connect_id"] or cid),
            "bot_id": sid_bot,
            "discord_user_id": sid_user,
            "account_email": (None if row["account_email"] is None else str(row["account_email"])),
            "scope": (None if row["scope"] is None else str(row["scope"])),
            "access_token": access_token,
            "refresh_token": (None if row["refresh_token"] is None else str(row["refresh_token"])),
            "token_expires_at": int(row["token_expires_at"] or 0),
        }
        now = _now_s()
        con.execute(
            """
            UPDATE oauth_flows
            SET claimed_at=?, status='claimed',
                access_token=NULL, refresh_token=NULL
            WHERE connect_id=?
            """,
            (int(now), cid),
        )
        con.commit()
    return out


def _http_post_form(*, url: str, data: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    payload = urllib.parse.urlencode([(str(k), str(v)) for k, v in data.items() if v is not None]).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        body = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {int(e.code)} from {url}: {_truncate(body, 1200)}") from None

    try:
        obj = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    except Exception:
        obj = {}
    if not isinstance(obj, dict):
        raise RuntimeError("Google returned invalid JSON object")
    return obj


def _http_get_json(*, url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(str(k), str(v))
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read()
    if not raw:
        return {}
    obj = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(obj, dict):
        raise RuntimeError("Google returned invalid JSON object")
    return obj


def _google_user_email(*, access_token: str, timeout_s: float) -> str | None:
    if not str(access_token or "").strip():
        return None
    try:
        obj = _http_get_json(
            url=GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout_s=float(timeout_s),
        )
    except Exception:
        return None
    email = str(obj.get("email") or "").strip()
    return email or None


def handle_google_oauth_callback(
    *,
    db_path: str,
    state: str,
    code: str | None,
    error: str | None,
    error_description: str | None,
) -> tuple[int, str]:
    st = str(state or "").strip()
    if not st:
        return 400, _html_page(
            title="Google connect failed",
            body_lines=["Missing state.", "Return to Discord and run /google connect again."],
        )

    with _connect(db_path) as con:
        row = con.execute("SELECT * FROM oauth_flows WHERE state=?", (st,)).fetchone()
        if row is None:
            return 400, _html_page(
                title="Google connect failed",
                body_lines=["Unknown state.", "Return to Discord and run /google connect again."],
            )

        connect_id = str(row["connect_id"] or "")
        now = _now_s()
        expires_at = int(row["expires_at"] or 0)
        if now >= expires_at:
            con.execute(
                "UPDATE oauth_flows SET status='expired', error=? WHERE connect_id=?",
                ("expired", connect_id),
            )
            con.commit()
            return 400, _html_page(
                title="Google connect expired",
                body_lines=["The connect flow expired.", "Return to Discord and run /google connect again."],
            )

        err = str(error or "").strip()
        if err:
            msg = str(error_description or err).strip() or "oauth_error"
            msg = _truncate(msg, 800)
            con.execute(
                "UPDATE oauth_flows SET status='error', error=? WHERE connect_id=?",
                (msg, connect_id),
            )
            con.commit()
            return 400, _html_page(
                title="Google connect failed",
                body_lines=[msg, "Return to Discord and run /google connect again."],
            )

        code_s = str(code or "").strip()
        if not code_s:
            con.execute(
                "UPDATE oauth_flows SET status='error', error=? WHERE connect_id=?",
                ("missing_code", connect_id),
            )
            con.commit()
            return 400, _html_page(
                title="Google connect failed",
                body_lines=["Missing code.", "Return to Discord and run /google connect again."],
            )

        try:
            client_id = _google_client_id()
            client_secret = _google_client_secret()
            redirect_uri = _google_redirect_uri_required()
            timeout_s = _google_oauth_timeout_s()

            token_payload: dict[str, Any] = {
                "client_id": client_id,
                "code": code_s,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            }
            if client_secret:
                token_payload["client_secret"] = client_secret
            tok = _http_post_form(url=GOOGLE_TOKEN_URL, data=token_payload, timeout_s=timeout_s)
            access_token = str(tok.get("access_token") or "").strip()
            refresh_token = str(tok.get("refresh_token") or "").strip()
            if not access_token:
                raise RuntimeError("token response missing access_token")
            try:
                expires_in = int(tok.get("expires_in") or 0)
            except Exception:
                expires_in = 0
            scope = str(tok.get("scope") or row["scope"] or "").strip()
            account_email = _google_user_email(access_token=access_token, timeout_s=timeout_s)
            token_expires_at = int(now + max(0, expires_in))

            con.execute(
                """
                UPDATE oauth_flows
                SET status='authorized', error=NULL,
                    account_email=?, access_token=?, refresh_token=?, token_expires_at=?, scope=?
                WHERE connect_id=?
                """,
                (
                    account_email,
                    access_token,
                    (refresh_token or None),
                    int(token_expires_at),
                    scope,
                    connect_id,
                ),
            )
            con.commit()
            return 200, _html_page(
                title="Google connected",
                body_lines=[
                    "Authorization completed.",
                    "Return to Discord and run /google poll.",
                ],
            )
        except Exception as e:
            msg = _truncate(str(e), 1200)
            con.execute(
                "UPDATE oauth_flows SET status='error', error=? WHERE connect_id=?",
                (msg, connect_id),
            )
            con.commit()
            return 500, _html_page(
                title="Google connect failed",
                body_lines=[msg, "Return to Discord and run /google connect again."],
            )

