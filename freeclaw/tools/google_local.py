from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage
from typing import Any

from .fs import ToolContext

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
CAL_API_BASE = "https://www.googleapis.com/calendar/v3"


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def _now_s() -> int:
    return int(time.time())


def _ensure_google_tokens_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS google_tokens_v1 (
          bot_id INTEGER NOT NULL,
          discord_user_id INTEGER NOT NULL,
          account_email TEXT,
          scope TEXT,
          access_token TEXT,
          refresh_token TEXT,
          token_expires_at INTEGER,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY(bot_id, discord_user_id)
        );
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_google_tokens_v1_updated_at ON google_tokens_v1(updated_at);"
    )
    con.commit()


def _http_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    full_url = url
    if params:
        q = urllib.parse.urlencode([(str(k), str(v)) for k, v in params.items() if v is not None], doseq=True)
        if q:
            full_url = f"{url}?{q}"

    data_b: bytes | None = None
    req = urllib.request.Request(full_url, method=method.upper())
    for k, v in headers.items():
        req.add_header(str(k), str(v))
    if json_body is not None:
        data_b = json.dumps(json_body, ensure_ascii=True).encode("utf-8")
        req.data = data_b
        req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
            if not raw:
                return {}
            obj = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(obj, dict):
                raise RuntimeError("Google API returned non-object JSON")
            return obj
    except urllib.error.HTTPError as e:
        raw = e.read()
        body = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {int(e.code)} from {full_url}: {body[:1200]}") from None


def _google_client_creds() -> tuple[str, str | None]:
    client_id = str(
        os.getenv("FREECLAW_GOOGLE_CLIENT_ID")
        or os.getenv("GAUTH_GOOGLE_CLIENT_ID")
        or ""
    ).strip()
    client_secret_raw = str(
        os.getenv("FREECLAW_GOOGLE_CLIENT_SECRET")
        or os.getenv("GAUTH_GOOGLE_CLIENT_SECRET")
        or ""
    ).strip()
    client_secret = client_secret_raw or None
    if not client_id:
        raise ValueError(
            "Missing Google client settings for token refresh. Set FREECLAW_GOOGLE_CLIENT_ID "
            "(legacy fallback: GAUTH_GOOGLE_CLIENT_ID)."
        )
    return client_id, client_secret


def _get_token_row(*, ctx: ToolContext, bot_id: int, discord_user_id: int) -> sqlite3.Row:
    with _connect(str(ctx.memory_db_path)) as con:
        _ensure_google_tokens_schema(con)
        row = con.execute(
            """
            SELECT bot_id, discord_user_id, account_email, scope,
                   access_token, refresh_token, token_expires_at, updated_at
            FROM google_tokens_v1
            WHERE bot_id=? AND discord_user_id=?
            """,
            (int(bot_id), int(discord_user_id)),
        ).fetchone()
    if row is None:
        raise ValueError("No Google tokens for bot_id/discord_user_id. Run google connect first.")
    return row


def _upsert_token_row(
    *,
    ctx: ToolContext,
    bot_id: int,
    discord_user_id: int,
    account_email: str | None,
    scope: str,
    access_token: str,
    refresh_token: str | None,
    token_expires_at: int,
) -> None:
    with _connect(str(ctx.memory_db_path)) as con:
        _ensure_google_tokens_schema(con)
        con.execute(
            """
            INSERT INTO google_tokens_v1 (
              bot_id, discord_user_id, account_email, scope,
              access_token, refresh_token, token_expires_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bot_id, discord_user_id) DO UPDATE SET
              account_email=excluded.account_email,
              scope=excluded.scope,
              access_token=excluded.access_token,
              refresh_token=excluded.refresh_token,
              token_expires_at=excluded.token_expires_at,
              updated_at=excluded.updated_at
            """,
            (
                int(bot_id),
                int(discord_user_id),
                account_email,
                str(scope or ""),
                str(access_token or ""),
                (None if refresh_token is None else str(refresh_token)),
                int(token_expires_at or 0),
                _now_s(),
            ),
        )
        con.commit()


def _refresh_access_token(*, refresh_token: str) -> dict[str, Any]:
    client_id, client_secret = _google_client_creds()
    data = {
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if client_secret:
        data["client_secret"] = client_secret
    req = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        method="POST",
        data=urllib.parse.urlencode(data).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            raw = resp.read()
            obj = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
            if not isinstance(obj, dict):
                raise RuntimeError("Invalid token refresh JSON response")
            return obj
    except urllib.error.HTTPError as e:
        raw = e.read()
        body = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"Google token refresh failed HTTP {int(e.code)}: {body[:1200]}") from None


def _ensure_access_token(*, ctx: ToolContext, bot_id: int, discord_user_id: int) -> tuple[str, sqlite3.Row]:
    row = _get_token_row(ctx=ctx, bot_id=bot_id, discord_user_id=discord_user_id)
    now = _now_s()
    access_token = str(row["access_token"] or "")
    refresh_token = str(row["refresh_token"] or "")
    expires_at = int(row["token_expires_at"] or 0)

    if access_token and expires_at > (now + 30):
        return access_token, row

    if not refresh_token:
        raise ValueError("Google access token expired and no refresh token is stored. Reconnect Google.")

    obj = _refresh_access_token(refresh_token=refresh_token)
    new_access = str(obj.get("access_token") or "")
    if not new_access:
        raise RuntimeError("Google refresh response missing access_token")
    expires_in = int(obj.get("expires_in") or 0)
    new_scope = str(obj.get("scope") or row["scope"] or "")
    new_refresh = str(obj.get("refresh_token") or "")
    new_expires = now + max(0, expires_in)

    _upsert_token_row(
        ctx=ctx,
        bot_id=bot_id,
        discord_user_id=discord_user_id,
        account_email=(None if row["account_email"] is None else str(row["account_email"])),
        scope=new_scope,
        access_token=new_access,
        refresh_token=(new_refresh or refresh_token),
        token_expires_at=new_expires,
    )

    row2 = _get_token_row(ctx=ctx, bot_id=bot_id, discord_user_id=discord_user_id)
    return new_access, row2


def _gmail_api_json(
    *,
    access_token: str,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _http_json(
        method=method,
        url=f"{GMAIL_API_BASE}{path}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        params=params,
        json_body=json_body,
        timeout_s=25.0,
    )


def _calendar_api_json(
    *,
    access_token: str,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _http_json(
        method=method,
        url=f"{CAL_API_BASE}{path}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        params=params,
        json_body=json_body,
        timeout_s=25.0,
    )


def google_email_list(
    ctx: ToolContext,
    *,
    bot_id: int,
    discord_user_id: int,
    query: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    lim = max(1, min(25, int(max_results)))
    access_token, _ = _ensure_access_token(ctx=ctx, bot_id=int(bot_id), discord_user_id=int(discord_user_id))

    params: dict[str, Any] = {"maxResults": int(lim)}
    q = str(query or "").strip()
    if q:
        params["q"] = q
    data = _gmail_api_json(access_token=access_token, method="GET", path="/users/me/messages", params=params)
    msgs = data.get("messages") if isinstance(data.get("messages"), list) else []

    out: list[dict[str, Any]] = []
    for m in msgs[:lim]:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "")
        if not mid:
            continue
        item = _gmail_api_json(
            access_token=access_token,
            method="GET",
            path=f"/users/me/messages/{urllib.parse.quote(mid, safe='')}",
            params={"format": "metadata", "metadataHeaders": ["From", "To", "Subject", "Date"]},
        )
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        headers = payload.get("headers") if isinstance(payload.get("headers"), list) else []
        hmap: dict[str, str] = {}
        for h in headers:
            if isinstance(h, dict):
                k = str(h.get("name") or "").strip().lower()
                v = str(h.get("value") or "")
                if k:
                    hmap[k] = v
        out.append(
            {
                "id": mid,
                "threadId": str(item.get("threadId") or ""),
                "snippet": str(item.get("snippet") or ""),
                "from": hmap.get("from", ""),
                "to": hmap.get("to", ""),
                "subject": hmap.get("subject", ""),
                "date": hmap.get("date", ""),
            }
        )

    return {
        "ok": True,
        "tool": "google_email_list",
        "count": len(out),
        "messages": out,
    }


def google_email_get(
    ctx: ToolContext,
    *,
    bot_id: int,
    discord_user_id: int,
    message_id: str,
) -> dict[str, Any]:
    mid = str(message_id or "").strip()
    if not mid:
        raise ValueError("message_id is required")
    access_token, _ = _ensure_access_token(ctx=ctx, bot_id=int(bot_id), discord_user_id=int(discord_user_id))
    item = _gmail_api_json(
        access_token=access_token,
        method="GET",
        path=f"/users/me/messages/{urllib.parse.quote(mid, safe='')}",
        params={"format": "full"},
    )
    return {
        "ok": True,
        "tool": "google_email_get",
        "message": item,
    }


def google_email_send(
    ctx: ToolContext,
    *,
    bot_id: int,
    discord_user_id: int,
    to: str,
    subject: str,
    body_text: str,
    cc: str | None = None,
    bcc: str | None = None,
) -> dict[str, Any]:
    to_s = str(to or "").strip()
    if not to_s:
        raise ValueError("to is required")

    access_token, _ = _ensure_access_token(ctx=ctx, bot_id=int(bot_id), discord_user_id=int(discord_user_id))

    msg = EmailMessage()
    msg["To"] = to_s
    if cc and str(cc).strip():
        msg["Cc"] = str(cc).strip()
    if bcc and str(bcc).strip():
        msg["Bcc"] = str(bcc).strip()
    msg["Subject"] = str(subject or "")
    msg.set_content(str(body_text or ""))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    sent = _gmail_api_json(
        access_token=access_token,
        method="POST",
        path="/users/me/messages/send",
        json_body={"raw": raw},
    )
    return {
        "ok": True,
        "tool": "google_email_send",
        "id": str(sent.get("id") or ""),
        "threadId": str(sent.get("threadId") or ""),
    }


def google_calendar_list(
    ctx: ToolContext,
    *,
    bot_id: int,
    discord_user_id: int,
    calendar_id: str = "primary",
    time_min: str | None = None,
    time_max: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    lim = max(1, min(25, int(max_results)))
    cid = str(calendar_id or "primary").strip() or "primary"
    access_token, _ = _ensure_access_token(ctx=ctx, bot_id=int(bot_id), discord_user_id=int(discord_user_id))
    params: dict[str, Any] = {
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": int(lim),
    }
    if time_min:
        params["timeMin"] = str(time_min)
    if time_max:
        params["timeMax"] = str(time_max)
    data = _calendar_api_json(
        access_token=access_token,
        method="GET",
        path=f"/calendars/{urllib.parse.quote(cid, safe='')}/events",
        params=params,
    )
    items = data.get("items") if isinstance(data.get("items"), list) else []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append(
            {
                "id": str(it.get("id") or ""),
                "status": str(it.get("status") or ""),
                "summary": str(it.get("summary") or ""),
                "description": str(it.get("description") or ""),
                "location": str(it.get("location") or ""),
                "start": it.get("start"),
                "end": it.get("end"),
                "htmlLink": str(it.get("htmlLink") or ""),
            }
        )
    return {
        "ok": True,
        "tool": "google_calendar_list",
        "count": len(out),
        "events": out,
    }


def google_calendar_create(
    ctx: ToolContext,
    *,
    bot_id: int,
    discord_user_id: int,
    summary: str,
    start_iso: str,
    end_iso: str,
    calendar_id: str = "primary",
    description: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    s = str(summary or "").strip()
    if not s:
        raise ValueError("summary is required")
    start_s = str(start_iso or "").strip()
    end_s = str(end_iso or "").strip()
    if not start_s or not end_s:
        raise ValueError("start_iso and end_iso are required")
    cid = str(calendar_id or "primary").strip() or "primary"

    access_token, _ = _ensure_access_token(ctx=ctx, bot_id=int(bot_id), discord_user_id=int(discord_user_id))

    body: dict[str, Any] = {
        "summary": s,
        "start": {"dateTime": start_s},
        "end": {"dateTime": end_s},
    }
    if description:
        body["description"] = str(description)
    if location:
        body["location"] = str(location)

    it = _calendar_api_json(
        access_token=access_token,
        method="POST",
        path=f"/calendars/{urllib.parse.quote(cid, safe='')}/events",
        json_body=body,
    )
    return {
        "ok": True,
        "tool": "google_calendar_create",
        "id": str(it.get("id") or ""),
        "status": str(it.get("status") or ""),
        "htmlLink": str(it.get("htmlLink") or ""),
        "summary": str(it.get("summary") or ""),
    }
