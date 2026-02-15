import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fs import ToolContext


def _now() -> int:
    return int(time.time())


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def _fts5_available(con: sqlite3.Connection) -> bool:
    try:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x);")
        con.execute("DROP TABLE IF EXISTS __fts5_probe;")
        return True
    except sqlite3.OperationalError:
        return False


def _init_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mem_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          key TEXT UNIQUE,
          content TEXT NOT NULL,
          tags TEXT,
          meta_json TEXT,
          pinned INTEGER NOT NULL DEFAULT 0,
          expires_at INTEGER,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        """
    )
    # Migrate older DBs in-place.
    cols = {r[1] for r in con.execute("PRAGMA table_info(mem_items);").fetchall()}
    if "pinned" not in cols:
        con.execute("ALTER TABLE mem_items ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;")
    if "expires_at" not in cols:
        con.execute("ALTER TABLE mem_items ADD COLUMN expires_at INTEGER;")
    con.execute("CREATE INDEX IF NOT EXISTS idx_mem_items_updated_at ON mem_items(updated_at);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_mem_items_expires_at ON mem_items(expires_at);")

    if _fts5_available(con):
        con.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts
            USING fts5(content, tags, content='mem_items', content_rowid='id');
            """
        )
        con.execute(
            """
            CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON mem_items BEGIN
              INSERT INTO mem_fts(rowid, content, tags) VALUES (new.id, new.content, coalesce(new.tags,''));
            END;
            """
        )
        con.execute(
            """
            CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON mem_items BEGIN
              INSERT INTO mem_fts(mem_fts, rowid, content, tags) VALUES('delete', old.id, old.content, coalesce(old.tags,''));
            END;
            """
        )
        con.execute(
            """
            CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON mem_items BEGIN
              INSERT INTO mem_fts(mem_fts, rowid, content, tags) VALUES('delete', old.id, old.content, coalesce(old.tags,''));
              INSERT INTO mem_fts(rowid, content, tags) VALUES (new.id, new.content, coalesce(new.tags,''));
            END;
            """
        )

    con.commit()


@dataclass(frozen=True)
class MemoryItem:
    id: int
    key: str | None
    content: str
    tags: list[str]
    meta: dict[str, Any]
    pinned: bool
    expires_at: int | None
    created_at: int
    updated_at: int


def _row_to_item(row: tuple[Any, ...]) -> MemoryItem:
    (mid, key, content, tags_s, meta_json, pinned, expires_at, created_at, updated_at) = row
    tags: list[str] = []
    if isinstance(tags_s, str) and tags_s.strip():
        tags = [t.strip() for t in tags_s.split(",") if t.strip()]
    meta: dict[str, Any] = {}
    if isinstance(meta_json, str) and meta_json.strip():
        try:
            mj = json.loads(meta_json)
            if isinstance(mj, dict):
                meta = mj
        except Exception:
            meta = {}
    return MemoryItem(
        id=int(mid),
        key=(str(key) if key is not None else None),
        content=str(content),
        tags=tags,
        meta=meta,
        pinned=bool(int(pinned) if pinned is not None else 0),
        expires_at=(int(expires_at) if expires_at is not None else None),
        created_at=int(created_at),
        updated_at=int(updated_at),
    )


def memory_add(
    ctx: ToolContext,
    *,
    content: str,
    key: str | None = None,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    pinned: bool = False,
    ttl_seconds: int | None = None,
) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        raise ValueError("content is required")
    if len(text.encode("utf-8")) > 50_000:
        raise ValueError("content too large (max ~50KB)")

    k = (key.strip() if isinstance(key, str) and key.strip() else None)
    tlist = [str(x).strip() for x in (tags or []) if str(x).strip()]
    tags_s = ",".join(sorted(set(tlist), key=str.lower)) if tlist else ""
    meta_json = json.dumps(meta or {}, ensure_ascii=True) if meta else "{}"

    now = _now()
    pin = 1 if bool(pinned) else 0
    exp: int | None = None
    if not pin and ttl_seconds is not None:
        ttl = int(ttl_seconds)
        if ttl > 0:
            exp = now + ttl
    with _connect(ctx.memory_db_path) as con:
        _init_schema(con)
        try:
            con.execute(
                """
                INSERT INTO mem_items(key, content, tags, meta_json, pinned, expires_at, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?);
                """,
                (k, text, tags_s, meta_json, pin, exp, now, now),
            )
        except sqlite3.IntegrityError:
            # Key collision: treat as upsert.
            if not k:
                raise
            con.execute(
                """
                UPDATE mem_items
                SET content=?, tags=?, meta_json=?, pinned=?, expires_at=?, updated_at=?
                WHERE key=?;
                """,
                (text, tags_s, meta_json, pin, exp, now, k),
            )
        con.commit()

        if k:
            row = con.execute(
                "SELECT id,key,content,tags,meta_json,pinned,expires_at,created_at,updated_at FROM mem_items WHERE key=?;",
                (k,),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT id,key,content,tags,meta_json,pinned,expires_at,created_at,updated_at FROM mem_items ORDER BY id DESC LIMIT 1;"
            ).fetchone()
        assert row is not None
        item = _row_to_item(row)

    return {"ok": True, "tool": "memory_add", "item": item.__dict__}


def memory_get(
    ctx: ToolContext,
    *,
    id: int | None = None,
    key: str | None = None,
    include_expired: bool = False,
) -> dict[str, Any]:
    if id is None and (not key or not str(key).strip()):
        raise ValueError("Provide id or key")

    with _connect(ctx.memory_db_path) as con:
        _init_schema(con)
        now = _now()
        exp_clause = "" if include_expired else "AND (expires_at IS NULL OR expires_at > ?)"
        if id is not None:
            row = con.execute(
                "SELECT id,key,content,tags,meta_json,pinned,expires_at,created_at,updated_at FROM mem_items WHERE id=? "
                + exp_clause
                + ";",
                ((int(id),) if include_expired else (int(id), now)),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT id,key,content,tags,meta_json,pinned,expires_at,created_at,updated_at FROM mem_items WHERE key=? "
                + exp_clause
                + ";",
                ((str(key).strip(),) if include_expired else (str(key).strip(), now)),
            ).fetchone()
        if not row:
            return {"ok": True, "tool": "memory_get", "found": False}
        item = _row_to_item(row)
        return {"ok": True, "tool": "memory_get", "found": True, "item": item.__dict__}


def memory_search(
    ctx: ToolContext,
    *,
    query: str,
    limit: int = 5,
    include_expired: bool = False,
) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    lim = int(limit)
    if lim < 1:
        raise ValueError("limit must be >= 1")
    if lim > 20:
        lim = 20

    with _connect(ctx.memory_db_path) as con:
        _init_schema(con)
        now = _now()
        exp_sql = "" if include_expired else "AND (mi.expires_at IS NULL OR mi.expires_at > ?)"

        use_fts = False
        try:
            con.execute("SELECT 1 FROM mem_fts LIMIT 1;")
            use_fts = True
        except sqlite3.OperationalError:
            use_fts = False

        rows: list[tuple[Any, ...]] = []
        if use_fts:
            rows = con.execute(
                """
                SELECT mi.id,mi.key,mi.content,mi.tags,mi.meta_json,mi.pinned,mi.expires_at,mi.created_at,mi.updated_at
                FROM mem_fts
                JOIN mem_items mi ON mi.id = mem_fts.rowid
                WHERE mem_fts MATCH ?
                """
                + exp_sql
                + """
                ORDER BY mi.updated_at DESC
                LIMIT ?;
                """,
                ((q, lim) if include_expired else (q, now, lim)),
            ).fetchall()
        else:
            like = f"%{q}%"
            rows = con.execute(
                """
                SELECT id,key,content,tags,meta_json,pinned,expires_at,created_at,updated_at
                FROM mem_items
                WHERE (content LIKE ? OR tags LIKE ? OR key LIKE ?)
                """
                + ("" if include_expired else "AND (expires_at IS NULL OR expires_at > ?)")
                + """
                ORDER BY updated_at DESC
                LIMIT ?;
                """,
                ((like, like, like, lim) if include_expired else (like, like, like, now, lim)),
            ).fetchall()

        items = [_row_to_item(r).__dict__ for r in rows]
        return {"ok": True, "tool": "memory_search", "query": q, "limit": lim, "results": items}


def memory_delete(ctx: ToolContext, *, id: int | None = None, key: str | None = None) -> dict[str, Any]:
    if id is None and (not key or not str(key).strip()):
        raise ValueError("Provide id or key")
    with _connect(ctx.memory_db_path) as con:
        _init_schema(con)
        if id is not None:
            cur = con.execute("DELETE FROM mem_items WHERE id=?;", (int(id),))
        else:
            cur = con.execute("DELETE FROM mem_items WHERE key=?;", (str(key).strip(),))
        con.commit()
        return {"ok": True, "tool": "memory_delete", "deleted": int(cur.rowcount)}
