from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .fs import ToolContext, _resolve_in_root


_TEXT_EXTS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".py",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".sql",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".log",
    ".env",
}


def _db_path(ctx: ToolContext) -> Path:
    return ctx.workspace / ".freeclaw" / "docs.sqlite3"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def _fts5_available(con: sqlite3.Connection) -> bool:
    try:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __docs_fts_probe USING fts5(x);")
        con.execute("DROP TABLE IF EXISTS __docs_fts_probe;")
        return True
    except sqlite3.OperationalError:
        return False


def _init_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          doc_key TEXT UNIQUE,
          source_path TEXT,
          title TEXT,
          content TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          content_chars INTEGER NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_docs_updated_at ON docs(updated_at);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_docs_source_path ON docs(source_path);")

    if _fts5_available(con):
        con.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(title, content, content='docs', content_rowid='id');
            """
        )
        con.execute(
            """
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
              INSERT INTO docs_fts(rowid, title, content) VALUES (new.id, coalesce(new.title,''), new.content);
            END;
            """
        )
        con.execute(
            """
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
              INSERT INTO docs_fts(docs_fts, rowid, title, content) VALUES('delete', old.id, coalesce(old.title,''), old.content);
            END;
            """
        )
        con.execute(
            """
            CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
              INSERT INTO docs_fts(docs_fts, rowid, title, content) VALUES('delete', old.id, coalesce(old.title,''), old.content);
              INSERT INTO docs_fts(rowid, title, content) VALUES (new.id, coalesce(new.title,''), new.content);
            END;
            """
        )
    con.commit()


def _read_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover - runtime-only dependency branch
        raise RuntimeError(f"PDF support requires pypdf (pip install -e '.[discord]'): {e}") from None

    import io

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts)


def _extract_text(path: Path, data: bytes) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_text(data), "pdf"

    # Prefer UTF-8 for known text types.
    if ext in _TEXT_EXTS:
        return data.decode("utf-8", errors="replace"), "text"

    # Best-effort text decode fallback for unknown extensions.
    txt = data.decode("utf-8", errors="replace")
    return txt, "text"


def doc_ingest(
    ctx: ToolContext,
    *,
    path: str,
    key: str | None = None,
    title: str | None = None,
    replace: bool = True,
    max_chars: int = 200_000,
) -> dict[str, Any]:
    p = _resolve_in_root(ctx, path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))

    raw = p.read_bytes()
    hard_limit = max(int(ctx.max_write_bytes), 2_000_000)
    if len(raw) > hard_limit:
        raise ValueError(f"file too large to ingest (bytes={len(raw)} > limit={hard_limit})")

    content, doc_type = _extract_text(p, raw)
    text = (content or "").strip()
    if not text:
        raise ValueError("No extractable text found in document")
    lim = max(1_000, int(max_chars))
    if len(text) > lim:
        text = text[:lim]

    now = int(time.time())
    k = (str(key).strip() if isinstance(key, str) and str(key).strip() else None)
    t = (str(title).strip() if isinstance(title, str) and str(title).strip() else p.name)
    src = str(p.relative_to(ctx.root))
    sha = hashlib.sha256(raw).hexdigest()

    with _connect(_db_path(ctx)) as con:
        _init_schema(con)
        if k:
            row = con.execute("SELECT id FROM docs WHERE doc_key=?;", (k,)).fetchone()
            if row and not replace:
                raise ValueError(f"doc key already exists: {k}")
            if row:
                con.execute(
                    """
                    UPDATE docs
                    SET source_path=?, title=?, content=?, sha256=?, content_chars=?, updated_at=?
                    WHERE doc_key=?;
                    """,
                    (src, t, text, sha, len(text), now, k),
                )
            else:
                con.execute(
                    """
                    INSERT INTO docs(doc_key, source_path, title, content, sha256, content_chars, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?);
                    """,
                    (k, src, t, text, sha, len(text), now, now),
                )
            out_row = con.execute(
                "SELECT id, doc_key, source_path, title, sha256, content_chars, created_at, updated_at FROM docs WHERE doc_key=?;",
                (k,),
            ).fetchone()
        else:
            con.execute(
                """
                INSERT INTO docs(doc_key, source_path, title, content, sha256, content_chars, created_at, updated_at)
                VALUES(NULL,?,?,?,?,?,?,?);
                """,
                (src, t, text, sha, len(text), now, now),
            )
            out_row = con.execute(
                "SELECT id, doc_key, source_path, title, sha256, content_chars, created_at, updated_at FROM docs ORDER BY id DESC LIMIT 1;"
            ).fetchone()
        con.commit()

    assert out_row is not None
    return {
        "ok": True,
        "tool": "doc_ingest",
        "doc": {
            "id": int(out_row[0]),
            "key": out_row[1],
            "source_path": str(out_row[2]),
            "title": str(out_row[3] or ""),
            "sha256": str(out_row[4]),
            "content_chars": int(out_row[5]),
            "created_at": int(out_row[6]),
            "updated_at": int(out_row[7]),
            "doc_type": doc_type,
        },
        "db_path": str(_db_path(ctx)),
    }


def doc_search(
    ctx: ToolContext,
    *,
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    lim = max(1, min(int(limit), 20))

    with _connect(_db_path(ctx)) as con:
        _init_schema(con)
        use_fts = False
        try:
            con.execute("SELECT 1 FROM docs_fts LIMIT 1;")
            use_fts = True
        except sqlite3.OperationalError:
            use_fts = False

        if use_fts:
            rows = con.execute(
                """
                SELECT d.id, d.doc_key, d.source_path, d.title, d.content_chars, d.updated_at, substr(d.content, 1, 240)
                FROM docs_fts f
                JOIN docs d ON d.id=f.rowid
                WHERE docs_fts MATCH ?
                ORDER BY d.updated_at DESC
                LIMIT ?;
                """,
                (q, lim),
            ).fetchall()
        else:
            like = f"%{q}%"
            rows = con.execute(
                """
                SELECT id, doc_key, source_path, title, content_chars, updated_at, substr(content, 1, 240)
                FROM docs
                WHERE title LIKE ? OR content LIKE ? OR source_path LIKE ? OR doc_key LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?;
                """,
                (like, like, like, like, lim),
            ).fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "id": int(r[0]),
                "key": r[1],
                "source_path": str(r[2] or ""),
                "title": str(r[3] or ""),
                "content_chars": int(r[4] or 0),
                "updated_at": int(r[5] or 0),
                "snippet": str(r[6] or "").replace("\n", " "),
            }
        )
    return {"ok": True, "tool": "doc_search", "query": q, "limit": lim, "results": results}


def doc_get(
    ctx: ToolContext,
    *,
    id: int | None = None,
    key: str | None = None,
    include_content: bool = True,
    max_chars: int = 8_000,
) -> dict[str, Any]:
    if id is None and (not key or not str(key).strip()):
        raise ValueError("Provide id or key")

    with _connect(_db_path(ctx)) as con:
        _init_schema(con)
        if id is not None:
            row = con.execute(
                "SELECT id, doc_key, source_path, title, sha256, content_chars, created_at, updated_at, content FROM docs WHERE id=?;",
                (int(id),),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT id, doc_key, source_path, title, sha256, content_chars, created_at, updated_at, content FROM docs WHERE doc_key=?;",
                (str(key).strip(),),
            ).fetchone()

    if not row:
        return {"ok": True, "tool": "doc_get", "found": False}

    content = str(row[8] or "")
    lim = max(500, int(max_chars))
    if len(content) > lim:
        content = content[:lim]

    out: dict[str, Any] = {
        "ok": True,
        "tool": "doc_get",
        "found": True,
        "doc": {
            "id": int(row[0]),
            "key": row[1],
            "source_path": str(row[2] or ""),
            "title": str(row[3] or ""),
            "sha256": str(row[4] or ""),
            "content_chars": int(row[5] or 0),
            "created_at": int(row[6] or 0),
            "updated_at": int(row[7] or 0),
        },
    }
    if include_content:
        out["doc"]["content"] = content
    return out


def doc_list(
    ctx: ToolContext,
    *,
    limit: int = 20,
    offset: int = 0,
    query: str | None = None,
) -> dict[str, Any]:
    lim = max(1, min(int(limit), 200))
    off = max(0, int(offset))
    q = (str(query).strip() if query is not None else "")

    with _connect(_db_path(ctx)) as con:
        _init_schema(con)
        if q:
            like = f"%{q}%"
            rows = con.execute(
                """
                SELECT id, doc_key, source_path, title, sha256, content_chars, created_at, updated_at
                FROM docs
                WHERE title LIKE ? OR source_path LIKE ? OR doc_key LIKE ? OR content LIKE ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?;
                """,
                (like, like, like, like, lim, off),
            ).fetchall()
            total = con.execute(
                """
                SELECT COUNT(*)
                FROM docs
                WHERE title LIKE ? OR source_path LIKE ? OR doc_key LIKE ? OR content LIKE ?;
                """,
                (like, like, like, like),
            ).fetchone()
        else:
            rows = con.execute(
                """
                SELECT id, doc_key, source_path, title, sha256, content_chars, created_at, updated_at
                FROM docs
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?;
                """,
                (lim, off),
            ).fetchall()
            total = con.execute("SELECT COUNT(*) FROM docs;").fetchone()

    items: list[dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "id": int(r[0]),
                "key": r[1],
                "source_path": str(r[2] or ""),
                "title": str(r[3] or ""),
                "sha256": str(r[4] or ""),
                "content_chars": int(r[5] or 0),
                "created_at": int(r[6] or 0),
                "updated_at": int(r[7] or 0),
            }
        )

    return {
        "ok": True,
        "tool": "doc_list",
        "limit": lim,
        "offset": off,
        "query": (q or None),
        "total": int(total[0] if total else 0),
        "results": items,
    }


def doc_delete(
    ctx: ToolContext,
    *,
    id: int | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    if id is None and (not key or not str(key).strip()):
        raise ValueError("Provide id or key")

    with _connect(_db_path(ctx)) as con:
        _init_schema(con)
        if id is not None:
            cur = con.execute("DELETE FROM docs WHERE id=?;", (int(id),))
        else:
            cur = con.execute("DELETE FROM docs WHERE doc_key=?;", (str(key).strip(),))
        con.commit()
    return {"ok": True, "tool": "doc_delete", "deleted": int(cur.rowcount)}
