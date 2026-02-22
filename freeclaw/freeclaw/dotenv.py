import os
from pathlib import Path

from .paths import env_path


def _default_config_env_path() -> Path:
    return env_path()


def default_config_env_path() -> Path:
    return _default_config_env_path()


def _parse_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[len("export ") :].lstrip()
    if "=" not in s:
        return None
    k, v = s.split("=", 1)
    key = k.strip()
    if not key:
        return None
    val = v.strip()
    if val and val[0] in {"'", '"'} and val[-1:] == val[:1]:
        val = val[1:-1]
    return key, val


def load_dotenv(path: Path, *, override: bool = False) -> int:
    """
    Loads KEY=VALUE lines into os.environ.
    Returns count of keys set.
    """
    if not path.exists() or not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    n = 0
    for raw in text.splitlines():
        parsed = _parse_line(raw)
        if not parsed:
            continue
        k, v = parsed
        if not override and k in os.environ:
            continue
        os.environ[k] = v
        n += 1
    return n


def autoload_dotenv(*, override: bool = False) -> list[Path]:
    """
    Load env vars from:
    - FREECLAW_ENV_FILE (if set), else
    - ./config/.env, then
    - ./.env
    Returns list of paths that were loaded (existed and were read).
    """
    loaded: list[Path] = []
    env_file = os.getenv("FREECLAW_ENV_FILE")
    if env_file and env_file.strip():
        p = Path(env_file).expanduser()
        if p.exists() and p.is_file():
            load_dotenv(p, override=override)
            loaded.append(p)
        return loaded

    candidates = [_default_config_env_path(), Path.cwd() / ".env"]
    for p in candidates:
        if p.exists() and p.is_file():
            load_dotenv(p, override=override)
            loaded.append(p)
    return loaded


def _quote_env_value(v: str) -> str:
    if v == "":
        return ""
    needs_quote = any(c in v for c in [" ", "\t", "#"]) or v.startswith('"') or v.endswith('"')
    if not needs_quote:
        return v
    esc = v.replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{esc}\""


def set_env_var(path: Path, key: str, value: str) -> None:
    """
    Upsert KEY=VALUE into an env file, preserving existing lines as much as possible.
    """
    key = key.strip()
    if not key:
        raise ValueError("key is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    out: list[str] = []
    found = False
    for ln in lines:
        parsed = _parse_line(ln)
        if parsed and parsed[0] == key and not found:
            out.append(f"{key}={_quote_env_value(value)}\n")
            found = True
        else:
            out.append(ln)
    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        if out and out[-1].strip() != "":
            out.append("\n")
        out.append(f"{key}={_quote_env_value(value)}\n")

    path.write_text("".join(out), encoding="utf-8")
    # Best-effort: keep secrets in env files readable only by the current user.
    # (No-op on platforms where chmod is unsupported.)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
