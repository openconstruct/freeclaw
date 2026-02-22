from __future__ import annotations

import re
from pathlib import Path

from .paths import config_dir

_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_agent_name(name: str) -> str:
    n = str(name or "").strip()
    if not n:
        raise ValueError("agent name is required")
    if "/" in n or "\\" in n:
        raise ValueError("agent name must not contain path separators")
    if not _AGENT_NAME_RE.fullmatch(n):
        raise ValueError(
            "invalid agent name (use 1-64 chars: letters, numbers, '_' or '-', starting with a letter/number)"
        )
    return n


def agents_dir() -> Path:
    return (config_dir() / "agents").resolve()


def agent_dir(name: str) -> Path:
    n = validate_agent_name(name)
    return (agents_dir() / n).resolve()


def agent_config_path(name: str) -> Path:
    return agent_dir(name) / "config.json"


def agent_env_path(name: str) -> Path:
    return agent_dir(name) / ".env"


def iter_agents() -> list[str]:
    base = agents_dir()
    if not base.exists() or not base.is_dir():
        return []
    out: list[str] = []
    for child in sorted(base.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if (child / "config.json").exists():
            out.append(child.name)
    return out


def resolve_agent_name(name: str) -> str:
    """
    Resolve an agent name to an existing profile directory if it matches
    case-insensitively.

    This avoids surprising duplicates like "Free2" vs "free2" when users type
    agent names on the command line.
    """
    n = validate_agent_name(name)
    if agent_dir(n).exists():
        return n
    matches = [a for a in iter_agents() if a.lower() == n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous agent name {n!r}; matches: {', '.join(matches)}")
    return n
