from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    """
    Default config directory lives in the current working directory to keep
    all state local to the project by default.

    Override with FREECLAW_CONFIG_DIR if desired.
    """
    override = os.getenv("FREECLAW_CONFIG_DIR")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    return (Path.cwd() / "config").resolve()


def config_path() -> Path:
    return config_dir() / "config.json"


def env_path() -> Path:
    return config_dir() / ".env"


def memory_db_path() -> Path:
    return config_dir() / "memory.sqlite3"


def skills_dir() -> Path:
    return (config_dir() / "skills").resolve()
