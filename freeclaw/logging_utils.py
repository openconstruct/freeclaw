from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging(*, level: str | None = None, log_file: str | None = None) -> None:
    """
    Configure stdlib logging for both CLI and Discord.

    Defaults:
    - level: FREECLAW_LOG_LEVEL or "warning"
    - log_file: FREECLAW_LOG_FILE (optional)
    """
    lvl = (level or os.getenv("FREECLAW_LOG_LEVEL") or "warning").strip().upper()
    numeric = getattr(logging, lvl, None)
    if not isinstance(numeric, int):
        numeric = logging.WARNING

    lf = (log_file or os.getenv("FREECLAW_LOG_FILE") or "").strip()
    handlers: list[logging.Handler] = []
    if lf:
        p = Path(lf).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(p), encoding="utf-8"))
    handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=handlers,
        force=True,
    )

