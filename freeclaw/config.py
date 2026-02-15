import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import config_path, skills_dir


def _default_config_path() -> Path:
    return config_path()


def default_config_path() -> Path:
    return _default_config_path()


def load_config_dict(path: str | None) -> tuple[Path, dict[str, Any]]:
    cfg_path = Path(path) if path else _default_config_path()
    if cfg_path.exists():
        return cfg_path, json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg_path, {}


def save_config_dict(cfg_path: Path, data: dict[str, Any]) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def default_skills_dir() -> Path:
    return skills_dir()


@dataclass(frozen=True)
class ClawConfig:
    onboarded: bool = False
    provider: str = "nim"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str | None = None
    assistant_name: str = "Freeclaw"
    assistant_tone: str = "Direct, pragmatic, concise. Ask clarifying questions when needed."
    temperature: float = 0.7
    max_tokens: int = 1024
    workspace_dir: str = "workspace"
    tool_root: str = "."
    tool_max_read_bytes: int = 200_000
    tool_max_write_bytes: int = 2_000_000
    tool_max_list_entries: int = 2_000
    skills_dirs: list[str] = field(default_factory=list)
    enabled_skills: list[str] = field(default_factory=list)
    discord_prefix: str = "!claw"
    discord_history_messages: int = 40
    discord_respond_to_all: bool = False
    discord_app_id: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ClawConfig":
        skills_dirs = d.get("skills_dirs")
        if not isinstance(skills_dirs, list):
            skills_dirs = []
        skills_dirs = [str(x) for x in skills_dirs if str(x).strip()]
        if not skills_dirs:
            skills_dirs = [str(default_skills_dir())]

        enabled_skills = d.get("enabled_skills")
        if not isinstance(enabled_skills, list):
            enabled_skills = []
        enabled_skills = [str(x) for x in enabled_skills if str(x).strip()]

        return ClawConfig(
            onboarded=bool(d.get("onboarded", False)),
            provider=str(d.get("provider") or "nim"),
            base_url=str(d.get("base_url") or "https://integrate.api.nvidia.com/v1"),
            model=(str(d["model"]) if d.get("model") else None),
            assistant_name=str(d.get("assistant_name") or "Freeclaw"),
            assistant_tone=str(
                d.get("assistant_tone")
                or "Direct, pragmatic, concise. Ask clarifying questions when needed."
            ),
            temperature=float(d.get("temperature", 0.7)),
            max_tokens=int(d.get("max_tokens", 1024)),
            workspace_dir=str(d.get("workspace_dir") or "workspace"),
            tool_root=str(d.get("tool_root") or "."),
            tool_max_read_bytes=int(d.get("tool_max_read_bytes", 200_000)),
            tool_max_write_bytes=int(d.get("tool_max_write_bytes", 2_000_000)),
            tool_max_list_entries=int(d.get("tool_max_list_entries", 2_000)),
            skills_dirs=skills_dirs,
            enabled_skills=enabled_skills,
            discord_prefix=str(d.get("discord_prefix") or "!claw"),
            discord_history_messages=int(d.get("discord_history_messages", 40)),
            discord_respond_to_all=bool(d.get("discord_respond_to_all", False)),
            discord_app_id=(str(d["discord_app_id"]).strip() if d.get("discord_app_id") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "onboarded": bool(self.onboarded),
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "assistant_name": self.assistant_name,
            "assistant_tone": self.assistant_tone,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "workspace_dir": self.workspace_dir,
            "tool_root": self.tool_root,
            "tool_max_read_bytes": self.tool_max_read_bytes,
            "tool_max_write_bytes": self.tool_max_write_bytes,
            "tool_max_list_entries": self.tool_max_list_entries,
            "skills_dirs": list(self.skills_dirs or []),
            "enabled_skills": list(self.enabled_skills or []),
            "discord_prefix": self.discord_prefix,
            "discord_history_messages": self.discord_history_messages,
            "discord_respond_to_all": bool(self.discord_respond_to_all),
            "discord_app_id": (self.discord_app_id or None),
        }


def load_config(path: str | None) -> ClawConfig:
    # Env overrides then config file; callers should load dotenv first.
    cfg_path, file_cfg = load_config_dict(path)
    cfg = ClawConfig.from_dict(file_cfg)

    provider = os.getenv("FREECLAW_PROVIDER")
    base_url = os.getenv("FREECLAW_BASE_URL")
    model = os.getenv("FREECLAW_MODEL")
    assistant_name = os.getenv("FREECLAW_ASSISTANT_NAME")
    assistant_tone = os.getenv("FREECLAW_ASSISTANT_TONE")
    temperature = os.getenv("FREECLAW_TEMPERATURE")
    max_tokens = os.getenv("FREECLAW_MAX_TOKENS")
    tool_root = os.getenv("FREECLAW_TOOL_ROOT")
    workspace_dir = os.getenv("FREECLAW_WORKSPACE_DIR")
    tool_max_read_bytes = os.getenv("FREECLAW_TOOL_MAX_READ_BYTES")
    tool_max_write_bytes = os.getenv("FREECLAW_TOOL_MAX_WRITE_BYTES")
    tool_max_list_entries = os.getenv("FREECLAW_TOOL_MAX_LIST_ENTRIES")
    discord_prefix = os.getenv("FREECLAW_DISCORD_PREFIX")
    discord_history_messages = os.getenv("FREECLAW_DISCORD_HISTORY_MESSAGES")
    discord_respond_to_all = os.getenv("FREECLAW_DISCORD_RESPOND_TO_ALL")
    discord_app_id = os.getenv("FREECLAW_DISCORD_APP_ID")

    return ClawConfig(
        onboarded=cfg.onboarded,
        provider=(provider or cfg.provider),
        base_url=(base_url or cfg.base_url),
        model=(model or cfg.model),
        assistant_name=(assistant_name or cfg.assistant_name),
        assistant_tone=(assistant_tone or cfg.assistant_tone),
        temperature=(float(temperature) if temperature else cfg.temperature),
        max_tokens=(int(max_tokens) if max_tokens else cfg.max_tokens),
        workspace_dir=(workspace_dir or cfg.workspace_dir),
        tool_root=(tool_root or cfg.tool_root),
        tool_max_read_bytes=(int(tool_max_read_bytes) if tool_max_read_bytes else cfg.tool_max_read_bytes),
        tool_max_write_bytes=(
            int(tool_max_write_bytes) if tool_max_write_bytes else cfg.tool_max_write_bytes
        ),
        tool_max_list_entries=(
            int(tool_max_list_entries) if tool_max_list_entries else cfg.tool_max_list_entries
        ),
        skills_dirs=cfg.skills_dirs,
        enabled_skills=cfg.enabled_skills,
        discord_prefix=(discord_prefix or cfg.discord_prefix),
        discord_history_messages=(
            int(discord_history_messages)
            if discord_history_messages
            else cfg.discord_history_messages
        ),
        discord_respond_to_all=(
            str(discord_respond_to_all).strip().lower() in {"1", "true", "yes", "y", "on"}
            if discord_respond_to_all is not None
            else cfg.discord_respond_to_all
        ),
        discord_app_id=(discord_app_id.strip() if discord_app_id and discord_app_id.strip() else cfg.discord_app_id),
    )


def write_default_config(path: str | None) -> Path:
    cfg_path = Path(path) if path else _default_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists():
        raise SystemExit(f"Config already exists: {cfg_path}")

    cfg = {
        "onboarded": False,
        "provider": "nim",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": None,
        "assistant_name": "Freeclaw",
        "assistant_tone": "Direct, pragmatic, concise. Ask clarifying questions when needed.",
        "temperature": 0.7,
        "max_tokens": 1024,
        "workspace_dir": "workspace",
        "tool_root": ".",
        "tool_max_read_bytes": 200_000,
        "tool_max_write_bytes": 2_000_000,
        "tool_max_list_entries": 2_000,
        "skills_dirs": [str(default_skills_dir())],
        "enabled_skills": [],
        "discord_prefix": "!claw",
        "discord_history_messages": 40,
        "discord_respond_to_all": False,
        "discord_app_id": None,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg_path
