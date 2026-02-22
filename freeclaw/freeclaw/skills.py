from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ClawConfig


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path
    skill_md: Path


def _candidate_skill_dirs(cfg: ClawConfig) -> list[Path]:
    dirs: list[Path] = []
    for d in (cfg.skills_dirs or []):
        p = Path(str(d)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        dirs.append(p)

    # Also allow a local ./skills folder without explicitly configuring it.
    local = (Path.cwd() / "skills").resolve()
    if local not in dirs:
        dirs.append(local)
    return dirs


def iter_skills(cfg: ClawConfig) -> list[Skill]:
    skills: dict[str, Skill] = {}
    for base in _candidate_skill_dirs(cfg):
        if not base.exists() or not base.is_dir():
            continue
        for child in sorted(base.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            md = child / "SKILL.md"
            if not md.exists() or not md.is_file():
                continue
            name = child.name
            # First directory in search path wins.
            if name not in skills:
                skills[name] = Skill(name=name, path=child, skill_md=md)
    return list(skills.values())


def find_skill(cfg: ClawConfig, name: str) -> Skill | None:
    for s in iter_skills(cfg):
        if s.name == name:
            return s
    return None


def load_skill_text(skill: Skill, *, max_bytes: int = 80_000) -> str:
    data = skill.skill_md.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def render_enabled_skills_system(cfg: ClawConfig) -> str:
    enabled = [s for s in (cfg.enabled_skills or []) if str(s).strip()]
    if not enabled:
        return ""
    all_skills = {s.name: s for s in iter_skills(cfg)}

    blocks: list[str] = []
    blocks.append("Enabled skills (follow these instructions when relevant):")
    for name in enabled:
        sk = all_skills.get(name)
        if not sk:
            continue
        txt = load_skill_text(sk)
        blocks.append(f"\n=== SKILL: {name} ===\n{txt.strip()}\n")
    return "\n".join(blocks).strip() + "\n"

