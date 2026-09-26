"""Резюме пользователя из локального Markdown-файла.

Путь берётся только из настроек (RESUME_PATH): модель не может попросить сервер
прочитать произвольный файл. Файл перечитывается при изменении — можно править
резюме, не перезапуская сервер.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from career_mcp.skills import SkillDictionary

_SKILLS_HEADING = re.compile(r"^#+\s*.*(навык|skills|стек|технологи|компетенци)", re.I)
_ANY_HEADING = re.compile(r"^#+\s")


class ResumeNotFound(Exception):
    pass


@dataclass
class Resume:
    path: Path
    text: str
    skills: list[str] = field(default_factory=list)
    explicit_skills: list[str] = field(default_factory=list)


def _skills_section(text: str) -> list[str]:
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if _SKILLS_HEADING.match(line):
            inside = True
            continue
        if inside and _ANY_HEADING.match(line):
            inside = False
        if inside:
            item = re.sub(r"^[\s>*\-•]+", "", line)
            item = re.sub(r"\*\*([^*]+)\*\*:?", "", item)  # «**Языки:** Python» → «Python»
            out.extend(p for p in re.split(r"[,;·|/]| — ", item) if p.strip())
    return out


class ResumeStore:
    def __init__(self, path: Path, skills: SkillDictionary) -> None:
        self.path = Path(path)
        self._skills = skills
        self._cached: tuple[float, Resume] | None = None

    def load(self) -> Resume:
        if not self.path.is_file():
            raise ResumeNotFound(
                f"Файл резюме не найден: {self.path.name}. Укажите путь к Markdown-файлу в RESUME_PATH (.env)."
            )
        mtime = self.path.stat().st_mtime
        if self._cached and self._cached[0] == mtime:
            return self._cached[1]
        text = self.path.read_text(encoding="utf-8")
        explicit = [s for s in (self._skills.lookup(x) for x in _skills_section(text)) if s is not None]
        explicit_names = list(dict.fromkeys(s.name for s in explicit))
        found = self._skills.find_in_text(text)
        all_skills = list(dict.fromkeys([*explicit_names, *found]))
        resume = Resume(path=self.path, text=text, skills=all_skills, explicit_skills=explicit_names)
        self._cached = (mtime, resume)
        return resume
