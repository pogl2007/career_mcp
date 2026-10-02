"""Нормализация и сопоставление навыков.

Основа — словарь синонимов (data/skills_synonyms.yaml): детерминированно,
объяснимо и быстро. «Postgres» и «PostgreSQL» сливаются в один навык, «ML»
находится в тексте, а «R» в «R&D» — нет.

Эмбеддинги — опция для того, чего нет в словаре: «PyTorch Lightning» и
«Lightning» или «LLM» и «большие языковые модели» на другом языке. Модель
тяжёлая (torch), поэтому включается флагом EMBEDDINGS_ENABLED.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from career_mcp.text import norm

log = logging.getLogger("career_mcp.skills")

# категории, которые не считаем «технологиями» для поля stack
NON_TECH = {"human_language", "soft"}


@dataclass(frozen=True)
class Skill:
    name: str
    category: str | None = None


def _clean(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip().strip(".,;:·•").strip()


def _alias_key(alias: str) -> str:
    return norm(_clean(alias))


def _alias_regex(alias: str) -> re.Pattern[str] | None:
    alias = _clean(alias).replace("ё", "е").replace("Ё", "Е")
    if len(alias) <= 1:
        return None
    body = re.escape(alias).replace(r"\*", r"\w*").replace(r"\ ", r"[\s\-]+")
    flags = 0 if len(alias) == 2 else re.IGNORECASE
    return re.compile(rf"(?<![\w+#]){body}(?![\w+#])", flags)


class SkillDictionary:
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._exact: dict[str, Skill] = {}
        self._wildcards: list[tuple[re.Pattern[str], Skill]] = []
        self._text: list[tuple[re.Pattern[str], Skill]] = []
        self._by_name: dict[str, Skill] = {}
        self._implies: dict[str, list[str]] = {}
        for entry in entries:
            skill = Skill(entry["name"], entry.get("category"))
            self._by_name[norm(skill.name)] = skill
            if entry.get("implies"):
                self._implies[skill.name] = [str(x) for x in entry["implies"]]
            for alias in [skill.name, *(entry.get("aliases") or [])]:
                alias = str(alias)
                if "*" in alias:
                    body = re.escape(_alias_key(alias)).replace(r"\*", r"\w*")
                    self._wildcards.append((re.compile(rf"{body}", re.IGNORECASE), skill))
                else:
                    self._exact.setdefault(_alias_key(alias), skill)
                pattern = _alias_regex(alias)
                if pattern is not None:
                    self._text.append((pattern, skill))

    @classmethod
    def load(cls, path: Path) -> SkillDictionary:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(data.get("skills") or [])

    @property
    def skills(self) -> list[Skill]:
        return list(self._by_name.values())

    def lookup(self, raw: str) -> Skill | None:
        key = _alias_key(raw)
        if not key:
            return None
        if key in self._exact:
            return self._exact[key]
        for pattern, skill in self._wildcards:
            if pattern.fullmatch(key):
                return skill
        return None

    def normalize(self, raw: str) -> str:
        """Каноническое название навыка; неизвестный навык возвращается очищенным как есть."""
        skill = self.lookup(raw)
        return skill.name if skill else _clean(raw)

    def category(self, name: str) -> str | None:
        skill = self._by_name.get(norm(name)) or self.lookup(name)
        return skill.category if skill else None

    def is_tech(self, name: str) -> bool:
        cat = self.category(name)
        return cat is not None and cat not in NON_TECH

    def find_in_text(self, text: str) -> list[str]:
        """Канонические навыки, упомянутые в тексте, в порядке первого упоминания."""
        if not text:
            return []
        text = text.replace("ё", "е").replace("Ё", "Е")
        first_pos: dict[str, int] = {}
        for pattern, skill in self._text:
            m = pattern.search(text)
            if m and (skill.name not in first_pos or m.start() < first_pos[skill.name]):
                first_pos[skill.name] = m.start()
        return sorted(first_pos, key=first_pos.__getitem__)

    def implied(self, names: list[str]) -> list[str]:
        """Навыки, которые следуют из имеющихся (PyTorch → Deep Learning), и которых ещё нет в списке."""
        have = set(names)
        out: list[str] = []
        queue = list(names)
        while queue:
            for implied in self._implies.get(queue.pop(0), []):
                if implied not in have:
                    have.add(implied)
                    out.append(implied)
                    queue.append(implied)
        return out

    def normalize_many(self, raws: list[str]) -> list[str]:
        out: list[str] = []
        for raw in raws:
            name = self.normalize(raw)
            if name and name not in out:
                out.append(name)
        return out


class EmbeddingMatcher:
    """Нечёткое сопоставление навыков по косинусной близости эмбеддингов."""

    def __init__(self, model_name: str, threshold: float) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Эмбеддинги включены, но sentence-transformers не установлен: "
                'pip install -e ".[embeddings]"'
            ) from exc
        self.threshold = threshold
        self._model = SentenceTransformer(model_name)

    def best_matches(self, required: list[str], available: list[str]) -> dict[str, tuple[str, float]]:
        """Для каждого требуемого навыка — ближайший из имеющихся, если близость ≥ порога."""
        if not required or not available:
            return {}
        req = self._model.encode(required, normalize_embeddings=True)
        have = self._model.encode(available, normalize_embeddings=True)
        sims = req @ have.T
        out: dict[str, tuple[str, float]] = {}
        for i, name in enumerate(required):
            j = int(sims[i].argmax())
            score = float(sims[i][j])
            if score >= self.threshold:
                out[name] = (available[j], round(score, 3))
        return out


def match_skills(
    required: list[str],
    have: set[str],
    dictionary: SkillDictionary,
    embeddings: EmbeddingMatcher | None = None,
) -> tuple[list[tuple[str, str, str, float | None]], list[str]]:
    """Сопоставляет требуемые навыки с имеющимися.

    Возвращает (совпадения [(требуемый, чем закрыт, способ, близость)], чего не хватает).
    """
    have_norm = {dictionary.normalize(h) for h in have}
    have_keys = {norm(h): h for h in have_norm}
    matched: list[tuple[str, str, str, float | None]] = []
    missing: list[str] = []
    for req in required:
        canon = dictionary.normalize(req)
        if norm(canon) in have_keys:
            method = "exact" if norm(req) == norm(canon) else "synonym"
            matched.append((req, have_keys[norm(canon)], method, None))
        else:
            missing.append(req)
    if embeddings is not None and missing:
        fuzzy = embeddings.best_matches(missing, sorted(have_norm))
        for req, (by, score) in fuzzy.items():
            matched.append((req, by, "embedding", score))
        missing = [m for m in missing if m not in fuzzy]
    return matched, missing
