"""Метрики для оценки качества (скрипты в eval/). Вынесены в пакет, чтобы покрыть тестами."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from career_mcp.skills import SkillDictionary
from career_mcp.text import norm


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return None if p is None or r is None else 0.0
        return 2 * p * r / (p + r)

    def add(self, other: PRF) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


def normalize_set(items: list[str], skills: SkillDictionary) -> set[str]:
    return {norm(skills.normalize(i)) for i in items if i and i.strip()}


def compare(gold: list[str], pred: list[str], skills: SkillDictionary) -> PRF:
    g, p = normalize_set(gold, skills), normalize_set(pred, skills)
    return PRF(tp=len(g & p), fp=len(p - g), fn=len(g - p))


def macro(prfs: list[PRF]) -> tuple[float | None, float | None]:
    ps = [x.precision for x in prfs if x.precision is not None]
    rs = [x.recall for x in prfs if x.recall is not None]
    return (sum(ps) / len(ps) if ps else None, sum(rs) / len(rs) if rs else None)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Ранговая корреляция Спирмена (с учётом одинаковых значений)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def split_labels(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").replace(",", ";").split(";") if x.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"
