"""Срез рынка: выборка вакансий, дедупликация, зарплаты, распределения, навыки.

Как считаются зарплаты:
- берём только вакансии с указанной зарплатой; остальные не «нули», а отсутствие
  данных — иначе медиана уехала бы вниз. Размер выборки и доля вакансий с зарплатой
  всегда в ответе, чтобы было видно, на скольких числах держится медиана;
- только помесячные (salary_range.mode = MONTH): почасовые ставки и оплата за смену
  в одной выборке с месячными окладами исказили бы статистику;
- одна точка на вакансию: середина вилки, если указаны обе границы, иначе та, что есть;
- валюта → рубли по курсу из справочника hh (rate — сколько единиц валюты стоит 1 ₽);
- «до вычета налогов» → «на руки» по прогрессивной шкале НДФЛ (с 2025 года).
"""

from __future__ import annotations

import asyncio
import hashlib
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from career_mcp.hh_client import MAX_DEPTH, HHClient, HHError, HHNotFound
from career_mcp.models import GapItem, MarketSnapshot, SalaryStats, SkillCount
from career_mcp.skills import SkillDictionary
from career_mcp.text import html_to_text, norm

# Годовой доход → ставка НДФЛ на часть дохода в этом диапазоне (с 2025 года).
NDFL_BRACKETS: list[tuple[float, float]] = [
    (2_400_000, 0.13),
    (5_000_000, 0.15),
    (20_000_000, 0.18),
    (50_000_000, 0.20),
    (float("inf"), 0.22),
]

RUB_CODES = {"RUR", "RUB"}

Progress = Callable[[int, int], Awaitable[None]]


# ---------------------------------------------------------------- зарплата


def ndfl(annual_income: float) -> float:
    tax, lower = 0.0, 0.0
    for upper, rate in NDFL_BRACKETS:
        if annual_income <= lower:
            break
        tax += (min(annual_income, upper) - lower) * rate
        lower = upper
    return tax


def net_from_gross_monthly(gross: float) -> float:
    annual = gross * 12
    return (annual - ndfl(annual)) / 12


def to_rub(amount: float, currency: str | None, rates: dict[str, float]) -> float | None:
    if currency is None or currency in RUB_CODES:
        return float(amount)
    rate = rates.get(currency)
    if not rate:
        return None
    return float(amount) / rate


def salary_point(s: dict[str, Any]) -> float | None:
    lo, hi = s.get("from"), s.get("to")
    if lo is not None and hi is not None:
        return (lo + hi) / 2
    return lo if lo is not None else hi


def monthly_net_rub(vacancy: dict[str, Any], rates: dict[str, float]) -> tuple[str, float | None]:
    """Возвращает (статус, сумма): ok | none | not_monthly | unknown_currency."""
    s = vacancy.get("salary_range") or vacancy.get("salary")
    if not s:
        return "none", None
    point = salary_point(s)
    if point is None:
        return "none", None
    mode = (s.get("mode") or {}).get("id") if isinstance(s.get("mode"), dict) else None
    if mode is not None and mode != "MONTH":
        return "not_monthly", None
    rub = to_rub(point, s.get("currency"), rates)
    if rub is None:
        return "unknown_currency", None
    if s.get("gross"):
        rub = net_from_gross_monthly(rub)
    return "ok", rub


def quartiles(values: list[float]) -> tuple[float, float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        v = values[0]
        return v, v, v
    q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q2, q3


def salary_stats(vacancies: list[dict[str, Any]], rates: dict[str, float]) -> SalaryStats:
    counts: Counter[str] = Counter()
    values: list[float] = []
    for v in vacancies:
        status, amount = monthly_net_rub(v, rates)
        counts[status] += 1
        if amount is not None:
            values.append(amount)
    total = len(vacancies)
    with_salary = total - counts["none"]
    q = quartiles(values)
    return SalaryStats(
        vacancies_total=total,
        with_salary=with_salary,
        share_with_salary=round(with_salary / total, 3) if total else None,
        used_in_stats=len(values),
        excluded_not_monthly=counts["not_monthly"],
        excluded_unknown_currency=counts["unknown_currency"],
        median=round(q[1]) if q else None,
        q1=round(q[0]) if q else None,
        q3=round(q[2]) if q else None,
        min=round(min(values)) if values else None,
        max=round(max(values)) if values else None,
    )


# ---------------------------------------------------------------- дедупликация


def dedup_key(v: dict[str, Any]) -> tuple[str, str, str]:
    """Одна вакансия, размещённая в нескольких городах, у hh — разные id с тем же
    работодателем, названием и текстом. Ключ: работодатель + название + хеш описания
    (без описания — хеш зарплаты, чтобы не склеить разные вакансии одного работодателя)."""
    employer = str((v.get("employer") or {}).get("id") or norm((v.get("employer") or {}).get("name") or ""))
    name = norm(v.get("name") or "")
    body = v.get("description")
    if body:
        digest = hashlib.sha1(norm(html_to_text(body)).encode("utf-8")).hexdigest()
    else:
        s = v.get("salary_range") or v.get("salary") or {}
        digest = f"{s.get('from')}-{s.get('to')}-{s.get('currency')}"
    return employer, name, digest


def dedup(vacancies: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    removed = 0
    for v in vacancies:
        key = dedup_key(v)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(v)
    return unique, removed


# ---------------------------------------------------------------- навыки и распределения


def vacancy_skills(v: dict[str, Any], skills: SkillDictionary) -> list[str]:
    """Навыки вакансии: key_skills после нормализации + словарные совпадения в описании."""
    found = skills.normalize_many([s.get("name", "") for s in v.get("key_skills") or []])
    for name in skills.find_in_text(html_to_text(v.get("description"))):
        if name not in found:
            found.append(name)
    return found


def skill_counts(vacancies: list[dict[str, Any]], skills: SkillDictionary) -> Counter[str]:
    counter: Counter[str] = Counter()
    for v in vacancies:
        counter.update(set(vacancy_skills(v, skills)))
    return counter


def top_skills(vacancies: list[dict[str, Any]], skills: SkillDictionary, n: int = 20) -> list[SkillCount]:
    total = len(vacancies) or 1
    counter = skill_counts(vacancies, skills)
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [SkillCount(skill=k, count=c, share=round(c / total, 3), category=skills.category(k)) for k, c in ranked]


def distribution(vacancies: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for v in vacancies:
        value = v.get(field)
        if isinstance(value, list):
            counter.update(x.get("name") or x.get("id") for x in value if isinstance(x, dict))
        elif isinstance(value, dict):
            counter[value.get("name") or value.get("id")] += 1
        else:
            counter["не указано"] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------- выборка


async def collect_sample(
    hh: HHClient,
    params: dict[str, Any],
    sample_size: int,
    progress: Progress | None = None,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Ищет вакансии, берёт первые `sample_size` и загружает каждую целиком
    (key_skills есть только в полной вакансии). Возвращает (найдено всего, вакансии, заметки)."""
    notes: list[str] = []
    per_page = min(100, sample_size)  # 100 — максимум API; у инструмента поиска свой лимит 50
    first = await hh.search_vacancies({**params, "page": 0, "per_page": per_page})
    found = int(first.get("found", 0))
    pages = int(first.get("pages", 1))
    items = list(first.get("items") or [])
    page = 1
    while len(items) < sample_size and page < pages and (page + 1) * per_page <= MAX_DEPTH:
        data = await hh.search_vacancies({**params, "page": page, "per_page": per_page})
        items.extend(data.get("items") or [])
        page += 1
    items = items[:sample_size]
    if found < sample_size:
        notes.append(f"hh нашёл только {found} вакансий — выборка меньше запрошенной.")

    semaphore = asyncio.Semaphore(4)
    done = 0

    async def fetch(item: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal done
        async with semaphore:
            try:
                return await hh.get_vacancy(str(item["id"]))
            except HHNotFound:
                return None
            finally:
                done += 1
                if progress is not None:
                    await progress(done, len(items))

    results = await asyncio.gather(*(fetch(i) for i in items), return_exceptions=True)
    details: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, HHError):
            raise r
        if isinstance(r, BaseException):
            raise r
        if r is not None:
            details.append(r)
    skipped = len(items) - len(details)
    if skipped:
        notes.append(f"{skipped} вакансий уже удалены или скрыты — пропущены.")
    return found, details, notes


def build_snapshot(
    *,
    query: str,
    area: str,
    found: int,
    details: list[dict[str, Any]],
    sample_requested: int,
    rates: dict[str, float],
    skills: SkillDictionary,
    notes: list[str] | None = None,
) -> MarketSnapshot:
    unique, removed = dedup(details)
    all_notes = list(notes or [])
    all_notes.append(
        f"Распределения, зарплаты и навыки посчитаны по {len(unique)} уникальным вакансиям "
        f"из выборки, а не по всем {found} найденным."
    )
    all_notes.append(
        "Зарплата: середина вилки, только помесячные, в рублях по курсу hh, "
        "«до вычета налогов» пересчитано на руки по прогрессивной шкале НДФЛ."
    )
    return MarketSnapshot(
        query=query,
        area=area,
        found_total=found,
        sample_requested=sample_requested,
        sample_fetched=len(details),
        duplicates_removed=removed,
        sample_unique=len(unique),
        experience=distribution(unique, "experience"),
        work_format=distribution(unique, "work_format"),
        salary=salary_stats(unique, rates),
        top_skills=top_skills(unique, skills, n=20),
        notes=all_notes,
    )


def gap_items(
    vacancies: list[dict[str, Any]], skills: SkillDictionary, resume_skills: set[str], limit: int = 30
) -> tuple[list[GapItem], list[GapItem]]:
    total = len(vacancies) or 1
    counter = skill_counts(vacancies, skills)
    have = {norm(s) for s in resume_skills}
    missing: list[GapItem] = []
    present: list[GapItem] = []
    for name, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if skills.category(name) == "soft":
            continue
        item = GapItem(skill=name, count=count, share=round(count / total, 3), category=skills.category(name))
        (present if norm(name) in have else missing).append(item)
    return missing[:limit], present[:limit]
