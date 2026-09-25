"""Зарплаты, валюты, дедупликация, агрегаты среза рынка."""

from __future__ import annotations

import pytest

from career_mcp.config import PROJECT_ROOT
from career_mcp.market import (
    build_snapshot,
    dedup,
    monthly_net_rub,
    ndfl,
    net_from_gross_monthly,
    salary_stats,
    to_rub,
)
from career_mcp.skills import SkillDictionary

from .conftest import load_fixture

VACANCIES = load_fixture("vacancies.json")
RATES = {c["code"]: c["rate"] for c in load_fixture("dictionaries.json")["currency"]}


@pytest.fixture(scope="module")
def skills():
    return SkillDictionary.load(PROJECT_ROOT / "data" / "skills_synonyms.yaml")


def test_usd_converted_to_rubles_by_hh_rate():
    # rate USD = 0.0125: 1 ₽ стоит 0,0125 $, значит 2000 $ = 160 000 ₽
    assert to_rub(2000, "USD", RATES) == pytest.approx(160_000)
    assert to_rub(50_000, "RUR", RATES) == 50_000


def test_unknown_currency_is_excluded():
    v = {"salary_range": {"from": 100, "to": None, "currency": "XYZ", "gross": False, "mode": {"id": "MONTH"}}}
    assert monthly_net_rub(v, RATES) == ("unknown_currency", None)


def test_gross_to_net_progressive_ndfl():
    assert ndfl(2_400_000) == pytest.approx(312_000)  # 13% до 2,4 млн
    assert ndfl(3_600_000) == pytest.approx(312_000 + 180_000)  # 15% сверх 2,4 млн
    assert net_from_gross_monthly(100_000) == pytest.approx(87_000)
    assert net_from_gross_monthly(300_000) == pytest.approx(259_000)


def test_usd_gross_vacancy_to_monthly_net_rub():
    status, amount = monthly_net_rub(VACANCIES["100003"], RATES)
    # середина 2500 $ → 200 000 ₽ до вычета → 174 000 на руки
    assert status == "ok"
    assert amount == pytest.approx(174_000)


def test_vacancy_without_salary_not_in_salary_stats():
    stats = salary_stats([VACANCIES["100001"], VACANCIES["100004"]], RATES)
    assert stats.vacancies_total == 2
    assert stats.with_salary == 1
    assert stats.share_with_salary == 0.5
    assert stats.used_in_stats == 1
    assert stats.median == 100_000  # только 100001; 100004 не считается нулём


def test_hourly_salary_excluded_from_monthly_stats():
    stats = salary_stats([VACANCIES["100005"], VACANCIES["100006"]], RATES)
    assert stats.with_salary == 2
    assert stats.excluded_not_monthly == 1
    assert stats.used_in_stats == 1
    assert stats.median == 259_000


def test_duplicate_in_two_cities_counted_once():
    unique, removed = dedup([VACANCIES["100001"], VACANCIES["100002"], VACANCIES["100003"]])
    assert removed == 1
    assert [v["id"] for v in unique] == ["100001", "100003"]


def test_same_employer_different_vacancies_not_merged():
    unique, removed = dedup([VACANCIES["100001"], VACANCIES["100006"]])
    assert removed == 0 and len(unique) == 2


def test_snapshot_on_fixture(skills):
    snap = build_snapshot(
        query="ml", area="Москва", found=6, details=list(VACANCIES.values()),
        sample_requested=100, rates=RATES, skills=skills,
    )
    assert snap.duplicates_removed == 1
    assert snap.sample_unique == 5
    assert snap.experience == {"Нет опыта": 2, "От 1 года до 3 лет": 2, "От 3 до 6 лет": 1}
    assert snap.work_format == {"Гибрид": 2, "Из дома": 2, "На месте работодателя": 1}

    s = snap.salary
    assert (s.vacancies_total, s.with_salary, s.used_in_stats, s.excluded_not_monthly) == (5, 4, 3, 1)
    assert s.share_with_salary == 0.8
    # на руки в месяц: 100 000 (100001), 174 000 (100003), 259 000 (100006)
    assert (s.q1, s.median, s.q3) == (137_000, 174_000, 216_500)

    top = {t.skill: t.count for t in snap.top_skills}
    assert top["Python"] == 4
    assert top["PostgreSQL"] == 2  # «Postgres» и «PostgreSQL» — один навык
    assert "Postgres" not in top
    assert snap.top_skills[0].skill == "Python"
