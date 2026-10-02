"""Тесты на настоящих ответах hh.ru (записаны eval/record_fixtures.py 2026-09-25, обезличены).

Синтетические фикстуры проверяют логику на заранее известных ответах, эти —
что разбор не ломается на реальных данных: пустые key_skills, зарплата null,
справочные названия с неразрывными пробелами и т. п.
"""

from __future__ import annotations

import pytest

from career_mcp.config import PROJECT_ROOT
from career_mcp.extract import rule_requirements
from career_mcp.market import build_snapshot
from career_mcp.models import short_from_item
from career_mcp.server import vacancy_detail
from career_mcp.skills import SkillDictionary

from .conftest import FIXTURES, load_fixture

pytestmark = pytest.mark.skipif(not (FIXTURES / "recorded").exists(), reason="нет записанных фикстур")


@pytest.fixture(scope="module")
def recorded():
    return {
        "search": load_fixture("recorded/search.json"),
        "vacancies": load_fixture("recorded/vacancies.json"),
        "dictionaries": load_fixture("recorded/dictionaries.json"),
    }


@pytest.fixture(scope="module")
def skills():
    return SkillDictionary.load(PROJECT_ROOT / "data" / "skills_synonyms.yaml")


def test_personal_data_is_stripped(recorded):
    for v in recorded["vacancies"].values():
        assert "contacts" not in v and "manager" not in v and "address" not in v


def test_search_items_compress_cleanly(recorded):
    for item in recorded["search"]["items"]:
        short = short_from_item(item)
        assert short.id and short.name
        assert all("\xa0" not in f for f in short.work_format)


def test_every_vacancy_parses(recorded, skills):
    for v in recorded["vacancies"].values():
        detail = vacancy_detail(v)
        assert "<" not in detail.description
        req = rule_requirements(v, skills)
        assert req.vacancy_id == str(v["id"])
        assert req.grade in {None, "intern", "junior", "middle", "senior", "lead"}


def test_most_real_vacancies_yield_skills(recorded, skills):
    with_skills = sum(bool(rule_requirements(v, skills).must_have) for v in recorded["vacancies"].values())
    assert with_skills >= len(recorded["vacancies"]) * 0.7


def test_snapshot_on_real_data_is_consistent(recorded, skills):
    rates = {c["code"]: c["rate"] for c in recorded["dictionaries"]["currency"]}
    details = list(recorded["vacancies"].values())
    snap = build_snapshot(
        query="ML стажёр", area="Москва", found=recorded["search"]["found"], details=details,
        sample_requested=len(details), rates=rates, skills=skills,
    )
    s = snap.salary
    assert snap.sample_unique + snap.duplicates_removed == len(details)
    assert s.used_in_stats <= s.with_salary <= s.vacancies_total == snap.sample_unique
    if s.median is not None:
        assert s.q1 <= s.median <= s.q3
    assert sum(snap.experience.values()) == snap.sample_unique
