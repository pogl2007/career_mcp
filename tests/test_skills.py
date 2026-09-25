"""Нормализация навыков и поиск в тексте."""

from __future__ import annotations

import pytest

from career_mcp.config import PROJECT_ROOT
from career_mcp.skills import SkillDictionary, match_skills


@pytest.fixture(scope="module")
def skills():
    return SkillDictionary.load(PROJECT_ROOT / "data" / "skills_synonyms.yaml")


@pytest.mark.parametrize("raw", ["Postgres", "PostgreSQL", "postgresql", " psql ", "Постгрес", "postgres."])
def test_postgres_synonyms_merge(skills, raw):
    assert skills.normalize(raw) == "PostgreSQL"


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("sklearn", "scikit-learn"),
        ("Scikit-Learn", "scikit-learn"),
        ("pytorch", "PyTorch"),
        ("k8s", "Kubernetes"),
        ("A/B тесты", "A/B-тестирование"),
        ("Машинное обучение", "Machine Learning"),
        ("HuggingFace", "Hugging Face Transformers"),
        ("английский", "Английский язык"),
        ("R", "R"),
    ],
)
def test_synonyms(skills, raw, canonical):
    assert skills.normalize(raw) == canonical


def test_unknown_skill_is_kept_as_is(skills):
    assert skills.normalize("  Внимательность; ") == "Внимательность"
    assert skills.category("Внимательность") is None


def test_find_in_text_respects_word_boundaries(skills):
    text = "Опыт с PostgreSQL и Postgres, знание C++ и C#, R&D-отдел, Go и Golang, ML-модели, NoSQL."
    found = skills.find_in_text(text)
    assert found.count("PostgreSQL") == 1
    assert "C++" in found and "C#" in found
    assert "R" not in found  # «R» из одного символа в тексте не ищем
    assert "Go" in found
    assert "Machine Learning" in found
    assert "NoSQL" in found
    assert "SQL" not in found  # SQL внутри PostgreSQL и NoSQL не считается


def test_find_in_text_handles_russian_endings(skills):
    found = skills.find_in_text("Знание математической статистики и теории вероятностей, опыт A/B-тестов.")
    assert {"Статистика", "Теория вероятностей", "A/B-тестирование"} <= set(found)


def test_find_in_text_keeps_order_of_first_mention(skills):
    assert skills.find_in_text("Docker, затем Python, потом снова Docker") == ["Docker", "Python"]


def test_short_aliases_are_case_sensitive(skills):
    assert "Go" not in skills.find_in_text("let's go and build it")
    assert "Machine Learning" not in skills.find_in_text("html и xml")


def test_match_skills_uses_synonyms(skills):
    matched, missing = match_skills(["Postgres", "PyTorch", "Kubernetes"], {"PostgreSQL", "pytorch"}, skills)
    got = {m[0]: m[2] for m in matched}
    assert got == {"Postgres": "synonym", "PyTorch": "exact"}
    assert missing == ["Kubernetes"]
