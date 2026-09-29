"""Формулы метрик оценки: чтобы цифры в README считались правильно."""

from __future__ import annotations

import pytest

from career_mcp.config import PROJECT_ROOT
from career_mcp.evaluation import PRF, compare, macro, spearman, split_labels
from career_mcp.skills import SkillDictionary


@pytest.fixture(scope="module")
def skills():
    return SkillDictionary.load(PROJECT_ROOT / "data" / "skills_synonyms.yaml")


def test_compare_normalizes_synonyms(skills):
    prf = compare(gold=["Postgres", "PyTorch", "Docker"], pred=["PostgreSQL", "pytorch", "Kubernetes"], skills=skills)
    assert (prf.tp, prf.fp, prf.fn) == (2, 1, 1)
    assert prf.precision == pytest.approx(2 / 3)
    assert prf.recall == pytest.approx(2 / 3)
    assert prf.f1 == pytest.approx(2 / 3)


def test_micro_and_macro(skills):
    a = compare(["Python"], ["Python"], skills)  # P=1, R=1
    b = compare(["SQL", "Docker"], ["SQL", "Git", "Linux"], skills)  # P=1/3, R=1/2
    total = PRF()
    total.add(a)
    total.add(b)
    assert total.precision == pytest.approx(2 / 4)
    assert total.recall == pytest.approx(2 / 3)
    p, r = macro([a, b])
    assert p == pytest.approx((1 + 1 / 3) / 2)
    assert r == pytest.approx((1 + 1 / 2) / 2)


def test_empty_prediction_has_undefined_precision(skills):
    prf = compare(["Python"], [], skills)
    assert prf.precision is None and prf.recall == 0


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_split_labels():
    assert split_labels("Python; PyTorch,  SQL ;") == ["Python", "PyTorch", "SQL"]
