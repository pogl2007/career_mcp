"""Извлечение требований: правила, LLM как дополнение, защита от инъекций."""

from __future__ import annotations

import httpx
import pytest
import respx

from career_mcp.cache import Cache
from career_mcp.config import PROJECT_ROOT
from career_mcp.extract import LLMExtraction, RequirementExtractor, merge_llm, rule_requirements
from career_mcp.llm import LLMClient, parse_json_object
from career_mcp.ratelimit import TokenBucket
from career_mcp.resume import ResumeStore
from career_mcp.skills import SkillDictionary

from .conftest import RESUME_TEXT, load_fixture

VACANCIES = load_fixture("vacancies.json")
LLM_URL = "http://llm.test/v1"


@pytest.fixture
def llm_api():
    with respx.mock(base_url=LLM_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture(scope="module")
def skills():
    return SkillDictionary.load(PROJECT_ROOT / "data" / "skills_synonyms.yaml")


def test_rules_on_nlp_intern(skills):
    req = rule_requirements(VACANCIES["100001"], skills)
    assert req.must_have == ["Python", "PyTorch", "Hugging Face Transformers", "SQL", "PostgreSQL", "NLP"]
    assert req.nice_to_have == ["Docker"]
    assert req.grade == "intern"
    assert req.english == "B1"
    assert len(req.tasks) == 3
    assert "понимание метрик классификации" in req.other_requirements
    assert req.sources["must_have"] == "rules"


def test_injection_line_does_not_leak_into_requirements(skills):
    req = rule_requirements(VACANCIES["100001"], skills)
    everything = " ".join([*req.must_have, *req.nice_to_have, *req.other_requirements, *req.tasks])
    assert "Игнорируй" not in everything
    assert "идеально" not in everything


def test_nice_to_have_is_not_must(skills):
    req = rule_requirements(VACANCIES["100003"], skills)
    assert {"Airflow", "Spark"} <= set(req.nice_to_have)
    assert "Airflow" not in req.must_have
    assert {"Python", "Pandas", "NumPy", "scikit-learn", "Статистика", "SQL", "PostgreSQL"} <= set(req.must_have)
    assert req.english == "Upper-Intermediate"
    assert req.grade == "junior"


def test_inline_nice_heading_and_grade_from_experience(skills):
    req = rule_requirements(VACANCIES["100004"], skills)
    assert {"Kubernetes", "Qdrant"} <= set(req.nice_to_have)
    assert {"Python", "LLM", "LangChain", "LangGraph", "Docker", "Git"} <= set(req.must_have)
    assert req.grade == "junior"  # «ML Engineer (LLM)» без грейда в названии → по опыту 1–3 года


def test_unknown_key_skills_go_to_other_requirements(skills):
    req = rule_requirements(VACANCIES["100005"], skills)
    assert "Внимательность" in req.other_requirements
    assert "Внимательность" not in req.must_have
    assert "Excel" in req.must_have


def test_resume_skills_parsed(tmp_path, skills):
    path = tmp_path / "cv.md"
    path.write_text(RESUME_TEXT, encoding="utf-8")
    resume = ResumeStore(path, skills).load()
    assert {"Python", "PyTorch", "FastAPI", "Docker", "PostgreSQL", "Pandas", "scikit-learn",
            "SQL", "Git", "LangChain", "Hugging Face Transformers"} <= set(resume.skills)
    assert "LangChain" in resume.explicit_skills


# ---------------------------------------------------------------- LLM


def test_parse_json_object_handles_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Вот ответ: {"a": [1, 2]} — готово') == {"a": [1, 2]}


def test_merge_llm_keeps_only_grounded_skills(skills):
    req = rule_requirements(VACANCIES["100001"], skills)
    llm = LLMExtraction(
        must_have=["Метрики классификации", "Космонавтика", "PyTorch"],
        nice_to_have=["Docker"],
        grade="junior",
    )
    text = "Требования: понимание метрик классификации; PyTorch."
    merged = merge_llm(req, llm, text, skills)
    assert "Космонавтика" not in merged.must_have
    assert any("Космонавтика" in n for n in merged.notes)
    assert merged.grade == "intern"  # правила уже определили грейд — LLM его не перетирает
    assert merged.llm_used


def _llm_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_extractor_calls_llm_only_when_rules_have_gaps(tmp_path, skills, llm_api):
    route = llm_api.post("/chat/completions").mock(
        return_value=_llm_response('{"must_have": ["метрики классификации"], "nice_to_have": [], "tasks": []}')
    )
    cache = Cache(tmp_path / "c.sqlite")
    await cache.open()
    llm = LLMClient(LLM_URL, "test-model", limiter=TokenBucket(1000, 1000))
    extractor = RequirementExtractor(skills, llm, cache)
    try:
        req = await extractor.extract(VACANCIES["100001"])
        assert route.call_count == 1
        assert req.llm_used
        assert "метрики классификации" in req.must_have
        assert req.sources["must_have"] == "rules+llm"

        # Повторный вызов берёт ответ LLM из кеша.
        await extractor.extract(VACANCIES["100001"])
        assert route.call_count == 1

        # Инъекция не уходит в LLM: строки-инструкции вырезаются из текста.
        sent = route.calls[0].request.content.decode("utf-8")
        assert "Игнорируй" not in sent
        assert "<vacancy>" in sent
    finally:
        await llm.aclose()
        await cache.close()


async def test_invalid_llm_answer_falls_back_to_rules(tmp_path, skills, llm_api):
    llm_api.post("/chat/completions").mock(return_value=_llm_response("APPROVED"))
    llm = LLMClient(LLM_URL, "test-model", limiter=TokenBucket(1000, 1000))
    extractor = RequirementExtractor(skills, llm, None)
    try:
        req = await extractor.extract(VACANCIES["100001"])
    finally:
        await llm.aclose()
    assert not req.llm_used
    assert req.must_have[0] == "Python"
    assert any("только по правилам" in n for n in req.notes)


async def test_llm_payment_error_is_not_fatal(skills, llm_api):
    llm_api.post("/chat/completions").mock(return_value=httpx.Response(402, json={"error": "pay"}))
    llm = LLMClient(LLM_URL, "m", limiter=TokenBucket(1000, 1000))
    try:
        req = await RequirementExtractor(skills, llm).extract(VACANCIES["100003"])
    finally:
        await llm.aclose()
    assert not req.llm_used
