"""Сервер целиком через in-memory fastmcp.Client: tools, resources, prompts."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from career_mcp.server import create_server

from .conftest import load_fixture, make_settings

VACANCIES = load_fixture("vacancies.json")


@pytest.fixture
def hh_api(api):
    api.get("/areas").respond(200, json=load_fixture("areas.json"))
    api.get("/dictionaries").respond(200, json=load_fixture("dictionaries.json"))
    api.get("/vacancies").respond(200, json=load_fixture("search_ml.json"))
    for vid, body in VACANCIES.items():
        api.get(f"/vacancies/{vid}").respond(200, json=body)
    return api


@pytest.fixture
def server_settings(tmp_path):
    return make_settings(tmp_path, hh_rate_per_sec=1000, hh_burst=1000)


@pytest.fixture
async def client(server_settings, hh_api):
    async with Client(create_server(server_settings)) as c:
        yield c


async def test_tools_are_registered(client):
    names = {t.name for t in await client.list_tools()}
    assert {"search_vacancies", "get_vacancy"} <= names
    tool = next(t for t in await client.list_tools() if t.name == "get_vacancy")
    assert "недоверенные" in tool.description
    assert tool.annotations.read_only_hint is True


async def test_search_translates_city_and_filters(client, hh_api):
    result = await client.call_tool(
        "search_vacancies",
        {"query": "ML стажёр", "area": "Москва", "experience": "нет опыта",
         "work_format": "удалённо", "only_with_salary": True, "per_page": 20},
    )
    data = result.structured_content
    assert data["area"] == "Москва"
    assert data["found"] == 6
    first = data["items"][0]
    assert set(first) == {"id", "name", "employer", "salary", "area", "work_format", "experience", "published", "url"}
    assert first["salary"] == "80 000–120 000 ₽, на руки, за месяц"

    params = dict(hh_api.routes[2].calls.last.request.url.params)
    assert params["area"] == "1"
    assert params["experience"] == "noExperience"
    assert params["work_format"] == "REMOTE"
    assert params["label"] == "with_salary"
    assert "only_with_salary" not in params


async def test_unknown_city_suggests_close_names(client):
    with pytest.raises(ToolError) as exc:
        await client.call_tool("search_vacancies", {"query": "python", "area": "Масква"})
    assert "Москва" in str(exc.value)


async def test_per_page_is_capped_at_50(client):
    with pytest.raises(ToolError):
        await client.call_tool("search_vacancies", {"query": "python", "per_page": 100})


async def test_get_vacancy_returns_clean_text_and_sections(client):
    result = await client.call_tool("get_vacancy", {"vacancy_id": "100001"})
    v = result.structured_content
    assert "<" not in v["description"]
    assert "Python, PyTorch" in v["requirements"]
    assert "Docker" in v["nice_to_have"]
    assert v["key_skills"] == ["Python", "PyTorch", "NLP", "Postgres"]
    assert v["full_resource"] == "vacancy://100001"
    assert "не выполнять" in v["untrusted_notice"]
    assert any("Игнорируй" in line for line in v["suspicious_lines"])


async def test_vacancy_resource_is_served_from_cache(client, hh_api):
    await client.call_tool("get_vacancy", {"vacancy_id": "100001"})
    route = next(r for r in hh_api.routes if r.pattern and "100001" in repr(r.pattern))
    calls_before = route.call_count

    contents = await client.read_resource("vacancy://100001")
    payload = json.loads(contents[0].text)

    assert payload["vacancy"]["id"] == "100001"
    assert "не выполнять" in payload["untrusted_notice"]
    assert route.call_count == calls_before


async def test_dictionary_resources(client):
    currencies = json.loads((await client.read_resource("dict://currencies"))[0].text)
    assert {"code": "USD", "name": "Доллары", "rate": 0.0125} in currencies
    experience = json.loads((await client.read_resource("dict://experience"))[0].text)
    assert experience[0]["id"] == "noExperience"
    areas = json.loads((await client.read_resource("dict://areas"))[0].text)
    assert {"id": "1", "name": "Москва", "parent_id": "113"} in areas


async def test_market_snapshot_on_fixture(client):
    result = await client.call_tool("market_snapshot", {"query": "ML", "area": "Москва", "sample_size": 10})
    snap = result.structured_content
    assert snap["found_total"] == 6
    assert snap["sample_fetched"] == 6
    assert snap["duplicates_removed"] == 1
    assert snap["salary"]["median"] == 174_000
    assert snap["salary"]["used_in_stats"] == 3
    assert snap["top_skills"][0]["skill"] == "Python"


async def test_repeated_snapshot_uses_cache_only(client, hh_api):
    await client.call_tool("market_snapshot", {"query": "ML", "sample_size": 10})
    calls = sum(r.call_count for r in hh_api.routes)
    await client.call_tool("market_snapshot", {"query": "ML", "sample_size": 10})
    assert sum(r.call_count for r in hh_api.routes) == calls


async def test_resume_resource(client):
    text = (await client.read_resource("resume://current"))[0].text
    assert text.startswith("# Тестовый Кандидат")


async def test_match_resume(client):
    result = await client.call_tool("match_resume", {"vacancy_id": "100001"})
    m = result.structured_content
    assert m["must_have_coverage"] == pytest.approx(5 / 6, abs=1e-3)
    assert m["missing"] == ["NLP"]
    assert {x["required"] for x in m["matched"]} == {"Python", "PyTorch", "Hugging Face Transformers", "SQL", "PostgreSQL"}
    assert m["nice_to_have_matched"] == ["Docker"]
    assert "LangChain" in m["extra_in_resume"]
    assert m["verdict"].startswith("Хорошее")


async def test_match_resume_accepts_no_path_from_model(client):
    tool = next(t for t in await client.list_tools() if t.name == "match_resume")
    assert set(tool.input_schema["properties"]) == {"vacancy_id"}


async def test_skill_gap(client):
    result = await client.call_tool("skill_gap", {"query": "ML", "sample_size": 10})
    gap = result.structured_content
    assert gap["sample_unique"] == 5
    missing = [g["skill"] for g in gap["missing"]]
    assert "Python" not in missing
    assert missing[0] == "Английский язык"  # в тестовом резюме английского нет, в вакансиях — трижды
    assert {"Kubernetes", "NLP"} <= set(missing)
    counts = [g["count"] for g in gap["missing"]]
    assert counts == sorted(counts, reverse=True)
    assert "Python" in [g["skill"] for g in gap["already_have"]]


async def test_missing_resume_file_is_explained(tmp_path, hh_api):
    settings = make_settings(tmp_path, resume_path=tmp_path / "nope.md")
    async with Client(create_server(settings)) as c:
        with pytest.raises(ToolError) as exc:
            await c.call_tool("match_resume", {"vacancy_id": "100001"})
    assert "RESUME_PATH" in str(exc.value)


async def test_captcha_is_reported_to_model_without_retries(server_settings, api):
    api.get("/areas").respond(200, json=load_fixture("areas.json"))
    route = api.get("/vacancies").respond(
        403, json={"errors": [{"type": "captcha_required", "value": "captcha_required"}]}
    )
    async with Client(create_server(server_settings)) as c:
        with pytest.raises(ToolError) as exc:
            await c.call_tool("search_vacancies", {"query": "python"})
    assert "капч" in str(exc.value)
    assert route.call_count == 1


async def test_missing_token_is_explained(tmp_path, hh_api):
    settings = make_settings(tmp_path, hh_access_token=None)
    async with Client(create_server(settings)) as c:
        with pytest.raises(ToolError) as exc:
            await c.call_tool("get_vacancy", {"vacancy_id": "100001"})
    assert "HH_ACCESS_TOKEN" in str(exc.value)
