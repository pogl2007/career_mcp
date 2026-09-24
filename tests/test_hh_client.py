"""Клиент hh: кеш, лимиты, повторы, ошибки, капча. Сеть не используется."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from career_mcp.cache import Cache
from career_mcp.hh_client import (
    HHAuthError,
    HHAuthRequired,
    HHBadRequest,
    HHCaptchaRequired,
    HHClient,
    HHConfigError,
    HHNetworkError,
    HHNotFound,
    HHRateLimited,
    HHServerError,
)
from career_mcp.ratelimit import TokenBucket

from .conftest import TOKEN, FakeClock, FakeSleep, load_fixture, make_settings

VACANCY = load_fixture("vacancies.json")["100001"]


async def test_repeat_request_served_from_cache(hh, api):
    route = api.get("/vacancies/100001").respond(200, json=VACANCY)

    first = await hh.get_vacancy("100001")
    second = await hh.get_vacancy("100001")

    assert first == second == VACANCY
    assert route.call_count == 1
    assert hh.metrics.cache_hits == 1
    assert hh.metrics.cache_hit_ratio == 0.5


async def test_headers_contain_user_agent_contact_and_token(hh, api):
    route = api.get("/vacancies/100001").respond(200, json=VACANCY)
    await hh.get_vacancy("100001")

    headers = route.calls.last.request.headers
    assert headers["HH-User-Agent"] == "career-mcp/0.1.0 (dev@example.com)"
    assert headers["User-Agent"] == headers["HH-User-Agent"]
    assert headers["Authorization"] == f"Bearer {TOKEN}"


async def test_429_with_retry_after_waits_and_retries(hh, api, fake_sleep):
    route = api.get("/vacancies/100001").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}, json={"errors": [{"type": "too_many_requests"}]}),
            httpx.Response(200, json=VACANCY),
        ]
    )

    result = await hh.get_vacancy("100001")

    assert result["id"] == "100001"
    assert route.call_count == 2
    assert fake_sleep.delays == [3.0]


async def test_429_gives_up_after_two_retries(hh, api, fake_sleep):
    route = api.get("/vacancies/100001").respond(429)

    with pytest.raises(HHRateLimited) as exc:
        await hh.get_vacancy("100001")

    assert route.call_count == 3  # первая попытка + 2 повтора
    assert fake_sleep.delays == [1.0, 2.0]  # экспоненциальная пауза без Retry-After
    assert "попробуйте" in exc.value.message.lower()


async def test_too_long_retry_after_is_not_awaited(hh, api, fake_sleep):
    route = api.get("/vacancies/100001").respond(429, headers={"Retry-After": "120"})

    with pytest.raises(HHRateLimited) as exc:
        await hh.get_vacancy("100001")

    assert route.call_count == 1
    assert fake_sleep.delays == []
    assert "120" in exc.value.message


async def test_5xx_retried_with_exponential_backoff(hh, api, fake_sleep):
    route = api.get("/vacancies/100001").mock(
        side_effect=[httpx.Response(503), httpx.Response(500, text="<html>oops</html>"), httpx.Response(200, json=VACANCY)]
    )

    await hh.get_vacancy("100001")

    assert route.call_count == 3
    assert fake_sleep.delays == [1.0, 2.0]


async def test_5xx_gives_up(hh, api):
    route = api.get("/vacancies/100001").respond(502)
    with pytest.raises(HHServerError):
        await hh.get_vacancy("100001")
    assert route.call_count == 3


async def test_404_is_not_retried(hh, api, fake_sleep):
    route = api.get("/vacancies/999").respond(404, json={"errors": [{"type": "not_found"}]})

    with pytest.raises(HHNotFound):
        await hh.get_vacancy("999")

    assert route.call_count == 1
    assert fake_sleep.delays == []


async def test_400_bad_argument_is_not_retried(hh, api):
    route = api.get("/vacancies").respond(400, json={"errors": [{"type": "bad_argument", "value": "experience"}]})
    with pytest.raises(HHBadRequest) as exc:
        await hh.search_vacancies({"text": "ml", "experience": "oops"})
    assert route.call_count == 1
    assert "experience" in exc.value.message


async def test_captcha_stops_client_without_retries(hh, api, fake_sleep):
    captcha = {"errors": [{"type": "captcha_required", "value": "captcha_required",
                           "captcha_url": "https://hh.ru/account/captcha?state=x", "fallback_url": None}]}
    route = api.get("/vacancies").respond(403, json=captcha)
    other = api.get("/vacancies/100001").respond(200, json=VACANCY)

    with pytest.raises(HHCaptchaRequired) as exc:
        await hh.search_vacancies({"text": "python"})

    assert route.call_count == 1
    assert fake_sleep.delays == []
    assert "капч" in exc.value.message
    assert "не будет обходить" in exc.value.message

    # После капчи клиент остановлен: следующие запросы не уходят в сеть.
    with pytest.raises(HHCaptchaRequired):
        await hh.get_vacancy("100001")
    assert other.call_count == 0


async def test_oauth_error_does_not_leak_token(hh, api, caplog):
    caplog.set_level(logging.DEBUG)
    api.get("/vacancies/100001").respond(
        403, json={"errors": [{"type": "oauth", "value": "bad_authorization"}]}
    )

    with pytest.raises(HHAuthError) as exc:
        await hh.get_vacancy("100001")

    assert "bad_authorization" in exc.value.message
    assert TOKEN not in exc.value.message
    assert TOKEN not in caplog.text
    assert TOKEN not in repr(hh.settings)


async def test_without_token_vacancies_are_not_requested(tmp_path, api, cache):
    client = HHClient(make_settings(tmp_path, hh_access_token=None), cache, limiter=TokenBucket(1000, 1000))
    search = api.get("/vacancies").respond(200, json=load_fixture("search_ml.json"))
    dicts = api.get("/dictionaries").respond(200, json=load_fixture("dictionaries.json"))
    try:
        with pytest.raises(HHAuthRequired) as exc:
            await client.search_vacancies({"text": "python"})
        assert "HH_ACCESS_TOKEN" in exc.value.message
        assert search.call_count == 0

        # Справочники доступны анонимно.
        data = await client.get_dictionaries()
        assert "currency" in data
        assert dicts.call_count == 1
        assert "Authorization" not in dicts.calls.last.request.headers
    finally:
        await client.aclose()


async def test_missing_contact_email_blocks_network(tmp_path, api, cache):
    client = HHClient(make_settings(tmp_path, hh_contact_email=None), cache)
    route = api.get("/dictionaries").respond(200, json={})
    try:
        with pytest.raises(HHConfigError) as exc:
            await client.get_dictionaries()
        assert "HH_CONTACT_EMAIL" in exc.value.message
        assert route.call_count == 0
    finally:
        await client.aclose()


async def test_bad_user_agent_is_config_error(hh, api):
    api.get("/dictionaries").respond(400, json={"errors": [{"type": "bad_user_agent", "value": "blacklisted"}]})
    with pytest.raises(HHConfigError) as exc:
        await hh.get_dictionaries()
    assert "blacklisted" in exc.value.message


async def test_expired_dictionary_revalidated_with_etag(tmp_path, api):
    clock = FakeClock()
    settings = make_settings(tmp_path)
    cache = Cache(settings.db_path, clock=clock)
    await cache.open()
    client = HHClient(settings, cache, limiter=TokenBucket(1000, 1000), clock=clock)
    dicts = load_fixture("dictionaries.json")
    route = api.get("/dictionaries").mock(
        side_effect=[
            httpx.Response(200, json=dicts, headers={"ETag": 'W/"v1"'}),
            httpx.Response(304, headers={"ETag": 'W/"v1"'}),
        ]
    )
    try:
        await client.get_dictionaries()
        clock.advance(settings.cache_ttl_dict + 1)
        again = await client.get_dictionaries()

        assert again == dicts
        assert route.call_count == 2
        assert route.calls.last.request.headers["If-None-Match"] == 'W/"v1"'
        assert client.metrics.revalidated == 1

        # После 304 запись снова свежая — третий вызов без сети.
        await client.get_dictionaries()
        assert route.call_count == 2
    finally:
        await client.aclose()
        await cache.close()


async def test_timeout_is_not_retried(hh, api):
    route = api.get("/vacancies/100001").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(HHNetworkError) as exc:
        await hh.get_vacancy("100001")
    assert route.call_count == 1
    assert "не ответил" in exc.value.message


async def test_depth_limit_checked_before_network(hh, api):
    route = api.get("/vacancies").respond(200, json={})
    with pytest.raises(HHBadRequest) as exc:
        await hh.search_vacancies({"text": "python", "page": 40, "per_page": 50})
    assert "2000" in exc.value.message
    assert route.call_count == 0


async def test_vacancy_id_is_validated(hh, api):
    with pytest.raises(HHBadRequest):
        await hh.get_vacancy("../../me")


async def test_rate_limit_holds_under_parallel_calls(tmp_path, api):
    """10 параллельных запросов при лимите 2 в секунду: между запросами не меньше 0,5 с."""
    clock = FakeClock()
    sleep = FakeSleep(clock)
    limiter = TokenBucket(rate=2.0, capacity=1, clock=clock, sleep=sleep)
    settings = make_settings(tmp_path)
    cache = Cache(settings.db_path, clock=clock)
    await cache.open()
    client = HHClient(settings, cache, limiter=limiter, sleep=sleep, clock=clock)

    sent_at: list[float] = []

    def record(request):
        sent_at.append(clock())
        return httpx.Response(200, json={"id": request.url.path.rsplit("/", 1)[-1]})

    api.get(url__regex=r"/vacancies/\d+").mock(side_effect=record)
    try:
        await asyncio.gather(*(client.get_vacancy(str(100 + i)) for i in range(10)))
    finally:
        await client.aclose()
        await cache.close()

    assert len(sent_at) == 10
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert min(gaps) >= 0.5 - 1e-9
    assert sent_at[-1] - sent_at[0] == pytest.approx(4.5)
