"""Клиент официального API hh.ru.

Что здесь сделано и почему:
- заголовки User-Agent и HH-User-Agent: «приложение/версия (контакт)», контакт из окружения;
- токен приложения только из окружения, в логи и тексты ошибок не попадает;
- token bucket перед каждым сетевым запросом, в том числе перед повторами;
- кеш с TTL; просроченные справочники переспрашиваются с If-None-Match (ETag → 304);
- повторы только на 429 и 5xx, не больше `hh_max_retries`, с экспоненциальной паузой
  или по Retry-After; остальные 4xx не повторяем;
- капча и 403 останавливают клиент до перезапуска: без повторов и без попыток обхода;
- все ошибки превращаются в HHError с понятным текстом для модели.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any
from urllib.parse import urlencode

import httpx

from career_mcp.cache import Cache
from career_mcp.config import Settings
from career_mcp.ratelimit import TokenBucket

log = logging.getLogger("career_mcp.hh")

MAX_DEPTH = 2000  # hh отдаёт не больше 2000 результатов поиска


# ---------------------------------------------------------------- ошибки


class HHError(Exception):
    """Ошибка работы с hh.ru. `message` — текст для модели, без секретов и стек-трейсов."""

    def __init__(self, message: str, *, status: int | None = None, error_type: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type


class HHConfigError(HHError):
    """Не хватает настроек (контактная почта)."""


class HHAuthRequired(HHConfigError):
    """Метод требует токен приложения, а его нет."""


class HHAuthError(HHError):
    """Токен есть, но hh его не принял (403 oauth)."""


class HHCaptchaRequired(HHError):
    """hh требует капчу. Не обходим — останавливаемся."""


class HHForbidden(HHError):
    """403 без подробностей."""


class HHRateLimited(HHError):
    """429 после всех повторов или слишком долгий Retry-After."""


class HHNotFound(HHError):
    pass


class HHBadRequest(HHError):
    pass


class HHServerError(HHError):
    pass


class HHNetworkError(HHError):
    pass


def _auth_required_error() -> HHAuthRequired:
    return HHAuthRequired(
        "Поиск и просмотр вакансий в API hh.ru требуют токен приложения. "
        "Добавьте HH_ACCESS_TOKEN в .env (токен виден в dev.hh.ru/admin после одобрения "
        "заявки на приложение) и перезапустите сервер."
    )


# ---------------------------------------------------------------- метрики


@dataclass
class ClientMetrics:
    requests: int = 0  # обращений к get_json
    cache_hits: int = 0  # ответ целиком из кеша, без сети
    revalidated: int = 0  # сеть ответила 304, тело взяли из кеша
    network_calls: int = 0  # фактических HTTP-запросов, включая повторы
    retries: int = 0
    latency_cache: list[float] = field(default_factory=list)
    latency_network: list[float] = field(default_factory=list)

    @property
    def cache_hit_ratio(self) -> float | None:
        return self.cache_hits / self.requests if self.requests else None


# ---------------------------------------------------------------- клиент


def check_depth(page: int, per_page: int) -> None:
    if page < 0 or per_page < 1:
        raise HHBadRequest("page должен быть ≥ 0, per_page ≥ 1.")
    if (page + 1) * per_page > MAX_DEPTH:
        raise HHBadRequest(
            f"hh.ru отдаёт не больше {MAX_DEPTH} результатов поиска: page={page} при "
            f"per_page={per_page} выходит за предел. Уточните запрос фильтрами."
        )


def _normalize_params(params: dict[str, Any] | None) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key, value in (params or {}).items():
        if value is None:
            continue
        values = value if isinstance(value, list | tuple | set) else [value]
        for v in values:
            if isinstance(v, bool):
                v = "true" if v else "false"
            items.append((key, str(v)))
    return sorted(items)


def _parse_retry_after(value: str | None, now: float) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, dt.timestamp() - now)


def _error_items(response: httpx.Response) -> list[dict[str, Any]]:
    try:
        body = response.json()
    except ValueError:
        return []
    if isinstance(body, dict) and isinstance(body.get("errors"), list):
        return [e for e in body["errors"] if isinstance(e, dict)]
    return []


class HHClient:
    def __init__(
        self,
        settings: Settings,
        cache: Cache,
        *,
        limiter: TokenBucket | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.limiter = limiter or TokenBucket(settings.hh_rate_per_sec, settings.hh_burst)
        self.metrics = ClientMetrics()
        self._sleep = sleep
        self._clock = clock
        self._halted: HHError | None = None
        self._http = httpx.AsyncClient(
            base_url=settings.hh_base_url,
            timeout=settings.hh_timeout,
            transport=transport,
            trust_env=settings.hh_trust_env,
            headers={
                # По документации hh основной заголовок — User-Agent; HH-User-Agent
                # для клиентов, которые не дают его задать. Шлём оба с одним значением.
                "User-Agent": settings.user_agent,
                "HH-User-Agent": settings.user_agent,
                "Accept": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def has_token(self) -> bool:
        return self.settings.hh_access_token is not None and bool(
            self.settings.hh_access_token.get_secret_value()
        )

    def ensure_token(self) -> None:
        """Инструментам, которым нужны вакансии, лучше сразу сказать про токен, чем сначала
        ходить за справочниками и падать на чём-то другом."""
        if not self.has_token:
            raise _auth_required_error()

    @property
    def halted_reason(self) -> str | None:
        return self._halted.message if self._halted else None

    # ------------------------------------------------------------ публичные методы

    async def search_vacancies(self, params: dict[str, Any]) -> dict[str, Any]:
        check_depth(int(params.get("page", 0)), int(params.get("per_page", 20)))
        return await self.get_json(
            "/vacancies", params, ttl=self.settings.cache_ttl_search, auth=True
        )

    async def get_vacancy(self, vacancy_id: str) -> dict[str, Any]:
        vacancy_id = validate_vacancy_id(vacancy_id)
        return await self.get_json(
            f"/vacancies/{vacancy_id}", ttl=self.settings.cache_ttl_vacancy, auth=True
        )

    async def cached_vacancy(self, vacancy_id: str) -> dict[str, Any] | None:
        """Вакансия только из кеша, без сети (даже просроченная)."""
        entry = await self.cache.get(_cache_key(f"/vacancies/{validate_vacancy_id(vacancy_id)}", []))
        return entry.value if entry else None

    async def get_areas(self) -> list[dict[str, Any]]:
        return await self.get_json("/areas", ttl=self.settings.cache_ttl_dict, auth=False)

    async def get_dictionaries(self) -> dict[str, Any]:
        return await self.get_json("/dictionaries", ttl=self.settings.cache_ttl_dict, auth=False)

    # ------------------------------------------------------------ ядро

    async def get_json(
        self, path: str, params: dict[str, Any] | None = None, *, ttl: float, auth: bool
    ) -> Any:
        started = time.perf_counter()
        self.metrics.requests += 1
        query = _normalize_params(params)
        key = _cache_key(path, query)

        entry = await self.cache.get(key)
        if entry is not None and entry.is_fresh(self._clock()):
            self.metrics.cache_hits += 1
            self.metrics.latency_cache.append(time.perf_counter() - started)
            log.debug("cache hit %s", key)
            return entry.value

        if self._halted is not None:
            raise type(self._halted)(
                self._halted.message, status=self._halted.status, error_type=self._halted.error_type
            )
        if auth and not self.has_token:
            raise _auth_required_error()
        if not self.settings.hh_contact_email:
            raise HHConfigError(
                "Не задан HH_CONTACT_EMAIL: hh.ru требует контактную почту в заголовке "
                "User-Agent. Добавьте её в .env и перезапустите сервер."
            )

        headers: dict[str, str] = {}
        if self.has_token:
            headers["Authorization"] = f"Bearer {self.settings.hh_access_token.get_secret_value()}"
        if entry is not None and entry.etag:
            headers["If-None-Match"] = entry.etag

        response = await self._send_with_retries(path, query, headers)

        if response.status_code == 304 and entry is not None:
            self.metrics.revalidated += 1
            await self.cache.refresh(key, ttl)
            self.metrics.latency_network.append(time.perf_counter() - started)
            return entry.value

        try:
            data = response.json()
        except ValueError:
            raise HHServerError(
                "hh.ru вернул ответ не в формате JSON. Попробуйте позже.", status=response.status_code
            ) from None
        await self.cache.set(key, data, ttl, etag=response.headers.get("ETag"))
        self.metrics.latency_network.append(time.perf_counter() - started)
        return data

    async def _send_with_retries(
        self, path: str, query: list[tuple[str, str]], headers: dict[str, str]
    ) -> httpx.Response:
        max_retries = self.settings.hh_max_retries
        attempt = 0
        while True:
            await self.limiter.acquire()
            self.metrics.network_calls += 1
            t0 = time.perf_counter()
            try:
                response = await self._http.get(path, params=query, headers=headers)
            except httpx.TimeoutException:
                raise HHNetworkError(
                    f"api.hh.ru не ответил за {self.settings.hh_timeout:g} с. Попробуйте позже."
                ) from None
            except httpx.TransportError as exc:
                raise HHNetworkError(
                    f"Не удалось соединиться с api.hh.ru ({type(exc).__name__}). Проверьте сеть; "
                    "если мешает системный прокси или VPN — выставьте HH_TRUST_ENV=false."
                ) from None
            log.info(
                "GET %s -> %s за %.2f с (попытка %d)",
                path, response.status_code, time.perf_counter() - t0, attempt + 1,
            )

            status = response.status_code
            if status < 400:
                return response

            if status == 429 or status >= 500:
                if attempt >= max_retries:
                    raise self._final_retry_error(response)
                delay = self._retry_delay(response, attempt)
                if delay is None:
                    raise self._final_retry_error(response)
                attempt += 1
                self.metrics.retries += 1
                log.warning("hh ответил %s, повтор %d через %.1f с", status, attempt, delay)
                await self._sleep(delay)
                continue

            raise self._client_error(response)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float | None:
        delay = self.settings.hh_backoff_base * (2**attempt)
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"), self._clock())
            if retry_after is not None:
                if retry_after > self.settings.hh_max_retry_wait:
                    return None  # слишком долго ждать внутри вызова инструмента
                delay = max(delay, retry_after)
        return delay

    def _final_retry_error(self, response: httpx.Response) -> HHError:
        status = response.status_code
        if status == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"), self._clock())
            when = f"через {int(retry_after)} с" if retry_after else "через минуту"
            return HHRateLimited(
                f"hh.ru ограничил частоту запросов (429). Попробуйте {when}.", status=429
            )
        return HHServerError(
            f"hh.ru временно недоступен (ошибка {status}), повторы не помогли. Попробуйте позже.",
            status=status,
        )

    def _client_error(self, response: httpx.Response) -> HHError:
        status = response.status_code
        errors = _error_items(response)
        types = {e.get("type") for e in errors}
        value = next((str(e["value"]) for e in errors if e.get("value")), None)

        captcha = next((e for e in errors if e.get("type") == "captcha_required"), None)
        if captcha is not None:
            hint = ""
            if captcha.get("captcha_url"):
                hint = f" Пройти капчу вручную можно на странице: {captcha['captcha_url']}"
            err: HHError = HHCaptchaRequired(
                "hh.ru требует пройти капчу. Сервер остановил запросы к API до перезапуска и не "
                "будет обходить проверку. Обычно причина — запрос без токена приложения или "
                "слишком частые запросы." + hint,
                status=status,
                error_type="captcha_required",
            )
            self._halt(err)
            return err

        if status == 403:
            if "oauth" in types:
                err = HHAuthError(
                    f"hh.ru не принял токен приложения ({value or 'oauth'}). Проверьте "
                    "HH_ACCESS_TOKEN в .env: актуальный токен виден в dev.hh.ru/admin. "
                    "Запросы к API остановлены до перезапуска.",
                    status=403,
                    error_type=value or "oauth",
                )
            else:
                err = HHForbidden(
                    "hh.ru отказал в доступе (403). Запросы к API остановлены до перезапуска "
                    "сервера; обходить ограничение сервер не будет.",
                    status=403,
                    error_type=next(iter(types), None) if types else None,
                )
            self._halt(err)
            return err

        if status == 404:
            return HHNotFound(
                "Не найдено на hh.ru (404): вакансия удалена, скрыта или id неверный.", status=404
            )
        if status == 400 and "bad_user_agent" in types:
            return HHConfigError(
                f"hh.ru отклонил заголовок User-Agent ({value or 'bad_user_agent'}). Проверьте "
                "HH_APP_NAME и HH_CONTACT_EMAIL в .env.",
                status=400,
                error_type="bad_user_agent",
            )
        if status == 400:
            what = f": {value}" if value else ""
            return HHBadRequest(
                f"hh.ru отклонил параметры запроса{what}. Проверьте значения фильтров.",
                status=400,
                error_type=next(iter(types), None) if types else None,
            )
        return HHError(f"hh.ru вернул ошибку {status}.", status=status)

    def _halt(self, err: HHError) -> None:
        self._halted = err
        log.error("Запросы к hh остановлены: %s", err.error_type or err.status)


async def request_app_token(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> str:
    """Разово получает токен приложения (grant_type=client_credentials).

    Сервер сам этого не делает: каждый новый запрос отзывает прежний токен,
    а hh выдаёт их не чаще раза в 5 минут. Поэтому это отдельная ручная команда.
    """
    if not settings.hh_client_id or not settings.hh_client_secret:
        raise HHConfigError("Нужны HH_CLIENT_ID и HH_CLIENT_SECRET в .env (dev.hh.ru/admin → ваше приложение).")
    if not settings.hh_contact_email:
        raise HHConfigError("Не задан HH_CONTACT_EMAIL: hh требует контакт в User-Agent.")
    async with httpx.AsyncClient(
        base_url=settings.hh_base_url,
        timeout=settings.hh_timeout,
        transport=transport,
        trust_env=settings.hh_trust_env,
        headers={"User-Agent": settings.user_agent, "HH-User-Agent": settings.user_agent},
    ) as http:
        try:
            response = await http.post(
                "/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": settings.hh_client_id,
                    "client_secret": settings.hh_client_secret.get_secret_value(),
                },
            )
        except httpx.HTTPError as exc:
            raise HHNetworkError(f"Не удалось соединиться с api.hh.ru ({type(exc).__name__}).") from None

    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code == 200 and body.get("access_token"):
        return str(body["access_token"])
    error = str(body.get("error") or response.status_code)
    description = str(body.get("error_description") or "")
    if response.status_code == 403 and "too early" in description:
        raise HHRateLimited("hh выдаёт токен приложения не чаще раза в 5 минут — подождите и повторите.", status=403)
    raise HHAuthError(
        f"hh не выдал токен приложения: {error} {description}".strip()
        + ". Проверьте HH_CLIENT_ID и HH_CLIENT_SECRET.",
        status=response.status_code,
        error_type=error,
    )


def _cache_key(path: str, query: list[tuple[str, str]]) -> str:
    return f"GET {path}?{urlencode(query)}" if query else f"GET {path}"


def validate_vacancy_id(vacancy_id: str | int) -> str:
    value = str(vacancy_id).strip()
    if not value.isdigit() or len(value) > 12:
        raise HHBadRequest(f"Некорректный id вакансии: {value[:20]!r}. Нужны только цифры.")
    return value
