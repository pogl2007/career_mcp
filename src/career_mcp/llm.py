"""Минимальный клиент OpenAI-совместимого API (Ollama, apinex, OpenAI и т. п.).

Своя обёртка на httpx вместо SDK: нужен ровно один вызов /chat/completions,
а так тот же respx мокает его в тестах, и лимит частоты общий с остальным кодом.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from career_mcp.ratelimit import TokenBucket

log = logging.getLogger("career_mcp.llm")


class LLMError(Exception):
    pass


def parse_json_object(content: str) -> dict[str, Any]:
    """Достаёт JSON-объект из ответа модели, даже если он обёрнут в ```json ... ```."""
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("LLM вернула ответ без JSON-объекта.")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM вернула некорректный JSON: {exc.msg}.") from None
    if not isinstance(data, dict):
        raise LLMError("LLM вернула JSON, но не объект.")
    return data


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        rate_per_min: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
        limiter: TokenBucket | None = None,
    ) -> None:
        self.model = model
        self.limiter = limiter or TokenBucket(rate_per_min / 60.0, 1)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers, transport=transport
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        await self.limiter.acquire()
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = await self._http.post("/chat/completions", json=payload)
        except httpx.TimeoutException:
            raise LLMError("LLM не ответила вовремя.") from None
        except httpx.TransportError as exc:
            raise LLMError(f"Нет соединения с LLM ({type(exc).__name__}).") from None

        status = response.status_code
        if status == 429:
            raise LLMError("Лимит запросов к LLM исчерпан, попробуйте через минуту.")
        if status == 402:
            raise LLMError("LLM-провайдер требует оплату, подписку или ежедневную отметку.")
        if status in (401, 403):
            raise LLMError("Ключ LLM отклонён провайдером.")
        if status >= 400:
            raise LLMError(f"LLM вернула ошибку {status}.")
        try:
            content = response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("Неожиданный формат ответа LLM.") from None
        log.info("LLM %s ответила за %.1f с", self.model, response.elapsed.total_seconds())
        return parse_json_object(content)
