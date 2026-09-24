"""Ограничение частоты запросов: token bucket.

Корзина вмещает `capacity` токенов и пополняется со скоростью `rate` токенов
в секунду. Каждый запрос забирает один токен; если токенов нет — ждёт.
Блокировка держится и во время ожидания, поэтому параллельные вызовы
выстраиваются в очередь и суммарная частота не превышает лимит.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class TokenBucket:
    def __init__(
        self,
        rate: float,
        capacity: int = 1,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate должен быть больше нуля")
        self.rate = rate
        self.capacity = max(1, capacity)
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(self.capacity)
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Забирает токен, при необходимости ожидая. Возвращает время ожидания в секундах."""
        async with self._lock:
            waited = 0.0
            while True:
                now = self._clock()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return waited
                delay = (1 - self._tokens) / self.rate
                waited += delay
                await self._sleep(delay)
