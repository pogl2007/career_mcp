"""Кеш ответов API с TTL в SQLite.

Почему SQLite через aiosqlite, а не diskcache: сервер асинхронный, и синхронный
diskcache блокировал бы event loop (пришлось бы оборачивать каждый вызов в поток).
Кроме того, в той же базе живёт шорт-лист, а файл можно открыть обычным sqlite3
и посмотреть глазами.

Просроченная запись не удаляется сразу: у неё может быть ETag, и тогда клиент
переспросит hh с If-None-Match и получит дешёвый 304 вместо полного ответа.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    etag       TEXT,
    expires_at REAL NOT NULL,
    stored_at  REAL NOT NULL
)
"""


@dataclass(frozen=True)
class CacheEntry:
    value: Any
    etag: str | None
    expires_at: float

    def is_fresh(self, now: float) -> bool:
        return now < self.expires_at


class Cache:
    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self._path = str(path)
        self._clock = clock
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        if self._path != ":memory:":
            await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Кеш не открыт: вызовите open()")
        return self._db

    async def get(self, key: str) -> CacheEntry | None:
        """Возвращает запись, даже просроченную: свежесть проверяет вызывающий."""
        async with self.db.execute(
            "SELECT value, etag, expires_at FROM http_cache WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return CacheEntry(value=json.loads(row[0]), etag=row[1], expires_at=row[2])

    async def set(self, key: str, value: Any, ttl: float, etag: str | None = None) -> None:
        now = self._clock()
        await self.db.execute(
            "INSERT OR REPLACE INTO http_cache (key, value, etag, expires_at, stored_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), etag, now + ttl, now),
        )
        await self.db.commit()

    async def refresh(self, key: str, ttl: float) -> None:
        """Продлевает срок жизни записи после ответа 304 Not Modified."""
        await self.db.execute(
            "UPDATE http_cache SET expires_at = ? WHERE key = ?", (self._clock() + ttl, key)
        )
        await self.db.commit()

    async def purge_expired(self) -> int:
        cur = await self.db.execute(
            "DELETE FROM http_cache WHERE expires_at < ? AND etag IS NULL", (self._clock(),)
        )
        await self.db.commit()
        return cur.rowcount
