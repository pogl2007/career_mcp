"""Шорт-лист вакансий в SQLite.

Единственная операция записи в сервере, и она локальная: на hh.ru ничего не
отправляется. Таблица живёт в той же базе, что и кеш.
"""

from __future__ import annotations

from datetime import datetime, timezone

from career_mcp.cache import Cache
from career_mcp.models import SavedVacancy

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shortlist (
    vacancy_id TEXT PRIMARY KEY,
    name       TEXT,
    employer   TEXT,
    url        TEXT,
    note       TEXT NOT NULL DEFAULT '',
    saved_at   TEXT NOT NULL
)
"""


class Shortlist:
    def __init__(self, cache: Cache) -> None:
        self._cache = cache

    async def init(self) -> None:
        await self._cache.db.execute(_SCHEMA)
        await self._cache.db.commit()

    async def save(
        self,
        vacancy_id: str,
        note: str = "",
        *,
        name: str | None = None,
        employer: str | None = None,
        url: str | None = None,
    ) -> SavedVacancy:
        saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        await self._cache.db.execute(
            "INSERT INTO shortlist (vacancy_id, name, employer, url, note, saved_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(vacancy_id) DO UPDATE SET note = excluded.note, saved_at = excluded.saved_at, "
            "name = COALESCE(excluded.name, shortlist.name), employer = COALESCE(excluded.employer, shortlist.employer), "
            "url = COALESCE(excluded.url, shortlist.url)",
            (vacancy_id, name, employer, url, note, saved_at),
        )
        await self._cache.db.commit()
        return (await self.get(vacancy_id))  # type: ignore[return-value]

    async def get(self, vacancy_id: str) -> SavedVacancy | None:
        async with self._cache.db.execute(
            "SELECT vacancy_id, name, employer, url, note, saved_at FROM shortlist WHERE vacancy_id = ?",
            (vacancy_id,),
        ) as cur:
            row = await cur.fetchone()
        return _row(row) if row else None

    async def list(self) -> list[SavedVacancy]:
        async with self._cache.db.execute(
            "SELECT vacancy_id, name, employer, url, note, saved_at FROM shortlist ORDER BY saved_at DESC, vacancy_id"
        ) as cur:
            rows = await cur.fetchall()
        return [_row(r) for r in rows]


def _row(row) -> SavedVacancy:
    return SavedVacancy(
        vacancy_id=row[0], name=row[1], employer=row[2], url=row[3], note=row[4] or "", saved_at=row[5]
    )
