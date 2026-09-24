from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import respx

from career_mcp.cache import Cache
from career_mcp.config import Settings
from career_mcp.hh_client import HHClient
from career_mcp.ratelimit import TokenBucket

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token-SECRET-123"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Тесты не должны зависеть от переменных окружения разработчика.
    for key in list(os.environ):
        if key.upper().startswith(("HH_", "LLM_", "EMBEDDINGS_", "CACHE_TTL_", "RESUME_PATH", "DB_PATH")):
            monkeypatch.delenv(key, raising=False)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    """Не спит, а запоминает задержки. Может двигать поддельные часы."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.delays: list[float] = []
        self.clock = clock

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if self.clock is not None:
            self.clock.advance(delay)


RESUME_TEXT = """# Тестовый Кандидат

## Проекты
- Сервис классификации текстов. Стек: Python, PyTorch, FastAPI, Docker, PostgreSQL.

## Ключевые навыки
Python, pandas, scikit-learn, SQL, Git, LangChain, Hugging Face Transformers
"""


def make_settings(tmp_path: Path, **overrides) -> Settings:
    resume = tmp_path / "resume.md"
    if not resume.exists():
        resume.write_text(RESUME_TEXT, encoding="utf-8")
    values = {
        "hh_access_token": TOKEN,
        "hh_contact_email": "dev@example.com",
        "db_path": tmp_path / "cache.sqlite",
        "resume_path": resume,
        "llm_enabled": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def fake_sleep(clock):
    return FakeSleep(clock)


@pytest.fixture
async def cache(settings, clock):
    c = Cache(settings.db_path, clock=clock)
    await c.open()
    yield c
    await c.close()


@pytest.fixture
async def hh(settings, cache, fake_sleep, clock):
    client = HHClient(
        settings, cache, limiter=TokenBucket(1000, 1000), sleep=fake_sleep, clock=clock
    )
    yield client
    await client.aclose()


@pytest.fixture
def api():
    with respx.mock(base_url="https://api.hh.ru", assert_all_called=False) as router:
        yield router
