"""Общее для скриптов оценки: настройки из .env, пути к разметке и результатам."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from career_mcp.config import Settings
from career_mcp.server import configure_logging

EVAL_DIR = Path(__file__).resolve().parent
LABELS = EVAL_DIR / "labels"
RESULTS = EVAL_DIR / "results"


def load_settings(**overrides) -> Settings:
    configure_logging("WARNING")
    return Settings(**overrides)


def need_token(settings: Settings) -> None:
    if not settings.hh_access_token or not settings.hh_access_token.get_secret_value():
        print(
            "Нужен HH_ACCESS_TOKEN в .env: без токена приложения API hh.ru не отдаёт вакансии. "
            "Скрипт остановлен, цифры не посчитаны.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def write_result(name: str, text: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(f"<!-- посчитано {date.today().isoformat()} скриптом eval/ -->\n{text}", encoding="utf-8")
    return path
