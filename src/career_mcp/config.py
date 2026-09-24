"""Настройки сервера. Всё берётся из окружения и файла .env в корне проекта.

Секреты (токен hh, ключ LLM) хранятся как SecretStr: при печати настроек
или попадании в лог они видны как '**********'.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from career_mcp import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- hh.ru ---
    hh_access_token: SecretStr | None = None
    hh_contact_email: str | None = None
    hh_app_name: str = "career-mcp"
    hh_base_url: str = "https://api.hh.ru"
    hh_timeout: float = 10.0
    hh_max_retries: int = 2
    hh_backoff_base: float = 1.0
    # Дольше этого не ждём по Retry-After внутри вызова инструмента: лучше честно
    # сказать модели «попробуйте позже», чем держать запрос минутами.
    hh_max_retry_wait: float = 30.0
    hh_rate_per_sec: float = 2.0
    hh_burst: int = 4
    # Брать ли прокси из HTTPS_PROXY/HTTP_PROXY. Если системный VPN не пускает
    # к hh.ru, выставьте false — клиент пойдёт напрямую.
    hh_trust_env: bool = True

    # --- кеш (секунды) ---
    cache_ttl_search: int = 3600
    cache_ttl_vacancy: int = 86400
    cache_ttl_dict: int = 86400
    db_path: Path = PROJECT_ROOT / "data" / "career_mcp.sqlite"

    # --- локальные файлы ---
    resume_path: Path = PROJECT_ROOT / "resume.md"
    synonyms_path: Path = PROJECT_ROOT / "data" / "skills_synonyms.yaml"

    # --- LLM (OpenAI-совместимый API) ---
    llm_enabled: bool = False
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "qwen2.5:7b-instruct"
    llm_timeout: float = 60.0
    llm_rate_per_min: float = 5.0

    # --- эмбеддинги (опционально) ---
    embeddings_enabled: bool = False
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings_threshold: float = 0.8

    # --- транспорт http ---
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    log_level: str = "INFO"

    @field_validator("db_path", "resume_path", "synonyms_path", mode="after")
    @classmethod
    def _relative_to_project(cls, value: Path) -> Path:
        # Claude Desktop запускает сервер из произвольной папки, поэтому
        # относительные пути считаем от корня проекта, а не от текущего каталога.
        if str(value) == ":memory:" or value.is_absolute():
            return value
        return (PROJECT_ROOT / value).resolve()

    @property
    def user_agent(self) -> str:
        contact = self.hh_contact_email or "contact-not-set"
        return f"{self.hh_app_name}/{__version__} ({contact})"
