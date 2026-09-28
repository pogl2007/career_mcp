"""Оба транспорта в отдельном процессе: stdio (Claude Desktop) и http /mcp (агенты).

stdio-тест заодно проверяет, что в stdout не попадает ничего, кроме протокола:
лишняя строка лога или баннер сломали бы JSON-RPC и клиент бы не подключился.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from career_mcp.server import configure_logging

from .conftest import RESUME_TEXT


def _server_env(tmp_path) -> dict[str, str]:
    resume = tmp_path / "resume.md"
    resume.write_text(RESUME_TEXT, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("HH_", "LLM_"))}
    env.update(
        {
            "HH_CONTACT_EMAIL": "dev@example.com",
            "HH_ACCESS_TOKEN": "",
            "DB_PATH": str(tmp_path / "db.sqlite"),
            "RESUME_PATH": str(resume),
            "LLM_ENABLED": "false",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return env


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_stdio_transport(tmp_path):
    transport = StdioTransport(
        command=sys.executable, args=["-m", "career_mcp"], env=_server_env(tmp_path), cwd=str(tmp_path)
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        assert len(tools) == 8
        text = (await client.read_resource("resume://current"))[0].text
        assert text.startswith("# Тестовый Кандидат")


async def test_http_transport(tmp_path):
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "career_mcp", "--transport", "http", "--port", str(port)],
        env=_server_env(tmp_path), cwd=str(tmp_path), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                await asyncio.sleep(0.2)
        else:
            pytest.fail("http-сервер не поднялся")
        async with Client(f"http://127.0.0.1:{port}/mcp") as client:
            names = {t.name for t in await client.list_tools()}
            assert "market_snapshot" in names
            saved = await client.call_tool("list_saved", {})
            assert saved.structured_content == {"result": []}
    finally:
        proc.terminate()
        _, err = proc.communicate(timeout=15)
    assert "career-mcp: транспорт http" in err.decode("utf-8", "replace")  # лог ушёл в stderr


def test_logging_goes_to_stderr():
    configure_logging("INFO")
    handlers = logging.getLogger("career_mcp").handlers
    assert len(handlers) == 1
    assert handlers[0].stream is sys.stderr
