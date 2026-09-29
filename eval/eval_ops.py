"""Эксплуатация: доля запросов из кеша и латентность с кешем и без.

    python eval/eval_ops.py --query "ML стажёр" --area Москва

Сценарий прогоняется дважды на чистой временной базе: первый раз — холодный кеш
(всё идёт в сеть), второй — тёплый. Меряется время каждого обращения к клиенту hh.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
from pathlib import Path

from _common import load_settings, need_token, write_result

from career_mcp.market import collect_sample
from career_mcp.server import open_services


async def scenario(svc, query: str, area: str, sample: int) -> None:
    area_id, _ = await svc.directory.resolve_area(area)
    data = await svc.hh.search_vacancies({"text": query, "area": area_id, "page": 0, "per_page": 20})
    for item in (data.get("items") or [])[:10]:
        await svc.hh.get_vacancy(str(item["id"]))
    await collect_sample(svc.hh, {"text": query, "area": area_id}, sample)


def ms(values: list[float]) -> str:
    return f"{statistics.mean(values) * 1000:.1f} мс (медиана {statistics.median(values) * 1000:.1f})" if values else "—"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="ML стажёр")
    parser.add_argument("--area", default="Москва")
    parser.add_argument("--sample", type=int, default=30)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        settings = load_settings(db_path=Path(tmp) / "ops.sqlite")
        need_token(settings)
        runs = []
        for label in ("холодный кеш", "тёплый кеш"):
            async with open_services(settings) as svc:
                await scenario(svc, args.query, args.area, args.sample)
                runs.append((label, svc.hh.metrics))

    lines = [
        f"## Эксплуатация: сценарий «{args.query}», {args.area}, выборка {args.sample}\n",
        "| Прогон | Обращений | Из кеша | Доля из кеша | HTTP-запросов | Латентность из сети | Латентность из кеша |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, m in runs:
        lines.append(
            f"| {label} | {m.requests} | {m.cache_hits} | {m.cache_hit_ratio:.0%} | {m.network_calls} | "
            f"{ms(m.latency_network)} | {ms(m.latency_cache)} |"
        )
    total_req = sum(m.requests for _, m in runs)
    total_hits = sum(m.cache_hits for _, m in runs)
    lines.append(f"\nЗа оба прогона из кеша обслужено {total_hits} из {total_req} обращений ({total_hits / total_req:.0%}).")
    out = write_result("ops.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nСохранено: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
