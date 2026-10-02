"""Готовит таблицы для ручной разметки: 30 вакансий для извлечения навыков и 10 — для сравнения с резюме.

    python eval/make_labeling_sheet.py --query "ML стажёр" --area Москва

В таблице нет ответов системы — только текст требований и key_skills, чтобы
разметка не подстраивалась под то, что выдаёт сервер.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys

from _common import LABELS, load_settings, need_token

from career_mcp.market import collect_sample, dedup
from career_mcp.server import open_services
from career_mcp.text import html_to_text, split_sections


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="ML стажёр")
    parser.add_argument("--area", default="Москва")
    parser.add_argument("--experience", nargs="*", default=[], help="id опыта: noExperience between1And3 ...")
    parser.add_argument("-n", type=int, default=30)
    parser.add_argument("--match", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="перезаписать уже начатую разметку")
    args = parser.parse_args()

    extraction = LABELS / "extraction_labels.csv"
    match = LABELS / "match_labels.csv"
    if extraction.exists() and not args.force:
        print(f"{extraction} уже есть — чтобы не потерять разметку, запустите с --force.", file=sys.stderr)
        return 1

    settings = load_settings()
    need_token(settings)
    async with open_services(settings) as svc:
        area_id, _ = await svc.directory.resolve_area(args.area)
        params = {"text": args.query, "area": area_id}
        if args.experience:
            params["experience"] = args.experience
        _, details, _ = await collect_sample(svc.hh, params, min(200, args.n * 2))
    unique, _ = dedup(details)
    chosen = unique[: args.n]

    LABELS.mkdir(parents=True, exist_ok=True)
    with extraction.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vacancy_id", "name", "url", "key_skills", "requirements", "gold_must_have", "comment"])
        for v in chosen:
            sections = split_sections(html_to_text(v.get("description")))
            w.writerow([
                v["id"], v.get("name"), v.get("alternate_url"),
                "; ".join(s["name"] for s in v.get("key_skills") or []),
                " | ".join(sections["requirements"])[:800], "", "",
            ])
    with match.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vacancy_id", "name", "url", "my_score_0_100", "comment"])
        for v in chosen[: args.match]:
            w.writerow([v["id"], v.get("name"), v.get("alternate_url"), "", ""])
    print(f"Готово: {extraction.name} ({len(chosen)} вакансий) и {match.name} ({min(args.match, len(chosen))}).")
    print("Заполните gold_must_have (навыки через «;») и my_score_0_100 (насколько вы подходите, 0–100).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
