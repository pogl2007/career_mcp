"""Записывает настоящие ответы API hh в фикстуры и обезличивает их.

    python eval/record_fixtures.py --query "ML стажёр" -n 10

Из вакансий убираются контакты, данные менеджера, точный адрес и брендированное
оформление — в фикстурах остаётся только то, что нужно тестам. Файлы кладутся
в tests/fixtures/recorded/ рядом с синтетическими.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from _common import EVAL_DIR, load_settings, need_token

from career_mcp.server import open_services

OUT = EVAL_DIR.parent / "tests" / "fixtures" / "recorded"
DROP = {"contacts", "manager", "address", "branded_description", "branded_template", "insider_interview",
        "response_url", "apply_alternate_url", "negotiations_url", "suitable_resumes_url", "relations"}


def anonymize(v: dict) -> dict:
    v = {k: val for k, val in v.items() if k not in DROP}
    if isinstance(v.get("employer"), dict):
        v["employer"] = {k: v["employer"].get(k) for k in ("id", "name", "trusted", "alternate_url")}
    return v


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="ML стажёр")
    parser.add_argument("--area", default="Москва")
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()

    settings = load_settings()
    need_token(settings)
    async with open_services(settings) as svc:
        area_id, _ = await svc.directory.resolve_area(args.area)
        search = await svc.hh.search_vacancies({"text": args.query, "area": area_id, "page": 0, "per_page": args.n})
        vacancies = {str(i["id"]): anonymize(await svc.hh.get_vacancy(str(i["id"]))) for i in search["items"]}
        dictionaries = await svc.hh.get_dictionaries()
    search["items"] = [anonymize(i) for i in search["items"]]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in {"search.json": search, "vacancies.json": vacancies, "dictionaries.json": dictionaries}.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано в {OUT}: поиск, {len(vacancies)} вакансий, справочники.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
