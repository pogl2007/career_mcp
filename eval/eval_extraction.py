"""Precision и recall извлечения обязательных навыков: «словарь + key_skills» против «словарь + LLM».

    python eval/eval_extraction.py

Нужны: HH_ACCESS_TOKEN (или вакансии уже в кеше) и заполненный столбец gold_must_have
в eval/labels/extraction_labels.csv. Навыки сравниваются после нормализации синонимов.
"""

from __future__ import annotations

import asyncio
import sys

from _common import LABELS, load_settings, need_token, write_result

from career_mcp.evaluation import PRF, compare, fmt, macro, read_csv, split_labels
from career_mcp.server import open_services


async def main() -> int:
    path = LABELS / "extraction_labels.csv"
    if not path.exists():
        print("Нет разметки: сначала eval/make_labeling_sheet.py и ручное заполнение.", file=sys.stderr)
        return 1
    rows = [r for r in read_csv(path) if r.get("gold_must_have", "").strip()]
    if not rows:
        print("В extraction_labels.csv не заполнен ни один gold_must_have — считать нечего.", file=sys.stderr)
        return 1

    settings = load_settings()
    need_token(settings)
    rules_all, llm_all = PRF(), PRF()
    rules_rows: list[PRF] = []
    llm_rows: list[PRF] = []
    llm_calls = 0
    async with open_services(settings) as svc:
        for r in rows:
            v = await svc.hh.get_vacancy(r["vacancy_id"])
            gold = split_labels(r["gold_must_have"])
            rules = await svc.extractor.extract(v, use_llm=False)
            prf = compare(gold, rules.must_have, svc.skills)
            rules_all.add(prf)
            rules_rows.append(prf)
            if svc.llm is not None:
                with_llm = await svc.extractor.extract(v)
                llm_calls += int(with_llm.llm_used)
                prf = compare(gold, with_llm.must_have, svc.skills)
                llm_all.add(prf)
                llm_rows.append(prf)

    lines = [
        f"## Извлечение обязательных навыков ({len(rows)} размеченных вакансий)\n",
        "| Подход | Precision (micro) | Recall (micro) | F1 (micro) | Precision (macro) | Recall (macro) |",
        "|---|---|---|---|---|---|",
    ]
    p, rec = macro(rules_rows)
    lines.append(f"| словарь + key_skills | {fmt(rules_all.precision)} | {fmt(rules_all.recall)} | "
                 f"{fmt(rules_all.f1)} | {fmt(p)} | {fmt(rec)} |")
    if llm_rows:
        p, rec = macro(llm_rows)
        lines.append(f"| словарь + key_skills + LLM ({settings.llm_model}) | {fmt(llm_all.precision)} | "
                     f"{fmt(llm_all.recall)} | {fmt(llm_all.f1)} | {fmt(p)} | {fmt(rec)} |")
        lines.append(f"\nLLM понадобилась в {llm_calls} из {len(rows)} вакансий (в остальных правила извлекли всё).")
    else:
        lines.append("\nLLM выключена (LLM_ENABLED=false) — второй подход не считался.")
    out = write_result("extraction.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nСохранено: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
