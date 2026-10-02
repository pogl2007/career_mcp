"""Сравнение match_resume с вашей собственной оценкой соответствия (10 вакансий).

    python eval/eval_match.py

Нужен заполненный my_score_0_100 в eval/labels/match_labels.csv. Считаем ранговую
корреляцию Спирмена (правильно ли упорядочены вакансии), среднюю абсолютную ошибку
покрытия в процентах и совпадение категорий «хорошо / частично / слабо».
"""

from __future__ import annotations

import asyncio
import sys

from _common import LABELS, load_settings, need_token, write_result

from career_mcp.evaluation import fmt, read_csv, spearman
from career_mcp.server import build_match, open_services


def bucket(score_0_100: float) -> str:
    return "хорошо" if score_0_100 >= 70 else "частично" if score_0_100 >= 40 else "слабо"


async def main() -> int:
    path = LABELS / "match_labels.csv"
    if not path.exists():
        print("Нет разметки: сначала eval/make_labeling_sheet.py.", file=sys.stderr)
        return 1
    rows = [r for r in read_csv(path) if r.get("my_score_0_100", "").strip()]
    if len(rows) < 3:
        print("Заполните my_score_0_100 хотя бы для 3 вакансий (лучше для всех 10).", file=sys.stderr)
        return 1

    settings = load_settings()
    need_token(settings)
    table = ["| id | вакансия | моя оценка | покрытие match_resume | категории совпали |", "|---|---|---|---|---|"]
    mine, system = [], []
    agree = 0
    async with open_services(settings) as svc:
        resume = svc.resume.load()
        for r in rows:
            v = await svc.hh.get_vacancy(r["vacancy_id"])
            req = await svc.extractor.extract(v)
            m = build_match(req, set(resume.skills), svc.skills, svc.embeddings, set(resume.implied_skills))
            coverage = (m.must_have_coverage or 0.0) * 100
            score = float(r["my_score_0_100"])
            mine.append(score)
            system.append(coverage)
            same = bucket(score) == bucket(coverage)
            agree += same
            table.append(f"| {r['vacancy_id']} | {v.get('name')} | {score:.0f} | {coverage:.0f}% | {'да' if same else 'нет'} |")

    mae = sum(abs(a - b) for a, b in zip(mine, system)) / len(mine)
    rho = spearman(mine, system)
    summary = [
        f"## Сравнение с резюме ({len(rows)} вакансий)\n",
        f"- Корреляция Спирмена с моей оценкой: {fmt(rho)}",
        f"- Средняя абсолютная разница, п.п.: {mae:.1f}",
        f"- Совпадение категорий: {agree} из {len(rows)}",
        "",
        *table,
    ]
    out = write_result("match.md", "\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nСохранено: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
