"""Скрипты eval/ целиком на подставном API: чтобы они работали, когда появится токен.

Цифры отсюда ничего не значат (фикстуры синтетические) и в README не попадают —
проверяется только, что конвейер «лист разметки → разметка → метрики» не сломан.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from career_mcp.config import PROJECT_ROOT

from .conftest import load_fixture, make_settings

EVAL = PROJECT_ROOT / "eval"


@pytest.fixture
def eval_env(tmp_path, monkeypatch, api):
    monkeypatch.syspath_prepend(str(EVAL))
    for name in ["_common", "make_labeling_sheet", "eval_extraction", "eval_match", "eval_ops"]:
        sys.modules.pop(name, None)
    import _common  # noqa: PLC0415

    base = make_settings(tmp_path, hh_rate_per_sec=1000, hh_burst=1000)
    monkeypatch.setattr(_common, "load_settings", lambda **kw: base.model_copy(update=kw))
    monkeypatch.setattr(_common, "RESULTS", tmp_path / "results")

    api.get("/areas").respond(200, json=load_fixture("areas.json"))
    api.get("/dictionaries").respond(200, json=load_fixture("dictionaries.json"))
    api.get("/vacancies").respond(200, json=load_fixture("search_ml.json"))
    for vid, body in load_fixture("vacancies.json").items():
        api.get(f"/vacancies/{vid}").respond(200, json=body)
    return tmp_path


def _fill(path: Path, column: str, values: list[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row, value in zip(rows, values):
        row[column] = value
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


async def test_eval_pipeline(eval_env, monkeypatch):
    import eval_extraction  # noqa: PLC0415
    import eval_match  # noqa: PLC0415
    import eval_ops  # noqa: PLC0415
    import make_labeling_sheet  # noqa: PLC0415

    labels = eval_env / "labels"
    for module in (make_labeling_sheet, eval_extraction, eval_match):
        monkeypatch.setattr(module, "LABELS", labels)

    monkeypatch.setattr(sys, "argv", ["make_labeling_sheet.py", "-n", "5", "--match", "3"])
    assert await make_labeling_sheet.main() == 0
    sheet = labels / "extraction_labels.csv"
    with sheet.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 and rows[0]["gold_must_have"] == ""

    _fill(sheet, "gold_must_have", ["Python; PyTorch; SQL", "Python; Pandas", "Python; LLM", "Excel", "Python"])
    _fill(labels / "match_labels.csv", "my_score_0_100", ["80", "50", "30"])

    assert await eval_extraction.main() == 0
    assert "словарь + key_skills" in (eval_env / "results" / "extraction.md").read_text(encoding="utf-8")

    assert await eval_match.main() == 0
    assert "Спирмена" in (eval_env / "results" / "match.md").read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["eval_ops.py", "--sample", "10"])
    assert await eval_ops.main() == 0
    ops = (eval_env / "results" / "ops.md").read_text(encoding="utf-8")
    assert "| тёплый кеш |" in ops and "100%" in ops
