# Оценка качества

| Что меряем | Скрипт | Что нужно | Статус |
|---|---|---|---|
| Выбор LLM | `model_probe/round*.py` | ключ LLM | **измерено**, см. [model_selection.md](model_selection.md) |
| Извлечение навыков: precision и recall, «словарь + key_skills» против «+ LLM» | `eval_extraction.py` | токен hh + разметка 30 вакансий | ждёт токен |
| Сравнение с резюме против вашей оценки на 10 вакансиях | `eval_match.py` | токен hh + разметка | ждёт токен |
| Доля запросов из кеша, латентность с кешем и без | `eval_ops.py` | токен hh | ждёт токен |

Пока токена нет, скрипты останавливаются с кодом 2 и ничего не пишут в `results/`. В README проекта попадают только цифры из `results/`.

## Порядок, когда появится токен

```bash
.venv/Scripts/python eval/make_labeling_sheet.py --query "ML стажёр" --area Москва
```

1. Скрипт создаст `labels/extraction_labels.csv` (30 вакансий) и `labels/match_labels.csv` (10).
2. В первой таблице заполните `gold_must_have`: какие навыки реально обязательны, через «;». Ответов системы в таблице нет, чтобы не подсказывать.
3. Во второй таблице заполните `my_score_0_100`: насколько вы подходите на вакансию.
4. Запустите три оценки:

```bash
.venv/Scripts/python eval/eval_extraction.py
```

```bash
.venv/Scripts/python eval/eval_match.py
```

```bash
.venv/Scripts/python eval/eval_ops.py
```

Настоящие обезличенные ответы для тестов: `eval/record_fixtures.py`. Формулы метрик — в `career_mcp/evaluation.py`, покрыты тестами в `tests/test_evaluation.py`.
