"""Справочники hh: регионы, опыт, формат работы, валюты.

Модель передаёт город названием («Москва», «Питер» не поймём — только как в hh),
опыт и формат работы — id из справочника или человеческими словами. Здесь всё это
переводится в id, которые понимает API.
"""

from __future__ import annotations

import difflib
from typing import Any

from career_mcp.hh_client import HHBadRequest, HHClient
from career_mcp.text import norm

RUSSIA_ID = "113"

_EXPERIENCE_ALIASES = {
    "noexperience": "noExperience", "нет опыта": "noExperience", "без опыта": "noExperience",
    "стажер": "noExperience", "0": "noExperience",
    "between1and3": "between1And3", "от 1 до 3": "between1And3", "1-3": "between1And3",
    "от 1 года до 3 лет": "between1And3",
    "between3and6": "between3And6", "от 3 до 6": "between3And6", "3-6": "between3And6",
    "morethan6": "moreThan6", "более 6": "moreThan6", "6+": "moreThan6", "больше 6": "moreThan6",
}

_WORK_FORMAT_ALIASES = {
    "remote": "REMOTE", "удаленно": "REMOTE", "удаленка": "REMOTE", "удаленная": "REMOTE",
    "из дома": "REMOTE",
    "on_site": "ON_SITE", "офис": "ON_SITE", "в офисе": "ON_SITE", "на месте работодателя": "ON_SITE",
    "hybrid": "HYBRID", "гибрид": "HYBRID", "гибридный": "HYBRID",
    "field_work": "FIELD_WORK", "разъездная": "FIELD_WORK",
}


def _clean_area_name(value: str) -> str:
    v = norm(value)
    for prefix in ("г. ", "город "):
        if v.startswith(prefix):
            v = v[len(prefix):]
    return v


def flatten_areas(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]], root: str | None, parent_name: str | None) -> None:
        for node in nodes:
            node_root = root or node["id"]
            out.append({
                "id": str(node["id"]),
                "name": node["name"],
                "parent_id": node.get("parent_id"),
                "parent_name": parent_name,
                "country_id": node_root,
            })
            walk(node.get("areas") or [], node_root, node["name"])

    walk(tree, None, None)
    return out


class Directory:
    def __init__(self, hh: HHClient) -> None:
        self.hh = hh

    async def areas_flat(self) -> list[dict[str, Any]]:
        return flatten_areas(await self.hh.get_areas())

    async def resolve_area(self, value: str) -> tuple[str, str]:
        """Название или id региона → (id, название). Предпочитаем регионы России."""
        value = (value or "").strip()
        if not value:
            raise HHBadRequest("Укажите город или регион, например «Москва».")
        areas = await self.areas_flat()
        if value.isdigit():
            for a in areas:
                if a["id"] == value:
                    return a["id"], a["name"]
            raise HHBadRequest(f"Региона с id {value} нет в справочнике hh.")

        wanted = _clean_area_name(value)
        exact = [a for a in areas if _clean_area_name(a["name"]) == wanted]
        if exact:
            exact.sort(key=lambda a: a["country_id"] != RUSSIA_ID)
            best = exact[0]
            name = best["name"]
            if len(exact) > 1 and best["parent_name"]:
                name = f"{name} ({best['parent_name']})"
            return best["id"], name

        names = sorted({a["name"] for a in areas})
        close = difflib.get_close_matches(value, names, n=5, cutoff=0.6)
        hint = f" Возможно, вы имели в виду: {', '.join(close)}." if close else ""
        raise HHBadRequest(f"Регион «{value}» не найден в справочнике hh.{hint}")

    async def dictionary(self, name: str) -> list[dict[str, Any]]:
        data = await self.hh.get_dictionaries()
        return list(data.get(name) or [])

    async def resolve_option(self, dict_name: str, value: str, aliases: dict[str, str]) -> str:
        options = await self.dictionary(dict_name)
        ids = {o["id"] for o in options}
        v = value.strip()
        if v in ids:
            return v
        key = norm(v)
        by_name = {norm(o["name"]): o["id"] for o in options}
        found = aliases.get(key) or by_name.get(key)
        if found is None:
            found = next((o["id"] for o in options if o["id"].lower() == key), None)
        if found is None or (ids and found not in ids):
            variants = ", ".join(f"{o['id']} ({o['name']})" for o in options)
            raise HHBadRequest(f"Неизвестное значение «{value}» для {dict_name}. Допустимые: {variants}.")
        return found

    async def resolve_experience(self, value: str) -> str:
        return await self.resolve_option("experience", value, _EXPERIENCE_ALIASES)

    async def resolve_work_format(self, value: str) -> str:
        return await self.resolve_option("work_format", value, _WORK_FORMAT_ALIASES)

    async def currency_rates(self) -> dict[str, float]:
        return {c["code"]: float(c["rate"]) for c in await self.dictionary("currency") if c.get("rate")}
