"""Работа с текстом вакансий: HTML → текст, разделы описания, подозрительные строки.

Описание вакансии пишет работодатель, поэтому это недоверенные данные. Здесь же
ищем строки, похожие на инструкции для ИИ: сами по себе они ничего не ломают
(мы возвращаем их как данные), но модели полезно видеть предупреждение, а в LLM
такие строки не отправляются.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p", "div", "br", "ul", "ol", "table", "tr", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
}


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _BLOCK_TAGS or tag == "li":
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    parser = _HTMLToText()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
    out: list[str] = []
    for line in lines:
        if line in {"", "-"}:
            continue
        out.append(line)
    return "\n".join(out)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


# ---------------------------------------------------------------- разделы описания

_HEADINGS: list[tuple[str, tuple[str, ...]]] = [
    # «плюсы» проверяем раньше требований: «желательно» не должно стать обязательным
    ("nice_to_have", (
        "будет плюсом", "будет преимуществом", "большим плюсом", "дополнительным плюсом",
        "будет большим плюсом", "желательно", "приветствуется", "nice to have", "plus",
        "преимуществом будет", "плюсом будет",
    )),
    ("requirements", (
        "требования", "мы ждем", "мы ожидаем", "ожидаем", "что мы ждем", "что нужно",
        "от вас", "нам важно", "кого мы ищем", "наши ожидания", "необходимые навыки",
        "ключевые требования", "requirements", "you have", "must have", "что важно",
        "мы ищем",
    )),
    ("responsibilities", (
        "обязанности", "задачи", "чем предстоит заниматься", "чем заниматься",
        "что предстоит делать", "что нужно делать", "ваши задачи", "функционал",
        "responsibilities", "what you will do", "чем ты будешь заниматься",
    )),
    ("conditions", (
        "условия", "мы предлагаем", "что мы предлагаем", "предлагаем", "we offer",
        "что мы даем", "бонусы", "плюшки",
    )),
]

_NICE_INLINE = re.compile(
    r"будет плюсом|будет преимуществом|плюсом будет|преимуществом будет|желательн|приветству|nice to have",
    re.I,
)


def _match_heading(line: str) -> tuple[str, str] | None:
    """Если строка — заголовок раздела, возвращает (раздел, текст после двоеточия)."""
    if line.startswith("- "):
        return None
    low = norm(line).lstrip("•*#").strip()
    for kind, keys in _HEADINGS:
        for key in keys:
            if not low.startswith(key):
                continue
            rest_pos = line.find(":")
            if rest_pos != -1 and rest_pos < 70:
                return kind, line[rest_pos + 1 :].strip()
            if len(low) <= 60:
                return kind, ""
    return None


def split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "intro": [], "responsibilities": [], "requirements": [], "nice_to_have": [], "conditions": [],
    }
    current = "intro"
    for line in text.splitlines():
        heading = _match_heading(line)
        if heading is not None:
            current, rest = heading
            if rest:
                sections[current].append(rest)
            continue
        item = line[2:] if line.startswith("- ") else line
        item = item.strip().rstrip(";").strip()
        if not item:
            continue
        target = current
        if current == "requirements" and _NICE_INLINE.search(item):
            target = "nice_to_have"
        sections[target].append(item)
    return sections


# ---------------------------------------------------------------- инъекции

_INJECTION = re.compile(
    r"игнорир\w*\s+(?:все\s+)?(?:предыдущ|прошл|системн|ранее)"
    r"|ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)"
    r"|(?:\bии\b|\bai\b|llm|нейросет\w*)[\s-]*(?:ассистент|assistant|модел)"
    r"|system\s+prompt|системн\w+\s+(?:промпт|инструкц)"
    r"|you\s+are\s+now|ты\s+теперь"
    r"|напиши,?\s+что\s+кандидат",
    re.I,
)


def find_suspicious_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if _INJECTION.search(line)]


def strip_suspicious_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _INJECTION.search(line))
