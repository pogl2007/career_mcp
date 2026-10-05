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


def label(value: str | None) -> str | None:
    """Названия из справочников hh приходят с неразрывными пробелами («На\xa0месте работодателя»)."""
    return value.replace("\xa0", " ").strip() if value else value


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
    # «что нужно будет делать» — обязанности, хотя начинается как «что нужно уметь»
    ("responsibilities", (
        "обязанности", "задачи", "чем предстоит заниматься", "чем заниматься",
        "что предстоит делать", "что нужно делать", "что нужно будет делать", "ваши задачи",
        "твои задачи", "какие задачи", "примеры задач", "функционал", "вы будете", "ты будешь",
        "чем вы будете заниматься", "responsibilities", "what you will do", "чем ты будешь заниматься",
    )),
    # описание компании, этапы отбора, «чему научишься» — не требования и не условия
    ("other", (
        "о нашей команде", "о команде", "о компании", "о нас", "как попасть", "этапы отбора",
        "как проходит отбор", "чему научишься", "чему ты научишься", "about us",
    )),
    ("conditions", (
        "условия", "мы предлагаем", "что мы предлагаем", "предлагаем", "we offer",
        "что мы даем", "что вас ждет", "что тебя ждет", "что ты получишь", "что вы получите",
        "бонусы", "плюшки",
    )),
    ("requirements", (
        "требования", "мы ждем", "мы ожидаем", "ожидаем", "что мы ждем", "что ждем", "ждем от",
        "что нужно уметь", "что нужно знать", "что нужно иметь", "от вас", "нам важно",
        "кого мы ищем", "наши ожидания", "необходимые навыки", "ключевые требования",
        "requirements", "you have", "must have", "что важно", "мы ищем", "ты идеально",
        "вы идеально", "подходишь, если", "подходите, если", "наши пожелания",
    )),
]
_SECTION_KEYS = ("intro", "responsibilities", "requirements", "nice_to_have", "conditions", "other")

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
    # «Пожалуйста, обрати внимание на требования, это важно:» — короткая строка с двоеточием
    # в конце, где ключевое слово стоит не первым.
    if low.endswith(":") and len(low) <= 70:
        for kind, keys in _HEADINGS:
            if any(key in low for key in keys if len(key) >= 6):
                return kind, ""
    return None


def split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {key: [] for key in _SECTION_KEYS}
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

# Ищем обращение к модели с командой, а не упоминание ИИ: в вакансиях про LLM фразы
# «AI-ассистент», «System Prompt», «AI-модели» — обычное описание работы (живые данные 2026-09-29).
_INJECTION = re.compile(
    r"игнорир\w*\s+(?:все\s+)?(?:предыдущ|прошл|системн|ранее)"
    r"|ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)"
    r"|you\s+are\s+now|ты\s+теперь"
    r"|напиши,?\s+что\s+кандидат"
    r"|new\s+instructions|новые\s+инструкции|system\s+message\s*:"
    r"|(?:note|attention|instructions?)\s+(?:for|to)\s+(?:ai|llm|language\s+models?|assistants?|screening)"
    r"|(?:language\s+model|языков\w+\s+модел\w+|\bии\b|\bai\b|ассистент\w*|assistant|нейросет\w+)\s*[:,]\s*"
    r"(?:игнорир|ignore|напиши|write|ответь|reply|добавь|add|оцени|rate|выведи|output|the\s+correct)",
    re.I,
)


def find_suspicious_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if _INJECTION.search(line)]


def strip_suspicious_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _INJECTION.search(line))
