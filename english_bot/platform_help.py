"""Локальный поиск по базе знаний справочного агента English Lab."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


KNOWLEDGE_PATH = Path(__file__).with_name("platform_knowledge.json")
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
ARTICLE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
STOP_WORDS = {
    "а", "без", "бот", "бы", "в", "вы", "где", "для", "до", "его", "если",
    "и", "из", "или", "как", "мне", "можно", "мой", "на", "не", "нужно", "о",
    "он", "она", "по", "почему", "платформа", "про", "с", "сколько", "так", "то", "у", "что",
    "это", "я", "english", "lab", "the", "how", "is", "to", "what", "where",
}


class KnowledgeError(ValueError):
    """База знаний повреждена и не должна молча кормить модель мусором."""


@dataclass(frozen=True)
class KnowledgeArticle:
    id: str
    title_ru: str
    keywords: tuple[str, ...]
    answer_ru: str


def _required_string(raw: dict[str, Any], field: str, article_id: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError(f"{article_id}: поле {field} должно быть непустой строкой")
    return value.strip()


def load_knowledge(path: Path = KNOWLEDGE_PATH) -> tuple[KnowledgeArticle, ...]:
    """Загружает и строго проверяет пользовательские факты платформы."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"не удалось прочитать базу знаний: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise KnowledgeError("база знаний должна быть объектом версии 1")
    raw_articles = payload.get("articles")
    if not isinstance(raw_articles, list) or not raw_articles:
        raise KnowledgeError("articles должен быть непустым списком")

    articles: list[KnowledgeArticle] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_articles):
        if not isinstance(raw, dict):
            raise KnowledgeError(f"articles[{index}] должен быть объектом")
        article_id = _required_string(raw, "id", f"articles[{index}]")
        if not ARTICLE_ID_RE.fullmatch(article_id) or article_id in seen:
            raise KnowledgeError(f"некорректный или повторный id: {article_id}")
        keywords = raw.get("keywords")
        if (
            not isinstance(keywords, list)
            or len(keywords) < 2
            or any(not isinstance(item, str) or not item.strip() for item in keywords)
        ):
            raise KnowledgeError(f"{article_id}: keywords должен содержать минимум две строки")
        answer = _required_string(raw, "answer_ru", article_id)
        if len(answer) > 1200:
            raise KnowledgeError(f"{article_id}: ответ длиннее 1200 символов")
        seen.add(article_id)
        articles.append(
            KnowledgeArticle(
                id=article_id,
                title_ru=_required_string(raw, "title_ru", article_id),
                keywords=tuple(item.strip() for item in keywords),
                answer_ru=answer,
            )
        )
    return tuple(articles)


@lru_cache(maxsize=1)
def platform_knowledge() -> tuple[KnowledgeArticle, ...]:
    return load_knowledge()


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.casefold() not in STOP_WORDS
    }


def _matches(left: str, right: str) -> bool:
    if left == right:
        return True
    # Для частых русских окончаний достаточно устойчивого пятибуквенного начала:
    # «диагностика», «диагностике», «диагностический» найдут одну статью.
    return len(left) >= 6 and len(right) >= 6 and left[:5] == right[:5]


def _score(question: str, article: KnowledgeArticle) -> int:
    question_folded = question.casefold()
    question_tokens = _tokens(question)
    if not question_tokens:
        return 0
    keyword_text = " ".join(article.keywords).casefold()
    keyword_tokens = _tokens(keyword_text)
    title_tokens = _tokens(article.title_ru)
    answer_tokens = _tokens(article.answer_ru)

    score = 0
    for keyword in article.keywords:
        phrase = keyword.casefold().strip()
        # Бонус только за составную фразу. Одиночный токен иначе учитывался бы
        # дважды и общая статья «результаты» обгоняла бы точную «диагностику».
        if (" " in phrase or phrase.startswith("/")) and phrase in question_folded:
            score += 8
    for token in question_tokens:
        if any(_matches(token, candidate) for candidate in title_tokens):
            score += 5
        elif any(_matches(token, candidate) for candidate in keyword_tokens):
            score += 4
        elif any(_matches(token, candidate) for candidate in answer_tokens):
            score += 1
    return score


def find_articles(
    question: str,
    *,
    limit: int = 4,
    articles: Iterable[KnowledgeArticle] | None = None,
) -> tuple[KnowledgeArticle, ...]:
    """Возвращает самые релевантные статьи; нулевая близость означает «не по теме»."""
    if limit < 1:
        return ()
    candidates = articles if articles is not None else platform_knowledge()
    ranked = sorted(
        ((_score(question, article), article) for article in candidates),
        key=lambda item: (-item[0], item[1].id),
    )
    # Совпадение лишь со случайным словом из текста ответа слишком шумное. Для
    # допуска нужен хотя бы title/keyword match (4 балла).
    return tuple(article for score, article in ranked[:limit] if score >= 4)


def knowledge_excerpt(articles: Iterable[KnowledgeArticle]) -> str:
    return "\n\n".join(
        f"[{article.id}] {article.title_ru}\n{article.answer_ru}" for article in articles
    )
