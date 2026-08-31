"""Сверка свободного ответа с эталоном.

Ученик печатает в Telegram, поэтому эталон нельзя сравнивать посимвольно: нужно
прощать регистр, финальную точку, типографские апострофы и лишние пробелы, но не
прощать другую грамматическую форму. Всё, что реально допустимо, автор контента
перечисляет в `accept`.
"""

from __future__ import annotations

import re
import unicodedata

from ..content.schema import Exercise


APOSTROPHES = {"’": "'", "ʼ": "'", "´": "'", "`": "'"}
DASHES = {"–": "-", "—": "-", "−": "-"}
QUOTES = {"“": '"', "”": '"', "„": '"', "«": '"', "»": '"'}

# Только однозначные пары. Сокращения на `'s` и `'d` намеренно отсутствуют:
# "he's" — это и "he is", и "he has"; "I'd" — и "I would", и "I had". Разворачивать
# их значило бы засчитывать верным ответ в другом времени.
CONTRACTIONS: tuple[tuple[str, str], ...] = (
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
    ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"),
    ("were not", "weren't"), ("have not", "haven't"), ("has not", "hasn't"),
    ("had not", "hadn't"), ("will not", "won't"), ("would not", "wouldn't"),
    ("could not", "couldn't"), ("should not", "shouldn't"), ("must not", "mustn't"),
    ("i am", "i'm"), ("they are", "they're"), ("we are", "we're"), ("you are", "you're"),
    ("i have", "i've"), ("we have", "we've"), ("they have", "they've"), ("you have", "you've"),
    ("i will", "i'll"), ("we will", "we'll"), ("they will", "they'll"), ("you will", "you'll"),
)


def normalize(text: str) -> str:
    """Приводит ответ к сравнимому виду, не меняя грамматику."""
    value = unicodedata.normalize("NFKC", text).strip().lower()
    for source, target in {**APOSTROPHES, **DASHES, **QUOTES}.items():
        value = value.replace(source, target)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .!?;:,")
    return value


def expand(text: str) -> str:
    """Сводит сокращения к полной форме, чтобы don't и do not совпадали.

    Разворачивается только короткая форма: обратное направление породило бы
    неоднозначность, а нормализация обеих сторон и так приводит их к одному виду.
    """
    value = f" {text} "
    # "cannot" слитное, поэтому идёт до пробельных правил — иначе правило мёртвое.
    value = value.replace(" cannot ", " can not ").replace(" can't ", " can not ")
    for full, short in CONTRACTIONS:
        value = value.replace(f" {short} ", f" {full} ")
    return " ".join(value.split())


def matches(exercise: Exercise, given: str) -> bool:
    candidate = normalize(given)
    if not candidate:
        return False
    variants = {normalize(item) for item in exercise.expected if item}
    if candidate in variants:
        return True
    expanded = expand(candidate)
    return any(expand(variant) == expanded for variant in variants)


def parse_choice(exercise: Exercise, text: str) -> int | None:
    """Принимает букву A–H, номер 1–8 или текст самого варианта."""
    raw = text.strip()
    if not raw or not exercise.options:
        return None

    # Сначала точное совпадение с текстом варианта: иначе ответ "a" на задание
    # про артикли был бы прочитан как метка варианта A.
    lowered = normalize(raw)
    for index, option in enumerate(exercise.options):
        if normalize(option) == lowered:
            return index

    head = raw.split()[0].strip(".)-:").upper()
    if len(head) == 1 and head.isalpha():
        index = ord(head) - ord("A")
        if 0 <= index < len(exercise.options):
            return index
    if head.isdigit():
        index = int(head) - 1
        if 0 <= index < len(exercise.options):
            return index
    # Ученик мог прислать «C) has lived» — отрезаем метку и сверяем остаток.
    stripped = normalize(re.sub(r"^[A-Ha-h1-8][).:-]\s*", "", raw))
    for index, option in enumerate(exercise.options):
        if normalize(option) == stripped:
            return index
    return None


def labelled_options(exercise: Exercise) -> list[str]:
    return [
        f"{chr(ord('A') + index)}) {option}" for index, option in enumerate(exercise.options)
    ]


def correct_answer_text(exercise: Exercise) -> str:
    if exercise.kind == "choice" and exercise.correct_index is not None:
        return exercise.options[exercise.correct_index]
    return exercise.answer
