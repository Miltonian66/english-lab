"""Сверка свободного ответа с эталоном.

Ученик печатает в Telegram, поэтому эталон нельзя сравнивать посимвольно: нужно
прощать регистр, пунктуацию, типографские апострофы, сокращения и опечатку в
слове, которое ему дали, но не прощать другую грамматическую форму. Всё, что
реально допустимо сверх этого, автор контента перечисляет в `accept`.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import itertools
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

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
    # Остальные отрицания однозначны: needn't, mightn't, shan't.
    value = value.replace(" shan't ", " shall not ")
    value = re.sub(r"(?<=\w)n't\b", " not", value)
    return " ".join(value.split())


# Двусмысленные сокращения: he's — это и he is, и he has; I'd — и I would, и
# I had. Раскрывать их одним способом значило бы отвергать верный ответ, а
# любым — засчитывать чужое время. Поэтому ответ читается всеми способами, и
# он верен, если хотя бы одно чтение совпало с эталоном: ученик написал ровно
# то, что допускает запись эталона. У 's оставлен и исходный вид — это ещё и
# притяжательный падеж (Anna's laptop).
_CONTRACTION = re.compile(r"^(.*?)'(s|d|ve|re|ll|m)$")
_READINGS_LIMIT = 64


def readings(text: str) -> set[str]:
    """Все допустимые чтения нормализованного ответа."""
    base = expand(text)
    choices: list[tuple[str, ...]] = []
    for token in base.split():
        found = _CONTRACTION.match(token)
        if not found:
            choices.append((token,))
            continue
        stem, tail = found.groups()
        # Пустая основа — сокращение отдельным словом в пропуске: «If I ___» → «'d known».
        if tail == "s":
            choices.append((token, f"{stem} is".strip(), f"{stem} has".strip()))
        elif tail == "d":
            choices.append((f"{stem} would".strip(), f"{stem} had".strip()))
        else:
            full = {"ve": "have", "re": "are", "ll": "will", "m": "am"}[tail]
            choices.append((f"{stem} {full}".strip(),))
    total = 1
    for options in choices:
        total *= len(options)
    if total > _READINGS_LIMIT:
        return {base}
    return {" ".join(parts) for parts in itertools.product(*choices)}


def _without_commas(text: str) -> str:
    # 50,000 и 50000 — одно число: запятая между цифрами убирается без пробела.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return " ".join(text.replace(",", " ").split())


# Невозможно отличить по тексту, где запятая — пунктуация, а где предмет
# задания. Предметом она бывает в трёх случаях: пункт про пунктуацию,
# неопределительное придаточное (my brother, who lives in Berlin, — один брат)
# и «исправь ошибку», где эталон отличается от условия только запятыми.
_NON_DEFINING = re.compile(r",\s*(which|who|whom|whose|where)\b")


def comma_sensitive(exercise: Exercise) -> bool:
    if "punctuation" in exercise.id:
        return True
    variants = [normalize(item) for item in exercise.expected if item]
    if any(_NON_DEFINING.search(item) for item in variants):
        return True
    if exercise.kind == "correct":
        stripped = _without_commas(normalize(exercise.prompt))
        return any(_without_commas(item) == stripped for item in variants)
    return False


@dataclass(frozen=True)
class Grade:
    """Итог сверки: верно ли, с каким вариантом эталона совпало и что сказать сверх «Верно»."""

    correct: bool
    matched: str = ""
    note: str = ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", expand(normalize(text)))


# «Keep 'will' and correct the sentence: I will can send…» — до двоеточия
# инструкция, а не исправляемое предложение. Сравнивать с эталоном нужно только
# само предложение, иначе вся инструкция попала бы в «исправленный участок».
_INSTRUCTION = re.compile(r"^(?:keep|correct|replace|rewrite|fix)\b[^:]*:\s*", re.IGNORECASE)


def _sentence_tokens(exercise: Exercise) -> list[str]:
    return _tokens(_INSTRUCTION.sub("", exercise.prompt, count=1))


def _changed_span(prompt: list[str], answer: list[str]) -> tuple[int, int] | None:
    """Границы исправленного участка эталона: всё, что отличается от условия.

    Удаление слова само по себе не оставляет в эталоне токенов, поэтому его место
    обозначают соседи: «I can to send» → «can send».
    """
    positions: list[int] = []
    matcher = difflib.SequenceMatcher(None, prompt, answer, autojunk=False)
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            positions.extend(range(j1, j2))
        elif tag == "delete":
            positions.extend(index for index in (j1 - 1, j1) if 0 <= index < len(answer))
    if not positions:
        return None
    return min(positions), max(positions)


def _fragment(exercise: Exercise, candidate: list[str]) -> str | None:
    """«Исправь ошибку»: ученик прислал только исправленную часть, а не всё предложение.

    Ответ засчитывается, если он — непрерывный кусок эталона и целиком покрывает
    исправление. Кусок без исправления («polite») или его половина не пройдут.
    """
    if exercise.kind != "correct" or not candidate:
        return None
    prompt = _sentence_tokens(exercise)
    for variant in exercise.expected:
        answer = _tokens(variant)
        span = _changed_span(prompt, answer)
        if span is None or len(candidate) >= len(answer):
            continue
        low, high = span
        size = len(candidate)
        for start in range(0, low + 1):
            if start + size - 1 >= high and answer[start : start + size] == candidate:
                return variant
    return None


def _one_edit(left: str, right: str) -> bool:
    """Расстояние Дамерау — Левенштейна не больше единицы."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        diff = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(diff) == 1:
            return True
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and left[diff[0]] == right[diff[1]]
            and left[diff[1]] == right[diff[0]]
        )
    short, long_ = sorted((left, right), key=len)
    return any(long_[:index] + long_[index + 1 :] == short for index in range(len(long_)))


_TYPO_KINDS = frozenset({"correct", "order", "transform"})
_DATA = Path(__file__).resolve().parent.parent / "content" / "data"


@functools.lru_cache(maxsize=1)
def known_words() -> frozenset[str]:
    """Все английские слова учебного контента: условия, варианты, ответы, примеры, словарь.

    Опечатка — это не слово. Если набранное есть в контенте («use», «then»,
    «were»), это другое слово или другая форма, то есть грамматическая ошибка,
    и прощать её нельзя.
    """
    words: set[str] = set()
    for path in _DATA.glob("*.json"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for value in re.findall(r'"((?:[^"\\]|\\.)*)"', text):
            words.update(re.findall(r"[a-z]+(?:'[a-z]+)?", value.lower()))
    return frozenset(words)


def _inflection_of(left: str, right: str) -> bool:
    """use/used, work/works, make/making: одна форма слова вместо другой."""
    short, long_ = sorted((left, right), key=len)
    tail = long_[len(short):] if long_.startswith(short) else ""
    if tail in ("s", "es", "d", "ed", "ing"):
        return True
    return short.endswith("e") and long_ == f"{short[:-1]}ing"


def _typo(exercise: Exercise, candidate: list[str]) -> tuple[str, list[tuple[str, str]]] | None:
    """Опечатка в слове, которое ученик не должен был менять.

    Прощается только в словах из самого условия и не в исправленном участке:
    «We hve three old computers» — опечатка, а «He goed» — та самая ошибка,
    которую задание проверяет. Пропуск с одним словом опечаток не прощает: там
    слово и есть ответ. Лишний или пропущенный апостроф — тоже не опечатка:
    customs' officer и its/it's — это грамматика притяжательных форм.
    """
    if exercise.kind not in _TYPO_KINDS or not candidate:
        return None
    given = set(_tokens(exercise.prompt))
    prompt = _sentence_tokens(exercise)
    for variant in exercise.expected:
        answer = _tokens(variant)
        if len(answer) != len(candidate):
            continue
        span = _changed_span(prompt, answer) if exercise.kind == "correct" else None
        changed = set(range(span[0], span[1] + 1)) if span else set()
        slips = [
            (typed, wanted, index)
            for index, (typed, wanted) in enumerate(zip(candidate, answer))
            if typed != wanted
        ]
        if not slips or len(slips) > 2:
            continue
        if all(
            wanted in given
            and len(wanted) >= 4
            and index not in changed
            and _one_edit(typed, wanted)
            and not _inflection_of(typed, wanted)
            and typed.replace("'", "") != wanted.replace("'", "")
            and typed not in known_words()
            for typed, wanted, index in slips
        ):
            return variant, [(typed, wanted) for typed, wanted, _ in slips]
    return None


def grade(exercise: Exercise, given: str) -> Grade:
    candidate = normalize(given)
    if not candidate:
        return Grade(False)
    variants = [item for item in exercise.expected if item]
    for variant in variants:
        if normalize(variant) == candidate or expand(normalize(variant)) == expand(candidate):
            return Grade(True, variant)

    loose_commas = not comma_sensitive(exercise)
    heard = readings(candidate)
    if loose_commas:
        heard |= {_without_commas(item) for item in heard}
    for variant in variants:
        wanted = readings(normalize(variant))
        if loose_commas:
            wanted = {_without_commas(item) for item in wanted}
        if heard & wanted:
            return Grade(True, variant)

    tokens = re.findall(r"[a-z0-9']+", expand(candidate))
    fragment = _fragment(exercise, tokens)
    if fragment:
        return Grade(True, fragment, f"Целиком: {fragment}")
    typo = _typo(exercise, tokens)
    if typo:
        variant, slips = typo
        fixes = ", ".join(f"{typed} → {wanted}" for typed, wanted in slips)
        return Grade(True, variant, f"Засчитано, но проверь написание: {fixes}.")
    return Grade(False)


def matches(exercise: Exercise, given: str) -> bool:
    return grade(exercise, given).correct


def option_order(key: str, count: int) -> tuple[int, ...]:
    """Порядок показа вариантов: устойчивый для задания, но не порядок из файла.

    В банке правильный вариант стоит первым почти в половине заданий, а на C2 —
    в двух третях: «жать A» проходило блок диагностики чаще, чем знание языка.
    Порядок выводится из id, поэтому одно и то же задание всегда показывается
    одинаково — иначе показанный вопрос и проверяемый ответ разошлись бы.
    """
    if count <= 1:
        return tuple(range(count))
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    order = list(range(count))
    random.Random(int.from_bytes(digest, "big")).shuffle(order)
    return tuple(order)


def display_options(exercise: Exercise) -> tuple[tuple[int, str], ...]:
    """Пары «исходный индекс, текст» в порядке показа."""
    order = option_order(exercise.id, len(exercise.options))
    return tuple((index, exercise.options[index]) for index in order)


def parse_choice(exercise: Exercise, text: str, shuffle: bool = True) -> int | None:
    """Принимает букву A–H, номер 1–8 или текст самого варианта.

    Буква и номер относятся к порядку показа, а возвращается исходный индекс:
    ученик видит перемешанные варианты, а проверка идёт по данным задания.
    `shuffle=False` — когда варианты уже разложены в порядке показа и второе
    перемешивание развело бы кнопку «A» и букву «A».
    """
    raw = text.strip()
    if not raw or not exercise.options:
        return None
    shown = (
        display_options(exercise)
        if shuffle
        else tuple(enumerate(exercise.options))
    )

    # Сначала точное совпадение с текстом варианта: иначе ответ "a" на задание
    # про артикли был бы прочитан как метка варианта A.
    lowered = normalize(raw)
    for index, option in enumerate(exercise.options):
        if normalize(option) == lowered:
            return index

    head = raw.split()[0].strip(".)-:").upper()
    if len(head) == 1 and head.isalpha():
        position = ord(head) - ord("A")
        if 0 <= position < len(shown):
            return shown[position][0]
    if head.isdigit():
        position = int(head) - 1
        if 0 <= position < len(shown):
            return shown[position][0]
    # Ученик мог прислать «C) has lived» — отрезаем метку и сверяем остаток.
    stripped = normalize(re.sub(r"^[A-Ha-h1-8][).:-]\s*", "", raw))
    for index, option in enumerate(exercise.options):
        if normalize(option) == stripped:
            return index
    return None


def labelled_options(exercise: Exercise) -> list[str]:
    """Подписанные варианты в порядке показа, а не в порядке из файла."""
    return [
        f"{chr(ord('A') + position)}) {option}"
        for position, (_, option) in enumerate(display_options(exercise))
    ]


def correct_answer_text(exercise: Exercise) -> str:
    if exercise.kind == "choice" and exercise.correct_index is not None:
        return exercise.options[exercise.correct_index]
    return exercise.answer
