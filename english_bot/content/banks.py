"""Схемы вспомогательных банков: лексика, речь, письмо, аудирование и ошибки.

Грамматика описана в `schema.py`; здесь — всё остальное, что платформа выдаёт
пользователю. Каждый банк лежит отдельным JSON в `content/data/` и проверяется
теми же принципами: строгая валидация при загрузке, никаких молчаливых пропусков.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import ID_PATTERN, LEVELS, ContentError, _require, _text, _tuple


PARTS_OF_SPEECH: tuple[str, ...] = (
    "noun", "verb", "adjective", "adverb", "phrase", "phrasal verb", "preposition", "conjunction",
)
SPEAKING_MODES: tuple[str, ...] = ("monologue", "roleplay", "opinion", "describe", "interview")
LISTENING_SKILLS: tuple[str, ...] = ("gist", "detail", "inference", "attitude", "sequence")
ERROR_CATEGORIES: tuple[str, ...] = (
    "grammar", "word_choice", "article", "preposition", "word_order", "spelling",
    "punctuation", "expression", "pronunciation",
)


@dataclass(frozen=True)
class VocabItem:
    id: str
    level: str
    word: str
    pos: str
    ipa_us: str
    translation_ru: str
    example_en: str
    collocations: tuple[str, ...] = ()
    topic: str = ""


@dataclass(frozen=True)
class SpeakingTask:
    id: str
    level: str
    mode: str
    title_ru: str
    prompt_en: str
    guidance_ru: str
    seconds_min: int
    seconds_max: int
    focus: tuple[str, ...] = ()


@dataclass(frozen=True)
class WritingTask:
    id: str
    level: str
    title_ru: str
    prompt_en: str
    guidance_ru: str
    words_min: int
    words_max: int
    focus: tuple[str, ...] = ()


@dataclass(frozen=True)
class ListeningTask:
    id: str
    level: str
    skill: str
    title_ru: str
    script_en: str
    question_en: str
    options: tuple[str, ...]
    correct_index: int
    explanation_ru: str


@dataclass(frozen=True)
class ErrorPattern:
    """Ярлык ошибки с триггером в родном языке — модель из claude-english-immersion,
    переписанная под русскоязычных."""

    id: str
    category: str
    label_ru: str
    what_ru: str
    trigger_ru: str
    fix_ru: str
    wrong_en: str
    right_en: str
    levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class SoundNote:
    id: str
    ipa: str
    title_ru: str
    problem_ru: str
    advice_ru: str
    minimal_pairs: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


def _level(raw: object, where: str) -> str:
    value = _text(raw, "level", where)
    _require(value in LEVELS, f"{where}: неизвестный level {value!r}")
    return value


def _ident(raw: object, where: str) -> str:
    value = _text(raw, "id", where, 3)
    _require(bool(ID_PATTERN.match(value)), f"{where}: некорректный id {value!r}")
    return value


def _int_range(raw: object, name: str, where: str, low: int, high: int) -> int:
    _require(isinstance(raw, int), f"{where}: поле {name} должно быть целым")
    value = int(raw)  # type: ignore[arg-type]
    _require(low <= value <= high, f"{where}: поле {name} вне диапазона {low}..{high}")
    return value


def parse_vocab(raw: object, where: str) -> VocabItem:
    _require(isinstance(raw, dict), f"{where}: элемент лексики должен быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    item_id = _ident(data.get("id"), where)
    where = f"{where}/{item_id}"
    pos = _text(data.get("pos"), "pos", where)
    _require(pos in PARTS_OF_SPEECH, f"{where}: неизвестная часть речи {pos!r}")
    ipa = _text(data.get("ipa_us"), "ipa_us", where, 2)
    _require(ipa.startswith("/") and ipa.endswith("/"), f"{where}: ipa_us должен быть в слэшах")
    return VocabItem(
        id=item_id,
        level=_level(data.get("level"), where),
        word=_text(data.get("word"), "word", where, 2),
        pos=pos,
        ipa_us=ipa,
        translation_ru=_text(data.get("translation_ru"), "translation_ru", where, 2),
        example_en=_text(data.get("example_en"), "example_en", where, 8),
        collocations=_tuple(data.get("collocations"), "collocations", where),
        topic=str(data.get("topic") or "").strip(),
    )


def parse_speaking(raw: object, where: str) -> SpeakingTask:
    _require(isinstance(raw, dict), f"{where}: speaking-задание должно быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    task_id = _ident(data.get("id"), where)
    where = f"{where}/{task_id}"
    mode = _text(data.get("mode"), "mode", where)
    _require(mode in SPEAKING_MODES, f"{where}: неизвестный mode {mode!r}")
    seconds_min = _int_range(data.get("seconds_min"), "seconds_min", where, 20, 300)
    seconds_max = _int_range(data.get("seconds_max"), "seconds_max", where, 30, 600)
    _require(seconds_max > seconds_min, f"{where}: seconds_max должен быть больше seconds_min")
    return SpeakingTask(
        id=task_id,
        level=_level(data.get("level"), where),
        mode=mode,
        title_ru=_text(data.get("title_ru"), "title_ru", where, 3),
        prompt_en=_text(data.get("prompt_en"), "prompt_en", where, 20),
        guidance_ru=_text(data.get("guidance_ru"), "guidance_ru", where, 20),
        seconds_min=seconds_min,
        seconds_max=seconds_max,
        focus=_tuple(data.get("focus"), "focus", where),
    )


def parse_writing(raw: object, where: str) -> WritingTask:
    _require(isinstance(raw, dict), f"{where}: writing-задание должно быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    task_id = _ident(data.get("id"), where)
    where = f"{where}/{task_id}"
    words_min = _int_range(data.get("words_min"), "words_min", where, 20, 400)
    words_max = _int_range(data.get("words_max"), "words_max", where, 40, 800)
    _require(words_max > words_min, f"{where}: words_max должен быть больше words_min")
    return WritingTask(
        id=task_id,
        level=_level(data.get("level"), where),
        title_ru=_text(data.get("title_ru"), "title_ru", where, 3),
        prompt_en=_text(data.get("prompt_en"), "prompt_en", where, 20),
        guidance_ru=_text(data.get("guidance_ru"), "guidance_ru", where, 20),
        words_min=words_min,
        words_max=words_max,
        focus=_tuple(data.get("focus"), "focus", where),
    )


def parse_listening(raw: object, where: str) -> ListeningTask:
    _require(isinstance(raw, dict), f"{where}: listening-задание должно быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    task_id = _ident(data.get("id"), where)
    where = f"{where}/{task_id}"
    skill = _text(data.get("skill"), "skill", where)
    _require(skill in LISTENING_SKILLS, f"{where}: неизвестный skill {skill!r}")
    options = _tuple(data.get("options"), "options", where)
    _require(len(options) == 4, f"{where}: options должен содержать ровно 4 варианта")
    _require(
        len({option.casefold() for option in options}) == len(options),
        f"{where}: варианты ответа повторяются",
    )
    correct_index = _int_range(data.get("correct_index"), "correct_index", where, 0, 3)
    return ListeningTask(
        id=task_id,
        level=_level(data.get("level"), where),
        skill=skill,
        title_ru=_text(data.get("title_ru"), "title_ru", where, 3),
        script_en=_text(data.get("script_en"), "script_en", where, 20),
        question_en=_text(data.get("question_en"), "question_en", where, 8),
        options=options,
        correct_index=correct_index,
        explanation_ru=_text(data.get("explanation_ru"), "explanation_ru", where, 10),
    )


def parse_error(raw: object, where: str) -> ErrorPattern:
    _require(isinstance(raw, dict), f"{where}: паттерн ошибки должен быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    pattern_id = _ident(data.get("id"), where)
    where = f"{where}/{pattern_id}"
    category = _text(data.get("category"), "category", where)
    _require(category in ERROR_CATEGORIES, f"{where}: неизвестная категория {category!r}")
    levels = _tuple(data.get("levels"), "levels", where)
    for level in levels:
        _require(level in LEVELS, f"{where}: неизвестный level {level!r}")
    return ErrorPattern(
        id=pattern_id,
        category=category,
        label_ru=_text(data.get("label_ru"), "label_ru", where, 3),
        what_ru=_text(data.get("what_ru"), "what_ru", where, 10),
        trigger_ru=_text(data.get("trigger_ru"), "trigger_ru", where, 10),
        fix_ru=_text(data.get("fix_ru"), "fix_ru", where, 10),
        wrong_en=_text(data.get("wrong_en"), "wrong_en", where, 3),
        right_en=_text(data.get("right_en"), "right_en", where, 3),
        levels=levels,
    )


def parse_sound(raw: object, where: str) -> SoundNote:
    _require(isinstance(raw, dict), f"{where}: звук должен быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    sound_id = _ident(data.get("id"), where)
    where = f"{where}/{sound_id}"
    return SoundNote(
        id=sound_id,
        ipa=_text(data.get("ipa"), "ipa", where, 1),
        title_ru=_text(data.get("title_ru"), "title_ru", where, 3),
        problem_ru=_text(data.get("problem_ru"), "problem_ru", where, 10),
        advice_ru=_text(data.get("advice_ru"), "advice_ru", where, 10),
        minimal_pairs=_tuple(data.get("minimal_pairs"), "minimal_pairs", where),
        examples=_tuple(data.get("examples"), "examples", where),
    )


BANKS: dict[str, tuple[str, object]] = {
    "vocabulary": ("items", parse_vocab),
    "speaking_tasks": ("tasks", parse_speaking),
    "writing_tasks": ("tasks", parse_writing),
    "listening_tasks": ("tasks", parse_listening),
    "error_patterns": ("patterns", parse_error),
    "sounds": ("sounds", parse_sound),
}

def bank_of_stem(stem: str) -> str | None:
    """Банк по имени файла без расширения: `vocabulary_b1` → `vocabulary`.

    Единственный источник правды о зарезервированных именах: и загрузчик курса, и
    валидатор обязаны спрашивать эту функцию, иначе файл грамматики с именем вроде
    `sounds_extra.json` попадёт в один инструмент и не попадёт в другой.
    """
    for name in BANKS:
        if stem == name or stem.startswith(f"{name}_"):
            return name
    return None




def parse_bank(path: Path, bank: str) -> list[object]:
    key, parser = BANKS[bank]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContentError(f"{path.name}: некорректный JSON — {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name}: корень должен быть объектом")
    rows = payload.get(key)
    _require(isinstance(rows, list), f"{path.name}: поле {key} должно быть списком")
    parsed = [parser(row, path.name) for row in rows]  # type: ignore[operator, union-attr]
    ids = [getattr(item, "id") for item in parsed]
    _require(len(set(ids)) == len(ids), f"{path.name}: id повторяются")
    return parsed
