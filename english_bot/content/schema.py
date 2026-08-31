"""Схема учебного контента и строгая валидация JSON-файлов курса.

Контент хранится данными в `english_bot/content/data/*.json`, а не литералами в
коде: файлы генерируются и проверяются отдельно, а загрузчик отвечает только за
разбор и инварианты. Источник истины по форме данных — этот модуль.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


LEVELS: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")
BANDS: tuple[str, ...] = ("A1", "A2", "B1", "B2.1", "B2.2", "C1", "C2")
EXERCISE_KINDS: tuple[str, ...] = ("choice", "gap", "correct", "transform", "order")

LEVEL_ORDER = {level: index for index, level in enumerate(LEVELS)}
BAND_ORDER = {band: index for index, band in enumerate(BANDS)}

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{3,79}$")
MIN_EXERCISES = 6


class ContentError(ValueError):
    """Контент нарушает контракт схемы."""


@dataclass(frozen=True)
class Exercise:
    id: str
    kind: str
    prompt: str
    explanation_ru: str
    difficulty: int = 2
    options: tuple[str, ...] = ()
    correct_index: int | None = None
    answer: str = ""
    accept: tuple[str, ...] = ()

    @property
    def expected(self) -> tuple[str, ...]:
        """Все строки, которые считаются верным ответом."""
        if self.kind == "choice":
            if self.correct_index is None:
                return ()
            return (self.options[self.correct_index],)
        return (self.answer, *self.accept)


@dataclass(frozen=True)
class GrammarPoint:
    id: str
    level: str
    band: str
    topic: str
    title_en: str
    title_ru: str
    summary_ru: str
    ru_interference: str
    forms: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    exercises: tuple[Exercise, ...] = ()
    source_slug: str = ""

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (LEVEL_ORDER[self.level], BAND_ORDER.get(self.band, 0), self.title_en)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContentError(message)


def _text(raw: object, field_name: str, where: str, minimum: int = 1) -> str:
    _require(isinstance(raw, str), f"{where}: поле {field_name} должно быть строкой")
    value = str(raw).strip()
    _require(len(value) >= minimum, f"{where}: поле {field_name} короче {minimum} символов")
    return value


def _tuple(raw: object, field_name: str, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    _require(isinstance(raw, list), f"{where}: поле {field_name} должно быть списком")
    return tuple(_text(item, field_name, where) for item in raw)


def parse_exercise(raw: object, where: str) -> Exercise:
    _require(isinstance(raw, dict), f"{where}: упражнение должно быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    exercise_id = _text(data.get("id"), "id", where, 4)
    _require(bool(ID_PATTERN.match(exercise_id)), f"{where}: некорректный id упражнения {exercise_id!r}")
    where = f"{where}/{exercise_id}"

    kind = _text(data.get("kind"), "kind", where)
    _require(kind in EXERCISE_KINDS, f"{where}: неизвестный kind {kind!r}")
    prompt = _text(data.get("prompt"), "prompt", where, 4)
    explanation = _text(data.get("explanation_ru"), "explanation_ru", where, 8)

    difficulty = data.get("difficulty", 2)
    _require(isinstance(difficulty, int) and 1 <= difficulty <= 3, f"{where}: difficulty вне 1..3")

    options = _tuple(data.get("options"), "options", where)
    correct_index = data.get("correct_index")
    answer = str(data.get("answer") or "").strip()
    accept = _tuple(data.get("accept"), "accept", where)

    if kind == "choice":
        _require(len(options) >= 3, f"{where}: у choice нужно минимум 3 варианта")
        _require(len(set(options)) == len(options), f"{where}: варианты choice повторяются")
        _require(
            isinstance(correct_index, int) and 0 <= correct_index < len(options),
            f"{where}: correct_index вне диапазона вариантов",
        )
    else:
        _require(not options, f"{where}: options допустимы только для kind=choice")
        _require(correct_index is None, f"{where}: correct_index допустим только для kind=choice")
        _require(len(answer) >= 1, f"{where}: у {kind} обязателен answer")

    return Exercise(
        id=exercise_id,
        kind=kind,
        prompt=prompt,
        explanation_ru=explanation,
        difficulty=int(difficulty),
        options=options,
        correct_index=correct_index if kind == "choice" else None,
        answer=answer,
        accept=accept,
    )


def parse_point(raw: object, where: str) -> GrammarPoint:
    _require(isinstance(raw, dict), f"{where}: пункт должен быть объектом")
    data = dict(raw)  # type: ignore[arg-type]
    point_id = _text(data.get("id"), "id", where, 4)
    _require(bool(ID_PATTERN.match(point_id)), f"{where}: некорректный id пункта {point_id!r}")
    where = f"{where}/{point_id}"

    level = _text(data.get("level"), "level", where)
    _require(level in LEVELS, f"{where}: неизвестный level {level!r}")
    band = _text(data.get("band"), "band", where)
    _require(band in BANDS, f"{where}: неизвестный band {band!r}")

    exercises_raw = data.get("exercises")
    _require(isinstance(exercises_raw, list), f"{where}: exercises должен быть списком")
    exercises = tuple(parse_exercise(item, where) for item in exercises_raw)  # type: ignore[union-attr]
    _require(
        len(exercises) >= MIN_EXERCISES,
        f"{where}: нужно минимум {MIN_EXERCISES} упражнений, получено {len(exercises)}",
    )
    ids = [exercise.id for exercise in exercises]
    _require(len(set(ids)) == len(ids), f"{where}: id упражнений повторяются")
    _require(
        sum(1 for exercise in exercises if exercise.kind == "choice") >= 3,
        f"{where}: нужно минимум 3 упражнения kind=choice",
    )
    _require(
        any(exercise.kind != "choice" for exercise in exercises),
        f"{where}: нужно хотя бы одно упражнение со свободным вводом",
    )

    return GrammarPoint(
        id=point_id,
        level=level,
        band=band,
        topic=_text(data.get("topic"), "topic", where),
        title_en=_text(data.get("title_en"), "title_en", where, 3),
        title_ru=_text(data.get("title_ru"), "title_ru", where, 3),
        summary_ru=_text(data.get("summary_ru"), "summary_ru", where, 80),
        ru_interference=_text(data.get("ru_interference"), "ru_interference", where, 20),
        forms=_tuple(data.get("forms"), "forms", where),
        examples=_tuple(data.get("examples"), "examples", where),
        prerequisites=_tuple(data.get("prerequisites"), "prerequisites", where),
        exercises=exercises,
        source_slug=str(data.get("source_slug") or "").strip(),
    )


def parse_file(path: Path) -> list[GrammarPoint]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContentError(f"{path.name}: некорректный JSON — {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name}: корень должен быть объектом")
    points_raw = payload.get("points")
    _require(isinstance(points_raw, list), f"{path.name}: поле points должно быть списком")
    return [parse_point(item, path.name) for item in points_raw]  # type: ignore[union-attr]


@dataclass
class ValidationReport:
    files: int = 0
    points: int = 0
    exercises: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_directory(
    directory: Path, skip: "Callable[[Path], bool] | None" = None
) -> ValidationReport:
    """Проверяет грамматические файлы каталога и возвращает отчёт вместо исключения.

    `skip` отсекает файлы, которые грамматикой не являются (банки лексики и заданий).
    """
    report = ValidationReport()
    seen_points: dict[str, str] = {}
    seen_exercises: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        if skip is not None and skip(path):
            continue
        report.files += 1
        try:
            points = parse_file(path)
        except ContentError as exc:
            report.errors.append(str(exc))
            continue
        for point in points:
            if point.id in seen_points:
                report.errors.append(
                    f"{path.name}: дубликат пункта {point.id} (уже в {seen_points[point.id]})"
                )
                continue
            seen_points[point.id] = path.name
            report.points += 1
            for exercise in point.exercises:
                if exercise.id in seen_exercises:
                    report.errors.append(
                        f"{path.name}: дубликат упражнения {exercise.id} "
                        f"(уже в {seen_exercises[exercise.id]})"
                    )
                    continue
                seen_exercises[exercise.id] = path.name
                report.exercises += 1
    return report
