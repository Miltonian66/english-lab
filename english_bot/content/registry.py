"""Реестр учебного контента: единая точка доступа к грамматике, лексике и банкам.

Загрузка терпима к частично готовому каталогу: сломанный файл пропускается с
записью в лог, остальная платформа продолжает работать. Строгий шлюз — это
`python3 -m english_bot.content.validate`, который гоняется в тестах.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .banks import (
    ErrorPattern,
    ListeningTask,
    SoundNote,
    SpeakingTask,
    VocabItem,
    WritingTask,
    bank_of_stem,
    parse_bank,
)
from .schema import (
    BAND_ORDER,
    LEVEL_ORDER,
    LEVELS,
    ContentError,
    Exercise,
    GrammarPoint,
    parse_file,
)


LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"

BANK_FILES: dict[str, str] = {
    "speaking_tasks": "speaking_tasks.json",
    "writing_tasks": "writing_tasks.json",
    "listening_tasks": "listening_tasks.json",
    "error_patterns": "error_patterns.json",
    "sounds": "sounds.json",
}


@dataclass
class Curriculum:
    points: dict[str, GrammarPoint] = field(default_factory=dict)
    exercises: dict[str, tuple[Exercise, str]] = field(default_factory=dict)
    vocabulary: dict[str, list[VocabItem]] = field(default_factory=dict)
    speaking: dict[str, list[SpeakingTask]] = field(default_factory=dict)
    writing: dict[str, list[WritingTask]] = field(default_factory=dict)
    listening: dict[str, list[ListeningTask]] = field(default_factory=dict)
    error_patterns: dict[str, ErrorPattern] = field(default_factory=dict)
    sounds: list[SoundNote] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)

    # ── короткие коды для callback_data ──────────────────────────
    # Telegram ограничивает callback_data 64 байтами, а id пункта бывает длиннее.
    # Код — устойчивый хеш от самого id, поэтому не ломается при росте каталога.

    def point_code(self, point_id: str) -> str:
        return hashlib.sha1(point_id.encode()).hexdigest()[:8]

    def by_code(self, code: str) -> GrammarPoint | None:
        for point in self.points.values():
            if self.point_code(point.id) == code:
                return point
        return None

    def topic_code(self, level: str, topic: str) -> str:
        return hashlib.sha1(f"{level}|{topic}".encode()).hexdigest()[:8]

    def topic_by_code(self, level: str, code: str) -> str | None:
        for topic, _ in self.topics_of_level(level):
            if self.topic_code(level, topic) == code:
                return topic
        return None

    # ── грамматика ────────────────────────────────────────────────

    def levels(self) -> list[str]:
        """Уровни, для которых есть хотя бы один пункт, в порядке CEFR."""
        present = {point.level for point in self.points.values()}
        return [level for level in LEVELS if level in present]

    def points_of_level(self, level: str) -> list[GrammarPoint]:
        rows = [point for point in self.points.values() if point.level == level]
        rows.sort(key=lambda point: point.sort_key)
        return rows

    def topics_of_level(self, level: str) -> list[tuple[str, int]]:
        """Темы уровня с числом пунктов, в порядке первого появления по band."""
        counter: dict[str, int] = defaultdict(int)
        order: dict[str, tuple[int, str]] = {}
        for point in self.points_of_level(level):
            counter[point.topic] += 1
            order.setdefault(point.topic, (BAND_ORDER.get(point.band, 0), point.topic))
        return sorted(counter.items(), key=lambda item: order[item[0]])

    def points_of_topic(self, level: str, topic: str) -> list[GrammarPoint]:
        return [point for point in self.points_of_level(level) if point.topic == topic]

    def point(self, point_id: str) -> GrammarPoint | None:
        return self.points.get(point_id)

    def exercise(self, exercise_id: str) -> tuple[Exercise, GrammarPoint] | None:
        row = self.exercises.get(exercise_id)
        if row is None:
            return None
        exercise, point_id = row
        point = self.points.get(point_id)
        return (exercise, point) if point else None

    def search(self, query: str, limit: int = 12) -> list[GrammarPoint]:
        """Поиск пункта по русскому или английскому названию и теме."""
        needle = query.strip().lower()
        if len(needle) < 2:
            return []
        scored: list[tuple[int, GrammarPoint]] = []
        for point in self.points.values():
            haystacks = (point.title_en.lower(), point.title_ru.lower(), point.topic.lower())
            if any(needle == text for text in haystacks):
                scored.append((0, point))
            elif any(text.startswith(needle) for text in haystacks):
                scored.append((1, point))
            elif any(needle in text for text in haystacks):
                scored.append((2, point))
            elif needle in point.summary_ru.lower():
                scored.append((3, point))
        scored.sort(key=lambda item: (item[0], item[1].sort_key))
        return [point for _, point in scored[:limit]]

    # ── банки ─────────────────────────────────────────────────────

    def vocab_of_level(self, level: str) -> list[VocabItem]:
        return self.vocabulary.get(level, [])

    def vocab_upto(self, level: str) -> list[VocabItem]:
        """Лексика уровня и всех уровней ниже — база для интервального повторения."""
        ceiling = LEVEL_ORDER[level]
        rows: list[VocabItem] = []
        for name in LEVELS:
            if LEVEL_ORDER[name] <= ceiling:
                rows.extend(self.vocabulary.get(name, []))
        return rows

    def speaking_of_level(self, level: str) -> list[SpeakingTask]:
        return self.speaking.get(level, [])

    def writing_of_level(self, level: str) -> list[WritingTask]:
        return self.writing.get(level, [])

    def listening_of_level(self, level: str) -> list[ListeningTask]:
        return self.listening.get(level, [])

    def listening_code(self, task_id: str) -> str:
        return hashlib.sha1(task_id.encode()).hexdigest()[:8]

    def listening_by_code(self, code: str) -> ListeningTask | None:
        for tasks in self.listening.values():
            for task in tasks:
                if self.listening_code(task.id) == code:
                    return task
        return None

    def sound(self, ipa: str) -> SoundNote | None:
        cleaned = ipa.strip().strip("/[]")
        for note in self.sounds:
            if note.ipa == cleaned:
                return note
        return None

    def pick_speaking(self, level: str, exclude: set[str], rng: random.Random) -> SpeakingTask | None:
        pool = [task for task in self.speaking_of_level(level) if task.id not in exclude]
        if not pool:
            pool = self.speaking_of_level(level)
        return rng.choice(pool) if pool else None

    def pick_writing(self, level: str, exclude: set[str], rng: random.Random) -> WritingTask | None:
        pool = [task for task in self.writing_of_level(level) if task.id not in exclude]
        if not pool:
            pool = self.writing_of_level(level)
        return rng.choice(pool) if pool else None

    def pick_listening(
        self, level: str, exclude: set[str], rng: random.Random
    ) -> ListeningTask | None:
        pool = [task for task in self.listening_of_level(level) if task.id not in exclude]
        if not pool:
            pool = self.listening_of_level(level)
        return rng.choice(pool) if pool else None

    @property
    def total_exercises(self) -> int:
        return len(self.exercises)


def _load_grammar(curriculum: Curriculum, directory: Path) -> None:
    for path in sorted(directory.glob("*.json")):
        if bank_of_stem(path.stem) is not None:
            continue
        try:
            points = parse_file(path)
        except ContentError as exc:
            curriculum.load_errors.append(str(exc))
            LOGGER.error("Пропущен файл грамматики: %s", exc)
            continue
        for point in points:
            if point.id in curriculum.points:
                curriculum.load_errors.append(f"{path.name}: дубликат пункта {point.id}")
                continue
            curriculum.points[point.id] = point
            for exercise in point.exercises:
                if exercise.id in curriculum.exercises:
                    curriculum.load_errors.append(
                        f"{path.name}: дубликат упражнения {exercise.id}"
                    )
                    continue
                curriculum.exercises[exercise.id] = (exercise, point.id)


def _load_banks(curriculum: Curriculum, directory: Path) -> None:
    # Слово вводится один раз, на самом низком уровне, где встретилось: иначе
    # `vocab_upto` вернул бы его дважды и повторение задвоилось бы.
    introduced: set[str] = set()
    for level in LEVELS:
        path = directory / f"vocabulary_{level.lower()}.json"
        if not path.exists():
            continue
        try:
            items = parse_bank(path, "vocabulary")
        except ContentError as exc:
            curriculum.load_errors.append(str(exc))
            LOGGER.error("Пропущен банк лексики: %s", exc)
            continue
        unique: list[VocabItem] = []
        for item in items:  # type: ignore[assignment]
            word = item.word.strip().lower()  # type: ignore[attr-defined]
            if word in introduced:
                continue
            introduced.add(word)
            unique.append(item)  # type: ignore[arg-type]
        curriculum.vocabulary[level] = unique

    simple: dict[str, str] = {
        "speaking_tasks": "speaking",
        "writing_tasks": "writing",
        "listening_tasks": "listening",
    }
    for bank, attribute in simple.items():
        path = directory / BANK_FILES[bank]
        if not path.exists():
            continue
        try:
            rows = parse_bank(path, bank)
        except ContentError as exc:
            curriculum.load_errors.append(str(exc))
            LOGGER.error("Пропущен банк %s: %s", bank, exc)
            continue
        grouped: dict[str, list] = defaultdict(list)
        for row in rows:
            grouped[getattr(row, "level")].append(row)
        setattr(curriculum, attribute, dict(grouped))

    path = directory / BANK_FILES["error_patterns"]
    if path.exists():
        try:
            rows = parse_bank(path, "error_patterns")
            curriculum.error_patterns = {getattr(row, "id"): row for row in rows}  # type: ignore[misc]
        except ContentError as exc:
            curriculum.load_errors.append(str(exc))
            LOGGER.error("Пропущен банк ошибок: %s", exc)

    path = directory / BANK_FILES["sounds"]
    if path.exists():
        try:
            curriculum.sounds = list(parse_bank(path, "sounds"))  # type: ignore[arg-type]
        except ContentError as exc:
            curriculum.load_errors.append(str(exc))
            LOGGER.error("Пропущен банк звуков: %s", exc)


def load_curriculum(directory: Path | None = None) -> Curriculum:
    """Читает весь контент с диска. Без кэша — для тестов и перезагрузки."""
    target = directory or DATA_DIR
    curriculum = Curriculum()
    if not target.exists():
        LOGGER.warning("Каталог контента не найден: %s", target)
        return curriculum
    _load_grammar(curriculum, target)
    _load_banks(curriculum, target)
    LOGGER.info(
        "Контент загружен: %d пунктов, %d упражнений, %d слов, ошибок загрузки %d",
        len(curriculum.points),
        len(curriculum.exercises),
        sum(len(rows) for rows in curriculum.vocabulary.values()),
        len(curriculum.load_errors),
    )
    return curriculum


@lru_cache(maxsize=1)
def curriculum() -> Curriculum:
    """Кэшированный реестр для рантайма бота."""
    return load_curriculum()
