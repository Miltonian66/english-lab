"""Адаптивная диагностика уровня A1–C2.

Отдельного банка вопросов нет: диагностика берёт задания из того же курса, что и
практика, поэтому растёт вместе с контентом. Логика — лестница по уровням, как в
адаптивных тестах: блок заданий, решение подняться или опуститься, остановка на
уровне, где ученик перестаёт справляться.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..content.registry import Curriculum
from ..content.schema import LEVEL_ORDER, LEVELS, Exercise, GrammarPoint


BLOCK_SIZE = 4
MAX_ITEMS = 28
START_LEVEL = "A2"
UP_THRESHOLD = 0.75
DOWN_THRESHOLD = 0.34
PASS_THRESHOLD = 0.6


@dataclass(frozen=True)
class ProfileQuestion:
    id: str
    prompt: str
    options: tuple[str, ...]


PROFILE_QUESTIONS: tuple[ProfileQuestion, ...] = (
    ProfileQuestion(
        "profile_goal",
        "Зачем тебе английский в первую очередь?",
        ("работа и собеседования", "общение с коллегами и заказчиками",
         "чтение и контент", "экзамен или переезд"),
    ),
    ProfileQuestion(
        "profile_time",
        "Сколько времени в неделю реально готов уделять?",
        ("меньше 2 часов", "2–4 часа", "5–7 часов", "8+ часов"),
    ),
    ProfileQuestion(
        "profile_self",
        "Как сам оцениваешь свой уровень?",
        ("почти с нуля", "читаю, но говорю плохо", "middle, хочу увереннее",
         "свободно, шлифую детали"),
    ),
)

SELF_START: dict[int, str] = {0: "A1", 1: "A2", 2: "B1", 3: "B2"}


@dataclass
class PlacementResult:
    level: str
    per_level: dict[str, tuple[int, int]]
    weak_topics: list[str]
    strong_topics: list[str]
    asked: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.asked if self.asked else 0.0


@dataclass
class PlacementState:
    session_id: int
    level: str = START_LEVEL
    asked: list[str] = field(default_factory=list)
    results: dict[str, list[int]] = field(default_factory=dict)
    visited: list[str] = field(default_factory=list)
    current: str = ""
    profile: dict[str, str] = field(default_factory=dict)
    profile_index: int = 0
    wrong_topics: list[str] = field(default_factory=list)
    right_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "level": self.level,
            "asked": self.asked,
            "results": self.results,
            "visited": self.visited,
            "current": self.current,
            "profile": self.profile,
            "profile_index": self.profile_index,
            "wrong_topics": self.wrong_topics,
            "right_topics": self.right_topics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlacementState":
        return cls(
            session_id=int(data.get("session_id") or 1),
            level=str(data.get("level") or START_LEVEL),
            asked=list(data.get("asked") or []),
            results={str(k): list(v) for k, v in (data.get("results") or {}).items()},
            visited=list(data.get("visited") or []),
            current=str(data.get("current") or ""),
            profile=dict(data.get("profile") or {}),
            profile_index=int(data.get("profile_index") or 0),
            wrong_topics=list(data.get("wrong_topics") or []),
            right_topics=list(data.get("right_topics") or []),
        )

    @property
    def total_asked(self) -> int:
        return len(self.asked)


def _pool(curriculum: Curriculum, level: str) -> list[tuple[Exercise, GrammarPoint]]:
    """Задания уровня, пригодные для диагностики: только выбор из вариантов."""
    rows: list[tuple[Exercise, GrammarPoint]] = []
    for point in curriculum.points_of_level(level):
        for exercise in point.exercises:
            if exercise.kind == "choice" and exercise.difficulty <= 2:
                rows.append((exercise, point))
    return rows


def available_levels(curriculum: Curriculum) -> list[str]:
    return [level for level in LEVELS if _pool(curriculum, level)]


def start_level_from_profile(profile: dict[str, str]) -> str:
    """Самооценка задаёт стартовую ступень лестницы; без неё берётся START_LEVEL."""
    raw = profile.get("profile_self_index")
    if raw is None or raw == "":
        return START_LEVEL
    try:
        return SELF_START.get(int(raw), START_LEVEL)
    except (TypeError, ValueError):
        return START_LEVEL


def next_item(
    state: PlacementState, curriculum: Curriculum, rng: random.Random
) -> tuple[Exercise, GrammarPoint] | None:
    """Следующее задание текущего уровня, ещё не показанное ученику."""
    levels = available_levels(curriculum)
    if not levels:
        return None
    if state.level not in levels:
        state.level = min(levels, key=lambda name: abs(LEVEL_ORDER[name] - LEVEL_ORDER[state.level]))
    seen = set(state.asked)
    pool = [row for row in _pool(curriculum, state.level) if row[0].id not in seen]
    if not pool:
        return None
    # Разные пункты в блоке важнее разных заданий одного пункта.
    used_points = {
        curriculum.exercise(item)[1].id  # type: ignore[index]
        for item in state.asked
        if curriculum.exercise(item)
    }
    fresh = [row for row in pool if row[1].id not in used_points]
    return rng.choice(fresh or pool)


def record(state: PlacementState, exercise: Exercise, point: GrammarPoint, correct: bool) -> None:
    state.asked.append(exercise.id)
    state.results.setdefault(state.level, []).append(int(correct))
    if state.level not in state.visited:
        state.visited.append(state.level)
    if correct:
        state.right_topics.append(point.topic)
    else:
        state.wrong_topics.append(point.topic)


def advance(state: PlacementState, curriculum: Curriculum) -> bool:
    """После полного блока решает, куда идти. False — диагностика окончена."""
    levels = available_levels(curriculum)
    if not levels or state.total_asked >= MAX_ITEMS:
        return False
    block = state.results.get(state.level, [])
    if len(block) < BLOCK_SIZE:
        return True

    accuracy = sum(block) / len(block)
    index = levels.index(state.level) if state.level in levels else 0

    if accuracy >= UP_THRESHOLD:
        if index + 1 >= len(levels):
            return False
        target = levels[index + 1]
        if target in state.visited:
            return False
        state.level = target
        return True

    if accuracy <= DOWN_THRESHOLD:
        if index == 0:
            return False
        target = levels[index - 1]
        if target in state.visited:
            return False
        state.level = target
        return True

    return False


def finish(state: PlacementState, curriculum: Curriculum) -> PlacementResult:
    levels = available_levels(curriculum) or list(LEVELS)
    per_level = {
        level: (sum(values), len(values)) for level, values in state.results.items() if values
    }

    passed = [
        level
        for level, (correct, total) in per_level.items()
        if total >= 2 and correct / total >= PASS_THRESHOLD
    ]
    if passed:
        level = max(passed, key=lambda name: LEVEL_ORDER[name])
    elif per_level:
        lowest_tested = min(per_level, key=lambda name: LEVEL_ORDER[name])
        index = levels.index(lowest_tested) if lowest_tested in levels else 0
        level = levels[max(0, index - 1)] if index > 0 else levels[0]
    else:
        level = START_LEVEL

    weak = _top_topics(state.wrong_topics, state.right_topics)
    strong = _top_topics(state.right_topics, state.wrong_topics)
    asked = state.total_asked
    correct = sum(sum(values) for values in state.results.values())
    return PlacementResult(
        level=level,
        per_level=per_level,
        weak_topics=weak,
        strong_topics=strong,
        asked=asked,
        correct=correct,
    )


def _top_topics(primary: list[str], secondary: list[str], limit: int = 4) -> list[str]:
    counts: dict[str, int] = {}
    for topic in primary:
        counts[topic] = counts.get(topic, 0) + 1
    for topic in secondary:
        counts[topic] = counts.get(topic, 0) - 1
    ranked = [topic for topic, score in sorted(counts.items(), key=lambda item: -item[1]) if score > 0]
    return ranked[:limit]


def next_level(level: str) -> str:
    index = LEVEL_ORDER.get(level, 2)
    return LEVELS[min(index + 1, len(LEVELS) - 1)]
