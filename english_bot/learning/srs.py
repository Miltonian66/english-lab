"""Интервальное повторение по SM-2.

Алгоритм тот же, что в `fluent`: оценка 0–5, коэффициент лёгкости с нижней
границей 1.3, удвоение интервала через ease. Отличие одно — оценка выводится из
бинарного «верно/неверно» плюс контекст попытки, потому что в Telegram ученик не
ставит себе баллы вручную.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..storage import Card


MIN_EASE = 1.3
DEFAULT_EASE = 2.5
LEARNING_STEPS_DAYS: tuple[float, ...] = (1.0, 6.0)
LAPSE_INTERVAL_DAYS = 0.5


def quality(correct: bool, used_hint: bool = False, slow: bool = False) -> int:
    """Переводит исход попытки в шкалу SM-2 0–5."""
    if not correct:
        return 1 if used_hint else 2
    if used_hint:
        return 3
    return 4 if slow else 5


def new_card(card_type: str, card_key: str, now: datetime | None = None) -> Card:
    moment = now or datetime.now(UTC)
    return Card(
        card_type=card_type,
        card_key=card_key,
        ease=DEFAULT_EASE,
        interval_days=0.0,
        repetitions=0,
        lapses=0,
        due_at=moment.isoformat(timespec="seconds"),
        mastery=0,
    )


def derive_mastery(repetitions: int, lapses: int, ease: float) -> int:
    """Мастерство 0–5 в звёздах: рост от повторов, штраф за срывы и низкий ease."""
    if repetitions <= 0:
        return 0
    score = min(repetitions, 6) - min(lapses, 3)
    if ease < 2.0:
        score -= 1
    elif ease >= 2.6:
        score += 1
    return max(0, min(5, score))


def review(card: Card, grade: int, now: datetime | None = None) -> Card:
    """Возвращает новое состояние карточки после ответа."""
    moment = now or datetime.now(UTC)
    grade = max(0, min(5, grade))

    ease = card.ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ease = max(MIN_EASE, round(ease, 3))

    if grade < 3:
        repetitions = 0
        lapses = card.lapses + 1
        interval = LAPSE_INTERVAL_DAYS
    else:
        repetitions = card.repetitions + 1
        lapses = card.lapses
        if repetitions == 1:
            interval = LEARNING_STEPS_DAYS[0]
        elif repetitions == 2:
            interval = LEARNING_STEPS_DAYS[1]
        else:
            interval = round(max(card.interval_days, 1.0) * ease, 2)
        interval = min(interval, 365.0)

    due = moment + timedelta(days=interval)
    return Card(
        card_type=card.card_type,
        card_key=card.card_key,
        ease=ease,
        interval_days=interval,
        repetitions=repetitions,
        lapses=lapses,
        due_at=due.isoformat(timespec="seconds"),
        mastery=derive_mastery(repetitions, lapses, ease),
    )


def interval_note_ru(card: Card) -> str:
    days = card.interval_days
    if days < 1:
        return "вернём сегодня же"
    if days < 2:
        return "повторим завтра"
    if days < 30:
        return f"повторим через {round(days)} дн."
    return f"повторим через {round(days / 30, 1)} мес."


def stars(mastery: int) -> str:
    filled = max(0, min(5, mastery))
    return "★" * filled + "☆" * (5 - filled)
