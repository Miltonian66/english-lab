"""Сессия практики: сбор очереди, адаптивная сложность и проверка ответов.

Адаптивность работает как в `fluent`: цель — удержать долю верных ответов в
коридоре 60–70 %. Слишком легко — поднимаем сложность оставшихся заданий, слишком
тяжело — опускаем. Очередь смешивает пункты, чтобы ученик различал похожие
правила, а не задалбливал один шаблон.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

from ..content.banks import VocabItem
from ..content.registry import Curriculum
from ..content.schema import Exercise, GrammarPoint
from .answers import matches, normalize, parse_choice


TARGET_LOW = 0.6
TARGET_HIGH = 0.75
DEFAULT_LENGTH = 10


@dataclass(frozen=True)
class Question:
    """Единица практики: обычное упражнение курса или карточка лексики."""

    ref: str
    kind: str
    prompt: str
    options: tuple[str, ...]
    expected: tuple[str, ...]
    explanation_ru: str
    difficulty: int
    point_id: str
    level: str
    title_ru: str
    topic: str
    card_type: str
    card_key: str

    @property
    def is_choice(self) -> bool:
        return bool(self.options)


@dataclass
class PracticeState:
    kind: str
    subject: str
    queue: list[str] = field(default_factory=list)
    index: int = 0
    correct: int = 0
    answered: int = 0
    session_id: int = 0
    level: str = ""
    helped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "subject": self.subject, "queue": self.queue,
            "index": self.index, "correct": self.correct, "answered": self.answered,
            "session_id": self.session_id, "level": self.level, "helped": self.helped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PracticeState":
        return cls(
            kind=str(data.get("kind") or "mixed"),
            subject=str(data.get("subject") or ""),
            queue=list(data.get("queue") or []),
            index=int(data.get("index") or 0),
            correct=int(data.get("correct") or 0),
            answered=int(data.get("answered") or 0),
            session_id=int(data.get("session_id") or 0),
            level=str(data.get("level") or ""),
            helped=bool(data.get("helped")),
        )

    @property
    def finished(self) -> bool:
        return self.index >= len(self.queue)

    @property
    def remaining(self) -> int:
        return max(0, len(self.queue) - self.index)

    @property
    def accuracy(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    def current_ref(self) -> str | None:
        return self.queue[self.index] if self.index < len(self.queue) else None


# ── сборка очередей ──────────────────────────────────────────────


def _shuffled_by_difficulty(
    exercises: list[Exercise], rng: random.Random, ascending: bool = True
) -> list[Exercise]:
    buckets: dict[int, list[Exercise]] = {}
    for exercise in exercises:
        buckets.setdefault(exercise.difficulty, []).append(exercise)
    order = sorted(buckets, reverse=not ascending)
    result: list[Exercise] = []
    for level in order:
        bucket = buckets[level][:]
        rng.shuffle(bucket)
        result.extend(bucket)
    return result


def queue_for_point(point: GrammarPoint, rng: random.Random) -> list[str]:
    return [f"ex:{exercise.id}" for exercise in _shuffled_by_difficulty(list(point.exercises), rng)]


def queue_for_topic(
    curriculum: Curriculum, level: str, topic: str, rng: random.Random, length: int = 12
) -> list[str]:
    points = curriculum.points_of_topic(level, topic)
    if not points:
        return []
    per_point = max(1, length // max(1, len(points)))
    refs: list[str] = []
    for point in points:
        chosen = _shuffled_by_difficulty(list(point.exercises), rng)[:per_point]
        refs.extend(f"ex:{exercise.id}" for exercise in chosen)
    rng.shuffle(refs)
    return refs[:length]


def queue_for_level(
    curriculum: Curriculum,
    level: str,
    rng: random.Random,
    weak_points: list[str] | None = None,
    length: int = DEFAULT_LENGTH,
) -> list[str]:
    """Смешанная тренировка уровня с перевесом в сторону слабых пунктов."""
    points = curriculum.points_of_level(level)
    if not points:
        return []
    weak = set(weak_points or [])
    weighted: list[GrammarPoint] = []
    for point in points:
        weighted.append(point)
        if point.id in weak:
            weighted.extend([point, point])
    rng.shuffle(weighted)

    refs: list[str] = []
    used: set[str] = set()
    for point in weighted:
        pool = [exercise for exercise in point.exercises if exercise.id not in used]
        if not pool:
            continue
        exercise = rng.choice(pool)
        used.add(exercise.id)
        refs.append(f"ex:{exercise.id}")
        if len(refs) >= length:
            break
    return refs


RECOGNISE_SHARE = 0.4


def vocab_ref(vocab_id: str, rng: random.Random) -> str:
    """Режим спрашивания фиксируется в ссылке, а не выбирается при каждом разборе.

    Иначе показанный вопрос и проверяемый расходились бы: показали выбор варианта,
    а при нажатии кнопки задание оказалось бы со свободным вводом.
    """
    mode = "r" if rng.random() < RECOGNISE_SHARE else "p"
    return f"vocab:{mode}:{vocab_id}"


def queue_of_vocab(
    curriculum: Curriculum,
    level: str,
    seen_keys: set[str],
    rng: random.Random,
    length: int,
) -> list[str]:
    """Новые слова уровня и ниже. Именно отсюда рождаются карточки лексики:
    без этого банк слов был бы недостижим, а `/review` показывал бы только грамматику."""
    if length <= 0:
        return []
    pool = [item for item in curriculum.vocab_upto(level) if item.id not in seen_keys]
    if not pool:
        pool = curriculum.vocab_upto(level)
    if not pool:
        return []
    picks = rng.sample(pool, k=min(length, len(pool)))
    return [vocab_ref(item.id, rng) for item in picks]


def queue_for_review(
    curriculum: Curriculum, cards: list[Any], rng: random.Random, length: int = 15
) -> list[str]:
    """Очередь из карточек, у которых подошёл срок повторения."""
    refs: list[str] = []
    for card in cards:
        if card.card_type == "vocab":
            refs.append(vocab_ref(card.card_key, rng))
        elif card.card_type == "point":
            point = curriculum.point(card.card_key)
            if point and point.exercises:
                exercise = rng.choice(list(point.exercises))
                refs.append(f"ex:{exercise.id}")
        if len(refs) >= length:
            break
    return refs


# ── адаптивность ─────────────────────────────────────────────────


def adapt(state: PracticeState, curriculum: Curriculum, rng: random.Random) -> None:
    """Пересобирает хвост очереди под текущую успешность ученика."""
    if state.answered < 3 or state.remaining < 2:
        return
    accuracy = state.accuracy
    if TARGET_LOW <= accuracy <= TARGET_HIGH:
        return
    harder = accuracy > TARGET_HIGH

    tail = state.queue[state.index :]
    resolved: list[tuple[str, int]] = []
    for ref in tail:
        question = resolve(ref, curriculum, rng)
        resolved.append((ref, question.difficulty if question else 2))
    resolved.sort(key=lambda item: item[1], reverse=harder)
    state.queue = state.queue[: state.index] + [ref for ref, _ in resolved]


# ── разрешение ссылок в вопросы ──────────────────────────────────


def resolve(ref: str, curriculum: Curriculum, rng: random.Random) -> Question | None:
    prefix, _, key = ref.partition(":")
    if prefix == "ex":
        found = curriculum.exercise(key)
        if not found:
            return None
        exercise, point = found
        return Question(
            ref=ref,
            kind=exercise.kind,
            prompt=exercise.prompt,
            options=exercise.options,
            expected=exercise.expected,
            explanation_ru=exercise.explanation_ru,
            difficulty=exercise.difficulty,
            point_id=point.id,
            level=point.level,
            title_ru=point.title_ru,
            topic=point.topic,
            card_type="point",
            card_key=point.id,
        )
    if prefix == "vocab":
        mode, _, vocab_id = key.partition(":")
        if not vocab_id:  # старая ссылка без режима
            mode, vocab_id = "p", key
        item = _find_vocab(curriculum, vocab_id)
        if item is None:
            return None
        return vocab_question(item, curriculum, rng, recognise=mode == "r")
    return None


def _find_vocab(curriculum: Curriculum, vocab_id: str) -> VocabItem | None:
    for items in curriculum.vocabulary.values():
        for item in items:
            if item.id == vocab_id:
                return item
    return None


def vocab_question(
    item: VocabItem, curriculum: Curriculum, rng: random.Random, recognise: bool = False
) -> Question:
    """Лексику спрашиваем в обе стороны: узнавание и активное вспоминание.

    Дистракторы и их порядок берутся из генератора, засеянного самой ссылкой, а не
    из общего `rng`. Иначе один и тот же вопрос при показе и при проверке ответа
    получал бы разный порядок вариантов, и нажатая буква оценивала бы чужой вариант.
    """
    pool = [
        other
        for other in curriculum.vocab_of_level(item.level)
        if other.id != item.id and other.pos == item.pos
    ]
    if recognise and pool:
        seed = int(hashlib.sha1(f"vocab:r:{item.id}".encode()).hexdigest()[:12], 16)
        stable = random.Random(seed)
        distractors = stable.sample(pool, k=min(3, len(pool)))
        options = [item.word] + [other.word for other in distractors]
        stable.shuffle(options)
        return Question(
            ref=f"vocab:r:{item.id}",
            kind="choice",
            prompt=f"Какое слово значит «{item.translation_ru}»?",
            options=tuple(options),
            expected=(item.word,),
            explanation_ru=f"{item.word} {item.ipa_us} — {item.translation_ru}. {item.example_en}",
            difficulty=1,
            point_id="",
            level=item.level,
            title_ru="Лексика",
            topic="Лексика",
            card_type="vocab",
            card_key=item.id,
        )

    hint = item.example_en.replace(item.word, "___") if item.word in item.example_en else ""
    prompt = f"Как по-английски «{item.translation_ru}»? ({item.pos})"
    if hint:
        prompt += f"\nПодсказка: {hint}"
    return Question(
        ref=f"vocab:p:{item.id}",
        kind="gap",
        prompt=prompt,
        options=(),
        expected=(item.word,),
        explanation_ru=f"{item.word} {item.ipa_us} — {item.translation_ru}. {item.example_en}",
        difficulty=2,
        point_id="",
        level=item.level,
        title_ru="Лексика",
        topic="Лексика",
        card_type="vocab",
        card_key=item.id,
    )


# ── справка по теме ──────────────────────────────────────────────


def point_help(point: GrammarPoint, limit_examples: int = 5) -> str:
    """Разбор правила для подсказки — как вкладка Explanation на test-english.com.

    Подсказка обязана учить, а не сдавать ответ: сужение вариантов («точно не A»)
    и первые буквы ответа ничего не объясняют и на следующем таком же задании не
    помогут. Здесь ученик получает само правило, его формы и разобранные примеры.
    """
    lines = [f"💡 {point.title_ru}", "", point.summary_ru]
    if point.forms:
        lines.append("")
        lines.append("Как строится:")
        lines.extend(f"• {form}" for form in point.forms)
    if point.examples:
        lines.append("")
        lines.append("Примеры:")
        lines.extend(f"• {example}" for example in point.examples[:limit_examples])
    if point.ru_interference:
        lines.append("")
        lines.append(f"⚠️ Ловушка для русскоязычных: {point.ru_interference}")
    return "\n".join(lines)


BLANK = "…"


def word_forms(word: str) -> list[str]:
    """Слово и его частотные формы — чтобы «schedule» не утекло как «scheduled»."""
    base = word.strip().lower()
    forms = {base, f"{base}s", f"{base}es", f"{base}ed", f"{base}ing"}
    if base.endswith("e"):
        forms |= {f"{base[:-1]}ing", f"{base}d"}
    if base.endswith("y") and len(base) > 2:
        forms |= {f"{base[:-1]}ies", f"{base[:-1]}ied"}
    return sorted(forms, key=len, reverse=True)


def mask_word(text: str, word: str) -> str:
    """Прячет слово во всех формах, но только целиком: «go» не должно съесть «good»."""
    masked = text
    for form in word_forms(word):
        masked = re.sub(rf"\b{re.escape(form)}\b", BLANK, masked, flags=re.IGNORECASE)
    return masked


def vocab_help(item: VocabItem) -> str:
    """Справка по слову: употребление и сочетаемость, но без самого слова."""
    lines = [f"💡 {item.translation_ru} · {item.pos} · {item.ipa_us}"]
    lines.extend(["", f"В предложении: {mask_word(item.example_en, item.word)}"])
    if item.collocations:
        hidden = [mask_word(collocation, item.word) for collocation in item.collocations[:3]]
        lines.append("Сочетается: " + ", ".join(hidden))
    return "\n".join(lines)


def task_hint(question: "Question") -> str:
    """Что именно от ученика хотят: одной строкой под условием.

    Без неё `correct` неотличим от обычного предложения — человек видит текст без
    пропуска и без вопроса и не понимает, что в нём спрятана ошибка. У `gap`
    постановка видна из самого пропуска, но только если он там есть: часть заданий
    несёт инструкцию прямо в условии, и вторая строка ей противоречила бы.
    """
    if question.kind == "correct":
        return "Здесь есть ошибка. Пришли исправленное предложение целиком."
    if question.kind == "order":
        return "Составь предложение из этих слов и пришли целиком."
    if question.kind == "transform":
        return "Пришли переписанное предложение целиком."
    if question.kind == "gap" and "___" in question.prompt:
        return "Напиши только то, что стоит вместо пропуска."
    return "Напиши ответ сообщением."


def help_for(question: "Question", curriculum: Curriculum) -> str:
    """Материал по теме текущего задания."""
    if question.card_type == "vocab":
        item = _find_vocab(curriculum, question.card_key)
        return vocab_help(item) if item else "По этому слову справки нет."
    point = curriculum.point(question.point_id)
    return point_help(point) if point else "По этой теме справки нет."


# ── проверка ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    correct: bool
    understood: bool
    selected_index: int | None
    expected_text: str


def check(question: Question, text: str) -> Verdict:
    """`understood=False` — ответ не распознан как выбор варианта, а не «неверно»."""
    expected_text = question.expected[0] if question.expected else ""
    if question.is_choice:
        index = parse_choice(
            Exercise(
                id=question.ref, kind="choice", prompt=question.prompt,
                explanation_ru="", options=question.options,
                correct_index=_expected_index(question),
            ),
            text,
        )
        if index is None:
            return Verdict(False, False, None, expected_text)
        chosen = question.options[index]
        return Verdict(
            normalize(chosen) == normalize(expected_text), True, index, expected_text
        )

    probe = Exercise(
        id=question.ref, kind=question.kind, prompt=question.prompt, explanation_ru="",
        answer=question.expected[0] if question.expected else "",
        accept=tuple(question.expected[1:]),
    )
    return Verdict(matches(probe, text), True, None, expected_text)


def _expected_index(question: Question) -> int | None:
    if not question.expected or not question.options:
        return None
    target = normalize(question.expected[0])
    for index, option in enumerate(question.options):
        if normalize(option) == target:
            return index
    return None
