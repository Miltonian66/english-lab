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
from ..content.schema import LEVEL_ORDER, Exercise, GrammarPoint
from .answers import display_options, grade, normalize, parse_choice


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


def _unseen_first(exercises: list[Exercise], seen: set[str]) -> list[Exercise]:
    """Невиденные задания вперёд, порядок внутри групп сохраняется."""
    return sorted(exercises, key=lambda exercise: f"ex:{exercise.id}" in seen)


def queue_for_topic(
    curriculum: Curriculum,
    level: str,
    topic: str,
    rng: random.Random,
    length: int = 12,
    seen: set[str] | None = None,
) -> list[str]:
    points = curriculum.points_of_topic(level, topic)
    if not points:
        return []
    per_point = max(1, length // max(1, len(points)))
    refs: list[str] = []
    for point in points:
        ordered = _unseen_first(_shuffled_by_difficulty(list(point.exercises), rng), seen or set())
        chosen = sorted(ordered[:per_point], key=lambda exercise: exercise.difficulty)
        refs.extend(f"ex:{exercise.id}" for exercise in chosen)
    rng.shuffle(refs)
    return refs[:length]


def queue_for_level(
    curriculum: Curriculum,
    level: str,
    rng: random.Random,
    weak_points: list[str] | None = None,
    length: int = DEFAULT_LENGTH,
    seen: set[str] | None = None,
) -> list[str]:
    """Смешанная тренировка уровня с перевесом в сторону слабых пунктов.

    Невиденное идёт первым: и пункты, где остались новые задания, и сами задания
    внутри пункта. Случайный выбор без истории уже к десятому дню занятий давал
    треть повторов, хотя банк уровня не был пройден и наполовину.
    """
    points = curriculum.points_of_level(level)
    if not points:
        return []
    seen = seen or set()
    weak = set(weak_points or [])
    weighted: list[GrammarPoint] = []
    for point in points:
        weighted.append(point)
        if point.id in weak:
            weighted.extend([point, point])
    rng.shuffle(weighted)
    weighted.sort(key=lambda point: all(f"ex:{e.id}" in seen for e in point.exercises))

    refs: list[str] = []
    used: set[str] = set()
    for point in weighted:
        pool = [exercise for exercise in point.exercises if exercise.id not in used]
        if not pool:
            continue
        fresh = [exercise for exercise in pool if f"ex:{exercise.id}" not in seen]
        exercise = rng.choice(fresh or pool)
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
    curriculum: Curriculum,
    cards: list[Any],
    rng: random.Random,
    length: int = 15,
    level: str = "",
    seen: set[str] | None = None,
) -> list[str]:
    """Очередь из карточек, у которых подошёл срок повторения.

    Карточка правила повторяется новым заданием этого правила, пока они есть:
    повторение проверяет правило, а не память о конкретной фразе.

    Материал выше уровня ученика не поднимается: после исправления уровня в
    очереди оставались карточки прежнего, и «Повторение» превращалось в чужой
    курс.
    """
    ceiling = LEVEL_ORDER.get(level) if level else None
    refs: list[str] = []
    for card in cards:
        if card.card_type == "vocab":
            refs.append(vocab_ref(card.card_key, rng))
        elif card.card_type == "point":
            point = curriculum.point(card.card_key)
            if point is None or not point.exercises:
                continue
            if ceiling is not None and LEVEL_ORDER.get(point.level, 0) > ceiling:
                continue
            pool = list(point.exercises)
            fresh = [exercise for exercise in pool if f"ex:{exercise.id}" not in (seen or set())]
            exercise = rng.choice(fresh or pool)
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
        # Варианты раскладываются в порядке показа один раз здесь: дальше и
        # клавиатура, и разбор ответа работают с этим порядком.
        shown = tuple(option for _, option in display_options(exercise))
        return Question(
            ref=ref,
            kind=exercise.kind,
            prompt=exercise.prompt,
            options=shown,
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


def senses(translation_ru: str) -> set[str]:
    """Значения русского перевода: «стол, письменный стол» → {стол, письменный стол}."""
    plain = re.sub(r"\([^)]*\)", "", translation_ru.lower())
    return {part.strip() for part in re.split(r"[,;/]", plain) if part.strip()}


def _all_vocab(curriculum: Curriculum) -> list[VocabItem]:
    return [item for items in curriculum.vocabulary.values() for item in items]


def _covers(asked: set[str], offered: set[str]) -> bool:
    """Подходит ли слово с переводом `offered` на вопрос с переводом `asked`.

    Только точное совпадение значения. Более узкое слово не годится: на «боль»
    headache («головная боль») — уже не ответ. Где русский перевод честно
    допускает другое слово (стол — table и desk), его перечисляет `accept`.
    """
    return bool(asked & offered)


def accepted_forms(item: VocabItem) -> list[str]:
    """Формы, которые засчитываются во вспоминании: глаголу — времена, существительному — число."""
    base = item.word.strip().lower()
    if item.pos in ("verb", "phrasal verb"):
        return [form for form in word_forms(base) if form != base]
    if item.pos == "noun" and " " not in base:
        if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
            return [f"{base[:-1]}ies"]
        return [f"{base}es" if re.search(r"(s|x|z|ch|sh)$", base) else f"{base}s"]
    return []


def vocab_alternatives(item: VocabItem, curriculum: Curriculum) -> tuple[str, ...]:
    """Что ещё засчитать во вспоминании, кроме заголовочного слова.

    Перевод «стол» честно допускает и table, и desk: ученик, который знает оба,
    не должен получать «Мимо». Засчитываются синонимы из `accept`, слова банка той
    же части речи с тем же или более узким значением и формы самого слова — с
    пометкой, какое слово было загадано.
    """
    mine = senses(item.translation_ru)
    found: list[str] = list(item.accept)
    found.extend(accepted_forms(item))
    for other in _all_vocab(curriculum):
        if (
            other.id != item.id
            and other.pos == item.pos
            and _covers(mine, senses(other.translation_ru))
        ):
            found.append(other.word)
    unique: list[str] = []
    for word in found:
        if word.lower() != item.word.lower() and word.lower() not in (x.lower() for x in unique):
            unique.append(word)
    return tuple(unique)


def _distractor_pool(item: VocabItem, curriculum: Curriculum) -> list[VocabItem]:
    """Обманки того же уровня и части речи, ни одна из которых не верна сама.

    Слово с общим значением («на» — on и at) или из `accept` выбывает: иначе
    кнопка с верным ответом засчитывалась бы ошибкой. Если на уровне не хватает
    слов этой части речи, пул добирается с соседних уровней.
    """
    mine = senses(item.translation_ru)
    blocked = {word.lower() for word in item.accept} | {item.word.lower()}

    def fits(other: VocabItem) -> bool:
        theirs = senses(other.translation_ru)
        return (
            other.id != item.id
            and other.pos == item.pos
            and other.word.lower() not in blocked
            and item.word.lower() not in (word.lower() for word in other.accept)
            and not _covers(mine, theirs)
        )

    pool = [other for other in curriculum.vocab_of_level(item.level) if fits(other)]
    if len(pool) < 3:
        order = LEVEL_ORDER.get(item.level, 0)
        for level, index in sorted(LEVEL_ORDER.items(), key=lambda pair: abs(pair[1] - order)):
            if level == item.level:
                continue
            pool.extend(other for other in curriculum.vocab_of_level(level) if fits(other))
            if len(pool) >= 3:
                break
    return pool


def vocab_question(
    item: VocabItem, curriculum: Curriculum, rng: random.Random, recognise: bool = False
) -> Question:
    """Лексику спрашиваем в обе стороны: узнавание и активное вспоминание.

    Дистракторы и их порядок берутся из генератора, засеянного самой ссылкой, а не
    из общего `rng`. Иначе один и тот же вопрос при показе и при проверке ответа
    получал бы разный порядок вариантов, и нажатая буква оценивала бы чужой вариант.
    """
    pool = _distractor_pool(item, curriculum)
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

    masked = mask_word(item.example_en, item.word, blank="___")
    prompt = f"Как по-английски «{item.translation_ru}»? ({item.pos})"
    if masked != item.example_en:
        prompt += f"\nПодсказка: {masked}"
    return Question(
        ref=f"vocab:p:{item.id}",
        kind="gap",
        prompt=prompt,
        options=(),
        expected=(item.word, *vocab_alternatives(item, curriculum)),
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


def _plain(text: str) -> str:
    """Текст для сравнения фраз: без регистра, пунктуации и сокращений."""
    return " ".join(re.findall(r"[a-z0-9']+", normalize(text).replace("’", "'")))


def revealing_texts(question: "Question") -> tuple[str, ...]:
    """Фразы, которые выдали бы ответ на текущее задание: само верное предложение.

    Для пропуска и выбора — условие с подставленным ответом и без подсказки в
    скобках, для остальных видов — эталон и допустимые варианты.
    """
    if question.card_type != "point":
        return ()
    found: list[str] = []
    for answer in question.expected:
        if "___" in question.prompt:
            filled = re.sub(r"___(?:\s*___)*", answer, question.prompt, count=1)
            filled = re.sub(r"\s*\([^)]*\)", "", filled)
            found.extend(line for line in filled.split("\n") if answer.lower() in line.lower())
        else:
            found.append(answer)
    return tuple(_plain(text) for text in found if len(_plain(text).split()) >= 3)


def _reveals(text: str, hidden: tuple[str, ...]) -> bool:
    plain = _plain(text)
    return any(secret in plain or (len(plain.split()) >= 3 and plain in secret) for secret in hidden)


_ENGLISH_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’\- ,]*[A-Za-z0-9][.!?]?")


def _mask_secrets(text: str, hidden: tuple[str, ...]) -> str:
    """Заменяет многоточием английские фразы, совпавшие с ответом; русский текст правила остаётся."""
    if not hidden:
        return text

    def replace(found: re.Match[str]) -> str:
        chunk = found.group(0)
        return "…" if len(_plain(chunk).split()) >= 3 and _reveals(chunk, hidden) else chunk

    return _ENGLISH_RUN.sub(replace, text)


def point_help(point: GrammarPoint, limit_examples: int = 5, hide: tuple[str, ...] = ()) -> str:
    """Разбор правила для подсказки — как вкладка Explanation на test-english.com.

    Подсказка обязана учить, а не сдавать ответ: сужение вариантов («точно не A»)
    и первые буквы ответа ничего не объясняют и на следующем таком же задании не
    помогут. Здесь ученик получает само правило, его формы и разобранные примеры.
    """
    # Карточка открывается, пока вопрос не решён: фраза-ответ прячется везде —
    # в объяснении, схемах, примерах и ловушке, а само правило остаётся.
    lines = [f"💡 {point.title_ru}", "", _mask_secrets(point.summary_ru, hide)]
    forms = [_mask_secrets(form, hide) for form in point.forms]
    examples = [example for example in point.examples if not _reveals(example, hide)]
    if forms:
        lines.append("")
        lines.append("Как строится:")
        lines.extend(f"• {form}" for form in forms)
    if examples:
        lines.append("")
        lines.append("Примеры:")
        lines.extend(f"• {example}" for example in examples[:limit_examples])
    if point.ru_interference:
        lines.append("")
        lines.append(f"⚠️ Ловушка для русскоязычных: {_mask_secrets(point.ru_interference, hide)}")
    return "\n".join(lines)


BLANK = "…"

# Неправильные глаголы из банков лексики: без них «went» выдавало бы «go» в
# подсказке, а во вспоминании «came up with» не засчитывалось бы формой слова.
IRREGULAR: dict[str, tuple[str, ...]] = {
    "arise": ("arose", "arisen"), "be": ("am", "is", "are", "was", "were", "been", "being"),
    "bear": ("bore", "borne"), "become": ("became",), "begin": ("began", "begun"),
    "bend": ("bent",), "bind": ("bound",), "bite": ("bit", "bitten"), "blow": ("blew", "blown"),
    "break": ("broke", "broken"), "breed": ("bred",), "bring": ("brought",), "build": ("built",),
    "buy": ("bought",), "catch": ("caught",), "choose": ("chose", "chosen"), "cling": ("clung",),
    "come": ("came",), "creep": ("crept",), "deal": ("dealt",), "dig": ("dug",),
    "do": ("did", "done", "does"), "draw": ("drew", "drawn"), "drink": ("drank", "drunk"),
    "drive": ("drove", "driven"), "eat": ("ate", "eaten"), "fall": ("fell", "fallen"),
    "feed": ("fed",), "feel": ("felt",), "fight": ("fought",), "find": ("found",),
    "flee": ("fled",), "fly": ("flew", "flown", "flies"), "forbid": ("forbade", "forbidden"),
    "foresee": ("foresaw", "foreseen"), "forget": ("forgot", "forgotten"),
    "forgive": ("forgave", "forgiven"), "freeze": ("froze", "frozen"),
    "get": ("got", "gotten"), "give": ("gave", "given"), "go": ("went", "gone", "goes"),
    "grow": ("grew", "grown"), "hang": ("hung",), "have": ("had", "has"), "hear": ("heard",),
    "hide": ("hid", "hidden"), "hold": ("held",), "keep": ("kept",), "know": ("knew", "known"),
    "lay": ("laid",), "lead": ("led",), "leave": ("left",), "lend": ("lent",), "lie": ("lay", "lain"),
    "lose": ("lost",), "make": ("made",), "mean": ("meant",), "meet": ("met",),
    "overcome": ("overcame",), "pay": ("paid",), "ride": ("rode", "ridden"), "ring": ("rang", "rung"),
    "rise": ("rose", "risen"), "run": ("ran",), "say": ("said",), "see": ("saw", "seen"),
    "seek": ("sought",), "sell": ("sold",), "send": ("sent",), "shake": ("shook", "shaken"),
    "shine": ("shone",), "shoot": ("shot",), "show": ("shown",), "sing": ("sang", "sung"),
    "sit": ("sat",), "sleep": ("slept",), "speak": ("spoke", "spoken"), "spend": ("spent",),
    "stand": ("stood",), "steal": ("stole", "stolen"), "stick": ("stuck",), "strike": ("struck",),
    "strive": ("strove", "striven"), "swear": ("swore", "sworn"), "sweep": ("swept",),
    "swim": ("swam", "swum"), "take": ("took", "taken"), "teach": ("taught",),
    "tear": ("tore", "torn"), "tell": ("told",), "think": ("thought",), "throw": ("threw", "thrown"),
    "undertake": ("undertook", "undertaken"), "understand": ("understood",),
    "wake": ("woke", "woken"), "wear": ("wore", "worn"), "weep": ("wept",), "win": ("won",),
    "wind": ("wound",), "withdraw": ("withdrew", "withdrawn"), "write": ("wrote", "written"),
}


# Двусложные с ударением на последнем слоге удваивают согласную, как односложные.
DOUBLING = frozenset({
    "admit", "commit", "submit", "permit", "omit", "emit", "refer", "prefer", "occur",
    "deter", "regret", "control", "compel", "expel", "propel", "rebel", "equip", "upset",
})


def _single_forms(base: str) -> set[str]:
    """Настоящие формы одного слова: без «comeed» и «stoped».

    Формы идут и в маску подсказки, и в засчитанные ответы вспоминания, поэтому
    выдуманная форма здесь означала бы засчитанную орфографическую ошибку.
    """
    forms = {base}
    vowels = len(re.findall(r"[aeiouy]+", base))
    if base.endswith("ee") or base == "be":
        forms |= {f"{base}s", f"{base}d", f"{base}ing"}
    elif base.endswith("e"):
        forms |= {f"{base}s", f"{base}d", f"{base[:-1]}ing"}
    elif base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
        forms |= {f"{base[:-1]}ies", f"{base[:-1]}ied", f"{base}ing"}
    elif (vowels == 1 or base in DOUBLING) and re.fullmatch(r"[a-z]*[^aeiou][aeiou][bdgklmnprt]", base):
        forms |= {f"{base}s", f"{base}{base[-1]}ed", f"{base}{base[-1]}ing"}
    else:
        plural = f"{base}es" if re.search(r"(s|x|z|ch|sh|o)$", base) else f"{base}s"
        forms |= {plural, f"{base}ed", f"{base}ing"}
    if base in IRREGULAR:
        # Прошедшее у неправильных глаголов своё: «comed» и «maked» не формы.
        forms = {form for form in forms if not form.endswith("ed") and form != f"{base}d"}
        forms |= set(IRREGULAR[base])
        if base in {"be", "have", "do", "go"}:
            forms -= {f"{base}s", f"{base}es"}
    return forms


def word_forms(word: str) -> list[str]:
    """Слово и его частотные формы — чтобы «schedule» не утекло как «scheduled».

    У фразового глагола меняется только первое слово: came up with, checked in.
    """
    base = word.strip().lower()
    head, _, tail = base.partition(" ")
    forms = {f"{form} {tail}".strip() for form in _single_forms(head)} if tail else _single_forms(base)
    forms.add(base)
    return sorted(forms, key=len, reverse=True)


def mask_word(text: str, word: str, blank: str = BLANK) -> str:
    """Прячет слово во всех формах, но только целиком: «go» не должно съесть «good»."""
    masked = text
    for form in word_forms(word):
        masked = re.sub(rf"\b{re.escape(form)}\b", blank, masked, flags=re.IGNORECASE)
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
    if point is None:
        return "По этой теме справки нет."
    return point_help(point, hide=revealing_texts(question))


# ── проверка ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    correct: bool
    understood: bool
    selected_index: int | None
    expected_text: str
    # Пояснение к засчитанному ответу: опечатка, фрагмент вместо предложения,
    # другое слово с тем же значением.
    note: str = ""


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
            shuffle=False,
        )
        if index is None:
            return Verdict(False, False, None, expected_text)
        chosen = question.options[index]
        return Verdict(
            normalize(chosen) == normalize(expected_text), True, index, expected_text
        )

    # Ссылка на упражнение несёт его id: по нему сверка узнаёт пункт про
    # пунктуацию, где запятая — предмет задания.
    probe = Exercise(
        id=question.ref.partition(":")[2] or question.ref, kind=question.kind,
        prompt=question.prompt, explanation_ru="",
        answer=question.expected[0] if question.expected else "",
        accept=tuple(question.expected[1:]),
    )
    result = grade(probe, text)
    note = result.note
    if (
        result.correct
        and question.card_type == "vocab"
        and normalize(result.matched) != normalize(expected_text)
    ):
        note = f"Засчитано. Загадано слово: {expected_text}."
    return Verdict(result.correct, True, None, expected_text, note)


def _expected_index(question: Question) -> int | None:
    if not question.expected or not question.options:
        return None
    target = normalize(question.expected[0])
    for index, option in enumerate(question.options):
        if normalize(option) == target:
            return index
    return None
