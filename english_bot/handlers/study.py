"""Диагностика уровня, курс по темам, тренировка и повторение."""

from __future__ import annotations

import logging

from ..content.schema import LEVELS, GrammarPoint
from ..context import Context
from ..learning import placement as pl
from ..learning import practice as pr
from ..learning import srs
from ..learning.answers import display_options, labelled_options, parse_choice
from ..learning.progress import levels_overview, mastery_map, study_plan
from ..storage import User
from ..telegram_api import inline, inline_grid
from .menu import MAIN_KEYBOARD, RESUME_KEYBOARD


LOGGER = logging.getLogger(__name__)

SESSION_LENGTH = 10
REVIEW_LENGTH = 15
VOCAB_PER_SESSION = 3
STALE_STEP = "это задание уже закрыто"


def _tag(state: pr.PracticeState) -> int:
    """Короткая метка сессии в callback_data.

    Номера шага мало: кнопка «B» из третьего вопроса прошлой сессии совпадала по
    шагу с текущим и засчитывалась как ответ на другое задание.
    """
    return int(state.session_id or 0) % 1000


def _parse_step(payload: str, state: pr.PracticeState) -> tuple[int, int | None]:
    """Разбирает `tag:step[:choice]`; (-1, None) — кнопка из чужой сессии или битая."""
    parts = payload.split(":")
    if len(parts) < 2:
        return -1, None
    try:
        tag, step = int(parts[0]), int(parts[1])
    except ValueError:
        return -1, None
    if tag != _tag(state):
        return -1, None
    if len(parts) < 3:
        return step, None
    try:
        return step, int(parts[2])
    except ValueError:
        return step, None


def _session_buttons(state: pr.PracticeState) -> list[tuple[str, str]]:
    head = f"{_tag(state)}:{state.index}"
    return [("💡 Правило", f"hint:{head}"), ("Пропустить", f"skip:{head}"), ("Стоп", "endses")]


def _placement_tag(state: pl.PlacementState) -> int:
    return int(state.session_id or 0) % 1000


def _parse_placement(payload: str, state: pl.PlacementState) -> tuple[int, str]:
    """Разбирает `tag:step:choice` диагностики; (-1, "") — кнопка чужой попытки."""
    parts = payload.split(":")
    if len(parts) < 3:
        return -1, ""
    try:
        tag, step = int(parts[0]), int(parts[1])
    except ValueError:
        return -1, ""
    if tag != _placement_tag(state):
        return -1, ""
    return step, parts[2]


def _clip(text: str, limit: int) -> str:
    """Обрезает подпись кнопки по границе слова и честно ставит многоточие."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rstrip(" ,;:-—")
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-—") + "…"


# ── диагностика ──────────────────────────────────────────────────


def command_test(ctx: Context, user: User, text: str) -> None:
    if not ctx.curriculum.points:
        ctx.say(user, "Курс ещё не загружен — сообщи админу.")
        return
    session_id = ctx.storage.next_placement_session(user.user_id)
    state = pl.PlacementState(session_id=session_id)
    ctx.storage.set_state(user.user_id, "placement", state.to_dict())
    ctx.log_start(user, "placement")
    ctx.say(
        user,
        "Диагностика уровня A1–C2. Сначала три вопроса о тебе, потом задания: "
        "они подстраиваются под ответы, поэтому теста ровно столько, сколько нужно. "
        "10–15 минут.\n\n"
        "Можно отвлечься: кнопки «📚 Курс» и «📊 Я» тест не прервут, "
        "а «🎯 Продолжить» вернёт к текущему вопросу.",
        RESUME_KEYBOARD,
    )
    _ask_placement(ctx, user, state)


def resume_placement(ctx: Context, user: User) -> None:
    """Возвращает к незаконченной диагностике вместо запуска новой.

    Уровень пишется только в конце, поэтому потерянный тест — это потерянные
    10–15 минут: состояние полностью сериализуемо, и перерисовать шаг дешевле.
    """
    state = pl.PlacementState.from_dict(user.state_data)
    ctx.say(user, "Продолжаем диагностику.", RESUME_KEYBOARD)
    _ask_placement(ctx, user, state)


def _ask_placement(ctx: Context, user: User, state: pl.PlacementState) -> None:
    if state.profile_index < len(pl.PROFILE_QUESTIONS):
        question = pl.PROFILE_QUESTIONS[state.profile_index]
        tag = _placement_tag(state)
        ctx.say(
            user,
            f"{state.profile_index + 1}/3 · {question.prompt}",
            inline(
                [
                    [(option, f"pf:{tag}:{state.profile_index}:{index}")]
                    for index, option in enumerate(question.options)
                ]
            ),
        )
        return

    found = pl.next_item(state, ctx.curriculum, ctx.rng)
    if found is None:
        _finish_placement(ctx, user, state)
        return
    exercise, point = found
    state.current = exercise.id
    ctx.storage.set_state(user.user_id, "placement", state.to_dict())
    options = labelled_options(exercise)
    ctx.say(
        user,
        f"Задание {state.total_asked + 1} · уровень {state.level}\n\n{exercise.prompt}\n\n"
        + "\n".join(options),
        inline_grid(
            [
                # Метка кнопки — позиция показа, payload — исходный индекс задания:
                # варианты перемешаны, а проверка идёт по данным банка.
                (chr(ord("A") + position), f"pa:{_placement_tag(state)}:{state.total_asked}:{index}")
                for position, (index, _) in enumerate(display_options(exercise))
            ],
            columns=4,
            tail=[("Не знаю", f"pa:{_placement_tag(state)}:{state.total_asked}:x")],
        ),
    )


def callback_profile(ctx: Context, user: User, payload: str) -> str:
    state = pl.PlacementState.from_dict(user.state_data)
    if state.profile_index >= len(pl.PROFILE_QUESTIONS):
        return ""
    step, raw = _parse_placement(payload, state)
    if step != state.profile_index:
        return STALE_STEP
    question = pl.PROFILE_QUESTIONS[state.profile_index]
    try:
        index = int(raw)
    except ValueError:
        return "не понял выбор"
    if not 0 <= index < len(question.options):
        return "не понял выбор"

    state.profile[question.id] = question.options[index]
    state.profile[f"{question.id}_index"] = str(index)
    ctx.storage.save_placement_answer(
        user.user_id, state.session_id, question.id, question.options[index], index, None, ""
    )
    state.profile_index += 1
    if state.profile_index == len(pl.PROFILE_QUESTIONS):
        state.level = pl.start_level_from_profile(state.profile)
    ctx.storage.set_state(user.user_id, "placement", state.to_dict())
    _ask_placement(ctx, user, state)
    return question.options[index]


def callback_placement_answer(ctx: Context, user: User, payload: str) -> str:
    state = pl.PlacementState.from_dict(user.state_data)
    step, raw = _parse_placement(payload, state)
    if step != state.total_asked:
        return STALE_STEP
    found = ctx.curriculum.exercise(state.current)
    if not found:
        _ask_placement(ctx, user, state)
        return ""
    exercise, point = found

    if raw == "x":
        correct = False
        chosen = "не знаю"
        index = None
    else:
        try:
            index = int(raw)
        except ValueError:
            return "не понял выбор"
        if not 0 <= index < len(exercise.options):
            return "не понял выбор"
        chosen = exercise.options[index]
        correct = index == exercise.correct_index

    pl.record(state, exercise, point, correct)
    ctx.storage.save_placement_answer(
        user.user_id, state.session_id, exercise.id, chosen, index, correct, point.level
    )
    if pl.advance(state, ctx.curriculum):
        ctx.storage.set_state(user.user_id, "placement", state.to_dict())
        _ask_placement(ctx, user, state)
    else:
        _finish_placement(ctx, user, state)
    return "верно" if correct else "мимо"


def handle_placement_text(ctx: Context, user: User, text: str) -> None:
    """Во время диагностики принимаем и букву варианта, а не только нажатие кнопки."""
    state = pl.PlacementState.from_dict(user.state_data)
    if state.profile_index < len(pl.PROFILE_QUESTIONS):
        question = pl.PROFILE_QUESTIONS[state.profile_index]
        index = _letter_index(text, len(question.options))
        if index is None:
            ctx.say(user, "Выбери вариант кнопкой или пришли номер: 1, 2, 3 или 4.")
            return
        callback_profile(ctx, user, f"{_placement_tag(state)}:{state.profile_index}:{index}")
        return

    found = ctx.curriculum.exercise(state.current)
    if not found:
        _ask_placement(ctx, user, state)
        return
    exercise, _ = found
    index = parse_choice(exercise, text)
    if index is None:
        ctx.say(user, "Выбери вариант кнопкой или пришли букву: A, B, C или D.")
        return
    callback_placement_answer(
        ctx, user, f"{_placement_tag(state)}:{state.total_asked}:{index}"
    )


def _letter_index(text: str, count: int) -> int | None:
    head = text.strip().split()[0].strip(".)-:").upper() if text.strip() else ""
    if len(head) == 1 and head.isalpha():
        index = ord(head) - ord("A")
        return index if 0 <= index < count else None
    if head.isdigit():
        index = int(head) - 1
        return index if 0 <= index < count else None
    return None


def _finish_placement(ctx: Context, user: User, state: pl.PlacementState) -> None:
    result = pl.finish(state, ctx.curriculum)
    target = pl.next_level(result.level)
    ctx.storage.update_user(user.user_id, level=result.level, target_level=target)
    ctx.reset_state(user)

    lines = [
        "Диагностика закончена.",
        "",
        f"Уровень: {result.level}. Цель: {target}.",
        f"Заданий: {result.asked}, верно {result.correct} ({round(result.accuracy * 100)}%).",
    ]
    if result.per_level:
        lines.append("")
        lines.append("По уровням:")
        for level in LEVELS:
            if level in result.per_level:
                correct, total = result.per_level[level]
                lines.append(f"• {level}: {correct}/{total}")
    if result.weak_topics:
        lines.append("")
        lines.append("Проседает: " + ", ".join(result.weak_topics))
    if result.strong_topics:
        lines.append("Держится: " + ", ".join(result.strong_topics))
    lines.append("")
    lines.append(
        "Это рабочая оценка для подбора материала, а не сертификат CEFR: "
        "аудирование и произношение так не измеряются."
    )

    # Ноль верных почти всегда значит, что задания прокликали, а не что человек
    # ничего не знает. Молча выдать нижний уровень — обречь его на скучный курс.
    suspicious = result.asked >= 4 and result.accuracy <= 0.30

    # Зеркальная страховка: тест меряет узнавание, и человек, оценивший себя
    # заметно ниже вердикта, чаще прав, чем тест. Молча ставить ему потолок
    # значит отправить в курс, где непонятно всё сразу.
    self_level = pl.start_level_from_profile(state.profile)
    gap = LEVELS.index(result.level) - LEVELS.index(self_level)
    if not suspicious and gap >= 2:
        lines.append("")
        lines.append(
            f"⚠️ Ты оценил себя как {self_level}, а тест показал {result.level}. "
            "Тест проверяет узнавание грамматики, а не речь и письмо, поэтому "
            "часто завышает. Выбери, с чего начать."
        )
        ctx.storage.log_event(user.user_id, "placement_gap", f"{self_level}->{result.level}", gap)
        ctx.say(
            user,
            "\n".join(lines),
            inline(
                [
                    [(f"Оставить {result.level}", f"setlvl:{result.level}")],
                    [(f"Начать с {LEVELS[min(LEVELS.index(self_level) + 1, len(LEVELS) - 1)]}",
                      f"setlvl:{LEVELS[min(LEVELS.index(self_level) + 1, len(LEVELS) - 1)]}")],
                ]
            ),
        )
        return
    if suspicious:
        lines.append("")
        lines.append(
            "⚠️ Верных ответов почти нет. Если ты кликал наугад — уровень занижен, "
            "и курс будет слишком лёгким. Лучше перепройти внимательно."
        )
    ctx.storage.log_event(user.user_id, "finish", "placement", result.asked)
    if suspicious:
        # Предлагать «Начать тренировку» сразу после «уровень занижен» — значит
        # звать в курс, о котором только что предупредили. Один выбор, не два.
        ctx.say(
            user,
            "\n".join(lines),
            inline(
                [
                    [("🧪 Перепройти диагностику", "retest")],
                    [(f"Оставить {result.level} и тренировать", f"setlvl:{result.level}")],
                ]
            ),
        )
        return

    ctx.say(user, "\n".join(lines))
    fresh = ctx.reload_user(user)
    ctx.say(
        user,
        study_plan(ctx.storage, ctx.curriculum, fresh),
        inline([[("Начать тренировку", "startpractice"), ("Открыть курс", "lvls")]]),
    )


# ── курс ─────────────────────────────────────────────────────────


def callback_retest(ctx: Context, user: User, payload: str) -> str:
    command_test(ctx, ctx.reload_user(user), "")
    return "начинаем заново"


def command_learn(ctx: Context, user: User, text: str) -> None:
    """Поиск правила по названию — единственное, чего не даёт кнопка «📚 Курс»."""
    query = text.partition(" ")[2].strip()
    if not query:
        ctx.say(
            user,
            "Напиши, что искать: /learn present perfect\n"
            "Весь курс по уровням и темам — кнопка «📚 Курс».",
        )
        return
    _show_search(ctx, user, query)


def show_levels(ctx: Context, user: User) -> None:
    mastery = mastery_map(ctx.storage, user.user_id)
    buttons = levels_overview(ctx.curriculum, mastery)
    if not buttons:
        ctx.say(user, "Курс ещё не загружен.")
        return
    hint = (
        f"Твой уровень: {user.level}."
        if user.level
        else "Уровень пока не определён — его подберёт диагностика."
    )
    rows = [buttons[index : index + 2] for index in range(0, len(buttons), 2)]
    # Каталог из 221 правила глубок по устройству, но продолжить с места должно
    # быть можно первым же нажатием.
    resume = _continue_point(ctx, user, mastery)
    if resume is not None:
        code = ctx.curriculum.point_code(resume.id)
        rows.insert(0, [(f"▶️ Продолжить: {_clip(resume.title_ru, 24)}", f"pt:{code}")])
    ctx.edit(
        user,
        f"Курс грамматики A1–C2, структура как на test-english.com.\n{hint}\n\n"
        "Выбери уровень — внутри темы, внутри тем правила с упражнениями.\n"
        "Ищешь конкретное правило? Напиши /learn present perfect",
        inline(rows),
    )
    ctx.storage.log_event(user.user_id, "screen", "course")


def _continue_point(ctx: Context, user: User, mastery: dict[str, int]) -> GrammarPoint | None:
    """Ближайшее незакрытое правило текущего уровня — точка возврата в курс."""
    if not user.level:
        return None
    from ..learning.progress import unmastered_points

    rows = unmastered_points(ctx.curriculum, mastery, user.level, limit=1)
    return rows[0] if rows else None


def _show_search(ctx: Context, user: User, query: str) -> None:
    found = ctx.curriculum.search(query)
    if not found:
        ctx.say(user, f"По запросу «{query}» ничего не нашёл. Весь курс — кнопка «📚 Курс».")
        return
    ctx.say(
        user,
        f"Нашёл по запросу «{query}»:",
        inline_grid(
            [
                (f"{point.level} · {_clip(point.title_ru, 34)}", f"pt:{ctx.curriculum.point_code(point.id)}")
                for point in found
            ],
            columns=1,
        ),
    )


def callback_levels(ctx: Context, user: User, payload: str) -> str:
    show_levels(ctx, user)
    return ""


def callback_level(ctx: Context, user: User, payload: str) -> str:
    level = payload.strip().upper()
    topics = ctx.curriculum.topics_of_level(level)
    if not topics:
        return "на этом уровне пока пусто"
    mastery = mastery_map(ctx.storage, user.user_id)
    buttons: list[tuple[str, str]] = []
    for topic, count in topics:
        points = ctx.curriculum.points_of_topic(level, topic)
        done = sum(1 for point in points if mastery.get(point.id, 0) >= 3)
        buttons.append(
            (f"{topic} · {done}/{count}", f"tp:{level}:{ctx.curriculum.topic_code(level, topic)}")
        )
    ctx.edit(
        user,
        f"Уровень {level} — {len(topics)} тем, "
        f"{len(ctx.curriculum.points_of_level(level))} правил.",
        inline_grid(buttons, columns=1, tail=[("← Уровни", "lvls")]),
    )
    return level


def callback_topic(ctx: Context, user: User, payload: str) -> str:
    level, _, code = payload.partition(":")
    topic = ctx.curriculum.topic_by_code(level, code)
    if topic is None:
        return "тема не найдена"
    points = ctx.curriculum.points_of_topic(level, topic)
    mastery = mastery_map(ctx.storage, user.user_id)
    buttons = [
        (
            f"{srs.stars(mastery.get(point.id, 0))} {_clip(point.title_ru, 36)}",
            f"pt:{ctx.curriculum.point_code(point.id)}",
        )
        for point in points
    ]
    ctx.edit(
        user,
        f"{level} · {topic}",
        inline_grid(
            buttons,
            columns=1,
            tail=[("Тренировать всю тему", f"prt:{level}:{code}"), ("← Темы", f"lvl:{level}")],
        ),
    )
    return topic


def callback_point(ctx: Context, user: User, payload: str) -> str:
    point = ctx.curriculum.by_code(payload.strip())
    if point is None:
        return "правило не найдено"
    ctx.edit(user, _lesson_text(ctx, user, point), _lesson_buttons(ctx, point))
    return point.title_en[:40]


def _lesson_text(ctx: Context, user: User, point: GrammarPoint) -> str:
    mastery = mastery_map(ctx.storage, user.user_id).get(point.id, 0)
    lines = [
        f"{point.level} · {point.topic}",
        point.title_ru,
        f"({point.title_en})",
        "",
        point.summary_ru,
    ]
    if point.forms:
        lines.extend(["", "Форма:", *[f"• {form}" for form in point.forms]])
    if point.examples:
        lines.extend(["", "Примеры:", *[f"• {example}" for example in point.examples[:5]]])
    lines.extend(["", f"Ловушка для русскоязычных: {point.ru_interference}"])
    stats = ctx.storage.point_stats(user.user_id).get(point.id)
    if stats:
        correct, total = stats
        lines.append("")
        lines.append(f"Твоя статистика: {correct}/{total} · {srs.stars(mastery)}")
    return "\n".join(lines)


def _lesson_buttons(ctx: Context, point: GrammarPoint) -> dict:
    code = ctx.curriculum.point_code(point.id)
    rows = [[("Тренировать", f"pr:{code}")]]
    if point.prerequisites:
        prior = ctx.curriculum.point(point.prerequisites[0])
        if prior:
            rows.append(
                [(f"Сначала: {_clip(prior.title_ru, 26)}", f"pt:{ctx.curriculum.point_code(prior.id)}")]
            )
    rows.append([("Объясни подробнее", f"ex:{code}")])
    # Назад к своей теме, а не сразу к уровням: возвращаться к списку, из
    # которого пришёл, — обычное ожидание от навигации.
    topic_code = ctx.curriculum.topic_code(point.level, point.topic)
    back: list[tuple[str, str]] = []
    if topic_code:
        back.append((f"← {_clip(point.topic, 22)}", f"tp:{point.level}:{topic_code}"))
    back.append(("← Уровни", "lvls"))
    rows.append(back)
    return inline(rows)


# ── тренировка ───────────────────────────────────────────────────


def command_practice(ctx: Context, user: User, text: str) -> None:
    level = user.level or "A2"
    mastery = mastery_map(ctx.storage, user.user_id)
    weak = [
        point.id
        for point in ctx.curriculum.points_of_level(level)
        if mastery.get(point.id, 0) < 3
    ]
    queue = pr.queue_for_level(
        ctx.curriculum, level, ctx.rng, weak, SESSION_LENGTH - VOCAB_PER_SESSION,
        seen=ctx.storage.seen_exercises(user.user_id),
    )
    queue += pr.queue_of_vocab(
        ctx.curriculum, level, ctx.storage.vocab_card_keys(user.user_id), ctx.rng,
        VOCAB_PER_SESSION,
    )
    if not queue:
        ctx.say(user, f"На уровне {level} пока нет заданий.")
        return
    ctx.rng.shuffle(queue)
    _order_by_recent_accuracy(ctx, user, queue)
    _start_session(ctx, user, "mixed", level, queue, level)


def _order_by_recent_accuracy(ctx: Context, user: User, queue: list[str]) -> None:
    """Обещание «в следующий раз будет сложнее» должно исполняться следующей сессией.

    Внутрисессионная адаптация подстраивает только хвост текущей очереди, поэтому
    сильный ученик каждый день начинал с тех же лёгких заданий.
    """
    accuracy = ctx.storage.recent_accuracy(user.user_id)
    if accuracy is None:
        return
    if accuracy > pr.TARGET_HIGH:
        harder = True
    elif accuracy < pr.TARGET_LOW:
        harder = False
    else:
        return
    resolved = [
        (ref, question.difficulty if question else 2)
        for ref, question in ((ref, pr.resolve(ref, ctx.curriculum, ctx.rng)) for ref in queue)
    ]
    resolved.sort(key=lambda item: item[1], reverse=harder)
    queue[:] = [ref for ref, _ in resolved]


def callback_start_practice(ctx: Context, user: User, payload: str) -> str:
    command_practice(ctx, ctx.reload_user(user), "")
    return ""


def callback_practice_point(ctx: Context, user: User, payload: str) -> str:
    point = ctx.curriculum.by_code(payload.strip())
    if point is None:
        return "правило не найдено"
    queue = pr.queue_for_point(point, ctx.rng)
    _start_session(ctx, user, "point", point.title_ru, queue, point.level)
    return _clip(point.title_ru, 40)


def callback_practice_topic(ctx: Context, user: User, payload: str) -> str:
    level, _, code = payload.partition(":")
    topic = ctx.curriculum.topic_by_code(level, code)
    if topic is None:
        return "тема не найдена"
    queue = pr.queue_for_topic(
        ctx.curriculum, level, topic, ctx.rng, seen=ctx.storage.seen_exercises(user.user_id)
    )
    if not queue:
        return "в теме нет заданий"
    _start_session(ctx, user, "topic", topic, queue, level)
    return topic


def command_review(ctx: Context, user: User, text: str) -> None:
    cards = ctx.storage.due_cards(user.user_id, limit=REVIEW_LENGTH * 2)
    queue = pr.queue_for_review(
        ctx.curriculum, cards, ctx.rng, REVIEW_LENGTH, level=user.level or "",
        seen=ctx.storage.seen_exercises(user.user_id),
    )
    if not queue:
        counts = ctx.storage.card_counts(user.user_id)
        total = sum(value[0] for value in counts.values())
        if total:
            ctx.say(
                user,
                "Сейчас повторять нечего — все карточки ждут своего срока. "
                "Хочешь новое — «🎯 Заниматься» или «📚 Курс».",
            )
        else:
            ctx.say(
                user,
                "Карточек ещё нет. Они появляются сами после занятий и разборов — "
                "начни с «🎯 Заниматься».",
            )
        return
    _start_session(ctx, user, "review", "повторение", queue, user.level or "")


def callback_start_review(ctx: Context, user: User, payload: str) -> str:
    command_review(ctx, ctx.reload_user(user), "")
    return ""


def _start_session(
    ctx: Context, user: User, kind: str, subject: str, queue: list[str], level: str
) -> None:
    session_id = ctx.storage.start_session(user.user_id, kind, subject)
    state = pr.PracticeState(
        kind=kind, subject=subject, queue=queue, session_id=session_id, level=level
    )
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())
    ctx.log_start(user, kind)
    # У смешанной тренировки подводка уже была, а заголовок задания несёт счётчик:
    # объявлять «поехали, 10 заданий» второй раз — шум.
    if kind != "mixed":
        ctx.say(user, f"{subject} · {len(queue)} заданий.", RESUME_KEYBOARD)
    _ask_question(ctx, user, state)


def resume_practice(ctx: Context, user: User) -> None:
    """Возвращает к текущему заданию тренировки, не начиная новую сессию."""
    state = pr.PracticeState.from_dict(user.state_data)
    ctx.say(user, "Продолжаем тренировку.", RESUME_KEYBOARD)
    _ask_question(ctx, user, state, repeat=True)


def finish_current_practice(ctx: Context, user: User) -> None:
    """Закрывает начатую тренировку по-человечески: с итогом, серией и нормой дня.

    Раньше переключение кнопкой просто стирало состояние: ответы оставались в
    базе, а сессия висела незакрытой — пропадали и серия, и дневная норма.
    """
    _finish_session(ctx, user, pr.PracticeState.from_dict(user.state_data))


def _ask_question(
    ctx: Context,
    user: User,
    state: pr.PracticeState,
    repeat: bool = False,
    prefix: str = "",
    extra_rows: list[list[tuple[str, str]]] | None = None,
) -> None:
    """Показывает текущее задание.

    `repeat` — тот же шаг ещё раз, без сдвига очереди. `prefix` — вердикт по
    предыдущему ответу: он идёт тем же сообщением, что и новый вопрос, иначе на
    десять заданий приходится двадцать сообщений.
    """
    if repeat:
        ref = state.current_ref()
        question = pr.resolve(ref, ctx.curriculum, ctx.rng) if ref else None
        if question is None:
            return
        text, keyboard = _question_view(state, question, extra_rows)
        ctx.say(user, _join(prefix, text), keyboard)
        return
    while True:
        ref = state.current_ref()
        if ref is None:
            if prefix:
                ctx.say(user, prefix, inline(extra_rows) if extra_rows else None)
            _finish_session(ctx, user, state)
            return
        question = pr.resolve(ref, ctx.curriculum, ctx.rng)
        if question is not None:
            break
        state.index += 1  # ссылка протухла после обновления контента

    state.helped = False
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())
    text, keyboard = _question_view(state, question, extra_rows)
    ctx.say(user, _join(prefix, text), keyboard)


def _join(prefix: str, text: str) -> str:
    return f"{prefix}\n\n{text}" if prefix else text


def _question_view(
    state: pr.PracticeState,
    question: pr.Question,
    extra_rows: list[list[tuple[str, str]]] | None = None,
) -> tuple[str, dict]:
    """Текст задания и его клавиатура — одним куском, чтобы повтор был точным."""
    # В смешанной тренировке название правила — это подсказка: показываем только тему.
    label = question.title_ru if state.kind == "point" else question.topic
    body = [f"{state.index + 1}/{len(state.queue)} · {label}", "", question.prompt]
    if question.is_choice:
        body.append("")
        body.extend(
            f"{chr(ord('A') + index)}) {option}"
            for index, option in enumerate(question.options)
        )
        rows = [
            [
                (chr(ord("A") + index), f"an:{_tag(state)}:{state.index}:{index}")
                for index in range(len(question.options))
            ],
            _session_buttons(state),
        ]
    else:
        body.append("")
        body.append(pr.task_hint(question))
        rows = [_session_buttons(state)]
    keyboard = inline((extra_rows or []) + rows)
    return "\n".join(body), keyboard


def handle_practice_text(ctx: Context, user: User, text: str) -> None:
    state = pr.PracticeState.from_dict(user.state_data)
    ref = state.current_ref()
    if ref is None:
        _finish_session(ctx, user, state)
        return
    question = pr.resolve(ref, ctx.curriculum, ctx.rng)
    if question is None:
        state.index += 1
        _ask_question(ctx, user, state)
        return
    _grade(ctx, user, state, question, text)


def callback_answer(ctx: Context, user: User, payload: str) -> str:
    state = pr.PracticeState.from_dict(user.state_data)
    step, index = _parse_step(payload, state)
    if step != state.index:
        return STALE_STEP
    ref = state.current_ref()
    if ref is None:
        _finish_session(ctx, user, state)
        return ""
    question = pr.resolve(ref, ctx.curriculum, ctx.rng)
    if question is None:
        # Ссылка протухла после обновления контента — двигаем очередь, а не молчим.
        state.index += 1
        ctx.storage.set_state(user.user_id, "practice", state.to_dict())
        _ask_question(ctx, user, state)
        return "задание обновилось"
    if not question.is_choice:
        return "здесь нужен свободный ответ"
    if index is None or not 0 <= index < len(question.options):
        return "не понял"
    _grade(ctx, user, state, question, question.options[index])
    return ""


def _grade(
    ctx: Context, user: User, state: pr.PracticeState, question: pr.Question, text: str
) -> None:
    verdict = pr.check(question, text)
    if not verdict.understood:
        if question.kind == "cloze":
            ctx.say(
                user,
                f"Нужно {len(question.gaps)} {pr.answers_word(len(question.gaps))} по порядку — каждый с новой строки "
                "или через точку с запятой.",
            )
        else:
            ctx.say(user, "Выбери вариант кнопкой или пришли букву: A, B, C или D.")
        return

    state.answered += 1
    if verdict.correct:
        state.correct += 1

    ctx.storage.record_attempt(
        user.user_id, question.ref, question.point_id, question.level, verdict.correct, text,
    )

    card = ctx.storage.card(user.user_id, question.card_type, question.card_key) or srs.new_card(
        question.card_type, question.card_key
    )
    updated = srs.review(card, srs.quality(verdict.correct, used_hint=state.helped))
    ctx.storage.upsert_card(user.user_id, updated)

    if verdict.correct:
        head = "Верно." if not state.helped else "Верно, с подсказкой."
    else:
        head = f"Мимо. Правильно: {verdict.expected_text}"
    reply = [head]
    if verdict.note:
        reply.append(verdict.note)
    if question.explanation_ru:
        reply.append(question.explanation_ru)
    reply.append(srs.interval_note_ru(updated) + f" · {srs.stars(updated.mastery)}")
    # Разбор правила после ошибки — самый полезный момент: человек уже понял,
    # что не знает, и готов прочитать объяснение.
    extra_rows: list[list[tuple[str, str]]] | None = None
    if not verdict.correct and question.point_id:
        code = ctx.curriculum.point_code(question.point_id)
        extra_rows = [[("💡 Разобрать правило", f"rule:{code}")]]

    state.index += 1
    state.helped = False
    pr.adapt(state, ctx.curriculum, ctx.rng)
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())
    # Вердикт и следующий вопрос — одно сообщение: разбор больше не оказывается
    # выше нового задания, а лента не растёт вдвое быстрее нужного.
    _ask_question(ctx, user, state, prefix="\n".join(reply), extra_rows=extra_rows)


def callback_hint(ctx: Context, user: User, payload: str) -> str:
    state = pr.PracticeState.from_dict(user.state_data)
    step, _ = _parse_step(payload, state)
    if step != state.index:
        return STALE_STEP
    ref = state.current_ref()
    question = pr.resolve(ref, ctx.curriculum, ctx.rng) if ref else None
    if question is None:
        return ""
    state.helped = True
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())
    ctx.say(user, pr.help_for(question, ctx.curriculum))
    # Разбор занимает полтора-три экрана, и кнопки ответа уезжают вверх ровно в
    # тот момент, когда человек не уверен. Повторяем вопрос под разбором.
    _ask_question(ctx, user, state, repeat=True)
    return "разбор темы"


def callback_rule(ctx: Context, user: User, payload: str) -> str:
    """Разбор правила по коду пункта — работает и после ошибки, и из курса."""
    point = ctx.curriculum.by_code(payload.strip())
    if point is None:
        return "правило не найдено"
    fresh = ctx.reload_user(user)
    if fresh.state == "practice":
        # Внутри занятия «Тренировать эту тему» начала бы новую сессию поверх
        # текущей — предлагаем вернуться к заданию, которое человек уже решает.
        # Следующий вопрос уже открыт и может быть по той же теме: его ответ прячем.
        state = pr.PracticeState.from_dict(fresh.state_data)
        ref = state.current_ref()
        current = pr.resolve(ref, ctx.curriculum, ctx.rng) if ref else None
        hide = pr.revealing_texts(current) if current else ()
        ctx.say(user, pr.point_help(point, hide=hide), inline([[("▶️ Продолжить", "rsm")]]))
    else:
        ctx.say(
            user,
            pr.point_help(point),
            inline([[("Тренировать эту тему", f"pr:{ctx.curriculum.point_code(point.id)}")]]),
        )
    return _clip(point.title_ru, 40)


def callback_skip(ctx: Context, user: User, payload: str) -> str:
    state = pr.PracticeState.from_dict(user.state_data)
    step, _ = _parse_step(payload, state)
    if step != state.index:
        return STALE_STEP
    ref = state.current_ref()
    question = pr.resolve(ref, ctx.curriculum, ctx.rng) if ref else None
    if question is not None:
        expected = question.expected[0] if question.expected else ""
        card = ctx.storage.card(
            user.user_id, question.card_type, question.card_key
        ) or srs.new_card(question.card_type, question.card_key)
        ctx.storage.upsert_card(user.user_id, srs.review(card, srs.quality(False)))
        ctx.say(user, f"Пропустил. Ответ: {expected}\n{question.explanation_ru}")
        state.answered += 1
    state.index += 1
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())
    _ask_question(ctx, user, state)
    return "пропущено"


def callback_end_session(ctx: Context, user: User, payload: str) -> str:
    state = pr.PracticeState.from_dict(user.state_data)
    _finish_session(ctx, user, state)
    return "закончили"


def _finish_session(ctx: Context, user: User, state: pr.PracticeState) -> None:
    ctx.reset_state(user)
    if state.session_id:
        ctx.storage.finish_session(state.session_id, state.answered, state.correct)
    if not state.answered:
        ctx.say(user, "Сессия закрыта.", MAIN_KEYBOARD)
        ctx.storage.log_event(user.user_id, "abandon", state.kind)
        return

    streak = ctx.storage.bump_streak(user.user_id)
    ctx.storage.log_event(user.user_id, "finish", state.kind, state.answered)
    share = round(state.accuracy * 100)
    lines = [
        f"Готово: {state.correct} из {state.answered} ({share}%).",
        f"Серия: {streak} дн.",
    ]
    if share >= 85:
        lines.append("Слишком легко — следующая тренировка пойдёт сложнее.")
    elif share < 50:
        lines.append("Тяжеловато. Разбери правило и вернись — так быстрее.")
    else:
        lines.append("Нормальный коридор: сложность подобрана верно.")

    counts = ctx.storage.card_counts(user.user_id)
    due = sum(value[1] for value in counts.values())
    buttons: list[list[tuple[str, str]]] = []
    if due:
        lines.append(f"К повторению уже готово: {due} карточек.")
        buttons.append([(f"🔁 Повторить ({due})", "startreview")])
    # После тренировки одной темы «Ещё раунд» уводил в смешанную тренировку
    # уровня: продолжать логичнее ту же тему, которую человек и выбрал.
    again = _repeat_button(ctx, state)
    buttons.append([again, ("Что дальше", "daily")])
    ctx.say(user, "\n".join(lines), inline(buttons))


def _repeat_button(ctx: Context, state: pr.PracticeState) -> tuple[str, str]:
    """Кнопка «ещё раз» ведёт туда же, где человек занимался."""
    if state.kind == "point":
        point = next(
            (item for item in ctx.curriculum.points.values() if item.title_ru == state.subject),
            None,
        )
        if point is not None:
            return ("Ещё по этой теме", f"pr:{ctx.curriculum.point_code(point.id)}")
    if state.kind == "topic" and state.level:
        code = ctx.curriculum.topic_code(state.level, state.subject)
        if code:
            return ("Ещё по этой теме", f"prt:{state.level}:{code}")
    return ("Ещё раунд", "startpractice")


def callback_progress(ctx: Context, user: User, payload: str) -> str:
    from .core import command_progress

    command_progress(ctx, ctx.reload_user(user), "")
    return ""


def callback_plan(ctx: Context, user: User, payload: str) -> str:
    from .core import command_plan

    command_plan(ctx, ctx.reload_user(user), "")
    return ""
