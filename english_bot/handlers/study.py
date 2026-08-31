"""Диагностика уровня, курс по темам, тренировка и повторение."""

from __future__ import annotations

import logging

from ..content.schema import LEVELS, GrammarPoint
from ..context import Context
from ..learning import placement as pl
from ..learning import practice as pr
from ..learning import srs
from ..learning.answers import labelled_options, parse_choice
from ..learning.progress import levels_overview, mastery_map, study_plan
from ..storage import User
from ..telegram_api import inline, inline_grid
from .menu import MAIN_KEYBOARD


LOGGER = logging.getLogger(__name__)

SESSION_LENGTH = 10
REVIEW_LENGTH = 15
VOCAB_PER_SESSION = 3
STALE_STEP = "это задание уже закрыто"


def _step_of(payload: str) -> int:
    """Номер шага из callback_data; -1, если его там нет или он битый."""
    head = payload.partition(":")[0]
    try:
        return int(head)
    except ValueError:
        return -1


def _step_and_choice(payload: str) -> tuple[int, int | None]:
    step, _, rest = payload.partition(":")
    try:
        parsed_step = int(step)
    except ValueError:
        return -1, None
    try:
        return parsed_step, int(rest)
    except ValueError:
        return parsed_step, None


def _session_buttons(step: int) -> list[tuple[str, str]]:
    return [("💡 Правило", f"hint:{step}"), ("Пропустить", f"skip:{step}"), ("Стоп", "endses")]


# ── диагностика ──────────────────────────────────────────────────


def command_test(ctx: Context, user: User, text: str) -> None:
    if not ctx.curriculum.points:
        ctx.say(user, "Курс ещё не загружен — сообщи админу.")
        return
    session_id = ctx.storage.next_placement_session(user.user_id)
    state = pl.PlacementState(session_id=session_id)
    ctx.storage.set_state(user.user_id, "placement", state.to_dict())
    ctx.say(
        user,
        "Диагностика уровня A1–C2. Сначала три вопроса о тебе, потом задания: "
        "они подстраиваются под ответы, поэтому теста ровно столько, сколько нужно. "
        "10–15 минут.",
        MAIN_KEYBOARD,
    )
    _ask_placement(ctx, user, state)


def _ask_placement(ctx: Context, user: User, state: pl.PlacementState) -> None:
    if state.profile_index < len(pl.PROFILE_QUESTIONS):
        question = pl.PROFILE_QUESTIONS[state.profile_index]
        ctx.say(
            user,
            f"{state.profile_index + 1}/3 · {question.prompt}",
            inline(
                [
                    [(option, f"pf:{state.profile_index}:{index}")]
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
                (chr(ord("A") + index), f"pa:{state.total_asked}:{index}")
                for index in range(len(exercise.options))
            ],
            columns=4,
            tail=[("Не знаю", f"pa:{state.total_asked}:x")],
        ),
    )


def callback_profile(ctx: Context, user: User, payload: str) -> str:
    state = pl.PlacementState.from_dict(user.state_data)
    if state.profile_index >= len(pl.PROFILE_QUESTIONS):
        return ""
    step, index = _step_and_choice(payload)
    if step != state.profile_index:
        return STALE_STEP
    question = pl.PROFILE_QUESTIONS[state.profile_index]
    if index is None:
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
    step, raw = payload.partition(":")[0], payload.partition(":")[2]
    try:
        if int(step) != state.total_asked:
            return STALE_STEP
    except ValueError:
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
        callback_profile(ctx, user, f"{state.profile_index}:{index}")
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
    callback_placement_answer(ctx, user, f"{state.total_asked}:{index}")


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
    suspicious = result.asked >= 4 and result.accuracy <= 0.15
    if suspicious:
        lines.append("")
        lines.append(
            "⚠️ Верных ответов почти нет. Если ты кликал наугад — уровень занижен, "
            "и курс будет слишком лёгким. Лучше перепройти внимательно."
        )
    ctx.say(
        user,
        "\n".join(lines),
        inline([[("Перепройти диагностику", "retest")]]) if suspicious else None,
    )

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
        else "Уровень не определён — нажми «🎯 Заниматься»."
    )
    ctx.say(
        user,
        f"Курс грамматики A1–C2, структура как на test-english.com.\n{hint}\n\n"
        "Выбери уровень — внутри темы, внутри тем правила с упражнениями.\n"
        "Ищешь конкретное правило? Напиши /learn present perfect",
        inline_grid(buttons, columns=2),
    )


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
                (f"{point.level} · {point.title_ru[:40]}", f"pt:{ctx.curriculum.point_code(point.id)}")
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
    ctx.say(
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
            f"{srs.stars(mastery.get(point.id, 0))} {point.title_ru[:38]}",
            f"pt:{ctx.curriculum.point_code(point.id)}",
        )
        for point in points
    ]
    ctx.say(
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
    ctx.say(user, _lesson_text(ctx, user, point), _lesson_buttons(ctx, point))
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
            rows.append([(f"Сначала: {prior.title_ru[:28]}", f"pt:{ctx.curriculum.point_code(prior.id)}")])
    rows.append([("Объясни подробнее", f"ex:{code}"), ("← Уровни", "lvls")])
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
        ctx.curriculum, level, ctx.rng, weak, SESSION_LENGTH - VOCAB_PER_SESSION
    )
    queue += pr.queue_of_vocab(
        ctx.curriculum, level, ctx.storage.vocab_card_keys(user.user_id), ctx.rng,
        VOCAB_PER_SESSION,
    )
    if not queue:
        ctx.say(user, f"На уровне {level} пока нет заданий.")
        return
    ctx.rng.shuffle(queue)
    _start_session(ctx, user, "mixed", level, queue, level)


def callback_start_practice(ctx: Context, user: User, payload: str) -> str:
    command_practice(ctx, ctx.reload_user(user), "")
    return ""


def callback_practice_point(ctx: Context, user: User, payload: str) -> str:
    point = ctx.curriculum.by_code(payload.strip())
    if point is None:
        return "правило не найдено"
    queue = pr.queue_for_point(point, ctx.rng)
    _start_session(ctx, user, "point", point.title_ru, queue, point.level)
    return point.title_ru[:40]


def callback_practice_topic(ctx: Context, user: User, payload: str) -> str:
    level, _, code = payload.partition(":")
    topic = ctx.curriculum.topic_by_code(level, code)
    if topic is None:
        return "тема не найдена"
    queue = pr.queue_for_topic(ctx.curriculum, level, topic, ctx.rng)
    if not queue:
        return "в теме нет заданий"
    _start_session(ctx, user, "topic", topic, queue, level)
    return topic


def command_review(ctx: Context, user: User, text: str) -> None:
    cards = ctx.storage.due_cards(user.user_id, limit=REVIEW_LENGTH * 2)
    queue = pr.queue_for_review(ctx.curriculum, cards, ctx.rng, REVIEW_LENGTH)
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
    # У смешанной тренировки подводка уже была, а заголовок задания несёт счётчик:
    # объявлять «поехали, 10 заданий» второй раз — шум.
    if kind != "mixed":
        ctx.say(user, f"{subject} · {len(queue)} заданий.", MAIN_KEYBOARD)
    _ask_question(ctx, user, state)


def _ask_question(ctx: Context, user: User, state: pr.PracticeState) -> None:
    while True:
        ref = state.current_ref()
        if ref is None:
            _finish_session(ctx, user, state)
            return
        question = pr.resolve(ref, ctx.curriculum, ctx.rng)
        if question is not None:
            break
        state.index += 1  # ссылка протухла после обновления контента

    state.helped = False
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())

    # В смешанной тренировке название правила — это подсказка: показываем только тему.
    label = question.title_ru if state.kind == "point" else question.topic
    header = f"{state.index + 1}/{len(state.queue)} · {label}"
    body = [header, "", question.prompt]
    if question.is_choice:
        body.append("")
        body.extend(
            f"{chr(ord('A') + index)}) {option}"
            for index, option in enumerate(question.options)
        )
        keyboard = inline_grid(
            [
                (chr(ord("A") + index), f"an:{state.index}:{index}")
                for index in range(len(question.options))
            ],
            columns=4,
            tail=_session_buttons(state.index),
        )
    else:
        body.append("")
        body.append(pr.task_hint(question))
        keyboard = inline([_session_buttons(state.index)])
    ctx.say(user, "\n".join(body), keyboard)


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
    step, index = _step_and_choice(payload)
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
    if question.explanation_ru:
        reply.append(question.explanation_ru)
    reply.append(srs.interval_note_ru(updated) + f" · {srs.stars(updated.mastery)}")
    # Разбор правила после ошибки — самый полезный момент: человек уже понял,
    # что не знает, и готов прочитать объяснение.
    rule_button = None
    if not verdict.correct and question.point_id:
        code = ctx.curriculum.point_code(question.point_id)
        rule_button = inline([[("💡 Разобрать правило", f"rule:{code}")]])
    ctx.say(user, "\n".join(reply), rule_button)

    state.index += 1
    state.helped = False
    pr.adapt(state, ctx.curriculum, ctx.rng)
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())
    _ask_question(ctx, user, state)


def callback_hint(ctx: Context, user: User, payload: str) -> str:
    state = pr.PracticeState.from_dict(user.state_data)
    if _step_of(payload) != state.index:
        return STALE_STEP
    ref = state.current_ref()
    question = pr.resolve(ref, ctx.curriculum, ctx.rng) if ref else None
    if question is None:
        return ""
    state.helped = True
    ctx.storage.set_state(user.user_id, "practice", state.to_dict())
    ctx.say(user, pr.help_for(question, ctx.curriculum))
    return "разбор темы"


def callback_rule(ctx: Context, user: User, payload: str) -> str:
    """Разбор правила по коду пункта — работает и после ошибки, и из курса."""
    point = ctx.curriculum.by_code(payload.strip())
    if point is None:
        return "правило не найдено"
    ctx.say(
        user,
        pr.point_help(point),
        inline([[("Тренировать эту тему", f"pr:{ctx.curriculum.point_code(point.id)}")]]),
    )
    return point.title_ru[:40]


def callback_skip(ctx: Context, user: User, payload: str) -> str:
    state = pr.PracticeState.from_dict(user.state_data)
    if _step_of(payload) != state.index:
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
        return

    streak = ctx.storage.bump_streak(user.user_id)
    share = round(state.accuracy * 100)
    lines = [
        f"Готово: {state.correct} из {state.answered} ({share}%).",
        f"Серия: {streak} дн.",
    ]
    if share >= 85:
        lines.append("Слишком легко — в следующий раз подниму сложность.")
    elif share < 50:
        lines.append("Тяжеловато. Прочитай правило в «📚 Курс» и вернись — так быстрее.")
    else:
        lines.append("Нормальный коридор: сложность подобрана верно.")

    counts = ctx.storage.card_counts(user.user_id)
    due = sum(value[1] for value in counts.values())
    buttons = [[("Ещё раунд", "startpractice"), ("Что дальше", "daily")]]
    if due:
        lines.append(f"К повторению уже готово: {due} карточек.")
        buttons.insert(0, [(f"🔁 Повторить ({due})", "startreview")])

    ctx.say(user, "\n".join(lines), inline(buttons))


def callback_progress(ctx: Context, user: User, payload: str) -> str:
    from .core import command_progress

    command_progress(ctx, ctx.reload_user(user), "")
    return ""


def callback_plan(ctx: Context, user: User, payload: str) -> str:
    from .core import command_plan

    command_plan(ctx, ctx.reload_user(user), "")
    return ""
