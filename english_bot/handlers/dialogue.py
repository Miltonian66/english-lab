"""Свободное общение, ролевые диалоги и письменные работы."""

from __future__ import annotations

import logging
import re

from ..ai.llm import LLMError
from ..ai.prompts import dialogue_system, explain_system, learner_block, tutor_system
from ..context import Context
from ..learning.progress import mastery_map
from ..learning.scoring import assess_writing, format_assessment
from ..storage import User
from ..telegram_api import inline, inline_grid
from .menu import MAIN_KEYBOARD


LOGGER = logging.getLogger(__name__)

# Разбираем видимый блок правок вместо второго вызова модели: дешевле и без рассинхрона.
CORRECTION_LINE = re.compile(
    r'^[•\-*]\s*["“«]?(?P<original>[^"”»]+)["”»]?\s*(?:→|->)\s*["“«]?(?P<corrected>[^"”»]+)["”»]?'
    r'\s*(?:[-—–]\s*)?(?:\[(?P<category>[^\]]+)\])?\s*(?P<note>.*)$'
)
CATEGORY_MAP = {
    "grammar": "grammar", "грамматика": "grammar",
    "word choice": "word_choice", "лексика": "word_choice",
    "article": "article", "артикль": "article",
    "preposition": "preposition", "предлог": "preposition",
    "word order": "word_order", "порядок слов": "word_order",
    "spelling": "spelling", "орфография": "spelling",
    "punctuation": "punctuation", "пунктуация": "punctuation",
    "expression": "expression", "выражение": "expression",
}


def _context_block(ctx: Context, user: User) -> str:
    mastery = mastery_map(ctx.storage, user.user_id)
    weak = [
        point.title_en
        for point in ctx.curriculum.points_of_level(user.level or "A2")
        if mastery.get(point.id, 0) < 3
    ][:6]
    errors = []
    for row in ctx.storage.error_summary(user.user_id, limit=6):
        pattern = ctx.curriculum.error_patterns.get(str(row["pattern_id"] or ""))
        label = pattern.label_ru if pattern else str(row["category"])
        errors.append(f"{label} ×{row['times']}")
    goal = ""
    for row in ctx.storage.placement_answers(user.user_id):
        if row["question_id"] == "profile_goal":
            goal = str(row["answer_text"])
    return learner_block(
        level=user.level,
        target_level=user.target_level,
        goal=goal,
        weak_points=weak,
        frequent_errors=errors,
        streak=user.streak_days,
    )


def parse_corrections(text: str) -> list[tuple[str, str, str, str]]:
    """Достаёт (было, стало, категория, заметка) из блока «Правки:» ответа наставника."""
    rows: list[tuple[str, str, str, str]] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(("правки:", "corrections:")):
            in_block = True
            continue
        if in_block and not stripped:
            continue
        if in_block and stripped.lower().startswith(("возьми себе", "ошибок нет")):
            break
        if not in_block:
            continue
        match = CORRECTION_LINE.match(stripped)
        if not match:
            continue
        original = match.group("original").strip()
        corrected = match.group("corrected").strip()
        if not original or not corrected or original == corrected:
            continue
        raw_category = (match.group("category") or "").strip().lower()
        category = CATEGORY_MAP.get(raw_category, "grammar")
        rows.append((original, corrected, category, match.group("note").strip()))
    return rows[:8]


def _log_corrections(ctx: Context, user: User, reply: str, source: str) -> None:
    for original, corrected, category, note in parse_corrections(reply):
        ctx.storage.log_error(
            user.user_id, category, original, corrected, note, source=source
        )


# ── свободное общение ────────────────────────────────────────────


def command_chat(ctx: Context, user: User, text: str) -> None:
    ctx.reset_state(user)
    ctx.say(
        user,
        "Режим свободного общения. Пиши по-английски — отвечу и разберу ошибки. "
        "По-русски тоже можно, если нужен разбор правила.\n"
        "Выйти: /stop.",
        MAIN_KEYBOARD,
    )


def handle_free_text(ctx: Context, user: User, text: str) -> None:
    llm = ctx.require_llm(user)
    if llm is None:
        ctx.storage.add_message(user.user_id, "user", text)
        return

    ctx.working(user, "Думаю над ответом, это до минуты.")
    ctx.storage.add_message(user.user_id, "user", text)
    history = ctx.storage.recent_messages(user.user_id, limit=12)
    try:
        reply = llm.complete(
            tutor_system(_context_block(ctx, user)),
            history,
            user_id=user.user_id,
            max_tokens=1000,
        )
    except LLMError as exc:
        LOGGER.warning("Наставник не ответил: %s", exc)
        ctx.say(user, "ИИ сейчас недоступен. Сообщение сохранил, попробуй ещё раз.")
        return

    ctx.storage.add_message(user.user_id, "assistant", reply)
    ctx.storage.trim_messages(user.user_id)
    _log_corrections(ctx, user, reply, source="chat")
    ctx.say(user, reply)


# Готовые сценарии: одно нажатие вместо придумывания темы. Рабочие ситуации
# идут первыми — платформа отдела, и говорить чаще приходится про работу.
ROLEPLAY_PRESETS: list[tuple[str, str]] = [
    ("Собеседование", "техническое собеседование на позицию backend-разработчика"),
    ("Стендап", "стендап команды: что сделал вчера, где застрял, что дальше"),
    ("Заказчик", "объясняю заказчику, почему сроки сдвинулись"),
    ("Инцидент", "разбор вчерашнего инцидента с коллегой из соседней команды"),
    ("Кафе", "заказываю еду в кафе и уточняю состав блюда"),
    ("Аэропорт", "регистрация на рейс, вопросы про багаж и пересадку"),
]


def show_roleplay_presets(ctx: Context, user: User) -> None:
    ctx.say(
        user,
        "Ролевой диалог: выбери ситуацию — начну первым.\n"
        "Свой сценарий: /roleplay объясняю на созвоне, почему упал прод",
        inline_grid(
            [(title, f"rp:{index}") for index, (title, _) in enumerate(ROLEPLAY_PRESETS)],
            columns=2,
        ),
    )


def callback_roleplay_start(ctx: Context, user: User, payload: str) -> str:
    try:
        title, scenario = ROLEPLAY_PRESETS[int(payload)]
    except (ValueError, IndexError):
        return "сценарий не найден"
    _start_roleplay(ctx, ctx.reload_user(user), scenario)
    return title


def command_roleplay(ctx: Context, user: User, text: str) -> None:
    scenario = text.partition(" ")[2].strip()
    if not scenario:
        show_roleplay_presets(ctx, user)
        return
    _start_roleplay(ctx, user, scenario)


def _start_roleplay(ctx: Context, user: User, scenario: str) -> None:
    ctx.storage.set_state(user.user_id, "roleplay", {"scenario": scenario[:200]})
    llm = ctx.require_llm(user)
    if llm is None:
        return
    ctx.typing(user)
    try:
        reply = llm.complete(
            dialogue_system(_context_block(ctx, user), scenario, user.level or "A2"),
            [{"role": "user", "content": "Start the roleplay with your first line."}],
            user_id=user.user_id,
            max_tokens=500,
        )
    except LLMError:
        ctx.say(user, "Не получилось запустить диалог. Попробуй ещё раз.")
        ctx.reset_state(user)
        return
    ctx.storage.add_message(user.user_id, "assistant", reply)
    ctx.say(user, f"Сценарий: {scenario}\n\n{reply}", inline([[("Закончить", "endroleplay")]]))


def handle_roleplay_text(ctx: Context, user: User, text: str) -> None:
    llm = ctx.require_llm(user)
    if llm is None:
        return
    scenario = str(user.state_data.get("scenario") or "conversation")
    ctx.typing(user)
    ctx.storage.add_message(user.user_id, "user", text)
    history = ctx.storage.recent_messages(user.user_id, limit=14)
    try:
        reply = llm.complete(
            dialogue_system(_context_block(ctx, user), scenario, user.level or "A2"),
            history,
            user_id=user.user_id,
            max_tokens=800,
        )
    except LLMError:
        ctx.say(user, "ИИ сейчас недоступен, попробуй ещё раз.")
        return
    ctx.storage.add_message(user.user_id, "assistant", reply)
    _log_corrections(ctx, user, reply, source="roleplay")
    ctx.say(user, reply, inline([[("Закончить", "endroleplay")]]))


def callback_end_roleplay(ctx: Context, user: User, payload: str) -> str:
    ctx.reset_state(user)
    ctx.say(user, "Диалог закончен. Разбор ошибок — «📊 Я» → «Прогресс».")
    return "закончили"


# ── письмо ───────────────────────────────────────────────────────


def command_writing(ctx: Context, user: User, text: str) -> None:
    level = user.level or "A2"
    done = {str(row["task_id"]) for row in ctx.storage.writings(user.user_id, limit=50)}
    task = ctx.curriculum.pick_writing(level, done, ctx.rng)
    if task is None:
        ctx.say(user, f"Для уровня {level} письменных заданий пока нет.")
        return
    ctx.storage.set_state(user.user_id, "writing", {"task_id": task.id})
    focus = ", ".join(task.focus) if task.focus else "свободно"
    ctx.say(
        user,
        f"✍️ {task.title_ru} · {task.level}\n\n"
        f"{task.prompt_en}\n\n"
        f"Как писать: {task.guidance_ru}\n"
        f"Объём: {task.words_min}–{task.words_max} слов. В фокусе: {focus}.\n\n"
        "Пришли текст одним сообщением. Разберу по четырём критериям: "
        "задача, связность, лексика, грамматика.",
        MAIN_KEYBOARD,
    )


def handle_writing_text(ctx: Context, user: User, text: str) -> None:
    task_id = str(user.state_data.get("task_id") or "")
    task = None
    for tasks in ctx.curriculum.writing.values():
        for candidate in tasks:
            if candidate.id == task_id:
                task = candidate
    if task is None:
        ctx.reset_state(user)
        ctx.say(user, "Задание потерялось. Возьми новое кнопкой «✍️ Письмо».")
        return

    words = len(text.split())
    minimum = max(20, int(task.words_min * 0.7))
    if words < minimum:
        ctx.say(
            user,
            f"Пока {words} слов, для разбора нужно хотя бы {minimum}. Допиши и пришли целиком.",
        )
        return

    llm = ctx.require_llm(user)
    if llm is None:
        return
    ctx.working(user, "Разбираю текст по четырём критериям, это до минуты.")
    try:
        assessment = assess_writing(llm, user.user_id, task, text, user.level or "A2")
    except LLMError as exc:
        LOGGER.warning("Разбор письма не удался: %s", exc)
        ctx.say(user, "Не получилось разобрать текст. Попробуй ещё раз через минуту.")
        return

    report = format_assessment(assessment, ctx.curriculum, f"✍️ Разбор: {task.title_ru}")
    ctx.storage.add_writing(user.user_id, task.id, text, assessment.scores, report)
    for correction in assessment.corrections:
        ctx.storage.log_error(
            user.user_id, correction.category, correction.original, correction.corrected,
            correction.note, correction.pattern_id, source="writing",
        )
    if assessment.scores:
        ctx.storage.set_skill(
            user.user_id, "writing", round(assessment.average / 9 * 5), max(5, words // 20)
        )
    ctx.storage.bump_streak(user.user_id)
    ctx.reset_state(user)
    ctx.say(user, report, inline([[("Ещё задание", "write"), ("Прогресс", "progress")]]))


def callback_writing(ctx: Context, user: User, payload: str) -> str:
    command_writing(ctx, ctx.reload_user(user), "")
    return ""


# ── объяснение правила ───────────────────────────────────────────


def callback_explain(ctx: Context, user: User, payload: str) -> str:
    point = ctx.curriculum.by_code(payload.strip())
    if point is None:
        return "правило не найдено"
    llm = ctx.require_llm(user)
    if llm is None:
        return ""
    ctx.typing(user)
    prompt = (
        f"Объясни правило «{point.title_en}» ({point.level}). "
        f"Краткое описание из курса: {point.summary_ru}\n"
        f"Типичная ошибка русскоязычных: {point.ru_interference}\n"
        "Дай объяснение под уровень ученика, с парой контрастных примеров."
    )
    try:
        reply = llm.complete(
            explain_system(_context_block(ctx, user)),
            [{"role": "user", "content": prompt}],
            user_id=user.user_id,
            max_tokens=900,
        )
    except LLMError:
        ctx.say(user, "ИИ сейчас недоступен, попробуй позже.")
        return ""
    ctx.say(
        user,
        reply,
        inline([[("Тренировать", f"pr:{ctx.curriculum.point_code(point.id)}")]]),
    )
    return point.title_en[:40]
