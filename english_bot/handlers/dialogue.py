"""Свободное общение, ролевые диалоги и письменные работы."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..ai.llm import LLMError
from ..ai.prompts import dialogue_system, explain_system, learner_block, tutor_system
from ..context import AI_OFF_TEXT, Context, Ticket
from ..learning.progress import mastery_map, skill_level
from ..learning.scoring import assess_writing, format_assessment
from ..runtime import Job
from ..storage import User
from ..telegram_api import inline, inline_grid
from .menu import MAIN_KEYBOARD, RESUME_KEYBOARD


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


def _job_context(ctx: Context, ticket: Ticket) -> tuple[User, str] | None:
    """Свежий профиль и блок о нём для запроса из фоновой задачи.

    Снимок `User`, сделанный минуту назад, к моменту вызова модели описывает
    уже не того человека: уровень и серия могли поменяться.
    """
    user = ctx.who(ticket)
    if user is None:
        return None
    return user, _context_block(ctx, user)


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


def handle_free_text(ctx: Context, user: User, text: str) -> None:
    """Свободный чат — это просто сообщение в покое, отдельной кнопки у него нет."""
    if ctx.llm is None:
        ctx.storage.add_message(user.user_id, "user", text)
        ctx.say(
            user,
            "Сообщение сохранил. Пока ИИ не настроен, работают курс, тренировка "
            "и повторение — нажми «🎯 Заниматься».",
            inline([[("🎯 Заниматься", "daily")]]),
        )
        return
    ctx.background(
        ctx.ticket(user),
        "ответ в чате",
        _free_text_job,
        text,
        notice=ctx.slow_note("Думаю над ответом, это до минуты."),
    )


def _free_text_job(job: Job, ctx: Context, ticket: Ticket, text: str) -> None:
    # Реплику записываем уже в задаче: при отказе «сначала закончу» история не
    # должна пополняться сообщением, на которое никто не ответит.
    ctx.storage.add_message(ticket.user_id, "user", text)
    llm = ctx.claim_llm(ticket)
    if llm is None:
        return
    found = _job_context(ctx, ticket)
    if found is None:
        return
    user, block = found
    history = ctx.storage.recent_messages(ticket.user_id, limit=12)
    job.checkpoint()
    try:
        reply = llm.complete(
            tutor_system(block), history, user_id=ticket.user_id, max_tokens=1000
        )
    except LLMError as exc:
        LOGGER.warning("Наставник не ответил: %s", exc)
        ctx.tell(ticket, "ИИ сейчас недоступен. Сообщение сохранил, попробуй ещё раз.")
        return
    job.checkpoint()
    ctx.storage.add_message(ticket.user_id, "assistant", reply)
    ctx.storage.trim_messages(ticket.user_id)
    _log_corrections(ctx, user, reply, source="chat")
    ctx.tell(ticket, reply)


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
    if ctx.llm is None:
        ctx.say(user, AI_OFF_TEXT)
        return
    ctx.edit(
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
    # Сначала проверка, потом состояние: иначе отказ ИИ оставлял человека в
    # невидимом режиме, где каждое сообщение получало один и тот же отказ.
    if ctx.llm is None:
        ctx.say(user, AI_OFF_TEXT)
        return
    scenario = scenario[:200]
    ctx.storage.set_state(user.user_id, "roleplay", {"scenario": scenario})
    ctx.log_start(user, "roleplay")
    # Ограда — сам сценарий: пока модель придумывает первую реплику, человек мог
    # успеть начать другой диалог, и откат по ошибке закрыл бы уже его.
    ticket = Ticket(user.user_id, user.chat_id, "roleplay", "scenario", scenario)
    ctx.background(ticket, "первую реплику диалога", _roleplay_start_job, scenario)


def _roleplay_start_job(job: Job, ctx: Context, ticket: Ticket, scenario: str) -> None:
    llm = ctx.claim_llm(ticket)
    if llm is None:
        ctx.release_state(ticket)
        return
    found = _job_context(ctx, ticket)
    if found is None:
        ctx.release_state(ticket)
        return
    user, block = found
    job.checkpoint()
    try:
        reply = llm.complete(
            dialogue_system(block, scenario, user.level or "A2"),
            [{"role": "user", "content": "Start the roleplay with your first line."}],
            user_id=ticket.user_id,
            max_tokens=500,
        )
    except LLMError:
        ctx.tell(ticket, "Не получилось запустить диалог. Попробуй ещё раз.")
        ctx.release_state(ticket)
        return
    job.checkpoint()
    # Ограда и на успехе: пока модель писала первую реплику, человек мог уйти в
    # другое занятие, и вываливать в него чужой диалог нельзя.
    user = ctx.who(ticket)
    if user is None or user.state != "roleplay" or str(
        user.state_data.get("scenario") or ""
    ) != scenario:
        LOGGER.info("Диалог уже закрыт, первая реплика не нужна")
        return
    ctx.storage.add_message(ticket.user_id, "assistant", reply)
    ctx.tell(
        ticket, f"Сценарий: {scenario}\n\n{reply}", inline([[("Закончить", "endroleplay")]])
    )


def handle_roleplay_text(ctx: Context, user: User, text: str) -> None:
    if ctx.llm is None:
        ctx.say(user, AI_OFF_TEXT)
        return
    scenario = str(user.state_data.get("scenario") or "conversation")
    if not ctx.background(ctx.ticket(user), "ответ по диалогу", _roleplay_job, scenario, text):
        return
    ctx.typing(user)


def _roleplay_job(job: Job, ctx: Context, ticket: Ticket, scenario: str, text: str) -> None:
    ctx.storage.add_message(ticket.user_id, "user", text)
    llm = ctx.claim_llm(ticket)
    if llm is None:
        return
    found = _job_context(ctx, ticket)
    if found is None:
        return
    user, block = found
    history = ctx.storage.recent_messages(ticket.user_id, limit=14)
    job.checkpoint()
    try:
        reply = llm.complete(
            dialogue_system(block, scenario, user.level or "A2"),
            history,
            user_id=ticket.user_id,
            max_tokens=800,
        )
    except LLMError:
        ctx.tell(ticket, "ИИ сейчас недоступен, попробуй ещё раз.")
        return
    job.checkpoint()
    ctx.storage.add_message(ticket.user_id, "assistant", reply)
    _log_corrections(ctx, user, reply, source="roleplay")
    ctx.tell(ticket, reply, inline([[("Закончить", "endroleplay")]]))


def callback_end_roleplay(ctx: Context, user: User, payload: str) -> str:
    ctx.reset_state(user)
    ctx.say(
        user,
        "Диалог закончен. Правки из него уже в журнале ошибок.",
        inline([[("📈 Прогресс", "progress"), ("🎯 Заниматься", "daily")]]),
    )
    return "закончили"


# ── письмо ───────────────────────────────────────────────────────


def command_writing(ctx: Context, user: User, text: str) -> None:
    # Просить сто слов и только потом сказать, что разбирать их нечем, —
    # худший способ потратить чужое время. Проверяем до выдачи задания.
    if ctx.llm is None:
        ctx.say(user, AI_OFF_TEXT, MAIN_KEYBOARD)
        return
    # Тот же принцип, что и в речи: объём письменного задания подбирается по
    # тому, что человек уже написал, а не по узнаванию грамматики.
    level = skill_level(ctx.storage, user, "writing")
    done = {str(row["task_id"]) for row in ctx.storage.writings(user.user_id, limit=50)}
    task = ctx.curriculum.pick_writing(level, done, ctx.rng)
    if task is None:
        ctx.say(user, f"Для уровня {level} письменных заданий пока нет.")
        return
    ctx.storage.set_state(user.user_id, "writing", {"task_id": task.id})
    ctx.say(user, _writing_task_text(task), RESUME_KEYBOARD)
    ctx.log_start(user, "writing")


def _writing_task_text(task: Any) -> str:
    focus = ", ".join(task.focus) if task.focus else "свободно"
    return (
        f"✍️ {task.title_ru} · {task.level}\n\n"
        f"{task.prompt_en}\n\n"
        f"Как писать: {task.guidance_ru}\n"
        f"Объём: {task.words_min}–{task.words_max} слов. В фокусе: {focus}.\n\n"
        "Пришли текст одним сообщением. Разберу по четырём критериям: "
        "задача, связность, лексика, грамматика."
    )


def remind_writing(ctx: Context, user: User) -> None:
    """Возврат к письму: показываем то же задание, а не выдаём новое."""
    task_id = str(user.state_data.get("task_id") or "")
    task = _find_writing_task(ctx, task_id)
    if task is None:
        ctx.reset_state(user)
        ctx.say(user, "Задание потерялось. Возьми новое кнопкой «✍️ Письмо».", MAIN_KEYBOARD)
        return
    ctx.say(user, "Продолжаем письменное задание.\n\n" + _writing_task_text(task), RESUME_KEYBOARD)


def _find_writing_task(ctx: Context, task_id: str):
    for tasks in ctx.curriculum.writing.values():
        for candidate in tasks:
            if candidate.id == task_id:
                return candidate
    return None


def handle_writing_text(ctx: Context, user: User, text: str) -> None:
    task_id = str(user.state_data.get("task_id") or "")
    task = _find_writing_task(ctx, task_id)
    if task is None:
        ctx.reset_state(user)
        ctx.say(user, "Задание потерялось. Возьми новое кнопкой «✍️ Письмо».", MAIN_KEYBOARD)
        return

    words = len(text.split())
    minimum = max(20, int(task.words_min * 0.7))
    if words < minimum:
        ctx.say(
            user,
            f"Пока {words} слов, для разбора нужно хотя бы {minimum}. Допиши и пришли целиком.",
        )
        return

    if ctx.llm is None:
        # Задание остаётся: лимит откроется завтра, и тот же текст можно прислать снова.
        ctx.say(user, "Текст сохраню за тобой — пришли его ещё раз, когда ИИ снова заработает.")
        return
    # Ограда — само задание: за минуту разбора человек мог взять другое.
    ctx.background(
        ctx.ticket(user, "task_id", task.id),
        "разбор письма",
        _writing_job,
        task.id,
        text,
        words,
        notice=ctx.slow_note("Разбираю текст по четырём критериям, это до минуты."),
    )


def _writing_job(
    job: Job, ctx: Context, ticket: Ticket, task_id: str, text: str, words: int
) -> None:
    task = _find_writing_task(ctx, task_id)
    if task is None:
        ctx.tell(ticket, "Задание к тексту потерялось. Возьми новое кнопкой «✍️ Письмо».")
        ctx.release_state(ticket)
        return
    llm = ctx.claim_llm(ticket)
    if llm is None:
        return
    user = ctx.who(ticket)
    job.checkpoint()
    try:
        assessment = assess_writing(
            llm, ticket.user_id, task, text, (user.level if user else "") or "A2"
        )
    except LLMError as exc:
        LOGGER.warning("Разбор письма не удался: %s", exc)
        ctx.tell(ticket, "Не получилось разобрать текст. Попробуй ещё раз через минуту.")
        return
    job.checkpoint()

    report = format_assessment(assessment, ctx.curriculum, f"✍️ Разбор: {task.title_ru}")
    ctx.storage.add_writing(ticket.user_id, task.id, text, assessment.scores, report)
    for correction in assessment.corrections:
        ctx.storage.log_error(
            ticket.user_id, correction.category, correction.original, correction.corrected,
            correction.note, correction.pattern_id, source="writing",
        )
    if assessment.scores:
        ctx.storage.set_skill(
            ticket.user_id, "writing", round(assessment.average / 9 * 5), max(5, words // 20)
        )
    ctx.storage.bump_streak(ticket.user_id)
    if ctx.release_state(ticket):
        ctx.storage.log_event(ticket.user_id, "finish", "writing")
    ctx.tell(ticket, report, inline([[("Ещё задание", "write"), ("📈 Прогресс", "progress")]]))


def callback_writing(ctx: Context, user: User, payload: str) -> str:
    command_writing(ctx, ctx.reload_user(user), "")
    return ""


# ── объяснение правила ───────────────────────────────────────────


def callback_explain(ctx: Context, user: User, payload: str) -> str:
    point = ctx.curriculum.by_code(payload.strip())
    if point is None:
        return "правило не найдено"
    if ctx.llm is None:
        ctx.say(user, AI_OFF_TEXT)
        return ""
    # Подпись возвращается сразу: объяснение придёт отдельным сообщением, а
    # кнопка не должна крутиться, пока модель пишет.
    if not ctx.background(ctx.ticket(user), "объяснение правила", _explain_job, point.id):
        return ""
    ctx.typing(user)
    return point.title_en[:40]


def _explain_job(job: Job, ctx: Context, ticket: Ticket, point_id: str) -> None:
    point = ctx.curriculum.points.get(point_id)
    if point is None:
        ctx.tell(ticket, "Правило потерялось — открой его заново через «📚 Курс».")
        return
    llm = ctx.claim_llm(ticket)
    if llm is None:
        return
    found = _job_context(ctx, ticket)
    if found is None:
        return
    _, block = found
    prompt = (
        f"Объясни правило «{point.title_en}» ({point.level}). "
        f"Краткое описание из курса: {point.summary_ru}\n"
        f"Типичная ошибка русскоязычных: {point.ru_interference}\n"
        "Дай объяснение под уровень ученика, с парой контрастных примеров."
    )
    job.checkpoint()
    try:
        reply = llm.complete(
            explain_system(block),
            [{"role": "user", "content": prompt}],
            user_id=ticket.user_id,
            max_tokens=900,
        )
    except LLMError:
        ctx.tell(ticket, "ИИ сейчас недоступен, попробуй позже.")
        return
    job.checkpoint()
    ctx.tell(
        ticket, reply, inline([[("Тренировать", f"pr:{ctx.curriculum.point_code(point.id)}")]])
    )
