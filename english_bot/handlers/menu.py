"""Навигация: постоянная клавиатура, «умная» кнопка занятия и экран «📊 Я».

Задача — убрать необходимость помнить команды и сократить путь до пользы.
Постоянная клавиатура висит внизу экрана всегда, поэтому от любого места
диалога до занятия ровно одно нажатие. Что именно запустить, решает
`choose_daily`, а не пользователь: ему незачем каждый день выбирать между
повторением и новой темой — платформа знает состояние его карточек лучше.

Два правила, которые определяют поведение кнопок:

1. Экраны чтения («📚 Курс», «📊 Я») не трогают состояние. Заглянуть в курс
   посреди диагностики и вернуться — нормальный ход, а не потеря десяти минут.
2. Кнопка, которая начинает новое занятие, не уничтожает начатое молча: если
   терять есть что, бот спрашивает «Продолжить или начать заново».

Интерфейсы разделены. Всё, до чего можно дойти отсюда, командой не
дублируется: две точки входа в одно действие не удобство, а лишний способ
не найти нужное. Командами остаётся только то, чего в меню нет.
"""

from __future__ import annotations

from ..context import Context
from ..storage import User
from ..telegram_api import inline, reply_keyboard


PRACTICE = "🎯 Заниматься"
RESUME = "🎯 Продолжить"
SPEAKING = "🎙 Речь"
WRITING = "✍️ Письмо"
COURSE = "📚 Курс"
PROFILE = "📊 Я"

# Экраны чтения: их обработчики состояние не читают и не меняют.
READ_ONLY_BUTTONS: frozenset[str] = frozenset({COURSE, PROFILE})

# Состояния, в которых человек уже вложил усилие: диагностика теряется целиком,
# у тренировки и аудирования открыта сессия. Их нельзя закрыть без спроса.
GUARDED_STATES: frozenset[str] = frozenset({"placement", "practice", "listening"})

# Занятия, которые «🎯» продолжает вместо запуска нового: это тот же вид работы,
# который он и выбрал бы сам.
RESUMABLE_STATES: frozenset[str] = frozenset({"placement", "practice"})

# Что показать вместо «🎯 Заниматься», пока занятие не закончено.
STATE_TITLES: dict[str, str] = {
    "placement": "диагностика уровня",
    "practice": "тренировка",
    "writing": "письменное задание",
    "speaking": "устное задание",
    "listening": "аудирование",
    "roleplay": "ролевой диалог",
}


def _keyboard(first: str) -> dict:
    return reply_keyboard(
        [[first], [SPEAKING, WRITING], [COURSE, PROFILE]],
        placeholder="Нажми кнопку внизу или напиши что-нибудь по-английски",
    )


MAIN_KEYBOARD = _keyboard(PRACTICE)
RESUME_KEYBOARD = _keyboard(RESUME)


def keyboard_for(user: User) -> dict:
    """Клавиатура под состояние: пока занятие не закрыто, главная кнопка — «Продолжить».

    Подпись — подсказка, а не режим: «🎯 Продолжить» в покое просто запускает
    занятие дня, поэтому устаревшая клавиатура ничего не ломает и обновляется
    лениво, вместе с очередным сообщением, а не отдельным.
    """
    return RESUME_KEYBOARD if user.state in STATE_TITLES else MAIN_KEYBOARD


# Порог, после которого повторение важнее новой темы: меньше — не стоит
# прерывать движение вперёд ради трёх карточек.
REVIEW_THRESHOLD = 5
SPEAKING_GAP_DAYS = 4
LISTENING_GAP_DAYS = 4
WRITING_GAP_DAYS = 7


def choose_daily(ctx: Context, user: User) -> tuple[str, str]:
    """Что запустить по кнопке «Заниматься» и одна строка объяснения.

    Порядок намеренный: сначала закрываем долг по срокам повторения, иначе
    интервальный алгоритм теряет смысл; потом дневная норма новой практики;
    и только когда всё закрыто — то, что подтянет отстающий навык.

    Кнопка обязана предлагать выполнимое: режим, для которого не настроен нужный
    контур, не выбирается вовсе. Иначе на боте без синтеза речи главная кнопка
    до конца дня отвечала бы «Голос сейчас не настроен».
    """
    if not user.level:
        return "test", "Сначала определим уровень — это 10–15 минут."

    counts = ctx.storage.card_counts(user.user_id)
    due = sum(value[1] for value in counts.values())
    practiced = ctx.storage.practiced_today(user.user_id)

    if due >= REVIEW_THRESHOLD:
        return "review", f"Сегодня повторение: {due} карточек подошло по сроку."
    if not practiced:
        return "practice", f"Тренировка уровня {user.level}, 10 заданий."
    if due:
        return "review", f"Осталось {due} карточек к повторению — закроем."

    speaking_ready = ctx.transcriber is not None
    listening_ready = ctx.speaker is not None
    writing_ready = ctx.llm is not None

    speaking_gap = ctx.storage.days_since_speaking(user.user_id)
    listening_gap = ctx.storage.days_since_session(user.user_id, "listening")
    writing_gap = ctx.storage.days_since_writing(user.user_id)

    if speaking_ready and speaking_gap is None:
        return "speaking", "Норма на сегодня закрыта. Давно не говорил вслух — устное задание."
    if listening_ready and listening_gap is None:
        return "listening", "Устная практика уже была. Теперь потренируем понимание на слух."
    if writing_ready and writing_gap is None:
        return "writing", "Речь и слух закрыты. Осталось письмо — разберу по критериям."
    if speaking_ready and speaking_gap is not None and speaking_gap >= SPEAKING_GAP_DAYS:
        if listening_ready and listening_gap is not None and listening_gap > speaking_gap:
            return "listening", "Давно не тренировали понимание на слух — короткое аудирование."
        return "speaking", "Давно не говорил вслух — устное задание."
    if listening_ready and listening_gap is not None and listening_gap >= LISTENING_GAP_DAYS:
        return "listening", "Давно не тренировали понимание на слух — короткое аудирование."
    if writing_ready and writing_gap is not None and writing_gap >= WRITING_GAP_DAYS:
        return "writing", "Неделю без письма — возьмём письменное задание."
    return "practice", "Норма закрыта, но лишний раунд не повредит."


RETURN_GAP_DAYS = 3


def welcome_back(ctx: Context, user: User) -> str:
    """Строка возвращения после перерыва. Пусто — перерыва не было.

    Серия обнуляется молча, и человек видел «Серия: 1 дн.» без объяснения.
    Сказать прямо честнее, чем сделать вид, что ничего не прерывалось.
    """
    gap = ctx.storage.days_since_practice(user.user_id)
    if gap is None or gap < RETURN_GAP_DAYS:
        return ""
    counts = ctx.storage.card_counts(user.user_id)
    due = sum(value[1] for value in counts.values())
    note = f"С возвращением: перерыв {gap} дн., серия начнётся заново."
    if due:
        note += f" Накопилось {due} карточек к повторению — начнём с них."
    return note


def refuse_if_busy(ctx: Context, user: User) -> bool:
    """Новое занятие поверх идущей длинной задачи не начинаем.

    Экраны чтения — курс, профиль, прогресс — работают всегда: занятость не
    повод отнимать у человека навигацию.
    """
    job = ctx.busy(user)
    if job is None:
        return False
    ctx.say(user, f"Сначала закончу {job.stage}. Прервать — /stop.")
    return True


def start_daily(ctx: Context, user: User) -> None:
    """Одно нажатие — и человек уже отвечает на первое задание."""
    from . import dialogue, listening, speech, study

    if refuse_if_busy(ctx, user):
        return
    action, reason = choose_daily(ctx, user)
    back = welcome_back(ctx, user)
    if back:
        ctx.say(user, back)
        ctx.storage.log_event(user.user_id, "return", str(ctx.storage.days_since_practice(user.user_id)))
    # У диагностики своё вступление, второй раз объяснять то же самое незачем.
    if action != "test":
        ctx.say(user, reason)
    ctx.storage.log_event(user.user_id, "daily", action)
    if action == "test":
        study.command_test(ctx, user, "")
    elif action == "review":
        study.command_review(ctx, user, "")
    elif action == "speaking":
        speech.command_speaking(ctx, user, "")
    elif action == "listening":
        listening.command_listening(ctx, user)
    elif action == "writing":
        dialogue.command_writing(ctx, user, "")
    else:
        study.command_practice(ctx, user, "")


# ── экран «📊 Я» ─────────────────────────────────────────────────


def command_menu(ctx: Context, user: User, text: str = "") -> None:
    """Один экран: статус, следующий шаг, навыки и редкие действия командами."""
    counts = ctx.storage.card_counts(user.user_id)
    due = sum(value[1] for value in counts.values())
    ctx.say(user, _hub_text(ctx, user, due), _hub_keyboard(ctx, user, due))
    ctx.storage.log_event(user.user_id, "screen", "hub")


def _hub_text(ctx: Context, user: User, due: int) -> str:
    total, correct = ctx.storage.attempts_count(user.user_id)
    streak = ctx.storage.effective_streak(user.user_id)
    head = f"{user.display_name or 'Без имени'} · {user.level or 'уровень не определён'}"
    if user.target_level:
        head += f" → {user.target_level}"
    lines = [
        head,
        f"Серия: {streak} дн. · решено заданий: {total}"
        + (f" ({round(correct * 100 / total)}% верно)" if total else ""),
        "",
    ]
    back = welcome_back(ctx, user)
    if back:
        lines.append(back)
        lines.append("")
    if user.level:
        _, reason = choose_daily(ctx, user)
        lines.append(f"Сейчас полезнее всего: {reason[0].lower() + reason[1:]}")
    else:
        lines.append("Уровень ещё не определён — начнём с диагностики.")
    lines.append("")
    lines.append(
        "Редкое — командами: /help вопрос · /say слово · /learn правило · "
        "/roleplay свой сценарий · /export · /anki"
    )
    return "\n".join(lines)


def _hub_keyboard(ctx: Context, user: User, due: int = 0) -> dict:
    """Первый ряд — следующий шаг, дальше навыки, ниже отдел и роли.

    Справка, произношение и свободный чат отсюда убраны намеренно: у них есть
    команда с аргументом, а обычный текст в покое и так уходит в чат.
    """
    from .dialogue import ROLEPLAY_PRESETS

    rows: list[list[tuple[str, str]]] = []
    if due:
        rows.append([(f"🔁 Повторить ({due})", "startreview")])
    else:
        rows.append([("🎯 Тренировка уровня", "startpractice")])
    preset_title = ROLEPLAY_PRESETS[0][0] if ROLEPLAY_PRESETS else "Ролевой диалог"
    rows += [
        [("🎧 Аудирование", "listen"), (f"🎭 {preset_title}", "rp:0")],
        [("🎭 Ещё сценарии", "roleplayhint"), ("🧭 План", "plan")],
        [("📈 Прогресс", "progress"), ("🧪 Диагностика", "retest")],
        [("⚙ Уровень", "levelpick"), ("🏢 Отдел", "team")],
    ]
    if user.is_admin:
        rows.append([("Пригласить коллегу", "invitenew")])
        rows.append([("Пригласить админа", "invitenew:admin")])
    return inline(rows)


# ── кнопки постоянной клавиатуры ─────────────────────────────────


def handle_button(ctx: Context, user: User, text: str) -> None:
    """Выполняет нажатие кнопки. Вызывать только после `is_button`."""
    if text in READ_ONLY_BUTTONS:
        _open_screen(ctx, user, text)
        return
    if text in (PRACTICE, RESUME) and user.state in RESUMABLE_STATES:
        # Диагностика и тренировка — это и есть «занятие дня»: главная кнопка
        # возвращает в них, а не начинает заново поверх начатого.
        resume_current(ctx, user)
        return
    if text == RESUME and not resume_current(ctx, user):
        start_daily(ctx, user)
        return
    if text == RESUME:
        return

    # Отказ до `close_active`: иначе занятие закрылось бы, а новое не началось.
    if refuse_if_busy(ctx, user):
        return
    target = {PRACTICE: "daily", SPEAKING: "speaking", WRITING: "writing"}[text]
    if user.state in GUARDED_STATES:
        ask_switch(ctx, user, target)
        return
    close_active(ctx, user)
    start_target(ctx, ctx.reload_user(user), target)


def _open_screen(ctx: Context, user: User, text: str) -> None:
    """Курс и профиль читают состояние, но не меняют его — сбрасывать нечего."""
    from . import study

    if text == COURSE:
        study.show_levels(ctx, user)
    else:
        command_menu(ctx, user, "")


def is_button(text: str) -> bool:
    return text in {PRACTICE, RESUME, SPEAKING, WRITING, COURSE, PROFILE}


# ── переключение занятий ─────────────────────────────────────────


def ask_switch(ctx: Context, user: User, target: str) -> None:
    """Спрашивает, прежде чем закрыть начатое: усилие теряется молча только раз."""
    title = STATE_TITLES.get(user.state, "занятие")
    labels = {
        "daily": "🎯 Новое занятие",
        "speaking": "🎙 Устное задание",
        "writing": "✍️ Письменное задание",
    }
    ctx.say(
        user,
        f"Сейчас идёт {title}. Продолжить или начать заново?",
        inline([[("▶️ Продолжить", "rsm")], [(labels[target], f"sw:{target}")]]),
    )
    ctx.storage.log_event(user.user_id, "switch_asked", user.state)


def resume_current(ctx: Context, user: User) -> bool:
    """Возвращает человека к текущему заданию. False — возвращать нечего."""
    from . import dialogue, listening, speech, study

    state = user.state
    if state == "placement":
        study.resume_placement(ctx, user)
    elif state == "practice":
        study.resume_practice(ctx, user)
    elif state == "writing":
        dialogue.remind_writing(ctx, user)
    elif state == "speaking":
        speech.remind_speaking(ctx, user)
    elif state == "listening":
        listening.remind_listening(ctx, user)
    elif state == "roleplay":
        ctx.say(user, "Продолжаем диалог — пиши следующую реплику по-английски.")
    else:
        return False
    ctx.storage.log_event(user.user_id, "resume", state)
    return True


def close_active(ctx: Context, user: User) -> None:
    """Закрывает текущее занятие, ничего не теряя молча.

    Тренировка засчитывается: её ответы уже сохранены, и терять серию и дневную
    норму из-за переключения было бы обиднее всего.
    """
    from . import study

    # Занятие закрывается — значит его фоновый разбор больше не нужен.
    ctx.cancel_job(user)
    state = user.state
    if state == "idle":
        return
    if state == "practice":
        study.finish_current_practice(ctx, user)
        return
    if state == "listening":
        session_id = int(user.state_data.get("session_id") or 0)
        if session_id:
            ctx.storage.finish_session(session_id, items=0, correct=0)
    if state in STATE_TITLES:
        ctx.say(user, f"Закрыл: {STATE_TITLES[state]}.", MAIN_KEYBOARD)
    ctx.reset_state(user)
    ctx.storage.log_event(user.user_id, "abandon", state)


def start_target(ctx: Context, user: User, target: str) -> None:
    from . import dialogue, speech

    if refuse_if_busy(ctx, user):
        return
    if target == "speaking":
        speech.command_speaking(ctx, user, "")
    elif target == "writing":
        dialogue.command_writing(ctx, user, "")
    else:
        start_daily(ctx, user)


# ── коллбэки меню ────────────────────────────────────────────────


def callback_daily(ctx: Context, user: User, payload: str) -> str:
    start_daily(ctx, ctx.reload_user(user))
    return ""


def callback_resume(ctx: Context, user: User, payload: str) -> str:
    if not resume_current(ctx, ctx.reload_user(user)):
        ctx.say(user, "Возвращаться не к чему — нажми «🎯 Заниматься».", MAIN_KEYBOARD)
    return ""


def callback_switch(ctx: Context, user: User, payload: str) -> str:
    target = payload.strip() or "daily"
    if target not in {"daily", "speaking", "writing"}:
        return "не понял выбор"
    close_active(ctx, ctx.reload_user(user))
    start_target(ctx, ctx.reload_user(user), target)
    return ""


def callback_roleplay_hint(ctx: Context, user: User, payload: str) -> str:
    from . import dialogue

    dialogue.show_roleplay_presets(ctx, user)
    return ""


def callback_team(ctx: Context, user: User, payload: str) -> str:
    from . import core

    core.command_team(ctx, user, "")
    return ""


def callback_level_pick(ctx: Context, user: User, payload: str) -> str:
    """Только ручная установка: диагностика вынесена отдельной кнопкой хаба."""
    from ..content.schema import LEVELS

    grid = [(level, f"setlvl:{level}") for level in LEVELS]
    rows = [grid[index : index + 3] for index in range(0, len(grid), 3)]
    ctx.edit(
        user,
        "Поставь уровень вручную, если знаешь его. Точнее определит «🧪 Диагностика» "
        "— это 10–15 минут.",
        inline(rows),
    )
    return ""


def callback_invite(ctx: Context, user: User, payload: str) -> str:
    from . import core

    core.send_invite(ctx, user, payload.strip() or "member")
    return ""
