"""Навигация: постоянная клавиатура и «умная» кнопка занятия.

Задача — убрать необходимость помнить команды и сократить путь до пользы.
Постоянная клавиатура висит внизу экрана всегда, поэтому от любого места
диалога до занятия ровно одно нажатие. Что именно запустить, решает
`choose_daily`, а не пользователь: ему незачем каждый день выбирать между
повторением и новой темой — платформа знает состояние его карточек лучше.

Интерфейсы разделены. Всё, до чего можно дойти отсюда, командой не
дублируется: две точки входа в одно действие не удобство, а лишний способ
не найти нужное. Командами остаётся только то, чего в меню нет.
"""

from __future__ import annotations

from ..context import Context
from ..storage import User
from ..telegram_api import inline, reply_keyboard


PRACTICE = "🎯 Заниматься"
SPEAKING = "🎙 Речь"
WRITING = "✍️ Письмо"
COURSE = "📚 Курс"
PROFILE = "📊 Я"

MAIN_KEYBOARD = reply_keyboard(
    [[PRACTICE], [SPEAKING, WRITING], [COURSE, PROFILE]],
    placeholder="Нажми «Заниматься» или напиши что-нибудь по-английски",
)

# Порог, после которого повторение важнее новой темы: меньше — не стоит
# прерывать движение вперёд ради трёх карточек.
REVIEW_THRESHOLD = 5
SPEAKING_GAP_DAYS = 4
LISTENING_GAP_DAYS = 4


def choose_daily(ctx: Context, user: User) -> tuple[str, str]:
    """Что запустить по кнопке «Заниматься» и одна строка объяснения.

    Порядок намеренный: сначала закрываем долг по срокам повторения, иначе
    интервальный алгоритм теряет смысл; потом дневная норма новой практики;
    и только когда всё закрыто — то, что подтянет отстающий навык.
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

    speaking_gap = ctx.storage.days_since_speaking(user.user_id)
    if speaking_gap is None:
        return "speaking", "Норма на сегодня закрыта. Давно не говорил вслух — устное задание."
    listening_gap = ctx.storage.days_since_session(user.user_id, "listening")
    if listening_gap is None:
        return "listening", "Устная практика уже была. Теперь потренируем понимание на слух."
    if speaking_gap >= SPEAKING_GAP_DAYS or listening_gap >= LISTENING_GAP_DAYS:
        if listening_gap > speaking_gap:
            return "listening", "Давно не тренировали понимание на слух — короткое аудирование."
        return "speaking", "Давно не говорил вслух — устное задание."
    return "practice", "Норма закрыта, но лишний раунд не повредит."


def start_daily(ctx: Context, user: User) -> None:
    """Одно нажатие — и человек уже отвечает на первое задание."""
    from . import dialogue, listening, speech, study

    action, reason = choose_daily(ctx, user)
    # У диагностики своё вступление, второй раз объяснять то же самое незачем.
    if action != "test":
        ctx.say(user, reason)
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


def command_menu(ctx: Context, user: User, text: str) -> None:
    counts = ctx.storage.card_counts(user.user_id)
    due = sum(value[1] for value in counts.values())
    ctx.say(user, _profile_text(ctx, user, due), MAIN_KEYBOARD)
    ctx.say(user, "Что ещё умею:", _extra_keyboard(user, due))


def _profile_text(ctx: Context, user: User, due: int) -> str:
    total, correct = ctx.storage.attempts_count(user.user_id)
    lines = [
        f"{user.display_name or 'Без имени'} · {user.level or 'уровень не определён'}"
        + (f" → {user.target_level}" if user.target_level else ""),
        f"Серия: {user.streak_days} дн. · решено заданий: {total}"
        + (f" ({round(correct * 100 / total)}% верно)" if total else ""),
    ]
    lines.append(
        f"К повторению сейчас: {due}" if due else "Всё повторено, очередь пуста."
    )
    return "\n".join(lines)


def _extra_keyboard(user: User, due: int = 0) -> dict:
    """Всё редкое живёт здесь: пять кнопок внизу закрывают ежедневное."""
    rows: list[list[tuple[str, str]]] = []
    # Повторение по требованию: в норме его запускает «🎯 Заниматься», но когда
    # очередь видна прямо над кнопками, закрыть её хочется сразу.
    if due:
        rows.append([(f"🔁 Повторить ({due})", "startreview")])
    rows += [
        [("Свободный чат", "chat"), ("Ролевой диалог", "roleplayhint")],
        [("Аудирование", "listen"), ("Произношение", "askword")],
        [("План", "plan"), ("Прогресс", "progress")],
        [("Отдел", "team")],
        [("Файл для Anki", "anki"), ("Мои данные", "export")],
        [("Уровень и диагностика", "levelpick"), ("Справка", "help")],
    ]
    if user.is_admin:
        rows.append([("Пригласить коллегу", "invitenew"), ("Пригласить админа", "invitenew:admin")])
    return inline(rows)


# ── кнопки постоянной клавиатуры ─────────────────────────────────


def handle_button(ctx: Context, user: User, text: str) -> None:
    """Выполняет нажатие кнопки. Вызывать только после `is_button`."""
    from . import dialogue, speech, study

    if text == PRACTICE:
        start_daily(ctx, user)
    elif text == SPEAKING:
        speech.command_speaking(ctx, user, "")
    elif text == WRITING:
        dialogue.command_writing(ctx, user, "")
    elif text == COURSE:
        study.show_levels(ctx, user)
    elif text == PROFILE:
        command_menu(ctx, user, "")


def is_button(text: str) -> bool:
    return text in {PRACTICE, SPEAKING, WRITING, COURSE, PROFILE}


# ── коллбэки меню ────────────────────────────────────────────────


def callback_daily(ctx: Context, user: User, payload: str) -> str:
    start_daily(ctx, ctx.reload_user(user))
    return ""


def callback_chat(ctx: Context, user: User, payload: str) -> str:
    from . import dialogue

    dialogue.command_chat(ctx, user, "")
    return ""


def callback_roleplay_hint(ctx: Context, user: User, payload: str) -> str:
    from . import dialogue

    dialogue.show_roleplay_presets(ctx, user)
    return ""


def callback_ask_word(ctx: Context, user: User, payload: str) -> str:
    ctx.storage.set_state(user.user_id, "awaiting_word", {})
    ctx.say(
        user,
        "Какое слово или фразу озвучить? Пришли следующим сообщением — "
        "верну транскрипцию и звучание с американским акцентом.",
    )
    return ""


def handle_awaiting_word(ctx: Context, user: User, text: str) -> None:
    from . import speech

    ctx.reset_state(user)
    speech.command_say(ctx, user, f"/say {text.strip()[:120]}")


def callback_help(ctx: Context, user: User, payload: str) -> str:
    from . import core

    core.command_help(ctx, user, "")
    return ""


def callback_team(ctx: Context, user: User, payload: str) -> str:
    from . import core

    core.command_team(ctx, user, "")
    return ""


def callback_anki(ctx: Context, user: User, payload: str) -> str:
    from . import core

    core.command_anki(ctx, user, "")
    return ""


def callback_export(ctx: Context, user: User, payload: str) -> str:
    from . import core

    core.command_export(ctx, user, "")
    return ""


def callback_level_pick(ctx: Context, user: User, payload: str) -> str:
    from ..content.schema import LEVELS

    grid = [(level, f"setlvl:{level}") for level in LEVELS]
    rows = [grid[index : index + 3] for index in range(0, len(grid), 3)]
    rows.append([("🎯 Пройти диагностику", "retest")])
    ctx.say(
        user,
        "Диагностика определит уровень точнее — 10–15 минут. "
        "Или поставь его вручную, если знаешь.",
        inline(rows),
    )
    return ""


def callback_invite(ctx: Context, user: User, payload: str) -> str:
    from . import core

    core.send_invite(ctx, user, payload.strip() or "member")
    return ""
