"""Базовые команды: вход, справка, профиль, прогресс, команда, администрирование."""

from __future__ import annotations

import logging
from pathlib import Path

from ..content.schema import LEVELS
from ..context import Context
from ..learning.placement import next_level
from ..learning.progress import (
    anki_export,
    build_export,
    progress_report,
    study_plan,
    team_board,
    write_export,
)
from ..storage import User, utc_now
from ..telegram_api import inline, inline_grid
from .menu import MAIN_KEYBOARD


LOGGER = logging.getLogger(__name__)

# Список для меню команд Telegram. Дублировать здесь кнопки нельзя: у каждого
# действия должна быть ровно одна точка входа, иначе человек всё равно не знает,
# чем пользоваться. /admin намеренно отсутствует — он не для всех.
COMMANDS: list[tuple[str, str]] = [
    ("start", "начать заново и вернуть кнопки"),
    ("say", "произношение слова или фразы: /say schedule"),
    ("learn", "найти правило по названию: /learn present perfect"),
    ("roleplay", "диалог по своему сценарию"),
    ("stop", "прервать текущее занятие"),
    ("privacy", "участие в таблице отдела"),
    ("forget", "удалить свои учебные данные"),
]

HELP_TEXT = "\n".join(
    [
        "English Lab — платформа отдела для английского.",
        "",
        "Всё основное — кнопками внизу экрана:",
        "🎯 Заниматься — сама выбирает, что тебе сегодня полезнее, и запускает",
        "🎙 Речь · ✍️ Письмо — задание и разбор",
        "📚 Курс — уровни, темы, правила",
        "📊 Я — профиль, прогресс, план, произношение, отдел, выгрузки",
        "",
        "Команды — только то, чего в кнопках нет:",
    ]
    + [f"/{name} — {text}" for name, text in COMMANDS]
    + [
        "",
        "Просто напиши мне что-нибудь по-английски — отвечу и разберу ошибки.",
    ]
)

WELCOME = (
    "English Lab на связи.\n\n"
    "Курс грамматики A1–C2 по темам, тренажёр с интервальным повторением, "
    "устная практика с расшифровкой, разбор письма и произношение с озвучкой.\n\n"
    "Внизу экрана — кнопки, команды помнить не нужно. Главная — «🎯 Заниматься»: "
    "она сама решает, что тебе сегодня полезнее, и сразу это запускает."
)


def command_start(ctx: Context, user: User, text: str) -> None:
    """Первый вход: владелец по claim-коду, остальные по приглашению."""
    ctx.reset_state(user)
    tail = []
    if not user.level:
        tail.append("Первым делом кнопка определит твой уровень — это 10–15 минут.")
    ctx.say(user, "\n\n".join([WELCOME, *tail]), MAIN_KEYBOARD)


def command_help(ctx: Context, user: User, text: str) -> None:
    ctx.say(user, HELP_TEXT)


def callback_set_level(ctx: Context, user: User, payload: str) -> str:
    level = payload.strip().upper()
    if level not in LEVELS:
        return "неизвестный уровень"
    ctx.storage.update_user(user.user_id, level=level, target_level=next_level(level))
    ctx.say(user, f"Уровень: {level}, цель {next_level(level)}. Дальше — «🎯 Заниматься».")
    return f"уровень {level}"


def command_progress(ctx: Context, user: User, text: str) -> None:
    ctx.say(
        user,
        progress_report(ctx.storage, ctx.curriculum, user),
        inline([[("План", "plan"), ("Повторение", "startreview")]]),
    )


def command_plan(ctx: Context, user: User, text: str) -> None:
    if not user.level:
        ctx.say(user, "Сначала определим уровень — нажми «🎯 Заниматься».")
        return
    ctx.say(user, study_plan(ctx.storage, ctx.curriculum, user))


def command_privacy(ctx: Context, user: User, text: str) -> None:
    new_value = not user.share_progress
    ctx.storage.update_user(user.user_id, share_progress=int(new_value))
    ctx.say(
        user,
        "Теперь ты в таблице отдела." if new_value else "Убрал тебя из таблицы отдела.",
    )


def command_team(ctx: Context, user: User, text: str) -> None:
    ctx.say(user, team_board(ctx.storage, ctx.curriculum))


def command_export(ctx: Context, user: User, text: str) -> None:
    # Ключ персональный: общая настройка показывала бы в моём документе время чужой выгрузки.
    ctx.storage.set_setting(f"last_export:{user.user_id}", utc_now())
    content = build_export(ctx.storage, ctx.curriculum, user)
    write_export(ctx.settings.export_dir, user.user_id, content)
    ctx.say(user, content)


def command_anki(ctx: Context, user: User, text: str) -> None:
    payload = anki_export(ctx.storage, ctx.curriculum, user.user_id)
    if payload.count("\n") <= 3:
        ctx.say(
            user,
            "Пока нечего выгружать: карточки появляются после занятий и разборов. "
            "Нажми «🎯 Заниматься».",
        )
        return
    path: Path = ctx.settings.export_dir / f"anki-{user.user_id}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    ctx.telegram.send_document(
        user.chat_id,
        path,
        "Импорт в Anki: File → Import, разделитель Tab, поля Front/Back/Tags, "
        "«Allow HTML» выключен.",
    )


def command_forget(ctx: Context, user: User, text: str) -> None:
    if text.partition(" ")[2].strip() != "YES":
        ctx.say(
            user,
            "Это удалит твой прогресс, карточки, журнал ошибок, письма и голосовые. "
            "Учётка и роль останутся. Подтверди точно так: /forget YES",
        )
        return
    paths = ctx.storage.voice_paths(user.user_id)
    ctx.storage.delete_learning_data(user.user_id)
    _delete_voice_files(ctx, paths)
    (ctx.settings.export_dir / f"learner-{user.user_id}.md").unlink(missing_ok=True)
    (ctx.settings.export_dir / f"anki-{user.user_id}.tsv").unlink(missing_ok=True)
    ctx.say(
        user,
        "Учебные данные удалены. Начать заново — «🎯 Заниматься».",
        MAIN_KEYBOARD,
    )


def _delete_voice_files(ctx: Context, paths: list[Path]) -> None:
    root = ctx.settings.voice_dir.resolve()
    for raw in paths:
        path = raw.resolve()
        if not path.is_relative_to(root):
            LOGGER.warning("Пропущен путь вне VOICE_DIR")
            continue
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass


def command_stop(ctx: Context, user: User, text: str) -> None:
    ctx.reset_state(user)
    ctx.say(user, "Остановился. Кнопки внизу — или просто напиши мне по-английски.",
            MAIN_KEYBOARD)


# ── администрирование ────────────────────────────────────────────


def send_invite(ctx: Context, user: User, role: str = "member") -> None:
    """Выдаёт одноразовый код. Точка входа одна — кнопки на экране «📊 Я»."""
    if not user.is_admin:
        ctx.say(user, "Коды приглашений создаёт владелец или админ.")
        return
    role = "admin" if role == "admin" else "member"
    code = ctx.storage.create_invite(user.user_id, role=role)
    username = ctx.storage.get_setting("bot_username") or ""
    link = f"https://t.me/{username}?start={code}" if username else f"/start {code}"
    ctx.say(
        user,
        f"Одноразовый код на роль «{role}», действует 14 дней:\n\n{link}\n\n"
        "Отправь коллеге. После использования код сгорает.",
    )


def command_admin(ctx: Context, user: User, text: str) -> None:
    if not user.is_admin:
        ctx.say(user, "Раздел только для владельца и админов.")
        return
    users = ctx.storage.all_users()
    invites = ctx.storage.invites()
    unused = sum(1 for row in invites if row["used_by"] is None)
    curriculum = ctx.curriculum
    lines = [
        "Администрирование",
        "",
        f"Пользователей: {len(users)}",
        f"Неиспользованных приглашений: {unused}",
        "",
        f"Контент: {len(curriculum.points)} тем, {len(curriculum.exercises)} упражнений, "
        f"{sum(len(rows) for rows in curriculum.vocabulary.values())} слов",
        f"Ошибок загрузки контента: {len(curriculum.load_errors)}",
        "",
        f"Текстовый ИИ: {'включён (' + ctx.settings.llm_provider + ')' if ctx.llm else 'выключен'}",
        f"Распознавание речи: "
        f"{ctx.settings.speech_backend if ctx.transcriber else 'выключено'}",
        f"Синтез речи: {ctx.settings.speech_backend if ctx.speaker else 'выключен'}",
        f"Лимит обращений к ИИ: {ctx.settings.daily_ai_calls} в сутки на человека",
        "",
        "Люди:",
    ]
    for row in users:
        lines.append(
            f"• {row.display_name or 'без имени'} — {row.role}, "
            f"{row.level or 'уровень не определён'}, серия {row.streak_days}"
        )
    ctx.say(user, "\n".join(lines))
