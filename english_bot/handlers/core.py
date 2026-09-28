"""Базовые команды: вход, справка, профиль, прогресс, команда, администрирование."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..ai.llm import LLMError
from ..ai.prompts import platform_help_system
from ..content.schema import LEVELS
from ..context import Context, Ticket
from ..learning.placement import next_level
from ..learning.progress import (
    anki_export,
    build_export,
    progress_report,
    study_plan,
    team_board,
    write_export,
)
from ..platform_help import KnowledgeError, find_articles, knowledge_excerpt
from ..runtime import Job
from ..storage import User, utc_now
from ..telegram_api import TelegramError, inline
from .menu import MAIN_KEYBOARD


LOGGER = logging.getLogger(__name__)

# Список для меню команд Telegram. Дублировать здесь кнопки нельзя: у каждого
# действия должна быть ровно одна точка входа, иначе человек всё равно не знает,
# чем пользоваться. /admin намеренно отсутствует — он не для всех.
COMMANDS: list[tuple[str, str]] = [
    ("start", "вернуть кнопки, не прерывая занятие"),
    ("help", "вопрос о платформе: /help как пройти диагностику?"),
    ("say", "произношение слова или фразы: /say schedule"),
    ("learn", "найти правило по названию: /learn present perfect"),
    ("roleplay", "ролевой диалог: готовые ситуации или свой сценарий"),
    ("export", "выгрузить свои учебные данные текстом"),
    ("anki", "файл карточек для Anki"),
    ("stop", "прервать текущее занятие"),
    ("privacy", "участие в таблице отдела"),
    ("forget", "удалить свои учебные данные"),
]

HELP_TEXT = "\n".join(
    [
        "English Lab — платформа отдела АБП для английского.",
        "",
        "Всё ежедневное — кнопками внизу экрана:",
        "🎯 Заниматься — сама выбирает, что тебе сегодня полезнее, и запускает",
        "🎙 Речь · ✍️ Письмо — задание и разбор",
        "📚 Курс — уровни, темы, правила",
        "📊 Я — статус и следующий шаг: повторение, аудирование, ролевой диалог, "
        "план, прогресс, диагностика, уровень, отдел",
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
    "аудирование, устная практика с расшифровкой, разбор письма и произношение.\n\n"
    "Внизу экрана — кнопки, команды помнить не нужно. Главная — «🎯 Заниматься»: "
    "она сама решает, что тебе сегодня полезнее, и сразу это запускает.\n\n"
    "Хочешь просто поговорить — напиши мне по-английски, отвечу и разберу ошибки."
)


def command_start(ctx: Context, user: User, text: str) -> None:
    """Первый вход и возврат клавиатуры.

    Состояние не сбрасывается: документация советует `/start`, когда клавиатура
    потерялась, и раньше этот совет стирал незаконченную диагностику.
    """
    from .menu import keyboard_for

    tail = []
    if not user.level:
        tail.append("Первым делом кнопка определит твой уровень — это 10–15 минут.")
    missing = _unavailable_features(ctx)
    if missing:
        tail.append(f"Сейчас недоступно: {missing}. Курс, тренировка и повторение работают.")
    ctx.say(user, "\n\n".join([WELCOME, *tail]), keyboard_for(user))
    if user.state != "idle":
        ctx.say(
            user,
            "Занятие не закрыто — можно вернуться к нему.",
            inline([[("▶️ Продолжить", "rsm")]]),
        )


def _unavailable_features(ctx: Context) -> str:
    """Что не заработает при текущей конфигурации — сказать сразу, а не после усилия."""
    missing = []
    if ctx.llm is None:
        missing.append("разбор письма, ролевой диалог и свободный чат")
    if ctx.transcriber is None:
        missing.append("устная практика")
    if ctx.speaker is None:
        missing.append("аудирование и озвучка")
    return ", ".join(missing)


def command_help(ctx: Context, user: User, text: str) -> None:
    question = text.partition(" ")[2].strip()
    if not question:
        ctx.say(user, HELP_TEXT)
        return
    if len(question) > 500:
        ctx.say(user, "Сократи вопрос до 500 символов — так я точнее найду нужную справку.")
        return
    try:
        articles = find_articles(question)
    except KnowledgeError:
        LOGGER.exception("Не удалось загрузить базу знаний платформы")
        ctx.say(user, "Справка временно недоступна. Попробуй позже или обратись к владельцу.")
        return
    if not articles:
        ctx.say(
            user,
            "Я отвечаю только о том, как пользоваться English Lab. "
            "Например: /help как пройти диагностику?",
        )
        return

    fallback = articles[0].answer_ru
    if ctx.llm is None:
        # Статья уже найдена локально: прятать её за общим отказом бессмысленно
        # именно тогда, когда справка нужнее всего.
        ctx.say(user, f"ИИ не настроен, но вот ближайшая справка:\n\n{fallback}")
        return
    # Вопрос «почему бот не отвечает» задают как раз тогда, когда идёт разбор.
    # Отказывать здесь нельзя: отдаём готовую статью без обращения к модели.
    if ctx.busy(user) is not None:
        ctx.say(user, f"Сейчас доделываю прошлое. Вот ближайшая справка:\n\n{fallback}")
        return
    excerpt = knowledge_excerpt(articles)
    ctx.background(
        ctx.ticket(user),
        "ответ по справке",
        _help_job,
        question,
        excerpt,
        fallback,
        notice=ctx.slow_note("Ищу ответ в справке платформы, это до минуты."),
    )


def _help_job(
    job: Job, ctx: Context, ticket: Ticket, question: str, excerpt: str, fallback: str
) -> None:
    llm = ctx.claim_llm(ticket)
    if llm is None:
        ctx.tell(ticket, f"Ближайшая справка из базы:\n\n{fallback}")
        return
    job.checkpoint()
    try:
        reply = llm.complete(
            platform_help_system(excerpt),
            [{"role": "user", "content": question}],
            user_id=ticket.user_id,
            max_tokens=500,
            temperature=0.1,
        ).strip()
    except LLMError as exc:
        LOGGER.warning("Справочный агент не ответил: %s", exc)
        ctx.tell(ticket, f"ИИ сейчас не ответил. Вот ближайшая справка:\n\n{fallback}")
        return
    job.checkpoint()

    allowed_commands = {f"/{name}" for name, _ in COMMANDS} | {"/cancel", "/admin"}
    mentioned_commands = set(re.findall(r"/[a-z]+", reply.casefold()))
    invented = mentioned_commands - allowed_commands
    if not reply or invented:
        if invented:
            LOGGER.warning("Справочный агент придумал команды: %s", sorted(invented))
        reply = fallback
    ctx.tell(ticket, reply)


def callback_set_level(ctx: Context, user: User, payload: str) -> str:
    """Ручная установка уровня. Правит и то, что уровень за собой тянет.

    Раньше менялись две колонки в таблице, а карточки прежнего уровня оставались
    в очереди навсегда: человек исправлял ошибочную диагностику и получал
    «Повторение» из чужого курса.
    """
    from ..content.schema import LEVEL_ORDER

    level = payload.strip().upper()
    if level not in LEVELS:
        return "неизвестный уровень"
    previous = user.level or ""
    ctx.storage.update_user(user.user_id, level=level, target_level=next_level(level))
    ctx.storage.log_event(user.user_id, "level_manual", f"{previous or '-'}->{level}")

    removed = 0
    if previous and LEVEL_ORDER.get(level, 0) < LEVEL_ORDER.get(previous, 0):
        ceiling = LEVEL_ORDER[level]
        above = [
            key
            for key in ctx.storage.card_keys(user.user_id, "point")
            if (point := ctx.curriculum.point(key)) is not None
            and LEVEL_ORDER.get(point.level, 0) > ceiling
        ]
        removed = ctx.storage.delete_cards(user.user_id, "point", above)

    lines = [f"Уровень: {level}, цель {next_level(level)}."]
    if removed:
        lines.append(f"Убрал из повторения {removed} карточек выше нового уровня.")
    lines.append("Дальше — «🎯 Заниматься».")
    ctx.say(user, " ".join(lines))
    return f"уровень {level}"


def command_progress(ctx: Context, user: User, text: str) -> None:
    counts = ctx.storage.card_counts(user.user_id)
    due = sum(value[1] for value in counts.values())
    # Кнопка «Повторение» при пустой очереди вела в текст «повторять нечего»:
    # предлагаем то, что действительно есть.
    second = (f"🔁 Повторить ({due})", "startreview") if due else ("🎯 Тренировка", "startpractice")
    ctx.say(
        user,
        progress_report(ctx.storage, ctx.curriculum, user),
        inline([[("🧭 План", "plan"), second]]),
    )


def command_plan(ctx: Context, user: User, text: str) -> None:
    if not user.level:
        ctx.say(
            user,
            "Сначала определим уровень — это 10–15 минут.",
            inline([[("🧪 Диагностика", "retest")]]),
        )
        return
    from ..learning.progress import mastery_map, unmastered_points

    mastery = mastery_map(ctx.storage, user.user_id)
    upcoming = unmastered_points(ctx.curriculum, mastery, user.level, limit=3)
    rows = [
        [(f"▶️ {point.title_ru[:28]}", f"pr:{ctx.curriculum.point_code(point.id)}")]
        for point in upcoming
    ]
    rows.append([("🎯 Заниматься", "daily")])
    ctx.say(user, study_plan(ctx.storage, ctx.curriculum, user), inline(rows))


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
    # Файл собран локально, а вот его загрузка в Telegram — сетевой multipart с
    # таймаутом 120 секунд, и держать ею дорожку обновлений нельзя.
    ctx.background(ctx.ticket(user), "отправку файла карточек", _anki_job, str(path))


def _anki_job(job: Job, ctx: Context, ticket: Ticket, path: str) -> None:
    try:
        ctx.telegram.send_document(
            ticket.chat_id,
            Path(path),
            "Импорт в Anki: File → Import, разделитель Tab, поля Front/Back/Tags, "
            "«Allow HTML» выключен.",
        )
    except TelegramError:
        LOGGER.exception("Не удалось отправить файл карточек")
        ctx.tell(ticket, "Файл собрал, но Telegram его не принял. Попробуй ещё раз.")


def command_forget(ctx: Context, user: User, text: str) -> None:
    if text.partition(" ")[2].strip() != "YES":
        ctx.say(
            user,
            "Это удалит твой прогресс, карточки, журнал ошибок, письма и голосовые. "
            "Учётка и роль останутся. Подтверди точно так: /forget YES",
        )
        return
    # Иначе фоновый разбор допишет ошибки и серию уже после удаления.
    ctx.cancel_job(user)
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
    # Единственный способ прервать идущий разбор. Отмена кооперативная: Whisper и
    # запущенный подпроцесс доработают, но результат уже не придёт и ничего не запишет.
    stopped = ctx.cancel_job(user)
    ctx.reset_state(user)
    note = f"Прервал {stopped}. " if stopped else ""
    ctx.say(user, note + "Остановился. Кнопки внизу — или просто напиши мне по-английски.",
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
        inline([[("Отозвать этот код", f"revoke:{code}")]]),
    )


def callback_revoke_invite(ctx: Context, user: User, payload: str) -> str:
    """Отозвать выданный код: раньше живые приглашения нельзя было погасить."""
    if not user.is_admin:
        return "только для владельца и админов"
    if ctx.storage.revoke_invite(payload.strip()):
        ctx.say(user, "Код отозван — по нему больше не войти.")
        return "отозван"
    return "код уже использован или не найден"


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
        f"{sum(len(rows) for rows in curriculum.vocabulary.values())} слов, "
        f"{sum(len(rows) for rows in curriculum.listening.values())} аудирований",
        f"Ошибок загрузки контента: {len(curriculum.load_errors)}",
        "",
        f"Текстовый ИИ: {'включён (' + ctx.settings.llm_provider + ')' if ctx.llm else 'выключен'}",
        f"Распознавание речи: "
        f"{ctx.settings.speech_backend if ctx.transcriber else 'выключено'}",
        f"Синтез речи: {ctx.settings.speech_backend if ctx.speaker else 'выключен'}",
        f"Лимит обращений к ИИ: {ctx.settings.daily_ai_calls} в сутки на человека",
        f"Пулы: users={ctx.settings.workers}, LLM={ctx.settings.llm_workers}, "
        f"STT={ctx.settings.stt_workers}, TTS={ctx.settings.tts_workers}",
        "",
        "Люди:",
    ]
    for row in users:
        lines.append(
            f"• {row.display_name or 'без имени'} — {row.role}, "
            f"{row.level or 'уровень не определён'}, "
            f"серия {ctx.storage.effective_streak(row.user_id)}"
        )

    live = [row for row in invites if row["used_by"] is None]
    if live:
        lines.append("")
        lines.append("Живые приглашения:")
        for row in live[:10]:
            lines.append(f"• {row['code']} — роль {row['role']}, до {str(row['expires_at'])[:10]}")

    lines.extend(_observability_lines(ctx))
    lines.extend(_usage_lines(ctx))
    ctx.say(user, "\n".join(lines), _revoke_keyboard(live))


def _revoke_keyboard(live: list[dict[str, Any]]) -> dict | None:
    if not live:
        return None
    return inline([[(f"Отозвать {row['code'][:8]}", f"revoke:{row['code']}")] for row in live[:5]])


# Исходы авторизации по-русски: в журнале они короткие метки, а на экране их
# читает человек.
ACCESS_LABELS: dict[str, str] = {
    "owner": "владелец привязан",
    "joined": "вошёл по приглашению",
    "open": "вошёл свободно",
    "no_owner": "бот ещё не привязан",
    "need_invite": "нет кода приглашения",
    "invite_used": "код уже использован",
    "invite_expired": "срок кода истёк",
    "invite_missing": "такого кода нет",
}
ADMITTED_OUTCOMES = frozenset({"owner", "joined", "open"})


def _duration_ru(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} с"
    if seconds < 3600:
        return f"{seconds // 60} мин"
    if seconds < 86400:
        return f"{seconds // 3600} ч {seconds % 3600 // 60} мин"
    return f"{seconds // 86400} сут {seconds % 86400 // 3600} ч"


def _observability_lines(ctx: Context) -> list[str]:
    """Что процесс делает прямо сейчас и кто пытался войти.

    Раньше и то и другое было видно только с сервера через `journalctl`, а
    отказанные во входе не оставляли следа вообще — выдать доступ вручную было
    нечему, Telegram ID взять неоткуда.
    """
    lines = ["", "Наблюдаемость"]
    meter = ctx.telemetry
    if meter is not None:
        counts = meter.counts()
        lines.append(f"Аптайм: {_duration_ru(meter.uptime_seconds)}")
        lines.append(
            f"Обновлений: {counts.get('updates', 0)} · ошибок: {counts.get('errors', 0)}"
            f" · медленных: {counts.get('slow', 0)}"
        )
        lines.append(
            f"Очередь обновлений: {meter.queue_depth}"
            f" · отклонено за запуск: {meter.queue_rejected}"
        )
        where, ago = meter.last_error
        if where:
            lines.append(f"Последний сбой: {where}, {_duration_ru(ago)} назад")

    if ctx.jobs is not None:
        jobs = ctx.jobs.stats()
        lines.append(
            f"Фоновые задачи: сейчас {jobs.get('running', 0)}, всего "
            f"{jobs.get('started', 0)} — готово {jobs.get('done', 0)}, "
            f"прервано {jobs.get('cancelled', 0)}, упало {jobs.get('failed', 0)}"
        )
        for key, stage, seconds in ctx.jobs.snapshot()[:5]:
            lines.append(f"• {key} — {stage}, {_duration_ru(seconds)}")
    else:
        lines.append("Фоновые задачи: выключены (JOB_WORKERS=0)")

    # Очередь тяжёлых контуров объясняет «почему долго» лучше любого счётчика:
    # при STT_WORKERS=1 вторая расшифровка ждёт первую целиком.
    heavy = [
        (label, int(getattr(circuit, "queue_ahead", 0) or 0))
        for label, circuit in (
            ("текст", ctx.llm), ("расшифровка", ctx.transcriber), ("синтез", ctx.speaker)
        )
        if circuit is not None
    ]
    if heavy:
        lines.append(
            "Тяжёлые очереди: " + ", ".join(f"{label} {depth}" for label, depth in heavy)
        )

    summary = ctx.storage.access_summary(days=7)
    if summary:
        lines.append("")
        lines.append("Входы за 7 дней (попыток · людей):")
        for row in summary:
            label = ACCESS_LABELS.get(str(row["outcome"]), str(row["outcome"]))
            lines.append(f"• {label} — {row['times']} · {row['people']}")

    attempts = ctx.storage.access_attempts(limit=8)
    if attempts:
        lines.append("")
        lines.append("Последние попытки входа:")
        for row in attempts:
            when = str(row["created_at"])[5:16].replace("T", " ")
            mark = "✅" if str(row["outcome"]) in ADMITTED_OUTCOMES else "⛔"
            label = ACCESS_LABELS.get(str(row["outcome"]), str(row["outcome"]))
            who = str(row["display_name"] or "без имени")
            lines.append(f"{mark} {when} · {row['user_id']} · {who} — {label}")
        lines.append("Дать доступ вручную можно по этому id.")
    return lines


def _usage_lines(ctx: Context) -> list[str]:
    """Анонимные счётчики интерфейса: без них приоритезация — гадание."""
    counts = ctx.storage.event_counts(days=14)
    if not counts:
        return ["", "Счётчики интерфейса: за две недели событий нет."]
    lines = ["", "Интерфейс за 14 дней (событий · людей):"]
    for row in counts[:12]:
        label = f"{row['name']}:{row['detail']}" if row["detail"] else str(row["name"])
        lines.append(f"• {label} — {row['times']} · {row['people']}")
    depth = ctx.storage.event_depth(days=14)
    if depth:
        lines.append(f"Глубина до старта функции: медиана {depth[0]}, p90 {depth[1]}")
    return lines
