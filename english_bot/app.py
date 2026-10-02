"""English Lab: многопользовательская платформа изучения английского в Telegram.

Процесс один: long polling складывает апдейты в общий пул, а отдельная очередь
каждого человека сохраняет порядок. Codex, Whisper и Piper работают в своих
ограниченных пулах и не останавливают приём новых обновлений.
"""

from __future__ import annotations

import logging
import random
import re
import signal
import sys
import time
from concurrent.futures import Future
from typing import Any, Callable
from urllib.parse import urlparse

from .ai import claude_cli, codex_cli
from .ai.claude_cli import ClaudeRunner
from .ai.codex_cli import CodexRunner
from .ai.llm import LLM
from .ai.stt import Transcriber
from .ai.tts import Speaker
from .config import Settings
from .content.registry import load_curriculum
from .context import Context
from .handlers import core, dialogue, listening, menu, speech, study
from .runtime import (
    HeavyPools,
    JobRunner,
    KeyedExecutor,
    OverloadedError,
    Telemetry,
    ThreadedLLM,
    ThreadedSpeaker,
    ThreadedTranscriber,
)
from .storage import Storage, User
from .telegram_api import TelegramAPI, TelegramError, sender_name


LOGGER = logging.getLogger(__name__)

CommandHandler = Callable[[Context, User, str], None]
CallbackHandler = Callable[[Context, User, str], str]

# Интерфейсы разделены: всё, до чего можно дойти кнопками, командой не
# дублируется. Здесь остаётся только то, чего в меню нет, — как правило это
# действия с произвольным аргументом (слово, запрос, сценарий) и то, что
# намеренно спрятано подальше от случайного нажатия.
COMMANDS: dict[str, CommandHandler] = {
    "/start": core.command_start,
    "/help": core.command_help,
    "/say": speech.command_say,
    "/learn": study.command_learn,
    "/roleplay": dialogue.command_roleplay,
    "/export": core.command_export,
    "/anki": core.command_anki,
    "/stop": core.command_stop,
    "/cancel": core.command_stop,
    "/privacy": core.command_privacy,
    "/forget": core.command_forget,
    "/admin": core.command_admin,
}

CALLBACKS: dict[str, CallbackHandler] = {
    "pf": study.callback_profile,
    "pa": study.callback_placement_answer,
    "lvls": study.callback_levels,
    "lvl": study.callback_level,
    "tp": study.callback_topic,
    "pt": study.callback_point,
    "pr": study.callback_practice_point,
    "prt": study.callback_practice_topic,
    "an": study.callback_answer,
    "hint": study.callback_hint,
    "rule": study.callback_rule,
    "retest": study.callback_retest,
    "skip": study.callback_skip,
    "endses": study.callback_end_session,
    "startpractice": study.callback_start_practice,
    "startreview": study.callback_start_review,
    "progress": study.callback_progress,
    "plan": study.callback_plan,
    "setlvl": core.callback_set_level,
    "daily": menu.callback_daily,
    "rsm": menu.callback_resume,
    "sw": menu.callback_switch,
    "roleplayhint": menu.callback_roleplay_hint,
    "rp": dialogue.callback_roleplay_start,
    "team": menu.callback_team,
    "levelpick": menu.callback_level_pick,
    "invitenew": menu.callback_invite,
    "revoke": core.callback_revoke_invite,
    "ex": dialogue.callback_explain,
    "write": dialogue.callback_writing,
    "endroleplay": dialogue.callback_end_roleplay,
    "speak": speech.callback_speaking,
    "listen": listening.callback_listening,
    "la": listening.callback_answer,
    "lr": listening.callback_replay,
    "say": speech.callback_say,
    "slow": speech.callback_say_slow,
}

# Эти команды не прерывают занятие: спросить про слово, про интерфейс, вернуть
# клавиатуру или переключить участие в таблице посреди упражнения — нормальный
# ход, а не смена вида деятельности. `/stop` прерывает намеренно.
STATE_SAFE_COMMANDS = {"/stop", "/cancel", "/say", "/help", "/start", "/privacy", "/admin"}

# Пока идёт длинная задача, команда, которая сбрасывает состояние, уничтожила бы
# занятие и тут же получила отказ «сначала закончу». Проходят только те, что
# состояние не трогают, и `/forget`, который сам снимает задачу.
JOB_SAFE_COMMANDS = STATE_SAFE_COMMANDS | {"/forget"}

# Нажатия, чей обработчик всегда возвращает пустую подсказку: подтверждаем их до
# работы, иначе спиннер на кнопке крутится, пока экран собирается и отправляется.
# Всё, что возвращает осмысленный `note` («Порядок прилагательных», «сценарий не
# найден», «прослушивание 2 из 3»), подтверждается после обработчика — иначе
# подсказка потеряется. Список сверен с возвращаемыми значениями обработчиков.
EARLY_ACK = frozenset({"listen", "speak", "write", "daily", "rsm"})
# Дольше этого обработка одного обновления в дорожке идти не должна: значит
# тяжёлый вызов снова остался в интерактивной фазе вместо `ctx.background`.
# При `JOB_WORKERS=0` предупреждение ожидаемо — это и есть инлайн-режим.
SLOW_HANDLER_SECONDS = 3.0

# Кнопки, начинающие новое занятие. Пока идёт длинная задача, отказ обязан
# случиться здесь: `menu.close_active` внутри этих обработчиков сначала закрыл бы
# текущее занятие и только потом упёрся бы в «одну задачу на человека».
# Продолжение начатого (`rsm`), ответы внутри занятия и экраны чтения не входят:
# занятость — не повод отнимать навигацию.
JOB_BLOCKED_CALLBACKS = frozenset(
    {"daily", "sw", "speak", "listen", "write", "rp", "pr", "prt",
     "startpractice", "startreview", "retest"}
)

# Кнопка действует только внутри своего занятия. Инлайн-клавиатуры Telegram живут
# в истории вечно, поэтому нажатие из прокрученной вверх переписки обязано быть
# отбито, а не выполнено поверх текущего состояния.
CALLBACK_STATES: dict[str, str] = {
    "pf": "placement",
    "pa": "placement",
    "an": "practice",
    "hint": "practice",
    "skip": "practice",
    "endses": "practice",
    "la": "listening",
    "lr": "listening",
    # «Закончить» живёт под каждой репликой диалога: нажатие из прокрученной
    # вверх переписки не должно закрывать тренировку, начатую позже.
    "endroleplay": "roleplay",
}
STALE_BUTTON = "эта кнопка уже не активна"

# Исходы авторизации, при которых человек оказался внутри. Остальные — отказы,
# и в журнал процесса они идут предупреждением: молчаливый отказ невидим.
ADMITTED = frozenset({"owner", "joined", "open"})

def _build_llm(settings: Settings) -> LLM | None:
    """Собирает текстовую модель по `LLM_PROVIDER`. Это и есть переключатель.

    У подписочных провайдеров ключа нет, зато есть бинарник: если его нет в
    PATH, честнее сразу выключить ИИ, чем обещать функцию и падать на каждом
    обращении с «попробуй позже».
    """
    if settings.llm_provider == "codex":
        if not codex_cli.available(settings.codex_binary):
            LOGGER.error(
                "LLM_PROVIDER=codex, но %s не найден в PATH — текстовый ИИ выключен",
                settings.codex_binary,
            )
            return None
        return LLM(
            "codex",
            codex=CodexRunner(
                binary=settings.codex_binary,
                model=settings.codex_model,
                effort=settings.codex_effort,
                timeout=settings.codex_timeout,
            ),
        )
    if settings.llm_provider == "claude":
        if not claude_cli.available(settings.claude_binary):
            LOGGER.error(
                "LLM_PROVIDER=claude, но %s не найден в PATH — текстовый ИИ выключен",
                settings.claude_binary,
            )
            return None
        return LLM(
            "claude",
            model=settings.claude_model,
            claude=ClaudeRunner(
                binary=settings.claude_binary,
                model=settings.claude_model,
                effort=settings.claude_effort,
                timeout=settings.claude_timeout,
                base_url=settings.claude_base_url,
                auth_token=settings.claude_auth_token,
            ),
        )
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            return None
        return LLM("anthropic", settings.anthropic_api_key, settings.anthropic_model)
    if not settings.openai_api_key:
        return None
    return LLM("openai", settings.openai_api_key, settings.openai_model)


def _build_speech(settings: Settings) -> tuple[Any, Any]:
    """Собирает распознавание и синтез по `SPEECH_BACKEND`.

    Локальные модели живут в `.venv`; если бота запустили системным Python,
    импорт не удастся — тогда речь просто выключена, остальное работает.
    """
    if settings.speech_backend == "local":
        from .ai.local_speech import (
            LocalSpeaker,
            LocalTranscriber,
            piper_available,
            whisper_available,
        )

        transcriber = (
            LocalTranscriber(
                settings.whisper_dir,
                settings.whisper_model,
                settings.whisper_compute,
                settings.whisper_threads,
            )
            if whisper_available()
            else None
        )
        speaker = (
            LocalSpeaker(
                settings.piper_dir,
                settings.audio_cache_dir,
                settings.piper_voice,
                settings.piper_female_voices,
                settings.piper_male_voices,
            )
            if piper_available()
            else None
        )
        if transcriber is None or speaker is None:
            LOGGER.warning(
                "SPEECH_BACKEND=local, но модели недоступны: whisper=%s, piper=%s. "
                "Запусти бота из .venv/bin/python.",
                whisper_available(),
                piper_available(),
            )
        return transcriber, speaker

    if not settings.openai_api_key:
        return None, None
    return (
        Transcriber(settings.openai_api_key, settings.stt_model),
        Speaker(
            settings.openai_api_key,
            settings.tts_model,
            settings.tts_voice,
            settings.audio_cache_dir,
            settings.tts_female_voices,
            settings.tts_male_voices,
        ),
    )


UNEXPECTED_ERROR = (
    "Что-то пошло не так на моей стороне — ошибка записана. "
    "Попробуй ещё раз или нажми «🎯 Заниматься»."
)

GROUP_REFUSAL = (
    "Я работаю только в личных сообщениях — там у каждого свой уровень и свой прогресс. "
    "Открой меня в личке и нажми «🎯 Заниматься»."
)

NOT_LINKED = (
    "Бот ещё не привязан к владельцу. Открой персональную ссылку запуска, "
    "которую дал создатель."
)
NEED_INVITE = (
    "English Lab — платформа отдела АБП, вход по приглашению. "
    "Попроси у владельца бота ссылку вида t.me/…?start=КОД.\n\n"
    "Ссылка уже есть, а по нажатию ничего не происходит? Так бывает, если диалог "
    "со мной уже открыт: Telegram подставляет код только в пустой чат. "
    "Пришли код сюда сообщением — можно целой ссылкой, можно как /start КОД."
)

# Код приглашения приходит в трёх видах, и все три должны работать. Человек,
# у которого диалог с ботом уже открыт, нажать на ссылку не может: Telegram
# просто откроет уже открытый чат, а `/start КОД` подставляет только в пустой.
# Тогда ссылку вставляют текстом — и раньше это молча не срабатывало.
INVITE_LINK = re.compile(r"[?&]start=([A-Za-z0-9_-]{4,64})")
BARE_INVITE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


INVITE_NOT_FOUND = (
    "Такого кода приглашения нет. Проверь, что ссылка скопирована целиком, "
    "или попроси у владельца бота новую."
)


def invite_payload(text: str) -> str:
    """Достаёт код приглашения из `/start КОД`, из вставленной ссылки или из голого кода.

    Хвост после `/start` тоже разбирается: меню команд Telegram подставляет
    `/start ` в поле ввода, и человек вставляет ссылку прямо за ним. Раньше
    такое сообщение уходило в отказ, хотя код в нём был живой.
    """
    text = text.strip()
    parts = text.split(maxsplit=1)
    if parts and parts[0].split("@", 1)[0].lower() == "/start" and len(parts) > 1:
        tail = parts[1].strip()
        found = INVITE_LINK.search(tail)
        if found:
            return found.group(1)
        head = tail.split(maxsplit=1)[0]
        # Хвост не похож на код — отдаём как есть: явное `/start что-то` должно
        # попасть в журнал как «такого кода нет», а не раствориться в общем
        # отказе. Заодно это единственный путь для произвольного BOT_CLAIM_CODE.
        return head if BARE_INVITE.match(head) else tail
    found = INVITE_LINK.search(text)
    if found:
        return found.group(1)
    return text if BARE_INVITE.match(text) else ""


def looks_deliberate(text: str) -> bool:
    """Человек явно прислал код, а не просто написал боту слово из восьми букв."""
    stripped = text.strip()
    return stripped.lower().startswith("/start") or bool(INVITE_LINK.search(stripped))
# Один текст на все причины заставлял гадать: пересланная ссылка выглядела так
# же, как истёкшая или чужая.
INVITE_REFUSALS = {
    "used": (
        "Этим приглашением уже воспользовались — код одноразовый. "
        "Попроси у владельца бота новую ссылку."
    ),
    "expired": (
        "Срок этого приглашения истёк: коды живут 14 дней. "
        "Попроси у владельца бота новую ссылку."
    ),
}


class EnglishLabBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = Storage(settings.database_path)
        self.telegram = TelegramAPI(settings.telegram_token)
        self.curriculum = load_curriculum()
        self.running = True
        self._closed = False
        self._dispatcher = KeyedExecutor(settings.workers)
        # `JOB_WORKERS=0` — инлайн-режим: длинные цепочки идут прямо в дорожке
        # обновления, как до разделения. Так проверяются обработчики по шагам.
        self._jobs = (
            JobRunner(settings.job_workers, timeout=settings.job_timeout)
            if settings.job_workers
            else None
        )
        self._heavy = HeavyPools(
            settings.llm_workers, settings.stt_workers, settings.tts_workers
        )
        self._telemetry = Telemetry(self._dispatcher)

        llm = _build_llm(settings)
        transcriber, speaker = _build_speech(settings)
        self.llm = ThreadedLLM(llm, self._heavy.llm) if llm is not None else None
        self.transcriber = (
            ThreadedTranscriber(transcriber, self._heavy.stt)
            if transcriber is not None
            else None
        )
        self.speaker = (
            ThreadedSpeaker(speaker, self._heavy.tts) if speaker is not None else None
        )

    # ── жизненный цикл ───────────────────────────────────────────

    def context(self, message_id: int | None = None) -> Context:
        return Context(
            settings=self.settings,
            storage=self.storage,
            telegram=self.telegram,
            curriculum=self.curriculum,
            llm=self.llm,
            transcriber=self.transcriber,
            speaker=self.speaker,
            rng=random.Random(),
            message_id=message_id,
            jobs=self._jobs,
            telemetry=self._telemetry,
        )

    def initialize(self) -> dict[str, Any]:
        self.storage.initialize()
        identity = self.telegram.get_me()
        self.telegram.delete_webhook()
        self.telegram.set_commands(core.COMMANDS)
        username = str(identity.get("username") or "")
        if username:
            self.storage.set_setting("bot_username", username)
        if self.curriculum.load_errors:
            LOGGER.error(
                "Контент загружен с ошибками (%d) — часть тем недоступна",
                len(self.curriculum.load_errors),
            )
        provider = self.settings.llm_provider if self.llm else "выключен"
        if self.llm and self.settings.llm_provider == "claude" and self.settings.claude_base_url:
            # Только хост: токен шлюза в журнал не попадает.
            provider += f" через {urlparse(self.settings.claude_base_url).hostname}"
        LOGGER.info(
            "Запущен как @%s · %d тем, %d упражнений · ИИ: %s · речь: %s",
            username,
            len(self.curriculum.points),
            len(self.curriculum.exercises),
            provider,
            self.settings.speech_backend if self.transcriber else "выключена",
        )
        return identity

    def stop(self, *_: object) -> None:
        self.running = False

    def run(self) -> None:
        self.initialize()
        offset: int | None = None
        try:
            while self.running:
                try:
                    updates = self.telegram.get_updates(offset)
                    for update in updates:
                        offset = int(update["update_id"]) + 1
                        self._submit(update)
                except TelegramError:
                    LOGGER.exception("Ошибка long polling")
                    time.sleep(3)
                except Exception:
                    LOGGER.exception("Неожиданная ошибка цикла обновлений")
                    time.sleep(1)
        finally:
            self.close()

    def close(self, wait: bool = True) -> None:
        """Останавливает очереди в безопасном порядке; повторный вызов безвреден."""
        if self._closed:
            return
        self._closed = True
        # Сначала перестаём принимать обновления и даём уже принятым закончить,
        # затем снимаем фоновые задачи и только потом закрываем пулы, внутри
        # которых они могли ждать тяжёлую работу. Фоновые задачи не ждём никогда:
        # `claude -p` живёт до 180 секунд, а systemd даёт на остановку 30.
        self._dispatcher.shutdown(wait=wait)
        if self._jobs is not None:
            self._jobs.shutdown(wait=False)
        self._heavy.shutdown(wait=wait)

    def _submit(self, update: dict[str, Any]) -> "Future[None]":
        """Ставит обновление в дорожку его отправителя и отдаёт его future.

        Future нужен не циклу — он нужен тестам: только по нему видно, что
        обновление действительно обработано, а не ждёт чужой расшифровки.
        """
        sender = (update.get("message") or update.get("callback_query") or {}).get("from") or {}
        user_id = int(sender.get("id") or 0)
        future = self._dispatcher.submit(user_id, self._safe_handle, update)
        if future.done():
            try:
                future.result()
            except OverloadedError:
                # Ветка подкласса обязана стоять первой: `OverloadedError` — это
                # `RuntimeError`, и общий перехват ниже забрал бы его себе.
                # Очереди имеют конечный размер: под флудом сохраняем память и polling,
                # даже если отдельное уже подтверждённое Telegram-обновление потеряется.
                LOGGER.warning("Очередь обновлений заполнена, апдейт отклонён")
            except RuntimeError as exc:
                # Гонка с остановкой: без перехвата она всплывает в `run` и
                # кладёт long polling на секундный сон.
                LOGGER.info("Обновление не принято: %s", exc)
        return future

    def _safe_handle(self, update: dict[str, Any]) -> None:
        try:
            self.handle_update(update)
        except Exception as exc:
            self._telemetry.note_error("обновление", exc)
            LOGGER.exception("Не удалось обработать обновление")
            # Молча проглоченная ошибка выглядит для человека как зависший бот.
            # Ответить хоть что-то важнее, чем аккуратно записать в лог.
            self._apologise(update)

    def _apologise(self, update: dict[str, Any]) -> None:
        message = update.get("message") or (update.get("callback_query") or {}).get("message")
        chat_id = ((message or {}).get("chat") or {}).get("id")
        if not chat_id:
            return
        try:
            self.telegram.send_message(int(chat_id), UNEXPECTED_ERROR)
        except TelegramError:
            LOGGER.info("Не удалось сообщить об ошибке пользователю")

    # ── маршрутизация ────────────────────────────────────────────

    def handle_update(self, update: dict[str, Any]) -> None:
        started = time.monotonic()
        self._telemetry.mark("updates")
        try:
            if "callback_query" in update:
                self._handle_callback(update["callback_query"])
                return
            message = update.get("message")
            if message:
                self._handle_message(message)
        finally:
            spent = time.monotonic() - started
            if spent > SLOW_HANDLER_SECONDS:
                # Иначе возврат тяжёлого вызова в интерактивную фазу снова стал
                # бы невидимым: снаружи это просто «бот подтормаживает».
                self._telemetry.mark("slow")
                LOGGER.warning(
                    "Обновление обрабатывалось %.1f с — тяжёлый вызов остался в дорожке",
                    spent,
                )

    def _handle_message(self, message: dict[str, Any]) -> None:
        chat = message.get("chat", {})
        sender = message.get("from", {})
        if not sender.get("id") or sender.get("is_bot"):
            return
        if chat.get("type") != "private":
            # Молчать в группе нельзя: человек решает, что бот сломался.
            self._refuse_group(chat, message)
            return
        user_id = int(sender["id"])
        chat_id = int(chat["id"])
        text = (message.get("text") or "").strip()

        ctx = self.context()
        name = sender_name(sender)
        user = self._authorize(ctx, user_id, chat_id, text, name)
        if user is None:
            return
        self.storage.touch(user_id, chat_id, name)
        user = ctx.reload_user(user)

        if text.startswith("/"):
            self._dispatch_command(ctx, user, text)
            return
        if message.get("voice"):
            if user.state == "listening":
                ctx.say(user, "Сейчас слушаем: выбери ответ кнопкой под аудио.")
                return
            speech.handle_voice(ctx, user, message)
            return
        if not text:
            ctx.say(user, "Понимаю текст и голосовые. Дальше — кнопками внизу.")
            return
        self._dispatch_text(ctx, user, text)

    def _refuse_group(self, chat: dict[str, Any], message: dict[str, Any]) -> None:
        """В группе отвечаем только на команду и только один раз за раз."""
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return
        chat_id = int(chat.get("id") or 0)
        if not chat_id:
            return
        try:
            self.telegram.send_message(chat_id, GROUP_REFUSAL)
        except TelegramError:
            LOGGER.info("Не удалось ответить в группе")

    def _handle_callback(self, query: dict[str, Any]) -> None:
        callback_id = str(query.get("id") or "")
        sender = query.get("from", {})
        message = query.get("message") or {}
        chat = message.get("chat", {})
        if not sender.get("id") or not chat.get("id"):
            return
        user_id = int(sender["id"])
        chat_id = int(chat["id"])
        data = str(query.get("data") or "")

        ctx = self.context(message_id=int(message.get("message_id") or 0) or None)
        user = self.storage.user(user_id)
        if user is None:
            self.telegram.answer_callback(callback_id, "Нужно приглашение", alert=True)
            return
        self.storage.touch(user_id, chat_id, sender_name(sender))
        user = ctx.reload_user(user)

        prefix, _, payload = data.partition(":")
        handler = CALLBACKS.get(prefix)
        if handler is None:
            self.telegram.answer_callback(callback_id)
            return
        required = CALLBACK_STATES.get(prefix)
        if required is not None and user.state != required:
            self.telegram.answer_callback(callback_id, STALE_BUTTON)
            return
        busy = ctx.busy(user) if prefix in JOB_BLOCKED_CALLBACKS else None
        if busy is not None:
            self.telegram.answer_callback(callback_id, f"сначала закончу {busy.stage}")
            ctx.say(user, f"Сначала закончу {busy.stage}. Прервать — /stop.")
            return
        # `answerCallbackQuery` не тратит токен исходящего лимита, поэтому ранний
        # ответ бесплатен, а спиннер на кнопке гаснет сразу.
        early = prefix in EARLY_ACK
        if early:
            self.telegram.answer_callback(callback_id)
        try:
            note = handler(ctx, user, payload)
        except Exception:
            LOGGER.exception("Ошибка в обработчике callback %s", prefix)
            self.telegram.answer_callback(callback_id, "Что-то сломалось")
            ctx.say(user, "Кнопка не сработала. Попробуй ещё раз или нажми «🎯 Заниматься».")
            return
        if not early:
            self.telegram.answer_callback(callback_id, note or "")

    def _dispatch_command(self, ctx: Context, user: User, text: str) -> None:
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        handler = COMMANDS.get(command)
        if handler is None:
            ctx.say(user, core.HELP_TEXT)
            return
        # Отказ обязан случиться до сброса состояния: иначе команда во время
        # длинной задачи уничтожает занятие и только потом получает отказ.
        busy = ctx.busy(user)
        if busy is not None and command not in JOB_SAFE_COMMANDS:
            ctx.say(user, f"Сначала закончу {busy.stage}. Прервать — /stop.")
            return
        # Любая команда вытаскивает из незавершённого занятия: иначе следующее
        # обычное сообщение уйдёт, например, в разбор письма вместо чата.
        if user.state != "idle" and command not in STATE_SAFE_COMMANDS:
            ctx.reset_state(user)
            user = ctx.reload_user(user)
        handler(ctx, user, text)

    def _dispatch_text(self, ctx: Context, user: User, text: str) -> None:
        # Кнопки постоянной клавиатуры приходят обычным текстом. Они работают из
        # любого состояния, но состояние больше не сбрасывается здесь: экраны
        # чтения его не трогают, а начало нового занятия проходит через
        # `menu.handle_button`, который сначала спрашивает.
        if menu.is_button(text):
            menu.handle_button(ctx, user, text)
            return
        if user.state == "practice":
            study.handle_practice_text(ctx, user, text)
            return
        if user.state == "placement":
            study.handle_placement_text(ctx, user, text)
            return
        if user.state == "writing":
            dialogue.handle_writing_text(ctx, user, text)
            return
        if user.state == "roleplay":
            dialogue.handle_roleplay_text(ctx, user, text)
            return
        if user.state == "speaking":
            ctx.say(user, "Жду голосовое по заданию. Или /stop, чтобы выйти.")
            return
        if user.state == "listening":
            ctx.say(user, "Выбери ответ кнопкой A–D под аудио. Или /stop, чтобы выйти.")
            return
        dialogue.handle_free_text(ctx, user, text)

    # ── доступ ───────────────────────────────────────────────────

    def _authorize(
        self, ctx: Context, user_id: int, chat_id: int, text: str, name: str = ""
    ) -> User | None:
        existing = self.storage.user(user_id)
        if existing is not None:
            return existing

        payload = invite_payload(text)
        owner = self.storage.get_setting("owner_user_id")
        if owner is None:
            if payload and payload == self.settings.claim_code:
                self.storage.set_setting("owner_user_id", str(user_id))
                user = self.storage.create_user(user_id, chat_id, "owner", name)
                self._record_access(user_id, chat_id, name, "owner")
                core.command_start(ctx, user, text)
                return None
            self._record_access(
                user_id, chat_id, name, "no_owner", "неверный код" if payload else ""
            )
            self.telegram.send_message(chat_id, NOT_LINKED)
            return None

        # Одна попытка — одна запись в журнале. Причина уточняется по ходу
        # разбора, но пишется в самом конце, иначе неверный код давал бы две
        # строки на одно сообщение.
        reason, detail = "need_invite", ""
        if payload:
            role = self.storage.redeem_invite(payload, user_id)
            if role:
                user = self.storage.create_user(user_id, chat_id, role, name)
                self._record_access(user_id, chat_id, name, "joined", role)
                core.command_start(ctx, user, text)
                return None
            # Причина выясняется только после неудачного списания: спросить её
            # заранее значит показать состояние, которое к моменту отказа уже
            # могло измениться.
            status = self.storage.invite_status(payload)
            refusal = INVITE_REFUSALS.get(status)
            if refusal:
                self._record_access(user_id, chat_id, name, f"invite_{status}", payload)
                self.telegram.send_message(chat_id, refusal)
                return None
            # `missing` — кода нет: опечатка, чужая ссылка или случайный текст,
            # принятый разбором за код. В журнал идёт только то, что похоже на
            # код: остальное — чужая переписка, и ей там не место.
            reason = "invite_missing"
            detail = payload if BARE_INVITE.match(payload) else "не похоже на код"

        if self.settings.team_open_registration:
            user = self.storage.create_user(user_id, chat_id, "member", name)
            self._record_access(user_id, chat_id, name, "open")
            core.command_start(ctx, user, text)
            return None

        self._record_access(user_id, chat_id, name, reason, detail)
        # Общий отказ здесь был бы враньём: человек прислал код, и ему нужно
        # знать, что дело в самом коде, а не в отсутствии приглашения.
        precise = reason == "invite_missing" and looks_deliberate(text)
        self.telegram.send_message(chat_id, INVITE_NOT_FOUND if precise else NEED_INVITE)
        return None

    def _record_access(
        self, user_id: int, chat_id: int, name: str, outcome: str, detail: str = ""
    ) -> None:
        """Сохраняет исход попытки входа и дублирует его в журнал процесса.

        Отказы не оставляли следа нигде, и выдать доступ вручную было нечему:
        Telegram ID отказанного взять неоткуда. Теперь он есть и в базе — для
        экрана `/admin`, — и в journald, где его видно без запроса к SQLite.
        """
        try:
            self.storage.log_access(user_id, chat_id, name, outcome, detail)
        except Exception:
            LOGGER.exception("Не удалось записать попытку входа")
        note = f" ({detail})" if detail else ""
        if outcome in ADMITTED:
            LOGGER.info("Вход: id=%s исход=%s%s", user_id, outcome, note)
        else:
            LOGGER.warning("Отказ во входе: id=%s причина=%s%s", user_id, outcome, note)


def main() -> None:
    try:
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = EnglishLabBot(settings)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.run()


if __name__ == "__main__":
    main()
