"""English Lab: многопользовательская платформа изучения английского в Telegram.

Процесс один: long polling складывает апдейты в общий пул, а отдельная очередь
каждого человека сохраняет порядок. Codex, Whisper и Piper работают в своих
ограниченных пулах и не останавливают приём новых обновлений.
"""

from __future__ import annotations

import logging
import random
import signal
import sys
import time
from typing import Any, Callable

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
    KeyedExecutor,
    OverloadedError,
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
    "chat": menu.callback_chat,
    "roleplayhint": menu.callback_roleplay_hint,
    "rp": dialogue.callback_roleplay_start,
    "askword": menu.callback_ask_word,
    "help": menu.callback_help,
    "team": menu.callback_team,
    "anki": menu.callback_anki,
    "export": menu.callback_export,
    "levelpick": menu.callback_level_pick,
    "invitenew": menu.callback_invite,
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

# Произношение и справочный вопрос не прерывают занятие: спросить про слово или
# интерфейс посреди упражнения — нормальный ход, а не смена вида деятельности.
STATE_SAFE_COMMANDS = {"/stop", "/cancel", "/say", "/help"}

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
}
STALE_BUTTON = "эта кнопка уже не активна"

def _build_llm(settings: Settings) -> LLM | None:
    """Собирает текстовую модель по `LLM_PROVIDER`. Это и есть переключатель."""
    if settings.llm_provider == "codex":
        return LLM(
            "codex",
            codex=CodexRunner(
                binary=settings.codex_binary,
                model=settings.codex_model,
                effort=settings.codex_effort,
                timeout=settings.codex_timeout,
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
            LocalSpeaker(settings.piper_dir, settings.audio_cache_dir, settings.piper_voice)
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
    "Попроси у владельца бота ссылку вида t.me/…?start=КОД."
)


class EnglishLabBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = Storage(settings.database_path)
        self.telegram = TelegramAPI(settings.telegram_token)
        self.curriculum = load_curriculum()
        self.running = True
        self._closed = False
        self._dispatcher = KeyedExecutor(settings.workers)
        self._heavy = HeavyPools(
            settings.llm_workers, settings.stt_workers, settings.tts_workers
        )

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

    def context(self) -> Context:
        return Context(
            settings=self.settings,
            storage=self.storage,
            telegram=self.telegram,
            curriculum=self.curriculum,
            llm=self.llm,
            transcriber=self.transcriber,
            speaker=self.speaker,
            rng=random.Random(),
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
        LOGGER.info(
            "Запущен как @%s · %d тем, %d упражнений · ИИ: %s · речь: %s",
            username,
            len(self.curriculum.points),
            len(self.curriculum.exercises),
            self.settings.llm_provider if self.llm else "выключен",
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
        # затем закрываем пулы, внутри которых они могли ждать тяжёлую работу.
        self._dispatcher.shutdown(wait=wait)
        self._heavy.shutdown(wait=wait)

    def _submit(self, update: dict[str, Any]) -> None:
        sender = (update.get("message") or update.get("callback_query") or {}).get("from") or {}
        user_id = int(sender.get("id") or 0)
        future = self._dispatcher.submit(user_id, self._safe_handle, update)
        if future.done():
            try:
                future.result()
            except OverloadedError:
                # Очереди имеют конечный размер: под флудом сохраняем память и polling,
                # даже если отдельное уже подтверждённое Telegram-обновление потеряется.
                LOGGER.warning("Очередь обновлений заполнена, апдейт отклонён")

    def _safe_handle(self, update: dict[str, Any]) -> None:
        try:
            self.handle_update(update)
        except Exception:
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
        if "callback_query" in update:
            self._handle_callback(update["callback_query"])
            return
        message = update.get("message")
        if message:
            self._handle_message(message)

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

        ctx = self.context()
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
        try:
            note = handler(ctx, user, payload)
        except Exception:
            LOGGER.exception("Ошибка в обработчике callback %s", prefix)
            self.telegram.answer_callback(callback_id, "Что-то сломалось")
            ctx.say(user, "Кнопка не сработала. Попробуй ещё раз или нажми «🎯 Заниматься».")
            return
        self.telegram.answer_callback(callback_id, note or "")

    def _dispatch_command(self, ctx: Context, user: User, text: str) -> None:
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        handler = COMMANDS.get(command)
        if handler is None:
            ctx.say(user, core.HELP_TEXT)
            return
        # Любая команда вытаскивает из незавершённого занятия: иначе следующее
        # обычное сообщение уйдёт, например, в разбор письма вместо чата.
        if user.state != "idle" and command not in STATE_SAFE_COMMANDS:
            ctx.reset_state(user)
            user = ctx.reload_user(user)
        handler(ctx, user, text)

    def _dispatch_text(self, ctx: Context, user: User, text: str) -> None:
        # Кнопки постоянной клавиатуры приходят обычным текстом. Они работают из
        # любого состояния — иначе «одно нажатие» ломалось бы посреди занятия.
        if menu.is_button(text):
            if user.state != "idle":
                ctx.reset_state(user)
                user = ctx.reload_user(user)
            menu.handle_button(ctx, user, text)
            return
        if user.state == "awaiting_word":
            menu.handle_awaiting_word(ctx, user, text)
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

        payload = ""
        parts = text.split(maxsplit=1)
        if parts and parts[0].split("@", 1)[0].lower() == "/start" and len(parts) > 1:
            payload = parts[1].strip()

        owner = self.storage.get_setting("owner_user_id")
        if owner is None:
            if payload and payload == self.settings.claim_code:
                self.storage.set_setting("owner_user_id", str(user_id))
                user = self.storage.create_user(user_id, chat_id, "owner", name)
                LOGGER.info("Владелец привязан")
                core.command_start(ctx, user, text)
                return None
            self.telegram.send_message(chat_id, NOT_LINKED)
            return None

        if payload:
            role = self.storage.redeem_invite(payload, user_id)
            if role:
                user = self.storage.create_user(user_id, chat_id, role, name)
                LOGGER.info("Новый участник принят по приглашению, роль %s", role)
                core.command_start(ctx, user, text)
                return None

        if self.settings.team_open_registration:
            user = self.storage.create_user(user_id, chat_id, "member", name)
            core.command_start(ctx, user, text)
            return None

        self.telegram.send_message(chat_id, NEED_INVITE)
        return None


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
