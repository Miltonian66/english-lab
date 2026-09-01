"""Контекст выполнения: всё, что нужно обработчику, в одном объекте.

Живёт отдельным модулем, чтобы обработчики не импортировали `app` и не создавали
цикл. Здесь же — общие помощники ответа и учёт лимита обращений к ИИ.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from .ai.llm import LLM
from .ai.stt import Transcriber
from .ai.tts import Speaker
from .config import Settings
from .content.registry import Curriculum
from .storage import Storage, User
from .telegram_api import TelegramAPI


LOGGER = logging.getLogger(__name__)

AI_OFF_TEXT = (
    "Эта функция работает через ИИ, а он сейчас не настроен. "
    "Скажи владельцу бота — нужен LLM_PROVIDER в .env. "
    "Курс, тренировка и повторение работают и без него."
)
SPEECH_OFF_TEXT = (
    "Голос сейчас не настроен: нет ни локальных моделей, ни ключа OpenAI. "
    "Скажи владельцу бота."
)
LIMIT_TEXT = (
    "На сегодня лимит обращений к ИИ исчерпан — так платформа не разоряет владельца ключа. "
    "Тренировки, повторение и тесты работают без ограничений."
)


@dataclass
class Context:
    settings: Settings
    storage: Storage
    telegram: TelegramAPI
    curriculum: Curriculum
    llm: LLM | None
    transcriber: Transcriber | None
    speaker: Speaker | None
    rng: random.Random

    # ── ответы ───────────────────────────────────────────────────

    def say(
        self,
        user: User,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str | None = None,
    ) -> None:
        self.telegram.send_message(user.chat_id, text, reply_markup, parse_mode)

    def typing(self, user: User, action: str = "typing") -> None:
        self.telegram.send_chat_action(user.chat_id, action)

    def working(self, user: User, note: str) -> None:
        """Предупреждает об ожидании, если провайдер отвечает долго.

        Индикатор «печатает» живёт пару секунд, а `codex exec` считает 8–12 секунд
        и дольше. Без явного сообщения человек решает, что бот умер.
        """
        self.typing(user)
        if self.settings.llm_provider == "codex":
            self.say(user, note)

    # ── доступ к ИИ ──────────────────────────────────────────────

    def require_llm(self, user: User) -> LLM | None:
        if self.llm is None:
            self.say(user, AI_OFF_TEXT)
            return None
        if not self.storage.take_ai_call(user.user_id, self.settings.daily_ai_calls):
            self.say(user, LIMIT_TEXT)
            return None
        return self.llm

    def require_speech(self, user: User) -> bool:
        if self.transcriber is None or self.speaker is None:
            self.say(user, SPEECH_OFF_TEXT)
            return False
        if not self.storage.take_ai_call(user.user_id, self.settings.daily_ai_calls):
            self.say(user, LIMIT_TEXT)
            return False
        return True

    def require_speaker(self, user: User) -> bool:
        """Синтез нужен и без распознавания — например, для аудирования."""
        if self.speaker is None:
            self.say(user, SPEECH_OFF_TEXT)
            return False
        if not self.storage.take_ai_call(user.user_id, self.settings.daily_ai_calls):
            self.say(user, LIMIT_TEXT)
            return False
        return True

    # ── состояние ────────────────────────────────────────────────

    def reload_user(self, user: User) -> User:
        fresh = self.storage.user(user.user_id)
        return fresh or user

    def reset_state(self, user: User) -> None:
        self.storage.set_state(user.user_id, "idle", {})
