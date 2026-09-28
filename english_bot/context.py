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
from .config import CLI_PROVIDERS, Settings
from .content.registry import Curriculum
from .runtime import Job, JobCancelled, JobRunner, Telemetry
from .storage import Storage, User
from .telegram_api import TelegramAPI, TelegramError


LOGGER = logging.getLogger(__name__)

AI_OFF_TEXT = (
    "Эта функция работает через ИИ, а он сейчас не настроен. "
    "Скажи владельцу платформы, что ИИ-функция недоступна. "
    "Курс, тренировка и повторение работают и без него."
)
SPEECH_OFF_TEXT = (
    "Голос сейчас не настроен. Скажи владельцу платформы, что голосовые функции недоступны."
)
LIMIT_TEXT = (
    "На сегодня лимит обращений к ИИ исчерпан — он защищает платформу от перерасхода. "
    "Тренировки, повторение и тесты работают без ограничений."
)
JOB_FAILED_TEXT = (
    "Не довёл дело до конца — ошибка на моей стороне, она записана. "
    "Попробуй ещё раз или нажми «🎯 Заниматься»."
)


@dataclass(frozen=True)
class Ticket:
    """Что фоновая задача знает о человеке. Объект `User` в неё не едет.

    Снимок `User` к моменту записи результата устаревает на минуту, а живое на
    вид поле `state_data` в нём — приглашение принять решение по чужим данным.
    Поэтому задача получает только неизменные идентификаторы и ограду
    `state`/`key`/`value`: по ней она проверяет, что человек всё ещё занят тем
    же делом, а не другим заданием того же вида.
    """

    user_id: int
    chat_id: int
    state: str = ""
    key: str = ""
    value: str = ""


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
    # Сообщение, на кнопку которого нажали. Есть только у callback-обработчиков и
    # нужно, чтобы навигация перерисовывала экран, а не плодила новые сообщения.
    message_id: int | None = None
    # Куда уходят длинные цепочки. `None` — выполнять их сразу, как раньше:
    # так работают тесты отдельных обработчиков и разовые вызовы вне бота.
    jobs: JobRunner | None = None
    # Счётчики процесса для экрана `/admin`. Без них состояние очередей и задач
    # видно только из журнала, то есть с сервера, а не из самого бота.
    telemetry: Telemetry | None = None

    # ── ответы ───────────────────────────────────────────────────

    def say(
        self,
        user: User,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str | None = None,
    ) -> None:
        self.telegram.send_message(user.chat_id, text, reply_markup, parse_mode)

    def edit(
        self,
        user: User,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str | None = None,
    ) -> None:
        """Перерисовывает экран навигации на месте; вне callback — обычная отправка.

        Каталог из четырёх уровней вложенности иначе оставляет за собой стопку
        сообщений с живыми клавиатурами.
        """
        if self.message_id is None:
            self.say(user, text, reply_markup, parse_mode)
            return
        try:
            self.telegram.edit_message_text(
                user.chat_id, self.message_id, text, reply_markup, parse_mode
            )
        except TelegramError:
            # Сообщение могли удалить или оно слишком старое — не теряем ответ.
            LOGGER.info("Не удалось отредактировать экран, отправляю новым сообщением")
            self.say(user, text, reply_markup, parse_mode)

    def log_start(self, user: User, kind: str) -> None:
        """Отмечает старт функции и глубину пути до неё.

        Глубина считается так же, как её видит человек: запуск с постоянной
        клавиатуры — одно нажатие, запуск с inline-кнопки экрана — два. Точнее
        без отслеживания всей сессии не измерить, а порядок величины важнее.
        """
        depth = 2 if self.message_id is not None else 1
        self.storage.log_event(user.user_id, "start", kind, depth)

    def typing(self, user: User, action: str = "typing") -> None:
        self.telegram.send_chat_action(user.chat_id, action)

    def slow_note(self, note: str) -> str:
        """Текст предупреждения об ожидании — или пусто, если провайдер быстрый.

        Индикатор «печатает» живёт пару секунд, а подписочный запуск через CLI
        считает от нескольких секунд до минуты: без явного сообщения человек
        решает, что бот умер. На API то же сообщение было бы шумом.
        """
        return note if self.settings.llm_provider in CLI_PROVIDERS else ""

    # ── длинная работа ───────────────────────────────────────────

    def ticket(self, user: User, key: str = "", value: str = "") -> Ticket:
        """Пропуск задачи: кто человек и чем он был занят в момент её запуска."""
        return Ticket(user.user_id, user.chat_id, user.state, key, value)

    def tell(self, ticket: Ticket, text: str, reply_markup: dict[str, Any] | None = None) -> None:
        """Ответ из фоновой задачи. Идёт по чату, а не по устаревшему `User`."""
        self.telegram.send_message(ticket.chat_id, text, reply_markup)

    def who(self, ticket: Ticket) -> User | None:
        """Свежий `User`, если задаче действительно нужны уровень или роль."""
        return self.storage.user(ticket.user_id)

    def busy(self, user: User) -> Job | None:
        """Идущая длинная задача человека, если она есть."""
        return self.jobs.active(user.user_id) if self.jobs is not None else None

    def cancel_job(self, user: User) -> str:
        """Снимает задачу человека. Возвращает её текущий этап; пусто — снимать нечего.

        Вешается точечно на `/stop`, `/cancel`, `/forget` и закрытие занятия.
        Вешать отмену на сам `reset_state` нельзя: его зовут четырнадцать мест,
        и «Закончить» под старой репликой диалога убивало бы чужой синтез.
        """
        return self.jobs.cancel(user.user_id) if self.jobs is not None else ""

    def background(
        self, ticket: Ticket, name: str, function: Any, *args: Any, notice: str = ""
    ) -> bool:
        """Уводит длинную цепочку с интерактивной дорожки человека.

        В очереди обновлений остаётся только быстрая часть — проверки, состояние
        и подтверждение приёма. Расшифровка, разбор и синтез идут отдельно,
        поэтому кнопки, экраны и `/stop` продолжают отвечать, пока они считаются.

        `function` получает `(job, ctx, ticket, *args)` и обязан звать
        `job.checkpoint()` между этапами. `notice` — предупреждение об ожидании;
        его отправляет сама `background` сразу после допуска задачи, иначе в
        инлайн-режиме оно приходило бы после готового ответа, а в фоновом —
        наперегонки с ним. `False` — задача не принята, человеку уже сказано почему.
        """

        def announce() -> None:
            # Первый пульс придёт только через интервал, а ждать человек начинает
            # сейчас.
            self.telegram.send_chat_action(ticket.chat_id)
            if notice:
                self.tell(ticket, notice)

        def run(job: Job) -> None:
            try:
                function(job, self, ticket, *args)
            except JobCancelled:
                raise  # отмену разбирает JobRunner, извиняться не за что
            except Exception as exc:
                if self.telemetry is not None:
                    self.telemetry.note_error(f"задача «{name}»", exc)
                LOGGER.exception("Фоновая задача «%s» не удалась", name)
                try:
                    self.tell(ticket, JOB_FAILED_TEXT)
                except TelegramError:
                    LOGGER.info("Не удалось сообщить об ошибке фоновой задачи")

        if self.jobs is None:
            # Инлайн-режим (`JOB_WORKERS=0`): те же ограды и тот же билет, только
            # без отдельного потока. Побочные эффекты обязаны совпадать.
            announce()
            run(Job(ticket.user_id, name, 0.0))
            return True

        running = self.jobs.active(ticket.user_id)
        if running is None:
            job = self.jobs.start(
                ticket.user_id,
                name,
                run,
                pulse=lambda: self.telegram.send_chat_action(ticket.chat_id),
            )
            if job is not None:
                announce()
                return True
            running = self.jobs.active(ticket.user_id)
        if running is None:
            # Слот свободен, а задача не встала: пул исчерпан или уже закрыт.
            LOGGER.warning("Не удалось поставить фоновую задачу «%s»", name)
            self.tell(ticket, JOB_FAILED_TEXT)
            return False
        # Очередь из длинных задач одного человека бесполезна: он всё равно ждёт
        # первую. Честный отказ с названием того, что идёт, и способом прервать.
        self.tell(ticket, f"Сначала закончу {running.stage}. Прервать — /stop.")
        return False

    def release_state(self, ticket: Ticket) -> bool:
        """Возвращает человека в покой, если он всё ещё занят тем же делом."""
        return self.storage.swap_state(
            ticket.user_id, ticket.state, "idle", {}, key=ticket.key, value=ticket.value
        )

    def hold_state(self, ticket: Ticket, state: str, data: dict[str, Any]) -> bool:
        """Ставит новое состояние с той же оградой: результат задачи не поверх чужого."""
        return self.storage.swap_state(
            ticket.user_id, ticket.state, state, data, key=ticket.key, value=ticket.value
        )

    # ── доступ к ИИ ──────────────────────────────────────────────

    # Лимит списывается там же, где делается вызов, — то есть внутри задачи.
    # Списать его в интерактивной фазе значило бы брать плату за отказ «сначала
    # закончу …», при котором к модели никто не обращался.

    def claim_llm(self, ticket: Ticket) -> LLM | None:
        if self.llm is None:
            self.tell(ticket, AI_OFF_TEXT)
            return None
        if not self.storage.take_ai_call(ticket.user_id, self.settings.daily_ai_calls):
            self.tell(ticket, LIMIT_TEXT)
            return None
        return self.llm

    def claim_speech(self, ticket: Ticket) -> bool:
        """Расшифровке нужен только распознаватель: синтез — отдельный контур.

        Раньше отсутствие `piper` отключало и голосовые, хотя `whisper` работал.
        """
        if self.transcriber is None:
            self.tell(ticket, SPEECH_OFF_TEXT)
            return False
        return self._take_speech_call(ticket)

    def claim_speaker(self, ticket: Ticket) -> bool:
        """Синтез нужен и без распознавания — например, для аудирования."""
        if self.speaker is None:
            self.tell(ticket, SPEECH_OFF_TEXT)
            return False
        return self._take_speech_call(ticket)

    def _take_speech_call(self, ticket: Ticket) -> bool:
        """Лимит считает обращения к платным сервисам, а не работу этой машины.

        Локальные Whisper и Piper денег не стоят, поэтому дневной лимит на них
        не тратится: иначе активный чат «закрывал» аудирование до завтра.
        """
        if self.settings.speech_backend == "local":
            return True
        if not self.storage.take_ai_call(ticket.user_id, self.settings.daily_ai_calls):
            self.tell(ticket, LIMIT_TEXT)
            return False
        return True

    # ── состояние ────────────────────────────────────────────────

    def reload_user(self, user: User) -> User:
        fresh = self.storage.user(user.user_id)
        return fresh or user

    def reset_state(self, user: User) -> None:
        self.storage.set_state(user.user_id, "idle", {})
