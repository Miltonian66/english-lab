"""Аудирование: синтезируем скрытый текст, задаём вопрос и раскрываем скрипт после ответа."""

from __future__ import annotations

import logging
from pathlib import Path

from ..ai.tts import SpeechError
from ..content.banks import ListeningTask
from ..content.registry import Curriculum
from ..context import SPEECH_OFF_TEXT, Context, Ticket
from ..learning.answers import option_order
from ..learning.progress import skill_level
from ..runtime import Job
from ..storage import User
from ..telegram_api import TelegramError, inline


LOGGER = logging.getLogger(__name__)
LETTERS = "ABCD"
MAX_PLAYS = 3

# Внутренние метки банка по-английски; в русском интерфейсе они читаются как
# отладочный вывод.
SKILL_LABELS: dict[str, str] = {
    "gist": "общий смысл",
    "detail": "детали",
    "inference": "выводы",
    "attitude": "отношение говорящего",
    "sequence": "последовательность",
}


def command_listening(ctx: Context, user: User, text: str = "") -> None:
    if ctx.speaker is None:
        ctx.say(user, SPEECH_OFF_TEXT)
        return
    level = skill_level(ctx.storage, user, "listening")
    done = ctx.storage.completed_session_subjects(user.user_id, "listening")
    task = ctx.curriculum.pick_listening(level, done, ctx.rng)
    if task is None:
        ctx.say(user, f"Для уровня {level} заданий на аудирование пока нет.")
        return
    # Ни сессия, ни состояние вперёд не ставятся: иначе кнопки A–D под ещё не
    # пришедшим аудио позволили бы ответить вслепую и сожгли бы задание.
    # Двойное нажатие закрывает правило «одна задача на человека».
    code = ctx.curriculum.listening_code(task.id)
    if not ctx.background(ctx.ticket(user), "подготовку аудирования", _listening_job, code):
        return
    ctx.log_start(user, "listening")
    ctx.typing(user, "record_voice")


def _listening_job(job: Job, ctx: Context, ticket: Ticket, code: str) -> None:
    task = ctx.curriculum.listening_by_code(code)
    if task is None:
        ctx.tell(ticket, "Задание на аудирование потерялось. Возьми новое — «🎧 Аудирование».")
        return
    if not ctx.claim_speaker(ticket):
        return
    assert ctx.speaker is not None
    try:
        audio: Path = ctx.speaker.synthesize(task.script_en)
    except SpeechError as exc:
        LOGGER.warning("Не удалось подготовить аудирование: %s", exc)
        ctx.tell(ticket, "Не смог подготовить аудио. Попробуй другое задание чуть позже.")
        return
    job.checkpoint()

    caption = _caption(task, ctx.curriculum)
    try:
        file_id = ctx.telegram.send_voice(
            ticket.chat_id, audio, caption, reply_markup=_answer_keyboard(task, code)
        )
    except TelegramError:
        LOGGER.exception("Не удалось отправить аудирование")
        ctx.tell(ticket, "Аудио готово, но Telegram его не принял. Попробуй ещё раз.")
        return
    # Загрузка Ogg идёт десятки секунд: `/stop` за это время не должен получить
    # взамен открытую сессию и состояние «слушаем».
    job.checkpoint()

    session_id = ctx.storage.start_session(ticket.user_id, "listening", task.id)
    if not ctx.hold_state(
        ticket,
        "listening",
        {
            "task_id": task.id,
            "task_code": code,
            "session_id": session_id,
            "file_id": file_id,
            "audio_path": str(audio),
            "plays": 1,
        },
    ):
        # Человек успел начать другое занятие: сессию закрываем сразу, иначе она
        # осталась бы открытой и испортила счётчик занятий дня.
        ctx.storage.finish_session(session_id, items=0, correct=0)
        LOGGER.info("Аудирование пришло в уже закрытое занятие")


def remind_listening(ctx: Context, user: User) -> None:
    """Возврат к аудированию: аудио уже в переписке, повторяем вопрос и кнопки."""
    code = str(user.state_data.get("task_code") or "")
    task = ctx.curriculum.listening_by_code(code) if code else None
    if task is None:
        ctx.reset_state(user)
        ctx.say(user, "Задание потерялось. Возьми новое — «📊 Я» → «🎧 Аудирование».")
        return
    ctx.say(user, _caption(task, ctx.curriculum), _answer_keyboard(task, code))


def callback_listening(ctx: Context, user: User, payload: str) -> str:
    from .menu import close_active

    # Отказ обязан опередить `close_active`: иначе выключенный синтез закрывал бы
    # текущее занятие и не давал взамен ничего.
    if ctx.speaker is None:
        ctx.say(user, SPEECH_OFF_TEXT)
        return "синтез не настроен"
    close_active(ctx, ctx.reload_user(user))
    command_listening(ctx, ctx.reload_user(user))
    return ""


def callback_answer(ctx: Context, user: User, payload: str) -> str:
    code, separator, raw_index = payload.partition(":")
    if not separator or code != str(user.state_data.get("task_code") or ""):
        return "это задание уже закрыто"
    task = ctx.curriculum.listening_by_code(code)
    try:
        position = int(raw_index)
    except ValueError:
        return "не понял вариант"
    if task is None or not 0 <= position < len(task.options):
        return "задание не найдено"

    # Кнопка несёт позицию в порядке показа, а ключ хранится в порядке файла.
    order = shown_order(task, ctx.curriculum)
    selected = order[position]
    correct = selected == task.correct_index
    key_letter = LETTERS[order.index(task.correct_index)]
    session_id = int(user.state_data.get("session_id") or 0)
    plays = int(user.state_data.get("plays") or 1)
    if session_id:
        ctx.storage.finish_session(session_id, items=1, correct=int(correct))
    ctx.storage.record_attempt(
        user.user_id,
        task.id,
        "",
        task.level,
        correct,
        task.options[selected],
    )
    total, right = ctx.storage.session_totals(user.user_id, "listening")
    accuracy = right / total if total else 0.0
    # Первые ответы дают максимум 3/5; объём добавляет ещё две ступени. Так одна
    # удачная догадка не превращается в полное мастерство навыка.
    mastery = min(5, round(accuracy * 3) + min(2, total // 5))
    minutes = max(1, round(len(task.script_en.split()) / 120))
    ctx.storage.set_skill(user.user_id, "listening", mastery, minutes)
    ctx.storage.bump_streak(user.user_id)
    ctx.reset_state(user)
    ctx.storage.log_event(user.user_id, "finish", "listening")

    status = "✅ Верно." if correct else "❌ Не тот вариант."
    answer = task.options[task.correct_index]
    ctx.say(
        user,
        f"{status}\n\n"
        f"Правильный ответ: {key_letter}. {answer}\n"
        f"Почему: {task.explanation_ru}\n\n"
        f"Текст записи:\n{task.script_en}\n\n"
        f"Прослушиваний: {plays} · результат по аудированию: {right}/{total}",
        inline(
            [[("🎧 Ещё аудирование", "listen"), ("🎙 Ответить голосом", "speak")],
             [("Прогресс", "progress")]]
        ),
    )
    return "верно" if correct else f"ответ {key_letter}"


def callback_replay(ctx: Context, user: User, payload: str) -> str:
    code = payload.strip()
    if code != str(user.state_data.get("task_code") or ""):
        return "это задание уже закрыто"
    task = ctx.curriculum.listening_by_code(code)
    if task is None:
        return "задание не найдено"
    plays = int(user.state_data.get("plays") or 1)
    if plays >= MAX_PLAYS:
        return "три прослушивания — теперь выбери ответ"

    file_id = str(user.state_data.get("file_id") or "")
    if not file_id:
        # Файл ушёл бы загрузкой на десятки секунд — это уже длинная работа.
        path = Path(str(user.state_data.get("audio_path") or ""))
        if not path.is_file():
            return "аудио потерялось — возьми новое задание"
        ticket = ctx.ticket(user, "task_code", code)
        if not ctx.background(ticket, "повтор записи", _replay_job, code, str(path), plays):
            return "секунду"
        return f"прослушивание {plays + 1} из {MAX_PLAYS}"

    try:
        ctx.telegram.send_voice_by_id(
            user.chat_id, file_id, _caption(task, ctx.curriculum), reply_markup=_answer_keyboard(task, code)
        )
    except TelegramError:
        LOGGER.warning("Не удалось повторно отправить аудирование")
        return "не получилось повторить"

    state = dict(user.state_data)
    state["plays"] = plays + 1
    ctx.storage.set_state(user.user_id, "listening", state)
    return f"прослушивание {plays + 1} из {MAX_PLAYS}"


def _replay_job(
    job: Job, ctx: Context, ticket: Ticket, code: str, audio_path: str, plays: int
) -> None:
    task = ctx.curriculum.listening_by_code(code)
    if task is None:
        ctx.tell(ticket, "Задание потерялось — возьми новое.")
        return
    try:
        file_id = ctx.telegram.send_voice(
            ticket.chat_id,
            Path(audio_path),
            _caption(task, ctx.curriculum),
            reply_markup=_answer_keyboard(task, code),
        )
    except TelegramError:
        LOGGER.warning("Не удалось повторно отправить аудирование")
        ctx.tell(ticket, "Не получилось повторить запись. Попробуй ещё раз.")
        return
    user = ctx.who(ticket)
    if user is None:
        return
    state = dict(user.state_data)
    state["plays"] = plays + 1
    state["file_id"] = file_id
    # Ограда билета сверяет и состояние, и код задания: человек мог за время
    # загрузки взять другое аудирование.
    ctx.hold_state(ticket, "listening", state)


def shown_order(task: ListeningTask, curriculum: Curriculum) -> tuple[int, ...]:
    """Порядок показа вариантов: устойчивый для задания и не порядок из файла.

    В банке ключ стоял на B в 16 из 30 заданий, и «всегда B» давало больше
    половины верных ответов без понимания записи. Случайное перемешивание на
    тридцати заданиях тоже даёт перекос, поэтому позиция ключа идёт по кругу
    A, B, C, D по всему банку, а остальные варианты перемешаны по id.
    """
    count = len(task.options)
    ordered = sorted(
        (item for rows in curriculum.listening.values() for item in rows), key=lambda item: item.id
    )
    position = next((index for index, item in enumerate(ordered) if item.id == task.id), 0) % count
    others = [index for index in option_order(task.id, count) if index != task.correct_index]
    return tuple(others[:position] + [task.correct_index] + others[position:])


def _caption(task: ListeningTask, curriculum: Curriculum) -> str:
    options = "\n".join(
        f"{LETTERS[position]}. {task.options[index]}"
        for position, index in enumerate(shown_order(task, curriculum))
    )
    skill = SKILL_LABELS.get(task.skill, task.skill)
    return (
        f"🎧 {task.title_ru} · {task.level} · {skill}\n\n"
        "Прослушай запись и выбери ответ. Текст покажу только после ответа.\n\n"
        f"{task.question_en}\n{options}"
    )


def _answer_keyboard(task: ListeningTask, code: str) -> dict:
    answers = [
        (LETTERS[index], f"la:{code}:{index}") for index in range(len(task.options))
    ]
    return inline(
        [answers[:2], answers[2:], [("↻ Прослушать ещё раз", f"lr:{code}")]]
    )
