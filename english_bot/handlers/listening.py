"""Аудирование: синтезируем скрытый текст, задаём вопрос и раскрываем скрипт после ответа."""

from __future__ import annotations

import logging
from pathlib import Path

from ..ai.tts import SpeechError
from ..content.banks import ListeningTask
from ..context import Context
from ..storage import User
from ..telegram_api import TelegramError, inline


LOGGER = logging.getLogger(__name__)
LETTERS = "ABCD"
MAX_PLAYS = 3


def command_listening(ctx: Context, user: User, text: str = "") -> None:
    level = user.level or "A2"
    done = ctx.storage.completed_session_subjects(user.user_id, "listening")
    task = ctx.curriculum.pick_listening(level, done, ctx.rng)
    if task is None:
        ctx.say(user, f"Для уровня {level} заданий на аудирование пока нет.")
        return
    if not ctx.require_speaker(user):
        return

    ctx.typing(user, "record_voice")
    assert ctx.speaker is not None
    try:
        audio: Path = ctx.speaker.synthesize(task.script_en)
    except SpeechError as exc:
        LOGGER.warning("Не удалось подготовить аудирование: %s", exc)
        ctx.say(user, "Не смог подготовить аудио. Попробуй другое задание чуть позже.")
        return

    code = ctx.curriculum.listening_code(task.id)
    caption = _caption(task)
    try:
        file_id = ctx.telegram.send_voice(
            user.chat_id, audio, caption, reply_markup=_answer_keyboard(task, code)
        )
    except TelegramError:
        LOGGER.exception("Не удалось отправить аудирование")
        ctx.say(user, "Аудио готово, но Telegram его не принял. Попробуй ещё раз.")
        return

    session_id = ctx.storage.start_session(user.user_id, "listening", task.id)
    ctx.storage.set_state(
        user.user_id,
        "listening",
        {
            "task_id": task.id,
            "task_code": code,
            "session_id": session_id,
            "file_id": file_id,
            "audio_path": str(audio),
            "plays": 1,
        },
    )


def callback_listening(ctx: Context, user: User, payload: str) -> str:
    ctx.reset_state(user)
    command_listening(ctx, ctx.reload_user(user))
    return ""


def callback_answer(ctx: Context, user: User, payload: str) -> str:
    code, separator, raw_index = payload.partition(":")
    if not separator or code != str(user.state_data.get("task_code") or ""):
        return "это задание уже закрыто"
    task = ctx.curriculum.listening_by_code(code)
    try:
        selected = int(raw_index)
    except ValueError:
        return "не понял вариант"
    if task is None or not 0 <= selected < len(task.options):
        return "задание не найдено"

    correct = selected == task.correct_index
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

    status = "✅ Верно." if correct else "❌ Не тот вариант."
    answer = task.options[task.correct_index]
    ctx.say(
        user,
        f"{status}\n\n"
        f"Правильный ответ: {LETTERS[task.correct_index]}. {answer}\n"
        f"Почему: {task.explanation_ru}\n\n"
        f"Текст записи:\n{task.script_en}\n\n"
        f"Прослушиваний: {plays} · результат по аудированию: {right}/{total}",
        inline(
            [[("🎧 Ещё аудирование", "listen"), ("🎙 Ответить голосом", "speak")],
             [("Прогресс", "progress")]]
        ),
    )
    return "верно" if correct else f"ответ {LETTERS[task.correct_index]}"


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

    try:
        file_id = str(user.state_data.get("file_id") or "")
        if file_id:
            ctx.telegram.send_voice_by_id(
                user.chat_id,
                file_id,
                _caption(task),
                reply_markup=_answer_keyboard(task, code),
            )
        else:
            path = Path(str(user.state_data.get("audio_path") or ""))
            if not path.is_file():
                return "аудио потерялось — возьми новое задание"
            file_id = ctx.telegram.send_voice(
                user.chat_id, path, _caption(task), reply_markup=_answer_keyboard(task, code)
            )
    except TelegramError:
        LOGGER.warning("Не удалось повторно отправить аудирование")
        return "не получилось повторить"

    state = dict(user.state_data)
    state["plays"] = plays + 1
    if file_id:
        state["file_id"] = file_id
    ctx.storage.set_state(user.user_id, "listening", state)
    return f"прослушивание {plays + 1} из {MAX_PLAYS}"


def _caption(task: ListeningTask) -> str:
    options = "\n".join(f"{LETTERS[index]}. {option}" for index, option in enumerate(task.options))
    return (
        f"🎧 {task.title_ru} · {task.level} · {task.skill}\n\n"
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
