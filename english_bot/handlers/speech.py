"""Устная практика: задания, расшифровка Whisper, разбор и произношение слов."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..ai.llm import LLMError
from ..ai.prompts import PRONUNCIATION_SYSTEM
from ..ai.stt import TranscriptionError
from ..ai.tts import SpeechError
from ..content.banks import VocabItem
from ..context import AI_OFF_TEXT, SPEECH_OFF_TEXT, Context, Ticket
from ..learning.progress import skill_level
from ..learning.scoring import assess_speaking, format_assessment
from ..runtime import Job
from ..storage import User
from ..telegram_api import TelegramError, inline
from .menu import MAIN_KEYBOARD, RESUME_KEYBOARD


LOGGER = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z '\-]{0,60}$")

# Короткую запись локальная модель обрабатывает быстрее, чем человек прочитает
# предупреждение: сообщение об ожидании нужно только длинным ответам.
LONG_VOICE_SECONDS = 30

# Внутренние метки банка заданий по-английски; ученику нужен русский.
MODE_LABELS: dict[str, str] = {
    "monologue": "монолог",
    "roleplay": "ролевая ситуация",
    "opinion": "мнение",
    "describe": "описание",
    "interview": "интервью",
}


# ── устные задания ───────────────────────────────────────────────


def command_speaking(ctx: Context, user: User, text: str) -> None:
    # Записывать голосовое в никуда обиднее, чем услышать отказ сразу.
    if ctx.transcriber is None:
        ctx.say(user, SPEECH_OFF_TEXT, MAIN_KEYBOARD)
        return
    # Уровень речи отдельный: общий измеряется узнаванием грамматики, а
    # говорить человек почти всегда умеет хуже, чем узнавать.
    level = skill_level(ctx.storage, user, "speaking")
    done = _reviewed_speaking_tasks(ctx, user)
    task = ctx.curriculum.pick_speaking(level, done, ctx.rng)
    if task is None:
        ctx.say(user, f"Для уровня {level} устных заданий пока нет.")
        return

    ctx.storage.set_state(user.user_id, "speaking", {"task_id": task.id})
    ctx.say(user, _speaking_task_text(ctx, task), RESUME_KEYBOARD)
    ctx.log_start(user, "speaking")


def _reviewed_speaking_tasks(ctx: Context, user: User) -> set[str]:
    """Задание считается пройденным только после разбора, а не после записи."""
    return {
        str(row["task_id"])
        for row in ctx.storage.voices(user.user_id)
        if str(row.get("feedback") or "")
    }


def _speaking_task_text(ctx: Context, task: Any) -> str:
    focus = ", ".join(task.focus) if task.focus else "свободно"
    mode = MODE_LABELS.get(task.mode, task.mode)
    lines = [
        f"🎙 {task.title_ru} · {task.level} · {mode}",
        "",
        task.prompt_en,
        "",
        f"Как отвечать: {task.guidance_ru}",
        f"Длительность: {task.seconds_min}–{task.seconds_max} секунд. В фокусе: {focus}.",
        "",
        "Пришли одно голосовое. Не переписывай из-за мелких запинок — мне нужна живая речь.",
    ]
    if ctx.llm is None:
        # Расшифровка есть, разбора не будет: сказать до записи, а не после.
        lines.append("")
        lines.append("Разбор по критериям сейчас недоступен — верну только расшифровку.")
    return "\n".join(lines)


def remind_speaking(ctx: Context, user: User) -> None:
    """Возврат к устному заданию: то же задание, а не новое."""
    task = _find_speaking_task(ctx, str(user.state_data.get("task_id") or ""))
    if task is None:
        ctx.reset_state(user)
        ctx.say(user, "Задание потерялось. Возьми новое кнопкой «🎙 Речь».", MAIN_KEYBOARD)
        return
    ctx.say(user, "Продолжаем устное задание.\n\n" + _speaking_task_text(ctx, task), RESUME_KEYBOARD)


def handle_voice(ctx: Context, user: User, message: dict[str, Any]) -> None:
    """Быстрая половина: проверить запись и отдать её фоновой задаче.

    Скачивание, Whisper и разбор занимают от десятков секунд до минут. Держать
    ими дорожку обновлений нельзя: пока она занята, у человека не работают ни
    кнопки, ни `/stop` — со стороны это выглядит как умерший бот.
    """
    voice = message.get("voice") or {}
    telegram_message_id = int(message.get("message_id") or 0)
    duration = int(voice.get("duration") or 0)
    file_id = str(voice.get("file_id") or "")

    if not file_id:
        ctx.say(user, "Telegram прислал запись без идентификатора. Пришли ещё раз.")
        return
    if duration > ctx.settings.max_voice_seconds:
        ctx.say(
            user,
            f"Запись {duration} с — длиннее лимита {ctx.settings.max_voice_seconds} с. "
            "Пришли покороче.",
        )
        return
    if int(voice.get("file_size") or 0) > 20_000_000:
        ctx.say(user, "Голосовое больше 20 МБ — Telegram не отдаст его боту.")
        return

    task_id = str(user.state_data.get("task_id") or "") if user.state == "speaking" else ""
    # Ограда задачи — пара «состояние + задание»: за минуту разбора человек мог
    # взять другое устное задание, и вернуть его в покой было бы кражей.
    ticket = ctx.ticket(user, "task_id", task_id)
    ctx.background(
        ticket,
        "расшифровку голосового",
        _process_voice,
        file_id,
        str(voice.get("file_unique_id") or ""),
        telegram_message_id,
        duration,
        task_id,
        notice=_wait_note(ctx, duration),
    )


def _wait_note(ctx: Context, duration: int) -> str:
    """Оценка ожидания. Короткую запись модель обработает быстрее, чем её прочитают."""
    if ctx.settings.speech_backend != "local" or duration <= LONG_VOICE_SECONDS:
        return ""
    from ..ai.local_speech import estimate_seconds

    ahead = int(getattr(ctx.transcriber, "queue_ahead", 0) or 0)
    queue_note = f" Перед тобой в очереди: {ahead}." if ahead else ""
    return (
        f"Расшифровываю запись на {duration} с — это займёт около "
        f"{estimate_seconds(duration)} с.{queue_note}"
    )


def _process_voice(
    job: Job,
    ctx: Context,
    ticket: Ticket,
    file_id: str,
    file_unique_id: str,
    telegram_message_id: int,
    duration: int,
    task_id: str,
) -> None:
    """Медленная половина: скачать, расшифровать, разобрать, записать результат."""
    destination = ctx.settings.voice_dir / str(ticket.user_id) / f"{telegram_message_id}.ogg"
    try:
        ctx.telegram.download_file(file_id, destination)
    except TelegramError:
        LOGGER.exception("Не удалось скачать голосовое")
        ctx.tell(ticket, "Не смог забрать запись у Telegram. Пришли ещё раз.")
        return
    # Скачивание длится секунды, а `/forget YES` за это время успевает очистить
    # данные: строку в уже вычищенную таблицу писать нельзя.
    job.checkpoint()

    ctx.storage.add_voice(
        user_id=ticket.user_id,
        telegram_message_id=telegram_message_id,
        file_id=file_id,
        file_unique_id=file_unique_id,
        duration_seconds=duration,
        local_path=destination,
        task_id=task_id or "free_speech",
    )
    job.checkpoint()

    if not ctx.claim_speech(ticket):
        _release_speaking(ctx, ticket)
        ctx.tell(ticket, f"Запись сохранил ({duration} с), но расшифровать её сейчас нечем.")
        return

    assert ctx.transcriber is not None
    try:
        transcript = ctx.transcriber.transcribe(destination, duration)
    except TranscriptionError as exc:
        LOGGER.warning("Whisper не справился: %s", exc)
        ctx.tell(ticket, "Не удалось расшифровать запись. Попробуй ещё раз, поближе к микрофону.")
        return
    job.checkpoint()

    ctx.storage.set_voice_transcript(
        ticket.user_id, telegram_message_id, transcript.text, transcript.words
    )
    ctx.tell(
        ticket,
        f"Расшифровка ({transcript.seconds} с, {transcript.words} слов, "
        f"{transcript.pace_note_ru}, заполнителей {transcript.fillers}):\n\n{transcript.text}",
    )

    if not task_id:
        # Голосовое пришло вне задания: расшифровать полезно, но оценивать не по чему.
        _release_speaking(ctx, ticket)
        ctx.tell(
            ticket,
            "Записал и расшифровал. Разбор по критериям делаю только по заданию — "
            "возьми его кнопкой «🎙 Речь».",
            inline([[("Взять задание", "speak")]]),
        )
        return
    task = _find_speaking_task(ctx, task_id)
    if task is None:
        _release_speaking(ctx, ticket)
        ctx.tell(
            ticket,
            "Расшифровка готова, но задание к ней потерялось — возьми новое.",
            inline([[("Ещё задание", "speak")]]),
        )
        return
    if ctx.llm is None:
        # Единственный путь, который раньше заканчивался молчанием: расшифровка
        # есть, разбирать нечем, и человек не понимал, ждать ли оценку.
        _release_speaking(ctx, ticket)
        ctx.tell(ticket, AI_OFF_TEXT, inline([[("Ещё задание", "speak")]]))
        return
    if not ctx.storage.take_ai_call(ticket.user_id, ctx.settings.daily_ai_calls):
        ctx.tell(ticket, "Расшифровка есть, но лимит обращений к ИИ на сегодня исчерпан.")
        _release_speaking(ctx, ticket)
        return

    job.progress("разбор устного ответа")
    user = ctx.who(ticket)
    try:
        assessment = assess_speaking(
            ctx.llm, ticket.user_id, task, transcript, (user.level if user else "") or "A2"
        )
    except LLMError as exc:
        LOGGER.warning("Разбор речи не удался: %s", exc)
        ctx.tell(
            ticket,
            "Расшифровка сохранена, но разбор не получился. "
            "Попробуй ещё раз через кнопку «🎙 Речь».",
        )
        _release_speaking(ctx, ticket)
        return
    job.checkpoint()

    report = format_assessment(assessment, ctx.curriculum, "🎙 Разбор устного ответа")
    if not ctx.storage.set_voice_feedback(ticket.user_id, telegram_message_id, report):
        # Разбор этой записи уже записан: второй раз считать навык и серию нельзя.
        LOGGER.info("Разбор голосового уже сохранён, повтор пропущен")
        return
    for correction in assessment.corrections:
        ctx.storage.log_error(
            ticket.user_id, correction.category, correction.original, correction.corrected,
            correction.note, correction.pattern_id, source="speaking",
        )
    if assessment.scores:
        ctx.storage.set_skill(
            ticket.user_id, "speaking", round(assessment.average / 9 * 5), max(1, duration // 60)
        )
    ctx.storage.bump_streak(ticket.user_id)
    if _release_speaking(ctx, ticket):
        ctx.storage.log_event(ticket.user_id, "finish", "speaking")

    buttons = [[("Ещё задание", "speak")]]
    if assessment.sounds:
        first = assessment.sounds[0]
        note = ctx.curriculum.sound(first)
        if note and note.minimal_pairs:
            word = note.minimal_pairs[0].split("/")[0].strip()
            if WORD_PATTERN.match(word):
                buttons[0].append((f"Послушать {word}", f"say:{word[:40]}"))
    ctx.tell(ticket, report, inline(buttons))


def _release_speaking(ctx: Context, ticket: Ticket) -> bool:
    """Снимает состояние только с того устного задания, ради которого шла задача.

    Голосовое, присланное посреди тренировки или диагностики, раньше молча
    стирало её безусловным `reset_state`.
    """
    if ticket.state != "speaking":
        return False
    return ctx.release_state(ticket)


def _find_speaking_task(ctx: Context, task_id: str):
    for tasks in ctx.curriculum.speaking.values():
        for task in tasks:
            if task.id == task_id:
                return task
    return None


def callback_speaking(ctx: Context, user: User, payload: str) -> str:
    command_speaking(ctx, ctx.reload_user(user), "")
    return ""


# ── произношение ─────────────────────────────────────────────────


def command_say(ctx: Context, user: User, text: str) -> None:
    word = text.partition(" ")[2].strip()
    if not word:
        ctx.say(
            user,
            "Как это произносится? Напиши так: /say schedule\n"
            "Работает и с фразой: /say I would rather not",
        )
        return
    _pronounce(ctx, user, word, slow=False)


def callback_say(ctx: Context, user: User, payload: str) -> str:
    _pronounce(ctx, user, payload.strip(), slow=False)
    return payload[:40]


def callback_say_slow(ctx: Context, user: User, payload: str) -> str:
    _pronounce(ctx, user, payload.strip(), slow=True)
    return "медленно"


def _pronounce(ctx: Context, user: User, phrase: str, slow: bool) -> None:
    phrase = phrase.strip()[:120]
    if not phrase:
        return

    cached = ctx.storage.pronunciation(phrase)
    # Готовые транскрипция и file_id — это один JSON-запрос: задача не нужна,
    # и человек получает звук мгновенно.
    if not slow and cached and cached.get("ipa") and cached.get("file_id"):
        caption = _caption(_cached_info(cached, phrase), slow)
        try:
            ctx.telegram.send_voice_by_id(
                user.chat_id, str(cached["file_id"]), caption, reply_markup=_slow_button(phrase)
            )
            return
        except TelegramError:
            LOGGER.info("Кэш file_id протух, синтезирую заново")

    ctx.background(ctx.ticket(user), "озвучку слова", _pronounce_job, phrase, slow)


def _pronounce_job(job: Job, ctx: Context, ticket: Ticket, phrase: str, slow: bool) -> None:
    """Транскрипция у модели, синтез и загрузка — всё, что считается минутами."""
    cached = ctx.storage.pronunciation(phrase)
    info = _lookup(ctx, ticket, phrase, cached)
    if info is None:
        return
    job.checkpoint()

    caption = _caption(info, slow)
    if ctx.speaker is None:
        ctx.tell(ticket, caption + "\n\nОзвучка выключена — транскрипция выше верна.")
        return

    if not slow and cached and cached.get("file_id"):
        try:
            ctx.telegram.send_voice_by_id(
                ticket.chat_id, str(cached["file_id"]), caption, reply_markup=_slow_button(phrase)
            )
            return
        except TelegramError:
            LOGGER.info("Кэш file_id протух, синтезирую заново")

    if not ctx.storage.take_ai_call(ticket.user_id, ctx.settings.daily_ai_calls):
        ctx.tell(ticket, caption + "\n\nОзвучку не сделал: лимит обращений к ИИ на сегодня.")
        return

    job.progress("озвучку слова")
    try:
        audio: Path = ctx.speaker.synthesize(info.get("say") or phrase, slow=slow)
    except SpeechError as exc:
        LOGGER.warning("Синтез не удался: %s", exc)
        ctx.tell(ticket, caption + "\n\nОзвучить не получилось, транскрипция выше верна.")
        return
    job.checkpoint()

    try:
        file_id = ctx.telegram.send_voice(
            ticket.chat_id, audio, caption, reply_markup=None if slow else _slow_button(phrase)
        )
    except TelegramError:
        LOGGER.exception("Не удалось отправить голосовое")
        ctx.tell(ticket, caption)
        return

    ctx.storage.save_pronunciation(
        phrase,
        str(info.get("display") or phrase),
        str(info.get("ipa") or ""),
        str(info.get("note_ru") or ""),
        audio_path=str(audio),
        file_id="" if slow else file_id,
        syllables=str(info.get("syllables") or ""),
    )


def _cached_info(cached: dict[str, Any], phrase: str) -> dict[str, Any]:
    """Разбор кэшированной строки в тот же вид, что отдаёт `_lookup`."""
    return {
        "display": cached.get("display") or phrase,
        "ipa": cached.get("ipa"),
        "syllables": cached.get("syllables") or "",
        "note_ru": cached.get("note") or "",
        "contrast": "",
        "say": cached.get("display") or phrase,
    }


def _slow_button(phrase: str) -> dict:
    """Кнопка живёт на самом голосовом, а не отдельным сообщением."""
    return inline([[("Медленно, по слогам", f"slow:{phrase[:40]}")]])


def _caption(info: dict[str, Any], slow: bool) -> str:
    lines = [f"🔊 {info.get('display') or ''}"]
    ipa = str(info.get("ipa") or "")
    if ipa:
        lines.append(ipa)
    syllables = str(info.get("syllables") or "")
    if syllables:
        lines.append(syllables)
    note = str(info.get("note_ru") or "")
    if note:
        lines.append("")
        lines.append(note)
    contrast = str(info.get("contrast") or "")
    if contrast:
        lines.append(contrast)
    if slow:
        lines.append("")
        lines.append("Медленный вариант.")
    return "\n".join(lines)


def _lookup(
    ctx: Context, ticket: Ticket, phrase: str, cached: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Ищет транскрипцию: кэш → банк лексики → модель."""
    if cached and cached.get("ipa"):
        return _cached_info(cached, phrase)

    item = _vocab_match(ctx, phrase)
    if item is not None:
        info = {
            "display": item.word,
            "ipa": item.ipa_us,
            "syllables": "",
            "note_ru": f"{item.translation_ru}. {item.example_en}",
            "contrast": "",
            "say": item.word,
        }
        ctx.storage.save_pronunciation(phrase, item.word, item.ipa_us, str(info["note_ru"]))
        return info

    if ctx.llm is None and ctx.speaker is not None:
        # Транскрипцию без модели не взять, но озвучить слово синтез умеет —
        # это лучше, чем отказать целиком.
        return {
            "display": phrase,
            "ipa": "",
            "syllables": "",
            "note_ru": "Транскрипции нет: ИИ не настроен. Озвучиваю как есть.",
            "contrast": "",
            "say": phrase,
        }
    llm = ctx.claim_llm(ticket)
    if llm is None:
        return None
    note = ctx.slow_note(f"Слова «{phrase[:40]}» нет в словаре — спрашиваю модель, это до минуты.")
    if note:
        ctx.tell(ticket, note)
    try:
        data = llm.complete_json(
            PRONUNCIATION_SYSTEM,
            [{"role": "user", "content": phrase}],
            user_id=ticket.user_id,
            max_tokens=500,
        )
    except LLMError as exc:
        LOGGER.warning("Транскрипция не получена: %s", exc)
        ctx.tell(ticket, "Не смог разобрать это слово. Проверь написание.")
        return None
    # Модель может вернуть число, список или null в любом поле — приводим к строкам.
    info = {
        key: str(data.get(key) or "").strip()
        for key in ("display", "ipa", "syllables", "note_ru", "contrast", "say")
    }
    info["display"] = info["display"] or phrase
    info["say"] = info["say"] or info["display"]
    ctx.storage.save_pronunciation(
        phrase, info["display"], info["ipa"], info["note_ru"], syllables=info["syllables"]
    )
    return info


def _vocab_match(ctx: Context, phrase: str) -> VocabItem | None:
    needle = phrase.strip().lower()
    for items in ctx.curriculum.vocabulary.values():
        for item in items:
            if item.word.lower() == needle:
                return item
    return None
