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
from ..context import Context
from ..learning.scoring import assess_speaking, format_assessment
from ..storage import User
from ..telegram_api import TelegramError, inline
from .menu import MAIN_KEYBOARD


LOGGER = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z '\-]{0,60}$")


# ── устные задания ───────────────────────────────────────────────


def command_speaking(ctx: Context, user: User, text: str) -> None:
    level = user.level or "A2"
    done = {str(row["task_id"]) for row in ctx.storage.voices(user.user_id)}
    task = ctx.curriculum.pick_speaking(level, done, ctx.rng)
    if task is None:
        ctx.say(user, f"Для уровня {level} устных заданий пока нет.")
        return

    ctx.storage.set_state(user.user_id, "speaking", {"task_id": task.id})
    focus = ", ".join(task.focus) if task.focus else "свободно"
    ctx.say(
        user,
        f"🎙 {task.title_ru} · {task.level} · {task.mode}\n\n"
        f"{task.prompt_en}\n\n"
        f"Как отвечать: {task.guidance_ru}\n"
        f"Длительность: {task.seconds_min}–{task.seconds_max} секунд. "
        f"В фокусе: {focus}.\n\n"
        "Пришли одно голосовое. Не переписывай из-за мелких запинок — мне нужна живая речь.",
        MAIN_KEYBOARD,
    )


def handle_voice(ctx: Context, user: User, message: dict[str, Any]) -> None:
    voice = message.get("voice") or {}
    telegram_message_id = int(message.get("message_id") or 0)
    duration = int(voice.get("duration") or 0)

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
    destination = ctx.settings.voice_dir / str(user.user_id) / f"{telegram_message_id}.ogg"
    try:
        ctx.telegram.download_file(str(voice["file_id"]), destination)
    except (KeyError, TelegramError):
        LOGGER.exception("Не удалось скачать голосовое")
        ctx.say(user, "Не смог забрать запись у Telegram. Пришли ещё раз.")
        return

    ctx.storage.add_voice(
        user_id=user.user_id,
        telegram_message_id=telegram_message_id,
        file_id=str(voice["file_id"]),
        file_unique_id=str(voice.get("file_unique_id") or ""),
        duration_seconds=duration,
        local_path=destination,
        task_id=task_id or "free_speech",
    )

    if not ctx.require_speech(user):
        ctx.say(user, f"Запись сохранил ({duration} с), но расшифровать нечем.")
        return

    ctx.typing(user)
    if ctx.settings.speech_backend == "local":
        from ..ai.local_speech import estimate_seconds

        ctx.say(
            user,
            f"Расшифровываю запись на {duration} с — это займёт около "
            f"{estimate_seconds(duration)} с. Модель работает прямо на этой машине.",
        )
    assert ctx.transcriber is not None
    try:
        transcript = ctx.transcriber.transcribe(destination, duration)
    except TranscriptionError as exc:
        LOGGER.warning("Whisper не справился: %s", exc)
        ctx.say(user, "Не удалось расшифровать запись. Попробуй ещё раз, поближе к микрофону.")
        return

    ctx.storage.set_voice_result(
        user.user_id, telegram_message_id, transcript.text, "", transcript.words
    )
    ctx.say(
        user,
        f"Расшифровка ({transcript.seconds} с, {transcript.words} слов, "
        f"{transcript.pace_note_ru}, заполнителей {transcript.fillers}):\n\n{transcript.text}",
    )

    if not task_id:
        # Голосовое пришло вне задания: расшифровать полезно, но оценивать не по чему.
        ctx.reset_state(user)
        ctx.say(
            user,
            "Записал и расшифровал. Разбор по критериям делаю только по заданию — "
            "возьми его кнопкой «🎙 Речь».",
            inline([[("Взять задание", "speak")]]),
        )
        return
    task = _find_speaking_task(ctx, task_id)
    if task is None or ctx.llm is None:
        ctx.reset_state(user)
        return
    if not ctx.storage.take_ai_call(user.user_id, ctx.settings.daily_ai_calls):
        ctx.say(user, "Расшифровка есть, но лимит обращений к ИИ на сегодня исчерпан.")
        ctx.reset_state(user)
        return

    ctx.typing(user)
    try:
        assessment = assess_speaking(ctx.llm, user.user_id, task, transcript, user.level or "A2")
    except LLMError as exc:
        LOGGER.warning("Разбор речи не удался: %s", exc)
        ctx.say(
            user,
            "Расшифровка сохранена, но разбор не получился. "
            "Попробуй ещё раз через кнопку «🎙 Речь».",
        )
        ctx.reset_state(user)
        return

    report = format_assessment(assessment, ctx.curriculum, "🎙 Разбор устного ответа")
    ctx.storage.set_voice_result(
        user.user_id, telegram_message_id, transcript.text, report, transcript.words
    )
    for correction in assessment.corrections:
        ctx.storage.log_error(
            user.user_id, correction.category, correction.original, correction.corrected,
            correction.note, correction.pattern_id, source="speaking",
        )
    if assessment.scores:
        ctx.storage.set_skill(
            user.user_id, "speaking", round(assessment.average / 9 * 5), max(1, duration // 60)
        )
    ctx.storage.bump_streak(user.user_id)
    ctx.reset_state(user)

    buttons = [[("Ещё задание", "speak")]]
    if assessment.sounds:
        first = assessment.sounds[0]
        note = ctx.curriculum.sound(first)
        if note and note.minimal_pairs:
            word = note.minimal_pairs[0].split("/")[0].strip()
            if WORD_PATTERN.match(word):
                buttons[0].append((f"Послушать {word}", f"say:{word[:40]}"))
    ctx.say(user, report, inline(buttons))


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
    info = _lookup(ctx, user, phrase, cached)
    if info is None:
        return

    caption = _caption(info, slow)
    if ctx.speaker is None:
        ctx.say(user, caption + "\n\nОзвучка выключена — транскрипция выше верна.")
        return

    # Готовый file_id переиспользуем — это бесплатно и мгновенно.
    if not slow and cached and cached.get("file_id"):
        try:
            ctx.telegram.send_voice_by_id(
                user.chat_id, str(cached["file_id"]), caption, reply_markup=_slow_button(phrase)
            )
            return
        except TelegramError:
            LOGGER.info("Кэш file_id протух, синтезирую заново")

    if not ctx.storage.take_ai_call(user.user_id, ctx.settings.daily_ai_calls):
        ctx.say(user, caption + "\n\nОзвучку не сделал: лимит обращений к ИИ на сегодня.")
        return

    ctx.typing(user, "record_voice")
    try:
        audio: Path = ctx.speaker.synthesize(info.get("say") or phrase, slow=slow)
    except SpeechError as exc:
        LOGGER.warning("Синтез не удался: %s", exc)
        ctx.say(user, caption + "\n\nОзвучить не получилось, транскрипция выше верна.")
        return

    try:
        file_id = ctx.telegram.send_voice(
            user.chat_id, audio, caption, reply_markup=None if slow else _slow_button(phrase)
        )
    except TelegramError:
        LOGGER.exception("Не удалось отправить голосовое")
        ctx.say(user, caption)
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
    ctx: Context, user: User, phrase: str, cached: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Ищет транскрипцию: кэш → банк лексики → модель."""
    if cached and cached.get("ipa"):
        return {
            "display": cached.get("display") or phrase,
            "ipa": cached.get("ipa"),
            "syllables": cached.get("syllables") or "",
            "note_ru": cached.get("note") or "",
            "contrast": "",
            "say": cached.get("display") or phrase,
        }

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

    llm = ctx.require_llm(user)
    if llm is None:
        return None
    ctx.working(user, f"Слова «{phrase[:40]}» нет в словаре — спрашиваю модель, это до минуты.")
    try:
        data = llm.complete_json(
            PRONUNCIATION_SYSTEM,
            [{"role": "user", "content": phrase}],
            user_id=user.user_id,
            max_tokens=500,
        )
    except LLMError as exc:
        LOGGER.warning("Транскрипция не получена: %s", exc)
        ctx.say(user, "Не смог разобрать это слово. Проверь написание.")
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
