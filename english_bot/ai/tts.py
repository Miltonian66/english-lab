"""Синтез речи через OpenAI: Ogg Opus, пригодный для Telegram sendVoice."""

from __future__ import annotations

import hashlib
import io
import os
import re
import secrets
import wave
from pathlib import Path

from .http import ProviderError, post_binary


class SpeechError(RuntimeError):
    pass


# По умолчанию платформа учит американскому варианту; инструкция задаёт акцент и темп.
US_INSTRUCTIONS = (
    "Speak in a clear, neutral General American accent. "
    "Use a natural but slightly unhurried pace suitable for a language learner. "
    "Pronounce every sound distinctly, keep rhotic /r/, and do not imitate a British accent."
)
US_SLOW_INSTRUCTIONS = US_INSTRUCTIONS + " Speak slowly and separate the syllables clearly."


def cache_name(text: str, voice: str, slow: bool) -> str:
    """Стабильное имя файла кэша: читаемый префикс плюс хеш от полного запроса."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "audio"
    digest = hashlib.sha256(f"{text}|{voice}|{slow}".encode()).hexdigest()[:12]
    return f"{slug}-{digest}.ogg"


DIALOGUE_LINE = re.compile(r"^([A-Z][A-Za-z]{0,20}):\s*(.+)$")


def split_dialogue(script: str) -> list[tuple[str, str]]:
    """Реплики диалога «Имя: текст» по строкам; пусто, если это не диалог.

    Диалогом считается скрипт, где каждая непустая строка — реплика, а
    говорящих не меньше двух.
    """
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    turns = [DIALOGUE_LINE.match(line) for line in lines]
    if len(lines) < 2 or not all(turns):
        return []
    pairs = [(found.group(1), found.group(2)) for found in turns if found]
    return pairs if len({speaker for speaker, _ in pairs}) >= 2 else []


def spoken_text(script: str) -> str:
    """Текст для одного голоса: без имён говорящих, если скрипт — диалог."""
    turns = split_dialogue(script)
    return " ".join(text for _, text in turns) if turns else script


# Сырые PCM хостингового синтеза: 24 кГц, 16 бит, моно.
PCM_RATE = 24_000
TURN_PAUSE_SECONDS = 0.35


class Speaker:
    def __init__(
        self, api_key: str, model: str, voice: str, cache_dir: Path, second_voice: str = "onyx"
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.second_voice = second_voice
        self.cache_dir = cache_dir

    def synthesize(self, text: str, slow: bool = False) -> Path:
        """Возвращает путь к Ogg Opus. Повторные запросы берутся из кэша на диске."""
        cleaned = text.strip()
        if not cleaned:
            raise SpeechError("нечего озвучивать")
        if len(cleaned) > 900:
            cleaned = cleaned[:900]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self.cache_dir / cache_name(cleaned, self.voice, slow)
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        audio = self._request(cleaned, self.voice, "opus", slow)

        # Кэш общий на всю платформу: имя временного файла обязано быть уникальным,
        # иначе одновременный /say одного слова двумя людьми затрёт файл под ногами.
        partial = destination.with_suffix(f".{os.getpid()}.{secrets.token_hex(4)}.part")
        try:
            partial.write_bytes(audio)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        destination.chmod(0o600)
        return destination

    def _request(self, text: str, voice: str, response_format: str, slow: bool = False) -> bytes:
        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
            "instructions": US_SLOW_INSTRUCTIONS if slow else US_INSTRUCTIONS,
        }
        try:
            audio = post_binary(
                "https://api.openai.com/v1/audio/speech",
                {"Authorization": f"Bearer {self.api_key}"},
                payload,
            )
        except ProviderError as exc:
            raise SpeechError(str(exc)) from exc
        if not audio:
            raise SpeechError("сервис синтеза вернул пустой ответ")
        return audio

    def synthesize_dialogue(self, lines: list[tuple[str, str]]) -> Path:
        """Реплики по голосам: первый говорящий — `voice`, второй — `second_voice`.

        Каждая реплика запрашивается сырым PCM, реплики склеиваются с паузой и
        кодируются в Ogg/Opus тем же PyAV, что и локальный синтез. Без PyAV
        склеить нечем — тогда диалог читается одним голосом, без имён.
        """
        turns = [(speaker, text.strip()[:900]) for speaker, text in lines if text.strip()]
        if not turns:
            raise SpeechError("нечего озвучивать")
        try:
            import av  # noqa: F401  (проверка, что кодер Opus есть)
        except ImportError:
            return self.synthesize(" ".join(text for _, text in turns))

        from .local_speech import pad_wav, wav_to_opus

        order: list[str] = []
        for speaker, _ in turns:
            if speaker not in order:
                order.append(speaker)
        voices = {
            speaker: (self.voice if index % 2 == 0 else self.second_voice)
            for index, speaker in enumerate(order)
        }
        script = "\n".join(f"{speaker}: {text}" for speaker, text in turns)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        destination = self.cache_dir / cache_name(
            script, f"openai-dialogue-{self.voice}-{self.second_voice}", False
        )
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        pause = b"\x00\x00" * int(PCM_RATE * TURN_PAUSE_SECONDS)
        pieces: list[bytes] = []
        for index, (speaker, text) in enumerate(turns):
            if index:
                pieces.append(pause)
            pcm = self._request(text, voices[speaker], "pcm")
            pieces.append(pcm[: len(pcm) - len(pcm) % 2])
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(PCM_RATE)
            target.writeframes(b"".join(pieces))
        audio = wav_to_opus(pad_wav(buffer.getvalue()))

        partial = destination.with_suffix(f".{os.getpid()}.{secrets.token_hex(4)}.part")
        try:
            partial.write_bytes(audio)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        destination.chmod(0o600)
        return destination
