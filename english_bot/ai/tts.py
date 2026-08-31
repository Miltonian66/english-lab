"""Синтез речи через OpenAI: Ogg Opus, пригодный для Telegram sendVoice."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
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


class Speaker:
    def __init__(self, api_key: str, model: str, voice: str, cache_dir: Path):
        self.api_key = api_key
        self.model = model
        self.voice = voice
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

        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": cleaned,
            "response_format": "opus",
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
