"""Распознавание речи через OpenAI Whisper."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .http import ProviderError, post_multipart


class TranscriptionError(RuntimeError):
    pass


FILLERS = ("um", "uh", "er", "erm", "hmm", "like", "you know", "i mean", "sort of", "kind of")


@dataclass(frozen=True)
class Transcript:
    text: str
    words: int
    seconds: int
    fillers: int

    @property
    def wpm(self) -> int:
        """Темп речи в словах в минуту; ниже 100 — медленно, выше 160 — быстро."""
        if self.seconds <= 0:
            return 0
        return round(self.words * 60 / self.seconds)

    @property
    def pace_note_ru(self) -> str:
        rate = self.wpm
        if rate == 0:
            return "темп не измерен"
        if rate < 95:
            return f"{rate} слов/мин — медленно, много пауз"
        if rate <= 165:
            return f"{rate} слов/мин — нормальный разговорный темп"
        return f"{rate} слов/мин — быстро, следи за разборчивостью"


def count_fillers(text: str) -> int:
    lowered = f" {text.lower()} "
    total = 0
    for filler in FILLERS:
        total += len(re.findall(rf"(?<![a-z]){re.escape(filler)}(?![a-z])", lowered))
    return total


class Transcriber:
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model

    def transcribe(self, path: Path, seconds: int, language: str = "en") -> Transcript:
        if not path.exists():
            raise TranscriptionError("файл записи не найден")
        fields = {
            "model": self.model,
            "language": language,
            "response_format": "json",
            # Подсказка не подставляет слова, но выравнивает орфографию и пунктуацию.
            "prompt": "Spoken English practice by a Russian-speaking learner.",
        }
        try:
            data = post_multipart(
                "https://api.openai.com/v1/audio/transcriptions",
                {"Authorization": f"Bearer {self.api_key}"},
                fields,
                {"file": path},
            )
        except ProviderError as exc:
            raise TranscriptionError(str(exc)) from exc

        text = str(data.get("text") or "").strip()
        if not text:
            raise TranscriptionError("Whisper вернул пустую расшифровку")
        words = len([token for token in re.split(r"\s+", text) if token.strip(".,!?;:—-")])
        return Transcript(text=text, words=words, seconds=seconds, fillers=count_fillers(text))
