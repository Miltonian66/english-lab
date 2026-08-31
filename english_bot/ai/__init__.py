"""Внешние ИИ-сервисы: текстовая модель, распознавание речи и синтез."""

from .llm import LLM, LLMError
from .stt import Transcriber, TranscriptionError
from .tts import Speaker, SpeechError

__all__ = ["LLM", "LLMError", "Transcriber", "TranscriptionError", "Speaker", "SpeechError"]
