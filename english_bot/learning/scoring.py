"""Оценка письма и речи: вызов модели по рубрике и разбор структурного ответа.

Рубрика четырёхмерная, как в экзаменационных наборах: задача, связность, лексика,
грамматика. Модель обязана вернуть JSON, поэтому результат можно и показать
ученику, и положить в журнал ошибок для интервального повторения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ai.llm import LLM, LLMError
from ..ai.prompts import SPEAKING_SYSTEM, WRITING_SYSTEM, corrections_json_system
from ..ai.stt import Transcript
from ..content.banks import ErrorPattern, SpeakingTask, WritingTask
from ..content.registry import Curriculum


DIMENSION_NAMES_RU = {
    "task": "Задача",
    "coherence": "Связность",
    "fluency": "Беглость",
    "lexis": "Лексика",
    "grammar": "Грамматика",
}


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    category: str
    pattern_id: str
    note: str


@dataclass
class Assessment:
    scores: dict[str, int] = field(default_factory=dict)
    band_ru: str = ""
    strengths_ru: list[str] = field(default_factory=list)
    priorities_ru: list[str] = field(default_factory=list)
    rewrite_en: str = ""
    upgrades_en: list[str] = field(default_factory=list)
    sounds: list[str] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)

    @property
    def average(self) -> float:
        values = [value for value in self.scores.values() if isinstance(value, (int, float))]
        return sum(values) / len(values) if values else 0.0


def _corrections_from(payload: Any) -> list[Correction]:
    rows: list[Correction] = []
    if not isinstance(payload, list):
        return rows
    for item in payload[:10]:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original") or "").strip()
        corrected = str(item.get("corrected") or "").strip()
        if not original or not corrected or original == corrected:
            continue
        rows.append(
            Correction(
                original=original[:300],
                corrected=corrected[:300],
                category=str(item.get("category") or "grammar").strip().lower(),
                pattern_id=str(item.get("pattern_id") or "").strip(),
                note=str(item.get("note") or "").strip()[:300],
            )
        )
    return rows


def _strings(payload: Any, limit: int = 5) -> list[str]:
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload[:limit] if str(item).strip()]


def _scores(payload: Any, keys: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(payload, dict):
        return result
    for key in keys:
        raw = payload.get(key)
        if isinstance(raw, (int, float)):
            result[key] = max(0, min(9, int(round(float(raw)))))
    return result


def assess_writing(llm: LLM, user_id: int, task: WritingTask, text: str, level: str) -> Assessment:
    prompt = (
        f"Уровень ученика: {level}. Задание ({task.level}): {task.prompt_en}\n"
        f"Ожидаемый объём: {task.words_min}–{task.words_max} слов. "
        f"Целевые конструкции: {', '.join(task.focus) or 'не заданы'}.\n\n"
        f"ТЕКСТ УЧЕНИКА ({len(text.split())} слов):\n{text}"
    )
    data = llm.complete_json(
        WRITING_SYSTEM, [{"role": "user", "content": prompt}], user_id=user_id, max_tokens=2000
    )
    return Assessment(
        scores=_scores(data.get("scores"), ("task", "coherence", "lexis", "grammar")),
        band_ru=str(data.get("band_ru") or "").strip(),
        strengths_ru=_strings(data.get("strengths_ru")),
        priorities_ru=_strings(data.get("priorities_ru"), 3),
        rewrite_en=str(data.get("rewrite_en") or "").strip(),
        corrections=_corrections_from(data.get("corrections")),
    )


def assess_speaking(
    llm: LLM, user_id: int, task: SpeakingTask, transcript: Transcript, level: str
) -> Assessment:
    prompt = (
        f"Уровень ученика: {level}. Задание ({task.level}, {task.mode}): {task.prompt_en}\n"
        f"Ожидаемая длительность: {task.seconds_min}–{task.seconds_max} с. "
        f"Целевые конструкции: {', '.join(task.focus) or 'не заданы'}.\n\n"
        f"ИЗМЕРЕНО: длительность {transcript.seconds} с, слов {transcript.words}, "
        f"темп {transcript.wpm} слов/мин, заполнителей речи {transcript.fillers}.\n\n"
        f"РАСШИФРОВКА WHISPER:\n{transcript.text}"
    )
    data = llm.complete_json(
        SPEAKING_SYSTEM, [{"role": "user", "content": prompt}], user_id=user_id, max_tokens=2000
    )
    return Assessment(
        scores=_scores(data.get("scores"), ("task", "fluency", "lexis", "grammar")),
        band_ru=str(data.get("band_ru") or "").strip(),
        strengths_ru=_strings(data.get("strengths_ru")),
        priorities_ru=_strings(data.get("priorities_ru"), 3),
        upgrades_en=_strings(data.get("upgrade_en"), 3),
        sounds=_strings(data.get("sounds"), 3),
        corrections=_corrections_from(data.get("corrections")),
    )


def extract_corrections(
    llm: LLM, user_id: int, text: str, patterns: list[ErrorPattern]
) -> list[Correction]:
    """Структурный разбор реплики для журнала ошибок — отдельно от видимого ответа."""
    try:
        data = llm.complete_json(
            corrections_json_system(patterns),
            [{"role": "user", "content": text}],
            user_id=user_id,
            max_tokens=900,
        )
    except LLMError:
        return []
    return _corrections_from(data.get("corrections"))


# ── отображение ──────────────────────────────────────────────────


def format_assessment(assessment: Assessment, curriculum: Curriculum, title: str) -> str:
    lines = [title]
    if assessment.scores:
        parts = [
            f"{DIMENSION_NAMES_RU.get(key, key)} {value}/9"
            for key, value in assessment.scores.items()
        ]
        lines.append(" · ".join(parts))
    if assessment.band_ru:
        lines.append(assessment.band_ru)

    if assessment.strengths_ru:
        lines.append("")
        lines.append("Сработало:")
        lines.extend(f"• {item}" for item in assessment.strengths_ru[:3])

    if assessment.priorities_ru:
        lines.append("")
        lines.append("Взять в работу:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(assessment.priorities_ru, 1))

    if assessment.corrections:
        lines.append("")
        lines.append("Правки:")
        for correction in assessment.corrections[:6]:
            note = f" — {correction.note}" if correction.note else ""
            lines.append(f'• "{correction.original}" → "{correction.corrected}"{note}')

    if assessment.upgrades_en:
        lines.append("")
        lines.append("Сказать естественнее:")
        lines.extend(f"• {item}" for item in assessment.upgrades_en)

    if assessment.sounds:
        notes = [curriculum.sound(symbol) for symbol in assessment.sounds]
        found = [note for note in notes if note]
        if found:
            lines.append("")
            lines.append("Звуки на отработку (гипотеза по словам, не по звуку):")
            for note in found[:3]:
                pairs = ", ".join(note.minimal_pairs[:2])
                lines.append(f"• /{note.ipa}/ {note.title_ru} — {pairs}")

    if assessment.rewrite_en:
        lines.append("")
        lines.append("Тот же текст на ступень выше:")
        lines.append(assessment.rewrite_en)

    return "\n".join(lines)
