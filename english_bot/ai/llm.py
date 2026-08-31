"""Текстовая модель за единым интерфейсом.

Три источника: OpenAI Responses, Anthropic Messages и локальный `codex exec`,
работающий по подписке ChatGPT. Провайдер выбирается переменной `LLM_PROVIDER`,
и весь остальной код о разнице не знает — это и есть переключатель.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from .codex_cli import CodexError, CodexRunner, compose_prompt
from .http import ProviderError, extract_json, post_json


LOGGER = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLM:
    """Один интерфейс поверх трёх источников текста.

    `codex` работает по подписке ChatGPT и денег не стоит, но отвечает медленнее;
    `openai` и `anthropic` ходят по HTTP за ключ. Речь ни один из них не закрывает:
    распознавание и синтез живут отдельно, в `stt.py`/`tts.py` или `local_speech.py`.
    """

    def __init__(
        self,
        provider: str,
        api_key: str = "",
        model: str = "",
        codex: CodexRunner | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.codex = codex if codex is not None else (CodexRunner() if provider == "codex" else None)

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        user_id: int = 0,
        max_tokens: int = 1200,
        temperature: float | None = None,
    ) -> str:
        history = [
            {"role": row["role"], "content": row["content"]}
            for row in messages
            if row.get("content")
        ]
        if not history:
            raise LLMError("пустой запрос к модели")
        try:
            if self.provider == "codex":
                return self._codex(system, history)
            if self.provider == "anthropic":
                return self._anthropic(system, history, max_tokens, temperature)
            return self._openai(system, history, user_id, max_tokens, temperature)
        except ProviderError as exc:
            raise LLMError(str(exc)) from exc
        except CodexError as exc:
            raise LLMError(str(exc)) from exc

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        user_id: int = 0,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        """Просит модель вернуть JSON и разбирает его снисходительно к обёрткам."""
        guarded = (
            f"{system}\n\nОТВЕЧАЙ ТОЛЬКО валидным JSON-объектом без пояснений, "
            "без markdown-ограждений и без текста вокруг."
        )
        raw = self.complete(guarded, messages, user_id=user_id, max_tokens=max_tokens)
        parsed = extract_json(raw)
        if parsed is None:
            LOGGER.warning("Модель вернула не-JSON длиной %d символов", len(raw))
            raise LLMError("модель вернула не-JSON")
        return parsed

    # ── провайдеры ───────────────────────────────────────────────

    def _codex(self, system: str, messages: list[dict[str, str]]) -> str:
        if self.codex is None:
            raise LLMError("провайдер codex не инициализирован")
        return self.codex.run(compose_prompt(system, messages))

    def _openai(
        self,
        system: str,
        messages: list[dict[str, str]],
        user_id: int,
        max_tokens: int,
        temperature: float | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": messages,
            "max_output_tokens": max_tokens,
            "store": False,
            "safety_identifier": hashlib.sha256(f"english-bot:{user_id}".encode()).hexdigest()[:32],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        data = post_json(
            "https://api.openai.com/v1/responses",
            {"Authorization": f"Bearer {self.api_key}"},
            payload,
        )
        parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        if not parts:
            raise LLMError("OpenAI не вернул текст")
        return "\n".join(parts).strip()

    def _anthropic(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        data = post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            payload,
        )
        parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            raise LLMError("Anthropic не вернул текст")
        return text
