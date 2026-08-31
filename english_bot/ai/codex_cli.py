"""Текстовая модель через подписку: запуск `codex exec` подпроцессом.

Codex — агент для кода, а не HTTP-эндпоинт, поэтому обращение к нему выглядит
иначе, чем к API: процесс, один составной промпт на входе и файл с финальным
ответом на выходе. Всё остальное платформе безразлично — `LLM` предъявляет тот же
интерфейс, что и для OpenAI с Anthropic, и провайдер переключается одной
переменной окружения.

Ограничение, которое здесь не обойти: у Codex нет ни распознавания речи, ни
синтеза. Голос закрывается локальными моделями в `local_speech.py`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


LOGGER = logging.getLogger(__name__)

# Агент не должен ни ходить в сеть, ни трогать файлы: он здесь как языковая модель.
NO_TOOLS = (
    "Отвечай сразу текстом. Не запускай команды, не читай и не создавай файлы, "
    "не ищи в интернете — вся нужная информация уже есть в этом сообщении."
)


class CodexError(RuntimeError):
    pass


def available(binary: str = "codex") -> bool:
    return shutil.which(binary) is not None


def compose_prompt(system: str, messages: list[dict[str, str]]) -> str:
    """Собирает системную часть и диалог в один промпт: `codex exec` принимает один."""
    lines = [system.strip(), "", NO_TOOLS, ""]
    if len(messages) == 1:
        lines.append(messages[0].get("content", "").strip())
        return "\n".join(lines)

    lines.append("Диалог с учеником:")
    for row in messages[:-1]:
        who = "Ученик" if row.get("role") == "user" else "Ты"
        lines.append(f"{who}: {row.get('content', '').strip()}")
    lines.append("")
    lines.append("Последняя реплика ученика, на неё и отвечай:")
    lines.append(messages[-1].get("content", "").strip())
    return "\n".join(lines)


class CodexRunner:
    """Обёртка над `codex exec`.

    Работает в пустой временной директории и в песочнице `read-only`, поэтому
    даже при попытке агента что-то сделать он не дотянется до данных платформы.
    """

    def __init__(
        self,
        binary: str = "codex",
        model: str = "",
        effort: str = "low",
        timeout: int = 180,
        attempts: int = 2,
    ):
        self.binary = binary
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.attempts = max(1, attempts)

    def run(self, prompt: str) -> str:
        """Запускает агента, при разовом сбое повторяет.

        Запуск изредка падает мгновенно и без внятной причины — похоже на
        транзиентную ошибку сессии. Одного повтора хватает, а ученику незачем
        видеть «не удалось» из-за моргнувшего подпроцесса.
        """
        last: CodexError | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return self._run_once(prompt)
            except CodexError as exc:
                last = exc
                if attempt < self.attempts:
                    LOGGER.warning("codex сорвался (попытка %d): %s", attempt, exc)
                    time.sleep(1.5)
        assert last is not None
        raise last

    def _run_once(self, prompt: str) -> str:
        if not available(self.binary):
            raise CodexError(
                f"{self.binary} не найден в PATH. Проверь установку Codex CLI "
                "или переключись на LLM_PROVIDER=openai."
            )

        with tempfile.TemporaryDirectory(prefix="english-lab-codex-") as sandbox:
            answer_path = Path(sandbox) / "answer.txt"
            command = [
                self.binary, "exec",
                "--skip-git-repo-check",
                "--ephemeral",              # не оставлять сессии на диске
                "--color", "never",
                "-s", "read-only",
                "-C", sandbox,              # рабочий корень агента — пустая папка
                "-c", f"model_reasoning_effort={self.effort}",
                "-c", 'web_search="disabled"',
                "--output-last-message", str(answer_path),
                "-",                        # промпт придёт через stdin
            ]
            if self.model:
                command[2:2] = ["-m", self.model]

            try:
                result = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=sandbox,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexError(f"codex не ответил за {self.timeout} с") from exc
            except OSError as exc:
                raise CodexError(f"не удалось запустить codex: {exc}") from exc

            if result.returncode != 0:
                # Причина сбоя всегда в конце вывода, в начале — баннер запуска.
                merged = f"{result.stderr or ''}\n{result.stdout or ''}".strip()
                raise CodexError(
                    f"codex завершился с кодом {result.returncode}: …{merged[-400:]}"
                )

            if answer_path.exists():
                answer = answer_path.read_text(encoding="utf-8").strip()
                if answer:
                    return answer

        # Резервный разбор: если файл пуст, берём хвост stdout.
        fallback = (result.stdout or "").strip()
        if fallback:
            LOGGER.warning("codex не записал файл ответа, разбираю stdout")
            return _tail_after_marker(fallback)
        raise CodexError("codex вернул пустой ответ")


def _tail_after_marker(text: str) -> str:
    """Достаёт содержательную часть из человекочитаемого вывода `codex exec`.

    Ответ идёт после строки `codex` и заканчивается блоком `tokens used`, за
    которым следует само число — обрезать нужно от маркера, а не одну строку.
    """
    marker = "\ncodex\n"
    index = text.rfind(marker)
    if index != -1:
        text = text[index + len(marker) :]
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("tokens used"):
            break
        lines.append(line)
    return "\n".join(lines).strip()
