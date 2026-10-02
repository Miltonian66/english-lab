"""Текстовая модель через подписку: запуск `claude -p` подпроцессом.

Второй провайдер «по подписке» рядом с `codex exec`. Разница в том, что у
Claude Code системный промпт передаётся отдельным флагом, а не склеивается с
диалогом: наставнику достаётся ровно тот системный текст, который собрал
`prompts.py`, без личности кодового агента.

Запуск изолирован намеренно: `--safe-mode` выключает пользовательские
настройки, память, скиллы, плагины и хуки, `--no-session-persistence` не
оставляет на диске переписку ученика, а список запрещённых инструментов не даёт
агенту читать файлы и ходить в сеть. Ключ не нужен — работает подписка,
поэтому провайдер не тратит деньги, но отвечает медленнее API.

Вместо подписки владельца запросы можно пустить через совместимый шлюз:
`CLAUDE_BASE_URL` и `CLAUDE_AUTH_TOKEN` уходят в подпроцесс как
`ANTHROPIC_BASE_URL` и `ANTHROPIC_AUTH_TOKEN`. Имена у бота свои, чтобы шлюз
включался только явной настройкой, а не забытой переменной окружения.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time


LOGGER = logging.getLogger(__name__)

# Агент здесь нужен как языковая модель, а не как исполнитель: файлы, команды и
# сеть отключены явно, чтобы промпт ученика не мог увести его в сторону.
DISALLOWED_TOOLS = (
    "Bash,Read,Write,Edit,NotebookEdit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite"
)

# Подпроцессу передаётся только то, без чего он не запустится. Причин две:
# токен Telegram и код приглашения не должны попадать в агента, а ключ API,
# случайно оставшийся в окружении, молча увёл бы запуск с подписки на платный
# контур — и владелец узнал бы об этом из счёта.
ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR",
    "XDG_RUNTIME_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
)


def child_env(base_url: str = "", auth_token: str = "") -> dict[str, str]:
    """Окружение подпроцесса: только разрешённые переменные, без секретов платформы.

    Шлюз добавляется явно и только парой: адрес без токена отправил бы запросы
    шлюзу от имени подписки владельца.
    """
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    if base_url and auth_token:
        env["ANTHROPIC_BASE_URL"] = base_url
        env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    return env


class ClaudeError(RuntimeError):
    pass


def available(binary: str = "claude") -> bool:
    return shutil.which(binary) is not None


def compose_dialogue(messages: list[dict[str, str]]) -> str:
    """Собирает диалог в один промпт: `claude -p` принимает одно сообщение.

    Системная часть сюда не входит — она уходит флагом `--system-prompt`.
    """
    if len(messages) == 1:
        return messages[0].get("content", "").strip()

    lines = ["Диалог с учеником:"]
    for row in messages[:-1]:
        who = "Ученик" if row.get("role") == "user" else "Ты"
        lines.append(f"{who}: {row.get('content', '').strip()}")
    lines.append("")
    lines.append("Последняя реплика ученика, на неё и отвечай:")
    lines.append(messages[-1].get("content", "").strip())
    return "\n".join(lines)


class ClaudeRunner:
    """Обёртка над `claude -p`.

    Работает в пустой временной директории: даже если агент попытается что-то
    прочитать, рядом не окажется ни данных платформы, ни файлов проекта.
    """

    def __init__(
        self,
        binary: str = "claude",
        model: str = "",
        effort: str = "low",
        timeout: int = 180,
        attempts: int = 2,
        base_url: str = "",
        auth_token: str = "",
    ):
        self.binary = binary
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self.base_url = base_url
        self.auth_token = auth_token

    def run(self, system: str, prompt: str) -> str:
        """Запускает агента, при разовом сбое повторяет.

        Причина та же, что у `codex`: подписочный запуск изредка срывается на
        старте, и одного повтора хватает, чтобы ученик этого не заметил.
        """
        last: ClaudeError | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return self._run_once(system, prompt)
            except ClaudeError as exc:
                last = exc
                if attempt < self.attempts:
                    LOGGER.warning("claude сорвался (попытка %d): %s", attempt, exc)
                    time.sleep(1.5)
        assert last is not None
        raise last

    def _run_once(self, system: str, prompt: str) -> str:
        if not available(self.binary):
            raise ClaudeError(
                f"{self.binary} не найден в PATH. Проверь установку Claude Code "
                "или переключись на другой LLM_PROVIDER."
            )

        with tempfile.TemporaryDirectory(prefix="english-lab-claude-") as sandbox:
            command = [
                self.binary,
                "-p",                          # неинтерактивный режим
                "--output-format", "text",
                "--safe-mode",                 # без CLAUDE.md, скиллов, плагинов и хуков
                "--no-session-persistence",    # переписка ученика не ложится на диск
                "--strict-mcp-config",         # никаких сторонних MCP-серверов
                "--disallowedTools", DISALLOWED_TOOLS,
                "--effort", self.effort,
                "--system-prompt", system.strip(),
            ]
            if self.model:
                command += ["--model", self.model]

            try:
                result = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=sandbox,
                    env=child_env(self.base_url, self.auth_token),
                )
            except subprocess.TimeoutExpired as exc:
                raise ClaudeError(f"claude не ответил за {self.timeout} с") from exc
            except OSError as exc:
                raise ClaudeError(f"не удалось запустить claude: {exc}") from exc

        if result.returncode != 0:
            # Причина сбоя всегда в конце вывода, в начале — служебные строки.
            merged = f"{result.stderr or ''}\n{result.stdout or ''}".strip()
            raise ClaudeError(
                f"claude завершился с кодом {result.returncode}: …{merged[-400:]}"
            )

        answer = (result.stdout or "").strip()
        if not answer:
            raise ClaudeError("claude вернул пустой ответ")
        return answer
