from __future__ import annotations

import html
import http.client
import json
import mimetypes
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MAX_MESSAGE = 3800


class TelegramError(RuntimeError):
    pass


class _RateLimiter:
    """Token bucket: общий лимит Telegram и короткий burst внутри одного чата."""

    def __init__(
        self,
        global_rate: float = 28.0,
        global_burst: float = 28.0,
        chat_rate: float = 1.0,
        chat_burst: float = 3.0,
        *,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        self.global_rate = global_rate
        self.global_burst = global_burst
        self.chat_rate = chat_rate
        self.chat_burst = chat_burst
        self._global_tokens = global_burst
        self._global_updated = clock()
        self._chats: dict[int, tuple[float, float]] = {}
        self._clock = clock
        self._sleep = sleeper
        self._lock = threading.Lock()

    def acquire(self, chat_id: int) -> None:
        while True:
            with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._global_updated)
                global_tokens = min(
                    self.global_burst, self._global_tokens + elapsed * self.global_rate
                )
                chat_tokens, chat_updated = self._chats.get(
                    chat_id, (self.chat_burst, now)
                )
                chat_tokens = min(
                    self.chat_burst,
                    chat_tokens + max(0.0, now - chat_updated) * self.chat_rate,
                )
                if global_tokens >= 1 and chat_tokens >= 1:
                    self._global_tokens = global_tokens - 1
                    self._global_updated = now
                    self._chats[chat_id] = (chat_tokens - 1, now)
                    if len(self._chats) > 2000:
                        cutoff = now - 3600
                        self._chats = {
                            key: value for key, value in self._chats.items()
                            if value[1] >= cutoff
                        }
                    return
                global_wait = max(0.0, (1 - global_tokens) / self.global_rate)
                chat_wait = max(0.0, (1 - chat_tokens) / self.chat_rate)
                delay = max(global_wait, chat_wait, 0.001)
                self._global_tokens = global_tokens
                self._global_updated = now
                self._chats[chat_id] = (chat_tokens, now)
            self._sleep(delay)


def _retry_after(detail: str) -> int:
    """Достаёт Telegram ``parameters.retry_after`` из тела ответа 429."""
    try:
        payload = json.loads(detail)
        value = int((payload.get("parameters") or {}).get("retry_after") or 0)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return 0
    return max(0, value)


def sender_name(sender: dict[str, Any]) -> str:
    """Имя для таблицы отдела из профиля Telegram.

    Спрашивать имя отдельной командой незачем: Telegram присылает его в каждом
    сообщении. Берём как есть, чтобы в таблице человек узнавал себя.
    """
    first = str(sender.get("first_name") or "").strip()
    last = str(sender.get("last_name") or "").strip()
    full = " ".join(part for part in (first, last) if part)
    if full:
        return full[:40]
    username = str(sender.get("username") or "").strip()
    return f"@{username}"[:40] if username else ""


def esc(text: str) -> str:
    """Экранирование для parse_mode=HTML."""
    return html.escape(text, quote=False)


def _multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    """Собирает тело multipart/form-data без внешних зависимостей."""
    boundary = "----english-bot-" + secrets.token_hex(16)
    parts: list[bytes] = []
    for name, value in fields.items():
        # Без явной charset текстовая часть по умолчанию считается US-ASCII,
        # и русская подпись приезжает на сервер искажённой.
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n\r\n"
            f"{value}\r\n".encode()
        )
    for name, path in files.items():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode()
        )
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class TelegramAPI:
    def __init__(self, token: str):
        self.base_url = f"https://api.telegram.org/bot{token}/"
        self.file_base_url = f"https://api.telegram.org/file/bot{token}/"
        self._outbound = _RateLimiter()

    # ── низкий уровень ───────────────────────────────────────────

    def call(self, method: str, payload: dict[str, Any] | None = None, timeout: int = 45) -> Any:
        body = json.dumps(payload or {}).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + method,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request, timeout)

    def call_multipart(
        self, method: str, fields: dict[str, str], files: dict[str, Path], timeout: int = 120
    ) -> Any:
        body, content_type = _multipart(fields, files)
        request = urllib.request.Request(
            self.base_url + method,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        return self._send(request, timeout)

    def _send(self, request: urllib.request.Request, timeout: int) -> Any:
        result: dict[str, Any] = {}
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                finally:
                    exc.close()
                retry_after = _retry_after(detail) if exc.code == 429 else 0
                if attempt == 0 and retry_after:
                    time.sleep(min(60, retry_after))
                    continue
                raise TelegramError(f"Telegram HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                raise TelegramError(f"Telegram request failed: {exc}") from exc
        if not result.get("ok"):
            raise TelegramError(result.get("description", "Unknown Telegram API error"))
        return result.get("result")

    # ── базовые вызовы ───────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        return self.call("getMe")

    def get_updates(self, offset: int | None, timeout: int = 30) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset
        return self.call("getUpdates", payload, timeout=timeout + 15)

    def delete_webhook(self) -> None:
        self.call("deleteWebhook", {"drop_pending_updates": False})

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str | None = None,
    ) -> dict[str, Any] | None:
        chunks = _split(text)
        result: dict[str, Any] | None = None
        for index, chunk in enumerate(chunks):
            self._outbound.acquire(chat_id)
            payload: dict[str, Any] = {"chat_id": chat_id, "text": chunk}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if index == len(chunks) - 1 and reply_markup is not None:
                payload["reply_markup"] = reply_markup
            result = self.call("sendMessage", payload)
        return result

    def edit_message_text(
        self, chat_id: int, message_id: int, text: str,
        reply_markup: dict[str, Any] | None = None, parse_mode: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id, "message_id": message_id, "text": _split(text)[0]
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            self.call("editMessageText", payload)
        except TelegramError as exc:
            if "message is not modified" not in str(exc):
                raise

    def answer_callback(self, callback_id: str, text: str = "", alert: bool = False) -> None:
        payload: dict[str, Any] = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text[:200]
        if alert:
            payload["show_alert"] = True
        try:
            self.call("answerCallbackQuery", payload)
        except TelegramError:
            pass  # просроченный callback — не повод падать

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self._outbound.acquire(chat_id)
        try:
            self.call("sendChatAction", {"chat_id": chat_id, "action": action}, timeout=10)
        except TelegramError:
            pass

    def send_voice(
        self,
        chat_id: int,
        path: Path,
        caption: str = "",
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> str:
        """Отправляет голосовое и возвращает file_id для повторного использования."""
        self._outbound.acquire(chat_id)
        fields = {"chat_id": str(chat_id)}
        if caption:
            fields["caption"] = caption[:1000]
        if parse_mode:
            fields["parse_mode"] = parse_mode
        if reply_markup is not None:
            fields["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
        result = self.call_multipart("sendVoice", fields, {"voice": path})
        return str((result or {}).get("voice", {}).get("file_id", ""))

    def send_voice_by_id(
        self,
        chat_id: int,
        file_id: str,
        caption: str = "",
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        self._outbound.acquire(chat_id)
        payload: dict[str, Any] = {"chat_id": chat_id, "voice": file_id}
        if caption:
            payload["caption"] = caption[:1000]
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        self.call("sendVoice", payload)

    def send_document(self, chat_id: int, path: Path, caption: str = "") -> None:
        self._outbound.acquire(chat_id)
        fields = {"chat_id": str(chat_id)}
        if caption:
            fields["caption"] = caption[:1000]
        self.call_multipart("sendDocument", fields, {"document": path})

    def set_commands(self, commands: list[tuple[str, str]]) -> None:
        self.call(
            "setMyCommands",
            {"commands": [{"command": name, "description": text} for name, text in commands]},
        )

    def download_file(self, file_id: str, destination: Path, max_bytes: int = 20_000_000) -> None:
        file_info = self.call("getFile", {"file_id": file_id})
        file_path = file_info.get("file_path")
        if not file_path:
            raise TelegramError("Telegram getFile returned no file_path")
        reported_size = int(file_info.get("file_size") or 0)
        if reported_size > max_bytes:
            raise TelegramError("Telegram voice file is larger than the configured limit")

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        url = self.file_base_url + urllib.parse.quote(str(file_path), safe="/")
        try:
            with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as target:
                copied = 0
                while chunk := source.read(64 * 1024):
                    copied += len(chunk)
                    if copied > max_bytes:
                        raise TelegramError("Downloaded Telegram voice file exceeded the size limit")
                    target.write(chunk)
            partial.replace(destination)
            destination.chmod(0o600)
        except TelegramError:
            partial.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
            # Обрыв на теле ответа не проходит через _send, поэтому оборачивается тут.
            partial.unlink(missing_ok=True)
            raise TelegramError(f"Не удалось скачать файл: {exc}") from exc
        except Exception:
            partial.unlink(missing_ok=True)
            raise


def _split(text: str) -> list[str]:
    """Режет длинный текст по границам строк, чтобы не рвать разметку посреди слова."""
    if len(text) <= MAX_MESSAGE:
        return [text or ""]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            chunks.append("\n".join(current))
            current, size = [], 0

    for line in text.split("\n"):
        # Строка длиннее лимита нарезается на куски, а не обрезается: иначе хвост
        # длинной расшифровки просто пропал бы.
        pieces = (
            [line]
            if len(line) <= MAX_MESSAGE
            else [line[start : start + MAX_MESSAGE] for start in range(0, len(line), MAX_MESSAGE)]
        )
        for piece in pieces:
            if size + len(piece) + 1 > MAX_MESSAGE:
                flush()
            current.append(piece)
            size += len(piece) + 1
    flush()
    return chunks or [""]


# ── клавиатуры ───────────────────────────────────────────────────

def reply_keyboard(rows: list[list[str]], placeholder: str = "") -> dict[str, Any]:
    """Постоянная клавиатура внизу экрана.

    В отличие от инлайн-кнопок она видна всегда и не привязана к сообщению —
    именно это даёт «одно нажатие до занятия» из любой точки диалога.
    """
    markup: dict[str, Any] = {
        "keyboard": [[{"text": text} for text in row] for row in rows],
        "resize_keyboard": True,
        "is_persistent": True,
    }
    if placeholder:
        markup["input_field_placeholder"] = placeholder[:64]
    return markup


def choice_keyboard(options: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {
        "keyboard": [[{"text": option}] for option in options],
        "resize_keyboard": True,
        "one_time_keyboard": True,
        "input_field_placeholder": "Выбери один вариант",
    }


def clip_callback(data: str) -> str:
    """Telegram считает лимит 64 в БАЙТАХ: кириллица в срезе по символам порвётся."""
    encoded = data.encode("utf-8")
    if len(encoded) <= 64:
        return data
    return encoded[:64].decode("utf-8", errors="ignore")


def inline(rows: list[list[tuple[str, str]]]) -> dict[str, Any]:
    """Инлайн-клавиатура из строк вида [(подпись, callback_data), ...]."""
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": clip_callback(data)} for label, data in row]
            for row in rows
        ]
    }


def inline_grid(
    items: list[tuple[str, str]], columns: int = 2, tail: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    rows = [items[index : index + columns] for index in range(0, len(items), columns)]
    if tail:
        rows.append(tail)
    return inline(rows)
