"""Общий HTTP-слой для внешних ИИ-сервисов: только стандартная библиотека."""

from __future__ import annotations

import http.client
import json
import mimetypes
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    """Внешний сервис не ответил или ответил ошибкой."""


def post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 120
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
        raise ProviderError(f"запрос не удался: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"ответ не является JSON: {exc}") from exc


def post_multipart(
    url: str,
    headers: dict[str, str],
    fields: dict[str, str],
    files: dict[str, Path],
    timeout: int = 180,
) -> dict[str, Any]:
    boundary = "----english-bot-" + secrets.token_hex(16)
    parts: list[bytes] = []
    for name, value in fields.items():
        # Явная charset обязательна: без неё текстовая часть трактуется как US-ASCII.
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

    request = urllib.request.Request(
        url,
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
        raise ProviderError(f"запрос не удался: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"ответ не является JSON: {exc}") from exc


def post_binary(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 120
) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as exc:
        raise ProviderError(f"запрос не удался: {exc}") from exc


def extract_json(text: str) -> dict[str, Any] | None:
    """Достаёт JSON-объект из ответа модели, даже если он обёрнут в текст или ```."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
