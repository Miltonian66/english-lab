"""Асинхронная диспетчеризация поверх ограниченных пулов потоков.

Telegram long polling не должен ждать ни пользователя, ни модель. Обновления
одного человека выполняются строго по порядку, но разные люди больше не делят
фиксированную дорожку по ``user_id % N``. Тяжёлые контуры изолированы: LLM,
распознавание и синтез имеют собственные пулы и не могут бесконтрольно занять
все ядра или запустить десятки процессов Codex одновременно.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .ai.llm import LLMError
from .ai.stt import TranscriptionError
from .ai.tts import SpeechError


T = TypeVar("T")


class OverloadedError(RuntimeError):
    """Ограниченная очередь заполнена: лучше отказать, чем съесть всю память."""


@dataclass(frozen=True)
class _WorkItem:
    future: Future[Any]
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class KeyedExecutor:
    """Общий пул с последовательной очередью для каждого ключа.

    Старые фиксированные дорожки создавали head-of-line blocking: длинный Whisper
    одного человека останавливал случайных пользователей той же дорожки. Здесь
    активный пользователь занимает один worker, а любой другой свободный worker
    может обслужить другой ключ. Для одного ключа параллелизма по-прежнему нет —
    состояние диалога и ответы сохраняют порядок Telegram.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        max_pending: int = 1000,
        max_pending_per_key: int = 50,
        thread_name_prefix: str = "english-lab-update",
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )
        self._max_pending = max_pending
        self._max_pending_per_key = max_pending_per_key
        self._queues: dict[int, deque[_WorkItem]] = defaultdict(deque)
        self._active: set[int] = set()
        self._pending = 0
        self._closed = False
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def submit(
        self, key: int, function: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> Future[T]:
        future: Future[T] = Future()
        schedule = False
        with self._lock:
            queue = self._queues[key]
            if self._closed:
                future.set_exception(RuntimeError("диспетчер уже остановлен"))
                return future
            if self._pending >= self._max_pending or len(queue) >= self._max_pending_per_key:
                future.set_exception(OverloadedError("очередь обновлений заполнена"))
                return future
            queue.append(_WorkItem(future, function, args, kwargs))
            self._pending += 1
            if key not in self._active:
                self._active.add(key)
                schedule = True
        if schedule:
            self._executor.submit(self._drain, key)
        return future

    def _drain(self, key: int) -> None:
        while True:
            with self._lock:
                queue = self._queues.get(key)
                if not queue:
                    self._queues.pop(key, None)
                    self._active.discard(key)
                    return
                item = queue.popleft()
                self._pending -= 1
            if not item.future.set_running_or_notify_cancel():
                continue
            try:
                result = item.function(*item.args, **item.kwargs)
            except BaseException as exc:
                item.future.set_exception(exc)
            else:
                item.future.set_result(result)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


class BoundedPool:
    """Пул тяжёлого контура с конечным числом выполняемых и ожидающих задач."""

    def __init__(self, name: str, max_workers: int, max_queued: int = 32) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"english-lab-{name}"
        )
        self._capacity = threading.BoundedSemaphore(max_workers + max_queued)
        self._queued = 0
        self._active = 0
        self._lock = threading.Lock()

    @property
    def queued(self) -> int:
        with self._lock:
            return self._queued

    @property
    def ahead(self) -> int:
        """Сколько задач уже выполняется или ждёт перед новым вызовом."""
        with self._lock:
            return self._active + self._queued

    def run(self, function: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        if not self._capacity.acquire(blocking=False):
            raise OverloadedError("очередь тяжёлых задач заполнена")
        with self._lock:
            self._queued += 1

        def execute() -> T:
            with self._lock:
                self._queued -= 1
                self._active += 1
            try:
                return function(*args, **kwargs)
            finally:
                with self._lock:
                    self._active -= 1

        try:
            return self._executor.submit(execute).result()
        finally:
            self._capacity.release()

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


class HeavyPools:
    """Независимые лимиты для внешнего текста, CPU-STT и CPU/API-TTS."""

    def __init__(self, llm_workers: int, stt_workers: int, tts_workers: int) -> None:
        self.llm = BoundedPool("llm", llm_workers)
        self.stt = BoundedPool("stt", stt_workers)
        self.tts = BoundedPool("tts", tts_workers)

    def shutdown(self, wait: bool = True) -> None:
        self.llm.shutdown(wait)
        self.stt.shutdown(wait)
        self.tts.shutdown(wait)


class ThreadedLLM:
    def __init__(self, delegate: Any, pool: BoundedPool) -> None:
        self._delegate = delegate
        self._pool = pool
        self.provider = delegate.provider

    @property
    def queue_ahead(self) -> int:
        return self._pool.ahead

    def complete(self, *args: Any, **kwargs: Any) -> str:
        try:
            return self._pool.run(self._delegate.complete, *args, **kwargs)
        except OverloadedError as exc:
            raise LLMError("очередь текстовой модели заполнена") from exc

    def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return self._pool.run(self._delegate.complete_json, *args, **kwargs)
        except OverloadedError as exc:
            raise LLMError("очередь текстовой модели заполнена") from exc


class ThreadedTranscriber:
    def __init__(self, delegate: Any, pool: BoundedPool) -> None:
        self._delegate = delegate
        self._pool = pool

    @property
    def queue_ahead(self) -> int:
        return self._pool.ahead

    def transcribe(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._pool.run(self._delegate.transcribe, *args, **kwargs)
        except OverloadedError as exc:
            raise TranscriptionError("очередь распознавания заполнена") from exc


class ThreadedSpeaker:
    def __init__(self, delegate: Any, pool: BoundedPool) -> None:
        self._delegate = delegate
        self._pool = pool

    @property
    def queue_ahead(self) -> int:
        return self._pool.ahead

    def synthesize(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._pool.run(self._delegate.synthesize, *args, **kwargs)
        except OverloadedError as exc:
            raise SpeechError("очередь синтеза заполнена") from exc
