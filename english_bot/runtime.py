"""Асинхронная диспетчеризация поверх ограниченных пулов потоков.

Telegram long polling не должен ждать ни пользователя, ни модель. Обновления
одного человека выполняются строго по порядку, но разные люди больше не делят
фиксированную дорожку по ``user_id % N``. Тяжёлые контуры изолированы: LLM,
распознавание и синтез имеют собственные пулы и не могут бесконтрольно занять
все ядра или запустить десятки процессов Codex одновременно.

Порядок внутри одной дорожки решал задачу согласованности, но создавал вторую:
минутная расшифровка держала очередь человека, и его собственные кнопки, команды
и даже ``/stop`` ждали её конца — со стороны это выглядело как умерший бот.
Поэтому дорожка теперь короткая и интерактивная, а длинные цепочки уходят в
``JobRunner``: одна фоновая задача на человека, с отменой и живым индикатором
ожидания.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from .ai.llm import LLMError
from .ai.stt import TranscriptionError
from .ai.tts import SpeechError


LOGGER = logging.getLogger(__name__)

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
        # Ключ → метка владеющего им `_drain`. Просто множества мало: `submit`
        # ставит `_drain` в очередь пула ДО того, как тот может отказать, и
        # осиротевший цикл иначе разбирал бы дэку параллельно со своим сменщиком —
        # то есть ломал бы единственную гарантию, ради которой очередь и нужна.
        self._active: dict[int, int] = {}
        self._ticket = 0
        self._pending = 0
        # Отклонённые обновления Telegram считает доставленными, поэтому счётчик —
        # единственный способ узнать, что под нагрузкой что-то пропало.
        self._rejected = 0
        self._closed = False
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    @property
    def rejected(self) -> int:
        with self._lock:
            return self._rejected

    def submit(
        self, key: int, function: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> Future[T]:
        future: Future[T] = Future()
        schedule = False
        with self._lock:
            if self._closed:
                future.set_exception(RuntimeError("диспетчер уже остановлен"))
                return future
            # Обращение к `defaultdict` до проверок оставляло бы пустую дэку на
            # каждый отказ: под флудом это утечка на ровном месте.
            queued = len(self._queues[key]) if key in self._queues else 0
            if self._pending >= self._max_pending or queued >= self._max_pending_per_key:
                self._rejected += 1
                future.set_exception(OverloadedError("очередь обновлений заполнена"))
                return future
            self._queues[key].append(_WorkItem(future, function, args, kwargs))
            self._pending += 1
            if key not in self._active:
                self._ticket += 1
                ticket = self._ticket
                self._active[key] = ticket
                schedule = True
        if schedule:
            try:
                self._executor.submit(self._drain, key, ticket)
            except RuntimeError:
                # Пул закрыли между проверкой и submit: без этого очередь ключа
                # осталась бы навсегда, а future — вечно PENDING.
                self._abandon(key, ticket)
        return future

    def _abandon(self, key: int, ticket: int) -> None:
        """Снимает очередь ключа, которую уже некому разобрать."""
        with self._lock:
            if self._active.get(key) != ticket:
                return
            queue = self._queues.pop(key, deque())
            self._pending -= len(queue)
            del self._active[key]
        for item in queue:
            if not item.future.done():
                item.future.set_exception(RuntimeError("диспетчер уже остановлен"))

    def _drain(self, key: int, ticket: int) -> None:
        while True:
            with self._lock:
                if self._active.get(key) != ticket:
                    return  # ключ уже отдан другому дрейну
                queue = self._queues.get(key)
                if not queue:
                    self._queues.pop(key, None)
                    del self._active[key]
                    return
                item = queue.popleft()
                self._pending -= 1
            if not item.future.set_running_or_notify_cancel():
                continue
            try:
                result = item.function(*item.args, **item.kwargs)
            except BaseException as exc:
                # Future обычно никто не читает, и без записи в журнал отказ
                # исчезал бесследно.
                LOGGER.exception("Обработка обновления не удалась")
                item.future.set_exception(exc)
            else:
                item.future.set_result(result)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


class JobCancelled(RuntimeError):
    """Человек прервал задачу: доводить её до конца и отвечать больше незачем."""


class Job:
    """Одна длинная работа: её имя для человека, отмена и индикатор ожидания.

    Имя пишется так, чтобы подставляться в ответ «сначала закончу …»: не «STT»,
    а «расшифровку голосового».
    """

    __slots__ = ("key", "name", "stage", "started", "began", "reason", "pulse", "_cancelled")

    def __init__(
        self, key: int, name: str, started: float, pulse: "Callable[[], None] | None" = None
    ) -> None:
        self.key = key
        self.name = name
        # Чем задача занята прямо сейчас. Цепочка длинная, и «расшифровку» через
        # минуту после старта человек прочитает как враньё.
        self.stage = name
        # `started` — когда задачу поставили, `began` — когда её взял worker.
        # Бюджет сторожа считается от второго: ожидание свободного потока не
        # должно съедать срок, за который человек ждёт ответа.
        self.started = started
        self.began = 0.0
        # Кто снял задачу: человек, сторож или остановка процесса. В журнале
        # это единственный способ отличить `/stop` от зависшего Whisper.
        self.reason = ""
        # Индикатор «печатает» живёт в Telegram считаные секунды, а задача идёт
        # минуту: без повторов человек снова видит тишину.
        self.pulse = pulse
        self._cancelled = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self, reason: str = "человек") -> None:
        if not self.reason:
            self.reason = reason
        self._cancelled.set()

    def progress(self, stage: str) -> None:
        """Отмечает новый этап: его имя видно в отказе «сначала закончу …»."""
        self.stage = stage

    def checkpoint(self) -> None:
        """Граница между этапами цепочки: дальше идти незачем.

        Прервать сам вызов Whisper или подпроцесс CLI нельзя, поэтому отмена
        срабатывает на ближайшем шве — до отправки ответа и до записи результата.
        """
        if self._cancelled.is_set():
            raise JobCancelled(self.name)


class JobRunner:
    """Фоновые задачи вне пользовательской дорожки: не больше одной на человека.

    Правило «одна задача» заменяет очередь: вторая запись подряд получает
    честный отказ, а не молчаливое место в хвосте, о котором никто не знает.
    Диспетчер при этом свободен — кнопки, экраны и ``/stop`` отвечают сразу.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        timeout: float = 900.0,
        pulse_interval: float = 4.0,
        thread_name_prefix: str = "english-lab-job",
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )
        # Слот человека: то, что видят ограды «одна задача на человека».
        self._jobs: dict[int, Job] = {}
        # Всё, что реально ещё выполняется, включая снятые задачи: слот они уже
        # отдали, но `wait_idle` и остановка обязаны их дождаться.
        self._live: set[Job] = set()
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._closed = False
        self._stopping = threading.Event()
        self._counts = {"started": 0, "done": 0, "cancelled": 0, "failed": 0}
        self._timeout = timeout
        self._pulse_interval = pulse_interval
        self._heartbeat = threading.Thread(
            target=self._beat, name="english-lab-pulse", daemon=True
        )
        self._heartbeat.start()

    # ── запуск и учёт ────────────────────────────────────────────

    def start(
        self,
        key: int,
        name: str,
        function: "Callable[[Job], None]",
        *,
        pulse: "Callable[[], None] | None" = None,
    ) -> Job | None:
        """Ставит задачу. ``None`` — у человека уже есть активная."""
        with self._lock:
            if self._closed or key in self._jobs:
                return None
            job = Job(key, name, time.monotonic(), pulse)
            self._jobs[key] = job
            self._live.add(job)
            self._counts["started"] += 1
            self._idle.clear()
        try:
            self._executor.submit(self._run, job, function)
        except RuntimeError:
            # `ThreadPoolExecutor.submit` кладёт задачу в очередь ДО того, как
            # пытается поднять поток, поэтому отказ «can't start new thread»
            # оставляет её там. Снимаем явно: иначе освободившийся worker
            # подхватит задачу, невидимую ни для `/stop`, ни для сторожа.
            job.cancel()
            self._release(job)
            return None
        return job

    def active(self, key: int) -> Job | None:
        with self._lock:
            return self._jobs.get(key)

    def cancel(self, key: int, reason: str = "человек") -> str:
        """Снимает задачу человека и возвращает её этап; пусто — снимать нечего.

        Слот освобождается сразу, а не по факту остановки: отменённая задача
        может доработать текущий этап ещё минуту, и всё это время человек не
        должен получать «сначала закончу …» за работу, которую сам же прервал.
        """
        with self._lock:
            job = self._jobs.pop(key, None)
        if job is None:
            return ""
        job.cancel(reason)
        return job.stage

    @property
    def running(self) -> int:
        with self._lock:
            return len(self._live)

    def stats(self) -> dict[str, int]:
        """Счётчики за время жизни процесса: сколько задач и чем закончились.

        `running` и `queued` разведены: задача, ждущая свободного потока, ещё
        ничего не делает, и показывать её как работающую — врать.
        """
        with self._lock:
            counts = dict(self._counts)
            counts["running"] = sum(1 for job in self._live if job.began)
            counts["queued"] = sum(1 for job in self._live if not job.began)
            return counts

    def snapshot(self) -> list[tuple[int, str, int]]:
        """Что выполняется прямо сейчас: `(ключ, этап, секунд в работе)`."""
        now = time.monotonic()
        with self._lock:
            live = [job for job in self._live if job.began]
        return sorted(
            ((job.key, job.stage, int(now - job.began)) for job in live),
            key=lambda row: -row[2],
        )

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Ждёт, пока не останется активных задач. Нужно тестам и остановке."""
        return self._idle.wait(timeout)

    def shutdown(self, wait: bool = False) -> None:
        """Останавливает приём и снимает активные задачи.

        Ждать здесь нечего: `claude -p` живёт до 180 секунд, а systemd даёт на
        остановку 30 (`deploy/english-tutor-bot.service`). Задачи отменяются, их
        результат теряется — это честнее, чем SIGKILL посреди записи в базу.
        """
        with self._lock:
            self._closed = True
            jobs = list(self._live)
            self._jobs.clear()
        for job in jobs:
            job.cancel("остановка")
        self._stopping.set()
        # Без `cancel_futures`: снятый future не вызовет `_run`, и задача из
        # очереди пула навсегда осталась бы в `_live` — `wait_idle` не дождался
        # бы её никогда. Отменённая задача и так умрёт на первом же checkpoint.
        self._executor.shutdown(wait=wait)

    # ── внутреннее ───────────────────────────────────────────────

    def _run(self, job: Job, function: "Callable[[Job], None]") -> None:
        job.began = time.monotonic()
        outcome = "done"
        try:
            job.checkpoint()  # могли снять, пока задача стояла в очереди пула
            function(job)
        except JobCancelled:
            outcome = "cancelled"
            LOGGER.info("Задача «%s» прервана человеком", job.name)
        except BaseException:
            outcome = "failed"
            # Обработчик отвечает человеку сам; здесь нужен только след в логе,
            # иначе исключение утонет в Future, которую никто не читает.
            LOGGER.exception("Фоновая задача «%s» упала", job.name)
        finally:
            spent = time.monotonic() - job.began
            if outcome == "done" and job.cancelled:
                # Задача могла успеть доработать после отмены или проглотить её
                # сама. Счётчик обязан считать по факту снятия, а не по тому,
                # долетело ли исключение до этого места.
                outcome = "cancelled"
            with self._lock:
                self._counts[outcome] += 1
            # Длительность видна только здесь: снаружи задача — это молчание.
            LOGGER.info(
                "Задача «%s» (%s) для id=%s: %s%s за %.1f с",
                job.name,
                job.stage,
                job.key,
                outcome,
                f" ({job.reason})" if job.reason else "",
                spent,
            )
            self._release(job)

    def _release(self, job: Job) -> None:
        with self._lock:
            self._live.discard(job)
            if self._jobs.get(job.key) is job:
                del self._jobs[job.key]
            if not self._live:
                self._idle.set()

    def _beat(self) -> None:
        """Один поток на весь процесс: сторож зависших задач и индикатор ожидания.

        Проходы разделены намеренно. Пульс ходит в сеть, и один медленный
        `sendChatAction` откладывал бы снятие всех остальных задач, если бы
        сторож стоял с ним в одном цикле.
        """
        while not self._stopping.wait(self._pulse_interval):
            now = time.monotonic()
            with self._lock:
                live = list(self._live)
            for job in live:
                # Бюджет тратит только та, что реально выполняется: задача в
                # очереди пула ещё ничего не делает, снимать её не за что.
                if job.began and not job.cancelled and now - job.began > self._timeout:
                    # Слот один на человека: зависшая задача заперла бы его
                    # навсегда, и человек остался бы с «сначала закончу …».
                    LOGGER.warning("Задача «%s» идёт слишком долго, снимаю", job.name)
                    self._retire(job)
            for job in live:
                if job.cancelled or job.pulse is None:
                    continue
                try:
                    job.pulse()
                except Exception:
                    LOGGER.debug("Индикатор ожидания не обновился", exc_info=True)

    def _retire(self, job: Job) -> None:
        """Снимает задачу и сразу отдаёт слот, не дожидаясь её остановки."""
        with self._lock:
            if self._jobs.get(job.key) is job:
                del self._jobs[job.key]
        job.cancel("таймаут")


class Telemetry:
    """Счётчики процесса: что бот успел сделать и что при этом ломалось.

    Живут в памяти и обнуляются рестартом — это осознанно. Здесь отвечают на
    вопрос «что происходит прямо сейчас»; история попыток входа и занятий лежит
    в SQLite и рестарт переживает.
    """

    def __init__(self, dispatcher: "KeyedExecutor | None" = None) -> None:
        self.started_at = time.time()
        # Аптайм считается по монотонным часам: перевод времени и поправка NTP
        # иначе давали бы отрицательные или скачущие значения.
        self._since = time.monotonic()
        self._lock = threading.Lock()
        self._counts = {"updates": 0, "errors": 0, "slow": 0}
        self._last_error = ""
        self._last_error_at = 0.0
        self._dispatcher = dispatcher

    def mark(self, name: str) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def note_error(self, where: str, exc: BaseException) -> None:
        """Запоминает последний сбой: тип и место, без текста исключения.

        Сообщение исключения может нести кусок ответа ученика или подписи к
        сообщению, а экран `/admin` видят владелец и админы — туда такому нельзя.
        """
        self.mark("errors")
        with self._lock:
            self._last_error = f"{where}: {type(exc).__name__}"[:80]
            self._last_error_at = time.time()

    @property
    def last_error(self) -> tuple[str, int]:
        """Последний сбой и сколько секунд назад он был; пусто — сбоев не было."""
        with self._lock:
            if not self._last_error:
                return "", 0
            return self._last_error, int(time.time() - self._last_error_at)

    @property
    def uptime_seconds(self) -> int:
        return int(time.monotonic() - self._since)

    @property
    def queue_depth(self) -> int:
        """Сколько обновлений ждёт своей очереди прямо сейчас."""
        return self._dispatcher.pending if self._dispatcher is not None else 0

    @property
    def queue_rejected(self) -> int:
        """Сколько обновлений потеряно из-за переполнения очереди."""
        return self._dispatcher.rejected if self._dispatcher is not None else 0


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

    def synthesize_dialogue(
        self,
        lines: list[tuple[str, str]],
        genders: dict[str, str] | None = None,
        gentle: bool = False,
    ) -> Any:
        if not hasattr(self._delegate, "synthesize_dialogue"):
            return self.synthesize(" ".join(text for _, text in lines), gentle=gentle)
        try:
            return self._pool.run(self._delegate.synthesize_dialogue, lines, genders, gentle=gentle)
        except OverloadedError as exc:
            raise SpeechError("очередь синтеза заполнена") from exc
