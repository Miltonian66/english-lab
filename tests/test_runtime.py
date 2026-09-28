"""Нагрузочные инварианты: порядок одного чата, параллелизм разных и изоляция моделей."""

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from english_bot.runtime import (
    HeavyPools,
    JobCancelled,
    JobRunner,
    KeyedExecutor,
    OverloadedError,
    Telemetry,
    ThreadedLLM,
    ThreadedSpeaker,
    ThreadedTranscriber,
)


class KeyedExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = KeyedExecutor(4)

    def tearDown(self) -> None:
        self.executor.shutdown()

    def test_one_user_is_processed_strictly_in_order(self) -> None:
        seen: list[int] = []

        def append(value: int) -> int:
            time.sleep(0.002)
            seen.append(value)
            return value

        futures = [self.executor.submit(7, append, value) for value in range(30)]
        self.assertEqual([future.result() for future in futures], list(range(30)))
        self.assertEqual(seen, list(range(30)))

    def test_slow_user_does_not_block_another_user(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow() -> str:
            started.set()
            release.wait(2)
            return "slow"

        first = self.executor.submit(1, slow)
        self.assertTrue(started.wait(1))
        fast = self.executor.submit(5, lambda: "fast")  # старая схема: 1 % 4 == 5 % 4
        self.assertEqual(fast.result(timeout=0.5), "fast")
        release.set()
        self.assertEqual(first.result(timeout=1), "slow")

    def test_same_user_never_runs_two_updates_at_once(self) -> None:
        active = 0
        maximum = 0
        lock = threading.Lock()

        def probe() -> None:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.005)
            with lock:
                active -= 1

        futures = [self.executor.submit(11, probe) for _ in range(20)]
        for future in futures:
            future.result()
        self.assertEqual(maximum, 1)

    def test_queue_is_bounded_under_single_user_flood(self) -> None:
        executor = KeyedExecutor(1, max_pending=3, max_pending_per_key=1)
        started = threading.Event()
        release = threading.Event()
        first = executor.submit(1, lambda: (started.set(), release.wait(2)))
        self.assertTrue(started.wait(1))
        queued = executor.submit(1, lambda: None)
        rejected = executor.submit(1, lambda: None)
        with self.assertRaises(OverloadedError):
            rejected.result()
        release.set()
        first.result()
        queued.result()
        executor.shutdown()

    def test_rejected_key_does_not_leak_an_empty_queue(self) -> None:
        """Отказ не должен оставлять за собой дэку: под флудом это утечка."""
        executor = KeyedExecutor(1, max_pending=2, max_pending_per_key=2)
        release = threading.Event()
        started = threading.Event()
        busy = executor.submit(1, lambda: (started.set(), release.wait(2)))
        self.assertTrue(started.wait(1))
        queued = [executor.submit(1, lambda: None) for _ in range(2)]
        for key in range(2, 60):
            with self.assertRaises(OverloadedError):
                executor.submit(key, lambda: None).result()
        # Отвергнутые ключи не должны оставить за собой ни одной дэки.
        self.assertEqual(set(executor._queues), {1})
        release.set()
        busy.result()
        for future in queued:
            future.result()
        executor.shutdown()

    def test_submit_after_shutdown_never_leaves_a_pending_future(self) -> None:
        """Гонка с остановкой обязана дать исключение, а не вечный PENDING."""
        executor = KeyedExecutor(1)
        executor._executor.shutdown(wait=True)  # пул закрыт, а диспетчер ещё нет
        future = executor.submit(1, lambda: None)
        self.assertTrue(future.done())
        with self.assertRaises(RuntimeError):
            future.result()
        executor.shutdown(wait=False)


class TelemetryTests(unittest.TestCase):
    """Счётчики процесса: без них состояние очереди видно только с сервера."""

    def test_counts_and_queue_gauges(self) -> None:
        executor = KeyedExecutor(1, max_pending=1, max_pending_per_key=1)
        meter = Telemetry(executor)
        self.assertEqual(meter.queue_depth, 0)
        self.assertEqual(meter.queue_rejected, 0)

        release = threading.Event()
        started = threading.Event()
        busy = executor.submit(1, lambda: (started.set(), release.wait(2)))
        self.assertTrue(started.wait(1))
        executor.submit(1, lambda: None)  # занимает единственное место в очереди
        with self.assertRaises(OverloadedError):
            executor.submit(1, lambda: None).result()
        self.assertEqual(meter.queue_depth, 1)
        self.assertEqual(meter.queue_rejected, 1)

        meter.mark("updates")
        meter.mark("updates")
        meter.mark("errors")
        self.assertEqual(meter.counts()["updates"], 2)
        self.assertEqual(meter.counts()["errors"], 1)
        self.assertGreaterEqual(meter.uptime_seconds, 0)

        release.set()
        busy.result()
        executor.shutdown()


class JobRunnerTests(unittest.TestCase):
    """Длинная работа живёт вне пользовательской дорожки и умеет отменяться."""

    def setUp(self) -> None:
        self.runner = JobRunner(4, timeout=30.0, pulse_interval=0.02)

    def tearDown(self) -> None:
        self.runner.shutdown()

    def test_second_job_of_one_user_is_refused(self) -> None:
        release = threading.Event()
        started = threading.Event()
        second = []

        def slow(job: object) -> None:
            started.set()
            release.wait(2)

        self.assertIsNotNone(self.runner.start(7, "разбор", slow))
        self.assertTrue(started.wait(1))
        self.assertIsNone(self.runner.start(7, "ещё разбор", lambda job: second.append(1)))
        release.set()
        self.assertTrue(self.runner.wait_idle(2))
        self.assertEqual(second, [])

    def test_jobs_of_different_users_run_in_parallel(self) -> None:
        first = threading.Event()
        second = threading.Event()
        release = threading.Event()

        def wait_for(flag: threading.Event):
            def body(job: object) -> None:
                flag.set()
                release.wait(2)

            return body

        self.runner.start(1, "разбор", wait_for(first))
        self.runner.start(2, "разбор", wait_for(second))
        self.assertTrue(first.wait(1))
        self.assertTrue(second.wait(1))
        release.set()
        self.assertTrue(self.runner.wait_idle(2))

    def test_cancel_stops_at_the_next_checkpoint(self) -> None:
        """Отмена кооперативная: текущий этап доработает, следующий не начнётся."""
        reached = threading.Event()
        stages: list[str] = []

        def body(job: object) -> None:
            stages.append("первый")
            reached.set()
            while not job.cancelled:  # type: ignore[attr-defined]
                time.sleep(0.005)
            job.checkpoint()  # type: ignore[attr-defined]
            stages.append("второй")

        self.runner.start(3, "разбор", body)
        self.assertTrue(reached.wait(1))
        self.assertEqual(self.runner.cancel(3), "разбор")
        self.assertTrue(self.runner.wait_idle(2))
        self.assertEqual(stages, ["первый"])

    def test_slot_is_released_after_a_failure(self) -> None:
        """Упавшая задача не должна запирать человека навсегда."""
        def boom(job: object) -> None:
            raise RuntimeError("сломалось")

        self.runner.start(4, "разбор", boom)
        self.assertTrue(self.runner.wait_idle(2))
        self.assertIsNone(self.runner.active(4))
        self.assertIsNotNone(self.runner.start(4, "ещё разбор", lambda job: None))

    def test_watchdog_cancels_an_overlong_job(self) -> None:
        """Зависший Whisper не должен занимать единственный слот человека."""
        runner = JobRunner(1, timeout=0.05, pulse_interval=0.02)
        cancelled = threading.Event()

        def body(job: object) -> None:
            while not job.cancelled:  # type: ignore[attr-defined]
                time.sleep(0.005)
            cancelled.set()

        runner.start(5, "разбор", body)
        self.assertTrue(cancelled.wait(3))
        runner.shutdown()

    def test_pulse_repeats_while_the_job_runs(self) -> None:
        """Индикатор «печатает» живёт секунды, а задача — минуты."""
        beats: list[int] = []
        release = threading.Event()
        started = threading.Event()

        def body(job: object) -> None:
            started.set()
            release.wait(2)

        self.runner.start(6, "разбор", body, pulse=lambda: beats.append(1))
        self.assertTrue(started.wait(1))
        deadline = time.monotonic() + 2
        while len(beats) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        release.set()
        self.assertTrue(self.runner.wait_idle(2))
        self.assertGreaterEqual(len(beats), 2)

    def test_shutdown_does_not_wait_for_a_running_job(self) -> None:
        """systemd даёт на остановку 30 секунд, а `claude -p` живёт до 180."""
        runner = JobRunner(1, pulse_interval=0.02)
        started = threading.Event()
        release = threading.Event()

        def body(job: object) -> None:
            started.set()
            release.wait(3)

        runner.start(8, "разбор", body)
        self.assertTrue(started.wait(1))
        began = time.monotonic()
        runner.shutdown()
        self.assertLess(time.monotonic() - began, 1.0)
        release.set()

    def test_stats_count_every_outcome(self) -> None:
        """Снаружи задача — это молчание: счётчики единственный её след."""
        release = threading.Event()
        started = threading.Event()

        self.runner.start(1, "разбор", lambda job: None)
        self.assertTrue(self.runner.wait_idle(2))
        self.runner.start(2, "разбор", lambda job: (_ for _ in ()).throw(RuntimeError("бум")))
        self.assertTrue(self.runner.wait_idle(2))

        def slow(job: object) -> None:
            started.set()
            release.wait(2)
            job.checkpoint()  # type: ignore[attr-defined]

        self.runner.start(3, "разбор", slow)
        self.assertTrue(started.wait(1))
        self.runner.cancel(3)
        release.set()
        self.assertTrue(self.runner.wait_idle(2))

        stats = self.runner.stats()
        self.assertEqual(stats["started"], 3)
        self.assertEqual(stats["done"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["cancelled"], 1)
        self.assertEqual(stats["running"], 0)

    def test_snapshot_shows_the_current_stage(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow(job: object) -> None:
            job.progress("разбор устного ответа")  # type: ignore[attr-defined]
            started.set()
            release.wait(2)

        self.runner.start(9, "расшифровку голосового", slow)
        self.assertTrue(started.wait(1))
        snapshot = self.runner.snapshot()
        self.assertEqual(len(snapshot), 1)
        key, stage, seconds = snapshot[0]
        self.assertEqual(key, 9)
        self.assertEqual(stage, "разбор устного ответа")
        self.assertGreaterEqual(seconds, 0)
        release.set()
        self.assertTrue(self.runner.wait_idle(2))

    def test_cancelled_job_raises_at_the_checkpoint(self) -> None:
        job_box: list[object] = []

        def body(job: object) -> None:
            job_box.append(job)

        self.runner.start(9, "разбор", body)
        self.assertTrue(self.runner.wait_idle(2))
        job = job_box[0]
        job.cancel()  # type: ignore[attr-defined]
        with self.assertRaises(JobCancelled):
            job.checkpoint()  # type: ignore[attr-defined]


class HeavyPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pools = HeavyPools(llm_workers=1, stt_workers=1, tts_workers=1)

    def tearDown(self) -> None:
        self.pools.shutdown()

    def test_llm_runs_in_its_named_pool(self) -> None:
        class LLM:
            provider = "test"

            def complete(self) -> str:
                return threading.current_thread().name

        name = ThreadedLLM(LLM(), self.pools.llm).complete()
        self.assertTrue(name.startswith("english-lab-llm"), name)

    def test_stt_concurrency_is_capped_at_one(self) -> None:
        active = 0
        maximum = 0
        lock = threading.Lock()

        class Transcriber:
            def transcribe(inner_self) -> str:
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return "ok"

        transcriber = ThreadedTranscriber(Transcriber(), self.pools.stt)
        with ThreadPoolExecutor(max_workers=6) as callers:
            results = [callers.submit(transcriber.transcribe) for _ in range(6)]
            self.assertEqual([result.result() for result in results], ["ok"] * 6)
        self.assertEqual(maximum, 1)

    def test_tts_is_not_stuck_behind_a_slow_llm(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class LLM:
            provider = "test"

            def complete(self) -> str:
                started.set()
                release.wait(2)
                return "done"

        class Speaker:
            def synthesize(self) -> str:
                return "audio"

        llm = ThreadedLLM(LLM(), self.pools.llm)
        speaker = ThreadedSpeaker(Speaker(), self.pools.tts)
        with ThreadPoolExecutor(max_workers=1) as caller:
            slow = caller.submit(llm.complete)
            self.assertTrue(started.wait(1))
            self.assertEqual(speaker.synthesize(), "audio")
            release.set()
            self.assertEqual(slow.result(), "done")


if __name__ == "__main__":
    unittest.main()
