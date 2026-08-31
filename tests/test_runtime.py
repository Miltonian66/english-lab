"""Нагрузочные инварианты: порядок одного чата, параллелизм разных и изоляция моделей."""

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from english_bot.runtime import (
    HeavyPools,
    KeyedExecutor,
    OverloadedError,
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
