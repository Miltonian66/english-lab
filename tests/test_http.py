"""Самодельные HTTP-тела: multipart, разбиение сообщений, разбор JSON от модели.

Multipart собран руками на `urllib`, поэтому проверяется тем же способом, каким его
прочитает сервер: стандартным парсером `email` с политикой HTTP.
"""

from __future__ import annotations

import email
import io
import tempfile
import unittest
from email.policy import HTTP
from pathlib import Path

from english_bot.ai.http import extract_json, post_multipart
from english_bot.telegram_api import TelegramAPI, _multipart, _RateLimiter, _split


BINARY = bytes(range(256)) * 40  # содержит \r\n и байты, похожие на границу


def parse(body: bytes, content_type: str) -> dict[str, email.message.Message]:
    message = email.message_from_bytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body,
        policy=HTTP,
    )
    return {
        part.get_param("name", header="content-disposition"): part
        for part in message.iter_parts()  # type: ignore[union-attr]
    }


class TelegramMultipartTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.audio = Path(self._dir.name) / "voice.ogg"
        self.audio.write_bytes(BINARY)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_body_is_well_formed_multipart(self) -> None:
        body, content_type = _multipart({"chat_id": "1"}, {"voice": self.audio})
        parts = parse(body, content_type)
        self.assertEqual(sorted(parts), ["chat_id", "voice"])

    def test_binary_file_survives_byte_for_byte(self) -> None:
        body, content_type = _multipart({"chat_id": "1"}, {"voice": self.audio})
        part = parse(body, content_type)["voice"]
        self.assertEqual(part.get_payload(decode=True), BINARY)
        self.assertEqual(part.get_filename(), "voice.ogg")
        self.assertEqual(part.get_content_type(), "audio/ogg")

    def test_cyrillic_caption_is_not_mangled(self) -> None:
        """Без явной charset текстовая часть трактуется как US-ASCII."""
        caption = "🔊 process — процесс, порядок действий"
        body, content_type = _multipart(
            {"chat_id": "1", "caption": caption}, {"voice": self.audio}
        )
        received = parse(body, content_type)["caption"].get_content().strip()
        self.assertEqual(received, caption)

    def test_document_gets_its_own_mime_type(self) -> None:
        tsv = self.audio.with_name("cards.tsv")
        tsv.write_text("front\tback\ttags\n", encoding="utf-8")
        body, content_type = _multipart({"chat_id": "1"}, {"document": tsv})
        self.assertEqual(parse(body, content_type)["document"].get_filename(), "cards.tsv")


class ProviderMultipartTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.audio = Path(self._dir.name) / "speech.ogg"
        self.audio.write_bytes(BINARY)
        self.captured: dict[str, object] = {}

        import english_bot.ai.http as module

        self._real = module.urllib.request.urlopen

        class Response:
            def __enter__(self_inner):  # noqa: N805
                return self_inner

            def __exit__(self_inner, *args):  # noqa: N805
                return False

            def read(self_inner):  # noqa: N805
                return b'{"text": "ok"}'

        def fake(request, timeout=0):
            self.captured["body"] = request.data
            self.captured["content_type"] = request.headers["Content-type"]
            return Response()

        module.urllib.request.urlopen = fake
        self._module = module

    def tearDown(self) -> None:
        self._module.urllib.request.urlopen = self._real
        self._dir.cleanup()

    def test_whisper_upload_carries_fields_and_audio(self) -> None:
        result = post_multipart(
            "https://api.openai.com/v1/audio/transcriptions",
            {"Authorization": "Bearer x"},
            {"model": "whisper-1", "prompt": "Русскоязычный ученик"},
            {"file": self.audio},
        )
        self.assertEqual(result["text"], "ok")
        parts = parse(self.captured["body"], str(self.captured["content_type"]))  # type: ignore[arg-type]
        self.assertEqual(parts["model"].get_content().strip(), "whisper-1")
        self.assertIn("Русскоязычный", parts["prompt"].get_content())
        self.assertEqual(parts["file"].get_payload(decode=True), BINARY)


class MessageSplitTests(unittest.TestCase):
    def test_short_text_stays_whole(self) -> None:
        self.assertEqual(_split("привет"), ["привет"])

    def test_empty_text_yields_one_empty_chunk(self) -> None:
        self.assertEqual(_split(""), [""])

    def test_long_text_splits_on_line_boundaries(self) -> None:
        text = "\n".join(f"строка {index}" for index in range(900))
        chunks = _split(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 4096)
        self.assertEqual("\n".join(chunks), text)

    def test_single_overlong_line_is_truncated_not_dropped(self) -> None:
        chunks = _split("x" * 9000)
        self.assertTrue(chunks[0])
        self.assertLessEqual(len(chunks[0]), 4096)


class TelegramLoadTests(unittest.TestCase):
    def test_per_chat_limiter_allows_short_burst_then_waits(self) -> None:
        now = 0.0
        sleeps: list[float] = []

        def clock() -> float:
            return now

        def sleep(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        limiter = _RateLimiter(clock=clock, sleeper=sleep)
        for _ in range(3):
            limiter.acquire(10)
        self.assertEqual(sleeps, [])
        limiter.acquire(10)
        self.assertAlmostEqual(sum(sleeps), 1.0, places=3)

    def test_telegram_429_is_retried_once_after_retry_after(self) -> None:
        import english_bot.telegram_api as module

        real_urlopen = module.urllib.request.urlopen
        real_sleep = module.time.sleep
        calls = 0
        sleeps: list[float] = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return b'{"ok": true, "result": {"message_id": 1}}'

        def fake_urlopen(request, timeout=0):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise module.urllib.error.HTTPError(
                    request.full_url,
                    429,
                    "Too Many Requests",
                    {},
                    io.BytesIO(
                        b'{"ok":false,"parameters":{"retry_after":2}}'
                    ),
                )
            return Response()

        module.urllib.request.urlopen = fake_urlopen
        module.time.sleep = sleeps.append
        try:
            api = TelegramAPI("test")
            result = api.call("sendMessage", {"chat_id": 1, "text": "hi"})
        finally:
            module.urllib.request.urlopen = real_urlopen
            module.time.sleep = real_sleep
        self.assertEqual(result, {"message_id": 1})
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [2])


class JsonExtractionTests(unittest.TestCase):
    def test_plain_json_object(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self) -> None:
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_json_wrapped_in_prose(self) -> None:
        self.assertEqual(extract_json('Вот ответ: {"a": 1} — готово'), {"a": 1})

    def test_non_object_and_garbage_return_none(self) -> None:
        self.assertIsNone(extract_json("[1, 2, 3]"))
        self.assertIsNone(extract_json("совсем не json"))
        self.assertIsNone(extract_json("{битый json"))


if __name__ == "__main__":
    unittest.main()
