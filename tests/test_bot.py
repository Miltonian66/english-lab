"""Сквозные сценарии бота на поддельных Telegram и ИИ: доступ, диагностика, тренировка."""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from english_bot.ai.llm import LLMError
from english_bot.ai.stt import Transcript
from english_bot.app import EnglishLabBot
from english_bot.config import Settings
from english_bot.handlers import menu
from english_bot.learning import practice as pr
from english_bot.runtime import JobRunner


CLAIM_CODE = "claim-code-for-tests"


class FakeTelegram:
    """Заменяет сетевой клиент: копит отправленное и раздаёт предсказуемые ответы."""

    def __init__(self, token: str = "test"):
        self.sent: list[tuple[int, str]] = []
        self.markups: list[dict[str, Any] | None] = []
        self.voices: list[tuple[int, Path, str]] = []
        self.documents: list[tuple[int, Path]] = []
        self.answered: list[str] = []
        self.edits: list[tuple[int, int, str]] = []
        self.actions: list[tuple[int, str]] = []

    def get_me(self) -> dict[str, Any]:
        return {"username": "english_lab_test_bot", "id": 1}

    def delete_webhook(self) -> None:
        return None

    def set_commands(self, commands: list[tuple[str, str]]) -> None:
        return None

    def send_message(
        self, chat_id: int, text: str, reply_markup: Any = None, parse_mode: Any = None
    ) -> dict[str, Any]:
        self.sent.append((chat_id, text))
        self.markups.append(reply_markup)
        return {"message_id": len(self.sent)}

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: Any = None,
        parse_mode: Any = None,
    ) -> None:
        """Правка экрана — тоже сообщение для человека: тесты видят её так же."""
        self.edits.append((chat_id, message_id, text))
        self.sent.append((chat_id, text))
        self.markups.append(reply_markup)

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append((chat_id, action))

    def answer_callback(self, callback_id: str, text: str = "", alert: bool = False) -> None:
        self.answered.append(text)

    def send_voice(
        self,
        chat_id: int,
        path: Path,
        caption: str = "",
        parse_mode: Any = None,
        reply_markup: Any = None,
    ) -> str:
        self.voices.append((chat_id, path, caption))
        self.markups.append(reply_markup)
        return "fake-file-id"

    def send_voice_by_id(
        self,
        chat_id: int,
        file_id: str,
        caption: str = "",
        parse_mode: Any = None,
        reply_markup: Any = None,
    ) -> None:
        self.voices.append((chat_id, Path(file_id), caption))
        self.markups.append(reply_markup)

    def send_document(self, chat_id: int, path: Path, caption: str = "") -> None:
        self.documents.append((chat_id, path))

    def download_file(self, file_id: str, destination: Path, max_bytes: int = 20_000_000) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-ogg")

    # ── помощники теста ──────────────────────────────────────────

    def last(self) -> str:
        return self.sent[-1][1] if self.sent else ""

    def all_text(self) -> str:
        return "\n".join(text for _, text in self.sent)

    def buttons(self) -> list[str]:
        for markup in reversed(self.markups):
            if markup and "inline_keyboard" in markup:
                return [
                    button["callback_data"]
                    for row in markup["inline_keyboard"]
                    for button in row
                ]
        return []

    def all_buttons(self) -> list[str]:
        """Все callback_data из всех отправленных клавиатур, а не только последней."""
        found: list[str] = []
        for markup in self.markups:
            if markup and "inline_keyboard" in markup:
                found.extend(
                    button["callback_data"]
                    for row in markup["inline_keyboard"]
                    for button in row
                )
        return found

    def reset(self) -> None:
        self.sent.clear()
        self.markups.clear()
        self.answered.clear()
        self.edits.clear()


class BotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        # Собираем настройки тем же путём, что и боевой запуск: через окружение.
        # Иначе каждое новое поле конфига ломает подготовку тестов.
        self._env = {
            "TELEGRAM_BOT_TOKEN": "test",
            "BOT_CLAIM_CODE": CLAIM_CODE,
            "DATABASE_PATH": str(root / "db.sqlite3"),
            "EXPORT_DIR": str(root / "exports"),
            "VOICE_DIR": str(root / "voices"),
            "AUDIO_CACHE_DIR": str(root / "audio"),
            "MODELS_DIR": str(root / "models"),
            "LOG_LEVEL": "ERROR",
            "LLM_PROVIDER": "openai",
            "SPEECH_BACKEND": "openai",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "WORKERS": "1",
            # Инлайн-режим: длинные цепочки выполняются в вызывающем потоке.
            # Сквозные сценарии так остаются пошаговыми и не зависят от гонок;
            # саму асинхронность проверяет `BackgroundJobTests` с живым раннером.
            "JOB_WORKERS": "0",
            "DAILY_AI_CALLS": "50",
            "TEAM_OPEN_REGISTRATION": "0",
        }
        self._saved = {key: os.environ.get(key) for key in self._env}
        os.environ.update(self._env)
        self.settings = Settings.from_env()
        self.bot = EnglishLabBot(self.settings)
        self.telegram = FakeTelegram()
        self.bot.telegram = self.telegram  # type: ignore[assignment]
        self.bot.initialize()

    def tearDown(self) -> None:
        self.bot.close(wait=False)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._dir.cleanup()

    # ── помощники ────────────────────────────────────────────────

    def sender(self, user_id: int, **profile: str) -> dict[str, object]:
        """Профиль в том же виде, в каком его присылает Telegram."""
        return {"id": user_id, "is_bot": False, **profile}

    def send_guarded(self, user_id: int, text: str) -> None:
        """Как в бою: через `_safe_handle`, чтобы проверить страховку от исключений.

        Обычный `send` зовёт `handle_update` напрямую — иначе страховка глотала бы
        настоящие ошибки в тестах и они переставали бы падать.
        """
        self.bot._safe_handle(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "chat": {"id": user_id, "type": "private"},
                    "from": self.sender(user_id),
                    "text": text,
                },
            }
        )

    def send_group(self, user_id: int, text: str, chat_id: int = -100500) -> None:
        self.bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "chat": {"id": chat_id, "type": "supergroup"},
                    "from": self.sender(user_id),
                    "text": text,
                },
            }
        )

    def send(self, user_id: int, text: str, **profile: str) -> None:
        self.bot.handle_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": len(self.telegram.sent) + 1,
                    "chat": {"id": user_id, "type": "private"},
                    "from": self.sender(user_id, **profile),
                    "text": text,
                },
            }
        )

    def press(self, user_id: int, data: str, **profile: str) -> None:
        self.bot.handle_update(
            {
                "update_id": 1,
                "callback_query": {
                    "id": "cb",
                    "data": data,
                    "from": self.sender(user_id, **profile),
                    "message": {"message_id": 1, "chat": {"id": user_id, "type": "private"}},
                },
            }
        )

    def current_question(self, user_id: int):
        user = self.bot.storage.user(user_id)
        assert user is not None
        state = pr.PracticeState.from_dict(user.state_data)
        ref = state.current_ref()
        assert ref is not None
        question = pr.resolve(ref, self.bot.curriculum, self.bot.context().rng)
        assert question is not None
        return question

    def step(self, user_id: int) -> int:
        user = self.bot.storage.user(user_id)
        assert user is not None
        return int(user.state_data.get("index", 0))

    def session_tag(self, user_id: int) -> int:
        """Метка сессии в callback_data: кнопка принадлежит своему занятию."""
        user = self.bot.storage.user(user_id)
        assert user is not None
        return int(user.state_data.get("session_id") or 0) % 1000

    def step_payload(self, user_id: int) -> str:
        return f"{self.session_tag(user_id)}:{self.step(user_id)}"

    def answer_current(self, user_id: int, correctly: bool = True) -> None:
        """Отвечает на текущее задание тем способом, который оно принимает."""
        question = self.current_question(user_id)
        expected = question.expected[0] if question.expected else ""
        if question.is_choice:
            index = next(
                (i for i, option in enumerate(question.options) if option == expected), 0
            )
            if not correctly:
                index = (index + 1) % len(question.options)
            self.press(user_id, f"an:{self.step_payload(user_id)}:{index}")
        elif question.kind == "cloze" and not correctly:
            # Неверный ответ — по слову на каждый пропуск: иначе бот не поймёт
            # ответ и попросит нужное число, а не засчитает ошибку.
            self.send(user_id, "; ".join("definitely wrong" for _ in question.gaps))
        else:
            self.send(user_id, expected if correctly else "definitely wrong answer")

    def answer_placement(self, user_id: int, choice: str = "0") -> None:
        user = self.bot.storage.user(user_id)
        assert user is not None
        data = user.state_data
        tag = int(data.get("session_id") or 0) % 1000
        if int(data.get("profile_index", 0)) < 3:
            self.press(user_id, f"pf:{tag}:{data.get('profile_index', 0)}:{choice}")
        else:
            self.press(user_id, f"pa:{tag}:{len(data.get('asked') or [])}:{choice}")

    def claim_owner(self, user_id: int = 100) -> None:
        self.send(user_id, f"/start {CLAIM_CODE}")
        self.telegram.reset()

    # ── подставные провайдеры ────────────────────────────────────

    def enable_llm(self, reply: str = "Ответ наставника") -> Any:
        """Включает текстовый контур: без него письмо и диалог теперь не выдаются."""

        class Stub:
            provider = "openai"

            def __init__(self) -> None:
                self.calls: list[tuple[str, list[dict[str, str]]]] = []
                self.reply = reply

            def complete(self, system: str, messages: list[dict[str, str]], **kwargs: object) -> str:
                self.calls.append((system, messages))
                return self.reply

            def complete_json(self, *args: object, **kwargs: object) -> dict[str, Any]:
                return {}

        stub = Stub()
        self.bot.llm = stub  # type: ignore[assignment]
        return stub

    def enable_speech(self) -> None:
        """Включает распознавание и синтез: устная практика проверяет их заранее."""

        class Transcriber:
            def transcribe(self, *args: object, **kwargs: object) -> Any:
                raise AssertionError("в тесте расшифровка не вызывается")

        class Speaker:
            def synthesize(self, *args: object, **kwargs: object) -> Any:
                raise AssertionError("в тесте синтез не вызывается")

        self.bot.transcriber = Transcriber()  # type: ignore[assignment]
        self.bot.speaker = Speaker()  # type: ignore[assignment]


class AccessTests(BotTestCase):
    def test_first_user_claims_ownership_with_the_code(self) -> None:
        self.send(100, f"/start {CLAIM_CODE}")
        owner = self.bot.storage.user(100)
        assert owner is not None
        self.assertEqual(owner.role, "owner")
        self.assertIn("English Lab", self.telegram.all_text())

    def test_wrong_code_does_not_create_a_user(self) -> None:
        self.send(100, "/start неверный-код")
        self.assertIsNone(self.bot.storage.user(100))
        self.assertIn("не привязан", self.telegram.all_text())

    def test_stranger_is_refused_after_owner_exists(self) -> None:
        self.claim_owner()
        self.send(200, "/start")
        self.assertIsNone(self.bot.storage.user(200))
        self.assertIn("по приглашению", self.telegram.all_text())

    def test_colleague_joins_with_an_invite(self) -> None:
        self.claim_owner()
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.telegram.reset()

        self.send(200, f"/start {code}")
        member = self.bot.storage.user(200)
        assert member is not None
        self.assertEqual(member.role, "member")

        # Код одноразовый: третий человек по нему не пройдёт.
        self.send(300, f"/start {code}")
        self.assertIsNone(self.bot.storage.user(300))

    def test_invite_link_pasted_as_text_still_works(self) -> None:
        """Открытый диалог не даёт Telegram подставить код: ссылку вставляют руками."""
        self.claim_owner()
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.telegram.reset()

        self.send(200, "/start")  # диалог уже начат, дальше deep link молчит
        self.assertIsNone(self.bot.storage.user(200))
        self.send(200, f"https://t.me/english_lab_test_bot?start={code}")
        member = self.bot.storage.user(200)
        assert member is not None
        self.assertEqual(member.role, "member")

    def test_bare_invite_code_is_accepted(self) -> None:
        self.claim_owner()
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.telegram.reset()

        self.send(200, code)
        member = self.bot.storage.user(200)
        assert member is not None
        self.assertEqual(member.role, "member")

    def test_used_code_pasted_as_a_link_explains_itself(self) -> None:
        """Отказ обязан назвать причину: раньше вставленная ссылка молчала про неё."""
        self.claim_owner()
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.send(200, f"/start {code}")
        self.telegram.reset()

        self.send(300, f"https://t.me/english_lab_test_bot?start={code}")
        self.assertIsNone(self.bot.storage.user(300))
        self.assertIn("уже воспользовались", self.telegram.all_text())

    def test_refusal_tells_how_to_send_the_code_by_hand(self) -> None:
        self.claim_owner()
        self.send(200, "/start")
        self.assertIn("/start КОД", self.telegram.all_text())

    def test_ordinary_text_from_a_stranger_is_not_taken_for_a_code(self) -> None:
        self.claim_owner()
        self.send(200, "Здравствуйте, а что это за бот")
        self.assertIsNone(self.bot.storage.user(200))
        self.assertIn("по приглашению", self.telegram.all_text())

    def test_refused_attempt_is_recorded_with_the_telegram_id(self) -> None:
        """Отказ не оставлял следа, и выдать доступ вручную было нечему."""
        self.claim_owner()
        self.send(200, "/start", first_name="Гость")
        rows = self.bot.storage.access_attempts()
        self.assertEqual(rows[0]["user_id"], 200)
        self.assertEqual(rows[0]["outcome"], "need_invite")
        self.assertEqual(rows[0]["display_name"], "Гость")

    def test_every_refusal_reason_is_distinguishable(self) -> None:
        self.claim_owner()
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.send(200, f"/start {code}")
        self.send(300, f"/start {code}")  # код уже сгорел
        self.send(400, "/start несуществующий")
        outcomes = [row["outcome"] for row in self.bot.storage.access_attempts()]
        self.assertIn("invite_used", outcomes)
        self.assertIn("invite_missing", outcomes)
        self.assertIn("joined", outcomes)

    def test_one_attempt_leaves_one_record(self) -> None:
        """Неверный код — это одна попытка, а не две: журнал не должен шуметь."""
        self.claim_owner()
        self.send(200, "/start abcdefghij")
        rows = self.bot.storage.access_attempts()
        mine = [row for row in rows if row["user_id"] == 200]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]["outcome"], "invite_missing")
        self.assertEqual(mine[0]["detail"], "abcdefghij")

    def test_free_text_after_start_is_not_written_to_the_log(self) -> None:
        """`/start` принимает любой хвост, а журнал видят владелец и админы."""
        self.claim_owner()
        self.send(200, "/start мой пароль от почты hunter2 и телефон")
        row = self.bot.storage.access_attempts()[0]
        self.assertEqual(row["outcome"], "invite_missing")
        self.assertEqual(row["detail"], "не похоже на код")

    def test_link_pasted_after_the_start_command_works(self) -> None:
        """Меню команд Telegram подставляет «/start », и ссылку вставляют за ним."""
        self.claim_owner()
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.telegram.reset()

        self.send(200, f"/start https://t.me/english_lab_test_bot?start={code}")
        member = self.bot.storage.user(200)
        assert member is not None
        self.assertEqual(member.role, "member")

    def test_code_with_a_trailing_word_still_works(self) -> None:
        self.claim_owner()
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.telegram.reset()

        self.send(200, f"/start {code} спасибо")
        self.assertIsNotNone(self.bot.storage.user(200))

    def test_wrong_code_gets_a_precise_refusal(self) -> None:
        """Общий отказ здесь врёт: приглашение у человека есть, дело в самом коде."""
        self.claim_owner()
        self.send(200, "/start abcdefghij")
        self.assertIn("Такого кода приглашения нет", self.telegram.all_text())

    def test_stranger_without_a_code_gets_the_general_refusal(self) -> None:
        self.claim_owner()
        self.send(200, "Hellothere")
        self.assertIn("по приглашению", self.telegram.all_text())
        self.assertNotIn("Такого кода приглашения нет", self.telegram.all_text())

    def test_admin_screen_shows_who_was_refused(self) -> None:
        self.claim_owner()
        self.send(200, "/start", first_name="Гость")
        self.telegram.reset()
        self.send(100, "/admin")
        report = self.telegram.all_text()
        self.assertIn("Последние попытки входа", report)
        self.assertIn("200", report)
        self.assertIn("нет кода приглашения", report)

    def test_admin_screen_shows_process_counters(self) -> None:
        self.claim_owner()
        self.send(100, "/admin")
        report = self.telegram.all_text()
        self.assertIn("Наблюдаемость", report)
        self.assertIn("Обновлений:", report)
        self.assertIn("Очередь обновлений:", report)

    def test_counters_reflect_real_traffic(self) -> None:
        """Боевая проводка счётчиков: экран должен показывать факт, а не ноль."""
        self.claim_owner()
        before = self.bot._telemetry.counts()["updates"]
        self.send(100, "/privacy")
        self.send(100, "/privacy")
        self.assertEqual(self.bot._telemetry.counts()["updates"], before + 2)
        self.telegram.reset()
        self.send(100, "/admin")
        self.assertIn(f"Обновлений: {before + 3}", self.telegram.all_text())

    def test_admin_screen_summarises_access_outcomes(self) -> None:
        self.claim_owner()
        self.send(200, "/start")
        self.send(300, "/start")
        self.telegram.reset()
        self.send(100, "/admin")
        report = self.telegram.all_text()
        self.assertIn("Входы за 7 дней", report)
        self.assertIn("нет кода приглашения — 2 · 2", report)

    def test_uptime_is_rendered_in_human_units(self) -> None:
        from english_bot.handlers.core import _duration_ru

        self.assertEqual(_duration_ru(0), "0 с")
        self.assertEqual(_duration_ru(59), "59 с")
        self.assertEqual(_duration_ru(60), "1 мин")
        self.assertEqual(_duration_ru(3599), "59 мин")
        self.assertEqual(_duration_ru(3660), "1 ч 1 мин")
        self.assertEqual(_duration_ru(90000), "1 сут 1 ч")

    def test_admin_screen_shows_the_last_failure_without_leaking_text(self) -> None:
        """Сообщение исключения может нести текст ученика — на экран идёт только тип."""
        self.claim_owner()

        class Boom:
            provider = "openai"

            def complete(self, *args: object, **kwargs: object) -> str:
                raise RuntimeError("секретный текст ученика")

            def complete_json(self, *args: object, **kwargs: object) -> dict[str, Any]:
                raise RuntimeError("секретный текст ученика")

        self.bot.llm = Boom()  # type: ignore[assignment]
        self.send(100, "Hello there")
        self.telegram.reset()
        self.send(100, "/admin")
        report = self.telegram.all_text()
        self.assertIn("Последний сбой", report)
        self.assertIn("RuntimeError", report)
        self.assertNotIn("секретный текст ученика", report)

    def test_only_admins_create_invites(self) -> None:
        self.claim_owner()
        self.bot.storage.create_user(200, 200, role="member")
        self.press(200, "invitenew")
        self.assertIn("владелец или админ", self.telegram.all_text())

    def test_callback_from_unknown_user_is_refused(self) -> None:
        self.claim_owner()
        self.press(999, "lvls")
        self.assertIn("Нужно приглашение", self.telegram.answered)


class PlacementFlowTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def test_placement_runs_and_assigns_a_level(self) -> None:
        self.press(100, "retest")
        guard = 0
        while True:
            user = self.bot.storage.user(100)
            assert user is not None
            if user.state != "placement":
                break
            guard += 1
            self.assertLess(guard, 60, "диагностика не сходится")
            self.answer_placement(100, "2" if guard <= 3 else "0")

        user = self.bot.storage.user(100)
        assert user is not None
        self.assertIn(user.level, {"A1", "A2", "B1", "B2", "C1", "C2"})
        self.assertTrue(user.target_level)
        self.assertIn("Диагностика закончена", self.telegram.all_text())

    def test_placement_answers_are_recorded(self) -> None:
        self.press(100, "retest")
        for _ in range(4):
            self.answer_placement(100, "1")
        rows = self.bot.storage.placement_answers(100)
        self.assertGreaterEqual(len(rows), 4)

    def test_text_during_placement_is_redirected_to_buttons(self) -> None:
        self.press(100, "retest")
        self.telegram.reset()
        self.send(100, "просто текст")
        self.assertIn("кнопкой", self.telegram.last())


class PracticeFlowTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")

    def test_practice_session_creates_cards_and_finishes(self) -> None:
        self.press(100, "startpractice")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")
        total = len(user.state_data["queue"])

        for _ in range(total + 2):
            current = self.bot.storage.user(100)
            assert current is not None
            if current.state != "practice":
                break
            self.answer_current(100)

        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")
        self.assertGreater(self.bot.storage.attempts_count(100)[0], 0)
        self.assertGreater(self.bot.storage.card_counts(100).get("point", (0, 0))[0], 0)
        self.assertIn("Готово:", self.telegram.all_text())

    def test_correct_answer_is_scored_and_explained(self) -> None:
        self.press(100, "startpractice")
        self.telegram.reset()
        self.answer_current(100, correctly=True)
        self.assertIn("Верно", self.telegram.all_text())
        self.assertEqual(self.bot.storage.attempts_count(100), (1, 1))

    def test_wrong_answer_shows_the_key(self) -> None:
        self.press(100, "startpractice")
        self.telegram.reset()
        self.answer_current(100, correctly=False)
        self.assertIn("Мимо", self.telegram.all_text())
        self.assertEqual(self.bot.storage.attempts_count(100), (1, 0))

    def test_stale_answer_button_on_a_free_input_task_is_not_silent(self) -> None:
        """Нажатие кнопки варианта на задании со свободным вводом должно что-то сказать."""
        self.press(100, "startpractice")
        user = self.bot.storage.user(100)
        assert user is not None
        question = self.current_question(100)
        if question.is_choice:  # подберём свободный ввод дальше по очереди
            for index, ref in enumerate(user.state_data["queue"]):
                probe = pr.resolve(ref, self.bot.curriculum, self.bot.context().rng)
                if probe is not None and not probe.is_choice:
                    data = dict(user.state_data)
                    data["index"] = index
                    self.bot.storage.set_state(100, "practice", data)
                    break
            else:
                self.skipTest("в очереди нет заданий со свободным вводом")
        self.telegram.reset()
        self.press(100, f"an:{self.step_payload(100)}:0")
        self.assertIn("свободный ответ", " ".join(self.telegram.answered))

    def test_hint_shows_the_rule_and_lowers_the_grade(self) -> None:
        """Подсказка обязана объяснять тему, а не сдавать ответ."""
        self.press(100, "startpractice")
        question = self.current_question(100)
        self.telegram.reset()
        self.press(100, f"hint:{self.step_payload(100)}")

        user = self.bot.storage.user(100)
        assert user is not None
        self.assertTrue(user.state_data["helped"])
        text = self.telegram.all_text()
        self.assertIn("💡", text)
        if question.point_id:  # грамматика: полный разбор правила
            point = self.bot.curriculum.point(question.point_id)
            assert point is not None
            self.assertIn("Как строится:", text)
            self.assertIn(point.summary_ru[:60], text)
            self.assertGreater(len(text), 300, "разбор должен быть разбором, а не строкой")
        else:  # лексика: употребление и сочетаемость
            self.assertIn("В предложении:", text)

    def test_hint_does_not_narrow_the_options(self) -> None:
        """Старое «точно не A, C» ничему не учило — его быть не должно."""
        self.press(100, "startpractice")
        self.telegram.reset()
        self.press(100, f"hint:{self.step_payload(100)}")
        text = self.telegram.all_text()
        self.assertNotIn("точно не", text)
        self.assertNotIn("ответ начинается", text)

    def test_cloze_text_is_answered_gap_by_gap(self) -> None:
        """Связный текст: пропуски пронумерованы, ответы — по порядку, разбор — по номерам."""
        point = next(
            point for point in self.bot.curriculum.points.values()
            if any(exercise.kind == "cloze" for exercise in point.exercises)
        )
        self.press(100, f"pr:{self.bot.curriculum.point_code(point.id)}")
        for _ in range(len(point.exercises)):
            if self.current_question(100).kind == "cloze":
                break
            self.answer_current(100)
        question = self.current_question(100)
        self.assertEqual(question.kind, "cloze")
        self.assertIn("(1) ___", self.telegram.all_text())
        self.telegram.reset()
        self.send(100, "только один ответ")
        self.assertIn("по порядку", self.telegram.all_text())
        self.assertEqual(self.current_question(100).ref, question.ref)
        self.telegram.reset()
        self.send(100, "\n".join(gap[0] for gap in question.gaps))
        self.assertIn("Верно", self.telegram.all_text())

    def test_vocabulary_hint_never_contains_the_word(self) -> None:
        self.press(100, "startpractice")
        for _ in range(len(self.bot.storage.user(100).state_data["queue"])):  # type: ignore[union-attr]
            user = self.bot.storage.user(100)
            assert user is not None
            if user.state != "practice":
                break
            question = self.current_question(100)
            if question.card_type == "vocab" and not question.is_choice:
                self.telegram.reset()
                self.press(100, f"hint:{self.step_payload(100)}")
                # Ищем слово целиком: «on» подстрокой сидит в «preposition».
                word = re.escape(question.expected[0])
                self.assertIsNone(re.search(
                    rf"(?<![A-Za-z]){word}(?![A-Za-z])",
                    self.telegram.all_text(),
                    re.IGNORECASE,
                ))
                return
            self.answer_current(100)
        self.skipTest("в очереди не было карточки лексики на вспоминание")

    def test_wrong_answer_offers_to_go_through_the_rule(self) -> None:
        self.press(100, "startpractice")
        while True:
            user = self.bot.storage.user(100)
            assert user is not None
            if user.state != "practice":
                self.skipTest("в очереди не было грамматики")
            if self.current_question(100).point_id:
                break
            self.answer_current(100)
        self.telegram.reset()
        self.answer_current(100, correctly=False)
        rule = [data for data in self.telegram.all_buttons() if data.startswith("rule:")]
        self.assertTrue(rule, "после ошибки нужна кнопка разбора правила")

        self.telegram.reset()
        self.press(100, rule[0])
        self.assertIn("Как строится:", self.telegram.all_text())

    def test_mixed_practice_header_does_not_spoil_the_rule(self) -> None:
        """В смешанной тренировке заголовок называет тему, а не само правило."""
        self.press(100, "startpractice")
        checked = 0
        for _ in range(10):
            user = self.bot.storage.user(100)
            assert user is not None
            if user.state != "practice":
                break
            question = self.current_question(100)
            if question.point_id:  # у карточек лексики тема и название совпадают
                # Вердикт по прошлому ответу идёт тем же сообщением, поэтому
                # заголовок задания ищем по счётчику, а не по первой строке.
                header = next(
                    line
                    for line in self.telegram.last().split("\n")
                    if re.match(r"^\d+/\d+ · ", line)
                )
                self.assertIn(question.topic, header)
                self.assertNotIn(question.title_ru, header)
                checked += 1
            self.answer_current(100)
        self.assertGreater(checked, 0, "в очереди не оказалось грамматики")

    def test_practice_mixes_in_vocabulary_cards(self) -> None:
        """Иначе банк из 708 слов недостижим: карточки vocab никто не создаёт."""
        self.press(100, "startpractice")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertTrue(
            any(ref.startswith("vocab:") for ref in user.state_data["queue"]),
            "в тренировке нет ни одного слова",
        )

        for _ in range(len(user.state_data["queue"]) + 2):
            current = self.bot.storage.user(100)
            assert current is not None
            if current.state != "practice":
                break
            self.answer_current(100)
        self.assertGreater(
            self.bot.storage.card_counts(100).get("vocab", (0, 0))[0],
            0,
            "карточки лексики так и не появились",
        )

    def test_point_practice_header_names_the_rule(self) -> None:
        point = self.bot.curriculum.points_of_level("B1")[0]
        self.press(100, f"pr:{self.bot.curriculum.point_code(point.id)}")
        self.assertIn(point.title_ru, self.telegram.last().split("\n")[0])

    def test_stop_button_ends_the_session(self) -> None:
        self.press(100, "startpractice")
        self.press(100, "endses")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")

    def test_command_during_practice_leaves_the_session(self) -> None:
        self.press(100, "startpractice")
        self.send(100, "/learn present perfect")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")

    def test_settings_command_does_not_close_a_pending_writing_task(self) -> None:
        """Переключить участие в таблице — не смена занятия: задание должно уцелеть."""
        self.enable_llm()
        self.send(100, menu.WRITING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "writing")

        self.send(100, "/privacy")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "writing")

    def test_stop_closes_a_pending_writing_task(self) -> None:
        """Прервать занятие можно намеренно — этим и занимается /stop."""
        self.enable_llm()
        self.send(100, menu.WRITING)
        self.send(100, "/stop")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")

    def test_say_does_not_interrupt_a_pending_task(self) -> None:
        self.enable_llm()
        self.send(100, menu.WRITING)
        item = self.bot.curriculum.vocab_of_level("A1")[0]
        self.send(100, f"/say {item.word}")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "writing")

    def test_review_is_empty_until_cards_exist(self) -> None:
        self.press(100, "startreview")
        self.assertIn("Карточек ещё нет", self.telegram.all_text())


class CourseBrowsingTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1")

    def test_levels_topics_and_lesson_are_reachable(self) -> None:
        self.send(100, menu.COURSE)
        self.assertIn("lvl:B1", self.telegram.buttons())

        self.press(100, "lvl:B1")
        topic_button = next(data for data in self.telegram.buttons() if data.startswith("tp:"))
        self.press(100, topic_button)
        point_button = next(data for data in self.telegram.buttons() if data.startswith("pt:"))
        self.telegram.reset()

        self.press(100, point_button)
        text = self.telegram.all_text()
        self.assertIn("Ловушка для русскоязычных", text)
        self.assertTrue(any(data.startswith("pr:") for data in self.telegram.buttons()))

    def test_search_finds_a_rule_by_name(self) -> None:
        self.send(100, "/learn present perfect")
        self.assertTrue(any(data.startswith("pt:") for data in self.telegram.buttons()))

    def test_practice_from_a_lesson_starts_a_session(self) -> None:
        point = self.bot.curriculum.points_of_level("B1")[0]
        self.press(100, f"pr:{self.bot.curriculum.point_code(point.id)}")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")
        self.assertEqual(user.state_data["kind"], "point")


class PronunciationTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def test_known_word_is_answered_from_the_bank_without_ai(self) -> None:
        item = self.bot.curriculum.vocab_of_level("A1")[0]
        self.send(100, f"/say {item.word}")
        text = self.telegram.all_text()
        self.assertIn(item.ipa_us, text)
        self.assertIn("Озвучка выключена", text)
        cached = self.bot.storage.pronunciation(item.word)
        assert cached is not None
        self.assertEqual(cached["ipa"], item.ipa_us)

    def test_unknown_word_needs_ai_and_says_so(self) -> None:
        self.send(100, "/say zzzqqq")
        self.assertIn("не настроен", self.telegram.all_text())

    def test_say_sends_one_voice_with_caption_and_slow_button(self) -> None:
        """Слово, транскрипция и звучание должны прийти ОДНИМ сообщением."""
        audio = Path(self._dir.name) / "voice.ogg"
        audio.write_bytes(b"fake-ogg")

        class StubSpeaker:
            def synthesize(self, text: str, slow: bool = False) -> Path:
                return audio

        self.bot.speaker = StubSpeaker()  # type: ignore[assignment]
        item = self.bot.curriculum.vocab_of_level("A1")[0]
        self.send(100, f"/say {item.word}")

        self.assertEqual(len(self.telegram.voices), 1, "озвучка должна быть одним сообщением")
        _, path, caption = self.telegram.voices[0]
        self.assertEqual(path, audio)
        self.assertIn(item.ipa_us, caption)
        self.assertIn(item.word, caption)
        self.assertTrue(any(data.startswith("slow:") for data in self.telegram.buttons()))

    def test_say_without_argument_explains_usage(self) -> None:
        self.send(100, "/say")
        self.assertIn("/say schedule", self.telegram.all_text())


class ListeningTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")
        self.audio = Path(self._dir.name) / "listening.ogg"
        self.audio.write_bytes(b"fake-ogg")

        class StubSpeaker:
            def __init__(stub) -> None:
                stub.calls: list[str] = []
                stub.gentle: list[bool] = []

            def synthesize(stub, text: str, slow: bool = False, gentle: bool = False) -> Path:
                stub.calls.append(text)
                stub.gentle.append(gentle)
                return self.audio

        self.speaker = StubSpeaker()
        self.bot.speaker = self.speaker  # type: ignore[assignment]

    def task(self):
        user = self.bot.storage.user(100)
        assert user is not None
        task = self.bot.curriculum.listening_by_code(str(user.state_data["task_code"]))
        assert task is not None
        return task

    def test_listening_sends_audio_without_leaking_the_script(self) -> None:
        self.press(100, "listen")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "listening")
        task = self.task()
        self.assertEqual(len(self.telegram.voices), 1)
        _, _, caption = self.telegram.voices[-1]
        self.assertIn(task.question_en, caption)
        self.assertNotIn(task.script_en, caption)
        self.assertTrue(any(data.startswith("la:") for data in self.telegram.buttons()))
        self.assertTrue(any(data.startswith("lr:") for data in self.telegram.buttons()))

    def test_listening_without_synthesis_fails_cleanly(self) -> None:
        self.bot.speaker = None
        self.press(100, "listen")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")
        self.assertIn("Голос сейчас не настроен", self.telegram.all_text())

    def key_position(self, task) -> int:
        """Кнопка несёт позицию в порядке показа, а не индекс из файла."""
        from english_bot.handlers.listening import shown_order

        return shown_order(task, self.bot.curriculum).index(task.correct_index)

    def test_correct_answer_records_progress_and_reveals_the_script(self) -> None:
        self.press(100, "listen")
        task = self.task()
        code = self.bot.curriculum.listening_code(task.id)
        self.telegram.reset()
        self.press(100, f"la:{code}:{self.key_position(task)}")

        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")
        self.assertEqual(self.bot.storage.session_totals(100, "listening"), (1, 1))
        self.assertIn("listening", self.bot.storage.skills(100))
        self.assertIn(task.script_en, self.telegram.all_text())
        self.assertIn("✅ Верно", self.telegram.all_text())

    def test_options_are_shown_in_balanced_order_not_file_order(self) -> None:
        """Ключ стоял на B в 16 из 30 заданий: «всегда B» проходило аудирование."""
        import collections

        from english_bot.handlers.listening import shown_order

        tasks = [task for rows in self.bot.curriculum.listening.values() for task in rows]
        letters = collections.Counter(
            shown_order(task, self.bot.curriculum).index(task.correct_index) for task in tasks
        )
        self.assertLessEqual(max(letters.values()) - min(letters.values()), 1)
        self.press(100, "listen")
        task = self.task()
        _, _, caption = self.telegram.voices[-1]
        letter = "ABCD"[self.key_position(task)]
        self.assertIn(f"{letter}. {task.options[task.correct_index]}", caption)

    def test_beginner_levels_hear_a_gentler_pace(self) -> None:
        """На A1–A2 запись спокойнее обычного, с B1 — обычный темп."""
        self.bot.storage.update_user(100, level="B2", target_level="C1")
        self.press(100, "listen")
        self.assertNotIn(self.task().level, ("A1", "A2"))
        self.assertEqual(self.speaker.gentle, [False])
        self.bot.storage.set_state(100, "idle")
        self.bot.storage.update_user(100, level="A1", target_level="A2")
        self.press(100, "listen")
        self.assertEqual(self.task().level, "A1")
        self.assertEqual(self.speaker.gentle, [False, True])

    def test_replay_reuses_telegram_file_and_does_not_resynthesize(self) -> None:
        self.press(100, "listen")
        task = self.task()
        code = self.bot.curriculum.listening_code(task.id)
        self.press(100, f"lr:{code}")
        self.assertEqual(len(self.speaker.calls), 1)
        self.assertEqual(len(self.telegram.voices), 2)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state_data["plays"], 2)

    def test_old_answer_button_cannot_answer_a_new_task(self) -> None:
        self.press(100, "listen")
        first = self.task()
        first_code = self.bot.curriculum.listening_code(first.id)
        self.press(100, f"la:{first_code}:{self.key_position(first)}")
        self.press(100, "listen")
        before = self.bot.storage.session_totals(100, "listening")
        self.press(100, f"la:{first_code}:{self.key_position(first)}")
        self.assertEqual(self.bot.storage.session_totals(100, "listening"), before)
        self.assertIn("уже закрыто", self.telegram.answered[-1])


class TeamTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def test_privacy_toggle_removes_from_the_board(self) -> None:
        self.press(100, "team", first_name="Владелец")
        self.assertIn("Владелец", self.telegram.all_text())

        self.send(100, "/privacy")
        self.telegram.reset()
        self.press(100, "team")
        self.assertNotIn("Владелец", self.telegram.all_text())

    def test_admin_panel_reports_content_and_people(self) -> None:
        self.send(100, "/admin")
        text = self.telegram.all_text()
        self.assertIn("Пользователей: 1", text)
        self.assertIn("Ошибок загрузки контента: 0", text)

    def test_name_comes_from_the_telegram_profile(self) -> None:
        """Спрашивать имя отдельной командой незачем — оно есть в каждом сообщении."""
        self.send(100, menu.PROFILE, first_name="Милтон", last_name="Иванов")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.display_name, "Милтон Иванов")

    def test_username_is_used_when_there_is_no_name(self) -> None:
        self.send(100, menu.PROFILE, username="milton")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.display_name, "@milton")

    def test_renaming_in_telegram_updates_the_board(self) -> None:
        self.send(100, menu.PROFILE, first_name="Милтон")
        self.send(100, menu.PROFILE, first_name="Миша")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.display_name, "Миша")

    def test_new_member_is_named_from_the_first_message(self) -> None:
        self.press(100, "invitenew")
        code = self.telegram.all_text().split("start=")[-1].split()[0].strip()
        self.telegram.reset()
        self.send(200, f"/start {code}", first_name="Коллега")
        member = self.bot.storage.user(200)
        assert member is not None
        self.assertEqual(member.display_name, "Коллега")

    def test_level_is_set_from_the_menu_not_a_command(self) -> None:
        self.press(100, "setlvl:B2")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.level, "B2")
        self.assertEqual(user.target_level, "C1")

    def test_forget_requires_explicit_confirmation(self) -> None:
        self.bot.storage.record_attempt(100, "e", "p", "B1", True, "x")
        self.send(100, "/forget")
        self.assertEqual(self.bot.storage.attempts_count(100)[0], 1)
        self.send(100, "/forget YES")
        self.assertEqual(self.bot.storage.attempts_count(100)[0], 0)


class GroupChatTests(BotTestCase):
    """Молчание в группе читается как «бот сломался» — так было у первого коллеги."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def test_command_in_a_group_gets_an_explanation(self) -> None:
        self.send_group(100, "/say hello")
        self.assertIn("только в личных сообщениях", self.telegram.all_text())

    def test_plain_chatter_in_a_group_is_ignored(self) -> None:
        self.send_group(100, "ребята, кто идёт обедать")
        self.assertEqual(self.telegram.sent, [])

    def test_group_message_does_not_touch_learning_data(self) -> None:
        before = self.bot.storage.user(100)
        assert before is not None
        self.send_group(100, "/practice")
        after = self.bot.storage.user(100)
        assert after is not None
        self.assertEqual(after.state, "idle")
        self.assertEqual(after.last_active_at, before.last_active_at)


class PlatformHelpTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

        class Stub:
            provider = "openai"

            def __init__(self) -> None:
                self.calls: list[tuple[str, list[dict[str, str]], dict[str, object]]] = []
                self.reply = "Открой «📊 Я» → «Уровень и диагностика»."
                self.fail = False

            def complete(
                self, system: str, messages: list[dict[str, str]], **kwargs: object
            ) -> str:
                self.calls.append((system, messages, kwargs))
                if self.fail:
                    raise LLMError("временный сбой")
                return self.reply

        self.stub = Stub()
        self.bot.llm = self.stub  # type: ignore[assignment]

    def test_bare_help_is_static_and_does_not_spend_an_ai_call(self) -> None:
        self.bot.llm = None
        self.send(100, "/help")
        self.assertIn("Всё ежедневное — кнопками", self.telegram.all_text())
        self.assertIn("/help", self.telegram.all_text())

    def test_question_is_grounded_in_retrieved_knowledge(self) -> None:
        self.send(100, "/help как пройти диагностику?")
        self.assertEqual(len(self.stub.calls), 1)
        system, messages, kwargs = self.stub.calls[0]
        self.assertIn("Уровень и диагностика", system)
        self.assertIn("Единственный источник фактов", system)
        self.assertEqual(messages, [{"role": "user", "content": "как пройти диагностику?"}])
        self.assertEqual(kwargs["user_id"], 100)
        self.assertEqual(kwargs["max_tokens"], 500)
        self.assertEqual(self.telegram.last(), self.stub.reply)

    def test_help_question_does_not_interrupt_an_active_lesson(self) -> None:
        self.bot.storage.set_state(100, "practice", {"marker": "keep"})
        self.send(100, "/help где найти аудирование?")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")
        self.assertEqual(user.state_data, {"marker": "keep"})

    def test_unrelated_question_is_rejected_without_calling_the_model(self) -> None:
        self.send(100, "/help сколько варить пельмени?")
        self.assertEqual(self.stub.calls, [])
        self.assertIn("только о том, как пользоваться English Lab", self.telegram.last())

    def test_too_long_question_is_rejected_without_calling_the_model(self) -> None:
        self.send(100, "/help " + "x" * 501)
        self.assertEqual(self.stub.calls, [])
        self.assertIn("до 500 символов", self.telegram.last())

    def test_invented_command_is_replaced_with_a_grounded_fallback(self) -> None:
        self.stub.reply = "Нажми /practice, чтобы запустить аудирование."
        with self.assertLogs("english_bot.handlers.core", level="WARNING"):
            self.send(100, "/help как включить аудирование?")
        self.assertNotIn("/practice", self.telegram.last())
        self.assertIn("Аудирование", self.telegram.last())

    def test_model_failure_returns_the_nearest_article(self) -> None:
        self.stub.fail = True
        with self.assertLogs("english_bot.handlers.core", level="WARNING"):
            self.send(100, "/help как открыть курс?")
        self.assertIn("ИИ сейчас не ответил", self.telegram.last())
        self.assertIn("📚 Курс", self.telegram.last())


class SlowProviderFeedbackTests(BotTestCase):
    """С Codex ответ идёт 8–12 секунд: без предупреждения человек решает, что бот умер."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

        class Stub:
            provider = "codex"

            def complete(self, *args: object, **kwargs: object) -> str:
                return "ответ"

            def complete_json(self, *args: object, **kwargs: object) -> dict:
                raise RuntimeError("в тесте в сеть не ходим")

        self.bot.llm = Stub()  # type: ignore[assignment]
        object.__setattr__(self.settings, "llm_provider", "codex")

    def test_unknown_word_warns_before_the_model_call(self) -> None:
        self.send(100, "/say zzqqxx")
        self.assertIn("спрашиваю модель", self.telegram.all_text())

    def test_unexpected_failure_still_answers_the_user(self) -> None:
        """Заглушка падает не LLMError, а чем попало — человек всё равно не должен молчать.

        Отказ теперь ловит обёртка фоновой задачи, а не страховка диспетчера:
        исключение внутри задачи не всплывает в дорожку обновлений.
        """
        with self.assertLogs("english_bot.context", level="ERROR"):
            self.send_guarded(100, "/say zzqqxx")
        self.assertIn("Не довёл дело до конца", self.telegram.all_text())

    def test_known_word_does_not_warn(self) -> None:
        item = self.bot.curriculum.vocab_of_level("A1")[0]
        self.send(100, f"/say {item.word}")
        self.assertNotIn("спрашиваю модель", self.telegram.all_text())

    def test_free_chat_warns_that_it_is_thinking(self) -> None:
        self.send(100, "Hello, how are you?")
        self.assertIn("Думаю над ответом", self.telegram.all_text())
        self.assertIn("ответ", self.telegram.all_text())

    def test_fast_provider_stays_silent(self) -> None:
        object.__setattr__(self.settings, "llm_provider", "openai")
        self.send(100, "Hello again")
        self.assertNotIn("Думаю над ответом", self.telegram.all_text())


class BackgroundJobTests(BotTestCase):
    """Одно голосовое не должно отнимать у человека интерфейс.

    Все обновления идут через `_submit`, то есть через ту самую дорожку, которую
    раньше занимала расшифровка: только так видно, что она освободилась.
    """

    def setUp(self) -> None:
        super().setUp()
        self.bot._jobs = JobRunner(2, timeout=30.0, pulse_interval=0.05)
        self.claim_owner()
        self.holding = threading.Event()
        self.release = threading.Event()
        self.transcribed = 0
        test = self

        class Transcriber:
            queue_ahead = 0

            def transcribe(self, path: Path, duration: int, *args: object) -> Transcript:
                test.transcribed += 1
                test.holding.set()
                test.release.wait(5)
                return Transcript(text="I have went to work", words=5, seconds=duration, fillers=0)

        self.bot.transcriber = Transcriber()  # type: ignore[assignment]

    def tearDown(self) -> None:
        self.release.set()
        if self.bot._jobs is not None:
            self.bot._jobs.wait_idle(5)
        super().tearDown()

    # ── обновления в том виде, в каком их отдаёт Telegram ────────

    def voice_update(self, user_id: int = 100, message_id: int = 7) -> dict[str, Any]:
        return {
            "update_id": message_id,
            "message": {
                "message_id": message_id,
                "chat": {"id": user_id, "type": "private"},
                "from": self.sender(user_id),
                "voice": {
                    "file_id": f"file-{message_id}",
                    "file_unique_id": f"uniq-{message_id}",
                    "duration": 12,
                    "file_size": 4096,
                },
            },
        }

    def text_update(self, text: str, user_id: int = 100) -> dict[str, Any]:
        return {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "chat": {"id": user_id, "type": "private"},
                "from": self.sender(user_id),
                "text": text,
            },
        }

    def press_update(self, data: str, user_id: int = 100) -> dict[str, Any]:
        return {
            "update_id": 1,
            "callback_query": {
                "id": "cb",
                "data": data,
                "from": self.sender(user_id),
                "message": {"message_id": 1, "chat": {"id": user_id, "type": "private"}},
            },
        }

    def start_voice(self, message_id: int = 7) -> Any:
        future = self.bot._submit(self.voice_update(message_id=message_id))
        self.assertTrue(self.holding.wait(3), "расшифровка так и не началась")
        return future

    # ── сам симптом ─────────────────────────────────────────────

    def test_voice_does_not_freeze_the_interface(self) -> None:
        """Прямой тест постановки: при живой расшифровке кнопка обязана ответить."""
        self.start_voice()
        pressed = self.bot._submit(self.press_update("progress"))
        pressed.result(timeout=3)  # до разделения дорожки здесь был бы таймаут
        self.assertTrue(self.telegram.answered)

    def test_stop_answers_while_the_voice_is_being_transcribed(self) -> None:
        self.start_voice()
        self.bot._submit(self.text_update("/stop")).result(timeout=3)
        self.assertIn("Остановился", self.telegram.all_text())

    def test_stop_cancels_the_running_job(self) -> None:
        """Отмена кооперативная: этап доработает, но результат уже не придёт."""
        self.start_voice()
        self.bot._submit(self.text_update("/stop")).result(timeout=3)
        self.assertIn("Прервал расшифровку голосового", self.telegram.all_text())
        self.release.set()
        self.assertTrue(self.bot._jobs.wait_idle(5))
        self.assertNotIn("Расшифровка (", self.telegram.all_text())

    def test_second_voice_is_refused_while_the_first_runs(self) -> None:
        """Очередь из длинных задач одного человека бесполезна: он ждёт первую."""
        self.start_voice(message_id=7)
        self.bot._submit(self.voice_update(message_id=8)).result(timeout=3)
        self.assertIn("Сначала закончу расшифровку голосового", self.telegram.all_text())
        self.release.set()
        self.assertTrue(self.bot._jobs.wait_idle(5))
        self.assertEqual(self.transcribed, 1)

    def test_command_during_a_job_does_not_erase_the_lesson(self) -> None:
        """Отказ обязан случиться до сброса состояния, а не после."""
        self.bot.storage.set_state(100, "speaking", {"task_id": "T1"})
        self.start_voice()
        self.bot._submit(self.text_update("/anki")).result(timeout=3)
        self.assertIn("Сначала закончу", self.telegram.all_text())
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "speaking")

    def test_late_report_does_not_erase_a_new_speaking_task(self) -> None:
        """Разбор задания T1 не имеет права закрыть взятое позже задание T2."""
        self.bot.storage.set_state(100, "speaking", {"task_id": "T1"})
        self.start_voice()
        self.bot.storage.set_state(100, "speaking", {"task_id": "T2"})
        self.release.set()
        self.assertTrue(self.bot._jobs.wait_idle(5))
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "speaking")
        self.assertEqual(user.state_data["task_id"], "T2")

    def test_voice_in_practice_keeps_the_lesson(self) -> None:
        """Голосовое посреди тренировки раньше молча стирало её безусловным сбросом."""
        self.bot.storage.set_state(100, "practice", {"index": 3})
        self.start_voice()
        self.release.set()
        self.assertTrue(self.bot._jobs.wait_idle(5))
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")
        self.assertEqual(user.state_data["index"], 3)

    def test_new_lesson_is_refused_before_the_current_one_is_closed(self) -> None:
        """Отказ обязан опередить `close_active`: иначе занятие закрыто, а нового нет."""
        self.bot.storage.set_state(100, "writing", {"task_id": "T1"})
        self.start_voice()
        self.bot._submit(self.press_update("sw:daily")).result(timeout=3)
        self.assertIn("Сначала закончу", self.telegram.all_text())
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "writing")
        self.assertEqual(user.state_data["task_id"], "T1")

    def test_help_falls_back_to_the_knowledge_base_while_busy(self) -> None:
        """Вопрос «почему бот молчит» задают как раз во время разбора."""
        self.enable_llm("ответ модели")
        self.start_voice()
        self.bot._submit(self.text_update("/help как пройти диагностику")).result(timeout=3)
        self.assertIn("Вот ближайшая справка", self.telegram.all_text())
        self.assertNotIn("ответ модели", self.telegram.all_text())

    def test_course_screen_works_while_the_job_runs(self) -> None:
        """Занятость — не повод отнимать навигацию."""
        self.start_voice()
        self.bot._submit(self.text_update("📚 Курс")).result(timeout=3)
        self.assertIn("A1", self.telegram.all_text())

    def test_forget_cancels_the_job_and_deletes_data(self) -> None:
        """`/forget` — единственная команда, которой занятость не мешает."""
        self.start_voice()
        self.bot._submit(self.text_update("/forget YES")).result(timeout=3)
        self.assertIn("Учебные данные удалены", self.telegram.all_text())
        self.release.set()
        self.assertTrue(self.bot._jobs.wait_idle(5))
        self.assertEqual(self.bot.storage.voices(100), [])

    def test_main_button_is_refused_while_the_job_runs(self) -> None:
        """Кнопка постоянной клавиатуры не должна начинать занятие поверх задачи."""
        self.start_voice()
        self.bot._submit(self.text_update(menu.SPEAKING)).result(timeout=3)
        self.assertIn("Сначала закончу", self.telegram.all_text())
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")

    def test_listening_leaves_no_session_after_stop(self) -> None:
        """Шов отмены стоит между отправкой аудио и открытием сессии."""
        holding = threading.Event()
        release = threading.Event()

        class Speaker:
            def synthesize(self, *args: object, **kwargs: object) -> Path:
                holding.set()
                release.wait(5)
                return Path(self.__class__.__name__)

        self.bot.speaker = Speaker()  # type: ignore[assignment]
        self.bot._submit(self.press_update("listen"))
        self.assertTrue(holding.wait(3))
        self.bot._submit(self.text_update("/stop")).result(timeout=3)
        release.set()
        self.assertTrue(self.bot._jobs.wait_idle(5))
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")
        self.assertEqual(self.bot.storage.sessions(100), [])

    def test_admin_screen_shows_the_running_job(self) -> None:
        """Иначе «бот молчит» и «бот занят» снаружи неразличимы."""
        self.start_voice()
        self.telegram.reset()
        self.send(100, "/admin")
        report = self.telegram.all_text()
        self.assertIn("Фоновые задачи: сейчас 1", report)
        self.assertIn("расшифровку голосового", report)

    def test_waiting_indicator_is_refreshed_while_the_job_runs(self) -> None:
        """Индикатор «печатает» живёт в Telegram секунды, а задача — минуты."""
        self.start_voice()
        deadline = time.monotonic() + 3
        while len(self.telegram.actions) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertGreaterEqual(len(self.telegram.actions), 2)


class VoiceStageTests(BotTestCase):
    """Имя этапа в отказе и в `/stop` должно меняться вместе с работой."""

    def setUp(self) -> None:
        super().setUp()
        self.bot._jobs = JobRunner(2, timeout=30.0, pulse_interval=0.05)
        self.claim_owner()
        self.holding = threading.Event()
        self.release = threading.Event()

        class Transcriber:
            queue_ahead = 0

            def transcribe(self, path: Path, duration: int, *args: object) -> Transcript:
                return Transcript(text="I have went to work", words=5, seconds=duration, fillers=0)

        test = self

        class Stub:
            provider = "openai"

            def complete(self, *args: object, **kwargs: object) -> str:
                return "ответ"

            def complete_json(self, *args: object, **kwargs: object) -> dict[str, Any]:
                test.holding.set()
                test.release.wait(5)
                return {}

        self.bot.transcriber = Transcriber()  # type: ignore[assignment]
        self.bot.llm = Stub()  # type: ignore[assignment]

    def tearDown(self) -> None:
        self.release.set()
        if self.bot._jobs is not None:
            self.bot._jobs.wait_idle(5)
        super().tearDown()

    def test_refusal_names_the_current_stage_not_the_first_one(self) -> None:
        self.press(100, "speak")
        self.telegram.reset()
        self.bot._submit(
            {
                "update_id": 7,
                "message": {
                    "message_id": 7,
                    "chat": {"id": 100, "type": "private"},
                    "from": self.sender(100),
                    "voice": {
                        "file_id": "file-7",
                        "file_unique_id": "uniq-7",
                        "duration": 12,
                        "file_size": 4096,
                    },
                },
            }
        )
        self.assertTrue(self.holding.wait(3), "разбор так и не начался")
        self.bot._submit(
            {
                "update_id": 8,
                "callback_query": {
                    "id": "cb",
                    "data": "listen",
                    "from": self.sender(100),
                    "message": {"message_id": 1, "chat": {"id": 100, "type": "private"}},
                },
            }
        ).result(timeout=3)
        self.assertIn("Сначала закончу разбор устного ответа", self.telegram.all_text())


class ProductionWiringTests(unittest.TestCase):
    """Боевая сборка раннера собирается из настроек, а не из тестовой подмены."""

    def test_job_runner_is_built_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            env = {
                "TELEGRAM_BOT_TOKEN": "test",
                "BOT_CLAIM_CODE": CLAIM_CODE,
                "DATABASE_PATH": str(root / "db.sqlite3"),
                "EXPORT_DIR": str(root / "exports"),
                "VOICE_DIR": str(root / "voices"),
                "AUDIO_CACHE_DIR": str(root / "audio"),
                "MODELS_DIR": str(root / "models"),
                "LLM_PROVIDER": "openai",
                "SPEECH_BACKEND": "openai",
                "OPENAI_API_KEY": "",
                "JOB_WORKERS": "2",
                "JOB_TIMEOUT": "120",
            }
            saved = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                bot = EnglishLabBot(Settings.from_env())
                try:
                    self.assertIsNotNone(bot._jobs)
                    self.assertEqual(bot.context().jobs, bot._jobs)
                    self.assertEqual(bot._jobs._timeout, 120)
                finally:
                    bot.close(wait=False)

                os.environ["JOB_WORKERS"] = "0"
                inline = EnglishLabBot(Settings.from_env())
                try:
                    self.assertIsNone(inline._jobs)
                finally:
                    inline.close(wait=False)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class ClickThroughPlacementTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def _run_placement(self, choice: str) -> None:
        self.press(100, "retest")
        for _ in range(60):
            user = self.bot.storage.user(100)
            assert user is not None
            if user.state != "placement":
                return
            self.answer_placement(100, choice)

    def test_all_wrong_offers_to_retake(self) -> None:
        """Ноль верных почти всегда значит «прокликал», а не «ничего не знает»."""
        self._run_placement("3")
        text = self.telegram.all_text()
        if "верно 0 (0%)" not in text:
            self.skipTest("случайный выбор угадал часть ответов")
        self.assertIn("кликал наугад", text)
        self.assertIn("retest", self.telegram.all_buttons())

    def test_retake_button_starts_a_new_session(self) -> None:
        self._run_placement("3")
        if "retest" not in self.telegram.all_buttons():
            self.skipTest("случайный выбор угадал часть ответов")
        before = self.bot.storage.placement_answers(100)
        self.press(100, "retest")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "placement")
        self.assertGreater(user.state_data["session_id"], before[0]["session_id"])


class AiDisabledTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def test_free_chat_explains_that_ai_is_off(self) -> None:
        self.send(100, "Hello, how are you?")
        self.assertIn("не настроен", self.telegram.all_text())

    def test_writing_refuses_before_the_text_is_written(self) -> None:
        """Просить сто слов и отказать после — худший способ потратить время."""
        self.bot.storage.update_user(100, level="A2")
        self.send(100, menu.WRITING)
        self.assertIn("не настроен", self.telegram.all_text())
        self.assertNotIn("Объём", self.telegram.all_text())
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")

    def test_speaking_refuses_before_the_voice_is_recorded(self) -> None:
        self.bot.storage.update_user(100, level="A2")
        self.send(100, menu.SPEAKING)
        self.assertIn("не настроен", self.telegram.all_text())
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")


class RetentionAndAdminTests(BotTestCase):
    """Возврат после перерыва, честная серия, счётчики и приглашения."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")

    def _last_practice(self, days_ago: int) -> None:
        from datetime import UTC, datetime, timedelta

        day = (datetime.now(UTC) - timedelta(days=days_ago)).date().isoformat()
        self.bot.storage.update_user(100, streak_days=7, last_practice_day=day)

    def test_streak_expires_by_itself(self) -> None:
        """Иначе после десяти дней паузы профиль показывает «Серия: 7 дн.»."""
        self._last_practice(10)
        self.assertEqual(self.bot.storage.effective_streak(100), 0)

        self._last_practice(1)
        self.assertEqual(self.bot.storage.effective_streak(100), 7)

    def test_hub_greets_after_a_break(self) -> None:
        self._last_practice(10)
        self.send(100, menu.PROFILE)
        self.assertIn("С возвращением", self.telegram.all_text())
        self.assertIn("Серия: 0 дн.", self.telegram.all_text())

    def test_interface_events_are_counted_without_content(self) -> None:
        """Счётчики нужны, чтобы приоритезация опиралась на данные, а не на догадки."""
        self.press(100, "startpractice")
        self.answer_current(100)
        names = {row["name"] for row in self.bot.storage.event_counts(days=1)}
        self.assertIn("start", names)
        self.send(100, "/admin")
        self.assertIn("Интерфейс за 14 дней", self.telegram.all_text())

    def test_used_and_expired_invites_explain_themselves(self) -> None:
        self.press(100, "invitenew")
        code = self.bot.storage.invites()[0]["code"]
        self.bot.storage.redeem_invite(code, 200)
        self.telegram.reset()
        self.send(300, f"/start {code}")
        self.assertIn("уже воспользовались", self.telegram.all_text())

        fresh = self.bot.storage.create_invite(100, role="member")
        with self.bot.storage.session() as db:
            db.execute(
                "UPDATE invites SET expires_at = ? WHERE code = ?",
                ("2000-01-01T00:00:00+00:00", fresh),
            )
        self.telegram.reset()
        self.send(400, f"/start {fresh}")
        self.assertIn("Срок этого приглашения истёк", self.telegram.all_text())

    def test_export_and_anki_are_commands_now(self) -> None:
        self.send(100, "/export")
        self.assertIn("English Lab", self.telegram.all_text())
        self.telegram.reset()
        self.send(100, "/anki")
        self.assertTrue(self.telegram.all_text() or self.telegram.documents)


class LevelSplitTests(BotTestCase):
    """Уровень должен что-то значить: и в подборе, и после ручной правки."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="C2", target_level="C2")

    def test_lowering_the_level_clears_cards_from_above(self) -> None:
        """Иначе «Повторение» навсегда остаётся курсом прежнего уровня."""
        from english_bot.learning.srs import new_card

        high = self.bot.curriculum.points_of_level("C2")[0]
        low = self.bot.curriculum.points_of_level("B1")[0]
        for point in (high, low):
            self.bot.storage.upsert_card(100, new_card("point", point.id))

        self.press(100, "setlvl:B1")
        keys = set(self.bot.storage.card_keys(100, "point"))
        self.assertIn(low.id, keys)
        self.assertNotIn(high.id, keys)
        self.assertIn("Убрал из повторения", self.telegram.all_text())

    def test_review_never_lifts_material_above_the_level(self) -> None:
        import random

        from english_bot.learning import practice as pr
        from english_bot.learning.srs import new_card

        high = self.bot.curriculum.points_of_level("C2")[0]
        self.bot.storage.upsert_card(100, new_card("point", high.id))
        self.bot.storage.update_user(100, level="B1")
        cards = self.bot.storage.due_cards(100, limit=30)
        refs = pr.queue_for_review(self.bot.curriculum, cards, random.Random(1), 15, level="B1")
        levels = {
            q.level
            for q in (pr.resolve(ref, self.bot.curriculum, random.Random(1)) for ref in refs)
            if q is not None
        }
        self.assertNotIn("C2", levels)

    def test_productive_skills_start_a_level_below(self) -> None:
        """Узнавание грамматики не доказывает, что человек так же говорит и пишет."""
        from english_bot.learning.progress import skill_level

        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(skill_level(self.bot.storage, user, "speaking"), "C1")

        self.bot.storage.set_skill(100, "speaking", mastery=4, minutes_delta=10)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(skill_level(self.bot.storage, user, "speaking"), "C2")

    def test_placement_offers_a_choice_when_it_beats_self_assessment(self) -> None:
        """Тест меряет узнавание и завышает: разрыв с самооценкой нельзя проглатывать."""
        from english_bot.learning import placement as pl

        self.bot.storage.update_user(100, level="")
        state = pl.PlacementState(session_id=1)
        state.profile["profile_self_index"] = "1"  # самооценка A2
        state.results = {
            "B2": [1] * 6,
            "C1": [1] * 6,
            "C2": [1, 1, 1, 1, 1, 0],
        }
        self.bot.storage.set_state(100, "placement", state.to_dict())
        user = self.bot.storage.user(100)
        assert user is not None
        from english_bot.handlers import study

        study._finish_placement(self.bot.context(), user, state)
        text = self.telegram.all_text()
        self.assertIn("оценил себя как A2", text)
        self.assertIn("setlvl:C2", self.telegram.buttons())
        self.assertIn("setlvl:B1", self.telegram.buttons())


if __name__ == "__main__":
    unittest.main()
