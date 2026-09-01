"""Сквозные сценарии бота на поддельных Telegram и ИИ: доступ, диагностика, тренировка."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from english_bot.ai.llm import LLMError
from english_bot.app import EnglishLabBot
from english_bot.config import Settings
from english_bot.handlers import menu
from english_bot.learning import practice as pr


CLAIM_CODE = "claim-code-for-tests"


class FakeTelegram:
    """Заменяет сетевой клиент: копит отправленное и раздаёт предсказуемые ответы."""

    def __init__(self, token: str = "test"):
        self.sent: list[tuple[int, str]] = []
        self.markups: list[dict[str, Any] | None] = []
        self.voices: list[tuple[int, Path, str]] = []
        self.documents: list[tuple[int, Path]] = []
        self.answered: list[str] = []

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

    def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        return None

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
            self.press(user_id, f"an:{self.step(user_id)}:{index}")
        else:
            self.send(user_id, expected if correctly else "definitely wrong answer")

    def answer_placement(self, user_id: int, choice: str = "0") -> None:
        user = self.bot.storage.user(user_id)
        assert user is not None
        data = user.state_data
        if int(data.get("profile_index", 0)) < 3:
            self.press(user_id, f"pf:{data.get('profile_index', 0)}:{choice}")
        else:
            self.press(user_id, f"pa:{len(data.get('asked') or [])}:{choice}")

    def claim_owner(self, user_id: int = 100) -> None:
        self.send(user_id, f"/start {CLAIM_CODE}")
        self.telegram.reset()


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
        self.press(100, f"an:{self.step(100)}:0")
        self.assertIn("свободный ответ", " ".join(self.telegram.answered))

    def test_hint_shows_the_rule_and_lowers_the_grade(self) -> None:
        """Подсказка обязана объяснять тему, а не сдавать ответ."""
        self.press(100, "startpractice")
        question = self.current_question(100)
        self.telegram.reset()
        self.press(100, f"hint:{self.step(100)}")

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
        self.press(100, f"hint:{self.step(100)}")
        text = self.telegram.all_text()
        self.assertNotIn("точно не", text)
        self.assertNotIn("ответ начинается", text)

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
                self.press(100, f"hint:{self.step(100)}")
                self.assertNotIn(question.expected[0], self.telegram.all_text())
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
                header = self.telegram.last().split("\n")[0]
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

    def test_command_leaves_a_pending_writing_task(self) -> None:
        """Иначе следующее обычное сообщение уйдёт в разбор письма вместо чата."""
        self.send(100, menu.WRITING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "writing")

        self.send(100, "/privacy")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")

    def test_say_does_not_interrupt_a_pending_task(self) -> None:
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

            def synthesize(stub, text: str, slow: bool = False) -> Path:
                stub.calls.append(text)
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

    def test_correct_answer_records_progress_and_reveals_the_script(self) -> None:
        self.press(100, "listen")
        task = self.task()
        code = self.bot.curriculum.listening_code(task.id)
        self.telegram.reset()
        self.press(100, f"la:{code}:{task.correct_index}")

        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")
        self.assertEqual(self.bot.storage.session_totals(100, "listening"), (1, 1))
        self.assertIn("listening", self.bot.storage.skills(100))
        self.assertIn(task.script_en, self.telegram.all_text())
        self.assertIn("✅ Верно", self.telegram.all_text())

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
        self.press(100, f"la:{first_code}:{first.correct_index}")
        self.press(100, "listen")
        before = self.bot.storage.session_totals(100, "listening")
        self.press(100, f"la:{first_code}:{first.correct_index}")
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
        self.assertIn("Всё основное — кнопками", self.telegram.all_text())
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
        with self.assertRaises(RuntimeError):
            self.send(100, "/say zzqqxx")
        self.assertIn("спрашиваю модель", self.telegram.all_text())

    def test_unexpected_failure_still_answers_the_user(self) -> None:
        """Заглушка падает не LLMError, а чем попало — человек всё равно не должен молчать."""
        with self.assertLogs("english_bot.app", level="ERROR"):
            self.send_guarded(100, "/say zzqqxx")
        self.assertIn("Что-то пошло не так", self.telegram.all_text())

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

    def test_writing_task_is_offered_but_review_needs_ai(self) -> None:
        self.bot.storage.update_user(100, level="A2")
        self.send(100, menu.WRITING)
        self.assertIn("Объём", self.telegram.all_text())
        self.telegram.reset()
        self.send(100, "word " * 60)
        self.assertIn("не настроен", self.telegram.all_text())


if __name__ == "__main__":
    unittest.main()
