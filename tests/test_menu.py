"""Навигация: постоянная клавиатура, «умная» кнопка занятия и экран профиля.

Главное требование к этому слою — путь до пользы. Тесты меряют его буквально:
сколько сообщений уходит от нажатия до первого задания на экране.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from english_bot.handlers import menu
from english_bot.learning.srs import new_card, review
from tests.test_bot import BotTestCase


class KeyboardTests(BotTestCase):
    def test_keyboard_arrives_with_the_first_message(self) -> None:
        self.send(100, f"/start {self.settings.claim_code}")
        markup = next(m for m in self.telegram.markups if m and "keyboard" in m)
        texts = [button["text"] for row in markup["keyboard"] for button in row]
        self.assertEqual(texts[0], menu.PRACTICE)
        self.assertEqual(len(texts), 5)

    def test_keyboard_is_persistent_and_not_one_time(self) -> None:
        self.claim_owner()
        self.send(100, menu.PROFILE)
        markup = next(m for m in self.telegram.markups if m and "keyboard" in m)
        self.assertTrue(markup["is_persistent"])
        self.assertNotIn("one_time_keyboard", markup)

    def test_nothing_removes_the_keyboard(self) -> None:
        """REMOVE_KEYBOARD снёс бы её навсегда — путь в одно нажатие сломался бы."""
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1")
        for action in (menu.PRACTICE, "/stop", menu.SPEAKING, menu.WRITING, menu.COURSE):
            self.send(100, action)
        for data in ("chat", "retest", "startreview"):
            self.press(100, data)
        removals = [m for m in self.telegram.markups if m and m.get("remove_keyboard")]
        self.assertEqual(removals, [])

    def test_buttons_are_recognised_and_others_are_not(self) -> None:
        self.assertTrue(menu.is_button(menu.PRACTICE))
        self.assertTrue(menu.is_button(menu.COURSE))
        self.assertFalse(menu.is_button("I have lived here for five years"))
        self.assertFalse(menu.is_button("Заниматься"))  # без эмодзи — обычный текст


class OneClickTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def _due_cards(self, count: int) -> None:
        past = (datetime.now(UTC) - timedelta(days=3)).isoformat(timespec="seconds")
        for point in self.bot.curriculum.points_of_level("B1")[:count]:
            card = new_card("point", point.id)
            self.bot.storage.upsert_card(100, card)
            with self.bot.storage.session() as db:
                db.execute(
                    "UPDATE srs_cards SET due_at = ? WHERE user_id = ? AND card_key = ?",
                    (past, 100, point.id),
                )

    def test_first_tap_starts_placement_when_level_is_unknown(self) -> None:
        self.send(100, menu.PRACTICE)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "placement")
        self.assertIn("Зачем тебе английский", self.telegram.all_text())

    def test_one_tap_puts_a_question_on_the_screen(self) -> None:
        """От нажатия до первого задания — не больше трёх сообщений."""
        self.bot.storage.update_user(100, level="B1", target_level="B2")
        self.send(100, menu.PRACTICE)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")
        self.assertIn("1/10", self.telegram.all_text())
        self.assertLessEqual(len(self.telegram.sent), 3)

    def test_due_cards_win_over_new_practice(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        self._due_cards(menu.REVIEW_THRESHOLD + 1)
        action, reason = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "review")
        self.assertIn("повторение", reason)

    def test_few_due_cards_do_not_interrupt_new_material(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        self._due_cards(2)
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "practice")

    def test_after_the_daily_norm_it_offers_speaking(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        action, reason = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "speaking")
        self.assertIn("вслух", reason)

    def test_recent_speaking_rotates_to_listening(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        self.bot.storage.add_voice(
            user_id=100, telegram_message_id=1, file_id="f", file_unique_id="u",
            duration_seconds=60, local_path=self.settings.voice_dir / "x.ogg", task_id="t",
        )
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "listening")

    def test_recent_speaking_and_listening_send_back_to_practice(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        self.bot.storage.add_voice(
            user_id=100, telegram_message_id=1, file_id="f", file_unique_id="u",
            duration_seconds=60, local_path=self.settings.voice_dir / "x.ogg", task_id="t",
        )
        listening = self.bot.storage.start_session(100, "listening", "b1_ls")
        self.bot.storage.finish_session(listening, items=1, correct=1)
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "practice")

    def test_speaking_button_gives_a_task_in_one_tap(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        self.send(100, menu.SPEAKING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "speaking")
        self.assertIn("🎙", self.telegram.all_text())

    def test_writing_button_gives_a_task_in_one_tap(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        self.send(100, menu.WRITING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "writing")

    def test_course_button_opens_levels_in_one_tap(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        self.send(100, menu.COURSE)
        self.assertIn("lvl:B1", self.telegram.buttons())


class ButtonsDuringSessionTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")

    def test_button_pressed_mid_session_switches_activity(self) -> None:
        self.send(100, menu.WRITING)
        self.send(100, menu.SPEAKING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "speaking")

    def test_button_pressed_mid_practice_restarts_the_daily_action(self) -> None:
        self.press(100, "startpractice")
        first = self.bot.storage.user(100)
        assert first is not None
        self.send(100, menu.PRACTICE)
        second = self.bot.storage.user(100)
        assert second is not None
        self.assertEqual(second.state, "practice")
        self.assertNotEqual(second.state_data["session_id"], first.state_data["session_id"])

    def test_english_answer_is_not_swallowed_by_the_menu(self) -> None:
        """Ответ ученика не должен случайно совпасть с кнопкой."""
        self.press(100, "startpractice")
        user = self.bot.storage.user(100)
        assert user is not None
        before = user.state_data["index"]
        self.send(100, "Course")  # похоже на кнопку «📚 Курс», но без эмодзи
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")
        self.assertGreaterEqual(user.state_data["index"], before)


class ProfileScreenTests(BotTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()

    def test_profile_shows_level_streak_and_queue(self) -> None:
        self.bot.storage.update_user(100, level="B1", target_level="B2", display_name="Милтон")
        self.send(100, menu.PROFILE)
        text = self.telegram.all_text()
        self.assertIn("Милтон", text)
        self.assertIn("B1 → B2", text)
        self.assertIn("Серия", text)

    def test_extra_actions_are_two_taps_away(self) -> None:
        self.send(100, menu.PROFILE)
        buttons = self.telegram.buttons()
        for expected in ("chat", "listen", "askword", "plan", "team", "anki", "export", "help"):
            with self.subTest(button=expected):
                self.assertIn(expected, buttons)

    def test_invite_button_only_for_admins(self) -> None:
        self.send(100, menu.PROFILE)
        self.assertIn("invitenew", self.telegram.buttons())

        self.bot.storage.create_user(200, 200, role="member")
        self.telegram.reset()
        self.send(200, menu.PROFILE)
        self.assertNotIn("invitenew", self.telegram.buttons())

    def test_pronunciation_button_asks_for_a_word_then_answers(self) -> None:
        self.press(100, "askword")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "awaiting_word")

        item = self.bot.curriculum.vocab_of_level("A1")[0]
        self.telegram.reset()
        self.send(100, item.word)
        self.assertIn(item.ipa_us, self.telegram.all_text())
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "idle")


class InterfaceSplitTests(BotTestCase):
    """Одно действие — одна точка входа.

    Пока одно и то же лежало и в кнопке, и в команде, человек не пользовался
    ни тем, ни другим: он просто не знал, что здесь главное. Меню закрывает всё
    ежедневное, командам остаётся то, чего кнопкой не сделать, — произвольный
    аргумент и опасные операции.
    """

    EXPECTED_COMMANDS = {
        "/start", "/help", "/say", "/learn", "/roleplay",
        "/stop", "/cancel", "/privacy", "/forget", "/admin",
    }
    # Всё это доступно кнопками, поэтому командой быть не должно.
    RETIRED = (
        "/menu", "/me", "/test", "/practice", "/review", "/speaking",
        "/writing", "/chat", "/progress", "/plan", "/team", "/anki", "/export",
        "/invite", "/pron",
    )

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")

    def test_registry_holds_only_what_the_menu_cannot_do(self) -> None:
        from english_bot.app import COMMANDS

        self.assertEqual(set(COMMANDS), self.EXPECTED_COMMANDS)

    def test_telegram_command_list_matches_the_registry(self) -> None:
        """Меню команд Telegram не должно предлагать то, чего бот не понимает."""
        from english_bot.app import COMMANDS
        from english_bot.handlers import core

        for name, _ in core.COMMANDS:
            with self.subTest(command=name):
                self.assertIn(f"/{name}", COMMANDS)

    def test_retired_commands_do_not_run_anything(self) -> None:
        for command in self.RETIRED:
            self.telegram.reset()
            self.send(100, command)
            user = self.bot.storage.user(100)
            assert user is not None
            with self.subTest(command=command):
                self.assertIn("кнопками внизу экрана", self.telegram.all_text())
                self.assertEqual(user.state, "idle")

    def test_help_text_does_not_advertise_retired_commands(self) -> None:
        from english_bot.handlers.core import HELP_TEXT

        for command in self.RETIRED:
            with self.subTest(command=command):
                self.assertNotIn(f"{command} ", HELP_TEXT)
                self.assertFalse(HELP_TEXT.endswith(command))

    def test_no_text_points_at_a_retired_command(self) -> None:
        """Подсказка на несуществующую команду — тупик, и её легко не заметить."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "english_bot"
        offenders: list[str] = []
        for source in sorted(root.rglob("*.py")):
            for number, line in enumerate(source.read_text().splitlines(), 1):
                stripped = line.rstrip()
                for command in self.RETIRED:
                    if f"{command} " in stripped or stripped.endswith(f'{command}"'):
                        offenders.append(f"{source.name}:{number} {command}")
        self.assertEqual(offenders, [])

    def test_bare_learn_asks_for_a_query_instead_of_repeating_the_course(self) -> None:
        self.send(100, "/learn")
        self.assertIn("📚 Курс", self.telegram.all_text())
        self.assertNotIn("lvl:B1", self.telegram.buttons())

    def test_learn_with_a_query_still_searches(self) -> None:
        self.send(100, "/learn present perfect")
        self.assertTrue(any(data.startswith("pt:") for data in self.telegram.buttons()))


class MenuCoverageTests(BotTestCase):
    """То, что перестало быть командой, обязано остаться достижимым кнопками."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")

    def _due_cards(self, count: int) -> None:
        past = (datetime.now(UTC) - timedelta(days=3)).isoformat(timespec="seconds")
        for point in self.bot.curriculum.points_of_level("B1")[:count]:
            self.bot.storage.upsert_card(100, new_card("point", point.id))
            with self.bot.storage.session() as db:
                db.execute(
                    "UPDATE srs_cards SET due_at = ? WHERE user_id = ? AND card_key = ?",
                    (past, 100, point.id),
                )

    def test_diagnostic_is_reachable_from_the_level_screen(self) -> None:
        self.press(100, "levelpick")
        buttons = self.telegram.buttons()
        self.assertIn("retest", buttons)
        self.assertIn("setlvl:B2", buttons)

        self.press(100, "retest")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "placement")

    def test_review_button_appears_only_when_something_is_due(self) -> None:
        self.send(100, menu.PROFILE)
        self.assertNotIn("startreview", self.telegram.all_buttons())

        self._due_cards(3)
        self.telegram.reset()
        self.send(100, menu.PROFILE)
        self.assertIn("startreview", self.telegram.all_buttons())

    def test_roleplay_button_starts_a_scenario_in_two_taps(self) -> None:
        from english_bot.handlers.dialogue import ROLEPLAY_PRESETS

        self.press(100, "roleplayhint")
        self.assertIn("rp:0", self.telegram.buttons())

        self.press(100, "rp:0")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "roleplay")
        self.assertEqual(user.state_data["scenario"], ROLEPLAY_PRESETS[0][1])

    def test_roleplay_command_still_takes_a_custom_scenario(self) -> None:
        self.send(100, "/roleplay объясняю на созвоне, почему упал прод")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "roleplay")
        self.assertIn("упал прод", user.state_data["scenario"])

    def test_broken_roleplay_payload_is_refused(self) -> None:
        self.press(100, "rp:99")
        self.assertIn("сценарий не найден", self.telegram.answered)

    def test_both_invite_codes_are_issued_from_the_menu(self) -> None:
        self.send(100, menu.PROFILE)
        buttons = self.telegram.all_buttons()
        self.assertIn("invitenew", buttons)
        self.assertIn("invitenew:admin", buttons)

        self.press(100, "invitenew:admin")
        self.press(100, "invitenew")
        roles = sorted(row["role"] for row in self.bot.storage.invites())
        self.assertEqual(roles, ["admin", "member"])


class SilencePaddingTests(unittest.TestCase):
    def test_padding_adds_exactly_the_configured_silence(self) -> None:
        import io
        import wave

        from english_bot.ai.local_speech import PAD_SECONDS, pad_wav

        rate, seconds = 22050, 0.5
        raw = io.BytesIO()
        with wave.open(raw, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(b"\x01\x02" * int(rate * seconds))
        padded = pad_wav(raw.getvalue())

        with wave.open(io.BytesIO(padded), "rb") as reader:
            self.assertEqual(reader.getframerate(), rate)
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getsampwidth(), 2)
            grown = reader.getnframes() / rate - seconds
        self.assertAlmostEqual(grown, 2 * PAD_SECONDS, places=3)

    def test_padding_is_silence_on_both_ends(self) -> None:
        import io
        import wave

        from english_bot.ai.local_speech import PAD_SECONDS, pad_wav

        rate = 22050
        raw = io.BytesIO()
        with wave.open(raw, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(b"\x7f\x7f" * rate)
        with wave.open(io.BytesIO(pad_wav(raw.getvalue())), "rb") as reader:
            frames = reader.readframes(reader.getnframes())
        edge = int(rate * PAD_SECONDS) * 2
        self.assertEqual(frames[:edge], b"\x00" * edge)
        self.assertEqual(frames[-edge:], b"\x00" * edge)

    def test_zero_padding_is_a_no_op(self) -> None:
        from english_bot.ai.local_speech import pad_wav

        self.assertEqual(pad_wav(b"whatever", seconds=0), b"whatever")


if __name__ == "__main__":
    unittest.main()
