"""Навигация: постоянная клавиатура, «умная» кнопка занятия и экран «📊 Я».

Два главных требования к этому слою — путь до пользы и сохранность начатого.
Тесты меряют оба буквально: сколько нажатий от постоянной клавиатуры до старта
функции и что происходит с незаконченным занятием, когда человек нажимает
другую кнопку.
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
        self.send(100, "/start")
        markup = next(m for m in self.telegram.markups if m and "keyboard" in m)
        self.assertTrue(markup["is_persistent"])
        self.assertNotIn("one_time_keyboard", markup)

    def test_nothing_removes_the_keyboard(self) -> None:
        """REMOVE_KEYBOARD снёс бы её навсегда — путь в одно нажатие сломался бы."""
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1")
        for action in (menu.PRACTICE, "/stop", menu.COURSE, menu.PROFILE):
            self.send(100, action)
        for data in ("retest", "startreview"):
            self.press(100, data)
        removals = [m for m in self.telegram.markups if m and m.get("remove_keyboard")]
        self.assertEqual(removals, [])

    def test_buttons_are_recognised_and_others_are_not(self) -> None:
        self.assertTrue(menu.is_button(menu.PRACTICE))
        self.assertTrue(menu.is_button(menu.RESUME))
        self.assertTrue(menu.is_button(menu.COURSE))
        self.assertFalse(menu.is_button("I have lived here for five years"))
        self.assertFalse(menu.is_button("Заниматься"))  # без эмодзи — обычный текст

    def test_main_button_says_continue_while_a_task_is_open(self) -> None:
        """Подпись — подсказка: пока занятие не закрыто, кнопка возвращает в него."""
        self.claim_owner()
        self.send(100, menu.PRACTICE)  # уровня нет — начнётся диагностика
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "placement")
        keyboard = menu.keyboard_for(user)
        self.assertEqual(keyboard["keyboard"][0][0]["text"], menu.RESUME)


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

    def test_listening_alone_does_not_close_the_daily_norm(self) -> None:
        """Один вопрос под запись — не дневная норма тренировки."""
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "listening", "b1_ls")
        self.bot.storage.finish_session(session, items=1, correct=1)
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "practice")

    def test_after_the_daily_norm_it_offers_speaking(self) -> None:
        self.enable_speech()
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        action, reason = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "speaking")
        self.assertIn("вслух", reason)

    def test_daily_skips_speech_when_the_provider_is_off(self) -> None:
        """Главная кнопка не имеет права предлагать невыполнимое."""
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "practice")

    def test_recent_speaking_rotates_to_listening(self) -> None:
        self.enable_speech()
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        self._reviewed_voice()
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "listening")

    def test_unreviewed_voice_does_not_count_as_speaking_practice(self) -> None:
        """Запись без разбора — не выполненная устная практика."""
        self.enable_speech()
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        self.bot.storage.add_voice(
            user_id=100, telegram_message_id=1, file_id="f", file_unique_id="u",
            duration_seconds=60, local_path=self.settings.voice_dir / "x.ogg", task_id="t",
        )
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "speaking")

    def test_writing_joins_the_rotation_when_speech_is_closed(self) -> None:
        """Письмо — кнопка главной, и умная кнопка обязана его предлагать."""
        self.enable_llm()
        self.enable_speech()
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        self._reviewed_voice()
        listening = self.bot.storage.start_session(100, "listening", "b1_ls")
        self.bot.storage.finish_session(listening, items=1, correct=1)
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "writing")

    def test_everything_closed_sends_back_to_practice(self) -> None:
        self.enable_llm()
        self.enable_speech()
        self.bot.storage.update_user(100, level="B1")
        session = self.bot.storage.start_session(100, "mixed", "B1")
        self.bot.storage.finish_session(session, items=10, correct=7)
        self._reviewed_voice()
        listening = self.bot.storage.start_session(100, "listening", "b1_ls")
        self.bot.storage.finish_session(listening, items=1, correct=1)
        self.bot.storage.add_writing(100, "w1", "text", {}, "report")
        action, _ = menu.choose_daily(self.bot.context(), self.bot.storage.user(100))
        self.assertEqual(action, "practice")

    def test_speaking_button_gives_a_task_in_one_tap(self) -> None:
        self.enable_speech()
        self.bot.storage.update_user(100, level="B1")
        self.send(100, menu.SPEAKING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "speaking")
        self.assertIn("🎙", self.telegram.all_text())

    def test_writing_button_gives_a_task_in_one_tap(self) -> None:
        self.enable_llm()
        self.bot.storage.update_user(100, level="B1")
        self.send(100, menu.WRITING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "writing")

    def test_course_button_opens_levels_in_one_tap(self) -> None:
        self.bot.storage.update_user(100, level="B1")
        self.send(100, menu.COURSE)
        self.assertIn("lvl:B1", self.telegram.buttons())

    def _reviewed_voice(self, message_id: int = 1) -> None:
        self.bot.storage.add_voice(
            user_id=100, telegram_message_id=message_id, file_id="f", file_unique_id="u",
            duration_seconds=60, local_path=self.settings.voice_dir / "x.ogg", task_id="t",
        )
        self.bot.storage.set_voice_transcript(100, message_id, "text", 12)
        self.bot.storage.set_voice_feedback(100, message_id, "разбор")


class StateGuardTests(BotTestCase):
    """Начатое занятие не исчезает молча — это главный контракт навигации."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")

    def _state(self, user_id: int = 100) -> str:
        user = self.bot.storage.user(user_id)
        assert user is not None
        return user.state

    def test_reading_screens_do_not_close_a_placement(self) -> None:
        self.send(100, "/stop")
        self.bot.storage.update_user(100, level="")
        self.send(100, menu.PRACTICE)
        self.assertEqual(self._state(), "placement")
        self.send(100, menu.COURSE)
        self.assertEqual(self._state(), "placement")
        self.send(100, menu.PROFILE)
        self.assertEqual(self._state(), "placement")

    def test_main_button_resumes_a_placement_instead_of_restarting_it(self) -> None:
        self.bot.storage.update_user(100, level="")
        self.send(100, menu.PRACTICE)
        self.answer_placement(100)
        before = self.bot.storage.user(100)
        assert before is not None
        self.telegram.reset()

        self.send(100, menu.PRACTICE)
        after = self.bot.storage.user(100)
        assert after is not None
        self.assertEqual(after.state, "placement")
        self.assertEqual(after.state_data["session_id"], before.state_data["session_id"])
        self.assertEqual(after.state_data["profile_index"], before.state_data["profile_index"])
        self.assertIn("Продолжаем диагностику", self.telegram.all_text())

    def test_start_command_does_not_wipe_a_placement(self) -> None:
        """Справка советует /start, когда клавиатура потерялась: совет не должен стоить теста."""
        self.bot.storage.update_user(100, level="")
        self.send(100, menu.PRACTICE)
        self.send(100, "/start")
        self.assertEqual(self._state(), "placement")
        self.assertIn("rsm", self.telegram.buttons())

    def test_switching_activity_mid_practice_asks_first(self) -> None:
        self.press(100, "startpractice")
        session = self.bot.storage.user(100).state_data["session_id"]  # type: ignore[union-attr]
        self.telegram.reset()

        self.send(100, menu.SPEAKING)
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")
        self.assertEqual(user.state_data["session_id"], session)
        self.assertIn("sw:speaking", self.telegram.buttons())
        self.assertIn("rsm", self.telegram.buttons())

    def test_confirmed_switch_closes_the_session_and_counts_the_day(self) -> None:
        self.press(100, "startpractice")
        self.answer_current(100)
        self.enable_speech()
        self.telegram.reset()

        self.send(100, menu.PRACTICE)
        self.press(100, "sw:speaking")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "speaking")
        self.assertIn("Готово:", self.telegram.all_text())
        self.assertTrue(self.bot.storage.practiced_today(100))

    def test_resume_returns_to_the_same_question(self) -> None:
        self.press(100, "startpractice")
        index = self.step(100)
        self.telegram.reset()

        self.send(100, menu.SPEAKING)
        self.press(100, "rsm")
        self.assertEqual(self.step(100), index)
        self.assertEqual(self._state(), "practice")
        self.assertIn(f"{index + 1}/10", self.telegram.all_text())

    def test_button_from_a_previous_session_does_not_count(self) -> None:
        """Кнопка «B» из прошлой тренировки не должна отвечать за текущую."""
        self.press(100, "startpractice")
        stale = f"an:{self.step_payload(100)}:0"
        self.press(100, "endses")
        self.press(100, "startpractice")
        self.telegram.reset()

        self.press(100, "stale" if False else stale)
        self.assertIn("это задание уже закрыто", " ".join(self.telegram.answered))

    def test_old_roleplay_end_button_does_not_close_a_practice(self) -> None:
        self.press(100, "startpractice")
        self.press(100, "endroleplay")
        self.assertEqual(self._state(), "practice")

    def test_hint_repeats_the_question_with_its_keyboard(self) -> None:
        """Разбор занимает экраны: без повтора кнопки ответа уезжают вверх."""
        self.press(100, "startpractice")
        index = self.step(100)
        self.telegram.reset()
        self.press(100, f"hint:{self.step_payload(100)}")
        self.assertIn(f"{index + 1}/10", self.telegram.all_text())
        self.assertTrue(
            any(
                data.startswith(("an:", "hint:", "skip:"))
                for data in self.telegram.all_buttons()
            ),
            "после разбора должна вернуться клавиатура задания",
        )


class HubTests(BotTestCase):
    """Экран «📊 Я» — статус и следующий шаг, а не склад возможностей."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2", display_name="Милтон")

    def _due_cards(self, count: int) -> None:
        past = (datetime.now(UTC) - timedelta(days=3)).isoformat(timespec="seconds")
        for point in self.bot.curriculum.points_of_level("B1")[:count]:
            self.bot.storage.upsert_card(100, new_card("point", point.id))
            with self.bot.storage.session() as db:
                db.execute(
                    "UPDATE srs_cards SET due_at = ? WHERE user_id = ? AND card_key = ?",
                    (past, 100, point.id),
                )

    def test_hub_is_one_message_with_status_and_next_step(self) -> None:
        self.send(100, menu.PROFILE)
        self.assertEqual(len(self.telegram.sent), 1)
        text = self.telegram.last()
        self.assertIn("Милтон", text)
        self.assertIn("B1 → B2", text)
        self.assertIn("Серия", text)
        self.assertIn("Сейчас полезнее всего", text)

    def test_next_step_button_follows_the_review_queue(self) -> None:
        self.send(100, menu.PROFILE)
        self.assertIn("startpractice", self.telegram.buttons())

        self._due_cards(3)
        self.telegram.reset()
        self.send(100, menu.PROFILE)
        self.assertIn("startreview", self.telegram.buttons())

    def test_rare_actions_are_two_taps_or_a_command(self) -> None:
        self.send(100, menu.PROFILE)
        buttons = self.telegram.buttons()
        for expected in ("listen", "rp:0", "roleplayhint", "plan", "progress", "retest",
                         "levelpick", "team"):
            with self.subTest(button=expected):
                self.assertIn(expected, buttons)

    def test_duplicates_of_commands_are_gone_from_the_hub(self) -> None:
        """Справка, произношение и свободный чат живут командами и обычным вводом."""
        self.send(100, menu.PROFILE)
        buttons = self.telegram.buttons()
        for gone in ("help", "askword", "chat", "anki", "export"):
            with self.subTest(button=gone):
                self.assertNotIn(gone, buttons)

    def test_invite_buttons_only_for_admins(self) -> None:
        self.send(100, menu.PROFILE)
        self.assertIn("invitenew", self.telegram.buttons())

        self.bot.storage.create_user(200, 200, role="member")
        self.bot.storage.update_user(200, level="A2")
        self.telegram.reset()
        self.send(200, menu.PROFILE)
        self.assertNotIn("invitenew", self.telegram.buttons())


class DepthFromHomeTests(BotTestCase):
    """Глубина считается только от постоянной клавиатуры — как её видит человек."""

    def setUp(self) -> None:
        super().setUp()
        self.claim_owner()
        self.bot.storage.update_user(100, level="B1", target_level="B2")
        self.enable_llm()
        self.enable_speech()

    def _from_home(self, *steps: str) -> None:
        """Проходит маршрут: первый шаг — кнопка внизу, дальше — inline-кнопки."""
        self.assertLessEqual(len(steps), 2, "путь до функции длиннее двух нажатий")
        self.telegram.reset()
        for index, step in enumerate(steps):
            if index == 0:
                self.send(100, step)
            else:
                self.press(100, step)

    def test_diagnostic_is_two_taps(self) -> None:
        self._from_home(menu.PROFILE, "retest")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "placement")

    def test_ready_roleplay_is_two_taps(self) -> None:
        self._from_home(menu.PROFILE, "rp:0")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "roleplay")

    def test_practice_on_demand_is_two_taps(self) -> None:
        self._from_home(menu.PROFILE, "startpractice")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "practice")

    def test_manual_level_grid_is_two_taps(self) -> None:
        self._from_home(menu.PROFILE, "levelpick")
        self.assertIn("setlvl:B2", self.telegram.buttons())

    def test_plan_and_progress_are_two_taps(self) -> None:
        self._from_home(menu.PROFILE, "plan")
        self.assertIn("План:", self.telegram.all_text())
        self._from_home(menu.PROFILE, "progress")
        self.assertIn("Профиль:", self.telegram.all_text())

    def test_course_continue_is_two_taps(self) -> None:
        """Каталог глубок, но вернуться к своей теме нужно с первого экрана."""
        self.send(100, menu.COURSE)
        resume = [data for data in self.telegram.buttons() if data.startswith("pt:")]
        self.assertTrue(resume, "на первом экране курса нет кнопки «Продолжить»")
        self.telegram.reset()
        self.press(100, resume[0])
        self.assertIn("Ловушка для русскоязычных", self.telegram.all_text())


class InterfaceSplitTests(BotTestCase):
    """Одно действие — одна точка входа.

    Пока одно и то же лежало и в кнопке, и в команде, человек не пользовался
    ни тем, ни другим: он просто не знал, что здесь главное. Меню закрывает всё
    ежедневное, командам остаётся то, чего кнопкой не сделать, — произвольный
    аргумент, редкая выгрузка и опасные операции.
    """

    EXPECTED_COMMANDS = {
        "/start", "/help", "/say", "/learn", "/roleplay", "/export", "/anki",
        "/stop", "/cancel", "/privacy", "/forget", "/admin",
    }
    # Всё это доступно кнопками, поэтому командой быть не должно.
    RETIRED = (
        "/menu", "/me", "/test", "/practice", "/review", "/speaking",
        "/writing", "/chat", "/progress", "/plan", "/team", "/invite", "/pron",
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

    def test_no_button_duplicates_a_command(self) -> None:
        """Список команд сверяется поимённо, а дубли ловятся по обработчику."""
        from english_bot.app import CALLBACKS, COMMANDS

        shared = {
            handler
            for name, handler in CALLBACKS.items()
            if handler in set(COMMANDS.values())
        }
        self.assertEqual(shared, set())

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

    def test_manual_level_and_diagnostic_are_separate_actions(self) -> None:
        self.press(100, "levelpick")
        buttons = self.telegram.buttons()
        self.assertIn("setlvl:B2", buttons)
        self.assertNotIn("retest", buttons)

        self.press(100, "retest")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "placement")

    def test_roleplay_command_still_takes_a_custom_scenario(self) -> None:
        from english_bot.handlers.dialogue import ROLEPLAY_PRESETS

        self.enable_llm()
        self.send(100, "/roleplay объясняю на созвоне, почему упал прод")
        user = self.bot.storage.user(100)
        assert user is not None
        self.assertEqual(user.state, "roleplay")
        self.assertIn("упал прод", user.state_data["scenario"])
        self.assertTrue(ROLEPLAY_PRESETS)

    def test_broken_roleplay_payload_is_refused(self) -> None:
        self.enable_llm()
        self.press(100, "rp:99")
        self.assertIn("сценарий не найден", self.telegram.answered)

    def test_both_invite_codes_are_issued_from_the_hub(self) -> None:
        self.send(100, menu.PROFILE)
        buttons = self.telegram.all_buttons()
        self.assertIn("invitenew", buttons)
        self.assertIn("invitenew:admin", buttons)

        self.press(100, "invitenew:admin")
        self.press(100, "invitenew")
        roles = sorted(row["role"] for row in self.bot.storage.invites())
        self.assertEqual(roles, ["admin", "member"])

    def test_issued_code_can_be_revoked(self) -> None:
        """Раньше живой код нельзя было погасить — только ждать 14 дней."""
        self.press(100, "invitenew")
        code = self.bot.storage.invites()[0]["code"]
        self.press(100, f"revoke:{code}")
        self.assertEqual(self.bot.storage.invites(), [])


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
