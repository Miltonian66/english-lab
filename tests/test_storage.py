"""Схема, миграция с версии 1 и изоляция данных между пользователями."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from english_bot.storage import SCHEMA_VERSION, Storage, utc_now


V1_SCHEMA = """
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'idle',
    question_index INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    completed_at TEXT
);
CREATE TABLE answers (
    user_id INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    selected_index INTEGER,
    is_correct INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, question_id)
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE voice_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    telegram_message_id INTEGER NOT NULL,
    file_id TEXT NOT NULL,
    file_unique_id TEXT NOT NULL,
    duration_seconds INTEGER NOT NULL,
    local_path TEXT NOT NULL,
    task_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, telegram_message_id)
);
"""


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "test.sqlite3"
        self.storage = Storage(self.path)
        self.storage.initialize()

    def tearDown(self) -> None:
        self._dir.cleanup()


class SchemaTests(StorageTestCase):
    def test_fresh_database_is_current_version(self) -> None:
        with self.storage.connect() as db:
            self.assertEqual(int(db.execute("PRAGMA user_version").fetchone()[0]), SCHEMA_VERSION)

    def test_initialize_is_idempotent(self) -> None:
        self.storage.create_user(1, 1, role="owner")
        self.storage.initialize()
        self.storage.initialize()
        self.assertIsNotNone(self.storage.user(1))


class MigrationTests(unittest.TestCase):
    def test_v1_data_survives_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.sqlite3"
            db = sqlite3.connect(path)
            db.executescript(V1_SCHEMA)
            db.execute("INSERT INTO settings VALUES ('owner_user_id', '777')")
            db.execute("INSERT INTO users(user_id, chat_id, state) VALUES (777, 777, 'completed')")
            for index in range(21):
                db.execute(
                    "INSERT INTO answers VALUES (777, ?, 'ответ', 0, 1, ?)",
                    (f"q{index}", utc_now()),
                )
            db.execute(
                "INSERT INTO voice_messages(user_id, telegram_message_id, file_id, "
                "file_unique_id, duration_seconds, local_path, task_id, created_at) "
                "VALUES (777, 1, 'f', 'u', 90, '/tmp/x.ogg', 'task', ?)",
                (utc_now(),),
            )
            db.commit()
            db.close()

            Storage(path).initialize()

            check = sqlite3.connect(path)
            check.row_factory = sqlite3.Row
            self.assertEqual(
                check.execute("SELECT count(*) FROM placement_answers").fetchone()[0], 21
            )
            self.assertEqual(
                check.execute("SELECT count(*) FROM voice_messages").fetchone()[0], 1
            )
            row = check.execute("SELECT role, share_progress FROM users").fetchone()
            self.assertEqual(row["role"], "owner")
            self.assertEqual(row["share_progress"], 1)
            self.assertEqual(int(check.execute("PRAGMA user_version").fetchone()[0]), SCHEMA_VERSION)
            check.close()

            # Повторная миграция не должна ломать уже переехавшую базу.
            Storage(path).initialize()


class UserTests(StorageTestCase):
    def test_roles_and_admin_flag(self) -> None:
        owner = self.storage.create_user(1, 1, role="owner")
        member = self.storage.create_user(2, 2, role="member")
        self.assertTrue(owner.is_admin)
        self.assertFalse(member.is_admin)

    def test_state_data_round_trips(self) -> None:
        self.storage.create_user(1, 1)
        self.storage.set_state(1, "practice", {"queue": ["ex:a"], "index": 2})
        user = self.storage.user(1)
        assert user is not None
        self.assertEqual(user.state, "practice")
        self.assertEqual(user.state_data["index"], 2)

    def test_corrupted_state_data_falls_back_to_empty(self) -> None:
        self.storage.create_user(1, 1)
        with self.storage.connect() as db:
            db.execute("UPDATE users SET state_data = 'не json' WHERE user_id = 1")
        user = self.storage.user(1)
        assert user is not None
        self.assertEqual(user.state_data, {})

    def test_learning_data_is_isolated_between_users(self) -> None:
        self.storage.create_user(1, 1)
        self.storage.create_user(2, 2)
        self.storage.record_attempt(1, "e1", "p1", "B1", True, "ok")
        self.storage.record_attempt(2, "e2", "p2", "A2", False, "no")
        self.storage.log_error(1, "grammar", "a", "b")

        self.assertEqual(self.storage.attempts_count(1), (1, 1))
        self.assertEqual(self.storage.attempts_count(2), (1, 0))
        self.assertEqual(len(self.storage.recent_errors(2)), 0)
        self.assertIn("p1", self.storage.point_stats(1))
        self.assertNotIn("p1", self.storage.point_stats(2))

    def test_delete_learning_data_keeps_account_and_role(self) -> None:
        self.storage.create_user(1, 1, role="admin")
        self.storage.record_attempt(1, "e1", "p1", "B1", True, "ok")
        self.storage.update_user(1, level="B1", streak_days=5)
        self.storage.delete_learning_data(1)
        user = self.storage.user(1)
        assert user is not None
        self.assertEqual(user.role, "admin")
        self.assertEqual(user.level, "")
        self.assertEqual(user.streak_days, 0)
        self.assertEqual(self.storage.attempts_count(1), (0, 0))


class BackgroundWriteTests(StorageTestCase):
    """Ограды записи: фоновая задача заканчивается минутой позже своего старта."""

    def setUp(self) -> None:
        super().setUp()
        self.storage.create_user(1, 1)

    def test_swap_state_ignores_a_foreign_task(self) -> None:
        """Разбор задания T1 не должен закрывать взятое позже задание T2."""
        self.storage.set_state(1, "speaking", {"task_id": "T2"})
        changed = self.storage.swap_state(1, "speaking", "idle", {}, key="task_id", value="T1")
        self.assertFalse(changed)
        user = self.storage.user(1)
        assert user is not None
        self.assertEqual(user.state, "speaking")
        self.assertEqual(user.state_data["task_id"], "T2")

    def test_swap_state_releases_its_own_task(self) -> None:
        self.storage.set_state(1, "speaking", {"task_id": "T1"})
        self.assertTrue(
            self.storage.swap_state(1, "speaking", "idle", {}, key="task_id", value="T1")
        )
        user = self.storage.user(1)
        assert user is not None
        self.assertEqual(user.state, "idle")

    def test_swap_state_matches_a_numeric_marker(self) -> None:
        """`session_id` лежит в JSON числом: без приведения к тексту ограда молчит."""
        self.storage.set_state(1, "listening", {"session_id": 7})
        self.assertTrue(
            self.storage.swap_state(1, "listening", "idle", {}, key="session_id", value="7")
        )

    def test_swap_state_without_a_marker_checks_only_the_state(self) -> None:
        self.storage.set_state(1, "practice", {"index": 3})
        self.assertFalse(self.storage.swap_state(1, "writing", "idle", {}))
        self.assertTrue(self.storage.swap_state(1, "practice", "idle", {}))

    def test_voice_transcript_does_not_touch_feedback(self) -> None:
        self._add_voice()
        self.storage.set_voice_feedback(1, 5, "разбор")
        self.storage.set_voice_transcript(1, 5, "новая расшифровка", 4)
        row = self.storage.voices(1)[0]
        self.assertEqual(row["transcript"], "новая расшифровка")
        self.assertEqual(row["feedback"], "разбор")

    def test_voice_feedback_is_written_once(self) -> None:
        """Повтор задачи не должен второй раз засчитывать навык и серию."""
        self._add_voice()
        self.assertTrue(self.storage.set_voice_feedback(1, 5, "первый разбор"))
        self.assertFalse(self.storage.set_voice_feedback(1, 5, "второй разбор"))
        self.assertEqual(self.storage.voices(1)[0]["feedback"], "первый разбор")

    def test_add_voice_keeps_the_task_of_a_reviewed_record(self) -> None:
        self._add_voice(task_id="T1")
        self.storage.set_voice_feedback(1, 5, "разбор")
        self._add_voice(task_id="free_speech")
        self.assertEqual(self.storage.voices(1)[0]["task_id"], "T1")

    def test_finish_session_does_not_reopen_a_closed_one(self) -> None:
        """Занятие дня считается по закрытым сессиям: обнулить их нельзя."""
        session_id = self.storage.start_session(1, "listening", "task")
        self.assertTrue(self.storage.finish_session(session_id, items=1, correct=1))
        self.assertFalse(self.storage.finish_session(session_id, items=0, correct=0))
        row = self.storage.sessions(1)[0]
        self.assertEqual(row["items"], 1)
        self.assertEqual(row["correct"], 1)

    def test_save_pronunciation_keeps_a_better_transcription(self) -> None:
        """Окно между чтением кэша и записью — минуты, и пустое поле не аргумент."""
        self.storage.save_pronunciation("schedule", "schedule", "/ˈskɛdʒul/", "заметка")
        self.storage.save_pronunciation("schedule", "", "", "", file_id="abc")
        cached = self.storage.pronunciation("schedule")
        assert cached is not None
        self.assertEqual(cached["ipa"], "/ˈskɛdʒul/")
        self.assertEqual(cached["note"], "заметка")
        self.assertEqual(cached["file_id"], "abc")

    def _add_voice(self, task_id: str = "T1") -> None:
        self.storage.add_voice(
            user_id=1,
            telegram_message_id=5,
            file_id="f",
            file_unique_id="u",
            duration_seconds=12,
            local_path=Path("/tmp/x.ogg"),
            task_id=task_id,
        )


class AccessLogTests(StorageTestCase):
    """Журнал входов: он нужен именно для тех, кого в `users` нет."""

    def test_refused_stranger_is_recorded_without_a_user_row(self) -> None:
        self.storage.log_access(555, 555, "Гость", "need_invite")
        row = self.storage.access_attempts()[0]
        self.assertEqual(row["user_id"], 555)
        self.assertEqual(row["outcome"], "need_invite")
        self.assertIsNone(self.storage.user(555))

    def test_summary_groups_outcomes_and_people(self) -> None:
        self.storage.log_access(1, 1, "A", "need_invite")
        self.storage.log_access(1, 1, "A", "need_invite")
        self.storage.log_access(2, 2, "B", "joined", "member")
        summary = {row["outcome"]: row for row in self.storage.access_summary(days=7)}
        self.assertEqual(summary["need_invite"]["times"], 2)
        self.assertEqual(summary["need_invite"]["people"], 1)
        self.assertEqual(summary["joined"]["times"], 1)

    def test_log_is_capped_and_keeps_the_newest(self) -> None:
        """Разбор «почему не пускает» смотрит в хвост, архив тут не нужен."""
        from english_bot.storage import ACCESS_LOG_LIMIT

        for number in range(ACCESS_LOG_LIMIT + 25):
            self.storage.log_access(number, number, "", "need_invite")
        with self.storage.session() as db:
            total = int(db.execute("SELECT count(*) FROM access_log").fetchone()[0])
        self.assertLessEqual(total, ACCESS_LOG_LIMIT + 1)
        self.assertEqual(self.storage.access_attempts(limit=1)[0]["user_id"], ACCESS_LOG_LIMIT + 24)

    def test_filter_by_outcome(self) -> None:
        self.storage.log_access(1, 1, "A", "joined", "member")
        self.storage.log_access(2, 2, "B", "need_invite")
        rows = self.storage.access_attempts(outcome="joined")
        self.assertEqual([row["user_id"] for row in rows], [1])


class InviteTests(StorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.storage.create_user(1, 1, role="owner")

    def test_invite_is_single_use(self) -> None:
        code = self.storage.create_invite(1)
        self.assertEqual(self.storage.redeem_invite(code, 2), "member")
        self.assertIsNone(self.storage.redeem_invite(code, 3))

    def test_invite_carries_role(self) -> None:
        code = self.storage.create_invite(1, role="admin")
        self.assertEqual(self.storage.redeem_invite(code, 2), "admin")

    def test_expired_invite_is_rejected(self) -> None:
        code = self.storage.create_invite(1)
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
        with self.storage.connect() as db:
            db.execute("UPDATE invites SET expires_at = ? WHERE code = ?", (past, code))
        self.assertIsNone(self.storage.redeem_invite(code, 2))

    def test_unknown_code_is_rejected(self) -> None:
        self.assertIsNone(self.storage.redeem_invite("выдуманный", 2))


class LimitAndStreakTests(StorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.storage.create_user(1, 1)

    def test_ai_calls_are_capped_per_day(self) -> None:
        for _ in range(3):
            self.assertTrue(self.storage.take_ai_call(1, limit=3))
        self.assertFalse(self.storage.take_ai_call(1, limit=3))
        self.assertEqual(self.storage.ai_calls_used(1), 3)

    def test_limit_is_per_user(self) -> None:
        self.storage.create_user(2, 2)
        self.assertTrue(self.storage.take_ai_call(1, limit=1))
        self.assertFalse(self.storage.take_ai_call(1, limit=1))
        self.assertTrue(self.storage.take_ai_call(2, limit=1))

    def test_streak_increments_once_per_day(self) -> None:
        self.assertEqual(self.storage.bump_streak(1), 1)
        self.assertEqual(self.storage.bump_streak(1), 1)

    def test_streak_continues_from_yesterday(self) -> None:
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        with self.storage.connect() as db:
            db.execute(
                "UPDATE users SET streak_days = 4, last_practice_day = ? WHERE user_id = 1",
                (yesterday,),
            )
        self.assertEqual(self.storage.bump_streak(1), 5)

    def test_streak_resets_after_a_gap(self) -> None:
        long_ago = (date.today() - timedelta(days=9)).isoformat()
        with self.storage.connect() as db:
            db.execute(
                "UPDATE users SET streak_days = 9, last_practice_day = ? WHERE user_id = 1",
                (long_ago,),
            )
        self.assertEqual(self.storage.bump_streak(1), 1)


class CardTests(StorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.storage.create_user(1, 1)

    def test_cards_upsert_and_report_due(self) -> None:
        from english_bot.learning.srs import new_card, review

        card = new_card("point", "b1_x")
        self.storage.upsert_card(1, card)
        self.assertEqual(len(self.storage.due_cards(1)), 1)

        future = review(card, 5)
        self.storage.upsert_card(1, future)
        self.assertEqual(len(self.storage.due_cards(1)), 0)
        self.assertEqual(self.storage.card_counts(1)["point"], (1, 0))

    def test_team_board_respects_privacy_flag(self) -> None:
        self.storage.create_user(2, 2)
        self.storage.update_user(2, share_progress=0)
        rows = self.storage.team_stats()
        self.assertEqual([row["user_id"] for row in rows], [1])


if __name__ == "__main__":
    unittest.main()
