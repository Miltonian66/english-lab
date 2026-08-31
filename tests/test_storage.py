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
