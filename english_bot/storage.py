"""Хранилище платформы: схема SQLite, миграции и операции с учебными данными.

Схема версионируется через `PRAGMA user_version`. Версия 1 — личный бот на одного
владельца; версия 2 — многопользовательская платформа. Миграция не удаляет данные:
ответы старой диагностики переезжают в `placement_answers` и остаются историей.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3

ROLES: tuple[str, ...] = ("owner", "admin", "member")
CARD_TYPES: tuple[str, ...] = ("point", "vocab", "error")
SKILLS: tuple[str, ...] = ("grammar", "vocabulary", "reading", "listening", "writing", "speaking")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def today() -> str:
    return datetime.now(UTC).date().isoformat()


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


@dataclass(frozen=True)
class User:
    user_id: int
    chat_id: int
    role: str
    display_name: str
    level: str
    target_level: str
    state: str
    state_data: dict[str, Any]
    streak_days: int
    last_practice_day: str | None
    share_progress: bool
    created_at: str | None
    last_active_at: str | None

    @property
    def is_admin(self) -> bool:
        return self.role in {"owner", "admin"}


@dataclass(frozen=True)
class Card:
    card_type: str
    card_key: str
    ease: float
    interval_days: float
    repetitions: int
    lapses: int
    due_at: str
    mastery: int


SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    display_name TEXT NOT NULL DEFAULT '',
    level TEXT NOT NULL DEFAULT '',
    target_level TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'idle',
    state_data TEXT NOT NULL DEFAULT '{}',
    question_index INTEGER NOT NULL DEFAULT 0,
    streak_days INTEGER NOT NULL DEFAULT 0,
    last_practice_day TEXT,
    ai_calls_day TEXT,
    ai_calls_today INTEGER NOT NULL DEFAULT 0,
    share_progress INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    last_active_at TEXT,
    started_at TEXT,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS invites (
    code TEXT PRIMARY KEY,
    role TEXT NOT NULL DEFAULT 'member',
    created_by INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    used_by INTEGER,
    used_at TEXT
);
CREATE TABLE IF NOT EXISTS placement_answers (
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    selected_index INTEGER,
    is_correct INTEGER,
    level TEXT NOT NULL DEFAULT '',
    session_id INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, session_id, question_id)
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    exercise_id TEXT NOT NULL,
    point_id TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT '',
    is_correct INTEGER NOT NULL,
    answer_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_attempts_point ON attempts(user_id, point_id);
CREATE TABLE IF NOT EXISTS srs_cards (
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    card_type TEXT NOT NULL,
    card_key TEXT NOT NULL,
    ease REAL NOT NULL DEFAULT 2.5,
    interval_days REAL NOT NULL DEFAULT 0,
    repetitions INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    mastery INTEGER NOT NULL DEFAULT 0,
    due_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, card_type, card_key)
);
CREATE INDEX IF NOT EXISTS idx_srs_due ON srs_cards(user_id, due_at);
CREATE TABLE IF NOT EXISTS skill_mastery (
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    skill TEXT NOT NULL,
    mastery INTEGER NOT NULL DEFAULT 0,
    minutes INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, skill)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);
CREATE TABLE IF NOT EXISTS voice_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    telegram_message_id INTEGER NOT NULL,
    file_id TEXT NOT NULL,
    file_unique_id TEXT NOT NULL,
    duration_seconds INTEGER NOT NULL,
    local_path TEXT NOT NULL,
    task_id TEXT NOT NULL,
    transcript TEXT NOT NULL DEFAULT '',
    feedback TEXT NOT NULL DEFAULT '',
    words INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, telegram_message_id)
);
CREATE TABLE IF NOT EXISTS error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    pattern_id TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'grammar',
    original TEXT NOT NULL,
    corrected TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'chat',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_errors_user ON error_log(user_id, created_at);
CREATE TABLE IF NOT EXISTS writing_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    text TEXT NOT NULL,
    scores TEXT NOT NULL DEFAULT '{}',
    feedback TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pronunciations (
    word TEXT PRIMARY KEY,
    display TEXT NOT NULL,
    ipa TEXT NOT NULL,
    syllables TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    audio_path TEXT NOT NULL DEFAULT '',
    file_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    items INTEGER NOT NULL DEFAULT 0,
    correct INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, started_at);
"""


class Storage:
    def __init__(self, path: Path):
        self.path = path

    # ── подключение и миграции ────────────────────────────────────

    def connect(self) -> sqlite3.Connection:
        """Сырое соединение. Закрывать обязан вызывающий — обычно нужен `session()`."""
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def session(self) -> "Iterator[sqlite3.Connection]":
        """Транзакция с гарантированным закрытием: `with connection` только коммитит."""
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.session() as db:
            db.execute("PRAGMA journal_mode = WAL")
            version = int(db.execute("PRAGMA user_version").fetchone()[0])
            if version < 2:
                self._migrate_to_v2(db)
            if version < 3:
                self._migrate_to_v3(db)
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _migrate_to_v2(self, db: sqlite3.Connection) -> None:
        """Переводит личного бота v1 на многопользовательскую схему без потери данных."""
        tables = {
            row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "answers" in tables and "placement_answers" not in tables:
            db.execute("ALTER TABLE answers RENAME TO placement_answers")
            tables.discard("answers")
            tables.add("placement_answers")
        if "placement_answers" in tables:
            existing = {row[1] for row in db.execute("PRAGMA table_info(placement_answers)")}
            if "level" not in existing:
                db.execute("ALTER TABLE placement_answers ADD COLUMN level TEXT NOT NULL DEFAULT ''")
            if "session_id" not in existing:
                db.execute(
                    "ALTER TABLE placement_answers ADD COLUMN session_id INTEGER NOT NULL DEFAULT 1"
                )
            # ALTER TABLE не меняет первичный ключ. Версия 1 держала PK
            # (user_id, question_id), из-за чего вторая диагностика падала бы на
            # UNIQUE constraint. Таблицу нужно пересобрать, сохранив строки.
            key = [row[1] for row in db.execute("PRAGMA table_info(placement_answers)") if row[5]]
            if "session_id" not in key:
                db.executescript(
                    """
                    CREATE TABLE placement_answers_v2 (
                        user_id INTEGER NOT NULL,
                        question_id TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        selected_index INTEGER,
                        is_correct INTEGER,
                        level TEXT NOT NULL DEFAULT '',
                        session_id INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, session_id, question_id)
                    );
                    INSERT INTO placement_answers_v2(
                        user_id, question_id, answer_text, selected_index,
                        is_correct, level, session_id, created_at
                    )
                    SELECT user_id, question_id, answer_text, selected_index,
                           is_correct, level, session_id, created_at
                    FROM placement_answers;
                    DROP TABLE placement_answers;
                    ALTER TABLE placement_answers_v2 RENAME TO placement_answers;
                    """
                )
        if "users" in tables:
            existing = {row[1] for row in db.execute("PRAGMA table_info(users)")}
            additions = {
                "role": "TEXT NOT NULL DEFAULT 'member'",
                "display_name": "TEXT NOT NULL DEFAULT ''",
                "level": "TEXT NOT NULL DEFAULT ''",
                "target_level": "TEXT NOT NULL DEFAULT ''",
                "state_data": "TEXT NOT NULL DEFAULT '{}'",
                "streak_days": "INTEGER NOT NULL DEFAULT 0",
                "last_practice_day": "TEXT",
                "ai_calls_day": "TEXT",
                "ai_calls_today": "INTEGER NOT NULL DEFAULT 0",
                "share_progress": "INTEGER NOT NULL DEFAULT 1",
                "created_at": "TEXT",
                "last_active_at": "TEXT",
            }
            for column, definition in additions.items():
                if column not in existing:
                    db.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
            owner = db.execute("SELECT value FROM settings WHERE key = 'owner_user_id'").fetchone()
            if owner is not None:
                db.execute(
                    "UPDATE users SET role = 'owner' WHERE user_id = ?", (int(owner["value"]),)
                )
        if "voice_messages" in tables:
            existing = {row[1] for row in db.execute("PRAGMA table_info(voice_messages)")}
            for column, definition in {
                "transcript": "TEXT NOT NULL DEFAULT ''",
                "feedback": "TEXT NOT NULL DEFAULT ''",
                "words": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if column not in existing:
                    db.execute(f"ALTER TABLE voice_messages ADD COLUMN {column} {definition}")

    def _migrate_to_v3(self, db: sqlite3.Connection) -> None:
        """Добавляет деление на слоги в кэш произношений."""
        tables = {
            row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "pronunciations" not in tables:
            return
        existing = {row[1] for row in db.execute("PRAGMA table_info(pronunciations)")}
        if "syllables" not in existing:
            db.execute("ALTER TABLE pronunciations ADD COLUMN syllables TEXT NOT NULL DEFAULT ''")

    # ── настройки ────────────────────────────────────────────────

    def get_setting(self, key: str) -> str | None:
        with self.session() as db:
            row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self.session() as db:
            db.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ── пользователи ─────────────────────────────────────────────

    def _row_to_user(self, row: sqlite3.Row) -> User:
        try:
            state_data = json.loads(row["state_data"] or "{}")
        except json.JSONDecodeError:
            state_data = {}
        return User(
            user_id=int(row["user_id"]),
            chat_id=int(row["chat_id"]),
            role=str(row["role"] or "member"),
            display_name=str(row["display_name"] or ""),
            level=str(row["level"] or ""),
            target_level=str(row["target_level"] or ""),
            state=str(row["state"] or "idle"),
            state_data=state_data if isinstance(state_data, dict) else {},
            streak_days=int(row["streak_days"] or 0),
            last_practice_day=row["last_practice_day"],
            share_progress=bool(row["share_progress"]),
            created_at=row["created_at"],
            last_active_at=row["last_active_at"],
        )

    def user(self, user_id: int) -> User | None:
        with self.session() as db:
            row = db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return self._row_to_user(row) if row else None

    def create_user(
        self, user_id: int, chat_id: int, role: str = "member", display_name: str = ""
    ) -> User:
        with self.session() as db:
            db.execute(
                "INSERT INTO users(user_id, chat_id, role, display_name, created_at, "
                "last_active_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET chat_id = excluded.chat_id",
                (user_id, chat_id, role, display_name, utc_now(), utc_now()),
            )
        user = self.user(user_id)
        assert user is not None
        return user

    def touch(self, user_id: int, chat_id: int, display_name: str = "") -> None:
        """Отмечает активность и подтягивает имя из профиля Telegram.

        Имя не спрашивается отдельно и не хранится «навсегда»: оно перезаписывается
        при каждом сообщении, поэтому в таблице отдела всегда актуальное.
        """
        with self.session() as db:
            if display_name:
                db.execute(
                    "UPDATE users SET chat_id = ?, display_name = ?, last_active_at = ? "
                    "WHERE user_id = ?",
                    (chat_id, display_name, utc_now(), user_id),
                )
            else:
                db.execute(
                    "UPDATE users SET chat_id = ?, last_active_at = ? WHERE user_id = ?",
                    (chat_id, utc_now(), user_id),
                )

    def update_user(self, user_id: int, **fields: Any) -> None:
        allowed = {
            "role", "display_name", "level", "target_level", "state",
            "streak_days", "last_practice_day", "share_progress", "chat_id",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.session() as db:
            db.execute(
                f"UPDATE users SET {assignments} WHERE user_id = ?",
                (*updates.values(), user_id),
            )

    def set_state(self, user_id: int, state: str, data: dict[str, Any] | None = None) -> None:
        with self.session() as db:
            db.execute(
                "UPDATE users SET state = ?, state_data = ? WHERE user_id = ?",
                (state, json.dumps(data or {}, ensure_ascii=False), user_id),
            )

    def all_users(self) -> list[User]:
        with self.session() as db:
            rows = db.execute("SELECT * FROM users ORDER BY created_at, user_id").fetchall()
        return [self._row_to_user(row) for row in rows]

    def count_users(self) -> int:
        with self.session() as db:
            return int(db.execute("SELECT count(*) FROM users").fetchone()[0])

    # ── лимит вызовов ИИ ─────────────────────────────────────────

    def take_ai_call(self, user_id: int, limit: int) -> bool:
        """Атомарно списывает один вызов ИИ за сутки. False — лимит исчерпан."""
        stamp = today()
        with self.session() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT ai_calls_day, ai_calls_today FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                return False
            used = 0 if row["ai_calls_day"] != stamp else int(row["ai_calls_today"] or 0)
            if used >= limit:
                return False
            db.execute(
                "UPDATE users SET ai_calls_day = ?, ai_calls_today = ? WHERE user_id = ?",
                (stamp, used + 1, user_id),
            )
        return True

    def ai_calls_used(self, user_id: int) -> int:
        with self.session() as db:
            row = db.execute(
                "SELECT ai_calls_day, ai_calls_today FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None or row["ai_calls_day"] != today():
            return 0
        return int(row["ai_calls_today"] or 0)

    # ── приглашения ──────────────────────────────────────────────

    def create_invite(self, created_by: int, role: str = "member", days: int = 14) -> str:
        code = secrets.token_urlsafe(9)
        expires = (datetime.now(UTC) + timedelta(days=days)).isoformat(timespec="seconds")
        with self.session() as db:
            db.execute(
                "INSERT INTO invites(code, role, created_by, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (code, role, created_by, utc_now(), expires),
            )
        return code

    def redeem_invite(self, code: str, user_id: int) -> str | None:
        """Возвращает роль при успехе, иначе None. Код одноразовый."""
        with self.session() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM invites WHERE code = ?", (code,)).fetchone()
            if row is None or row["used_by"] is not None:
                return None
            expires = parse_ts(row["expires_at"])
            if expires is not None and expires < datetime.now(UTC):
                return None
            db.execute(
                "UPDATE invites SET used_by = ?, used_at = ? WHERE code = ?",
                (user_id, utc_now(), code),
            )
            return str(row["role"])

    def invites(self, created_by: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM invites"
        params: tuple[Any, ...] = ()
        if created_by is not None:
            query += " WHERE created_by = ?"
            params = (created_by,)
        query += " ORDER BY created_at DESC LIMIT 50"
        with self.session() as db:
            return [dict(row) for row in db.execute(query, params).fetchall()]

    # ── диагностика ──────────────────────────────────────────────

    def next_placement_session(self, user_id: int) -> int:
        with self.session() as db:
            row = db.execute(
                "SELECT max(session_id) AS last FROM placement_answers WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int((row["last"] or 0)) + 1

    def save_placement_answer(
        self,
        user_id: int,
        session_id: int,
        question_id: str,
        answer_text: str,
        selected_index: int | None,
        is_correct: bool | None,
        level: str,
    ) -> None:
        with self.session() as db:
            db.execute(
                """
                INSERT INTO placement_answers(
                    user_id, question_id, answer_text, selected_index,
                    is_correct, level, session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id, question_id) DO UPDATE SET
                    answer_text = excluded.answer_text,
                    selected_index = excluded.selected_index,
                    is_correct = excluded.is_correct,
                    level = excluded.level,
                    created_at = excluded.created_at
                """,
                (
                    user_id, question_id, answer_text, selected_index,
                    None if is_correct is None else int(is_correct), level, session_id, utc_now(),
                ),
            )

    def placement_answers(self, user_id: int, session_id: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM placement_answers WHERE user_id = ?"
        params: list[Any] = [user_id]
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY session_id, created_at"
        with self.session() as db:
            return [dict(row) for row in db.execute(query, params).fetchall()]

    # ── попытки и мастерство ─────────────────────────────────────

    def record_attempt(
        self, user_id: int, exercise_id: str, point_id: str, level: str,
        is_correct: bool, answer_text: str,
    ) -> None:
        with self.session() as db:
            db.execute(
                "INSERT INTO attempts(user_id, exercise_id, point_id, level, is_correct, "
                "answer_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, exercise_id, point_id, level, int(is_correct), answer_text[:500], utc_now()),
            )

    def point_stats(self, user_id: int) -> dict[str, tuple[int, int]]:
        """point_id -> (верно, всего)."""
        with self.session() as db:
            rows = db.execute(
                "SELECT point_id, sum(is_correct) AS correct, count(*) AS total "
                "FROM attempts WHERE user_id = ? AND point_id != '' GROUP BY point_id",
                (user_id,),
            ).fetchall()
        return {str(row["point_id"]): (int(row["correct"] or 0), int(row["total"])) for row in rows}

    def recent_accuracy(self, user_id: int, limit: int = 12) -> float | None:
        with self.session() as db:
            rows = db.execute(
                "SELECT is_correct FROM attempts WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        if not rows:
            return None
        return sum(int(row["is_correct"]) for row in rows) / len(rows)

    def attempts_count(self, user_id: int) -> tuple[int, int]:
        with self.session() as db:
            row = db.execute(
                "SELECT count(*) AS total, sum(is_correct) AS correct FROM attempts WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["total"] or 0), int(row["correct"] or 0)

    def set_skill(self, user_id: int, skill: str, mastery: int, minutes_delta: int = 0) -> None:
        with self.session() as db:
            db.execute(
                "INSERT INTO skill_mastery(user_id, skill, mastery, minutes, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, skill) DO UPDATE SET "
                "mastery = excluded.mastery, minutes = skill_mastery.minutes + ?, "
                "updated_at = excluded.updated_at",
                (user_id, skill, mastery, minutes_delta, utc_now(), minutes_delta),
            )

    def skills(self, user_id: int) -> dict[str, tuple[int, int]]:
        with self.session() as db:
            rows = db.execute(
                "SELECT skill, mastery, minutes FROM skill_mastery WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {str(row["skill"]): (int(row["mastery"]), int(row["minutes"])) for row in rows}

    # ── интервальное повторение ──────────────────────────────────

    def card(self, user_id: int, card_type: str, card_key: str) -> Card | None:
        with self.session() as db:
            row = db.execute(
                "SELECT * FROM srs_cards WHERE user_id = ? AND card_type = ? AND card_key = ?",
                (user_id, card_type, card_key),
            ).fetchone()
        if row is None:
            return None
        return Card(
            card_type=str(row["card_type"]),
            card_key=str(row["card_key"]),
            ease=float(row["ease"]),
            interval_days=float(row["interval_days"]),
            repetitions=int(row["repetitions"]),
            lapses=int(row["lapses"]),
            due_at=str(row["due_at"]),
            mastery=int(row["mastery"]),
        )

    def upsert_card(self, user_id: int, card: Card) -> None:
        with self.session() as db:
            db.execute(
                """
                INSERT INTO srs_cards(
                    user_id, card_type, card_key, ease, interval_days, repetitions,
                    lapses, mastery, due_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, card_type, card_key) DO UPDATE SET
                    ease = excluded.ease,
                    interval_days = excluded.interval_days,
                    repetitions = excluded.repetitions,
                    lapses = excluded.lapses,
                    mastery = excluded.mastery,
                    due_at = excluded.due_at,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id, card.card_type, card.card_key, card.ease, card.interval_days,
                    card.repetitions, card.lapses, card.mastery, card.due_at, utc_now(), utc_now(),
                ),
            )

    def due_cards(self, user_id: int, card_type: str | None = None, limit: int = 40) -> list[Card]:
        query = "SELECT * FROM srs_cards WHERE user_id = ? AND due_at <= ?"
        params: list[Any] = [user_id, utc_now()]
        if card_type:
            query += " AND card_type = ?"
            params.append(card_type)
        query += " ORDER BY due_at LIMIT ?"
        params.append(limit)
        with self.session() as db:
            rows = db.execute(query, params).fetchall()
        return [
            Card(
                card_type=str(row["card_type"]), card_key=str(row["card_key"]),
                ease=float(row["ease"]), interval_days=float(row["interval_days"]),
                repetitions=int(row["repetitions"]), lapses=int(row["lapses"]),
                due_at=str(row["due_at"]), mastery=int(row["mastery"]),
            )
            for row in rows
        ]

    def card_counts(self, user_id: int) -> dict[str, tuple[int, int]]:
        """card_type -> (всего, к повторению сейчас)."""
        with self.session() as db:
            rows = db.execute(
                "SELECT card_type, count(*) AS total, "
                "sum(CASE WHEN due_at <= ? THEN 1 ELSE 0 END) AS due "
                "FROM srs_cards WHERE user_id = ? GROUP BY card_type",
                (utc_now(), user_id),
            ).fetchall()
        return {str(row["card_type"]): (int(row["total"]), int(row["due"] or 0)) for row in rows}

    def vocab_card_keys(self, user_id: int) -> set[str]:
        """Все слова, по которым уже заведена карточка, — чтобы не выдавать их снова как новые."""
        with self.session() as db:
            rows = db.execute(
                "SELECT card_key FROM srs_cards WHERE user_id = ? AND card_type = 'vocab'",
                (user_id,),
            ).fetchall()
        return {str(row["card_key"]) for row in rows}

    def mastered_vocab(self, user_id: int, minimum: int = 1) -> list[str]:
        with self.session() as db:
            rows = db.execute(
                "SELECT card_key FROM srs_cards WHERE user_id = ? AND card_type = 'vocab' "
                "AND mastery >= ? ORDER BY card_key",
                (user_id, minimum),
            ).fetchall()
        return [str(row["card_key"]) for row in rows]

    # ── серия дней ───────────────────────────────────────────────

    def bump_streak(self, user_id: int) -> int:
        stamp = today()
        with self.session() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT streak_days, last_practice_day FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                return 0
            last = row["last_practice_day"]
            streak = int(row["streak_days"] or 0)
            if last == stamp:
                return streak
            yesterday = (date.fromisoformat(stamp) - timedelta(days=1)).isoformat()
            streak = streak + 1 if last == yesterday else 1
            db.execute(
                "UPDATE users SET streak_days = ?, last_practice_day = ? WHERE user_id = ?",
                (streak, stamp, user_id),
            )
        return streak

    # ── диалог ───────────────────────────────────────────────────

    def add_message(self, user_id: int, role: str, content: str) -> None:
        with self.session() as db:
            db.execute(
                "INSERT INTO messages(user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (user_id, role, content, utc_now()),
            )

    def recent_messages(self, user_id: int, limit: int = 12) -> list[dict[str, str]]:
        with self.session() as db:
            rows = db.execute(
                "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def trim_messages(self, user_id: int, keep: int = 200) -> None:
        with self.session() as db:
            db.execute(
                "DELETE FROM messages WHERE user_id = ? AND id NOT IN "
                "(SELECT id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
                (user_id, user_id, keep),
            )

    # ── голос ────────────────────────────────────────────────────

    def add_voice(
        self, user_id: int, telegram_message_id: int, file_id: str, file_unique_id: str,
        duration_seconds: int, local_path: Path, task_id: str,
    ) -> None:
        with self.session() as db:
            db.execute(
                """
                INSERT INTO voice_messages(
                    user_id, telegram_message_id, file_id, file_unique_id,
                    duration_seconds, local_path, task_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, telegram_message_id) DO UPDATE SET
                    file_id = excluded.file_id,
                    file_unique_id = excluded.file_unique_id,
                    duration_seconds = excluded.duration_seconds,
                    local_path = excluded.local_path,
                    task_id = excluded.task_id,
                    created_at = excluded.created_at
                """,
                (
                    user_id, telegram_message_id, file_id, file_unique_id,
                    duration_seconds, str(local_path), task_id, utc_now(),
                ),
            )

    def set_voice_result(
        self, user_id: int, telegram_message_id: int, transcript: str, feedback: str, words: int
    ) -> None:
        with self.session() as db:
            db.execute(
                "UPDATE voice_messages SET transcript = ?, feedback = ?, words = ? "
                "WHERE user_id = ? AND telegram_message_id = ?",
                (transcript, feedback, words, user_id, telegram_message_id),
            )

    def voices(self, user_id: int) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(
                "SELECT telegram_message_id, duration_seconds, local_path, task_id, "
                "transcript, feedback, words, created_at "
                "FROM voice_messages WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def voice_paths(self, user_id: int) -> list[Path]:
        return [Path(str(row["local_path"])) for row in self.voices(user_id)]

    # ── журнал ошибок ────────────────────────────────────────────

    def log_error(
        self, user_id: int, category: str, original: str, corrected: str,
        note: str = "", pattern_id: str = "", source: str = "chat",
    ) -> None:
        with self.session() as db:
            db.execute(
                "INSERT INTO error_log(user_id, pattern_id, category, original, corrected, "
                "note, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, pattern_id, category, original[:400], corrected[:400],
                 note[:400], source, utc_now()),
            )

    def error_summary(self, user_id: int, limit: int = 12) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(
                "SELECT category, pattern_id, count(*) AS times, max(created_at) AS last_seen "
                "FROM error_log WHERE user_id = ? GROUP BY category, pattern_id "
                "ORDER BY times DESC, last_seen DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_errors(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(
                "SELECT * FROM error_log WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ── письмо ───────────────────────────────────────────────────

    def add_writing(
        self, user_id: int, task_id: str, text: str, scores: dict[str, Any], feedback: str
    ) -> None:
        with self.session() as db:
            db.execute(
                "INSERT INTO writing_submissions(user_id, task_id, text, scores, feedback, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, task_id, text, json.dumps(scores, ensure_ascii=False), feedback, utc_now()),
            )

    def writings(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(
                "SELECT * FROM writing_submissions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ── произношение ─────────────────────────────────────────────

    def pronunciation(self, word: str) -> dict[str, Any] | None:
        with self.session() as db:
            row = db.execute(
                "SELECT * FROM pronunciations WHERE word = ?", (word.strip().lower(),)
            ).fetchone()
        return dict(row) if row else None

    def save_pronunciation(
        self,
        word: str,
        display: str,
        ipa: str,
        note: str,
        audio_path: str = "",
        file_id: str = "",
        syllables: str = "",
    ) -> None:
        with self.session() as db:
            db.execute(
                """
                INSERT INTO pronunciations(
                    word, display, ipa, syllables, note, audio_path, file_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(word) DO UPDATE SET
                    display = excluded.display,
                    ipa = excluded.ipa,
                    syllables = CASE WHEN excluded.syllables != '' THEN excluded.syllables
                                     ELSE pronunciations.syllables END,
                    note = excluded.note,
                    audio_path = CASE WHEN excluded.audio_path != '' THEN excluded.audio_path
                                      ELSE pronunciations.audio_path END,
                    file_id = CASE WHEN excluded.file_id != '' THEN excluded.file_id
                                   ELSE pronunciations.file_id END
                """,
                (
                    word.strip().lower(), display, ipa, syllables, note,
                    audio_path, file_id, utc_now(),
                ),
            )

    # ── сессии ───────────────────────────────────────────────────

    def start_session(self, user_id: int, kind: str, subject: str = "") -> int:
        with self.session() as db:
            cursor = db.execute(
                "INSERT INTO sessions(user_id, kind, subject, started_at) VALUES (?, ?, ?, ?)",
                (user_id, kind, subject, utc_now()),
            )
            return int(cursor.lastrowid or 0)

    def finish_session(self, session_id: int, items: int, correct: int) -> None:
        with self.session() as db:
            db.execute(
                "UPDATE sessions SET items = ?, correct = ?, finished_at = ? WHERE id = ?",
                (items, correct, utc_now(), session_id),
            )

    def sessions(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with self.session() as db:
            rows = db.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND finished_at IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def completed_session_subjects(self, user_id: int, kind: str) -> set[str]:
        """Уже выполненные задания данного типа — для ротации банка."""
        with self.session() as db:
            rows = db.execute(
                "SELECT DISTINCT subject FROM sessions WHERE user_id = ? AND kind = ? "
                "AND finished_at IS NOT NULL AND items > 0",
                (user_id, kind),
            ).fetchall()
        return {str(row["subject"]) for row in rows if row["subject"]}

    def session_totals(self, user_id: int, kind: str) -> tuple[int, int]:
        """(всего, верно) по завершённым сессиям одного типа."""
        with self.session() as db:
            row = db.execute(
                "SELECT sum(items) AS total, sum(correct) AS correct FROM sessions "
                "WHERE user_id = ? AND kind = ? AND finished_at IS NOT NULL",
                (user_id, kind),
            ).fetchone()
        return int(row["total"] or 0), int(row["correct"] or 0)

    def days_since_session(self, user_id: int, kind: str) -> int | None:
        """Дней с последней завершённой сессии типа; None — её ещё не было."""
        with self.session() as db:
            row = db.execute(
                "SELECT max(finished_at) AS last FROM sessions "
                "WHERE user_id = ? AND kind = ? AND finished_at IS NOT NULL",
                (user_id, kind),
            ).fetchone()
        last = parse_ts(row["last"] if row else None)
        if last is None:
            return None
        return max(0, (datetime.now(UTC) - last).days)

    def practiced_today(self, user_id: int) -> bool:
        """Была ли сегодня хоть одна завершённая тренировка или повторение."""
        with self.session() as db:
            row = db.execute(
                "SELECT 1 FROM sessions WHERE user_id = ? AND finished_at IS NOT NULL "
                "AND items > 0 AND substr(started_at, 1, 10) = ? LIMIT 1",
                (user_id, today()),
            ).fetchone()
        return row is not None

    def days_since_speaking(self, user_id: int) -> int | None:
        """Сколько дней прошло с последнего голосового. None — их вообще не было."""
        with self.session() as db:
            row = db.execute(
                "SELECT max(created_at) AS last FROM voice_messages WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        last = parse_ts(row["last"] if row else None)
        if last is None:
            return None
        return max(0, (datetime.now(UTC) - last).days)

    def team_stats(self) -> list[dict[str, Any]]:
        """Сводка по команде: только те, кто не отключил обмен прогрессом."""
        with self.session() as db:
            rows = db.execute(
                """
                SELECT u.user_id, u.display_name, u.level, u.streak_days,
                       (SELECT count(*) FROM attempts a WHERE a.user_id = u.user_id) AS attempts,
                       (SELECT sum(is_correct) FROM attempts a WHERE a.user_id = u.user_id) AS correct,
                       (SELECT count(*) FROM srs_cards c
                        WHERE c.user_id = u.user_id AND c.mastery >= 3) AS mastered
                FROM users u
                WHERE u.share_progress = 1
                ORDER BY attempts DESC, u.user_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    # ── удаление ─────────────────────────────────────────────────

    def delete_learning_data(self, user_id: int) -> None:
        """Удаляет учебные данные, но сохраняет саму учётку и её роль."""
        with self.session() as db:
            for table in (
                "placement_answers", "attempts", "srs_cards", "skill_mastery", "messages",
                "voice_messages", "error_log", "writing_submissions", "sessions",
            ):
                db.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            db.execute(
                "UPDATE users SET state = 'idle', state_data = '{}', level = '', "
                "target_level = '', streak_days = 0, last_practice_day = NULL WHERE user_id = ?",
                (user_id,),
            )

    def delete_user(self, user_id: int) -> None:
        self.delete_learning_data(user_id)
        with self.session() as db:
            db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
