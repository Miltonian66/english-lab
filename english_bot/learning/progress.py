"""Прогресс, учебный план, экспорт и выгрузка в Anki."""

from __future__ import annotations

from pathlib import Path

from ..content.registry import Curriculum
from ..content.schema import LEVEL_ORDER, LEVELS, GrammarPoint
from ..storage import Storage, User
from .placement import next_level
from .srs import stars


MASTERED = 3


def mastery_map(storage: Storage, user_id: int) -> dict[str, int]:
    """point_id -> звёзды 0–5 по данным интервального повторения."""
    with storage.session() as db:
        rows = db.execute(
            "SELECT card_key, mastery FROM srs_cards WHERE user_id = ? AND card_type = 'point'",
            (user_id,),
        ).fetchall()
    return {str(row["card_key"]): int(row["mastery"]) for row in rows}


def unmastered_points(
    curriculum: Curriculum, mastery: dict[str, int], level: str, limit: int = 6
) -> list[GrammarPoint]:
    """Следующие пункты уровня, которые ещё не закрыты."""
    rows = [
        point
        for point in curriculum.points_of_level(level)
        if mastery.get(point.id, 0) < MASTERED
    ]
    return rows[:limit]


def level_progress(
    curriculum: Curriculum, mastery: dict[str, int], level: str
) -> tuple[int, int]:
    points = curriculum.points_of_level(level)
    done = sum(1 for point in points if mastery.get(point.id, 0) >= MASTERED)
    return done, len(points)


def progress_report(storage: Storage, curriculum: Curriculum, user: User) -> str:
    mastery = mastery_map(storage, user.user_id)
    level = user.level or "не определён"
    total_attempts, correct_attempts = storage.attempts_count(user.user_id)
    counts = storage.card_counts(user.user_id)

    lines = [f"Профиль: {user.display_name or 'без имени'}"]
    lines.append(f"Уровень: {level} → цель {user.target_level or '—'}")
    lines.append(f"Серия: {user.streak_days} дн. подряд")

    if total_attempts:
        share = round(correct_attempts * 100 / total_attempts)
        lines.append(f"Заданий решено: {total_attempts}, верно {share}%")
    else:
        lines.append("Заданий пока не решено — нажми «🎯 Заниматься»")

    if user.level:
        done, total = level_progress(curriculum, mastery, user.level)
        if total:
            bar_filled = round(done * 10 / total)
            bar = "▰" * bar_filled + "▱" * (10 - bar_filled)
            lines.append(f"Уровень {user.level}: {bar} {done}/{total} тем закрыто")

    point_total, point_due = counts.get("point", (0, 0))
    vocab_total, vocab_due = counts.get("vocab", (0, 0))
    lines.append("")
    lines.append(f"Карточки грамматики: {point_total}, к повторению {point_due}")
    lines.append(f"Карточки лексики: {vocab_total}, к повторению {vocab_due}")

    errors = storage.error_summary(user.user_id, limit=5)
    if errors:
        lines.append("")
        lines.append("Повторяющиеся ошибки:")
        for row in errors:
            pattern = curriculum.error_patterns.get(str(row["pattern_id"] or ""))
            label = pattern.label_ru if pattern else str(row["category"])
            lines.append(f"• {label} — {row['times']} раз")

    if user.level:
        weak = [
            point
            for point in curriculum.points_of_level(user.level)
            if 0 < mastery.get(point.id, 0) < MASTERED
        ][:5]
        if weak:
            lines.append("")
            lines.append("Начато, но не закрыто:")
            for point in weak:
                lines.append(f"• {stars(mastery.get(point.id, 0))} {point.title_ru}")

    return "\n".join(lines)


def study_plan(storage: Storage, curriculum: Curriculum, user: User) -> str:
    level = user.level or "A2"
    target = user.target_level or next_level(level)
    mastery = mastery_map(storage, user.user_id)

    lines = [f"План: {level} → {target}"]
    done, total = level_progress(curriculum, mastery, level)
    lines.append(f"Закрыто тем уровня {level}: {done} из {total}.")
    lines.append("")

    upcoming = unmastered_points(curriculum, mastery, level, limit=6)
    if upcoming:
        lines.append(f"Ближайшие темы уровня {level}:")
        for index, point in enumerate(upcoming, 1):
            lines.append(f"{index}. {point.title_ru} — {point.topic}")
    else:
        lines.append(f"Уровень {level} закрыт. Переходим к {target}.")
        for index, point in enumerate(
            unmastered_points(curriculum, mastery, target, limit=6), 1
        ):
            lines.append(f"{index}. {point.title_ru} — {point.topic}")

    lines.append("")
    lines.append("Недельный ритм:")
    lines.append("• 4 дня — «🎯 Заниматься», 10–12 заданий по теме дня")
    lines.append("• каждый день — повторение, пока очередь не пуста")
    lines.append("• 2 раза — «🎙 Речь», голосом, без чтения с листа")
    lines.append("• 1 раз — «✍️ Письмо», потом разбор правок")
    lines.append("• незнакомое слово — /say, чтобы сразу поставить произношение")
    return "\n".join(lines)


def build_export(storage: Storage, curriculum: Curriculum, user: User) -> str:
    """Markdown-срез данных ученика: и для человека, и для внешнего разбора."""
    mastery = mastery_map(storage, user.user_id)
    lines = [
        f"# English Lab — учебные данные (Telegram id {user.user_id})",
        "",
        f"- Уровень: {user.level or 'не определён'}, цель {user.target_level or '—'}",
        f"- Роль: {user.role}",
        f"- Серия: {user.streak_days} дн.",
        f"- Обновлено: {storage.get_setting(f'last_export:{user.user_id}') or '—'}",
        "",
        "## Прогресс по уровням",
        "",
    ]
    for level in LEVELS:
        points = curriculum.points_of_level(level)
        if not points:
            continue
        done = sum(1 for point in points if mastery.get(point.id, 0) >= MASTERED)
        lines.append(f"- {level}: {done}/{len(points)}")

    sessions = storage.sessions(user.user_id, limit=15)
    if sessions:
        lines.extend(["", "## Последние сессии", ""])
        for row in sessions:
            lines.append(
                f"- {row['started_at']} · {row['kind']} · {row['subject']} · "
                f"{row['correct']}/{row['items']}"
            )

    errors = storage.error_summary(user.user_id, limit=15)
    if errors:
        lines.extend(["", "## Частые ошибки", ""])
        for row in errors:
            pattern = curriculum.error_patterns.get(str(row["pattern_id"] or ""))
            label = pattern.label_ru if pattern else str(row["category"])
            lines.append(f"- {label}: {row['times']}")

    writings = storage.writings(user.user_id, limit=5)
    if writings:
        lines.extend(["", "## Письменные работы", ""])
        for row in writings:
            lines.append(f"- {row['created_at']} · {row['task_id']} · {row['scores']}")

    voices = storage.voices(user.user_id)
    if voices:
        lines.extend(["", "## Устные ответы", ""])
        for row in voices[-8:]:
            transcript = str(row.get("transcript") or "")
            preview = transcript[:160] + ("…" if len(transcript) > 160 else "")
            lines.append(
                f"- {row['created_at']} · {row['task_id']} · {row['duration_seconds']} с · "
                f"{row['words']} слов"
            )
            if preview:
                lines.append(f"  > {preview}")

    return "\n".join(lines)


def write_export(export_dir: Path, user_id: int, content: str) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"learner-{user_id}.md"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def anki_export(storage: Storage, curriculum: Curriculum, user_id: int) -> str:
    """TSV для импорта в Anki: лицо, оборот, теги. Разделитель — табуляция."""
    rows: list[str] = [
        "#separator:tab",
        "#html:false",
        "#tags column:3",
    ]
    known = set(storage.mastered_vocab(user_id, minimum=1))
    for level, items in curriculum.vocabulary.items():
        for item in items:
            if item.id not in known:
                continue
            back = f"{item.translation_ru} — {item.ipa_us} — {item.example_en}"
            rows.append(f"{item.word}\t{back}\tenglish-lab vocab {level}")

    for row in storage.recent_errors(user_id, limit=200):
        original = str(row["original"]).replace("\t", " ")
        corrected = str(row["corrected"]).replace("\t", " ")
        note = str(row["note"]).replace("\t", " ")
        if not original or not corrected:
            continue
        rows.append(
            f"Исправь: {original}\t{corrected}"
            + (f" — {note}" if note else "")
            + f"\tenglish-lab errors {row['category']}"
        )

    mastery = mastery_map(storage, user_id)
    for point_id, value in mastery.items():
        point = curriculum.point(point_id)
        if not point or value < 1:
            continue
        example = point.examples[0] if point.examples else ""
        rows.append(
            f"{point.title_en}\t{point.summary_ru[:300]} {example}"
            f"\tenglish-lab grammar {point.level}"
        )
    return "\n".join(rows) + "\n"


def team_board(storage: Storage, curriculum: Curriculum) -> str:
    rows = storage.team_stats()
    if not rows:
        return "В отделе пока никого. Пригласи коллег: «📊 Я» → «Пригласить коллегу»."
    lines = ["Команда English Lab", ""]
    ranked = sorted(rows, key=lambda row: (-(row["attempts"] or 0), row["user_id"]))
    for index, row in enumerate(ranked[:20], 1):
        name = str(row["display_name"] or f"Учащийся {row['user_id'] % 10000}")
        attempts = int(row["attempts"] or 0)
        correct = int(row["correct"] or 0)
        share = f"{round(correct * 100 / attempts)}%" if attempts else "—"
        level = str(row["level"] or "—")
        lines.append(
            f"{index}. {name} · {level} · {attempts} заданий · {share} · "
            f"серия {row['streak_days']} дн."
        )
    lines.append("")
    lines.append("Отключить себя из таблицы: /privacy")
    return "\n".join(lines)


def levels_overview(curriculum: Curriculum, mastery: dict[str, int]) -> list[tuple[str, str]]:
    """Кнопки уровней с долей закрытых тем."""
    buttons: list[tuple[str, str]] = []
    for level in LEVELS:
        points = curriculum.points_of_level(level)
        if not points:
            continue
        done = sum(1 for point in points if mastery.get(point.id, 0) >= MASTERED)
        buttons.append((f"{level} · {done}/{len(points)}", f"lvl:{level}"))
    return buttons


def sort_levels(levels: list[str]) -> list[str]:
    return sorted(levels, key=lambda name: LEVEL_ORDER.get(name, 99))
