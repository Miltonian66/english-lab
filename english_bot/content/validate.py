"""CLI-проверка учебного контента: `python3 -m english_bot.content.validate`."""

from __future__ import annotations

import sys
from pathlib import Path

from .banks import bank_of_stem, parse_bank
from .schema import ContentError, ValidationReport, parse_file, validate_directory


def bank_of(path: Path) -> str | None:
    """Определяет банк по имени файла: vocabulary_b1.json -> vocabulary."""
    return bank_of_stem(path.stem)


def validate_single(path: Path) -> ValidationReport:
    report = ValidationReport(files=1)
    bank = bank_of(path)
    if bank is not None:
        try:
            rows = parse_bank(path, bank)
        except ContentError as exc:
            report.errors.append(str(exc))
            return report
        report.points = len(rows)
        return report
    try:
        points = parse_file(path)
    except ContentError as exc:
        report.errors.append(str(exc))
        return report
    report.points = len(points)
    report.exercises = sum(len(point.exercises) for point in points)
    return report


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent / "data"
    if not target.exists():
        print(f"Путь не найден: {target}", file=sys.stderr)
        return 2
    if target.is_file():
        report = validate_single(target)
    else:
        report = validate_directory(target, skip=lambda path: bank_of(path) is not None)
        for path in sorted(target.glob("*.json")):
            if bank_of(path) is None:
                continue
            bank_report = validate_single(path)
            report.errors.extend(bank_report.errors)
    print(
        f"файлов: {report.files}, пунктов: {report.points}, упражнений: {report.exercises}, "
        f"ошибок: {len(report.errors)}"
    )
    for error in report.errors[:60]:
        print(f"  ✗ {error}")
    if len(report.errors) > 60:
        print(f"  … и ещё {len(report.errors) - 60}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
