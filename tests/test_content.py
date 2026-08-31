"""Инварианты учебного контента: схема, ключи ответов, покрытие уровней."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from english_bot.content.registry import DATA_DIR, load_curriculum
from english_bot.content.schema import LEVELS, validate_directory
from english_bot.content.validate import bank_of


CURRICULUM = load_curriculum()

# Символы, которых в General American не бывает вовсе.
BRITISH_IPA = re.compile(r"[ɒɐː]")
# Безэрность: слово пишется на -er/-or/-ar/-ure, а транскрипция кончается шва без r-окраски.
RHOTIC_SPELLING = re.compile(r"(er|or|ar|ure|our)$", re.IGNORECASE)
# Дифтонг — одно ядро, поэтому ищется раньше одиночных гласных.
VOWEL_NUCLEUS = re.compile(r"aɪ|aʊ|ɔɪ|eɪ|oʊ|[iɪeɛæɑʌəɝɚɔʊu]")


class ContentSchemaTests(unittest.TestCase):
    def test_directory_validates_without_errors(self) -> None:
        report = validate_directory(DATA_DIR, skip=lambda path: bank_of(path) is not None)
        self.assertEqual(report.errors, [], "\n".join(report.errors[:10]))
        self.assertGreater(report.points, 200)
        self.assertGreater(report.exercises, 1500)

    def test_no_load_errors(self) -> None:
        self.assertEqual(CURRICULUM.load_errors, [])

    def test_all_cefr_levels_present(self) -> None:
        self.assertEqual(CURRICULUM.levels(), list(LEVELS))
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertGreaterEqual(len(CURRICULUM.points_of_level(level)), 8)

    def test_every_point_has_exercises_and_topic(self) -> None:
        for point in CURRICULUM.points.values():
            with self.subTest(point=point.id):
                self.assertGreaterEqual(len(point.exercises), 6)
                self.assertTrue(point.topic)
                self.assertTrue(point.summary_ru)
                self.assertTrue(point.ru_interference)


class AnswerKeyTests(unittest.TestCase):
    def test_choice_answer_appears_once(self) -> None:
        """Верный вариант не должен дублироваться среди дистракторов."""
        for point in CURRICULUM.points.values():
            for exercise in point.exercises:
                if exercise.kind != "choice":
                    continue
                with self.subTest(exercise=exercise.id):
                    normalized = [option.strip().lower() for option in exercise.options]
                    self.assertEqual(len(set(normalized)), len(normalized))
                    self.assertIsNotNone(exercise.correct_index)

    def test_free_answers_are_not_empty(self) -> None:
        for point in CURRICULUM.points.values():
            for exercise in point.exercises:
                if exercise.kind == "choice":
                    continue
                with self.subTest(exercise=exercise.id):
                    self.assertTrue(exercise.answer.strip())
                    self.assertNotIn("___", exercise.answer)

    def test_exercise_ids_are_globally_unique(self) -> None:
        seen: set[str] = set()
        for point in CURRICULUM.points.values():
            for exercise in point.exercises:
                self.assertNotIn(exercise.id, seen)
                seen.add(exercise.id)

    def test_prerequisites_point_to_existing_points(self) -> None:
        missing: list[str] = []
        for point in CURRICULUM.points.values():
            for prerequisite in point.prerequisites:
                if prerequisite not in CURRICULUM.points:
                    missing.append(f"{point.id} → {prerequisite}")
        self.assertEqual(missing, [], f"битые prerequisites: {missing[:10]}")


class HintMaterialTests(unittest.TestCase):
    """У каждой темы должен быть разбор для подсказки — без исключений."""

    def test_every_point_produces_a_substantial_hint(self) -> None:
        from english_bot.learning.practice import point_help

        thin = []
        for point in CURRICULUM.points.values():
            text = point_help(point)
            if len(text) < 250 or "Как строится:" not in text or "Примеры:" not in text:
                thin.append(f"{point.id} ({len(text)} символов)")
        self.assertEqual(thin, [], f"без полноценного разбора: {thin[:10]}")

    def test_hint_carries_forms_examples_and_the_russian_trap(self) -> None:
        from english_bot.learning.practice import point_help

        for point in list(CURRICULUM.points.values())[:40]:
            with self.subTest(point=point.id):
                text = point_help(point)
                self.assertIn(point.title_ru, text)
                self.assertIn(point.forms[0], text)
                self.assertIn(point.examples[0], text)
                self.assertIn("Ловушка для русскоязычных", text)

    def test_vocabulary_hint_hides_the_word_and_its_forms(self) -> None:
        from english_bot.learning.practice import vocab_help, word_forms

        leaked = []
        for items in CURRICULUM.vocabulary.values():
            for item in items:
                text = vocab_help(item).lower()
                for form in word_forms(item.word):
                    if re.search(rf"\b{re.escape(form)}\b", text):
                        leaked.append(f"{item.word} → {form}")
                        break
        self.assertEqual(leaked, [], f"слово видно в подсказке: {leaked[:10]}")


class CallbackCodeTests(unittest.TestCase):
    def test_point_codes_are_unique_and_short(self) -> None:
        codes = {CURRICULUM.point_code(point_id) for point_id in CURRICULUM.points}
        self.assertEqual(len(codes), len(CURRICULUM.points))
        for point_id in CURRICULUM.points:
            self.assertLessEqual(len(f"pt:{CURRICULUM.point_code(point_id)}"), 64)

    def test_topic_codes_round_trip(self) -> None:
        for level in CURRICULUM.levels():
            for topic, _ in CURRICULUM.topics_of_level(level):
                code = CURRICULUM.topic_code(level, topic)
                self.assertEqual(CURRICULUM.topic_by_code(level, code), topic)


class BankTests(unittest.TestCase):
    def test_vocabulary_covers_all_levels(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertGreaterEqual(len(CURRICULUM.vocab_of_level(level)), 100)

    def test_vocabulary_has_no_cross_level_duplicates(self) -> None:
        seen: dict[str, str] = {}
        for level, items in CURRICULUM.vocabulary.items():
            for item in items:
                word = item.word.lower()
                self.assertNotIn(word, seen, f"{word} есть и в {seen.get(word)}, и в {level}")
                seen[word] = level

    def test_ipa_uses_no_british_only_symbols(self) -> None:
        offenders = [
            f"{item.word} {item.ipa_us}"
            for items in CURRICULUM.vocabulary.values()
            for item in items
            if BRITISH_IPA.search(item.ipa_us.strip("/"))
        ]
        self.assertEqual(offenders, [], f"британские символы: {offenders[:10]}")

    def test_ipa_keeps_final_r_colouring(self) -> None:
        """teacher = /ˈtitʃɚ/, а не безэрное /ˈtiːtʃə/."""
        offenders: list[str] = []
        for items in CURRICULUM.vocabulary.values():
            for item in items:
                body = item.ipa_us.strip("/").rstrip()
                if not RHOTIC_SPELLING.search(item.word.split()[-1]):
                    continue
                if body.endswith("ə"):
                    offenders.append(f"{item.word} {item.ipa_us}")
        self.assertEqual(offenders, [], f"безэрное окончание: {offenders[:10]}")

    def test_ipa_marks_stress_for_multisyllable_words(self) -> None:
        missing: list[str] = []
        for items in CURRICULUM.vocabulary.values():
            for item in items:
                if " " in item.word or "-" in item.word:
                    continue
                vowels = len(VOWEL_NUCLEUS.findall(item.ipa_us))
                if vowels >= 2 and "ˈ" not in item.ipa_us:
                    missing.append(f"{item.word} {item.ipa_us}")
        self.assertEqual(missing, [], f"нет знака ударения: {missing[:10]}")

    def test_speaking_and_writing_cover_all_levels(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertGreaterEqual(len(CURRICULUM.speaking_of_level(level)), 5)
                self.assertGreaterEqual(len(CURRICULUM.writing_of_level(level)), 3)

    def test_error_patterns_explain_russian_interference(self) -> None:
        self.assertGreaterEqual(len(CURRICULUM.error_patterns), 30)
        for pattern in CURRICULUM.error_patterns.values():
            with self.subTest(pattern=pattern.id):
                self.assertTrue(pattern.trigger_ru.strip())
                self.assertNotEqual(pattern.wrong_en, pattern.right_en)

    def test_sounds_have_minimal_pairs(self) -> None:
        self.assertGreaterEqual(len(CURRICULUM.sounds), 15)
        for note in CURRICULUM.sounds:
            with self.subTest(sound=note.id):
                self.assertTrue(note.advice_ru.strip())
                for pair in note.minimal_pairs:
                    self.assertIn("/", pair)


class CatalogTests(unittest.TestCase):
    def test_catalog_matches_loaded_points(self) -> None:
        """Каталог, снятый со структуры test-english.com, должен быть покрыт контентом."""
        import json

        catalog_path = Path(__file__).resolve().parent.parent / (
            "english_bot/content/catalog.json"
        )
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        missing = [row["id"] for row in catalog if row["id"] not in CURRICULUM.points]
        self.assertEqual(missing, [], f"нет контента для пунктов каталога: {missing[:10]}")


if __name__ == "__main__":
    unittest.main()
