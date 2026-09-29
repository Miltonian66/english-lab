"""Инварианты учебного контента: схема, ключи ответов, покрытие уровней."""

from __future__ import annotations

import json
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
                    if exercise.kind == "cloze":
                        self.assertTrue(all(gap and gap[0].strip() for gap in exercise.gaps))
                        continue
                    self.assertTrue(exercise.answer.strip())
                    self.assertNotIn("___", exercise.answer)

    def test_every_exercise_accepts_its_own_answers(self) -> None:
        """Эталон и каждый вариант accept проходят настоящую сверку бота.

        Ловит accept с другим знаком в конце order, опечатку в эталоне и cloze,
        чьи ответы не соответствуют пропускам.
        """
        import random

        from english_bot.learning import practice as pr

        rng = random.Random(1)
        broken: list[str] = []
        for point in CURRICULUM.points.values():
            for exercise in point.exercises:
                question = pr.resolve(f"ex:{exercise.id}", CURRICULUM, rng)
                assert question is not None
                if exercise.kind == "cloze":
                    answers = ["; ".join(gap[0] for gap in exercise.gaps),
                               "; ".join(gap[-1] for gap in exercise.gaps)]
                else:
                    answers = list(exercise.expected)
                for answer in answers:
                    if not pr.check(question, answer).correct:
                        broken.append(f"{exercise.id}: {answer!r}")
        self.assertEqual(broken, [], f"ключ не проходит сверку: {broken[:10]}")

    def test_correct_items_really_contain_an_error(self) -> None:
        """В «исправь ошибку» само условие обязано отклоняться: иначе ошибки нет."""
        import random

        from english_bot.learning import practice as pr

        rng = random.Random(1)
        silent: list[str] = []
        for point in CURRICULUM.points.values():
            for exercise in point.exercises:
                if exercise.kind != "correct":
                    continue
                question = pr.resolve(f"ex:{exercise.id}", CURRICULUM, rng)
                assert question is not None
                if pr.check(question, exercise.prompt).correct:
                    silent.append(exercise.id)
        self.assertEqual(silent, [], f"условие засчитывается как ответ: {silent[:10]}")

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


class ListeningBankTests(unittest.TestCase):
    def test_dialogue_scripts_are_well_formed(self) -> None:
        """Строка «Имя: …» в скрипте значит диалог: тогда так оформлены все строки.

        Иначе синтез прочитал бы имена вслух одним голосом.
        """
        from english_bot.ai.tts import DIALOGUE_LINE, split_dialogue

        broken: list[str] = []
        for tasks in CURRICULUM.listening.values():
            for task in tasks:
                lines = [line for line in task.script_en.splitlines() if line.strip()]
                if any(DIALOGUE_LINE.match(line) for line in lines) and len(lines) > 1:
                    if not split_dialogue(task.script_en):
                        broken.append(task.id)
        self.assertEqual(broken, [], f"диалог оформлен не целиком: {broken}")

    def test_dialogues_declare_every_speaker_gender(self) -> None:
        """Голос выбирается по полу: необъявленный говорящий получил бы случайный."""
        from english_bot.ai.tts import split_dialogue

        for tasks in CURRICULUM.listening.values():
            for task in tasks:
                names = {name for name, _ in split_dialogue(task.script_en)}
                if not names:
                    continue
                with self.subTest(task=task.id):
                    self.assertEqual({name for name, _ in task.speakers}, names)

    def test_every_script_fits_its_level_length(self) -> None:
        """Запись C2 в 60 слов — это 20 секунд письменной речи, а не C2."""
        from english_bot.ai.tts import split_dialogue

        ranges = {"A1": (25, 40), "A2": (35, 55), "B1": (50, 80), "B2": (70, 110),
                  "C1": (90, 130), "C2": (110, 150)}
        outside: list[str] = []
        for level, tasks in CURRICULUM.listening.items():
            low, high = ranges[level]
            for task in tasks:
                turns = split_dialogue(task.script_en)
                words = sum(len(text.split()) for _, text in turns) if turns else len(task.script_en.split())
                if not low <= words <= high:
                    outside.append(f"{task.id}: {words}")
        self.assertEqual(outside, [], "длина записи вне диапазона уровня")

    def test_dialogue_names_used_in_the_question_are_heard(self) -> None:
        """Имена говорящих синтез не читает: ученик узнаёт Megan, только если её так назвали."""
        from english_bot.ai.tts import split_dialogue

        unheard: list[str] = []
        for tasks in CURRICULUM.listening.values():
            for task in tasks:
                turns = split_dialogue(task.script_en)
                spoken = " ".join(text for _, text in turns)
                asked = " ".join((task.question_en, *task.options))
                for name in {name for name, _ in turns}:
                    if re.search(rf"\b{name}\b", asked) and not re.search(rf"\b{name}\b", spoken):
                        unheard.append(f"{task.id}: {name}")
        self.assertEqual(unheard, [], "имя из вопроса не звучит в записи")

    def test_every_level_covers_all_listening_skills(self) -> None:
        from english_bot.content.banks import LISTENING_SKILLS

        for level in LEVELS:
            skills = {task.skill for task in CURRICULUM.listening.get(level, [])}
            with self.subTest(level=level):
                self.assertEqual(skills, set(LISTENING_SKILLS))


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

    def test_listening_codes_are_unique_short_and_round_trip(self) -> None:
        tasks = [task for rows in CURRICULUM.listening.values() for task in rows]
        codes = {CURRICULUM.listening_code(task.id) for task in tasks}
        self.assertEqual(len(codes), len(tasks))
        for task in tasks:
            code = CURRICULUM.listening_code(task.id)
            self.assertIs(CURRICULUM.listening_by_code(code), task)
            self.assertLessEqual(len(f"la:{code}:3".encode()), 64)


class BankTests(unittest.TestCase):
    def test_card_level_matches_its_file(self) -> None:
        """Уровень карточки — это её файл: обманки узнавания берутся из того же уровня.

        id при переносе не меняется, поэтому префикс id уровнем не считается.
        """
        for level in LEVELS:
            path = DATA_DIR / f"vocabulary_{level.lower()}.json"
            for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
                with self.subTest(card=item["id"]):
                    self.assertEqual(item["level"], level)

    def test_vocabulary_covers_all_levels(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertGreaterEqual(len(CURRICULUM.vocab_of_level(level)), 100)

    def test_vocabulary_has_no_cross_level_duplicates(self) -> None:
        """Проверяются файлы, а не загруженный курс: загрузчик повтор молча отбросил бы."""
        seen: dict[tuple[str, str], str] = {}
        for level in LEVELS:
            path = DATA_DIR / f"vocabulary_{level.lower()}.json"
            for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
                word = (item["word"].strip().lower(), item["pos"])
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

    def test_listening_covers_every_level_and_comprehension_skill(self) -> None:
        expected = {"gist", "detail", "inference", "attitude", "sequence"}
        averages: list[float] = []
        for level in LEVELS:
            with self.subTest(level=level):
                tasks = CURRICULUM.listening_of_level(level)
                self.assertGreaterEqual(len(tasks), 5)
                self.assertEqual({task.skill for task in tasks}, expected)
                averages.append(sum(len(task.script_en.split()) for task in tasks) / len(tasks))
                for task in tasks:
                    self.assertEqual(len(task.options), 4)
                    self.assertEqual(len(set(task.options)), 4)
                    self.assertIn(task.options[task.correct_index], task.options)
        self.assertEqual(averages, sorted(averages), "скрипты должны усложняться по CEFR")

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


class DiagnosticPoolTests(unittest.TestCase):
    """Пул диагностики — измерительный инструмент, а не просто выборка курса."""

    def test_every_level_has_enough_items_for_two_blocks(self) -> None:
        from english_bot.learning import placement as pl

        for level in LEVELS:
            with self.subTest(level=level):
                pool = pl._pool(CURRICULUM, level)
                self.assertGreaterEqual(
                    len(pool),
                    2 * pl.BLOCK_SIZE,
                    f"на {level} не хватит заданий на подтверждающий блок",
                )

    def test_pool_has_no_cross_level_duplicates(self) -> None:
        """Задание, дословно повторяющее материал уровнем ниже, уровень не измеряет."""
        import collections
        import re

        from english_bot.learning import placement as pl
        from english_bot.learning.answers import normalize

        order = {level: index for index, level in enumerate(LEVELS)}
        by_key = collections.defaultdict(set)
        for level in LEVELS:
            for exercise, point in pl._pool(CURRICULUM, level):
                options = tuple(sorted(normalize(option) for option in exercise.options))
                key = (options, normalize(exercise.options[exercise.correct_index or 0]))
                by_key[key].add(point.level)
                # Типовая инструкция без английского текста («В каком предложении
                # ошибка?») задание не определяет: его определяют варианты.
                if re.search(r"[a-z]", normalize(exercise.prompt)):
                    by_key[re.sub(r"\d+", "#", normalize(exercise.prompt))].add(point.level)
        clashes = [key for key, levels in by_key.items() if len(levels) > 1]
        self.assertEqual(clashes, [], f"дубли между уровнями в пуле: {clashes[:3]}")

    def test_flagged_items_stay_available_for_practice(self) -> None:
        """Непригодное для теста задание остаётся полноценной тренировкой."""
        excluded = [
            exercise
            for point in CURRICULUM.points.values()
            for exercise in point.exercises
            if not exercise.diagnostic
        ]
        self.assertTrue(excluded, "флаг diagnostic нигде не проставлен")
        sample = excluded[0]
        found = CURRICULUM.exercise(sample.id)
        self.assertIsNotNone(found)


if __name__ == "__main__":
    unittest.main()
