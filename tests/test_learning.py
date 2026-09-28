"""Учебная логика: SM-2, сверка ответов, лестница диагностики, адаптивность."""

from __future__ import annotations

import random
import unittest
from datetime import UTC, datetime, timedelta

from english_bot.content.registry import load_curriculum
from english_bot.content.schema import Exercise
from english_bot.learning import placement as pl
from english_bot.learning import practice as pr
from english_bot.learning import srs
from english_bot.learning.answers import grade, matches, normalize, parse_choice


CURRICULUM = load_curriculum()
NOW = datetime(2026, 1, 1, tzinfo=UTC)


class SrsTests(unittest.TestCase):
    def test_new_card_is_due_immediately(self) -> None:
        card = srs.new_card("point", "b1_x", now=NOW)
        self.assertEqual(card.repetitions, 0)
        self.assertEqual(card.due_at, NOW.isoformat(timespec="seconds"))

    def test_intervals_follow_sm2_ladder(self) -> None:
        card = srs.new_card("point", "b1_x", now=NOW)
        first = srs.review(card, 5, now=NOW)
        self.assertEqual(first.interval_days, 1.0)
        second = srs.review(first, 5, now=NOW)
        self.assertEqual(second.interval_days, 6.0)
        third = srs.review(second, 5, now=NOW)
        self.assertGreater(third.interval_days, 6.0)

    def test_failure_resets_repetitions_and_counts_lapse(self) -> None:
        card = srs.new_card("point", "b1_x", now=NOW)
        for _ in range(3):
            card = srs.review(card, 5, now=NOW)
        failed = srs.review(card, 1, now=NOW)
        self.assertEqual(failed.repetitions, 0)
        self.assertEqual(failed.lapses, 1)
        self.assertLess(failed.interval_days, 1.0)

    def test_ease_never_drops_below_floor(self) -> None:
        card = srs.new_card("point", "b1_x", now=NOW)
        for _ in range(20):
            card = srs.review(card, 0, now=NOW)
        self.assertGreaterEqual(card.ease, srs.MIN_EASE)

    def test_due_date_moves_forward_by_interval(self) -> None:
        card = srs.review(srs.new_card("point", "b1_x", now=NOW), 5, now=NOW)
        due = datetime.fromisoformat(card.due_at)
        self.assertEqual(due, NOW + timedelta(days=1))

    def test_mastery_grows_and_is_capped(self) -> None:
        card = srs.new_card("vocab", "w", now=NOW)
        for _ in range(10):
            card = srs.review(card, 5, now=NOW)
        self.assertEqual(card.mastery, 5)
        self.assertEqual(srs.stars(card.mastery), "★★★★★")

    def test_quality_reflects_hint_usage(self) -> None:
        self.assertEqual(srs.quality(True), 5)
        self.assertEqual(srs.quality(True, used_hint=True), 3)
        self.assertEqual(srs.quality(False), 2)
        self.assertEqual(srs.quality(False, used_hint=True), 1)


class AnswerTests(unittest.TestCase):
    def _gap(self, answer: str, accept: tuple[str, ...] = ()) -> Exercise:
        return Exercise(
            id="t_01", kind="gap", prompt="p", explanation_ru="e", answer=answer, accept=accept
        )

    def test_normalization_ignores_case_punctuation_and_spacing(self) -> None:
        self.assertEqual(normalize("  I  saw him.  "), "i saw him")
        self.assertEqual(normalize("It’s fine!"), "it's fine")

    def test_contractions_match_expanded_forms(self) -> None:
        exercise = self._gap("didn't see")
        self.assertTrue(matches(exercise, "did not see"))
        self.assertTrue(matches(exercise, "Didn't see"))
        self.assertTrue(matches(exercise, "didn’t see"))

    def test_wrong_grammar_is_not_accepted(self) -> None:
        exercise = self._gap("has lived")
        self.assertFalse(matches(exercise, "have lived"))
        self.assertFalse(matches(exercise, "lived"))
        self.assertFalse(matches(exercise, ""))

    def test_accept_list_widens_the_key(self) -> None:
        exercise = self._gap("The bridge was built in 1990.", ("They built the bridge in 1990.",))
        self.assertTrue(matches(exercise, "they built the bridge in 1990"))

    def _ex(self, kind: str, prompt: str, answer: str, accept: tuple[str, ...] = (), ex_id: str = "b1_x_07") -> Exercise:
        return Exercise(id=ex_id, kind=kind, prompt=prompt, explanation_ru="e", answer=answer, accept=accept)

    def test_missing_comma_inside_the_sentence_is_not_an_error(self) -> None:
        exercise = self._ex("order", "you / I / were / if / I / would / call / him", "If I were you, I would call him.")
        self.assertTrue(matches(exercise, "If I were you I would call him"))
        self.assertTrue(matches(exercise, "if i were you, i would call him"))

    def test_comma_is_checked_where_it_is_the_point(self) -> None:
        """Неопределительное придаточное и «исправь ошибку» в одной запятой — запятая и есть ответ."""
        relative = self._ex("transform", "Join: My brother lives in Berlin. He is a doctor.",
                            "My brother, who lives in Berlin, is a doctor.")
        self.assertFalse(matches(relative, "My brother who lives in Berlin is a doctor"))
        splice = self._ex("correct", "The plan is good, however it is expensive.",
                          "The plan is good. However, it is expensive.")
        self.assertFalse(matches(splice, "The plan is good, however it is expensive."))
        punctuation = self._ex("correct", "Its late.", "It's late, isn't it?", ex_id="c2_punctuation_x_01")
        self.assertFalse(matches(punctuation, "It's late isn't it"))

    def test_ambiguous_contractions_are_read_every_way(self) -> None:
        there = self._ex("transform", "Add a tag: There is a problem.", "There is a problem, isn't there?")
        self.assertTrue(matches(there, "There's a problem, isn't there?"))
        perfect = self._ex("gap", "If I ___ (know), I would have come.", "had known")
        self.assertTrue(matches(perfect, "'d known"))
        self.assertTrue(matches(self._ex("gap", "p", "should have told"), "should've told"))
        # Чтение 's как has не делает верным другое время.
        self.assertFalse(matches(self._ex("gap", "p", "is working"), "has worked"))
        self.assertFalse(matches(self._ex("gap", "p", "had known"), "would know"))

    def test_generic_negations_expand(self) -> None:
        self.assertTrue(matches(self._ex("gap", "p", "need not worry"), "needn't worry"))
        self.assertTrue(matches(self._ex("gap", "p", "shall not"), "shan't"))

    def test_correct_accepts_the_fixed_fragment(self) -> None:
        """Реальная ошибка журнала: исправление верное, но прислано одним словом."""
        exercise = self._ex("correct", "She speaks to clients very polite.", "She speaks to clients very politely.")
        result = grade(exercise, "politely")
        self.assertTrue(result.correct)
        self.assertIn("She speaks to clients very politely.", result.note)
        self.assertTrue(matches(exercise, "very politely"))
        self.assertFalse(matches(exercise, "polite"))
        self.assertFalse(matches(exercise, "clients very"))

    def test_fragment_must_cover_a_deletion(self) -> None:
        exercise = self._ex("correct", "I can to send the report.", "I can send the report.")
        self.assertTrue(matches(exercise, "can send"))
        self.assertFalse(matches(exercise, "send"))

    def test_typo_in_a_given_word_is_forgiven_with_a_note(self) -> None:
        exercise = self._ex("correct", "We have three olds computers.", "We have three old computers.")
        result = grade(exercise, "We hve three old computers.")
        self.assertTrue(result.correct)
        self.assertIn("hve → have", result.note)

    def test_typo_in_the_corrected_word_is_still_an_error(self) -> None:
        exercise = self._ex("correct", "He goed to the office.", "He went to the office.")
        self.assertFalse(matches(exercise, "He wnet to the office."))
        self.assertFalse(matches(exercise, "He goes to the office."))
        gap = self._ex("gap", "He ___ (go) to the office.", "went")
        self.assertFalse(matches(gap, "wnet"))

    def test_choice_accepts_letter_number_and_text(self) -> None:
        """Буква и номер относятся к порядку показа, текст — к самому варианту."""
        from english_bot.learning.answers import display_options

        exercise = Exercise(
            id="t_02", kind="choice", prompt="p", explanation_ru="e",
            options=("lives", "lived", "has lived", "is living"), correct_index=2,
        )
        shown = display_options(exercise)
        position = next(i for i, (index, _) in enumerate(shown) if index == 2)
        letter = chr(ord("A") + position)
        for given in (letter, f"{letter.lower()})", str(position + 1),
                      "has lived", f"{letter}) has lived"):
            with self.subTest(given=given):
                self.assertEqual(parse_choice(exercise, given), 2)
        self.assertIsNone(parse_choice(exercise, "яблоко"))

    def test_key_position_is_not_the_file_order(self) -> None:
        """Ключ стоял первым в 47% заданий банка: «жать A» проходило блок теста."""
        import collections

        from english_bot.learning.answers import display_options

        items = [
            ex
            for point in CURRICULUM.points.values()
            for ex in point.exercises
            if ex.kind == "choice"
        ]
        positions = collections.Counter()
        for ex in items:
            shown = display_options(ex)
            positions[next(i for i, (idx, _) in enumerate(shown) if idx == ex.correct_index)] += 1
        share = max(positions.values()) / len(items)
        self.assertLess(share, 0.3, "порядок показа снова смещён к одной позиции")


class PlacementTests(unittest.TestCase):
    def test_profile_answer_sets_start_level(self) -> None:
        self.assertEqual(pl.start_level_from_profile({"profile_self_index": "0"}), "A1")
        self.assertEqual(pl.start_level_from_profile({"profile_self_index": "3"}), "B2")
        self.assertEqual(pl.start_level_from_profile({}), pl.START_LEVEL)

    def test_ladder_climbs_on_success(self) -> None:
        state = pl.PlacementState(session_id=1, level="A2")
        state.results["A2"] = [1, 1, 1, 1, 1, 0]  # 5 из 6 — порог подъёма
        state.visited = ["A2"]
        self.assertTrue(pl.advance(state, CURRICULUM))
        self.assertEqual(state.level, "B1")

    def test_ladder_descends_on_failure(self) -> None:
        state = pl.PlacementState(session_id=1, level="B1")
        state.results["B1"] = [0, 0, 0, 0, 1, 1]
        state.visited = ["B1"]
        self.assertTrue(pl.advance(state, CURRICULUM))
        self.assertEqual(state.level, "A2")

    def test_middle_band_asks_a_second_block_instead_of_stopping(self) -> None:
        """Половина верных — не приговор: раньше тест обрывался после одного блока."""
        state = pl.PlacementState(session_id=1, level="B1")
        state.results["B1"] = [1, 1, 1, 0, 0, 0]
        state.visited = ["B1"]
        self.assertTrue(pl.advance(state, CURRICULUM))
        self.assertEqual(state.level, "B1")

        state.results["B1"] += [1, 1, 0, 0, 0, 0]
        self.assertFalse(pl.advance(state, CURRICULUM))

    def test_top_level_needs_a_confirming_block(self) -> None:
        """C2 не с чем сравнить сверху, поэтому его подтверждает второй блок."""
        state = pl.PlacementState(session_id=1, level="C2")
        state.results["C2"] = [1, 1, 1, 1, 1, 0]
        state.visited = ["C1", "C2"]
        self.assertTrue(pl.advance(state, CURRICULUM))
        self.assertEqual(state.level, "C2")

    def test_visited_level_gets_a_second_block_not_an_abrupt_end(self) -> None:
        state = pl.PlacementState(session_id=1, level="B1")
        state.results["B1"] = [1, 1, 1, 1, 1, 1]
        state.visited = ["B1", "B2"]
        self.assertTrue(pl.advance(state, CURRICULUM))
        state.results["B1"] += [1, 1, 1, 1, 1, 1]
        self.assertFalse(pl.advance(state, CURRICULUM))

    def test_result_picks_highest_passed_level(self) -> None:
        state = pl.PlacementState(session_id=1)
        state.results = {
            "A2": [1, 1, 1, 1, 1, 1],
            "B1": [1, 1, 1, 1, 0, 0],
            "B2": [0, 0, 1, 0, 0, 1],
        }
        result = pl.finish(state, CURRICULUM)
        self.assertEqual(result.level, "B1")
        self.assertEqual(result.asked, 0)

    def test_incomplete_block_proves_nothing(self) -> None:
        state = pl.PlacementState(session_id=1)
        state.results = {"B1": [1, 1, 1, 1, 1, 1], "B2": [1, 1]}
        self.assertEqual(pl.finish(state, CURRICULUM).level, "B1")

    def test_top_level_needs_the_climb_threshold_to_be_assigned(self) -> None:
        """4 из 6 на C2 — это «продолжаем», а не потолок курса."""
        state = pl.PlacementState(session_id=1)
        state.results = {"C1": [1, 1, 1, 1, 1, 1], "C2": [1, 1, 1, 1, 0, 0]}
        self.assertEqual(pl.finish(state, CURRICULUM).level, "C1")
        state.results["C2"] = [1, 1, 1, 1, 1, 0]
        self.assertEqual(pl.finish(state, CURRICULUM).level, "C2")

    def test_result_drops_below_only_when_the_level_is_failed(self) -> None:
        state = pl.PlacementState(session_id=1)
        state.results = {"B1": [0, 0, 0, 0, 0, 0]}
        self.assertEqual(pl.finish(state, CURRICULUM).level, "A2")

    def test_half_right_keeps_the_tested_level(self) -> None:
        """Половина верных на A2 — это A2, а не A1, где не задали ни одного вопроса."""
        state = pl.PlacementState(session_id=1)
        state.results = {"A2": [1, 1, 1, 0, 0, 0]}
        self.assertEqual(pl.finish(state, CURRICULUM).level, "A2")

    def test_adaptive_run_converges_for_a_simulated_learner(self) -> None:
        """Ученик, стабильно решающий до B1 включительно, должен получить B1."""
        rng = random.Random(7)
        truth = {"A1": 0.95, "A2": 0.9, "B1": 0.8, "B2": 0.2, "C1": 0.05, "C2": 0.0}
        state = pl.PlacementState(session_id=1, level="A2")
        while True:
            found = pl.next_item(state, CURRICULUM, rng)
            if found is None:
                break
            exercise, point = found
            correct = rng.random() < truth[state.level]
            pl.record(state, exercise, point, correct)
            if not pl.advance(state, CURRICULUM):
                break
        result = pl.finish(state, CURRICULUM)
        self.assertIn(result.level, {"B1", "B2"})
        self.assertLessEqual(result.asked, pl.MAX_ITEMS)

    def test_next_item_never_repeats(self) -> None:
        rng = random.Random(3)
        state = pl.PlacementState(session_id=1, level="B1")
        for _ in range(12):
            found = pl.next_item(state, CURRICULUM, rng)
            self.assertIsNotNone(found)
            assert found is not None
            self.assertNotIn(found[0].id, state.asked)
            pl.record(state, found[0], found[1], True)


class ClozeTests(unittest.TestCase):
    RAW = {
        "id": "b1_test_cloze_09", "kind": "cloze",
        "prompt": "Last year I ___ to Spain. I ___ there before, so everything was new.",
        "gaps": [["went", "traveled"], ["had never been"]],
        "explanation_ru": "Прошлое событие и опыт до него.", "difficulty": 2,
    }

    def question(self) -> pr.Question:
        from english_bot.content.schema import parse_exercise

        exercise = parse_exercise(self.RAW, "t")
        return pr.Question(
            ref=f"ex:{exercise.id}", kind="cloze", prompt=pr.number_gaps(exercise.prompt), options=(),
            expected=exercise.expected, explanation_ru="", difficulty=2, point_id="p", level="B1",
            title_ru="", topic="", card_type="point", card_key="p", gaps=exercise.gaps,
        )

    def test_gaps_are_numbered_and_answers_split_many_ways(self) -> None:
        question = self.question()
        self.assertIn("(1) ___", question.prompt)
        self.assertIn("(2) ___", question.prompt)
        for given in ("went; had never been", "went\nhad never been", "1) traveled 2) had never been",
                      "went, had never been"):
            with self.subTest(given=given):
                self.assertTrue(pr.check(question, given).correct)

    def test_each_gap_is_graded_and_reported(self) -> None:
        verdict = pr.check(self.question(), "went; had been")
        self.assertFalse(verdict.correct)
        self.assertIn("1 ✓", verdict.note)
        self.assertIn("2 ✗ (had never been)", verdict.note)

    def test_wrong_number_of_answers_is_not_understood(self) -> None:
        verdict = pr.check(self.question(), "went")
        self.assertFalse(verdict.understood)

    def test_schema_rejects_inconsistent_cloze(self) -> None:
        from english_bot.content.schema import ContentError, parse_exercise

        for broken in (dict(self.RAW, gaps=[["went"]]), dict(self.RAW, gaps=[["a;b"], ["c"]]),
                       dict(self.RAW, prompt="No gaps in this text at all.")):
            with self.assertRaises(ContentError):
                parse_exercise(broken, "t")


class PracticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(11)

    def test_level_queue_has_requested_length_and_no_repeats(self) -> None:
        queue = pr.queue_for_level(CURRICULUM, "B1", self.rng, [], length=10)
        self.assertEqual(len(queue), 10)
        self.assertEqual(len(set(queue)), 10)

    def test_point_queue_starts_easy(self) -> None:
        point = CURRICULUM.points_of_level("B1")[0]
        queue = pr.queue_for_point(point, self.rng)
        difficulties = [
            pr.resolve(ref, CURRICULUM, self.rng).difficulty  # type: ignore[union-attr]
            for ref in queue
        ]
        self.assertEqual(difficulties, sorted(difficulties))

    def test_topic_queue_spreads_across_points(self) -> None:
        level = "B1"
        topic = CURRICULUM.topics_of_level(level)[0][0]
        queue = pr.queue_for_topic(CURRICULUM, level, topic, self.rng, length=12)
        self.assertTrue(queue)
        points = {
            pr.resolve(ref, CURRICULUM, self.rng).point_id  # type: ignore[union-attr]
            for ref in queue
        }
        self.assertGreaterEqual(len(points), min(2, len(CURRICULUM.points_of_topic(level, topic))))

    def test_level_queue_prefers_unseen_exercises(self) -> None:
        """Случайный выбор без истории к 10-му дню давал треть повторов при непройденном банке."""
        seen: set[str] = set()
        rng = random.Random(3)
        # Правило даёт в сессию одно задание, поэтому первые восемь сессий по
        # десять правил из пятнадцати гарантированно обходятся без повторов.
        for _ in range(8):
            queue = pr.queue_for_level(CURRICULUM, "C1", rng, [], length=10, seen=seen)
            self.assertFalse(seen & set(queue), "повтор, хотя невиденные задания уровня остались")
            seen |= set(queue)

    def test_topic_and_review_queues_prefer_unseen(self) -> None:
        level = "B1"
        topic = CURRICULUM.topics_of_level(level)[0][0]
        first = pr.queue_for_topic(CURRICULUM, level, topic, self.rng, length=4)
        second = pr.queue_for_topic(CURRICULUM, level, topic, self.rng, length=4, seen=set(first))
        self.assertFalse(set(first) & set(second))
        point = CURRICULUM.points_of_level(level)[0]
        seen = {f"ex:{exercise.id}" for exercise in point.exercises[:-1]}

        class Card:
            card_type = "point"
            card_key = point.id

        refs = pr.queue_for_review(CURRICULUM, [Card()], self.rng, level=level, seen=seen)
        self.assertEqual(refs, [f"ex:{point.exercises[-1].id}"])

    def test_rule_card_hides_the_answer_to_the_open_question(self) -> None:
        """Ответ стоял дословно в примерах карточки у 220 заданий, а кнопка доступна до ответа."""
        leaks = 0
        for point in CURRICULUM.points.values():
            for exercise in point.exercises:
                question = pr.resolve(f"ex:{exercise.id}", CURRICULUM, self.rng)
                assert question is not None
                card = pr.help_for(question, CURRICULUM)
                for secret in pr.revealing_texts(question):
                    if secret in pr._plain(card):
                        leaks += 1
                        break
        self.assertEqual(leaks, 0)

    def _vocab(self, word: str, pos: str = ""):
        for items in CURRICULUM.vocabulary.values():
            for item in items:
                if item.word == word and (not pos or item.pos == pos):
                    return item
        raise AssertionError(f"нет слова {word}")

    def test_recall_accepts_same_meaning_words_and_forms_with_a_note(self) -> None:
        """«делать» — это и do, и make; «вставать» в прошедшем — тоже знание слова."""
        question = pr.vocab_question(self._vocab("do"), CURRICULUM, self.rng)
        self.assertTrue(pr.check(question, "do").correct)
        verdict = pr.check(question, "make")
        self.assertTrue(verdict.correct)
        self.assertIn("Загадано слово: do", verdict.note)
        self.assertTrue(pr.check(question, "did").correct)
        self.assertFalse(pr.check(question, "go").correct)
        phrasal = pr.vocab_question(self._vocab("check in"), CURRICULUM, self.rng)
        self.assertTrue(pr.check(phrasal, "checked in").correct)

    def test_recall_forms_depend_on_part_of_speech(self) -> None:
        self.assertEqual(pr.accepted_forms(self._vocab("on")), [])
        self.assertEqual(pr.accepted_forms(self._vocab("table")), ["tables"])
        self.assertIn("came", pr.accepted_forms(self._vocab("come")))
        self.assertNotIn("comed", pr.word_forms("come"))

    def test_recognition_never_offers_a_second_correct_word(self) -> None:
        """«на» — это on и at: at среди обманок засчитывался бы ошибкой."""
        for items in CURRICULUM.vocabulary.values():
            for item in items:
                question = pr.vocab_question(item, CURRICULUM, self.rng, recognise=True)
                self.assertEqual(question.kind, "choice")
                self.assertEqual(len(question.options), 4, item.word)
                valid = {word.lower() for word in pr.vocab_alternatives(item, CURRICULUM)}
                wrong = [option for option in question.options if option != item.word and option.lower() in valid]
                self.assertEqual(wrong, [], item.word)

    def test_recall_hint_hides_the_word_even_at_sentence_start(self) -> None:
        item = self._vocab("close")
        question = pr.vocab_question(item, CURRICULUM, self.rng)
        self.assertIn("Подсказка: ___ the door", question.prompt)

    def test_same_spelling_different_part_of_speech_are_two_cards(self) -> None:
        self.assertEqual(self._vocab("book", "noun").level, "A1")
        self.assertEqual(self._vocab("book", "verb").level, "A2")
        self.assertEqual(self._vocab("cost", "noun").level, "B1")

    def test_check_accepts_the_corrected_fragment_of_a_real_exercise(self) -> None:
        question = pr.resolve("ex:a1_adverbs_manner_07", CURRICULUM, self.rng)
        assert question is not None
        verdict = pr.check(question, "politely")
        self.assertTrue(verdict.correct)
        self.assertIn("She speaks to clients very politely.", verdict.note)

    def test_adapt_raises_difficulty_when_learner_is_cruising(self) -> None:
        queue = pr.queue_for_level(CURRICULUM, "B2", self.rng, [], length=10)
        state = pr.PracticeState(kind="mixed", subject="B2", queue=queue, index=3,
                                 answered=4, correct=4)
        pr.adapt(state, CURRICULUM, self.rng)
        tail = [
            pr.resolve(ref, CURRICULUM, self.rng).difficulty  # type: ignore[union-attr]
            for ref in state.queue[state.index :]
        ]
        self.assertEqual(tail, sorted(tail, reverse=True))

    def test_adapt_lowers_difficulty_when_learner_struggles(self) -> None:
        queue = pr.queue_for_level(CURRICULUM, "B2", self.rng, [], length=10)
        state = pr.PracticeState(kind="mixed", subject="B2", queue=queue, index=3,
                                 answered=4, correct=0)
        pr.adapt(state, CURRICULUM, self.rng)
        tail = [
            pr.resolve(ref, CURRICULUM, self.rng).difficulty  # type: ignore[union-attr]
            for ref in state.queue[state.index :]
        ]
        self.assertEqual(tail, sorted(tail))

    def test_adapt_leaves_queue_alone_inside_target_band(self) -> None:
        queue = pr.queue_for_level(CURRICULUM, "B1", self.rng, [], length=10)
        state = pr.PracticeState(kind="mixed", subject="B1", queue=list(queue), index=3,
                                 answered=3, correct=2)
        pr.adapt(state, CURRICULUM, self.rng)
        self.assertEqual(state.queue, queue)

    def test_every_free_input_task_says_what_to_do(self) -> None:
        """Условие без постановки задачи — это загадка, а не упражнение."""
        seen = set()
        for exercise, _ in CURRICULUM.exercises.values():
            if exercise.kind == "choice":
                continue
            question = pr.resolve(f"ex:{exercise.id}", CURRICULUM, self.rng)
            assert question is not None
            hint = pr.task_hint(question)
            seen.add(exercise.kind)
            with self.subTest(exercise=exercise.id):
                self.assertTrue(hint)
                # «Напиши ответ сообщением» ничего не объясняет: оно допустимо
                # только там, где постановка уже стоит в самом условии.
                if hint == "Напиши ответ сообщением.":
                    self.assertNotEqual(exercise.kind, "correct")
                    self.assertNotEqual(exercise.kind, "order")
        self.assertEqual(seen, {"gap", "correct", "order", "transform"})

    def test_error_hunting_task_is_announced(self) -> None:
        exercise = next(e for e, _ in CURRICULUM.exercises.values() if e.kind == "correct")
        question = pr.resolve(f"ex:{exercise.id}", CURRICULUM, self.rng)
        assert question is not None
        self.assertIn("ошибка", pr.task_hint(question))
        self.assertIn("целиком", pr.task_hint(question))

    def test_gap_task_asks_only_for_the_missing_part(self) -> None:
        exercise = next(
            e for e, _ in CURRICULUM.exercises.values() if e.kind == "gap" and "___" in e.prompt
        )
        question = pr.resolve(f"ex:{exercise.id}", CURRICULUM, self.rng)
        assert question is not None
        self.assertIn("вместо пропуска", pr.task_hint(question))

    def test_gap_without_a_gap_does_not_promise_one(self) -> None:
        """Часть заданий несёт инструкцию в условии — вторая ей противоречила бы."""
        item = CURRICULUM.vocab_of_level("A1")[0]
        question = pr.vocab_question(item, CURRICULUM, self.rng, recognise=False)
        self.assertEqual(question.kind, "gap")
        if "___" not in question.prompt:
            self.assertNotIn("пропуск", pr.task_hint(question))

    def test_state_round_trips_through_json_shape(self) -> None:
        state = pr.PracticeState(kind="topic", subject="Tenses", queue=["ex:a"], index=1,
                                 correct=1, answered=2, session_id=9, level="B1", helped=True)
        restored = pr.PracticeState.from_dict(state.to_dict())
        self.assertEqual(restored, state)

    def test_check_understands_letters_and_free_text(self) -> None:
        point = CURRICULUM.points_of_level("B1")[0]
        choice = next(ex for ex in point.exercises if ex.kind == "choice")
        question = pr.resolve(f"ex:{choice.id}", CURRICULUM, self.rng)
        assert question is not None
        key = choice.options[choice.correct_index or 0]
        correct_letter = chr(ord("A") + question.options.index(key))
        self.assertTrue(pr.check(question, correct_letter).correct)
        self.assertFalse(pr.check(question, "полная ерунда").understood)

    def test_vocab_question_expects_the_word(self) -> None:
        item = CURRICULUM.vocab_of_level("B1")[0]
        question = pr.vocab_question(item, CURRICULUM, random.Random(1))
        self.assertEqual(question.card_type, "vocab")
        self.assertTrue(pr.check(question, item.word).correct)

    def test_vocab_reference_always_resolves_to_the_same_question(self) -> None:
        """Иначе показанный порядок вариантов и проверяемый разойдутся."""
        item = CURRICULUM.vocab_of_level("B1")[0]
        for ref in (f"vocab:r:{item.id}", f"vocab:p:{item.id}"):
            with self.subTest(ref=ref):
                shapes = {
                    (
                        pr.resolve(ref, CURRICULUM, random.Random(seed)).kind,  # type: ignore[union-attr]
                        pr.resolve(ref, CURRICULUM, random.Random(seed)).options,  # type: ignore[union-attr]
                        pr.resolve(ref, CURRICULUM, random.Random(seed)).expected,  # type: ignore[union-attr]
                    )
                    for seed in range(25)
                }
                self.assertEqual(len(shapes), 1)

    def test_vocab_queue_prefers_unseen_words(self) -> None:
        pool = CURRICULUM.vocab_upto("B1")
        seen = {item.id for item in pool[:-4]}
        refs = pr.queue_of_vocab(CURRICULUM, "B1", seen, self.rng, 3)
        self.assertEqual(len(refs), 3)
        for ref in refs:
            vocab_id = ref.split(":", 2)[2]
            self.assertNotIn(vocab_id, seen)

    def test_vocab_ref_carries_the_mode(self) -> None:
        refs = {pr.vocab_ref("b1_v_x", random.Random(seed)) for seed in range(30)}
        self.assertEqual({ref.split(":")[1] for ref in refs}, {"r", "p"})

    def test_unknown_reference_resolves_to_none(self) -> None:
        self.assertIsNone(pr.resolve("ex:does_not_exist", CURRICULUM, self.rng))
        self.assertIsNone(pr.resolve("junk", CURRICULUM, self.rng))


if __name__ == "__main__":
    unittest.main()
