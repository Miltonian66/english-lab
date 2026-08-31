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
from english_bot.learning.answers import matches, normalize, parse_choice


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

    def test_choice_accepts_letter_number_and_text(self) -> None:
        exercise = Exercise(
            id="t_02", kind="choice", prompt="p", explanation_ru="e",
            options=("lives", "lived", "has lived", "is living"), correct_index=2,
        )
        for given in ("C", "c)", "3", "has lived", "C) has lived"):
            with self.subTest(given=given):
                self.assertEqual(parse_choice(exercise, given), 2)
        self.assertIsNone(parse_choice(exercise, "яблоко"))


class PlacementTests(unittest.TestCase):
    def test_profile_answer_sets_start_level(self) -> None:
        self.assertEqual(pl.start_level_from_profile({"profile_self_index": "0"}), "A1")
        self.assertEqual(pl.start_level_from_profile({"profile_self_index": "3"}), "B2")
        self.assertEqual(pl.start_level_from_profile({}), pl.START_LEVEL)

    def test_ladder_climbs_on_success(self) -> None:
        state = pl.PlacementState(session_id=1, level="A2")
        state.results["A2"] = [1, 1, 1, 1]
        state.visited = ["A2"]
        self.assertTrue(pl.advance(state, CURRICULUM))
        self.assertEqual(state.level, "B1")

    def test_ladder_descends_on_failure(self) -> None:
        state = pl.PlacementState(session_id=1, level="B1")
        state.results["B1"] = [0, 0, 0, 1]
        state.visited = ["B1"]
        self.assertTrue(pl.advance(state, CURRICULUM))
        self.assertEqual(state.level, "A2")

    def test_ladder_stops_in_the_middle_band(self) -> None:
        state = pl.PlacementState(session_id=1, level="B1")
        state.results["B1"] = [1, 1, 0, 0]
        state.visited = ["B1"]
        self.assertFalse(pl.advance(state, CURRICULUM))

    def test_ladder_does_not_revisit_levels(self) -> None:
        state = pl.PlacementState(session_id=1, level="B1")
        state.results["B1"] = [1, 1, 1, 1]
        state.visited = ["B1", "B2"]
        self.assertFalse(pl.advance(state, CURRICULUM))

    def test_result_picks_highest_passed_level(self) -> None:
        state = pl.PlacementState(session_id=1)
        state.results = {"A2": [1, 1, 1, 1], "B1": [1, 1, 1, 0], "B2": [0, 0, 1, 0]}
        result = pl.finish(state, CURRICULUM)
        self.assertEqual(result.level, "B1")
        self.assertEqual(result.asked, 0)

    def test_result_drops_below_when_nothing_passed(self) -> None:
        state = pl.PlacementState(session_id=1)
        state.results = {"B1": [0, 0, 0, 0]}
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
        correct_letter = chr(ord("A") + (choice.correct_index or 0))
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
