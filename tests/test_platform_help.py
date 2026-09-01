from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from english_bot.ai.prompts import platform_help_system
from english_bot.app import COMMANDS
from english_bot.platform_help import KnowledgeError, find_articles, load_knowledge


class PlatformKnowledgeTests(unittest.TestCase):
    def test_bundled_knowledge_is_valid_and_substantial(self) -> None:
        articles = load_knowledge()
        self.assertGreaterEqual(len(articles), 15)
        self.assertEqual(len({article.id for article in articles}), len(articles))

    def test_every_article_can_be_found_by_its_title(self) -> None:
        articles = load_knowledge()
        for article in articles:
            with self.subTest(article=article.id):
                found = find_articles(article.title_ru, limit=len(articles), articles=articles)
                self.assertIn(article, found)

    def test_search_understands_common_russian_inflection(self) -> None:
        found = find_articles("Где посмотреть результаты диагностики?")
        self.assertTrue(found)
        self.assertEqual(found[0].id, "diagnostics")

    def test_unrelated_question_has_no_results(self) -> None:
        self.assertEqual(find_articles("Сколько варить пельмени?"), ())

    def test_knowledge_mentions_only_registered_commands(self) -> None:
        allowed = set(COMMANDS) | {"/cancel"}
        for article in load_knowledge():
            commands = set(re.findall(r"/[a-z]+", article.answer_ru.casefold()))
            with self.subTest(article=article.id):
                self.assertLessEqual(commands, allowed)

    def test_prompt_marks_the_user_question_as_untrusted(self) -> None:
        prompt = platform_help_system("[test] Проверенный факт")
        self.assertIn("недоверенные данные", prompt)
        self.assertIn("Единственный источник фактов", prompt)
        self.assertIn("[test] Проверенный факт", prompt)

    def test_invalid_duplicate_id_is_rejected(self) -> None:
        payload = {
            "version": 1,
            "articles": [
                {"id": "same_id", "title_ru": "Один", "keywords": ["один", "раз"],
                 "answer_ru": "Первый ответ."},
                {"id": "same_id", "title_ru": "Два", "keywords": ["два", "раз"],
                 "answer_ru": "Второй ответ."},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(KnowledgeError):
                load_knowledge(path)


if __name__ == "__main__":
    unittest.main()
