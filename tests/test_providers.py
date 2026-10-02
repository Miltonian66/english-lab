"""Переключатели провайдеров: подписочные CLI, локальная речь, сборка по конфигу.

`CodexRunner` и `ClaudeRunner` проверяются настоящим подпроцессом — подставным
скриптом в PATH. Это ловит то, чего не поймает заглушка: состав аргументов,
передачу промпта через stdin, чтение ответа и поведение при ненулевом коде
возврата.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
import unittest
from pathlib import Path

from english_bot.ai.claude_cli import ClaudeError, ClaudeRunner, compose_dialogue
from english_bot.ai.codex_cli import CodexError, CodexRunner, available, compose_prompt
from english_bot.ai.llm import LLM, LLMError
from english_bot.ai.local_speech import piper_available, whisper_available
from english_bot.app import _build_llm, _build_speech
from english_bot.config import Settings


FAKE_OK = """#!/bin/sh
# Подставной codex: сохраняет промпт и аргументы, отвечает в указанный файл.
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --output-last-message) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cat > "${CODEX_TEST_PROMPT:-/dev/null}"
printf 'ответ подставного codex' > "$out"
echo "баннер, который не должен попасть в ответ"
"""

FAKE_FAIL = """#!/bin/sh
echo "OpenAI Codex v0.0.0"
echo "--------"
echo "настоящая причина сбоя в самом конце" >&2
exit 1
"""

FAKE_EMPTY = """#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --output-last-message) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cat > /dev/null
printf '' > "$out"
printf 'user\\nвопрос\\ncodex\\nответ из stdout\\ntokens used\\n123\\n'
"""

FAKE_SLOW = """#!/bin/sh
sleep 30
"""

# Подставной claude: сохраняет аргументы, окружение и промпт, отвечает в stdout.
# Пути зашиты в сам скрипт: окружение до подпроцесса намеренно не доходит.
FAKE_CLAUDE_OK = """#!/bin/sh
printf '%s\\n' "$@" > {args}
env > {env}
cat > {prompt}
printf 'ответ подставного claude'
"""

# Тот же подставной claude без слежки: нужен там, где важен только код возврата.
FAKE_CLAUDE_MIN = """#!/bin/sh
cat > /dev/null
printf 'ответ подставного claude'
"""

FAKE_CLAUDE_FAIL = """#!/bin/sh
echo "служебная строка"
echo "настоящая причина сбоя в самом конце" >&2
exit 2
"""

FAKE_CLAUDE_EMPTY = """#!/bin/sh
cat > /dev/null
printf ''
"""


class FakeCli:
    """Кладёт подставной бинарник провайдера в начало PATH на время теста."""

    def __init__(self, script: str, name: str = "codex"):
        self.script = script
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)
        path = self.dir / name
        path.write_text(self.script)
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self._saved = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.dir}{os.pathsep}{self._saved}"

    def close(self) -> None:
        os.environ["PATH"] = self._saved
        self._dir.cleanup()


class PromptTests(unittest.TestCase):
    def test_single_message_keeps_system_and_question(self) -> None:
        prompt = compose_prompt("СИСТЕМА", [{"role": "user", "content": "ВОПРОС"}])
        self.assertTrue(prompt.startswith("СИСТЕМА"))
        self.assertIn("ВОПРОС", prompt)
        self.assertNotIn("Диалог с учеником", prompt)

    def test_dialogue_is_labelled_and_last_turn_is_singled_out(self) -> None:
        prompt = compose_prompt(
            "СИСТЕМА",
            [
                {"role": "user", "content": "первая"},
                {"role": "assistant", "content": "ответ"},
                {"role": "user", "content": "последняя"},
            ],
        )
        self.assertIn("Диалог с учеником", prompt)
        self.assertIn("Ученик: первая", prompt)
        self.assertIn("Ты: ответ", prompt)
        self.assertIn("на неё и отвечай", prompt)
        self.assertTrue(prompt.rstrip().endswith("последняя"))

    def test_agent_is_told_not_to_use_tools(self) -> None:
        prompt = compose_prompt("С", [{"role": "user", "content": "в"}])
        self.assertIn("Не запускай команды", prompt)


class CodexRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Часть тестов роняет codex намеренно — предупреждения в выводе только мешают.
        self._logger = logging.getLogger("english_bot.ai.codex_cli")
        self._level = self._logger.level
        self._logger.setLevel(logging.CRITICAL)

    def tearDown(self) -> None:
        self._logger.setLevel(self._level)
        fake = getattr(self, "fake", None)
        if fake is not None:
            fake.close()

    def test_reads_answer_file_and_ignores_banner(self) -> None:
        self.fake = FakeCli(FAKE_OK)
        answer = CodexRunner().run("промпт")
        self.assertEqual(answer, "ответ подставного codex")
        self.assertNotIn("баннер", answer)

    def test_prompt_goes_through_stdin(self) -> None:
        self.fake = FakeCli(FAKE_OK)
        with tempfile.TemporaryDirectory() as spy_dir:
            spy = Path(spy_dir) / "prompt.txt"
            os.environ["CODEX_TEST_PROMPT"] = str(spy)
            try:
                CodexRunner().run("уникальный-промпт-42")
            finally:
                os.environ.pop("CODEX_TEST_PROMPT", None)
            self.assertIn("уникальный-промпт-42", spy.read_text())

    def test_nonzero_exit_reports_the_tail_of_output(self) -> None:
        self.fake = FakeCli(FAKE_FAIL)
        with self.assertRaises(CodexError) as caught:
            CodexRunner(attempts=1).run("промпт")
        message = str(caught.exception)
        self.assertIn("настоящая причина сбоя", message)

    def test_retry_is_attempted_before_giving_up(self) -> None:
        self.fake = FakeCli(FAKE_FAIL)
        runner = CodexRunner(attempts=2)
        with self.assertRaises(CodexError):
            runner.run("промпт")
        self.assertEqual(runner.attempts, 2)

    def test_empty_answer_file_falls_back_to_stdout(self) -> None:
        self.fake = FakeCli(FAKE_EMPTY)
        answer = CodexRunner(attempts=1).run("промпт")
        self.assertEqual(answer, "ответ из stdout")
        self.assertNotIn("tokens used", answer)

    def test_timeout_is_reported_not_hung(self) -> None:
        self.fake = FakeCli(FAKE_SLOW)
        with self.assertRaises(CodexError) as caught:
            CodexRunner(timeout=1, attempts=1).run("промпт")
        self.assertIn("не ответил", str(caught.exception))

    def test_missing_binary_names_the_way_out(self) -> None:
        with self.assertRaises(CodexError) as caught:
            CodexRunner(binary="codex-которого-нет", attempts=1).run("промпт")
        self.assertIn("LLM_PROVIDER=openai", str(caught.exception))

    def test_llm_facade_routes_to_codex(self) -> None:
        self.fake = FakeCli(FAKE_OK)
        llm = LLM("codex")
        self.assertEqual(llm.complete("система", [{"role": "user", "content": "вопрос"}]),
                         "ответ подставного codex")

    def test_llm_facade_turns_codex_failure_into_llm_error(self) -> None:
        self.fake = FakeCli(FAKE_FAIL)
        llm = LLM("codex", codex=CodexRunner(attempts=1))
        with self.assertRaises(LLMError):
            llm.complete("система", [{"role": "user", "content": "вопрос"}])


class ClaudeDialogueTests(unittest.TestCase):
    """Системная часть уходит флагом, поэтому в промпте её быть не должно."""

    def test_single_message_is_passed_as_is(self) -> None:
        prompt = compose_dialogue([{"role": "user", "content": "ВОПРОС"}])
        self.assertEqual(prompt, "ВОПРОС")
        self.assertNotIn("Диалог с учеником", prompt)

    def test_dialogue_is_labelled_and_last_turn_is_singled_out(self) -> None:
        prompt = compose_dialogue(
            [
                {"role": "user", "content": "первая"},
                {"role": "assistant", "content": "ответ"},
                {"role": "user", "content": "последняя"},
            ]
        )
        self.assertIn("Ученик: первая", prompt)
        self.assertIn("Ты: ответ", prompt)
        self.assertTrue(prompt.rstrip().endswith("последняя"))


class ClaudeRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._logger = logging.getLogger("english_bot.ai.claude_cli")
        self._level = self._logger.level
        self._logger.setLevel(logging.CRITICAL)
        self._spy = tempfile.TemporaryDirectory()
        self.args_path = Path(self._spy.name) / "args.txt"
        self.env_path = Path(self._spy.name) / "env.txt"
        self.prompt_path = Path(self._spy.name) / "prompt.txt"
        self.spy_script = FAKE_CLAUDE_OK.format(
            args=self.args_path, env=self.env_path, prompt=self.prompt_path
        )

    def tearDown(self) -> None:
        self._logger.setLevel(self._level)
        self._spy.cleanup()
        fake = getattr(self, "fake", None)
        if fake is not None:
            fake.close()

    def test_answer_comes_from_stdout(self) -> None:
        self.fake = FakeCli(self.spy_script, name="claude")
        self.assertEqual(ClaudeRunner().run("СИСТЕМА", "промпт"), "ответ подставного claude")

    def test_system_prompt_is_a_flag_and_dialogue_goes_through_stdin(self) -> None:
        """Ученик не должен видеть системный текст, а модель — личность агента."""
        self.fake = FakeCli(self.spy_script, name="claude")
        ClaudeRunner().run("СИСТЕМНЫЙ-ТЕКСТ", "уникальный-промпт-42")
        args = self.args_path.read_text().splitlines()
        self.assertIn("--system-prompt", args)
        self.assertIn("СИСТЕМНЫЙ-ТЕКСТ", args)
        prompt = self.prompt_path.read_text()
        self.assertIn("уникальный-промпт-42", prompt)
        self.assertNotIn("СИСТЕМНЫЙ-ТЕКСТ", prompt)

    def test_run_is_isolated_from_settings_history_and_tools(self) -> None:
        """Настройки владельца, переписка ученика и инструменты — вне запуска."""
        self.fake = FakeCli(self.spy_script, name="claude")
        ClaudeRunner().run("система", "промпт")
        args = self.args_path.read_text().splitlines()
        for flag in ("-p", "--safe-mode", "--no-session-persistence", "--strict-mcp-config"):
            with self.subTest(flag=flag):
                self.assertIn(flag, args)
        tools = args[args.index("--disallowedTools") + 1]
        self.assertIn("Bash", tools)
        self.assertIn("WebSearch", tools)

    def test_model_and_effort_are_passed_when_set(self) -> None:
        self.fake = FakeCli(self.spy_script, name="claude")
        ClaudeRunner(model="sonnet", effort="medium").run("система", "промпт")
        args = self.args_path.read_text().splitlines()
        self.assertEqual(args[args.index("--model") + 1], "sonnet")
        self.assertEqual(args[args.index("--effort") + 1], "medium")

    def test_platform_secrets_and_api_keys_do_not_reach_the_agent(self) -> None:
        """Ключ в окружении молча увёл бы запуск с подписки на платный контур."""
        self.fake = FakeCli(self.spy_script, name="claude")
        os.environ.update(
            {
                "TELEGRAM_BOT_TOKEN": "секретный-токен-бота",
                "ANTHROPIC_API_KEY": "секретный-ключ",
                "BOT_CLAIM_CODE": "секретный-код",
            }
        )
        try:
            ClaudeRunner().run("система", "промпт")
        finally:
            for key in ("TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY", "BOT_CLAIM_CODE"):
                os.environ.pop(key, None)
        seen = self.env_path.read_text()
        self.assertNotIn("секретный", seen)
        self.assertIn("HOME=", seen)  # без него CLI не найдёт учётку подписки
        self.assertIn("PATH=", seen)

    def test_gateway_reaches_the_agent_only_when_configured(self) -> None:
        """Шлюз включается настройкой бота, а не забытой ANTHROPIC_* в окружении."""
        self.fake = FakeCli(self.spy_script, name="claude")
        os.environ.update({"ANTHROPIC_BASE_URL": "https://чужой", "ANTHROPIC_AUTH_TOKEN": "чужой-токен"})
        try:
            ClaudeRunner().run("система", "промпт")
            self.assertNotIn("ANTHROPIC", self.env_path.read_text())
            ClaudeRunner(base_url="https://gw.example", auth_token="токен-шлюза").run("система", "промпт")
        finally:
            for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
                os.environ.pop(key, None)
        seen = self.env_path.read_text()
        self.assertIn("ANTHROPIC_BASE_URL=https://gw.example", seen)
        self.assertIn("ANTHROPIC_AUTH_TOKEN=токен-шлюза", seen)
        self.assertNotIn("чужой", seen)

    def test_nonzero_exit_reports_the_tail_of_output(self) -> None:
        self.fake = FakeCli(FAKE_CLAUDE_FAIL, name="claude")
        with self.assertRaises(ClaudeError) as caught:
            ClaudeRunner(attempts=1).run("система", "промпт")
        self.assertIn("настоящая причина сбоя", str(caught.exception))

    def test_empty_output_is_an_error_not_an_empty_reply(self) -> None:
        self.fake = FakeCli(FAKE_CLAUDE_EMPTY, name="claude")
        with self.assertRaises(ClaudeError):
            ClaudeRunner(attempts=1).run("система", "промпт")

    def test_timeout_is_reported_not_hung(self) -> None:
        self.fake = FakeCli(FAKE_SLOW, name="claude")
        with self.assertRaises(ClaudeError) as caught:
            ClaudeRunner(timeout=1, attempts=1).run("система", "промпт")
        self.assertIn("не ответил", str(caught.exception))

    def test_missing_binary_names_the_way_out(self) -> None:
        with self.assertRaises(ClaudeError) as caught:
            ClaudeRunner(binary="claude-которого-нет", attempts=1).run("система", "промпт")
        self.assertIn("LLM_PROVIDER", str(caught.exception))

    def test_llm_facade_routes_to_claude(self) -> None:
        self.fake = FakeCli(self.spy_script, name="claude")
        llm = LLM("claude")
        answer = llm.complete("система", [{"role": "user", "content": "вопрос"}])
        self.assertEqual(answer, "ответ подставного claude")

    def test_llm_facade_turns_claude_failure_into_llm_error(self) -> None:
        self.fake = FakeCli(FAKE_CLAUDE_FAIL, name="claude")
        llm = LLM("claude", claude=ClaudeRunner(attempts=1))
        with self.assertRaises(LLMError):
            llm.complete("система", [{"role": "user", "content": "вопрос"}])


class SettingsSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(os.environ)
        os.environ.update({"TELEGRAM_BOT_TOKEN": "t", "BOT_CLAIM_CODE": "c"})
        for key in ("LLM_PROVIDER", "SPEECH_BACKEND", "CODEX_EFFORT", "CLAUDE_EFFORT",
                    "CLAUDE_BINARY", "CLAUDE_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                    "CLAUDE_BASE_URL", "CLAUDE_AUTH_TOKEN"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved)

    def test_codex_is_ready_without_any_key(self) -> None:
        os.environ["LLM_PROVIDER"] = "codex"
        settings = Settings.from_env()
        self.assertTrue(settings.llm_ready)
        fake = FakeCli(FAKE_OK)
        try:
            self.assertEqual(_build_llm(settings).provider, "codex")  # type: ignore[union-attr]
        finally:
            fake.close()

    def test_claude_is_ready_without_any_key(self) -> None:
        os.environ.update({"LLM_PROVIDER": "claude", "CLAUDE_MODEL": "sonnet"})
        settings = Settings.from_env()
        self.assertTrue(settings.llm_ready)
        fake = FakeCli(FAKE_CLAUDE_MIN, name="claude")
        try:
            llm = _build_llm(settings)
        finally:
            fake.close()
        assert llm is not None
        self.assertEqual((llm.provider, llm.model), ("claude", "sonnet"))
        assert llm.claude is not None
        self.assertEqual(llm.claude.model, "sonnet")

    def test_claude_gateway_is_passed_to_the_runner(self) -> None:
        os.environ.update({"LLM_PROVIDER": "claude", "CLAUDE_BASE_URL": "https://gw.example/",
                           "CLAUDE_AUTH_TOKEN": "t"})
        fake = FakeCli(FAKE_CLAUDE_MIN, name="claude")
        try:
            llm = _build_llm(Settings.from_env())
        finally:
            fake.close()
        assert llm is not None and llm.claude is not None
        self.assertEqual((llm.claude.base_url, llm.claude.auth_token), ("https://gw.example", "t"))

    def test_claude_gateway_needs_both_parts_and_https(self) -> None:
        """Адрес без токена ушёл бы от имени подписки, токен по http — в открытую."""
        for env in ({"CLAUDE_BASE_URL": "https://gw.example"}, {"CLAUDE_AUTH_TOKEN": "t"},
                    {"CLAUDE_BASE_URL": "http://gw.example", "CLAUDE_AUTH_TOKEN": "t"}):
            with self.subTest(env=env):
                for key in ("CLAUDE_BASE_URL", "CLAUDE_AUTH_TOKEN"):
                    os.environ.pop(key, None)
                os.environ.update({"LLM_PROVIDER": "claude", **env})
                with self.assertRaises(RuntimeError):
                    Settings.from_env()

    def test_missing_cli_binary_disables_the_provider(self) -> None:
        """Обещать ИИ без бинарника — значит падать на каждом обращении."""
        os.environ.update({"LLM_PROVIDER": "claude", "CLAUDE_BINARY": "claude-которого-нет"})
        settings = Settings.from_env()
        self.assertTrue(settings.llm_ready)
        self.assertIsNone(_build_llm(settings))

    def test_openai_without_key_is_not_ready(self) -> None:
        os.environ["LLM_PROVIDER"] = "openai"
        settings = Settings.from_env()
        self.assertFalse(settings.llm_ready)
        self.assertIsNone(_build_llm(settings))

    def test_anthropic_uses_its_own_key_and_model(self) -> None:
        os.environ.update({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"})
        llm = _build_llm(Settings.from_env())
        assert llm is not None
        self.assertEqual((llm.provider, llm.api_key), ("anthropic", "k"))

    def test_unknown_provider_and_backend_are_rejected(self) -> None:
        os.environ["LLM_PROVIDER"] = "выдуманный"
        with self.assertRaises(RuntimeError):
            Settings.from_env()
        os.environ["LLM_PROVIDER"] = "codex"
        os.environ["SPEECH_BACKEND"] = "выдуманный"
        with self.assertRaises(RuntimeError):
            Settings.from_env()

    def test_unknown_codex_effort_is_rejected(self) -> None:
        os.environ.update({"LLM_PROVIDER": "codex", "CODEX_EFFORT": "запредельное"})
        with self.assertRaises(RuntimeError):
            Settings.from_env()

    def test_unknown_claude_effort_is_rejected(self) -> None:
        os.environ.update({"LLM_PROVIDER": "claude", "CLAUDE_EFFORT": "запредельное"})
        with self.assertRaises(RuntimeError):
            Settings.from_env()

    def test_local_speech_needs_no_key(self) -> None:
        os.environ["SPEECH_BACKEND"] = "local"
        self.assertTrue(Settings.from_env().speech_ready)

    def test_openai_speech_without_key_is_off(self) -> None:
        os.environ["SPEECH_BACKEND"] = "openai"
        settings = Settings.from_env()
        self.assertFalse(settings.speech_ready)
        self.assertEqual(_build_speech(settings), (None, None))

    def test_model_directories_hang_off_models_dir(self) -> None:
        settings = Settings.from_env()
        self.assertEqual(settings.whisper_dir.name, "whisper")
        self.assertEqual(settings.piper_dir.parent, settings.models_dir)

    def test_concurrency_defaults_separate_user_and_heavy_pools(self) -> None:
        for key in ("WORKERS", "JOB_WORKERS", "LLM_WORKERS", "STT_WORKERS", "TTS_WORKERS"):
            os.environ.pop(key, None)
        settings = Settings.from_env()
        self.assertEqual(
            (
                settings.workers,
                settings.job_workers,
                settings.llm_workers,
                settings.stt_workers,
                settings.tts_workers,
            ),
            (16, 4, 2, 1, 1),
        )

    def test_zero_job_workers_means_inline_long_chains(self) -> None:
        """`JOB_WORKERS=0` — законное значение: длинные цепочки идут в дорожке."""
        os.environ["JOB_WORKERS"] = "0"
        self.assertEqual(Settings.from_env().job_workers, 0)
        os.environ["JOB_WORKERS"] = "17"
        with self.assertRaises(RuntimeError):
            Settings.from_env()
        os.environ.pop("JOB_WORKERS", None)


class HostedDialogueTests(unittest.TestCase):
    """Диалог через OpenAI: реплики разными голосами, склейка в одно голосовое."""

    def setUp(self) -> None:
        from english_bot.ai import tts

        self.tts = tts
        self.calls: list[dict] = []
        self._dir = tempfile.TemporaryDirectory()
        self._post = tts.post_binary

        def fake_post(url: str, headers: dict, payload: dict) -> bytes:
            self.calls.append(payload)
            return b"\x10\x00" * 2400  # 0.1 с PCM на реплику

        tts.post_binary = fake_post
        self.speaker = tts.Speaker(
            "key", "gpt-4o-mini-tts", "alloy", Path(self._dir.name), ("nova", "shimmer"), ("onyx", "echo")
        )

    def tearDown(self) -> None:
        self.tts.post_binary = self._post
        self._dir.cleanup()

    def test_each_speaker_gets_own_voice(self) -> None:
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("склейка диалога требует PyAV из .venv")
        lines = [("Tom", "Hi."), ("Anna", "Hello."), ("Tom", "Bye.")]
        genders = {"Tom": "male", "Anna": "female"}
        path = self.speaker.synthesize_dialogue(lines, genders)
        self.assertEqual([call["voice"] for call in self.calls], ["onyx", "nova", "onyx"])
        self.assertTrue(all(call["response_format"] == "pcm" for call in self.calls))
        self.assertTrue(path.read_bytes().startswith(b"OggS"))
        self.speaker.synthesize_dialogue(lines, genders)
        self.assertEqual(len(self.calls), 3, "повтор берётся из кэша")

    def test_without_pyav_dialogue_is_read_by_one_voice(self) -> None:
        import builtins

        real_import = builtins.__import__

        def no_av(name: str, *args, **kwargs):
            if name == "av":
                raise ImportError("нет PyAV")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = no_av
        try:
            self.speaker.synthesize_dialogue([("Anna", "Hi."), ("Tom", "Hello.")])
        finally:
            builtins.__import__ = real_import
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["input"], "Hi. Hello.")
        self.assertEqual(self.calls[0]["response_format"], "opus")


class VoiceAssignmentTests(unittest.TestCase):
    POOLS = {"female": ["lessac", "amy"], "male": ["ryan", "joe"]}

    def test_voice_follows_declared_gender_not_speaking_order(self) -> None:
        """Megan говорила мужским голосом, потому что говорила второй."""
        from english_bot.ai.tts import assign_voices

        voices = assign_voices(["Tyler", "Megan"], {"Tyler": "male", "Megan": "female"}, self.POOLS, "lessac")
        self.assertEqual(voices, {"Tyler": "ryan", "Megan": "lessac"})

    def test_two_women_get_two_different_voices(self) -> None:
        from english_bot.ai.tts import assign_voices

        voices = assign_voices(["Sophie", "Linda"], {"Sophie": "female", "Linda": "female"}, self.POOLS, "lessac")
        self.assertEqual(voices, {"Sophie": "lessac", "Linda": "amy"})

    def test_undeclared_speakers_still_sound_different(self) -> None:
        from english_bot.ai.tts import assign_voices

        voices = assign_voices(["A", "B"], {}, self.POOLS, "lessac")
        self.assertNotEqual(voices["A"], voices["B"])
        self.assertEqual(assign_voices(["A"], {"A": "male"}, {"female": [], "male": []}, "lessac"), {"A": "lessac"})


class DialogueScriptTests(unittest.TestCase):
    def test_dialogue_lines_are_split_by_speaker(self) -> None:
        from english_bot.ai.tts import split_dialogue, spoken_text

        script = "Anna: Are you free on Friday?\nTom: I think so. Why?\nAnna: Team lunch."
        self.assertEqual(
            split_dialogue(script),
            [("Anna", "Are you free on Friday?"), ("Tom", "I think so. Why?"), ("Anna", "Team lunch.")],
        )
        self.assertEqual(spoken_text(script), "Are you free on Friday? I think so. Why? Team lunch.")

    def test_monologue_is_not_a_dialogue(self) -> None:
        from english_bot.ai.tts import split_dialogue, spoken_text

        self.assertEqual(split_dialogue("Note: the meeting moved. It starts at ten."), [])
        self.assertEqual(split_dialogue("Anna: Hi.\nAnna: It's me again."), [])
        self.assertEqual(spoken_text("Just one voice here."), "Just one voice here.")


@unittest.skipUnless(
    whisper_available() and piper_available(),
    "локальные модели доступны только из .venv",
)
class LocalSpeechTests(unittest.TestCase):
    """Прогоняются только при запуске из .venv; из системного Python пропускаются."""

    @classmethod
    def setUpClass(cls) -> None:
        from english_bot.ai.local_speech import LocalSpeaker, LocalTranscriber

        cls._dir = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parent.parent
        cls.speaker = LocalSpeaker(root / "data/models/piper", Path(cls._dir.name))
        cls.transcriber = LocalTranscriber(root / "data/models/whisper")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_two_women_are_voiced_by_two_female_voices(self) -> None:
        if not (self.speaker.voice_dir / "en_US-amy-medium.onnx").exists():
            self.skipTest("нет второго женского голоса")
        self.speaker.synthesize_dialogue(
            [("Sophie", "Did you buy eggs?"), ("Linda", "Yes, a dozen.")],
            {"Sophie": "female", "Linda": "female"},
        )
        self.assertIn("en_US-amy-medium", self.speaker._voices)
        self.assertIn("en_US-lessac-medium", self.speaker._voices)

    def test_dialogue_is_one_ogg_with_both_voices(self) -> None:
        import av

        path = self.speaker.synthesize_dialogue([("Anna", "Are you free on Friday?"), ("Tom", "Yes, I am.")])
        with av.open(str(path)) as container:
            self.assertEqual(container.streams.audio[0].codec_context.name, "opus")
            self.assertGreater(container.duration / 1_000_000, 1.5)
        self.assertIn("en_US-ryan-medium", self.speaker._voices)

    def test_synthesis_produces_ogg_opus_for_telegram(self) -> None:
        import av

        path = self.speaker.synthesize("The schedule changed yesterday.")
        self.assertTrue(path.exists() and path.stat().st_size > 0)
        container = av.open(str(path))
        stream = container.streams.audio[0]
        self.assertEqual(container.format.name, "ogg")
        self.assertEqual(stream.codec_context.name, "opus")
        self.assertEqual(stream.codec_context.channels, 1)
        container.close()

    def test_second_call_comes_from_cache(self) -> None:
        first = self.speaker.synthesize("Cache me once.")
        second = self.speaker.synthesize("Cache me once.")
        self.assertEqual(first, second)

    def test_slow_variant_is_a_separate_longer_file(self) -> None:
        normal = self.speaker.synthesize("Comfortable.")
        slow = self.speaker.synthesize("Comfortable.", slow=True)
        self.assertNotEqual(normal, slow)
        self.assertGreater(slow.stat().st_size, normal.stat().st_size)

    def test_gentle_pace_is_between_normal_and_slow(self) -> None:
        """Аудирование A1–A2: спокойнее обычного, но не по слогам."""
        import av

        def seconds(path: Path) -> float:
            with av.open(str(path)) as container:
                return container.duration / 1_000_000

        text = "We usually have dinner at seven, but today we are going to a restaurant."
        normal = self.speaker.synthesize(text)
        gentle = self.speaker.synthesize(text, gentle=True)
        slow = self.speaker.synthesize(text, slow=True)
        self.assertNotEqual(normal, gentle)
        self.assertLess(seconds(normal), seconds(gentle))
        self.assertLess(seconds(gentle), seconds(slow))

    def test_round_trip_synthesis_then_transcription(self) -> None:
        spoken = "I have lived here for five years."
        audio = self.speaker.synthesize(spoken)
        transcript = self.transcriber.transcribe(audio, seconds=4)
        self.assertIn("lived here", transcript.text.lower())
        self.assertGreater(transcript.words, 4)
        self.assertEqual(transcript.seconds, 4)

    def test_missing_file_is_reported(self) -> None:
        from english_bot.ai.stt import TranscriptionError

        with self.assertRaises(TranscriptionError):
            self.transcriber.transcribe(Path("/нет/такого.ogg"), seconds=3)

    def test_empty_text_is_refused(self) -> None:
        from english_bot.ai.tts import SpeechError

        with self.assertRaises(SpeechError):
            self.speaker.synthesize("   ")


class CodexPresenceTests(unittest.TestCase):
    def test_available_reflects_path(self) -> None:
        self.assertFalse(available("codex-которого-точно-нет"))


class ProviderAgnosticTextTests(unittest.TestCase):
    """AGENTS.md: пользовательские сообщения не называют ключ или сервис.

    Провайдер переключается переменной окружения, а тексты остаются прежними;
    упоминание OpenAI или «ключа» при подписке ещё и просто неверно.
    """

    def test_refusals_do_not_name_a_provider_or_a_key(self) -> None:
        from english_bot import context

        forbidden = ("OpenAI", "Anthropic", "Codex", "Claude", "Whisper", "Piper", "ключ")
        for name in ("AI_OFF_TEXT", "SPEECH_OFF_TEXT", "LIMIT_TEXT"):
            text = getattr(context, name)
            for word in forbidden:
                with self.subTest(text=name, word=word):
                    self.assertNotIn(word.lower(), text.lower())


if __name__ == "__main__":
    unittest.main()
