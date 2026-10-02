from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Переключатели контуров. Текст и речь выбираются независимо: можно взять
# подписочный Codex или Claude и при этом озвучивать через локальные модели,
# а позже перевести текст на API, поменяв одну переменную.
LLM_PROVIDERS: tuple[str, ...] = ("codex", "claude", "openai", "anthropic")
# Провайдеры, которые работают подпроцессом по подписке, а не по ключу.
CLI_PROVIDERS: tuple[str, ...] = ("codex", "claude")
SPEECH_BACKENDS: tuple[str, ...] = ("local", "openai")
CODEX_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
CLAUDE_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _int_env(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} должен быть целым числом") from exc
    if not low <= value <= high:
        raise RuntimeError(f"{name} должен быть в диапазоне {low}..{high}")
    return value


def _voices_env(name: str, default: str) -> tuple[str, ...]:
    """Список голосов через запятую; пустое значение — список по умолчанию."""
    raw = os.environ.get(name, "").strip() or default
    return tuple(voice.strip() for voice in raw.split(",") if voice.strip())


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    claim_code: str
    database_path: Path
    export_dir: Path
    voice_dir: Path
    audio_cache_dir: Path
    log_level: str

    llm_provider: str
    codex_binary: str
    codex_model: str
    codex_effort: str
    codex_timeout: int
    claude_binary: str
    claude_model: str
    claude_effort: str
    claude_timeout: int
    claude_base_url: str
    claude_auth_token: str
    speech_backend: str
    models_dir: Path
    whisper_model: str
    whisper_compute: str
    whisper_threads: int
    piper_voice: str
    piper_female_voices: tuple[str, ...]
    piper_male_voices: tuple[str, ...]
    openai_api_key: str | None
    openai_model: str
    anthropic_api_key: str | None
    anthropic_model: str
    stt_model: str
    tts_model: str
    tts_voice: str
    tts_female_voices: tuple[str, ...]
    tts_male_voices: tuple[str, ...]

    workers: int
    job_workers: int
    job_timeout: int
    llm_workers: int
    stt_workers: int
    tts_workers: int
    max_voice_seconds: int
    daily_ai_calls: int
    team_open_registration: bool

    @property
    def llm_ready(self) -> bool:
        """Готов ли текстовый ИИ-наставник."""
        if self.llm_provider in CLI_PROVIDERS:
            return True  # ключ не нужен, наличие бинарника проверяет `app._build_llm`
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return bool(self.openai_api_key)

    @property
    def speech_ready(self) -> bool:
        """Готовы ли распознавание и синтез."""
        if self.speech_backend == "local":
            return True  # наличие моделей проверяется при первом обращении
        return bool(self.openai_api_key)

    @property
    def whisper_dir(self) -> Path:
        return self.models_dir / "whisper"

    @property
    def piper_dir(self) -> Path:
        return self.models_dir / "piper"

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

        claim_code = os.environ.get("BOT_CLAIM_CODE", "").strip()
        if not claim_code:
            raise RuntimeError("BOT_CLAIM_CODE is required for safe first-user setup")

        provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai"
        if provider not in LLM_PROVIDERS:
            raise RuntimeError(f"LLM_PROVIDER должен быть одним из {', '.join(LLM_PROVIDERS)}")

        backend = os.environ.get("SPEECH_BACKEND", "local").strip().lower() or "local"
        if backend not in SPEECH_BACKENDS:
            raise RuntimeError(f"SPEECH_BACKEND должен быть одним из {', '.join(SPEECH_BACKENDS)}")

        effort = os.environ.get("CODEX_EFFORT", "low").strip().lower() or "low"
        if effort not in CODEX_EFFORTS:
            raise RuntimeError(f"CODEX_EFFORT должен быть одним из {', '.join(CODEX_EFFORTS)}")

        claude_effort = os.environ.get("CLAUDE_EFFORT", "low").strip().lower() or "low"
        if claude_effort not in CLAUDE_EFFORTS:
            raise RuntimeError(f"CLAUDE_EFFORT должен быть одним из {', '.join(CLAUDE_EFFORTS)}")

        # Шлюз задаётся парой: адрес без токена ушёл бы от имени подписки
        # владельца, токен без адреса ничего не значит. Только https — токен
        # идёт в заголовке каждого запроса.
        claude_base_url = os.environ.get("CLAUDE_BASE_URL", "").strip().rstrip("/")
        claude_auth_token = os.environ.get("CLAUDE_AUTH_TOKEN", "").strip()
        if bool(claude_base_url) != bool(claude_auth_token):
            raise RuntimeError("CLAUDE_BASE_URL и CLAUDE_AUTH_TOKEN задаются только вместе")
        if claude_base_url and not claude_base_url.startswith("https://"):
            raise RuntimeError("CLAUDE_BASE_URL должен начинаться с https://")

        return cls(
            telegram_token=token,
            claim_code=claim_code,
            database_path=_project_path(
                os.environ.get("DATABASE_PATH", "data/english_bot.sqlite3")
            ),
            export_dir=_project_path(os.environ.get("EXPORT_DIR", "data/exports")),
            voice_dir=_project_path(os.environ.get("VOICE_DIR", "data/voices")),
            audio_cache_dir=_project_path(os.environ.get("AUDIO_CACHE_DIR", "data/audio")),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            llm_provider=provider,
            codex_binary=os.environ.get("CODEX_BINARY", "codex").strip() or "codex",
            codex_model=os.environ.get("CODEX_MODEL", "").strip(),
            codex_effort=effort,
            codex_timeout=_int_env("CODEX_TIMEOUT", 180, 30, 900),
            claude_binary=os.environ.get("CLAUDE_BINARY", "claude").strip() or "claude",
            claude_model=os.environ.get("CLAUDE_MODEL", "sonnet").strip(),
            claude_effort=claude_effort,
            claude_timeout=_int_env("CLAUDE_TIMEOUT", 180, 30, 900),
            claude_base_url=claude_base_url,
            claude_auth_token=claude_auth_token,
            speech_backend=backend,
            models_dir=_project_path(os.environ.get("MODELS_DIR", "data/models")),
            whisper_model=os.environ.get("WHISPER_MODEL", "small.en").strip() or "small.en",
            whisper_compute=os.environ.get("WHISPER_COMPUTE", "int8").strip() or "int8",
            whisper_threads=_int_env("WHISPER_THREADS", 3, 1, 32),
            piper_voice=(
                os.environ.get("PIPER_VOICE", "en_US-lessac-medium").strip()
                or "en_US-lessac-medium"
            ),
            piper_female_voices=_voices_env("PIPER_FEMALE_VOICES", "en_US-lessac-medium,en_US-amy-medium"),
            piper_male_voices=_voices_env("PIPER_MALE_VOICES", "en_US-ryan-medium,en_US-joe-medium"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip() or None,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip(),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip() or None,
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5").strip(),
            stt_model=os.environ.get("STT_MODEL", "whisper-1").strip(),
            tts_model=os.environ.get("TTS_MODEL", "gpt-4o-mini-tts").strip(),
            tts_voice=os.environ.get("TTS_VOICE", "alloy").strip(),
            tts_female_voices=_voices_env("TTS_FEMALE_VOICES", "nova,shimmer"),
            tts_male_voices=_voices_env("TTS_MALE_VOICES", "onyx,echo"),
            workers=_int_env("WORKERS", 16, 1, 64),
            # 0 — выполнять длинные цепочки прямо в дорожке обновления, как было
            # до фоновых задач. Нужен тестам обработчиков и отладке по шагам.
            job_workers=_int_env("JOB_WORKERS", 4, 0, 16),
            job_timeout=_int_env("JOB_TIMEOUT", 900, 60, 3600),
            llm_workers=_int_env("LLM_WORKERS", 2, 1, 8),
            stt_workers=_int_env("STT_WORKERS", 1, 1, 4),
            tts_workers=_int_env("TTS_WORKERS", 1, 1, 4),
            max_voice_seconds=_int_env("MAX_VOICE_SECONDS", 300, 30, 900),
            daily_ai_calls=_int_env("DAILY_AI_CALLS", 120, 5, 5000),
            team_open_registration=(
                os.environ.get("TEAM_OPEN_REGISTRATION", "").strip().lower() in {"1", "true", "yes"}
            ),
        )
