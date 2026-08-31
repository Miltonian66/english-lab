from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Переключатели контуров. Текст и речь выбираются независимо: можно взять
# бесплатный Codex по подписке и при этом озвучивать через локальные модели,
# а позже перевести текст на API, поменяв одну переменную.
LLM_PROVIDERS: tuple[str, ...] = ("codex", "openai", "anthropic")
SPEECH_BACKENDS: tuple[str, ...] = ("local", "openai")
CODEX_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")


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
    speech_backend: str
    models_dir: Path
    whisper_model: str
    whisper_compute: str
    whisper_threads: int
    piper_voice: str
    openai_api_key: str | None
    openai_model: str
    anthropic_api_key: str | None
    anthropic_model: str
    stt_model: str
    tts_model: str
    tts_voice: str

    workers: int
    max_voice_seconds: int
    daily_ai_calls: int
    team_open_registration: bool

    @property
    def llm_ready(self) -> bool:
        """Готов ли текстовый ИИ-наставник."""
        if self.llm_provider == "codex":
            return True  # ключ не нужен, проверка наличия бинарника — при запуске
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
            speech_backend=backend,
            models_dir=_project_path(os.environ.get("MODELS_DIR", "data/models")),
            whisper_model=os.environ.get("WHISPER_MODEL", "small.en").strip() or "small.en",
            whisper_compute=os.environ.get("WHISPER_COMPUTE", "int8").strip() or "int8",
            whisper_threads=_int_env("WHISPER_THREADS", 3, 1, 32),
            piper_voice=(
                os.environ.get("PIPER_VOICE", "en_US-lessac-medium").strip()
                or "en_US-lessac-medium"
            ),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip() or None,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip(),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip() or None,
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5").strip(),
            stt_model=os.environ.get("STT_MODEL", "whisper-1").strip(),
            tts_model=os.environ.get("TTS_MODEL", "gpt-4o-mini-tts").strip(),
            tts_voice=os.environ.get("TTS_VOICE", "alloy").strip(),
            workers=_int_env("WORKERS", 4, 1, 16),
            max_voice_seconds=_int_env("MAX_VOICE_SECONDS", 300, 30, 900),
            daily_ai_calls=_int_env("DAILY_AI_CALLS", 120, 5, 5000),
            team_open_registration=(
                os.environ.get("TEAM_OPEN_REGISTRATION", "").strip().lower() in {"1", "true", "yes"}
            ),
        )
