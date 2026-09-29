"""Локальная речь: распознавание faster-whisper и синтез piper.

Нужна, потому что подписка ChatGPT текст даёт, а звук — нет: у `codex exec` нет
ни распознавания, ни синтеза. Эти две модели закрывают голос целиком и бесплатно,
без единого сетевого запроса после установки.

Зависимости живут в `.venv` и импортируются лениво: платформа обязана
запускаться и на системном Python, просто с выключенной речью.
"""

from __future__ import annotations

import io
import logging
import os
import re
import secrets
import wave
from pathlib import Path
from typing import Any

from .stt import Transcript, TranscriptionError, count_fillers
from .tts import SpeechError, assign_voices, cache_name, speaking_order


LOGGER = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "small.en"
# Замер на этой машине: 1 поток — 0.79×, 2 — 1.61×, 3 — 1.68×, 4 — 1.22×, 6 — 0.93×
# реального времени. Больше потоков вредит: CTranslate2 на int8 захлёбывается.
DEFAULT_WHISPER_THREADS = 3
# Во столько раз расшифровка быстрее записи — для честной оценки ожидания:
# минутный ответ обрабатывается примерно за 40 секунд.
REALTIME_FACTOR = 1.6
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"
# Голоса диалогов аудирования по полу говорящих.
DEFAULT_FEMALE_VOICES = ("en_US-lessac-medium", "en_US-amy-medium")
DEFAULT_MALE_VOICES = ("en_US-ryan-medium", "en_US-joe-medium")
# Пауза между репликами диалога: без неё собеседники сливаются.
TURN_PAUSE_SECONDS = 0.35
SLOW_LENGTH_SCALE = 1.45  # растягивает слоги, не меняя высоту голоса
GENTLE_LENGTH_SCALE = 1.25  # аудирование A1–A2: на десятую часть спокойнее, слоги не растянуты
# piper обрывает звук ровно на последнем сэмпле: без полей слово в Telegram
# щёлкает на старте и рубится на конце. Десятая доля секунды это снимает.
PAD_SECONDS = 0.1


def estimate_seconds(duration: int) -> int:
    """Сколько примерно займёт расшифровка записи такой длины."""
    return max(3, round(duration / REALTIME_FACTOR) + 3)


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def piper_available() -> bool:
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return True


class LocalTranscriber:
    """Расшифровка через faster-whisper. Модель грузится один раз, при первом ГС."""

    def __init__(
        self,
        model_dir: Path,
        model_name: str = DEFAULT_WHISPER_MODEL,
        compute_type: str = "int8",
        threads: int = 3,
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.compute_type = compute_type
        self.threads = threads
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper не установлен. Запусти бота из .venv или "
                    "переключи SPEECH_BACKEND на openai."
                ) from exc
            LOGGER.info("Загружаю модель распознавания %s", self.model_name)
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.threads,
                download_root=str(self.model_dir),
            )
        return self._model

    def transcribe(self, path: Path, seconds: int, language: str = "en") -> Transcript:
        if not path.exists():
            raise TranscriptionError("файл записи не найден")
        model = self._load()
        try:
            segments, info = model.transcribe(
                str(path),
                language=language,
                beam_size=5,
                vad_filter=True,           # режет тишину: ученик часто думает перед фразой
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:  # библиотека кидает всё подряд, включая ошибки декодера
            raise TranscriptionError(f"распознавание не удалось: {exc}") from exc

        if not text:
            raise TranscriptionError("в записи не распозналась речь")
        # Длительность из Telegram надёжнее: она про исходный файл, а не про VAD.
        duration = seconds or int(getattr(info, "duration", 0) or 0)
        words = len([token for token in re.split(r"\s+", text) if token.strip(".,!?;:—-")])
        return Transcript(text=text, words=words, seconds=duration, fillers=count_fillers(text))


class LocalSpeaker:
    """Синтез через piper с последующей упаковкой в Ogg/Opus для Telegram.

    Диалогам аудирования нужны голоса по полу говорящих: иначе Megan звучит
    мужским голосом, и ученик путает, кто что сказал. Голоса без модели на диске
    из пула выпадают; пустой пул заменяется основным голосом.
    """

    def __init__(
        self,
        voice_dir: Path,
        cache_dir: Path,
        voice: str = DEFAULT_PIPER_VOICE,
        female_voices: tuple[str, ...] = DEFAULT_FEMALE_VOICES,
        male_voices: tuple[str, ...] = DEFAULT_MALE_VOICES,
    ):
        self.voice_dir = voice_dir
        self.cache_dir = cache_dir
        self.voice = voice
        self.female_voices = female_voices
        self.male_voices = male_voices
        self._voices: dict[str, Any] = {}

    @property
    def model_path(self) -> Path:
        return self.voice_dir / f"{self.voice}.onnx"

    def _load(self, name: str | None = None) -> Any:
        name = name or self.voice
        if name not in self._voices:
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise SpeechError(
                    "piper-tts не установлен. Запусти бота из .venv или "
                    "переключи SPEECH_BACKEND на openai."
                ) from exc
            path = self.voice_dir / f"{name}.onnx"
            if not path.exists():
                raise SpeechError(
                    f"голос {name} не найден в {self.voice_dir}. "
                    "Скачай его: python -m piper.download_voices <голос>"
                )
            LOGGER.info("Загружаю голос синтеза %s", name)
            self._voices[name] = PiperVoice.load(path)
        return self._voices[name]

    def _wav(
        self, text: str, voice_name: str | None = None, slow: bool = False, gentle: bool = False
    ) -> bytes:
        voice = self._load(voice_name)
        try:
            from piper.config import SynthesisConfig

            scale = SLOW_LENGTH_SCALE if slow else GENTLE_LENGTH_SCALE if gentle else None
            config = SynthesisConfig(length_scale=scale) if scale else None
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=config)
            return buffer.getvalue()
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(f"синтез не удался: {exc}") from exc

    def _store(self, destination: Path, wav_bytes: bytes) -> Path:
        audio = wav_to_opus(pad_wav(wav_bytes))
        # Кэш общий на всю платформу, поэтому имя временного файла уникально.
        partial = destination.with_suffix(f".{os.getpid()}.{secrets.token_hex(4)}.part")
        try:
            partial.write_bytes(audio)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        destination.chmod(0o600)
        return destination

    def synthesize(self, text: str, slow: bool = False, gentle: bool = False) -> Path:
        cleaned = text.strip()
        if not cleaned:
            raise SpeechError("нечего озвучивать")
        cleaned = cleaned[:900]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"piper-{self.voice}" + ("-gentle" if gentle else "")
        destination = self.cache_dir / cache_name(cleaned, tag, slow)
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        return self._store(destination, self._wav(cleaned, slow=slow, gentle=gentle))

    def _pools(self) -> dict[str, list[str]]:
        def present(names: tuple[str, ...]) -> list[str]:
            return [name for name in names if (self.voice_dir / f"{name}.onnx").exists()]

        return {"female": present(self.female_voices), "male": present(self.male_voices)}

    def synthesize_dialogue(
        self,
        lines: list[tuple[str, str]],
        genders: dict[str, str] | None = None,
        gentle: bool = False,
    ) -> Path:
        """Реплики голосами по полу говорящих (`tts.assign_voices`).

        Реплики склеиваются в одну запись с паузой между ними: Telegram отдаёт
        аудирование одним голосовым, а повтор идёт по его file_id.
        """
        turns = [(speaker, text.strip()) for speaker, text in lines if text.strip()]
        if not turns:
            raise SpeechError("нечего озвучивать")
        voices = assign_voices(speaking_order(turns), genders or {}, self._pools(), self.voice)
        script = "\n".join(f"{speaker}: {text}" for speaker, text in turns)[:2000]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = "piper-dialogue-" + ",".join(f"{name}={voice}" for name, voice in voices.items())
        destination = self.cache_dir / cache_name(script, tag + ("-gentle" if gentle else ""), False)
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        params = None
        frames: list[bytes] = []
        for index, (speaker, text) in enumerate(turns):
            with wave.open(io.BytesIO(self._wav(text, voices[speaker], gentle=gentle)), "rb") as source:
                current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                if params is None:
                    params = current
                elif current != params:
                    raise SpeechError("голоса диалога записаны в разных форматах")
                if index:
                    frames.append(b"\x00" * int(params[2] * TURN_PAUSE_SECONDS) * params[0] * params[1])
                frames.append(source.readframes(source.getnframes()))
        assert params is not None
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(params[0])
            target.setsampwidth(params[1])
            target.setframerate(params[2])
            target.writeframes(b"".join(frames))
        return self._store(destination, output.getvalue())


def pad_wav(wav_bytes: bytes, seconds: float = PAD_SECONDS) -> bytes:
    """Добавляет тишину в начало и в конец, сохраняя формат исходного WAV."""
    if seconds <= 0:
        return wav_bytes
    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())

    silence = b"\x00" * int(rate * seconds) * channels * width
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(silence + frames + silence)
    return output.getvalue()


def wav_to_opus(wav_bytes: bytes) -> bytes:
    """Telegram принимает голосовые только как Ogg/Opus, а piper отдаёт WAV.

    Перекодирование идёт через PyAV, который уже пришёл зависимостью
    faster-whisper и несёт в себе ffmpeg — системный ffmpeg не нужен.
    """
    try:
        import av
    except ImportError as exc:
        raise SpeechError("PyAV не установлен, перекодировать WAV в Opus нечем") from exc

    try:
        source = av.open(io.BytesIO(wav_bytes), mode="r", format="wav")
        output = io.BytesIO()
        target = av.open(output, mode="w", format="ogg")
        stream = target.add_stream("libopus", rate=48000)
        stream.layout = "mono"  # type: ignore[assignment]

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=48000)
        for frame in source.decode(audio=0):
            for resampled in resampler.resample(frame):
                for packet in stream.encode(resampled):
                    target.mux(packet)
        for packet in stream.encode(None):
            target.mux(packet)
        target.close()
        source.close()
    except Exception as exc:
        raise SpeechError(f"не удалось перекодировать в Opus: {exc}") from exc

    data = output.getvalue()
    if not data:
        raise SpeechError("перекодирование дало пустой файл")
    return data
