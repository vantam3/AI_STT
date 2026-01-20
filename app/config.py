import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


def _require(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise ConfigError(f"missing required env var: {name}")
    return value


def _require_int(name: str) -> int:
    raw = _require(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid int for {name}: {raw!r}") from exc


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid int for {name}: {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"invalid float for {name}: {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    raw = raw.strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    raise ConfigError(f"invalid bool for {name}: {raw!r}")


@dataclass(frozen=True)
class Settings:
    HOST: str
    PORT: int
    DEVICE: str
    COMPUTE_TYPE: str
    DEFAULT_MODEL_SIZE: str
    MAX_SESSIONS: int
    RTP_HOST: str
    RTP_PORT_MIN: int
    RTP_PORT_MAX: int
    BACKEND_URL: str
    AI_KEY: str
    FFMPEG_LOGLEVEL: str
    FFMPEG_RTBUF_SIZE: str
    FFMPEG_BUFFER_SIZE: str
    STT_WARMUP_WINDOW_SECONDS: float
    STT_WARMUP_WINDOWS: int
    STT_FFMPEG_READ_TIMEOUT: float
    STT_RTP_IDLE_MS: float
    STT_BEAM_SIZE: int
    STT_LANGUAGE: str
    STT_VAD_FILTER: bool
    STT_BUFFER_SECONDS: float
    STT_RMS_THRESHOLD: float
    STT_MIN_SPEECH_SECONDS: float
    STT_MIN_SPEECH_RATIO: float
    STT_STABLE_MIN_CHARS: int
    STT_PARTIAL_CHAR_LIMIT: int
    STT_LAG_DROP_SECONDS: float
    STT_LAG_KEEP_SECONDS: float
    STT_PROMPT_CHARS: int
    STT_WINDOW_SECONDS: float
    STT_EMIT_INTERVAL_MS: int
    STT_OVERLAP_SECONDS: float
    STT_AGREEMENT_HITS: int
    STT_SILENCE_SECONDS: float
    STT_DUMP_FULL: bool
    STT_MIN_WINDOW_SECONDS: float
    STT_CPU_THREADS: int
    STT_NUM_WORKERS: int
    STT_NO_SPEECH_THRESHOLD: float
    STT_LOGPROB_THRESHOLD: float
    STT_COMPRESSION_RATIO_THRESHOLD: float
    STT_CONDITION_ON_PREV_TEXT: bool


settings = Settings(
    HOST=_get("HOST", "0.0.0.0"),
    PORT=_get_int("PORT", 9000),
    DEVICE=_get("DEVICE", "cpu"),
    COMPUTE_TYPE=_get("COMPUTE_TYPE", "int8"),
    DEFAULT_MODEL_SIZE=_get("DEFAULT_MODEL_SIZE", "small"),
    MAX_SESSIONS=_get_int("MAX_SESSIONS", 4),
    RTP_HOST=_require("RTP_HOST"),
    RTP_PORT_MIN=_require_int("RTP_PORT_MIN"),
    RTP_PORT_MAX=_require_int("RTP_PORT_MAX"),
    BACKEND_URL=_require("BACKEND_URL"),
    AI_KEY=_require("AI_KEY"),
    FFMPEG_LOGLEVEL=_get("FFMPEG_LOGLEVEL", "error"),
    FFMPEG_RTBUF_SIZE=_get("FFMPEG_RTBUF_SIZE", ""),
    FFMPEG_BUFFER_SIZE=_get("FFMPEG_BUFFER_SIZE", ""),
    STT_WARMUP_WINDOW_SECONDS=_get_float("STT_WARMUP_WINDOW_SECONDS", 0.45),
    STT_WARMUP_WINDOWS=_get_int("STT_WARMUP_WINDOWS", 12),
    STT_FFMPEG_READ_TIMEOUT=_get_float("STT_FFMPEG_READ_TIMEOUT", 0.5),
    STT_RTP_IDLE_MS=_get_float("STT_RTP_IDLE_MS", 0.0),
    STT_BEAM_SIZE=_get_int("STT_BEAM_SIZE", 2),
    STT_LANGUAGE=_get("STT_LANGUAGE", "vi"),
    STT_VAD_FILTER=_get_bool("STT_VAD_FILTER", False),
    STT_BUFFER_SECONDS=_get_float("STT_BUFFER_SECONDS", 8.0),
    STT_RMS_THRESHOLD=_get_float("STT_RMS_THRESHOLD", 0.005),
    STT_MIN_SPEECH_SECONDS=_get_float("STT_MIN_SPEECH_SECONDS", 0.2),
    STT_MIN_SPEECH_RATIO=_get_float("STT_MIN_SPEECH_RATIO", 0.15),
    STT_STABLE_MIN_CHARS=_get_int("STT_STABLE_MIN_CHARS", 6),
    STT_PARTIAL_CHAR_LIMIT=_get_int("STT_PARTIAL_CHAR_LIMIT", 220),
    STT_LAG_DROP_SECONDS=_get_float("STT_LAG_DROP_SECONDS", 3.0),
    STT_LAG_KEEP_SECONDS=_get_float("STT_LAG_KEEP_SECONDS", 1.5),
    STT_PROMPT_CHARS=_get_int("STT_PROMPT_CHARS", 240),
    STT_WINDOW_SECONDS=_get_float("STT_WINDOW_SECONDS", 1.2),
    STT_EMIT_INTERVAL_MS=_get_int("STT_EMIT_INTERVAL_MS", 300),
    STT_OVERLAP_SECONDS=_get_float("STT_OVERLAP_SECONDS", 0.6),
    STT_AGREEMENT_HITS=_get_int("STT_AGREEMENT_HITS", 2),
    STT_SILENCE_SECONDS=_get_float("STT_SILENCE_SECONDS", 1.0),
    STT_DUMP_FULL=_get_bool("STT_DUMP_FULL", False),
    STT_MIN_WINDOW_SECONDS=_get_float("STT_MIN_WINDOW_SECONDS", 2.0),
    STT_CPU_THREADS=_get_int("STT_CPU_THREADS", 0),
    STT_NUM_WORKERS=_get_int("STT_NUM_WORKERS", 1),
    STT_NO_SPEECH_THRESHOLD=_get_float("STT_NO_SPEECH_THRESHOLD", 0.70),
    STT_LOGPROB_THRESHOLD=_get_float("STT_LOGPROB_THRESHOLD", -1.0),
    STT_COMPRESSION_RATIO_THRESHOLD=_get_float("STT_COMPRESSION_RATIO_THRESHOLD", 2.4),
    STT_CONDITION_ON_PREV_TEXT=_get_bool("STT_CONDITION_ON_PREV_TEXT", True),
)
