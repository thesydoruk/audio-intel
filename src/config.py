"""Configuration for the HTTP transcription service."""

from __future__ import annotations

import os
from dataclasses import dataclass

from audio_intel.env_file import load_env_file

load_env_file()


def _str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name, "").lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _strs(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _str(name, "")
    if not raw:
        return default
    out = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(out) if out else default


def _floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = _str(name, "")
    if not raw:
        return default
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            return default
    return tuple(out) if out else default


@dataclass(frozen=True)
class Config:
    """Resolved HTTP service configuration from environment variables."""

    model: str
    device: str
    compute_type: str
    beam_size: int
    cpu_threads: int
    whisper_parallel_workers: int
    max_concurrent_requests: int
    download_root: str
    forced_language: str | None
    lang_min_prob: float
    temperatures: tuple[float, ...]
    compression_ratio_threshold: float
    log_prob_threshold: float
    no_speech_threshold: float
    hallucination_silence_s: float
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    vad_stream_block_s: float
    vad_use_onnx: bool
    max_chunk_s: float
    chunk_pad_s: float
    aed_enabled: bool
    aed_device: str
    aed_sample_rate: int
    aed_window_s: float
    aed_overlap_s: float
    aed_min_score: float
    aed_top_k: int
    aed_merge_gap_s: float
    aed_min_duration_s: float
    aed_exclude_speech: bool
    aed_prompt_allowlist: tuple[str, ...]
    aed_parallel_workers: int
    diarization_enabled: bool
    diarization_device: str
    diarization_pipeline_workers: int
    diarization_embed_workers: int
    diarization_chunk_s: float
    diarization_chunk_overlap_s: float
    diarization_link_threshold: float
    diarization_link_min_speech_s: float
    hf_token: str | None
    port: int
    log_level: str
    # WhisperX CTC alignment (optional; gated by per-request `align` flag).
    alignment_enabled: bool
    alignment_device: str | None
    alignment_model: str | None
    alignment_interpolate_method: str
    alignment_language: str | None


@dataclass
class VADConfig:
    """Silero VAD thresholds."""

    sample_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 400
    speech_pad_ms: int = 200
    # ONNX avoids loading PyTorch Silero in the same process as CUDA ctranslate2
    # Whisper — that mix segfaults (exit 139) right after VAD on some hosts.
    use_onnx: bool = True
    stream_block_s: float = 30.0


@dataclass
class AlignmentConfig:
    """WhisperX CTC alignment options."""

    enabled: bool = True
    device: str | None = None
    model: str | None = None
    interpolate_method: str = "nearest"


def load_config() -> Config:
    """Build server :class:`Config` from environment variables (after ``.env`` load)."""
    language = _str("ASR_LANGUAGE", "auto")
    forced = None if language.lower() in {"", "auto", "none"} else language
    download_root = _str("MODEL_CACHE_DIR", "/var/lib/whisper")
    hf_token_raw = _str("HF_TOKEN", "")

    return Config(
        model=_str("ASR_MODEL", "large-v3"),
        device=_str("ASR_DEVICE", "cuda"),
        compute_type=_str("ASR_COMPUTE_TYPE", "float16"),
        beam_size=_int("ASR_BEAM_SIZE", 5),
        cpu_threads=_int("ASR_CPU_THREADS", 2),
        whisper_parallel_workers=max(1, min(_int("ASR_PARALLEL_WORKERS", 1), 8)),
        max_concurrent_requests=max(1, min(_int("MAX_CONCURRENT_REQUESTS", 2), 8)),
        download_root=download_root,
        forced_language=forced,
        lang_min_prob=_float("ASR_LANGUAGE_MIN_CONFIDENCE", 0.5),
        temperatures=_floats("ASR_TEMPERATURES", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        compression_ratio_threshold=_float("ASR_MAX_COMPRESSION_RATIO", 2.4),
        log_prob_threshold=_float("ASR_MIN_LOG_PROB", -1.0),
        no_speech_threshold=_float("ASR_NO_SPEECH_THRESHOLD", 0.6),
        hallucination_silence_s=_float("ASR_SKIP_SILENCE_LONGER_THAN_S", 2.0),
        vad_threshold=_float("VAD_SPEECH_THRESHOLD", 0.5),
        vad_min_speech_ms=_int("VAD_MIN_SPEECH_MS", 250),
        vad_min_silence_ms=_int("VAD_MIN_SILENCE_MS", 400),
        vad_speech_pad_ms=_int("VAD_SPEECH_PAD_MS", 200),
        vad_stream_block_s=_float("VAD_STREAM_BLOCK_S", 30.0),
        # Default ONNX: PyTorch Silero + CUDA ctranslate2 Whisper segfaults on some GPUs.
        vad_use_onnx=_bool("VAD_USE_ONNX", True),
        max_chunk_s=_float("VAD_MAX_CHUNK_S", 28.0),
        chunk_pad_s=_float("VAD_CHUNK_PAD_S", 0.2),
        aed_enabled=_bool("SOUND_EVENTS_ENABLED", True),
        aed_device=_str("SOUND_EVENTS_DEVICE", "cpu"),
        aed_sample_rate=_int("SOUND_EVENTS_SAMPLE_RATE", 32000),
        aed_window_s=_float("SOUND_EVENTS_WINDOW_S", 60.0),
        aed_overlap_s=_float("SOUND_EVENTS_OVERLAP_S", 1.0),
        aed_min_score=_float("SOUND_EVENTS_MIN_SCORE", 0.35),
        aed_top_k=_int("SOUND_EVENTS_TOP_K", 3),
        aed_merge_gap_s=_float("SOUND_EVENTS_MERGE_GAP_S", 0.6),
        aed_min_duration_s=_float("SOUND_EVENTS_MIN_DURATION_S", 0.5),
        aed_exclude_speech=_bool("SOUND_EVENTS_EXCLUDE_SPEECH", True),
        aed_prompt_allowlist=_strs("SOUND_EVENTS_PROMPT_ALLOWLIST", ()),
        aed_parallel_workers=max(1, min(_int("SOUND_EVENTS_PARALLEL_WORKERS", 1), 8)),
        diarization_enabled=_bool("SPEAKERS_ENABLED", False),
        diarization_device=_str("SPEAKERS_DEVICE", "cuda"),
        diarization_pipeline_workers=max(1, min(_int("SPEAKERS_PARALLEL_WORKERS", 1), 4)),
        diarization_embed_workers=max(1, min(_int("SPEAKERS_EMBED_WORKERS", 1), 8)),
        diarization_chunk_s=_float("SPEAKERS_CHUNK_S", 600.0),
        diarization_chunk_overlap_s=_float("SPEAKERS_CHUNK_OVERLAP_S", 30.0),
        diarization_link_threshold=_float("SPEAKERS_LINK_THRESHOLD", 0.75),
        diarization_link_min_speech_s=_float("SPEAKERS_LINK_MIN_SPEECH_S", 0.5),
        hf_token=hf_token_raw or None,
        port=_int("AUDIO_INTEL_PORT", 8080),
        log_level=_str("LOG_LEVEL", "INFO"),
        alignment_enabled=_bool("WORD_ALIGN_ENABLED", False),
        alignment_device=_str("WORD_ALIGN_DEVICE", "") or None,
        alignment_model=_str("WORD_ALIGN_MODEL", "") or None,
        alignment_interpolate_method=_str("WORD_ALIGN_INTERPOLATE_METHOD", "nearest"),
        alignment_language=_str("WORD_ALIGN_LANGUAGE", "") or None,
    )
