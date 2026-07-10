"""Audio decoding helpers built on faster-whisper's PyAV-backed decoder."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from faster_whisper.audio import decode_audio

# Whisper / Silero VAD both operate on 16 kHz mono PCM.
SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Decode any ffmpeg-supported media file into a 16 kHz mono float32 array.

    PyAV handles container demuxing and resampling, so audio and (already
    audio-extracted) video inputs are both accepted.
    """
    return decode_audio(path, sampling_rate=SAMPLE_RATE)


def load_audio_at(path: str, sample_rate: int) -> np.ndarray:
    """Decode media into a mono float32 array at an arbitrary sample rate.

    Used by audio-event detection, which runs PANNs at 32 kHz rather than the
    16 kHz Whisper/VAD rate.
    """
    return decode_audio(path, sampling_rate=sample_rate)


def probe_media_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def load_audio_window(path: str, sample_rate: int, start_s: float, duration_s: float) -> np.ndarray:
    """Decode a mono float32 slice without loading the full file into memory."""
    if duration_s <= 0:
        return np.array([], dtype=np.float32)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-i",
            path,
            "-t",
            f"{duration_s:.6f}",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def write_temp_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Persist decoded PCM to a temporary WAV file for pyannote I/O."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sample_rate)
    return path
