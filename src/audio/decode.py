"""Audio decoding helpers built on faster-whisper's PyAV-backed decoder."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Whisper / Silero VAD both operate on 16 kHz mono PCM.
SAMPLE_RATE = 16000

# Copy WAV slices in bounded blocks so a 10-minute diarization window never
# becomes a single ~40 MB float32 array in the Python process.
_WAV_COPY_BLOCK_S = 10.0


def load_audio(path: str) -> np.ndarray:
    """Decode any ffmpeg-supported media file into a 16 kHz mono float32 array.

    PyAV handles container demuxing and resampling, so audio and (already
    audio-extracted) video inputs are both accepted.
    """
    from faster_whisper.audio import decode_audio

    return decode_audio(path, sampling_rate=SAMPLE_RATE)


def load_audio_at(path: str, sample_rate: int) -> np.ndarray:
    """Decode media into a mono float32 array at an arbitrary sample rate.

    Used by audio-event detection, which runs PANNs at 32 kHz rather than the
    16 kHz Whisper/VAD rate.
    """
    from faster_whisper.audio import decode_audio

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


def _as_writable_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Return C-contiguous writable mono float32 (required by ctranslate2)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1:
        audio = np.reshape(audio, -1)
    if not audio.flags.c_contiguous or not audio.flags.writeable:
        return np.array(audio, dtype=np.float32, copy=True, order="C")
    return audio


def _read_wav_window(
    path: str, sample_rate: int, start_s: float, duration_s: float
) -> np.ndarray | None:
    """Read a PCM WAV slice via libsndfile, or ``None`` to fall back to ffmpeg."""
    start_frame = int(round(start_s * sample_rate))
    n_frames = int(round(duration_s * sample_rate))
    if n_frames < 1:
        return np.array([], dtype=np.float32)
    try:
        with sf.SoundFile(path) as handle:
            if handle.samplerate != sample_rate:
                return None
            if start_frame >= handle.frames:
                return np.array([], dtype=np.float32)
            handle.seek(start_frame)
            n = min(n_frames, max(handle.frames - start_frame, 0))
            if n < 1:
                return np.array([], dtype=np.float32)
            audio = handle.read(n, dtype="float32", always_2d=False)
    except (OSError, RuntimeError, ValueError):
        return None
    if getattr(audio, "size", 0) == 0:
        return np.array([], dtype=np.float32)
    return _as_writable_mono_float32(audio)


def _load_audio_window_ffmpeg(
    path: str, sample_rate: int, start_s: float, duration_s: float
) -> np.ndarray:
    """Decode a slice through ffmpeg stdout (used when the file is not a matching WAV)."""
    try:
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
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return np.array([], dtype=np.float32)
    if result.returncode != 0 or not result.stdout:
        return np.array([], dtype=np.float32)
    # Copy: np.frombuffer is read-only; ctranslate2/faster-whisper segfaults on
    # non-writable float32 views (exit 139) when used as Whisper chunk input.
    pcm = result.stdout
    del result
    audio = np.frombuffer(pcm, dtype=np.float32).copy()
    del pcm
    return audio


def load_audio_window(path: str, sample_rate: int, start_s: float, duration_s: float) -> np.ndarray:
    """Decode a mono float32 slice without loading the full file into memory.

    Prepared workspace WAVs are read with libsndfile (seek + slice) so ffmpeg is
    not spawned and stdout is not buffered as a second copy of the PCM. Returns
    an empty array at/past EOF or for sub-sample windows. ffmpeg seeks past the
    end of short clips can hang for a long time; a timeout turns that into an
    empty read so VAD/AED loops can terminate.
    """
    if duration_s <= 0 or start_s < 0:
        return np.array([], dtype=np.float32)
    # One PCM float32 sample; shorter requests are noise and can hang ffmpeg.
    if duration_s * sample_rate < 1.0:
        return np.array([], dtype=np.float32)
    wav = _read_wav_window(path, sample_rate, start_s, duration_s)
    if wav is not None:
        return wav
    return _load_audio_window_ffmpeg(path, sample_rate, start_s, duration_s)


def load_audio_spans(
    path: str,
    spans: list[tuple[float, float]],
    sample_rate: int,
    *,
    max_duration_s: float | None = None,
) -> np.ndarray:
    """Concatenate time spans from disk without decoding the rest of the file.

    Used for speaker embeddings: at most ``max_duration_s`` of speech per
    speaker (typically 30 s), not the whole recording.
    """
    if not spans:
        return np.array([], dtype=np.float32)

    budget = max_duration_s if max_duration_s is not None and max_duration_s > 0 else None
    chunks: list[np.ndarray] = []

    wav = _try_open_matching_wav(path, sample_rate)
    try:
        for start, end in spans:
            take = max(0.0, end - start)
            if take <= 0:
                continue
            if budget is not None:
                take = min(take, budget)
                if take <= 0:
                    break
            if wav is not None:
                piece = _read_open_wav_window(wav, sample_rate, start, take)
            else:
                piece = load_audio_window(path, sample_rate, start, take)
            if piece.size:
                chunks.append(piece)
                if budget is not None:
                    budget -= take
    finally:
        if wav is not None:
            wav.close()

    if not chunks:
        return np.array([], dtype=np.float32)
    return _as_writable_mono_float32(np.concatenate(chunks))


def _try_open_matching_wav(path: str, sample_rate: int) -> sf.SoundFile | None:
    try:
        handle = sf.SoundFile(path)
    except (OSError, RuntimeError, ValueError):
        return None
    if handle.samplerate != sample_rate:
        handle.close()
        return None
    return handle


def _read_open_wav_window(
    handle: sf.SoundFile, sample_rate: int, start_s: float, duration_s: float
) -> np.ndarray:
    start_frame = int(round(start_s * sample_rate))
    n_frames = int(round(duration_s * sample_rate))
    if n_frames < 1 or start_frame >= handle.frames:
        return np.array([], dtype=np.float32)
    handle.seek(max(start_frame, 0))
    n = min(n_frames, max(handle.frames - max(start_frame, 0), 0))
    if n < 1:
        return np.array([], dtype=np.float32)
    audio = handle.read(n, dtype="float32", always_2d=False)
    if getattr(audio, "size", 0) == 0:
        return np.array([], dtype=np.float32)
    return _as_writable_mono_float32(audio)


def extract_wav_window(
    source_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int,
    start_s: float,
    duration_s: float,
) -> None:
    """Write a mono PCM WAV slice to ``output_path`` without a full-file float32 buffer.

    Matching WAV sources are copied in ``_WAV_COPY_BLOCK_S`` blocks via libsndfile.
    Other containers fall back to ffmpeg writing the file directly (no Python PCM).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if duration_s <= 0 or start_s < 0:
        raise ValueError("extract_wav_window requires start_s >= 0 and duration_s > 0")
    if _extract_wav_window_soundfile(
        str(source_path), str(output_path), sample_rate, start_s, duration_s
    ):
        return
    _extract_wav_window_ffmpeg(str(source_path), str(output_path), sample_rate, start_s, duration_s)


def _extract_wav_window_soundfile(
    source_path: str,
    output_path: str,
    sample_rate: int,
    start_s: float,
    duration_s: float,
) -> bool:
    start_frame = int(round(start_s * sample_rate))
    n_frames = int(round(duration_s * sample_rate))
    if n_frames < 1:
        return False
    try:
        with sf.SoundFile(source_path) as src:
            if src.samplerate != sample_rate:
                return False
            if start_frame >= src.frames:
                return False
            src.seek(start_frame)
            remaining = min(n_frames, src.frames - start_frame)
            block = max(int(round(_WAV_COPY_BLOCK_S * sample_rate)), sample_rate)
            with sf.SoundFile(
                output_path,
                mode="w",
                samplerate=sample_rate,
                channels=1,
                subtype="PCM_16",
            ) as dst:
                while remaining > 0:
                    n = min(block, remaining)
                    data = src.read(n, dtype="float32", always_2d=False)
                    if getattr(data, "size", 0) == 0:
                        break
                    if getattr(data, "ndim", 1) > 1:
                        data = np.mean(data, axis=1, dtype=np.float32)
                    dst.write(data)
                    remaining -= len(data)
    except (OSError, RuntimeError, ValueError):
        return False
    return Path(output_path).is_file() and Path(output_path).stat().st_size > 0


def _extract_wav_window_ffmpeg(
    source_path: str,
    output_path: str,
    sample_rate: int,
    start_s: float,
    duration_s: float,
) -> None:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-i",
            source_path,
            "-t",
            f"{duration_s:.6f}",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output_path,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract WAV window from {source_path}:\n{proc.stderr}"
        )


def write_temp_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Persist decoded PCM to a temporary WAV file for pyannote I/O."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sample_rate)
    return path
