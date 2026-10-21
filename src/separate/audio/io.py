from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from audio_intel.separate.types import AudioInfo


def _ffmpeg_available() -> bool:
    return shutil.which("ffprobe") is not None and shutil.which("ffmpeg") is not None


def probe_audio(path: str | Path) -> AudioInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        info = sf.info(str(path))
        return AudioInfo(
            path=str(path.resolve()),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            duration=float(info.duration),
            num_frames=int(info.frames),
            format=str(info.format),
            subtype=str(info.subtype),
        )
    except (RuntimeError, sf.LibsndfileError):
        if not _ffmpeg_available():
            raise RuntimeError(
                f"Cannot decode {path} with libsndfile and ffmpeg/ffprobe are not on PATH. "
                "Install ffmpeg or convert the file to WAV/FLAC first."
            ) from None
        return _probe_with_ffprobe(path)


def _probe_with_ffprobe(path: Path) -> AudioInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    sr = int(float(stream.get("sample_rate", 48000)))
    ch = int(stream.get("channels", 1))
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    return AudioInfo(
        path=str(path.resolve()),
        sample_rate=sr,
        channels=ch,
        duration=duration,
        num_frames=int(round(duration * sr)),
        format=str(stream.get("codec_name", "unknown")),
        subtype="ffmpeg",
    )


def resolve_channels(source_channels: int, layout: str | int) -> int:
    """Map a ``master_channels`` setting to a concrete channel count (1 or 2)."""
    if isinstance(layout, int):
        return max(1, min(2, layout))
    layout = str(layout).lower()
    if layout in {"mono", "1"}:
        return 1
    if layout in {"stereo", "2"}:
        return 2
    return max(1, min(2, int(source_channels)))


def materialize_master_wav(
    source: str | Path,
    dest: str | Path,
    *,
    target_sr: int = 48000,
    channels: str | int = "keep",
    subtype: str = "PCM_24",
) -> AudioInfo:
    """Decode the source once into a lossless master WAV. Never overwrites the source."""
    source = Path(source).resolve()
    dest = Path(dest).resolve()
    if source == dest:
        raise ValueError("Refusing to overwrite the original audio file")
    dest.parent.mkdir(parents=True, exist_ok=True)

    info = probe_audio(source)
    n_ch = resolve_channels(info.channels, channels)

    if info.subtype != "ffmpeg":
        # libsndfile can decode it: read, fix layout, resample if needed.
        audio, sr = sf.read(str(source), always_2d=True, dtype="float32")
        audio = _fix_layout(audio, n_ch)
        if sr != target_sr:
            from audio_intel.separate.audio.resample import resample_audio

            audio = resample_audio(audio, sr, target_sr)
        sf.write(str(dest), audio, target_sr, subtype=subtype)
        return probe_audio(dest)

    codec = {"PCM_16": "pcm_s16le", "PCM_24": "pcm_s24le", "FLOAT": "pcm_f32le"}.get(
        subtype, "pcm_s24le"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(n_ch),
        "-ar",
        str(target_sr),
        "-c:a",
        codec,
        str(dest),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    return probe_audio(dest)


def _fix_layout(audio: np.ndarray, n_ch: int) -> np.ndarray:
    """``audio`` is (T, C). Return (T, n_ch)."""
    if audio.ndim == 1:
        audio = audio[:, None]
    c = audio.shape[1]
    if c == n_ch:
        return audio
    if n_ch == 1:
        return audio.mean(axis=1, keepdims=True)
    if c == 1:
        return np.repeat(audio, n_ch, axis=1)
    # c > n_ch == 2: keep the first two channels (L/R) rather than a lossy fold-down.
    return audio[:, :n_ch]


def read_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Read a whole file as float32. Returns (T, C), or (T,) when ``mono``."""
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if mono:
        audio = audio.mean(axis=1)
    if sample_rate is not None and sample_rate != sr:
        from audio_intel.separate.audio.resample import resample_audio

        audio = resample_audio(audio, sr, sample_rate)
        sr = sample_rate
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def read_mono_resampled(
    path: str | Path, target_sr: int, block_seconds: float = 600.0
) -> np.ndarray:
    """Stream a long file into a mono array at ``target_sr`` with bounded memory."""
    info = sf.info(str(path))
    sr = int(info.samplerate)
    if sr == target_sr:
        audio, _ = read_audio(path, mono=True)
        return audio
    stream = soxr.ResampleStream(sr, target_sr, 1, dtype="float32", quality="HQ")
    out: list[np.ndarray] = []
    block = int(block_seconds * sr)
    with sf.SoundFile(str(path)) as f:
        while True:
            data = f.read(block, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            mono = np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)
            last = f.tell() >= f.frames
            out.append(stream.resample_chunk(mono, last=last))
            if last:
                break
    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out).astype(np.float32)


def read_segment(
    path: str | Path,
    start: float,
    end: float,
    *,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read [start, end) seconds as mono float32. Returns (audio, sample_rate)."""
    path = Path(path)
    info = sf.info(str(path))
    sr = int(info.samplerate)
    start_frame = max(0, int(round(start * sr)))
    end_frame = min(int(info.frames), int(round(end * sr)))
    frames = max(0, end_frame - start_frame)
    if frames == 0:
        mono = np.zeros(0, dtype=np.float32)
    else:
        audio, _ = sf.read(
            str(path), start=start_frame, frames=frames, always_2d=True, dtype="float32"
        )
        mono = audio.mean(axis=1).astype(np.float32)
    if sample_rate is not None and sample_rate != sr:
        from audio_intel.separate.audio.resample import resample_audio

        mono = resample_audio(mono, sr, sample_rate)
        return mono, sample_rate
    return mono, sr


def write_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "PCM_24",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype=subtype)


def allocate_silent_track(num_samples: int) -> np.ndarray:
    return np.zeros(int(num_samples), dtype=np.float32)


def with_temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    write_wav(tmp.name, audio, sample_rate)
    return Path(tmp.name)
