"""Prepare arbitrary media into pipeline-ready PCM files for the duration of one run.

``MediaWorkspace`` decodes the upload once into ``whisper_16k.wav`` (and optionally
``aed_32k.wav``) so Whisper, VAD, alignment, and PANNs all read the same PCM
without repeated ffmpeg calls.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import soundfile as sf
from audio_intel.audio.decode import SAMPLE_RATE, load_audio_window, probe_media_duration

log = logging.getLogger("audio-intel")

WHISPER_WAV_NAME = "whisper_16k.wav"

# ffmpeg decodes audio far faster than real time; a run that takes longer than the
# media itself is stuck (corrupt stream, blocked pipe) and must not hold a request
# slot forever. ``_MIN_CONVERT_TIMEOUT_S`` covers short clips and process startup.
_MIN_CONVERT_TIMEOUT_S = 300.0


def convert_timeout_s(duration_s: float) -> float:
    """Wall-clock budget for decoding ``duration_s`` seconds of media."""
    return max(_MIN_CONVERT_TIMEOUT_S, 2.0 * max(duration_s, 0.0))


def convert_to_wavs(
    source_path: str | Path,
    outputs: list[tuple[str | Path, int]],
    *,
    channels: int = 1,
    timeout_s: float | None = None,
) -> None:
    """Decode one ffmpeg-supported input into mono PCM WAVs at several sample rates.

    A single ffmpeg process demuxes and decodes the source once and resamples it
    into every ``(output_path, sample_rate)`` requested, so a video upload is not
    decoded twice for the 16 kHz and 32 kHz copies.
    """
    if not outputs:
        raise ValueError("convert_to_wavs needs at least one output")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path)]
    for output_path, sample_rate in outputs:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += [
            "-vn",
            "-sn",
            "-dn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg did not finish decoding {source_path} within {timeout_s:.0f}s"
        ) from exc
    if proc.returncode != 0:
        rates = ", ".join(str(rate) for _, rate in outputs)
        raise RuntimeError(
            f"ffmpeg failed to convert {source_path} to {rates} Hz WAV:\n{proc.stderr}"
        )


def convert_to_wav(
    source_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int,
    channels: int = 1,
    timeout_s: float | None = None,
) -> None:
    """Decode any ffmpeg-supported input into one mono PCM WAV file."""
    convert_to_wavs(
        source_path, [(output_path, sample_rate)], channels=channels, timeout_s=timeout_s
    )


def wav_duration_s(path: str | Path) -> float:
    """Exact duration of a PCM WAV from its header (no subprocess)."""
    info = sf.info(str(path))
    return info.frames / info.samplerate if info.samplerate else 0.0


@dataclass
class MediaWorkspace:
    """Ephemeral decoded copies of one input file.

    Created at the start of a pipeline run; every downstream stage reads from
    ``whisper_16k`` / ``aed_32k`` instead of re-decoding the original upload.
    Removed automatically when the workspace is closed.
    """

    source_path: Path
    root: Path
    duration_s: float
    whisper_16k: Path
    aed_32k: Path | None = None
    aed_sample_rate: int = 32000

    @classmethod
    def prepare(
        cls,
        source_path: str | Path,
        *,
        need_aed: bool = False,
        aed_sample_rate: int = 32000,
    ) -> MediaWorkspace:
        """Decode ``source_path`` into all PCM variants needed for one pipeline run."""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        root = Path(tempfile.mkdtemp(prefix="audio-intel-"))
        try:
            # Only a hint for the decode timeout: streams without a container
            # duration report nothing, and the prepared WAV is the real timeline.
            source_duration_s = _probe_duration_or_zero(source_path)
            whisper_16k = root / WHISPER_WAV_NAME
            log.info(
                "Preparing media workspace: %s (%.1fs) → %s",
                source_path.name,
                source_duration_s,
                root,
            )

            outputs: list[tuple[Path, int]] = [(whisper_16k, SAMPLE_RATE)]
            aed_32k: Path | None = None
            if need_aed:
                aed_32k = root / f"aed_{aed_sample_rate}.wav"
                outputs.append((aed_32k, aed_sample_rate))
            convert_to_wavs(
                source_path,
                outputs,
                timeout_s=convert_timeout_s(source_duration_s),
            )

            # Container probe can be slightly longer than the decoded WAV; VAD must
            # use the prepared whisper timeline or the last window seeks past EOF.
            duration_s = wav_duration_s(whisper_16k)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise

        workspace = cls(
            source_path=source_path,
            root=root,
            duration_s=duration_s,
            whisper_16k=whisper_16k,
            aed_32k=aed_32k,
            aed_sample_rate=aed_sample_rate,
        )
        log.info(
            "Media workspace ready: %.1fs, whisper_16k=%s%s",
            duration_s,
            whisper_16k.name,
            f", aed_{aed_sample_rate}={aed_32k.name}" if aed_32k else "",
        )
        return workspace

    @property
    def whisper_path(self) -> str:
        """16 kHz mono WAV path for Whisper, VAD, alignment, and diarization."""
        return str(self.whisper_16k)

    @property
    def aed_path(self) -> str | None:
        """32 kHz mono WAV path for PANNs sound-event detection."""
        return str(self.aed_32k) if self.aed_32k is not None else None

    def read_whisper_window(self, start_s: float, end_s: float) -> np.ndarray:
        """Load a ``[start_s, end_s]`` slice from the prepared 16 kHz WAV."""
        duration_s = max(0.0, end_s - start_s)
        return load_audio_window(self.whisper_path, SAMPLE_RATE, start_s, duration_s)

    def read_aed_window(self, start_s: float, duration_s: float) -> np.ndarray:
        """Load a slice from the prepared AED WAV."""
        if self.aed_32k is None:
            raise RuntimeError("AED WAV was not prepared for this workspace")
        return load_audio_window(self.aed_path, self.aed_sample_rate, start_s, duration_s)  # type: ignore[arg-type]

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            log.info("Media workspace removed: %s", self.root)

    def __enter__(self) -> MediaWorkspace:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _probe_duration_or_zero(path: Path) -> float:
    try:
        return probe_media_duration(str(path))
    except (subprocess.CalledProcessError, ValueError, OSError):
        return 0.0
