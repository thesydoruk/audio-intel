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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
from audio_intel.audio.decode import SAMPLE_RATE, load_audio_window, probe_media_duration

log = logging.getLogger("audio-intel")

WHISPER_WAV_NAME = "whisper_16k.wav"


def convert_to_wav(
    source_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int,
    channels: int = 1,
) -> None:
    """Decode any ffmpeg-supported input into a mono PCM WAV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to convert {source_path} to {sample_rate} Hz WAV:\n{proc.stderr}"
        )


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
        duration_s = probe_media_duration(str(source_path))
        whisper_16k = root / WHISPER_WAV_NAME

        log.info(
            "Preparing media workspace: %s (%.1fs) → %s",
            source_path.name,
            duration_s,
            root,
        )

        aed_32k: Path | None = None
        if need_aed:
            aed_32k = root / f"aed_{aed_sample_rate}.wav"
            with ThreadPoolExecutor(max_workers=2) as pool:
                whisper_future = pool.submit(
                    convert_to_wav,
                    source_path,
                    whisper_16k,
                    sample_rate=SAMPLE_RATE,
                )
                aed_future = pool.submit(
                    convert_to_wav,
                    source_path,
                    aed_32k,
                    sample_rate=aed_sample_rate,
                )
                whisper_future.result()
                aed_future.result()
        else:
            convert_to_wav(source_path, whisper_16k, sample_rate=SAMPLE_RATE)

        workspace = cls(
            source_path=source_path,
            root=root,
            duration_s=duration_s,
            whisper_16k=whisper_16k,
            aed_32k=aed_32k,
            aed_sample_rate=aed_sample_rate,
        )
        log.info(
            "Media workspace ready: whisper_16k=%s%s",
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
