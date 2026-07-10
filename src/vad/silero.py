"""Silero VAD: detect speech regions before Whisper chunking.

Runs independently from faster-whisper's built-in VAD. Output is a list of
:class:`SpeechRegion` time spans used by :class:`~audio_intel.vad.pipeline.VadPipeline`
to build bounded decode chunks.
"""

from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from audio_intel.audio.decode import load_audio_at
from audio_intel.config import VADConfig
from audio_intel.vad.regions import SpeechRegion
from audio_intel.vad.streaming import (
    SILERO_WINDOW_SAMPLES,
    filter_min_speech_duration,
    iter_vad_windows,
    regions_from_vad_iterator_events,
    total_samples_from_duration,
)

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("audio-intel")


def load_audio_mono(audio_path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode audio to mono float32 PCM at ``sample_rate`` (PyAV-backed)."""
    return load_audio_at(str(audio_path), sample_rate)


class SileroVAD:
    """Speech activity detection using Silero VAD (independent from Whisper)."""

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self._model = None
        self._infer_lock = threading.Lock()

    def ensure_ready(self) -> None:
        """Load the Silero VAD model into memory."""
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise ImportError(
                "silero-vad is required for speech detection. Install with: pip install silero-vad"
            ) from exc

        with self._infer_lock:
            if self._model is not None:
                return
            log.info("Loading Silero VAD model (onnx=%s)", self.config.use_onnx)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module=r"silero_vad|torchaudio"
                )
                self._model = load_silero_vad(onnx=self.config.use_onnx)
            log.info("Silero VAD model loaded")

    def detect(
        self,
        audio_path: str | Path,
        *,
        duration_s: float | None = None,
    ) -> list[SpeechRegion]:
        """Run Silero VAD on ``audio_path`` using streaming disk reads."""
        path = str(audio_path)
        if duration_s is None:
            from audio_intel.audio.decode import probe_media_duration

            duration_s = probe_media_duration(path)
        return self.detect_stream(path, duration_s)

    def detect_stream(self, audio_path: str, duration_s: float) -> list[SpeechRegion]:
        """Stream ``audio_path`` in blocks and run Silero's stateful VADIterator."""
        try:
            from silero_vad import VADIterator
        except ImportError as exc:
            raise ImportError(
                "silero-vad is required for speech detection. Install with: pip install silero-vad"
            ) from exc

        if self._model is None:
            raise RuntimeError("Silero VAD is not loaded; call ensure_ready() at startup")

        with self._infer_lock:
            sample_rate = self.config.sample_rate
            total_samples = total_samples_from_duration(duration_s, sample_rate)
            iterator = VADIterator(
                self._model,
                threshold=self.config.threshold,
                sampling_rate=sample_rate,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms,
            )

            events: list[dict[str, float]] = []
            window_count = 0
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module=r"silero_vad|torchaudio"
                )
                for window in iter_vad_windows(
                    audio_path,
                    sample_rate=sample_rate,
                    total_samples=total_samples,
                    window_samples=SILERO_WINDOW_SAMPLES,
                    block_duration_s=self.config.stream_block_s,
                ):
                    window_count += 1
                    event = iterator(window, return_seconds=True, time_resolution=3)
                    if event is not None:
                        events.append(event)

            regions = regions_from_vad_iterator_events(
                events,
                duration_s=duration_s,
                speech_active_at_end=iterator.triggered,
            )
            regions = filter_min_speech_duration(
                regions,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
            )
            log.info(
                "VAD streaming finished: %.1fs audio, %d windows (%ds blocks), %d speech regions",
                duration_s,
                window_count,
                int(self.config.stream_block_s),
                len(regions),
            )
            return regions

    def detect_waveform(self, waveform: np.ndarray, sample_rate: int) -> list[SpeechRegion]:
        """Run VAD on an already-decoded mono float32 waveform."""
        try:
            import torch
            from silero_vad import get_speech_timestamps
        except ImportError as exc:
            raise ImportError(
                "silero-vad is required for speech detection. Install with: pip install silero-vad"
            ) from exc

        if self._model is None:
            raise RuntimeError("Silero VAD is not loaded; call ensure_ready() at startup")

        with self._infer_lock:
            tensor = torch.from_numpy(waveform)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module=r"silero_vad|torchaudio"
                )
                timestamps = get_speech_timestamps(
                    tensor,
                    self._model,
                    sampling_rate=sample_rate,
                    threshold=self.config.threshold,
                    min_speech_duration_ms=self.config.min_speech_duration_ms,
                    min_silence_duration_ms=self.config.min_silence_duration_ms,
                    speech_pad_ms=self.config.speech_pad_ms,
                    return_seconds=True,
                )

            return [
                SpeechRegion(start=float(item["start"]), end=float(item["end"]))
                for item in timestamps
            ]
