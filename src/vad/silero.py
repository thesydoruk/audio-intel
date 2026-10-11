"""Silero VAD: detect speech regions before Whisper chunking.

Runs independently from faster-whisper's built-in VAD. Output is a list of
:class:`SpeechRegion` time spans used by :class:`~audio_intel.vad.pipeline.VadPipeline`
to build bounded decode chunks.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from audio_intel.common.model_pool import ModelPool
from audio_intel.config import VADConfig
from audio_intel.vad.regions import SpeechRegion
from audio_intel.vad.streaming import (
    SILERO_WINDOW_SAMPLES,
    filter_min_speech_duration,
    iter_vad_windows,
    regions_from_vad_iterator_events,
    total_samples_from_duration,
)

log = logging.getLogger("audio-intel")


class SileroVAD:
    """Speech activity detection using Silero VAD (independent from Whisper).

    Silero is stateful per stream, so the pool hands each request its own copy
    (a few MB each) instead of serializing every request behind one model.
    """

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self._pool: ModelPool[object] = ModelPool(self._create_model)

    def _create_model(self) -> object:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise ImportError(
                "silero-vad is required for speech detection. Install with: pip install silero-vad"
            ) from exc

        log.info("Loading Silero VAD model (onnx=%s)", self.config.use_onnx)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"silero_vad|torchaudio")
            model = load_silero_vad(onnx=self.config.use_onnx)
        log.info("Silero VAD model loaded")
        return model

    def ensure_ready(self, num_workers: int = 1) -> None:
        """Load ``num_workers`` Silero VAD copies into memory."""
        self._pool.ensure_ready(num_workers)

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

        if self._pool.size == 0:
            raise RuntimeError("Silero VAD is not loaded; call ensure_ready() at startup")

        sample_rate = self.config.sample_rate
        total_samples = total_samples_from_duration(duration_s, sample_rate)
        model = self._pool.acquire()
        try:
            iterator = VADIterator(
                model,
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
            speech_active_at_end = bool(iterator.triggered)
        finally:
            self._pool.release(model)

        regions = regions_from_vad_iterator_events(
            events,
            duration_s=duration_s,
            speech_active_at_end=speech_active_at_end,
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
