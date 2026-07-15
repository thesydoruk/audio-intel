"""Silero VAD helpers and chunk preparation for the HTTP service.

``VadPipeline`` turns raw speech regions into :class:`~audio_intel.audio.chunking.Chunk`
objects that fit Whisper's ``max_chunk_s`` window with padding.
"""

from __future__ import annotations

from audio_intel.audio.chunking import Chunk, build_chunks
from audio_intel.audio.decode import SAMPLE_RATE
from audio_intel.config import Config, VADConfig
from audio_intel.vad.regions import SpeechRegion
from audio_intel.vad.silero import SileroVAD


def _vad_config_from_server(cfg: Config) -> VADConfig:
    return VADConfig(
        sample_rate=SAMPLE_RATE,
        threshold=cfg.vad_threshold,
        min_speech_duration_ms=cfg.vad_min_speech_ms,
        min_silence_duration_ms=cfg.vad_min_silence_ms,
        speech_pad_ms=cfg.vad_speech_pad_ms,
        stream_block_s=cfg.vad_stream_block_s,
        use_onnx=cfg.vad_use_onnx,
    )


def cap_region_duration(
    regions: list[SpeechRegion],
    max_duration_s: float,
) -> list[SpeechRegion]:
    """Split long VAD regions so each span fits the Whisper chunk window."""
    if max_duration_s <= 0:
        return regions

    capped: list[SpeechRegion] = []
    for region in regions:
        start = region.start
        while start < region.end:
            end = min(region.end, start + max_duration_s)
            capped.append(SpeechRegion(start=start, end=end))
            start = end
    return capped


def regions_to_vad_samples(
    regions: list[SpeechRegion],
    sample_rate: int = SAMPLE_RATE,
) -> list[dict]:
    """Convert second-based regions to sample indices for the chunk builder."""
    return [
        {
            "start": int(round(region.start * sample_rate)),
            "end": int(round(region.end * sample_rate)),
        }
        for region in regions
    ]


def total_samples_from_duration(duration_s: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Round file duration to an integer sample count at ``sample_rate``."""
    return int(round(duration_s * sample_rate))


class VadPipeline:
    """Detect speech regions and merge them into bounded Whisper chunks."""

    def __init__(self, config: VADConfig) -> None:
        self._vad = SileroVAD(config)

    @classmethod
    def from_server(cls, cfg: Config) -> VadPipeline:
        """Build a pipeline from HTTP service ``Config`` VAD thresholds."""
        return cls(_vad_config_from_server(cfg))

    def ensure_ready(self) -> None:
        """Pre-load the Silero VAD model."""
        self._vad.ensure_ready()

    def detect_regions(self, whisper_path: str, duration_s: float) -> list[SpeechRegion]:
        """Run Silero VAD on the prepared 16 kHz workspace WAV."""
        return self._vad.detect(whisper_path, duration_s=duration_s)

    def detect_capped_regions(
        self,
        whisper_path: str,
        max_region_s: float,
        *,
        duration_s: float,
    ) -> list[SpeechRegion]:
        """Detect regions, then split any span longer than ``max_region_s``."""
        return cap_region_duration(
            self.detect_regions(whisper_path, duration_s),
            max_region_s,
        )

    def build_chunks(
        self,
        regions: list[SpeechRegion],
        duration_s: float,
        *,
        max_chunk_s: float,
        chunk_pad_s: float,
    ) -> list[Chunk]:
        """Merge VAD samples into decode chunks with ``chunk_pad_s`` overlap."""
        samples = regions_to_vad_samples(regions, SAMPLE_RATE)
        return build_chunks(
            samples,
            SAMPLE_RATE,
            total_samples_from_duration(duration_s, SAMPLE_RATE),
            max_chunk_s,
            chunk_pad_s,
        )

    def prepare_chunks_for_server(
        self,
        whisper_path: str,
        duration_s: float,
        cfg: Config,
    ) -> list[Chunk]:
        """Full HTTP path: VAD → cap regions → build chunks from server config."""
        regions = self.detect_capped_regions(
            whisper_path,
            cfg.max_chunk_s,
            duration_s=duration_s,
        )
        return self.build_chunks(
            regions,
            duration_s,
            max_chunk_s=cfg.max_chunk_s,
            chunk_pad_s=cfg.chunk_pad_s,
        )
