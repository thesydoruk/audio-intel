"""Audio decoding and VAD chunking."""

from audio_intel.audio.chunking import Chunk, build_chunks
from audio_intel.audio.decode import (
    SAMPLE_RATE,
    load_audio,
    load_audio_at,
    load_audio_window,
    probe_media_duration,
)
from audio_intel.audio.vad_bridge import regions_to_vad_samples, total_samples_from_duration
from audio_intel.audio.workspace import MediaWorkspace, convert_to_wav

__all__ = [
    "Chunk",
    "MediaWorkspace",
    "SAMPLE_RATE",
    "build_chunks",
    "convert_to_wav",
    "load_audio",
    "load_audio_at",
    "load_audio_window",
    "probe_media_duration",
    "regions_to_vad_samples",
    "total_samples_from_duration",
]
