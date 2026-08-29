"""Re-exports from audio_intel.vad.pipeline."""

from __future__ import annotations

from audio_intel.vad.pipeline import (
    cap_region_duration,
    regions_to_vad_samples,
    total_samples_from_duration,
)

__all__ = ["cap_region_duration", "regions_to_vad_samples", "total_samples_from_duration"]
