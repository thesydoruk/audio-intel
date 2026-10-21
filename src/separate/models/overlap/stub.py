from __future__ import annotations

import numpy as np
from audio_intel.separate.models.base import OverlapDetector
from audio_intel.separate.models.overlap.hybrid import coactivity_overlaps
from audio_intel.separate.types import DiarizationResult, OverlapRegion


class StubOverlapDetector(OverlapDetector):
    name = "stub"

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        offset: float = 0.0,
        diarization: DiarizationResult | None = None,
    ) -> list[OverlapRegion]:
        _ = audio, sample_rate, offset
        if diarization is None:
            return []
        return coactivity_overlaps(diarization, min_overlap_duration=0.1)
