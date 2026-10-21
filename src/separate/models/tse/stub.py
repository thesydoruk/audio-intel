from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import TargetSpeakerExtractor, normalize_audio


class StubTSE(TargetSpeakerExtractor):
    name = "stub"
    sample_rate = 16000

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def extract(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        *,
        target_embedding: np.ndarray | None = None,
        reference_audio: np.ndarray | None = None,
        reference_sample_rate: int | None = None,
    ) -> np.ndarray:
        _ = target_embedding, reference_audio, reference_sample_rate
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate:
            mix = resample_audio(mix, sample_rate, self.sample_rate)
        return normalize_audio(mix * 0.9)
