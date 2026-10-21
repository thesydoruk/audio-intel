from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import SpeechEnhancer, normalize_audio


class IdentityEnhancer(SpeechEnhancer):
    name = "identity"

    def __init__(
        self, sample_rate: int = 48000, device: str = "cpu", dtype: str = "float32"
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.sample_rate = sample_rate

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate:
            audio = resample_audio(audio, sample_rate, self.sample_rate)
        return normalize_audio(audio)
