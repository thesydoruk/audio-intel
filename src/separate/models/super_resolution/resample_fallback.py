from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import SpeechSuperResolution, normalize_audio


class SoxrSuperResolution(SpeechSuperResolution):
    """Fallback bandwidth extension via high-quality resampling (not neural SR)."""

    name = "soxr_hq"
    output_sample_rate = 48000

    def __init__(
        self,
        output_sample_rate: int = 48000,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.output_sample_rate = output_sample_rate

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def upscale(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != self.output_sample_rate:
            audio = resample_audio(audio, sample_rate, self.output_sample_rate)
        return normalize_audio(audio)
