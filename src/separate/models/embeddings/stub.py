from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import SpeakerEmbedder


class StubEmbedder(SpeakerEmbedder):
    name = "stub"
    embedding_dim = 32

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            audio = resample_audio(audio, sample_rate, 16000)
        # Deterministic pseudo-embedding from spectral moments
        n_fft = 512
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        spec = np.abs(np.fft.rfft(audio[: n_fft * (len(audio) // n_fft)]))
        bins = np.array_split(spec, self.embedding_dim)
        vec = np.array([float(np.mean(b)) for b in bins], dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec
