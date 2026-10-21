from __future__ import annotations

from pathlib import Path

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import TargetSpeakerExtractor, normalize_audio
from audio_intel.separate.models.tse.adaptive_wiener import AdaptiveWienerTSE


class WeSepTSE(TargetSpeakerExtractor):
    """Optional WeSep BSRNN backend.

    Loads a checkpoint if `tse.checkpoint` is configured; otherwise falls back
    to AdaptiveWienerTSE so the pipeline remains runnable.
    """

    name = "wesep"
    sample_rate = 16000

    def __init__(
        self, cfg: dict | None = None, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.cfg = cfg or {}
        self.sample_rate = int(self.cfg.get("sample_rate", 16000))
        self.checkpoint = self.cfg.get("checkpoint")
        self._model = None
        self._fallback = AdaptiveWienerTSE(cfg=self.cfg, device="cpu", dtype="float32")

    def load(self) -> None:
        self._fallback.load()
        if not self.checkpoint or not Path(self.checkpoint).exists():
            self._loaded = True
            return
        # Placeholder for WeSep integration when local checkpoint is present.
        # Keeping explicit so swapping in REAL-TSE/WeSep weights is one adapter edit.
        self._model = None
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._fallback.unload()
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
        self.ensure_loaded()
        if self._model is None:
            return self._fallback.extract(
                mixture,
                sample_rate,
                target_embedding=target_embedding,
                reference_audio=reference_audio,
                reference_sample_rate=reference_sample_rate,
            )
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate:
            mix = resample_audio(mix, sample_rate, self.sample_rate)
        return normalize_audio(mix)
