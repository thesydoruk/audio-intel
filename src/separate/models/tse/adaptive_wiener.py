from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import TargetSpeakerExtractor, normalize_audio
from scipy import signal


class AdaptiveWienerTSE(TargetSpeakerExtractor):
    """Audio-only target speaker extraction conditioned on enrollment audio.

    Builds a speaker spectral prior from the reference and applies a soft
    Wiener-like mask to the mixture. Works for arbitrary speaker counts and
    does not assume a fixed-N blind separator.
    """

    name = "adaptive_wiener"
    sample_rate = 16000

    def __init__(
        self, cfg: dict | None = None, device: str = "cpu", dtype: str = "float32"
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        cfg = cfg or {}
        self.sample_rate = int(cfg.get("sample_rate", 16000))
        self.n_fft = int(cfg.get("n_fft", 512))
        self.hop = int(cfg.get("hop_length", 128))
        self.floor = float(cfg.get("wiener_floor", 0.05))
        self.power = float(cfg.get("wiener_power", 1.0))

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        _, _, zxx = signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop,
            window="hann",
            boundary="zeros",
            padded=True,
        )
        return zxx

    def _istft(self, zxx: np.ndarray, length: int) -> np.ndarray:
        _, x = signal.istft(
            zxx,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop,
            window="hann",
            boundary="zeros",
        )
        x = np.asarray(x, dtype=np.float32)
        if len(x) < length:
            x = np.pad(x, (0, length - len(x)))
        return x[:length]

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
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate:
            mix = resample_audio(mix, sample_rate, self.sample_rate)

        if (
            reference_audio is None
            or len(np.asarray(reference_audio).reshape(-1)) < self.sample_rate // 4
        ):
            # No usable enrollment: return lightly gated mixture
            return normalize_audio(mix)

        ref = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
        ref_sr = int(reference_sample_rate or sample_rate)
        if ref_sr != self.sample_rate:
            ref = resample_audio(ref, ref_sr, self.sample_rate)

        mix_stft = self._stft(mix)
        ref_stft = self._stft(ref)
        mix_mag = np.abs(mix_stft)
        ref_mag = np.abs(ref_stft)

        # Speaker prior: median spectrum of enrollment, broadcast over time
        prior = np.median(ref_mag, axis=1, keepdims=True) + 1e-8
        prior = prior / (np.linalg.norm(prior) + 1e-8)

        # Soft projection of mixture magnitude onto speaker prior
        # mask_f,t = prior_f^p / (prior_f^p + residual^p)
        proj = mix_mag * prior
        residual = np.maximum(mix_mag - proj, 0.0)
        num = np.power(proj, self.power)
        den = num + np.power(residual, self.power) + 1e-8
        mask = np.clip(num / den, self.floor, 1.0)

        # Mild temporal smoothing
        if mask.shape[1] >= 3:
            mask = signal.medfilt(mask, kernel_size=(1, 3))

        # Optional embedding confidence modulates aggressiveness
        if target_embedding is not None:
            emb = np.asarray(target_embedding, dtype=np.float32).reshape(-1)
            strength = float(np.clip(np.linalg.norm(emb), 0.5, 1.0))
            mask = self.floor + (mask - self.floor) * strength

        enhanced = mix_stft * mask
        out = self._istft(enhanced, length=len(mix))
        return normalize_audio(out)
