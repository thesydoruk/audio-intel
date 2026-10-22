from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from audio_intel.separate.types import DiarizationResult, OverlapRegion


class ModelHandle(ABC):
    name: str

    def __init__(self, device: str = "cuda", dtype: str = "float16") -> None:
        self.device = device
        self.dtype = dtype
        self._loaded = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    def loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


class Diarizer(ModelHandle):
    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        offset: float = 0.0,
    ) -> DiarizationResult: ...


class SpeakerEmbedder(ModelHandle):
    embedding_dim: int

    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return L2-normalized embedding vector."""


class OverlapDetector(ModelHandle):
    @abstractmethod
    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        offset: float = 0.0,
        diarization: DiarizationResult | None = None,
    ) -> list[OverlapRegion]: ...


class TargetSpeakerExtractor(ModelHandle):
    sample_rate: int

    @abstractmethod
    def extract(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        *,
        target_embedding: np.ndarray | None = None,
        reference_audio: np.ndarray | None = None,
        reference_sample_rate: int | None = None,
    ) -> np.ndarray:
        """Return target speaker waveform at self.sample_rate (typically 16 kHz)."""


class SpeechEnhancer(ModelHandle):
    sample_rate: int

    @abstractmethod
    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


class SpeechSuperResolution(ModelHandle):
    output_sample_rate: int

    @abstractmethod
    def upscale(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Upscale toward output_sample_rate (typically 48 kHz)."""


class SourceSeparator(ModelHandle):
    """Split a full mix into a speech (vocals) stem and a background (everything else) stem.

    Both stems are returned at the input sample rate with the input channel layout
    ``(samples, channels)`` and identical length to the input.
    """

    @abstractmethod
    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(speech, background)`` arrays shaped ``(samples, channels)``."""


def resolve_device(requested: str) -> str:
    """``auto`` -> the GPU when there is one; a bare ``cuda`` -> ``cuda:0``.

    SpeechBrain splits the device string on ``:`` to pick the card and, given
    plain ``cuda``, logs a warning and falls back to device 0 by itself, which
    is wrong on a multi-GPU host and noisy on every other one.
    """
    if requested == "auto":
        try:
            import torch

            requested = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return "cuda:0" if requested == "cuda" else requested


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[name.lower()]


def estimate_snr_db(audio: np.ndarray, frame: int = 400) -> float:
    """Crude segmental SNR proxy from active vs quiet frames."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < frame * 2:
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12) + 60.0
    n = len(audio) // frame
    frames = audio[: n * frame].reshape(n, frame)
    energies = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    noise = np.percentile(energies, 20)
    speech = np.percentile(energies, 80)
    return float(20.0 * np.log10((speech + 1e-12) / (noise + 1e-12)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def normalize_audio(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    m = float(np.max(np.abs(audio)) + 1e-12)
    if m > peak:
        audio = audio * (peak / m)
    return audio
