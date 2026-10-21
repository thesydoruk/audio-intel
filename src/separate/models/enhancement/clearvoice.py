from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import SpeechEnhancer, normalize_audio

log = logging.getLogger(__name__)


class ClearVoiceEnhancer(SpeechEnhancer):
    name = "clearvoice_enhancer"

    def __init__(
        self,
        model_name: str = "MossFormer2_SE_48K",
        sample_rate: int = 48000,
        mix: float = 0.85,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mix = float(mix)
        self._cv = None
        self.load_error: str | None = None

    def load(self) -> None:
        try:
            from clearvoice import ClearVoice

            self._cv = ClearVoice(task="speech_enhancement", model_names=[self.model_name])
        except Exception as exc:  # pragma: no cover - optional dependency
            self._cv = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "ClearerVoice %s unavailable (%s); enhancement is a pass-through.",
                self.model_name,
                self.load_error,
            )
        self._loaded = True

    def unload(self) -> None:
        self._cv = None
        self._loaded = False

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.ensure_loaded()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate:
            audio = resample_audio(audio, sample_rate, self.sample_rate)

        if self._cv is None or self.mix <= 0:
            return normalize_audio(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            in_path = Path(tmp.name)
        try:
            sf.write(str(in_path), audio, self.sample_rate)
            out = self._cv(input_path=str(in_path), online_write=False)
            if isinstance(out, list | tuple):
                out = out[0]
            enhanced = np.asarray(out, dtype=np.float32).reshape(-1)
        except Exception as exc:
            log.warning(
                "ClearerVoice %s failed on a segment (%s); keeping input.", self.model_name, exc
            )
            enhanced = audio
        finally:
            in_path.unlink(missing_ok=True)

        n = min(len(audio), len(enhanced))
        mixed = (1.0 - self.mix) * audio[:n] + self.mix * enhanced[:n]
        if len(mixed) < len(audio):
            mixed = np.pad(mixed, (0, len(audio) - len(mixed)))
        return normalize_audio(mixed[: len(audio)])
