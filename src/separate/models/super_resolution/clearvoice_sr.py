from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import SpeechSuperResolution, normalize_audio
from audio_intel.separate.models.super_resolution.resample_fallback import SoxrSuperResolution

log = logging.getLogger(__name__)


class ClearVoiceSuperResolution(SpeechSuperResolution):
    name = "clearvoice_sr"
    output_sample_rate = 48000

    def __init__(
        self,
        model_name: str = "MossFormer2_SR_48K",
        fallback: str = "soxr_hq",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name
        self.fallback_name = fallback
        self._cv = None
        self._fallback = SoxrSuperResolution(output_sample_rate=48000, device=device, dtype=dtype)

    def load(self) -> None:
        self._fallback.load()
        try:
            from clearvoice import ClearVoice

            self._cv = ClearVoice(task="speech_super_resolution", model_names=[self.model_name])
        except Exception as exc:
            self._cv = None
            log.warning(
                "ClearerVoice %s unavailable (%s); using soxr resampling for 16->48 kHz.",
                self.model_name,
                exc,
            )
        self._loaded = True

    def unload(self) -> None:
        self._cv = None
        self._fallback.unload()
        self._loaded = False

    def upscale(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.ensure_loaded()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate == self.output_sample_rate:
            return normalize_audio(audio)

        if self._cv is None:
            return self._fallback.upscale(audio, sample_rate)

        # ClearerVoice SR expects >=16 kHz content
        work_sr = sample_rate
        work = audio
        if sample_rate < 16000:
            work = resample_audio(audio, sample_rate, 16000)
            work_sr = 16000

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            in_path = Path(tmp.name)
        try:
            sf.write(str(in_path), work, work_sr)
            out = self._cv(input_path=str(in_path), online_write=False)
            if isinstance(out, list | tuple):
                out = out[0]
            up = np.asarray(out, dtype=np.float32).reshape(-1)
        except Exception:
            return self._fallback.upscale(audio, sample_rate)
        finally:
            in_path.unlink(missing_ok=True)

        if len(up) == 0:
            return self._fallback.upscale(audio, sample_rate)
        return normalize_audio(up)
