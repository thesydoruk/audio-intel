from __future__ import annotations

import logging

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import SourceSeparator

log = logging.getLogger(__name__)


class DemucsSeparator(SourceSeparator):
    """Speech / background separation with Demucs (``htdemucs_ft`` by default).

    Demucs is a music source separation model, but its ``vocals`` stem is the most
    robust generally available way to pull dialogue out of a mix that contains
    music and effects. Everything that is not ``vocals`` (drums, bass, other) is
    summed into the background stem, which is what a dubbing pipeline needs as
    its music & effects bed.
    """

    name = "demucs"

    def __init__(
        self,
        model_name: str = "htdemucs_ft",
        *,
        shifts: int = 1,
        overlap: float = 0.25,
        segment: float | None = None,
        device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name
        self.shifts = int(shifts)
        self.overlap = float(overlap)
        self.segment = segment
        self._model = None
        self.model_sample_rate = 44100
        self.sources: list[str] = []

    def load(self) -> None:
        import torch
        from demucs.pretrained import get_model

        model = get_model(self.model_name)
        model.eval()
        dev = (
            self.device if (self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        if dev == "cpu" and self.device.startswith("cuda"):
            log.warning("CUDA requested for Demucs but unavailable; running on CPU (slow).")
        self.device = dev
        model.to(dev)
        self._model = model
        self.model_sample_rate = int(model.samplerate)
        self.sources = list(model.sources)
        if "vocals" not in self.sources:
            raise RuntimeError(
                f"Demucs model {self.model_name} has no 'vocals' source: {self.sources}"
            )
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        self.ensure_loaded()
        assert self._model is not None
        import torch
        from demucs.apply import apply_model

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        n_in, n_ch = audio.shape

        work = audio
        if sample_rate != self.model_sample_rate:
            work = resample_audio(work, sample_rate, self.model_sample_rate)
        # Demucs models are stereo; duplicate mono input.
        if work.shape[1] == 1:
            work = np.repeat(work, 2, axis=1)
        elif work.shape[1] > 2:
            work = work[:, :2]

        wav = torch.from_numpy(np.ascontiguousarray(work.T))  # (C, T)
        ref = wav.mean(0)
        mean = float(ref.mean())
        std = float(ref.std()) + 1e-8
        wav = (wav - mean) / std

        with torch.inference_mode():
            out = apply_model(
                self._model,
                wav[None],
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=False,
                segment=self.segment,
            )[0]
        out = out * std + mean  # (S, C, T)
        out_np = out.cpu().numpy().astype(np.float32)

        v_idx = self.sources.index("vocals")
        vocals = out_np[v_idx].T  # (T, C)
        background = (out_np.sum(axis=0) - out_np[v_idx]).T

        vocals = self._to_input_layout(vocals, n_ch, sample_rate, n_in)
        background = self._to_input_layout(background, n_ch, sample_rate, n_in)
        return vocals, background

    def _to_input_layout(
        self, stem: np.ndarray, n_ch: int, sample_rate: int, n_in: int
    ) -> np.ndarray:
        if sample_rate != self.model_sample_rate:
            stem = resample_audio(stem, self.model_sample_rate, sample_rate)
        if n_ch == 1:
            stem = stem.mean(axis=1, keepdims=True)
        elif stem.shape[1] != n_ch:
            stem = stem[:, :n_ch]
        if len(stem) < n_in:
            stem = np.pad(stem, ((0, n_in - len(stem)), (0, 0)))
        return np.ascontiguousarray(stem[:n_in], dtype=np.float32)
