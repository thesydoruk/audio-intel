"""Speaker embeddings from the pyannote community-1 WeSpeaker model.

The transcription service already loads this model for its speaker roster
(``pyannote/speaker-diarization-community-1/embedding``, 256-d). Using the same
model here puts separation profiles and the service's stored speaker vectors
in one space, so a track's speaker can be matched against known voices with
the service's own thresholds. ECAPA (``speechbrain_ecapa``) remains available;
the thresholds in ``default.yaml`` were tuned for it, see MODEL_SELECTION.md.

``inference_provider`` lets the service hand in its already loaded pyannote
``Inference`` instead of this backend loading a second copy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models.base import SpeakerEmbedder

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "pyannote/speaker-diarization-community-1/embedding"


def _split_model_id(model_id: str) -> tuple[str, str | None]:
    parts = [part for part in model_id.strip().strip("/").split("/") if part]
    if len(parts) <= 2:
        return "/".join(parts), None
    return "/".join(parts[:2]), "/".join(parts[2:])


class WeSpeakerEmbedder(SpeakerEmbedder):
    """pyannote ``Inference(window="whole")`` over the community-1 embedding model."""

    name = "wespeaker"
    embedding_dim = 256

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cuda",
        dtype: str = "float32",
        inference_provider: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id
        self._provider = inference_provider
        self._inference: Any = None

    def load(self) -> None:
        import torch

        if self._provider is not None:
            self._inference = self._provider()
            self._loaded = True
            return
        import os

        from pyannote.audio import Inference, Model

        token = os.environ.get("HF_TOKEN") or None
        checkpoint, subfolder = _split_model_id(self.model_id)
        kwargs: dict[str, Any] = {"token": token} if token else {}
        if subfolder:
            kwargs["subfolder"] = subfolder
        model = Model.from_pretrained(checkpoint, **kwargs)
        if model is None:
            raise RuntimeError(
                f"Could not load {self.model_id}: accept its terms on Hugging Face "
                "for the account that owns HF_TOKEN"
            )
        dimension = getattr(model, "dimension", None)
        if isinstance(dimension, int) and dimension > 0:
            self.embedding_dim = dimension
        inference = Inference(model, window="whole")
        dev = (
            self.device if (self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        if dev == "cpu" and self.device.startswith("cuda"):
            log.warning("CUDA requested for WeSpeaker but unavailable; running on CPU.")
        self.device = dev
        inference.to(torch.device(dev))
        self._inference = inference
        self._loaded = True

    def unload(self) -> None:
        self._inference = None
        self._loaded = False
        if self._provider is not None:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.ensure_loaded()
        assert self._inference is not None
        import torch

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            audio = resample_audio(audio, sample_rate, 16000)
            sample_rate = 16000
        waveform = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sample_rate}
        with torch.inference_mode():
            vector = self._inference(waveform)
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        return vec / (np.linalg.norm(vec) + 1e-12)
