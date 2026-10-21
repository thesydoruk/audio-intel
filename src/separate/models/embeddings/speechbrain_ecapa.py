from __future__ import annotations

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models._speechbrain import speechbrain_fetch_kwargs
from audio_intel.separate.models.base import SpeakerEmbedder


class SpeechBrainECAPA(SpeakerEmbedder):
    name = "speechbrain_ecapa"
    embedding_dim = 192

    def __init__(
        self,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cuda",
        dtype: str = "float16",
        savedir: str = "./cache/embeddings/speechbrain_ecapa",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id
        self.savedir = savedir
        self._classifier = None

    def load(self) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        dev = (
            self.device if (self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        self.device = dev
        self._classifier = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=self.savedir,
            run_opts={"device": dev},
            **speechbrain_fetch_kwargs(),
        )
        self._classifier.eval()
        self._loaded = True

    def unload(self) -> None:
        self._classifier = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.ensure_loaded()
        assert self._classifier is not None
        import torch

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            audio = resample_audio(audio, sample_rate, 16000)
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.inference_mode():
            emb = self._classifier.encode_batch(wav)
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec
