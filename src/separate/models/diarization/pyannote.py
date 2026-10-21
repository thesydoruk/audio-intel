from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from audio_intel.separate.models.base import Diarizer
from audio_intel.separate.types import DiarizationResult, DiarizationSegment, TimeSpan

log = logging.getLogger(__name__)


def _hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    return None


class PyannoteDiarizer(Diarizer):
    """pyannote.audio diarization (community-1 by default; 3.1 also works).

    Runs on the full recording by default: pyannote performs its own sliding
    window inference plus *global* clustering, which keeps speaker labels
    consistent across the whole file far better than re-linking short chunks.
    """

    name = "pyannote"

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        *,
        use_exclusive: bool = False,
        device: str = "cuda",
        dtype: str = "float32",
        pipeline_provider: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id
        self.use_exclusive = use_exclusive
        self._pipeline = None
        # The transcription service hands in its own loaded pipeline this way,
        # so one copy serves both diarization and separation.
        self._provider = pipeline_provider

    def load(self) -> None:
        if self._provider is not None:
            self._pipeline = self._provider()
            self._loaded = True
            return
        import torch
        from pyannote.audio import Pipeline

        token = _hf_token()
        pipeline = None
        last_exc: Exception | None = None
        for kwargs in ({"token": token}, {"use_auth_token": token}, {}):
            try:
                pipeline = Pipeline.from_pretrained(self.model_id, **kwargs)
                break
            except TypeError as exc:  # older/newer signature
                last_exc = exc
                continue
        if pipeline is None:
            raise RuntimeError(
                f"Failed to load {self.model_id}. Accept the model terms on Hugging Face and set "
                f"HF_TOKEN (see .env.example). Last error: {last_exc}"
            )
        if self.device.startswith("cuda") and torch.cuda.is_available():
            pipeline.to(torch.device(self.device))
        elif self.device.startswith("cuda"):
            log.warning("CUDA requested for pyannote but unavailable; running on CPU (slow).")
        self._pipeline = pipeline
        self._loaded = True

    def unload(self) -> None:
        self._pipeline = None
        self._loaded = False
        if self._provider is not None:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        offset: float = 0.0,
    ) -> DiarizationResult:
        self.ensure_loaded()
        assert self._pipeline is not None
        import torch

        kwargs: dict[str, int] = {}
        if min_speakers is not None and max_speakers is not None and min_speakers == max_speakers:
            kwargs["num_speakers"] = int(min_speakers)
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers is not None:
                kwargs["max_speakers"] = int(max_speakers)

        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
        output = self._pipeline({"waveform": wav, "sample_rate": int(sample_rate)}, **kwargs)

        # pyannote 3.x returns an Annotation; 4.x (community-1) returns DiarizeOutput.
        annotation = output
        exclusive = None
        if hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization
            exclusive = getattr(output, "exclusive_speaker_diarization", None)
        elif isinstance(output, dict) and "speaker_diarization" in output:
            annotation = output["speaker_diarization"]
            exclusive = output.get("exclusive_speaker_diarization")
        if self.use_exclusive and exclusive is not None:
            annotation = exclusive

        segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = float(turn.start) + offset
            end = float(turn.end) + offset
            if end <= start:
                continue
            segments.append(
                DiarizationSegment(
                    local_speaker=str(speaker), span=TimeSpan(start, end), confidence=1.0
                )
            )
        segments.sort(key=lambda s: (s.span.start, s.span.end))
        return DiarizationResult(segments=segments, raw={"model": self.model_id})
