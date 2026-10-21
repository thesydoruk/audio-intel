from __future__ import annotations

import logging
import os

import numpy as np
from audio_intel.separate.models.base import OverlapDetector
from audio_intel.separate.types import DiarizationResult, OverlapRegion, TimeSpan

log = logging.getLogger(__name__)


def _merge_regions(regions: list[OverlapRegion], merge_gap: float) -> list[OverlapRegion]:
    if not regions:
        return []
    regions = sorted(regions, key=lambda r: r.span.start)
    merged: list[OverlapRegion] = [regions[0]]
    for reg in regions[1:]:
        prev = merged[-1]
        if reg.span.start <= prev.span.end + merge_gap:
            speakers = sorted(set(prev.speakers) | set(reg.speakers))
            merged[-1] = OverlapRegion(
                span=TimeSpan(prev.span.start, max(prev.span.end, reg.span.end)),
                n_speakers=max(prev.n_speakers, reg.n_speakers, len(speakers)),
                confidence=max(prev.confidence, reg.confidence),
                speakers=speakers,
            )
        else:
            merged.append(reg)
    return merged


def coactivity_overlaps(
    diarization: DiarizationResult,
    *,
    min_overlap_duration: float,
    offset_already_applied: bool = True,
) -> list[OverlapRegion]:
    """Infer overlap from simultaneous diarization labels."""
    _ = offset_already_applied
    events: list[tuple[float, int, str]] = []
    for seg in diarization.segments:
        events.append((seg.span.start, 1, seg.local_speaker))
        events.append((seg.span.end, -1, seg.local_speaker))
    events.sort(key=lambda x: (x[0], -x[1]))

    active: dict[str, int] = {}
    regions: list[OverlapRegion] = []
    prev_t: float | None = None
    for t, delta, spk in events:
        if prev_t is not None and t > prev_t:
            speakers = [s for s, c in active.items() if c > 0]
            if len(speakers) >= 2 and (t - prev_t) >= min_overlap_duration:
                regions.append(
                    OverlapRegion(
                        span=TimeSpan(prev_t, t),
                        n_speakers=len(speakers),
                        confidence=0.8,
                        speakers=sorted(speakers),
                    )
                )
        active[spk] = active.get(spk, 0) + delta
        if active[spk] <= 0:
            active.pop(spk, None)
        prev_t = t
    return regions


class HybridOverlapDetector(OverlapDetector):
    name = "hybrid_overlap"

    def __init__(
        self,
        mode: str = "hybrid",
        model_id: str = "pyannote/overlapped-speech-detection",
        min_overlap_duration: float = 0.15,
        merge_gap: float = 0.08,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.mode = mode
        self.model_id = model_id
        self.min_overlap_duration = min_overlap_duration
        self.merge_gap = merge_gap
        self._pipeline = None

    def load(self) -> None:
        if self.mode in {"hybrid", "pyannote_osd"}:
            try:
                import torch
                from pyannote.audio import Pipeline

                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                try:
                    self._pipeline = Pipeline.from_pretrained(self.model_id, token=token)
                except TypeError:
                    self._pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=token)
                if self._pipeline is None:
                    raise RuntimeError("pipeline is None (accept the model terms on Hugging Face)")
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    self._pipeline.to(torch.device(self.device))
            except Exception as exc:
                self._pipeline = None
                log.warning(
                    "Overlapped-speech model %s unavailable (%s); "
                    "using diarization co-activity only.",
                    self.model_id,
                    exc,
                )
        self._loaded = True

    def unload(self) -> None:
        self._pipeline = None
        self._loaded = False

    def _detect_osd(
        self, audio: np.ndarray, sample_rate: int, offset: float
    ) -> list[OverlapRegion]:
        if self._pipeline is None:
            return []
        import torch

        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
        output = self._pipeline({"waveform": wav, "sample_rate": int(sample_rate)})

        regions: list[OverlapRegion] = []
        timeline = output.get_timeline().support() if hasattr(output, "get_timeline") else output
        for speech in timeline:
            start = float(speech.start) + offset
            end = float(speech.end) + offset
            if end - start >= self.min_overlap_duration:
                regions.append(
                    OverlapRegion(
                        span=TimeSpan(start, end),
                        n_speakers=2,
                        confidence=0.9,
                        speakers=[],
                    )
                )
        return regions

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        offset: float = 0.0,
        diarization: DiarizationResult | None = None,
    ) -> list[OverlapRegion]:
        self.ensure_loaded()
        regions: list[OverlapRegion] = []
        if self.mode in {"hybrid", "pyannote_osd"}:
            regions.extend(self._detect_osd(audio, sample_rate, offset))
        if self.mode in {"hybrid", "coactivity"} and diarization is not None:
            regions.extend(
                coactivity_overlaps(diarization, min_overlap_duration=self.min_overlap_duration)
            )
        return _merge_regions(regions, self.merge_gap)
