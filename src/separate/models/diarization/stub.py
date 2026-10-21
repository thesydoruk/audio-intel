from __future__ import annotations

import numpy as np
from audio_intel.separate.models.base import Diarizer, estimate_snr_db
from audio_intel.separate.types import DiarizationResult, DiarizationSegment, TimeSpan


class StubDiarizer(Diarizer):
    """Energy-based single/dual speaker stub for offline tests without HF models."""

    name = "stub"

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        offset: float = 0.0,
    ) -> DiarizationResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        frame = max(1, int(0.02 * sample_rate))
        n = len(audio) // frame
        if n == 0:
            return DiarizationResult(segments=[])
        energies = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
        thr = max(0.01, float(np.percentile(energies, 40)))
        active = energies > thr
        segments: list[DiarizationSegment] = []
        i = 0
        spk = 0
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j < n and active[j]:
                j += 1
            start = offset + i * frame / sample_rate
            end = offset + j * frame / sample_rate
            if end - start >= 0.2:
                # Alternate local labels to exercise tracking permutation handling
                label = f"local_{spk % 2}"
                segments.append(
                    DiarizationSegment(
                        local_speaker=label, span=TimeSpan(start, end), confidence=0.5
                    )
                )
                spk += 1
            i = j
        if min_speakers and not segments:
            duration = len(audio) / sample_rate
            segments.append(
                DiarizationSegment(
                    local_speaker="local_0",
                    span=TimeSpan(offset, offset + duration),
                    confidence=0.2,
                )
            )
        _ = max_speakers
        _ = estimate_snr_db(audio)
        return DiarizationResult(segments=segments)
