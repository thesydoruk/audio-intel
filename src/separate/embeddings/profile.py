from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from audio_intel.separate.models.base import cosine_similarity
from audio_intel.separate.types import EmbeddingSample, TimeSpan


@dataclass
class SpeakerProfile:
    speaker_id: str
    samples: list[EmbeddingSample] = field(default_factory=list)
    enrollment_audio: np.ndarray | None = None
    enrollment_sr: int = 16000
    enrollment_spans: list[TimeSpan] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray | None:
        if not self.samples:
            return None
        mats = np.stack([np.asarray(s.vector, dtype=np.float32) for s in self.samples], axis=0)
        return mats.mean(axis=0)

    @property
    def confidence(self) -> float:
        if not self.samples:
            return 0.0
        total_dur = sum(s.span.duration for s in self.samples)
        avg_q = float(np.mean([s.quality for s in self.samples]))
        return float(np.clip(0.5 * avg_q + 0.5 * np.tanh(total_dur / 5.0), 0.0, 1.0))

    def similarity(self, vector: np.ndarray) -> float:
        c = self.robust_centroid()
        if c is None:
            return -1.0
        return cosine_similarity(c, vector)

    def robust_centroid(self, method: str = "trimmed_mean") -> np.ndarray | None:
        if not self.samples:
            return None
        mats = np.stack([np.asarray(s.vector, dtype=np.float32) for s in self.samples], axis=0)
        if method == "median":
            c = np.median(mats, axis=0)
        elif method == "trimmed_mean" and len(mats) >= 4:
            # Drop farthest 20% from provisional mean
            mean = mats.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-12)
            sims = mats @ mean
            keep = max(2, int(len(mats) * 0.8))
            idx = np.argsort(sims)[-keep:]
            c = mats[idx].mean(axis=0)
        else:
            c = mats.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        return c.astype(np.float32)


class SpeakerProfileBank:
    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.update_similarity_threshold = float(cfg.get("update_similarity_threshold", 0.55))
        self.update_min_duration = float(cfg.get("update_min_duration", 1.0))
        self.max_samples = int(cfg.get("max_samples_per_speaker", 64))
        self.centroid_method = str(cfg.get("centroid_method", "trimmed_mean"))
        self.profiles: dict[str, SpeakerProfile] = {}
        self._next_id = 0

    def speakers(self) -> list[str]:
        return sorted(self.profiles.keys())

    def create(
        self,
        sample: EmbeddingSample,
        enrollment: np.ndarray | None = None,
        sr: int = 16000,
        *,
        speaker_id: str | None = None,
    ) -> str:
        if speaker_id is None:
            sid = f"speaker_{self._next_id:02d}"
            self._next_id += 1
        else:
            sid = speaker_id
            self._next_id = max(self._next_id, len(self.profiles) + 1)
        sample.speaker_id = sid
        profile = SpeakerProfile(speaker_id=sid, samples=[sample])
        if enrollment is not None:
            profile.enrollment_audio = np.asarray(enrollment, dtype=np.float32)
            profile.enrollment_sr = sr
            profile.enrollment_spans = [sample.span]
        self.profiles[sid] = profile
        return sid

    def maybe_update(
        self,
        speaker_id: str,
        sample: EmbeddingSample,
        enrollment: np.ndarray | None = None,
        sr: int = 16000,
    ) -> bool:
        profile = self.profiles[speaker_id]
        sim = profile.similarity(sample.vector)
        if sample.span.duration < self.update_min_duration:
            return False
        if sim < self.update_similarity_threshold and profile.samples:
            return False
        sample.speaker_id = speaker_id
        profile.samples.append(sample)
        if len(profile.samples) > self.max_samples:
            # Keep highest-quality samples
            profile.samples.sort(key=lambda s: s.quality * s.span.duration, reverse=True)
            profile.samples = profile.samples[: self.max_samples]
        if enrollment is not None:
            if profile.enrollment_audio is None:
                profile.enrollment_audio = np.asarray(enrollment, dtype=np.float32)
                profile.enrollment_sr = sr
            elif sample.quality >= 0.7 and sample.span.duration >= 2.0:
                # Refresh enrollment with a strong clean clip
                profile.enrollment_audio = np.asarray(enrollment, dtype=np.float32)
                profile.enrollment_sr = sr
            profile.enrollment_spans.append(sample.span)
        return True

    def centroid(self, speaker_id: str) -> np.ndarray | None:
        return self.profiles[speaker_id].robust_centroid(self.centroid_method)

    def enrollment(self, speaker_id: str) -> tuple[np.ndarray | None, int]:
        p = self.profiles[speaker_id]
        return p.enrollment_audio, p.enrollment_sr
