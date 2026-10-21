from __future__ import annotations

from pathlib import Path

import numpy as np
from audio_intel.separate.audio.io import write_wav
from audio_intel.separate.audio.timeline import MasterTimeline
from audio_intel.separate.stitching.crossfade import blend_into_track
from audio_intel.separate.types import TimeSpan


class TrackStitcher:
    """Assembles one full-length, sample-aligned mono track per speaker.

    Every track has exactly ``timeline.num_samples`` samples; silence where the
    speaker is inactive. Segments are written with edge crossfades, and callers
    pad consecutive segments (see :meth:`padded`) so boundaries stay continuous.
    """

    def __init__(
        self,
        duration: float,
        sample_rate: int = 48000,
        crossfade_ms: float = 40.0,
        segment_pad_ms: float = 100.0,
    ) -> None:
        self.timeline = MasterTimeline(duration, sample_rate)
        self.crossfade_ms = float(crossfade_ms)
        # Padding must cover the crossfade so adjacent writes overlap, never dip.
        self.segment_pad_s = max(float(segment_pad_ms), self.crossfade_ms) * 0.001
        self.tracks: dict[str, np.ndarray] = {}

    @property
    def sample_rate(self) -> int:
        return self.timeline.sample_rate

    def padded(self, span: TimeSpan) -> TimeSpan:
        """Span extended by the segment padding and clamped to the timeline."""
        return TimeSpan(span.start - self.segment_pad_s, span.end + self.segment_pad_s).clamp(
            0.0, self.timeline.duration
        )

    def ensure_speaker(self, speaker_id: str) -> np.ndarray:
        if speaker_id not in self.tracks:
            self.tracks[speaker_id] = np.zeros(self.timeline.num_samples, dtype=np.float32)
        return self.tracks[speaker_id]

    def write(self, speaker_id: str, span: TimeSpan, audio: np.ndarray) -> None:
        """Write ``audio`` (covering exactly ``span``) into the speaker's track."""
        track = self.ensure_speaker(speaker_id)
        blend_into_track(track, self.timeline, span, audio, crossfade_ms=self.crossfade_ms)

    def speakers(self) -> list[str]:
        return sorted(self.tracks)

    def sum_tracks(self) -> np.ndarray:
        """Sum of all speaker tracks (used for the residual background fallback)."""
        total = np.zeros(self.timeline.num_samples, dtype=np.float32)
        for audio in self.tracks.values():
            total += audio
        return total

    def export(
        self,
        output_dir: str | Path,
        *,
        naming: str = "speaker_{id:02d}_48k.wav",
        subtype: str = "PCM_24",
    ) -> dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for idx, sid in enumerate(self.speakers()):
            audio = self.tracks[sid]
            filename = naming.format(id=idx, speaker=sid)
            path = output_dir / filename
            peak = float(np.max(np.abs(audio)) + 1e-12)
            if peak > 1.0:
                audio = audio / peak
            write_wav(path, audio, self.timeline.sample_rate, subtype=subtype)
            paths[sid] = str(path)
        return paths
