from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    SINGLE_SPEAKER = "single_speaker"
    OVERLAP = "overlap"
    UNCERTAIN = "uncertain"
    SILENCE = "silence"


class RegionClass(str, Enum):
    SILENCE = "silence"
    SINGLE = "single_speaker"
    TWO_SPEAKER_OVERLAP = "two_speaker_overlap"
    MULTI_SPEAKER_OVERLAP = "multi_speaker_overlap"


@dataclass(slots=True)
class TimeSpan:
    """Timeline span in seconds on the master timeline."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Invalid span: end {self.end} < start {self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def clamp(self, lo: float, hi: float) -> TimeSpan:
        return TimeSpan(max(lo, self.start), min(hi, self.end))

    def overlap(self, other: TimeSpan) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def to_samples(self, sample_rate: int) -> tuple[int, int]:
        start = int(round(self.start * sample_rate))
        end = int(round(self.end * sample_rate))
        return start, max(start, end)

    def shifted(self, delta: float) -> TimeSpan:
        return TimeSpan(self.start + delta, self.end + delta)


@dataclass(slots=True)
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    duration: float
    num_frames: int
    format: str
    subtype: str

    @property
    def sample_count_48k(self) -> int:
        """Sample count if remapped to 48 kHz with identical duration."""
        return int(round(self.duration * 48000))


@dataclass(slots=True)
class ChunkSpec:
    index: int
    span: TimeSpan
    # Exclusive core region (non-overlap with previous/next contribution)
    core: TimeSpan


@dataclass(slots=True)
class DiarizationSegment:
    local_speaker: str
    span: TimeSpan
    confidence: float = 1.0


@dataclass(slots=True)
class DiarizationResult:
    segments: list[DiarizationSegment] = field(default_factory=list)
    embeddings: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def speakers(self) -> list[str]:
        return sorted({s.local_speaker for s in self.segments})


@dataclass(slots=True)
class OverlapRegion:
    span: TimeSpan
    n_speakers: int
    confidence: float = 1.0
    speakers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegionLabel:
    span: TimeSpan
    kind: RegionClass
    speakers: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(slots=True)
class EmbeddingSample:
    speaker_id: str
    vector: Any  # np.ndarray
    span: TimeSpan
    quality: float
    snr_db: float | None = None
    from_overlap: bool = False


@dataclass(slots=True)
class ExtractedRegion:
    speaker_id: str
    span: TimeSpan
    source_type: SourceType
    diarization_confidence: float = 1.0
    matching_confidence: float = 1.0
    tse_confidence: float | None = None
    enhancement_confidence: float | None = None
    embedding_similarity: float | None = None
    estimated_snr_db: float | None = None
    chunk_index: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d["span"] = {"start": self.span.start, "end": self.span.end}
        return d


@dataclass(slots=True)
class SpeakerSummary:
    speaker_id: str
    total_speech_duration: float = 0.0
    high_confidence_duration: float = 0.0
    overlap_duration: float = 0.0
    low_confidence_duration: float = 0.0
    n_regions: int = 0


@dataclass(slots=True)
class PipelineResult:
    input_info: AudioInfo
    output_dir: str
    speaker_files: dict[str, str] = field(default_factory=dict)
    regions: list[ExtractedRegion] = field(default_factory=list)
    summaries: list[SpeakerSummary] = field(default_factory=list)
    global_speakers: list[str] = field(default_factory=list)
    # Global (already identity-resolved) speech turns, for ASR alignment
    segments: list[DiarizationSegment] = field(default_factory=list)
    overlaps: list[OverlapRegion] = field(default_factory=list)
    background_path: str | None = None
    speech_stem_path: str | None = None
    master_path: str | None = None
    metadata_path: str | None = None
    report_path: str | None = None
    rttm_path: str | None = None
