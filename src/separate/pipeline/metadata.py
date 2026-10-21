from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio_intel.separate.types import (
    DiarizationSegment,
    ExtractedRegion,
    PipelineResult,
    SpeakerSummary,
)


def build_summaries(regions: list[ExtractedRegion], high_thr: float = 0.6) -> list[SpeakerSummary]:
    by: dict[str, SpeakerSummary] = {}
    for r in regions:
        s = by.setdefault(r.speaker_id, SpeakerSummary(speaker_id=r.speaker_id))
        dur = r.span.duration
        s.total_speech_duration += dur
        s.n_regions += 1
        conf = min(r.diarization_confidence, r.matching_confidence)
        if r.source_type.value == "overlap":
            s.overlap_duration += dur
        if conf >= high_thr:
            s.high_confidence_duration += dur
        else:
            s.low_confidence_duration += dur
    return [by[k] for k in sorted(by)]


def segments_by_speaker(segments: list[DiarizationSegment]) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    for seg in sorted(segments, key=lambda s: (s.span.start, s.span.end)):
        out.setdefault(seg.local_speaker, []).append(
            {"start": round(seg.span.start, 3), "end": round(seg.span.end, 3)}
        )
    return out


def write_metadata(
    result: PipelineResult, path: str | Path, extra: dict[str, Any] | None = None
) -> Path:
    path = Path(path)
    payload: dict[str, Any] = {
        "input": {
            "path": result.input_info.path,
            "sample_rate": result.input_info.sample_rate,
            "channels": result.input_info.channels,
            "duration": result.input_info.duration,
            "format": result.input_info.format,
        },
        "output_dir": result.output_dir,
        "master": result.master_path,
        "speech_stem": result.speech_stem_path,
        "background": result.background_path,
        "speakers": result.global_speakers,
        "speaker_files": result.speaker_files,
        # Per-speaker speech turns on the master timeline (seconds): feed these to ASR
        "segments": segments_by_speaker(result.segments),
        "overlaps": [
            {
                "start": round(o.span.start, 3),
                "end": round(o.span.end, 3),
                "n_speakers": o.n_speakers,
                "speakers": o.speakers,
                "confidence": o.confidence,
            }
            for o in result.overlaps
        ],
        "regions": [r.to_dict() for r in result.regions],
        "summaries": [
            {
                "speaker_id": s.speaker_id,
                "total_speech_duration": s.total_speech_duration,
                "high_confidence_duration": s.high_confidence_duration,
                "overlap_duration": s.overlap_duration,
                "low_confidence_duration": s.low_confidence_duration,
                "n_regions": s.n_regions,
            }
            for s in result.summaries
        ],
    }
    if extra:
        payload["extra"] = extra
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_rttm(segments: list[DiarizationSegment], path: str | Path, uri: str = "audio") -> Path:
    """Write NIST RTTM (one SPEAKER line per turn), the lingua franca of diarization tools."""
    path = Path(path)
    lines = []
    for seg in sorted(segments, key=lambda s: (s.span.start, s.span.end)):
        lines.append(
            f"SPEAKER {uri} 1 {seg.span.start:.3f} {seg.span.duration:.3f} "
            f"<NA> <NA> {seg.local_speaker} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
