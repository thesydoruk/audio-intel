from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from audio_intel.separate.models.tse.adaptive_wiener import AdaptiveWienerTSE
from audio_intel.separate.pipeline.orchestrator import (
    merge_segments,
    relabel_by_duration,
    split_by_overlap,
)
from audio_intel.separate.stitching.stitcher import TrackStitcher
from audio_intel.separate.types import DiarizationSegment, OverlapRegion, TimeSpan


def test_stitcher_preserves_duration_and_silence(tmp_path: Path):
    st = TrackStitcher(duration=2.0, sample_rate=48000, crossfade_ms=10.0, segment_pad_ms=0.0)
    audio = np.ones(48000, dtype=np.float32) * 0.25
    st.write("speaker_00", TimeSpan(0.5, 1.5), audio)
    paths = st.export(tmp_path, naming="speaker_{id:02d}_48k.wav")
    y, sr = sf.read(paths["speaker_00"])
    assert sr == 48000
    assert len(y) == 96000
    assert float(np.max(np.abs(y[: int(0.4 * sr)]))) < 1e-3
    assert abs(float(y[int(1.0 * sr)]) - 0.25) < 1e-3


def test_adjacent_padded_segments_have_no_dip():
    """Two abutting turns of one speaker must stitch into a flat signal (no gain dip at the seam)."""
    sr = 48000
    st = TrackStitcher(duration=4.0, sample_rate=sr, crossfade_ms=40.0, segment_pad_ms=100.0)
    src = np.ones(4 * sr, dtype=np.float32) * 0.5
    for span in (TimeSpan(0.5, 2.0), TimeSpan(2.0, 3.5)):
        padded = st.padded(span)
        a, b = padded.to_samples(sr)
        st.write("speaker_00", padded, src[a:b])
    track = st.tracks["speaker_00"]
    seam = track[int(1.8 * sr) : int(2.2 * sr)]
    assert np.allclose(seam, 0.5, atol=1e-4)
    # Fades to silence outside the padded region
    assert float(np.abs(track[: int(0.35 * sr)]).max()) < 1e-6


def test_sum_tracks_and_export_indexing(tmp_path: Path):
    st = TrackStitcher(duration=1.0, sample_rate=16000, segment_pad_ms=0.0)
    st.write("speaker_01", TimeSpan(0.0, 1.0), np.full(16000, 0.1, dtype=np.float32))
    st.write("speaker_00", TimeSpan(0.0, 1.0), np.full(16000, 0.2, dtype=np.float32))
    total = st.sum_tracks()
    assert np.allclose(total[8000], 0.3, atol=1e-4)
    paths = st.export(tmp_path)
    assert Path(paths["speaker_00"]).name == "speaker_00_48k.wav"
    assert Path(paths["speaker_01"]).name == "speaker_01_48k.wav"


def test_split_by_overlap_partitions_turn():
    span = TimeSpan(0.0, 10.0)
    overlaps = [OverlapRegion(TimeSpan(2.0, 3.0), 2), OverlapRegion(TimeSpan(9.95, 12.0), 2)]
    parts = split_by_overlap(span, overlaps, min_part=0.2)
    kinds = [(round(p.start, 2), round(p.end, 2), ov) for p, ov in parts]
    # The 50 ms sliver at the end is absorbed into the trailing overlap part
    assert kinds == [(0.0, 2.0, False), (2.0, 3.0, True), (3.0, 9.95, False), (9.95, 10.0, True)]
    covered = sum(p.duration for p, _ in parts)
    assert abs(covered - span.duration) < 1e-6


def test_merge_and_relabel_segments():
    segs = [
        DiarizationSegment("B", TimeSpan(0.0, 1.0)),
        DiarizationSegment("B", TimeSpan(1.1, 2.0)),
        DiarizationSegment("A", TimeSpan(3.0, 3.05)),
        DiarizationSegment("A", TimeSpan(5.0, 5.5)),
    ]
    merged = merge_segments(segs, min_duration=0.1, merge_gap=0.25)
    assert [(s.local_speaker, s.span.start, s.span.end) for s in merged] == [
        ("B", 0.0, 2.0),
        ("A", 5.0, 5.5),
    ]
    relabeled = relabel_by_duration(merged)
    assert {s.local_speaker for s in relabeled} == {"speaker_00", "speaker_01"}
    assert relabeled[0].local_speaker == "speaker_00"  # B talks longest


def test_adaptive_wiener_tse_runs():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    ref = 0.2 * np.sin(2 * np.pi * 120 * t)
    mix = ref + 0.2 * np.sin(2 * np.pi * 200 * t)
    model = AdaptiveWienerTSE({"sample_rate": sr})
    model.load()
    out = model.extract(mix, sr, reference_audio=ref, reference_sample_rate=sr)
    assert out.shape[0] == mix.shape[0]
    assert np.isfinite(out).all()
