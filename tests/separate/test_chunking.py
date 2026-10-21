from __future__ import annotations

from audio_intel.separate.audio.chunking import build_chunks


def test_chunk_cores_cover_timeline_without_gaps_or_overlap():
    chunks = build_chunks(duration=100.0, chunk_duration=45.0, overlap=10.0)
    assert chunks[0].core.start == 0.0
    assert abs(chunks[-1].core.end - 100.0) < 1e-6
    for i in range(len(chunks) - 1):
        assert abs(chunks[i].core.end - chunks[i + 1].core.start) < 1e-6
        # Spans themselves overlap
        assert chunks[i].span.end > chunks[i + 1].span.start


def test_chunk_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        build_chunks(duration=10.0, chunk_duration=5.0, overlap=5.0)
