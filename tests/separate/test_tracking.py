from __future__ import annotations

import numpy as np
from audio_intel.separate.embeddings.profile import SpeakerProfileBank
from audio_intel.separate.tracking.assigner import GlobalSpeakerTracker, LocalSpeakerStats
from audio_intel.separate.types import EmbeddingSample, TimeSpan


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=32).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def test_permutation_resolved_across_chunks():
    bank = SpeakerProfileBank({"update_similarity_threshold": 0.1, "update_min_duration": 0.1})
    tracker = GlobalSpeakerTracker(
        bank,
        {
            "new_speaker_threshold": 0.35,
            "min_embedding_similarity": 0.1,
            "weight_embedding": 0.8,
            "weight_temporal_continuity": 0.1,
            "weight_overlap_consistency": 0.05,
            "weight_profile_confidence": 0.025,
            "weight_segment_quality": 0.025,
        },
    )

    a = _vec(1)
    b = _vec(2)

    # Chunk 1: local_0 -> A, local_1 -> B
    asg1 = tracker.assign_chunk(
        [
            LocalSpeakerStats("local_0", a, 0.9, 3.0, TimeSpan(0, 3)),
            LocalSpeakerStats("local_1", b, 0.9, 3.0, TimeSpan(1, 4)),
        ],
        chunk_span=TimeSpan(0, 10),
        overlap_with_prev=None,
    )
    map1 = {x.local_id: x.global_id for x in asg1}
    assert map1["local_0"] != map1["local_1"]

    # Chunk 2: permuted local IDs
    asg2 = tracker.assign_chunk(
        [
            LocalSpeakerStats("local_0", b, 0.9, 3.0, TimeSpan(8, 12)),
            LocalSpeakerStats("local_1", a, 0.9, 3.0, TimeSpan(9, 13)),
        ],
        chunk_span=TimeSpan(5, 15),
        overlap_with_prev=TimeSpan(5, 10),
    )
    map2 = {x.local_id: x.global_id for x in asg2}
    assert map2["local_0"] == map1["local_1"]
    assert map2["local_1"] == map1["local_0"]


def test_speaker_return_after_long_gap_uses_embeddings():
    bank = SpeakerProfileBank({"update_similarity_threshold": 0.1, "update_min_duration": 0.1})
    tracker = GlobalSpeakerTracker(
        bank, {"new_speaker_threshold": 0.4, "min_embedding_similarity": 0.2}
    )
    a = _vec(7)
    asg1 = tracker.assign_chunk(
        [LocalSpeakerStats("local_0", a, 0.9, 2.0, TimeSpan(0, 2))],
        chunk_span=TimeSpan(0, 5),
        overlap_with_prev=None,
    )
    gid = asg1[0].global_id
    # Add more samples to strengthen profile
    bank.maybe_update(
        gid,
        EmbeddingSample(speaker_id=gid, vector=a, span=TimeSpan(0, 2), quality=0.95),
    )
    asg2 = tracker.assign_chunk(
        [LocalSpeakerStats("local_3", a, 0.9, 2.0, TimeSpan(600, 602))],
        chunk_span=TimeSpan(595, 610),
        overlap_with_prev=None,
    )
    assert asg2[0].global_id == gid
    assert asg2[0].is_new is False
