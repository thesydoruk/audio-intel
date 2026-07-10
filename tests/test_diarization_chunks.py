"""Unit tests for chunked diarization planning and speaker linking."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.diarization.chunks import (  # noqa: E402
    DiarizationChunk,
    cosine_similarity,
    link_local_speakers,
    merge_chunk_intervals,
    offset_intervals,
    owned_span,
    plan_diarization_chunks,
    remap_speaker_ids,
)


class PlanDiarizationChunksTest(unittest.TestCase):
    def test_short_media_returns_single_chunk(self) -> None:
        chunks = plan_diarization_chunks(120.0, chunk_s=600.0, overlap_s=30.0)
        self.assertEqual(chunks, [DiarizationChunk(index=0, start_s=0.0, end_s=120.0)])

    def test_disabled_chunking_returns_single_chunk(self) -> None:
        chunks = plan_diarization_chunks(3600.0, chunk_s=0.0, overlap_s=30.0)
        self.assertEqual(chunks, [DiarizationChunk(index=0, start_s=0.0, end_s=3600.0)])

    def test_long_media_splits_with_overlap(self) -> None:
        chunks = plan_diarization_chunks(1300.0, chunk_s=600.0, overlap_s=30.0)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].start_s, 0.0)
        self.assertEqual(chunks[0].end_s, 600.0)
        self.assertEqual(chunks[1].start_s, 570.0)
        self.assertEqual(chunks[2].end_s, 1300.0)


class ChunkMergeTest(unittest.TestCase):
    def test_offset_intervals_shifts_times(self) -> None:
        local = [{"start": 1.0, "end": 2.0, "speaker_id": "spk_0"}]
        shifted = offset_intervals(local, 600.0)
        self.assertEqual(shifted[0]["start"], 601.0)
        self.assertEqual(shifted[0]["end"], 602.0)

    def test_owned_span_excludes_overlap_halves(self) -> None:
        chunk = DiarizationChunk(index=1, start_s=570.0, end_s=1170.0)
        owned_start, owned_end = owned_span(chunk, total_chunks=3, overlap_s=30.0)
        self.assertEqual(owned_start, 585.0)
        self.assertEqual(owned_end, 1155.0)

    def test_merge_keeps_only_owned_regions(self) -> None:
        chunks = [
            DiarizationChunk(index=0, start_s=0.0, end_s=600.0),
            DiarizationChunk(index=1, start_s=570.0, end_s=1200.0),
        ]
        chunk_intervals = [
            [
                {"start": 10.0, "end": 20.0, "speaker_id": "spk_0"},
                {"start": 590.0, "end": 595.0, "speaker_id": "spk_1"},
            ],
            [
                {"start": 580.0, "end": 590.0, "speaker_id": "spk_0"},
                {"start": 900.0, "end": 910.0, "speaker_id": "spk_1"},
            ],
        ]
        merged = merge_chunk_intervals(chunk_intervals, chunks, overlap_s=30.0)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["start"], 10.0)
        self.assertEqual(merged[1]["start"], 580.0)
        self.assertEqual(merged[2]["start"], 900.0)


class SpeakerLinkingTest(unittest.TestCase):
    def test_cosine_similarity_identical_vectors(self) -> None:
        vec = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0)

    def test_first_chunk_maps_local_ids_to_global(self) -> None:
        mapping, next_idx = link_local_speakers(
            local_intervals=[{"start": 0.0, "end": 1.0, "speaker_id": "spk_0"}],
            previous_global_intervals=[],
            chunk_start_s=0.0,
            overlap_s=30.0,
            local_embeddings={},
            global_embeddings={},
            next_global_index=0,
            threshold=0.75,
        )
        self.assertEqual(mapping, {"spk_0": "spk_0"})
        self.assertEqual(next_idx, 1)

    def test_links_matching_embeddings_in_overlap(self) -> None:
        previous = [
            {"start": 560.0, "end": 580.0, "speaker_id": "spk_0"},
        ]
        local = [
            {"start": 570.0, "end": 590.0, "speaker_id": "spk_0"},
            {"start": 900.0, "end": 910.0, "speaker_id": "spk_1"},
        ]
        same = [1.0, 0.0, 0.0]
        different = [0.0, 1.0, 0.0]
        mapping, next_idx = link_local_speakers(
            local_intervals=local,
            previous_global_intervals=previous,
            chunk_start_s=570.0,
            overlap_s=30.0,
            local_embeddings={"spk_0": same, "spk_1": different},
            global_embeddings={"spk_0": same},
            next_global_index=1,
            threshold=0.75,
        )
        self.assertEqual(mapping["spk_0"], "spk_0")
        self.assertEqual(mapping["spk_1"], "spk_1")
        self.assertEqual(next_idx, 2)

    def test_remap_speaker_ids_applies_mapping(self) -> None:
        remapped = remap_speaker_ids(
            [{"start": 1.0, "end": 2.0, "speaker_id": "spk_0"}],
            {"spk_0": "spk_7"},
        )
        self.assertEqual(remapped[0]["speaker_id"], "spk_7")


if __name__ == "__main__":
    unittest.main()
