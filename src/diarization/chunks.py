"""Chunk planning and cross-chunk speaker linking for long-form diarization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiarizationChunk:
    """One time window passed to pyannote as a separate decode."""

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def plan_diarization_chunks(
    duration_s: float,
    *,
    chunk_s: float,
    overlap_s: float,
) -> list[DiarizationChunk]:
    """Split a timeline into overlapping windows for chunked diarization.

    Returns a single chunk covering ``[0, duration_s]`` when chunking is disabled
    (``chunk_s <= 0``) or the media is shorter than one chunk.
    """
    if duration_s <= 0:
        return []

    if chunk_s <= 0 or duration_s <= chunk_s:
        return [DiarizationChunk(index=0, start_s=0.0, end_s=round(duration_s, 3))]

    overlap_s = max(0.0, min(overlap_s, chunk_s / 2))
    step_s = max(chunk_s - overlap_s, chunk_s / 2)
    chunks: list[DiarizationChunk] = []
    start = 0.0
    index = 0

    while start < duration_s - 1e-6:
        end = min(duration_s, start + chunk_s)
        chunks.append(DiarizationChunk(index=index, start_s=round(start, 3), end_s=round(end, 3)))
        if end >= duration_s - 1e-6:
            break
        start += step_s
        index += 1

    return chunks


def offset_intervals(intervals: list[dict], offset_s: float) -> list[dict]:
    """Shift chunk-local diarization intervals onto the global timeline."""
    if offset_s == 0:
        return [dict(item) for item in intervals]
    return [
        {
            "start": round(float(item["start"]) + offset_s, 3),
            "end": round(float(item["end"]) + offset_s, 3),
            "speaker_id": str(item["speaker_id"]),
        }
        for item in intervals
    ]


def intervals_in_span(
    intervals: list[dict],
    span_start: float,
    span_end: float,
    *,
    local_time: bool = False,
) -> list[dict]:
    """Return intervals whose midpoint lies in ``[span_start, span_end)``."""
    selected: list[dict] = []
    for interval in intervals:
        start = float(interval["start"])
        end = float(interval["end"])
        midpoint = (start + end) / 2.0
        if span_start <= midpoint < span_end:
            selected.append(interval)
    return selected


def owned_span(
    chunk: DiarizationChunk,
    *,
    total_chunks: int,
    overlap_s: float,
) -> tuple[float, float]:
    """Non-overlapping region whose intervals are kept from this chunk."""
    owned_start = chunk.start_s
    owned_end = chunk.end_s
    if chunk.index > 0 and overlap_s > 0:
        owned_start = chunk.start_s + overlap_s / 2.0
    if chunk.index < total_chunks - 1 and overlap_s > 0:
        owned_end = chunk.end_s - overlap_s / 2.0
    return owned_start, owned_end


def select_owned_intervals(
    intervals: list[dict],
    chunk: DiarizationChunk,
    *,
    total_chunks: int,
    overlap_s: float,
) -> list[dict]:
    """Keep intervals whose midpoint falls in this chunk's owned span."""
    owned_start, owned_end = owned_span(
        chunk,
        total_chunks=total_chunks,
        overlap_s=overlap_s,
    )
    return intervals_in_span(intervals, owned_start, owned_end)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for L2-normalized or arbitrary embedding vectors."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _speaker_spans_in_window(
    intervals: list[dict],
    window_start: float,
    window_end: float,
) -> dict[str, list[tuple[float, float]]]:
    spans: dict[str, list[tuple[float, float]]] = {}
    for interval in intervals:
        start = float(interval["start"])
        end = float(interval["end"])
        overlap_start = max(start, window_start)
        overlap_end = min(end, window_end)
        if overlap_end <= overlap_start:
            continue
        sid = str(interval["speaker_id"])
        spans.setdefault(sid, []).append((overlap_start, overlap_end))
    return spans


def link_local_speakers(
    *,
    local_intervals: list[dict],
    previous_global_intervals: list[dict],
    chunk_start_s: float,
    overlap_s: float,
    local_embeddings: dict[str, list[float]],
    global_embeddings: dict[str, list[float]],
    next_global_index: int,
    threshold: float,
) -> tuple[dict[str, str], int]:
    """Map chunk-local ``speaker_id`` values onto global ``spk_N`` ids."""
    if not local_intervals:
        return {}, next_global_index

    if not previous_global_intervals:
        mapping = {
            sid: f"spk_{index}" for index, sid in enumerate(_unique_speaker_ids(local_intervals))
        }
        return mapping, len(mapping)

    overlap_end = chunk_start_s + overlap_s
    local_overlap = _speaker_spans_in_window(local_intervals, chunk_start_s, overlap_end)
    global_overlap = _speaker_spans_in_window(previous_global_intervals, chunk_start_s, overlap_end)

    mapping: dict[str, str] = {}
    used_global: set[str] = set()

    local_ids = _rank_speakers_by_overlap(local_overlap)
    for local_id in local_ids:
        local_emb = local_embeddings.get(local_id)
        best_global: str | None = None
        best_score = threshold

        if local_emb is not None:
            for global_id in global_overlap:
                if global_id in used_global:
                    continue
                global_emb = global_embeddings.get(global_id)
                if global_emb is None:
                    continue
                score = cosine_similarity(local_emb, global_emb)
                if score > best_score:
                    best_score = score
                    best_global = global_id

        if best_global is not None:
            mapping[local_id] = best_global
            used_global.add(best_global)
        else:
            mapping[local_id] = f"spk_{next_global_index}"
            next_global_index += 1

    for sid in _unique_speaker_ids(local_intervals):
        if sid not in mapping:
            mapping[sid] = f"spk_{next_global_index}"
            next_global_index += 1

    return mapping, next_global_index


def remap_speaker_ids(intervals: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Apply a local→global speaker map to diarization intervals."""
    remapped: list[dict] = []
    for interval in intervals:
        sid = str(interval["speaker_id"])
        remapped.append(
            {
                "start": float(interval["start"]),
                "end": float(interval["end"]),
                "speaker_id": mapping.get(sid, sid),
            }
        )
    return remapped


def merge_chunk_intervals(
    chunk_intervals: list[list[dict]],
    chunks: list[DiarizationChunk],
    *,
    overlap_s: float,
) -> list[dict]:
    """Combine per-chunk global intervals, keeping each chunk's owned span only."""
    merged: list[dict] = []
    total = len(chunks)
    for chunk, intervals in zip(chunks, chunk_intervals, strict=True):
        owned = select_owned_intervals(
            intervals,
            chunk,
            total_chunks=total,
            overlap_s=overlap_s,
        )
        merged.extend(owned)
    merged.sort(key=lambda item: (float(item["start"]), str(item["speaker_id"])))
    return merged


def _unique_speaker_ids(intervals: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for interval in intervals:
        sid = str(interval["speaker_id"])
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    return ordered


def _rank_speakers_by_overlap(spans: dict[str, list[tuple[float, float]]]) -> list[str]:
    totals = {
        sid: sum(max(0.0, end - start) for start, end in speaker_spans)
        for sid, speaker_spans in spans.items()
    }
    return sorted(totals, key=lambda sid: totals[sid], reverse=True)
