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
    """Map chunk-local ``speaker_id`` values onto global ``spk_N`` ids.

    Matching order per local speaker:
    1. Best embedding among speakers active in the overlap zone
    2. Best embedding among the full global gallery (speakers who skipped the
       overlap still reattach when the voice matches)
    3. Temporal overlap fallback when embeddings are missing
    4. Otherwise mint a new global id
    """
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
    preferred_global_ids = set(global_overlap)

    mapping: dict[str, str] = {}
    used_global: set[str] = set()

    for local_id in _rank_speakers_for_linking(local_intervals, local_overlap):
        local_emb = local_embeddings.get(local_id)
        best_global = _best_embedding_match(
            local_emb,
            global_embeddings,
            candidates=preferred_global_ids,
            used_global=used_global,
            threshold=threshold,
        )
        if best_global is None:
            best_global = _best_embedding_match(
                local_emb,
                global_embeddings,
                candidates=set(global_embeddings),
                used_global=used_global,
                threshold=threshold,
            )
        if best_global is None:
            best_global = _best_temporal_overlap_match(
                local_id,
                local_overlap,
                global_overlap,
                used_global=used_global,
            )

        if best_global is not None:
            mapping[local_id] = best_global
            used_global.add(best_global)
        else:
            mapping[local_id] = f"spk_{next_global_index}"
            next_global_index += 1

    return mapping, next_global_index


def merge_speaker_ids_by_embedding(
    embeddings: dict[str, list[float]],
    *,
    threshold: float,
) -> dict[str, str]:
    """Collapse duplicate speaker ids whose embeddings exceed ``threshold``.

    Returns a map ``old_id → canonical_id`` (identity when no merge applies).
    Canonical ids prefer the lowest ``spk_N`` index in each cluster.
    """
    ids = sorted(embeddings, key=_speaker_sort_key)
    if len(ids) <= 1:
        return {sid: sid for sid in ids}

    parent = {sid: sid for sid in ids}

    def find(sid: str) -> str:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if _speaker_sort_key(root_left) <= _speaker_sort_key(root_right):
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for index, left in enumerate(ids):
        left_emb = embeddings[left]
        for right in ids[index + 1 :]:
            if cosine_similarity(left_emb, embeddings[right]) > threshold:
                union(left, right)

    return {sid: find(sid) for sid in ids}


def remap_segment_speaker_ids(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Rewrite ``speaker_id`` on speech segments using ``mapping``."""
    if not mapping:
        return segments
    remapped: list[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        speaker_id = new_seg.get("speaker_id")
        if speaker_id is not None:
            sid = str(speaker_id)
            new_seg["speaker_id"] = mapping.get(sid, sid)
        remapped.append(new_seg)
    return remapped


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


_MIN_TEMPORAL_OVERLAP_S = 0.25


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


def _speech_seconds_by_speaker(intervals: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for interval in intervals:
        sid = str(interval["speaker_id"])
        totals[sid] = totals.get(sid, 0.0) + max(
            0.0,
            float(interval["end"]) - float(interval["start"]),
        )
    return totals


def _rank_speakers_for_linking(
    local_intervals: list[dict],
    local_overlap: dict[str, list[tuple[float, float]]],
) -> list[str]:
    """Prefer overlap speakers (by overlap duration), then remaining by speech time."""
    overlap_ranked = _rank_speakers_by_overlap(local_overlap)
    overlap_set = set(overlap_ranked)
    speech_totals = _speech_seconds_by_speaker(local_intervals)
    rest = [sid for sid in _unique_speaker_ids(local_intervals) if sid not in overlap_set]
    rest.sort(key=lambda sid: speech_totals.get(sid, 0.0), reverse=True)
    return overlap_ranked + rest


def _best_embedding_match(
    local_emb: list[float] | None,
    global_embeddings: dict[str, list[float]],
    *,
    candidates: set[str],
    used_global: set[str],
    threshold: float,
) -> str | None:
    if local_emb is None or not candidates:
        return None
    best_global: str | None = None
    best_score = threshold
    for global_id in candidates:
        if global_id in used_global:
            continue
        global_emb = global_embeddings.get(global_id)
        if global_emb is None:
            continue
        score = cosine_similarity(local_emb, global_emb)
        if score > best_score:
            best_score = score
            best_global = global_id
    return best_global


def _best_temporal_overlap_match(
    local_id: str,
    local_overlap: dict[str, list[tuple[float, float]]],
    global_overlap: dict[str, list[tuple[float, float]]],
    *,
    used_global: set[str],
) -> str | None:
    """Reuse a global id when spans heavily overlap but embeddings are unavailable."""
    local_spans = local_overlap.get(local_id)
    if not local_spans:
        return None

    best_global: str | None = None
    best_duration = 0.0
    for global_id, global_spans in global_overlap.items():
        if global_id in used_global:
            continue
        duration = 0.0
        for local_start, local_end in local_spans:
            for global_start, global_end in global_spans:
                duration += max(0.0, min(local_end, global_end) - max(local_start, global_start))
        if duration > best_duration:
            best_duration = duration
            best_global = global_id

    if best_global is not None and best_duration >= _MIN_TEMPORAL_OVERLAP_S:
        return best_global
    return None


def _speaker_sort_key(speaker_id: str) -> tuple[int, int | str]:
    if speaker_id.startswith("spk_"):
        try:
            return (0, int(speaker_id[4:]))
        except ValueError:
            pass
    return (1, speaker_id)
