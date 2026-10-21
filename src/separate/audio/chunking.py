from __future__ import annotations

from audio_intel.separate.types import ChunkSpec, TimeSpan


def build_chunks(
    duration: float,
    chunk_duration: float = 45.0,
    overlap: float = 10.0,
    min_chunk_duration: float = 5.0,
) -> list[ChunkSpec]:
    """Create overlapping chunks with exclusive core regions for stitching.

    Overlap is split evenly between previous/next cores so speech is never duplicated.
    """
    if duration <= 0:
        return []
    if overlap >= chunk_duration:
        raise ValueError("chunk overlap must be < chunk duration")

    hop = chunk_duration - overlap
    chunks: list[ChunkSpec] = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(duration, start + chunk_duration)
        if end - start < min_chunk_duration and chunks:
            # Extend previous chunk to cover the remainder
            prev = chunks[-1]
            chunks[-1] = ChunkSpec(
                index=prev.index,
                span=TimeSpan(prev.span.start, duration),
                core=TimeSpan(prev.core.start, duration),
            )
            break
        chunks.append(ChunkSpec(index=index, span=TimeSpan(start, end), core=TimeSpan(start, end)))
        if end >= duration:
            break
        start += hop
        index += 1

    # Assign exclusive cores: midpoint of each overlap
    for i, chunk in enumerate(chunks):
        core_start = chunk.span.start
        core_end = chunk.span.end
        if i > 0:
            prev = chunks[i - 1]
            overlap_start = chunk.span.start
            overlap_end = prev.span.end
            mid = 0.5 * (overlap_start + overlap_end)
            core_start = mid
        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            overlap_start = nxt.span.start
            overlap_end = chunk.span.end
            mid = 0.5 * (overlap_start + overlap_end)
            core_end = mid
        chunks[i] = ChunkSpec(
            index=chunk.index,
            span=chunk.span,
            core=TimeSpan(core_start, max(core_start, core_end)),
        )
    return chunks
