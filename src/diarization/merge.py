"""Assign diarization speaker ids to Whisper speech segments."""

from __future__ import annotations

from bisect import bisect_right

from audio_intel.diarization.chunks import speaker_sort_key
from audio_intel.quality.normalization import normalize_segment_text


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the overlap duration between two half-open intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


class _IntervalIndex:
    """Diarization turns sorted by start, with a running max end for range queries.

    Lookups touch only the turns that can overlap the query instead of scanning
    every turn for every word, which on a one-hour file is thousands of words
    against thousands of turns.
    """

    def __init__(self, intervals: list[dict]) -> None:
        rows = sorted(
            (
                (float(item["start"]), float(item["end"]), str(item["speaker_id"]))
                for item in intervals
            ),
            key=lambda row: row[0],
        )
        self._starts = [row[0] for row in rows]
        self._ends = [row[1] for row in rows]
        self._speakers = [row[2] for row in rows]
        self._max_end_upto: list[float] = []
        running = float("-inf")
        for end in self._ends:
            running = max(running, end)
            self._max_end_upto.append(running)

    def __bool__(self) -> bool:
        return bool(self._starts)

    def _candidates(self, query_start: float, query_end: float) -> range:
        """Indexes of turns that may intersect ``[query_start, query_end)``, ascending."""
        hi = bisect_right(self._starts, query_end)
        lo = hi
        while lo > 0 and self._max_end_upto[lo - 1] > query_start:
            lo -= 1
        return range(lo, hi)

    def speaker_at(self, time_s: float) -> str | None:
        """Speaker active at ``time_s`` (first turn in start order wins)."""
        for idx in self._candidates(time_s, time_s):
            if self._starts[idx] <= time_s < self._ends[idx]:
                return self._speakers[idx]
        return None

    def speaker_for_span(self, start: float, end: float) -> str | None:
        """Speaker with the largest overlap on ``[start, end]``."""
        best_id: str | None = None
        best_overlap = 0.0
        for idx in self._candidates(start, end):
            overlap = overlap_seconds(start, end, self._starts[idx], self._ends[idx])
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = self._speakers[idx]
        return best_id if best_overlap > 0 else None


def _strip_internal_fields(seg: dict) -> dict:
    """Copy a segment for the public timeline.

    ``words`` stay on the segment so ``align=true`` + ``diarize=true`` still
    returns word-level timestamps after speaker regroup.
    """
    return dict(seg)


def _assign_speaker_by_overlap(seg: dict, index: _IntervalIndex) -> dict:
    """Attach one ``speaker_id`` to a segment using whole-span overlap (legacy path)."""
    speaker_id = index.speaker_for_span(float(seg["start"]), float(seg["end"]))
    new_seg = _strip_internal_fields(seg)
    if speaker_id:
        new_seg["speaker_id"] = speaker_id
    return new_seg


def _text_from_words(words: list[dict]) -> str:
    """Join word tokens into segment text with spaces.

    Whisper tokens often carry a leading space, but
    :func:`~audio_intel.transcribe.segments.segment_from_whisper` strips each
    token before storage. Rejoining with an empty separator then glues words
    together; always strip tokens and insert spaces explicitly.
    """
    parts = [str(word["word"]).strip() for word in words if str(word.get("word", "")).strip()]
    return normalize_segment_text(" ".join(parts))


def _build_segment_from_words(base: dict, words: list[dict], speaker_id: str | None) -> dict:
    """Build one speech segment from a consecutive run of word timestamps."""
    text = _text_from_words(words)
    new_seg = _strip_internal_fields(base)
    new_seg.update(
        {
            "start": round(float(words[0]["start"]), 3),
            "end": round(float(words[-1]["end"]), 3),
            "text": text,
            "words": words,
        }
    )
    if speaker_id:
        new_seg["speaker_id"] = speaker_id
    return new_seg


def _assign_speakers_by_words(seg: dict, words: list[dict], index: _IntervalIndex) -> list[dict]:
    """Split/regroup one Whisper segment by per-word speaker assignment."""
    grouped: list[dict] = []
    current_words: list[dict] = []
    current_speaker: str | None = None

    for word in words:
        w_start = float(word["start"])
        w_end = float(word["end"])
        midpoint = (w_start + w_end) / 2.0
        speaker_id = index.speaker_at(midpoint)
        if speaker_id is None:
            speaker_id = index.speaker_for_span(w_start, w_end)

        if current_words and speaker_id != current_speaker:
            grouped.append(_build_segment_from_words(seg, current_words, current_speaker))
            current_words = []

        current_speaker = speaker_id
        current_words.append(word)

    if current_words:
        grouped.append(_build_segment_from_words(seg, current_words, current_speaker))

    if not grouped:
        return [_assign_speaker_by_overlap(seg, index)]

    return grouped


def assign_speakers_to_segments(
    speech_segments: list[dict],
    intervals: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Attach ``speaker_id`` to speech segments and build a speaker roster.

    When Whisper word timestamps are present, speakers are assigned per word and
    consecutive words with the same speaker are regrouped. Otherwise each segment
    receives a single speaker via temporal overlap.

    @param speech_segments — Whisper segments with ``start`` / ``end`` / ``text``;
      optional ``words`` entries ``{start, end, word}`` from faster-whisper.
    @param intervals — diarization turns with ``start`` / ``end`` / ``speaker_id``.
    @returns ``(segments, roster)`` — roster entries use ``id``, ``speech_seconds``,
      ``segment_count``.
    """
    if not intervals:
        return [_strip_internal_fields(seg) for seg in speech_segments], []

    index = _IntervalIndex(intervals)
    assigned: list[dict] = []
    for seg in speech_segments:
        words = seg.get("words")
        if isinstance(words, list) and words:
            assigned.extend(_assign_speakers_by_words(seg, words, index))
        else:
            assigned.append(_assign_speaker_by_overlap(seg, index))

    return assigned, build_speaker_roster(assigned)


def build_speaker_roster(segments: list[dict]) -> list[dict]:
    """Aggregate per-speaker duration and segment counts from assigned segments."""
    stats: dict[str, dict] = {}
    for seg in segments:
        speaker_id = seg.get("speaker_id")
        if not speaker_id:
            continue
        sid = str(speaker_id)
        if sid not in stats:
            stats[sid] = {"id": sid, "speech_seconds": 0.0, "segment_count": 0}
        duration = max(float(seg["end"]) - float(seg["start"]), 0.0)
        stats[sid]["speech_seconds"] = round(stats[sid]["speech_seconds"] + duration, 3)
        stats[sid]["segment_count"] += 1

    roster = list(stats.values())
    roster.sort(key=lambda item: speaker_sort_key(item["id"]))
    return roster
