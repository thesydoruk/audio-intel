"""Assign diarization speaker ids to Whisper speech segments."""

from __future__ import annotations

from audio_intel.quality.normalization import normalize_segment_text


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the overlap duration between two half-open intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speaker_id_for_time(time_s: float, intervals: list[dict]) -> str | None:
    """Map a timestamp to the diarization speaker active at that instant."""
    for interval in intervals:
        start = float(interval["start"])
        end = float(interval["end"])
        if start <= time_s < end:
            return str(interval["speaker_id"])
    return None


def _speaker_id_for_span(start: float, end: float, intervals: list[dict]) -> str | None:
    """Pick the diarization speaker with the largest overlap on ``[start, end]``."""
    best_id: str | None = None
    best_overlap = 0.0
    for interval in intervals:
        overlap = overlap_seconds(
            start,
            end,
            float(interval["start"]),
            float(interval["end"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = str(interval["speaker_id"])
    return best_id if best_overlap > 0 else None


def _strip_internal_fields(seg: dict) -> dict:
    """Drop merge-only keys before returning segments to callers."""
    return {key: value for key, value in seg.items() if key != "words"}


def _assign_speaker_by_overlap(seg: dict, intervals: list[dict]) -> dict:
    """Attach one ``speaker_id`` to a segment using whole-span overlap (legacy path)."""
    speaker_id = _speaker_id_for_span(float(seg["start"]), float(seg["end"]), intervals)
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
        }
    )
    if speaker_id:
        new_seg["speaker_id"] = speaker_id
    return new_seg


def _assign_speakers_by_words(seg: dict, words: list[dict], intervals: list[dict]) -> list[dict]:
    """Split/regroup one Whisper segment by per-word speaker assignment."""
    grouped: list[dict] = []
    current_words: list[dict] = []
    current_speaker: str | None = None

    for word in words:
        w_start = float(word["start"])
        w_end = float(word["end"])
        midpoint = (w_start + w_end) / 2.0
        speaker_id = _speaker_id_for_time(midpoint, intervals)
        if speaker_id is None:
            speaker_id = _speaker_id_for_span(w_start, w_end, intervals)

        if current_words and speaker_id != current_speaker:
            grouped.append(_build_segment_from_words(seg, current_words, current_speaker))
            current_words = []

        current_speaker = speaker_id
        current_words.append(word)

    if current_words:
        grouped.append(_build_segment_from_words(seg, current_words, current_speaker))

    if not grouped:
        return [_assign_speaker_by_overlap(seg, intervals)]

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

    assigned: list[dict] = []
    for seg in speech_segments:
        words = seg.get("words")
        if isinstance(words, list) and words:
            assigned.extend(_assign_speakers_by_words(seg, words, intervals))
        else:
            assigned.append(_assign_speaker_by_overlap(seg, intervals))

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
    roster.sort(key=lambda item: item["id"])
    return roster
