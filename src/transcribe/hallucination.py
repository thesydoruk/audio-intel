"""Drop Whisper segments that no speech backs up.

Whisper decodes every chunk as if it contained speech, and when it does not
(music, ambience, the silence it pads a short clip with) it still produces
fluent text: "Thank you.", "Дякую за перегляд!", a credits line it learned
from captions. faster-whisper's own guard only drops a segment when
``no_speech_prob`` is high *and* the log-probability is low; these phrases
are decoded confidently, so they pass.

Measured on 265 files (podcast clips and full episodes with reference text,
Demucs background stems, synthetic noise; ``scripts/eval_hallucination.py``):

- Every hallucination had ``no_speech_prob`` >= 0.67, fourteen of fifteen >= 0.8,
  while real speech sits at a median of 0.04 with a long tail: 6 of 902 real
  segments scored >= 0.8 too, all single words under 1.3 s.
- Every hallucination was short: a canned phrase over 0.1–1.8 s of non-speech.
- Where it happened inside a chunk, Silero VAD had marked little or none of
  the span as speech; coverage below 0.3 never occurred on real speech.

So the filter drops a segment when (a) VAD covers less than
``min_speech_coverage`` of it, or (b) ``no_speech_prob`` reaches
``max_no_speech_prob`` on a span no longer than ``max_no_speech_span_s``.
At the defaults that removed 13 of 15 hallucinations and 3 of 902 real
segments (three one-word fragments of 0.3–0.4 s). Text placed past the end of
the chunk's audio (Whisper's 30 s padding) is clamped, or dropped when it
starts there.

With ``enabled=False`` the features and the reason are still attached and
nothing is dropped: the dry run the evaluation script uses.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audio_intel.config import Config

log = logging.getLogger("audio-intel")

#: Segment dict keys this module adds (internal; the HTTP body does not expose them).
FEATURE_KEYS = ("speech_coverage", "hallucination_reason")


@dataclass(frozen=True)
class HallucinationConfig:
    """Thresholds for :func:`assess_segment`.

    ``enabled=False`` keeps every segment but still annotates it.
    """

    enabled: bool = True
    #: Drop below this VAD-covered fraction of the span (0 disables the rule).
    min_speech_coverage: float = 0.3
    #: Drop at or above this ``no_speech_prob`` ... (1 disables the rule)
    max_no_speech_prob: float = 0.8
    #: ... but only when the span is at most this long (seconds).
    max_no_speech_span_s: float = 1.0
    #: Segments may end this far past the chunk's audio before they are clamped.
    overrun_s: float = 0.5

    @classmethod
    def from_config(cls, cfg: Config) -> HallucinationConfig:
        return cls(
            enabled=cfg.hallucination_filter,
            min_speech_coverage=cfg.hallucination_min_coverage,
            max_no_speech_prob=cfg.hallucination_max_no_speech_prob,
            max_no_speech_span_s=cfg.hallucination_max_no_speech_span_s,
            overrun_s=cfg.hallucination_overrun_s,
        )


def speech_coverage(start: float, end: float, spans: Iterable[tuple[float, float]]) -> float:
    """Fraction of ``[start, end)`` that the speech ``spans`` cover (0–1)."""
    span = end - start
    if span <= 0:
        return 0.0
    covered = 0.0
    for span_start, span_end in spans:
        covered += max(0.0, min(end, span_end) - max(start, span_start))
    return round(min(1.0, covered / span), 3)


def assess_segment(
    seg: dict,
    *,
    speech_spans: Iterable[tuple[float, float]],
    audio_end: float,
    cfg: HallucinationConfig,
) -> tuple[dict, str | None]:
    """Annotate one timeline segment and say why it should be dropped, if at all.

    Returns the (possibly clamped) segment and a reason, or ``None`` to keep it.
    The caller decides whether a reason actually drops the segment (``cfg.enabled``).
    """
    start = float(seg["start"])
    end = float(seg["end"])
    limit = audio_end + cfg.overrun_s

    if start >= limit:
        # Everything sits on Whisper's padding past the real audio.
        return {**seg, "speech_coverage": 0.0}, "beyond_audio"

    if end > limit:
        # Text started in real audio but ran onto the padding: keep what is
        # inside the audio, drop the words that are not.
        end = round(audio_end, 3)
        seg = {**seg, "end": end}
        if seg.get("words"):
            seg["words"] = [w for w in seg["words"] if float(w["start"]) < audio_end]

    coverage = speech_coverage(start, end, tuple(speech_spans))
    annotated = {**seg, "speech_coverage": coverage}

    if coverage < cfg.min_speech_coverage:
        return annotated, "no_vad_speech"
    no_speech = float(seg.get("no_speech_prob", 0.0))
    if no_speech >= cfg.max_no_speech_prob and end - start <= cfg.max_no_speech_span_s:
        return annotated, "no_speech_prob"
    return annotated, None


def filter_segments(
    segments: list[dict],
    *,
    speech_spans: Iterable[tuple[float, float]],
    audio_end: float,
    cfg: HallucinationConfig,
) -> tuple[list[dict], list[dict]]:
    """Split ``segments`` into (kept, dropped) per :func:`assess_segment`.

    Dropped segments carry ``hallucination_reason``; with ``cfg.enabled`` off
    they are returned in ``kept`` instead, reason attached.
    """
    spans = tuple(speech_spans)
    kept: list[dict] = []
    dropped: list[dict] = []
    for seg in segments:
        annotated, reason = assess_segment(seg, speech_spans=spans, audio_end=audio_end, cfg=cfg)
        if reason is None:
            kept.append(annotated)
            continue
        annotated["hallucination_reason"] = reason
        if cfg.enabled:
            dropped.append(annotated)
            log.info(
                "Dropped hallucinated segment [%.2f-%.2f] %s (coverage=%.2f, no_speech=%.2f): %r",
                float(annotated["start"]),
                float(annotated["end"]),
                reason,
                annotated.get("speech_coverage", 0.0),
                float(annotated.get("no_speech_prob", 0.0)),
                str(annotated.get("text", ""))[:80],
            )
        else:
            kept.append(annotated)
    return kept, dropped
