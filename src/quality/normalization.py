"""Normalize ASR segment text before it leaves audio-intel.

Whisper often emits Ukrainian/Cyrillic lines in ALL CAPS. We lowercase those
lines at the source so downstream storage and UI see normal prose.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """Collapse case/punctuation/whitespace for phrase blocklist matching."""
    lowered = text.lower().replace("\u2019", "'")
    stripped = _PUNCT.sub(" ", lowered)
    return _WS.sub(" ", stripped).strip()


def is_mostly_uppercase(text: str) -> bool:
    """True when a segment has letters but none in lowercase — typical Whisper caps artefact."""
    upper = 0
    lower = 0
    for ch in text:
        if not ch.isalpha():
            continue
        up = ch.upper()
        lo = ch.lower()
        if up == lo:
            continue
        if ch == up:
            upper += 1
        elif ch == lo:
            lower += 1
    total = upper + lower
    if total < 3:
        return False
    return lower == 0


def normalize_segment_text(text: str) -> str:
    """Fix common Whisper output artefacts on one speech segment."""
    stripped = text.strip()
    if not stripped:
        return stripped
    if is_mostly_uppercase(stripped):
        return stripped.lower()
    return stripped
