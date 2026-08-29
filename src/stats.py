"""Counters for optional alignment metadata on transcript results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TranscriptStats:
    marked_alignment_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TranscriptStats:
        if not data:
            return cls()
        return cls(marked_alignment_failed=int(data.get("marked_alignment_failed", 0)))
