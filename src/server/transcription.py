"""Shared transcription request/response helpers for HTTP handlers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TranscriptionParams:
    """Normalized per-request transcription options."""

    override_language: str | None
    align: bool
    diarize: bool
    sound_events: bool
    aed_overrides: dict[str, float | int | bool | None]
    aed_debug_top_n: int
    response_format: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptionParams:
        return cls(
            override_language=data.get("override_language"),
            align=bool(data.get("align", False)),
            diarize=bool(data.get("diarize", False)),
            sound_events=bool(data.get("sound_events", False)),
            aed_overrides=dict(data.get("aed_overrides") or {}),
            aed_debug_top_n=int(data.get("aed_debug_top_n", 0)),
            response_format=str(data.get("response_format", "verbose_json")),
        )


def normalize_language(language: str | None) -> str | None:
    """Map OpenAI-style language form values to a forced code or ``None`` (auto)."""
    if not language:
        return None
    stripped = language.strip()
    if stripped.lower() in {"", "auto"}:
        return None
    return stripped


def build_aed_overrides(
    *,
    aed_min_score: float | None,
    aed_top_k: int | None,
    aed_merge_gap_s: float | None,
    aed_min_duration_s: float | None,
    aed_exclude_speech: bool | None,
) -> dict[str, float | int | bool | None]:
    return {
        "min_score": aed_min_score,
        "top_k": aed_top_k,
        "merge_gap_s": aed_merge_gap_s,
        "min_duration_s": aed_min_duration_s,
        "exclude_speech": aed_exclude_speech,
    }


def segment_payload(seg: dict) -> dict:
    """Serialize one timeline entry (speech or sound) for the verbose JSON body."""
    if seg.get("kind") == "sound":
        return {
            "kind": "sound",
            "start": seg["start"],
            "end": seg["end"],
            "label": seg["label"],
            "index": seg["index"],
            "score": seg["score"],
            "prompt_relevant": seg.get("prompt_relevant", True),
        }
    payload = {
        "kind": "speech",
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"],
        "confidence": seg.get("confidence"),
        **({"language": seg["language"]} if seg.get("language") else {}),
        **({"speaker_id": seg["speaker_id"]} if seg.get("speaker_id") else {}),
    }
    if seg.get("words"):
        payload["words"] = seg["words"]
    if "alignment_failed" in seg:
        payload["alignment_failed"] = bool(seg["alignment_failed"])
    return payload


def build_transcription_body(result: dict) -> dict:
    """Shape a transcriber result dict into the verbose JSON response body."""
    languages = result["languages"]
    body: dict[str, Any] = {
        "task": "transcribe",
        "language": languages[0] if languages else "unknown",
        "languages": languages,
        "confidence": result["confidence"],
        "duration": result["duration"],
        "text": result["text"],
        "segments": [segment_payload(seg) for seg in result["segments"]],
    }
    if result.get("speakers"):
        body["speakers"] = result["speakers"]
    if result.get("aed_debug"):
        body["aed_debug"] = result["aed_debug"]
    if result.get("alignment_requested"):
        body["alignment_requested"] = True
        body["alignment_applied"] = result.get("alignment_applied", False)
        body["alignment_failed"] = result.get("alignment_failed", False)
        body["alignment_failed_segments"] = result.get("alignment_failed_segments", 0)
        if result.get("aligned_at"):
            body["aligned_at"] = result["aligned_at"]
    return body
