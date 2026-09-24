"""FastAPI HTTP server for OpenAI-compatible transcription."""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    # Importing ``app`` loads Whisper; keep request helpers importable without it.
    if name == "app":
        from audio_intel.server.app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
