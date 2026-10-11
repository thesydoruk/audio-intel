"""audio-intel — HTTP transcription service and shared audio analysis library."""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("audio-intel")
except importlib.metadata.PackageNotFoundError:  # bare source tree
    __version__ = "0+unknown"
__author__ = "Valerii Sydoruk"
__license__ = "MIT"

from audio_intel.config import AlignmentConfig, Config, VADConfig, load_config
from audio_intel.transcribe import Transcriber
from audio_intel.types import SpeechSegment, TranscriptResult, Word

__all__ = [
    "__version__",
    "AlignmentConfig",
    "Config",
    "SpeechSegment",
    "Transcriber",
    "TranscriptResult",
    "VADConfig",
    "Word",
    "load_config",
]
