"""audio-intel — HTTP transcription service and shared audio analysis library."""

from __future__ import annotations

__version__ = "1.0.0"
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
