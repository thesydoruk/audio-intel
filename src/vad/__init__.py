"""Silero VAD."""

from audio_intel.vad.regions import SpeechRegion
from audio_intel.vad.silero import SileroVAD, load_audio_mono

__all__ = ["SileroVAD", "SpeechRegion", "load_audio_mono"]
