"""Speaker diarization and embedding extraction."""

from audio_intel.diarization.embeddings import SpeakerEmbedder
from audio_intel.diarization.merge import assign_speakers_to_segments, build_speaker_roster
from audio_intel.diarization.pyannote import SpeakerDiarizer

__all__ = [
    "SpeakerDiarizer",
    "SpeakerEmbedder",
    "assign_speakers_to_segments",
    "build_speaker_roster",
]
