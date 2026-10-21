"""Speaker separation: one clean 48 kHz track per speaker plus a background bed.

Ported from the split-speakers project. The pipeline (Demucs, pyannote, speaker
embeddings, SepFormer in overlaps) lives in :mod:`audio_intel.separate.pipeline`;
``audio-intel separate`` is the CLI. It shares the diarization and embedding
stacks with the transcription service when run inside it.
"""
