"""Speaker separation: one clean 48 kHz track per speaker plus a background bed.

Ported from the split-speakers project. The pipeline (BS-RoFormer with a gated bed,
pyannote, speaker embeddings, SepFormer in overlaps) lives in
:mod:`audio_intel.separate.pipeline` and runs inside the service as a separation job
(``POST /v1/audio/transcriptions`` with ``separate=true``), sharing the service's
diarization and embedding stacks.
"""
