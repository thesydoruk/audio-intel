# Speaker separation: development

The separation pipeline lives in `src/separate/` (package `audio_intel.separate`). It was the
standalone split-speakers project and keeps its shape; the service's `scripts/setup.sh` sets up
the environment, and `pip install -e ".[separate]"` adds the model backends.

## Layout

```text
src/separate/
  default.yaml            all defaults, documented inline (ships inside the package)
  cli.py                  `audio-intel separate` argument parsing → config overrides
  audio/                  decode / master WAV / streaming resample / chunking / timeline math
  cache/                  content-addressed stage cache (JSON, npy, file paths)
  embeddings/             speaker profile bank (centroids, enrollment clips)
  evaluation/             synthetic scenarios A–J
  models/
    base.py               interfaces: SourceSeparator, Diarizer, SpeakerEmbedder, OverlapDetector,
                          TargetSpeakerExtractor, SpeechEnhancer, SpeechSuperResolution
    registry.py           config -> backend
    separation/           Demucs, identity
    diarization/          pyannote, stub
    embeddings/           SpeechBrain ECAPA, WeSpeaker (pyannote community-1), stub
    overlap/              co-activity / pyannote OSD, stub
    tse/                  SepFormer + embedding matching, adaptive Wiener, WeSep adapter, stub
    enhancement/          ClearerVoice, identity
    super_resolution/     ClearerVoice SR, soxr
  pipeline/orchestrator.py  the stages, in order
  pipeline/metadata.py      metadata.json + RTTM
  report/                 HTML report
  stitching/              crossfaded assembly of full-length tracks
  tracking/               cross-chunk identity assignment (chunked mode only)
tests/separate/           offline tests on the stub backends
```

## Pipeline stages (orchestrator.py)

1. `materialize_master_wav` — decode once to `_master_48k.wav` (mono/stereo per `audio.master_channels`).
2. `_separate` — Demucs → `_speech_48k.wav` + `background_48k.wav` (cached by master hash + model).
3. `_diarize_full` (or `_diarize_chunked`) on the 16 kHz speech stem → global turns; `merge_segments`,
   `relabel_by_duration` (speaker_00 = talks most).
4. Overlap detection on the turns.
5. `_build_profiles` — embeddings of each speaker's cleanest turns + an enrollment clip.
6. `_extract` — each turn is split by `split_by_overlap` into single / overlapped parts, padded by
   `stitching.segment_pad_ms`, and written with `TrackStitcher` (raised-cosine crossfades; padded
   neighbours overlap so there is no dip at the seam).
7. `_finish` — export tracks, residual background if configured, RTTM, metadata, report.

## Sharing models with the service

`registry.py` reads an optional `_providers` entry in the config: `_providers.diarization` and
`_providers.embedding` are callables that return an already loaded pyannote `Pipeline` /
`Inference`. The transcription service passes its own instances there, so one copy of each model
serves both transcription and separation in one process. Without providers the backends load
their own copies, as the CLI does.

## Tests

```bash
python -m pytest tests/separate      # offline, stub backends; no torch needed
ruff check src tests scripts
```

Tests cover chunking, tracking, stitching continuity, overlap splitting, cache/IO, and end-to-end
runs in full and chunked mode (mono and stereo, residual background reconstruction). They run in
the service's CI with the `dev` extra, which pulls `separate-core` (scipy, soxr, pyyaml, tqdm,
rich) but no model backends.

## Coding guidelines

- Never write to the source file; everything derives from `_master_48k.wav`.
- Time is always seconds on the master timeline; convert to samples only at the edges.
- Diarization labels are only trusted after `relabel_by_duration` (full mode) or the tracker (chunked).
- Heavy models live behind the interfaces in `models/` and are built by `registry.py`; adapters log a
  warning and fall back when an optional dependency is missing, never silently.
- Keep every output the same length as the master. `TrackStitcher` guarantees this for speaker tracks;
  separators must return stems the same length as their input.

## Optional dependencies

| Extra | Provides |
| --- | --- |
| `separate-core` | scipy, soxr, pyyaml, tqdm, rich — the pipeline with stub backends, and the tests |
| `separate` | `separate-core` + `demucs` (htdemucs_ft) + `speechbrain` (ECAPA, SepFormer); pyannote comes with `server` |
| `clearvoice` | ClearerVoice enhancement / super-resolution |
| `all` | server, vad, separate |

## Windows notes

- SpeechBrain normally symlinks model files from the Hugging Face cache, which needs Developer Mode on
  Windows. The adapters switch to copying automatically on Windows (`SPEAKER_EXTRACT_SB_COPY=1` forces
  it elsewhere).

## Hugging Face access

pyannote models are gated: accept the terms on the model card, then set `HF_TOKEN` in the
environment (the service reads `.env`; the CLI reads the environment only).
