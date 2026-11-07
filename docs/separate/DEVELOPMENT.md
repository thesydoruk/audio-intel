# Speaker separation: development

The separation pipeline lives in `src/separate/` (package `audio_intel.separate`). It was the
standalone split-speakers project and keeps its shape; the service's `scripts/setup.sh` sets up
the environment, and `pip install -e ".[separate]"` adds the model backends.

## Layout

```text
src/separate/
  default.yaml            all defaults, documented inline (ships inside the package)
  audio/                  decode / master WAV / streaming resample / chunking / timeline math,
                          bed.py: the gated background bed (master outside speech, instrumental under it)
  cache/                  content-addressed stage cache (JSON, npy, file paths)
  embeddings/             speaker profile bank (centroids, enrollment clips)
  evaluation/             synthetic scenarios A–J
  models/
    base.py               interfaces: SourceSeparator, Diarizer, SpeakerEmbedder, OverlapDetector,
                          TargetSpeakerExtractor, SpeechEnhancer, SpeechSuperResolution
    registry.py           config -> backend
    separation/           BS-RoFormer (audio-separator) + gated bed, identity
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
1. `_separate` — BS-RoFormer → `_speech_48k.wav` (vocals stem) + `background_48k.wav` (the gated bed:
   master where the vocals stem is quiet, instrumental under speech); cached by master hash + model + gate.
1. `_diarize_full` (or `_diarize_chunked`) on the 16 kHz speech stem → global turns; `merge_segments`,
   `relabel_by_duration` (speaker_00 = talks most).
1. Overlap detection on the turns.
1. `_build_profiles` — embeddings of each speaker's cleanest turns + an enrollment clip.
1. `_extract` — each turn is split by `split_by_overlap` into single / overlapped parts, padded by
   `stitching.segment_pad_ms`, and written with `TrackStitcher` (raised-cosine crossfades; padded
   neighbours overlap so there is no dip at the seam).
1. `_finish` — export tracks, residual background if configured, RTTM, metadata, report.

## Sharing models with the service

`registry.py` reads an optional `_providers` entry in the config: `_providers.diarization` and
`_providers.embedding` are callables that return an already loaded pyannote `Pipeline` /
`Inference`. The transcription service passes its own instances there, so one copy of each model
serves both transcription and separation in one process. Without providers (the tests, the eval
scripts) the backends load their own copies. The service passes `diarization` when `SPEAKERS_ENABLED=1`, and
`embedding` only when `SEPARATE_EMBEDDINGS=wespeaker` as well (the default).

`_providers.gpu_guard` is a context-manager factory the orchestrator wraps around every GPU stage;
the service passes one that takes a GPU token at batch priority, so interactive requests go first.
Without one the orchestrator uses `nullcontext`.

The service side lives in `src/jobs/`:

```text
src/jobs/
  store.py          SQLite rows + one directory per job under JOBS_DIR; TTL janitor; orphans fail on start
  lane.py           bounded worker pool (BATCH_CONCURRENT_JOBS, BATCH_QUEUE_MAX)
  separate_job.py   SeparationEngine (config overrides, the `separation` model group) and the job stages:
                    separating → transcribing → embedding → sound_events → finishing
  service.py        what the HTTP layer calls: submit, get, files, cancel/delete, purge
```

`SeparationEngine.build_config` overrides `default.yaml` (plus `SEPARATE_CONFIG`) for the service:
the embedding backend, `cache.dir = JOBS_DIR/cache`, the device, no HTML report, resume on, and
`device.unload_idle_models: false` (the model group handles unloading).

## Tests

```bash
python -m pytest tests/separate      # offline, stub backends; no torch needed
python -m pytest tests/test_separate_job.py tests/test_job_store.py tests/test_batch_lane.py tests/test_jobs_api.py
ruff check src tests scripts
```

`tests/separate/` covers chunking, tracking, stitching continuity, overlap splitting, cache/IO,
device resolution, the gated bed, the RoFormer adapter against a fake `audio_separator`, the
separation registry, and end-to-end runs in full and chunked mode (mono and stereo, residual
background reconstruction). The job tests cover the store, the lane, the job stages against a fake
transcriber, and the `/v1/jobs` API (skipped without `httpx`). They run in the service's CI with
the `dev` extra, which pulls `separate-core` (scipy, soxr, pyyaml, tqdm) but no model backends.

`scripts/eval_separate_embeddings.py` compares the embedding backends on real recordings (needs the
`separate` extra and a GPU); see [MODEL_SELECTION.md](MODEL_SELECTION.md).

## Coding guidelines

- Never write to the source file; everything derives from `_master_48k.wav`.
- Time is always seconds on the master timeline; convert to samples only at the edges.
- Diarization labels are only trusted after `relabel_by_duration` (full mode) or the tracker (chunked).
- Heavy models live behind the interfaces in `models/` and are built by `registry.py`; adapters log a
  warning and fall back when an optional dependency is missing, never silently.
- Keep every output the same length as the master. `TrackStitcher` guarantees this for speaker tracks;
  separators must return stems the same length as their input.

## Optional dependencies

| Extra           | Provides                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| `separate-core` | scipy, soxr, pyyaml, tqdm — the pipeline with stub backends, and the tests                                         |
| `separate`      | `separate-core` + `audio-separator` (BS-RoFormer) + `speechbrain` (ECAPA, SepFormer); pyannote comes with `server` |
| `clearvoice`    | ClearerVoice enhancement / super-resolution                                                                        |
| `all`           | server, vad, separate                                                                                              |

## Windows notes

- SpeechBrain normally symlinks model files from the Hugging Face cache, which needs Developer Mode on
  Windows. The adapters switch to copying automatically on Windows (`SPEAKER_EXTRACT_SB_COPY=1` forces
  it elsewhere).

## Hugging Face access

pyannote models are gated: accept the terms on the model card, then set `HF_TOKEN` in the
environment (the service reads `.env`). The `wespeaker`
backend loads from the same gated community-1 repository.

## audio-separator in the image

`audio-separator` declares the dependencies of every architecture it can run (Demucs, `diffq`,
`onnx2torch` with torchvision, onnx). The RoFormer path imports none of them, and `diffq` has
no wheel for the image's Python and needs a compiler the runtime image lacks, so the Dockerfile
installs `audio-separator --no-deps` and then the packages the RoFormer path does import
(`beartype`, `rotary-embedding-torch`, `onnxruntime`, `pydub`, `ml_collections`, `resampy`,
`audioread`; `librosa`, `einops`, `soundfile`, `tqdm`, `pyyaml`, `requests` come with the other
extras). When bumping `audio-separator`, import `audio_separator.separator` in the built image
and run one job; a missing module shows up there, not in the unit tests.

`RoformerSeparator` writes the input as a float32 WAV at half scale and reads the stems back
at double: audio-separator mirrors the input subtype for its outputs (float in, float out) and
rescales any stem whose peak exceeds `normalization_threshold` (1.0 at most), which would break
`Instrumental == source − Vocals`; at half scale no stem gets there. It also fixes the sample
rate per `Separator` instance, so the adapter rebuilds it when an input arrives at another
rate (the model runs at the input rate; 48 kHz measured at least as clean as a 44.1 kHz round
trip).
