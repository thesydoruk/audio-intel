# Speaker separation: usage

A separation job splits one recording into sample-aligned 48 kHz tracks: one clean mono
track per speaker (silence where they are not talking), one background track with everything
that is not speech (music, effects, ambience), and `diarization.rttm` + `metadata.json` with
who speaks when on the same timeline. Every file has exactly the length of the master, so the
tracks drop into a DAW or a dubbing pipeline as they are: run ASR per speaker track, replace it
with the dub, keep the background bed.

Separation runs inside the service as a background job: send the file to
`POST /v1/audio/transcriptions` with `separate=true`, and the job separates the recording,
then runs Whisper on every speaker track (see
[the README](../../README.md#post-v1audiotranscriptions-with-separatetrue) for the job API).
There is no standalone command-line entry point.

## Quick start

```bash
# the service, with the job lane on (SEPARATE_ENABLED=1 is the default) and HF_TOKEN for pyannote
docker run --rm --gpus all -p 8080:8080 -e HF_TOKEN=hf_xxx \
  -v audio-intel-models:/var/lib/whisper -v audio-intel-jobs:/var/lib/jobs \
  ghcr.io/thesydoruk/audio-intel:2.3.1

curl -F "file=@recording.mp4" -F separate=true http://localhost:8080/v1/audio/transcriptions
#   → 202 {"id": "<id>", "status": "queued", "url": "/v1/jobs/<id>"}
curl http://localhost:8080/v1/jobs/<id>                       # status, stage, progress, result
curl -O http://localhost:8080/v1/jobs/<id>/files/speaker_00_48k.wav
curl -O http://localhost:8080/v1/jobs/<id>/files/background_48k.wav
```

Result: `speaker_00_48k.wav …`, `background_48k.wav`, `_speech_48k.wav`, `diarization.rttm`,
`metadata.json` under `JOBS_DIR/<id>/out/`, served by `GET /v1/jobs/<id>/files/<name>`. All
WAVs have identical length.

The models volume keeps the pyannote, SpeechBrain and BS-RoFormer downloads between runs
(`HF_HOME`, `TORCH_HOME` and `AUDIO_SEPARATOR_MODEL_DIR` all point into `/var/lib/whisper`).

For local development, `pip install -e ".[separate]"` adds the model backends on top of the
service dependencies (see [DEVELOPMENT.md](DEVELOPMENT.md#optional-dependencies)).

## How it works

```text
input (any audio/video ffmpeg can read)
        |
        v
48 kHz master WAV (mono or stereo; the source is never touched)
        |
        v
BS-RoFormer (audio-separator)  ──► Vocals stem ──► gate ──►  background_48k.wav
        |  Vocals stem                                      (master where nobody speaks,
        v                                                    Instrumental stem under speech)
pyannote speaker-diarization-community-1 (whole file, global speaker labels)
        |
        +──► overlap regions (>= 2 speakers active at once)
        +──► speaker profiles: embeddings + enrollment clip per speaker
        |
        v
per-speaker assembly
   single-speaker turns  → speech stem, cut & crossfaded         (48 kHz native)
   overlapped turns      → SepFormer blind separation at 16 kHz,
                           pick the source matching the speaker,
                           16 → 48 kHz
        |
        v
speaker_00_48k.wav, speaker_01_48k.wav, …  +  metadata.json, diarization.rttm
```

Diarizing the *vocals stem* instead of the raw mix makes pyannote far more robust to music, and
BS-RoFormer is the strongest open vocals separator available. Blind fixed-N separation is used
only inside short overlap regions, where it is the right tool; the global speaker identity
always comes from pyannote's clustering plus embeddings.

### The background bed

A separation model's instrumental stem is a lossy re-synthesis of the mix. Played where
nobody speaks, it is pure artefact: there was no voice to remove. So the background is not the
model's `Instrumental` as such but a **gated bed** built from both stems
(`audio_intel.separate.audio.bed.gate_bed`):

- the vocals stem is scanned in 20 ms frames; frames within `bed.threshold_db` (default
  -45 dB) of the stem's loudest frame count as speech;
- every speech region is held for `bed.hold_ms` (150 ms) on both sides;
- inside a region the bed is the `Instrumental` stem, sample for sample; outside it, further
  than `bed.fade_ms` (150 ms) from any region, the bed is the **original master, bit for bit**;
  the ramps between are linear.

The vocals stem is the right gate because it marks where the *original* voice is at sample
precision, which is exactly where the bed must not contain it; where the dub lands later is
irrelevant to the bed. With BS-RoFormer the `Instrumental` stem is exactly `master − Vocals`,
so inside a region the bed is the master with the voice subtracted, and nothing else changes.

## Configuration

A job reads `SEPARATE_*` from the service environment (`SEPARATE_ENABLED`, `SEPARATE_DEVICE`,
`SEPARATE_EMBEDDINGS`, `SEPARATE_CONFIG`) and everything else from
[`src/separate/default.yaml`](../../src/separate/default.yaml), which documents every key
inline. `SEPARATE_CONFIG` points at a partial YAML: list only the keys to change, the rest keep
their defaults. For example:

```yaml
separation:
  bed:
    threshold_db: -50       # catch quieter voice tails (more of the bed comes from the model)
diarization:
  num_speakers: 2           # when the cast is known
stitching:
  segment_pad_ms: 200
```

Keys worth knowing:

| Key                                        | Meaning                                                                                                             |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `audio.master_channels`                    | `keep` (default, at most 2) / `mono` / `stereo`: layout of the master and so of `background_48k.wav`                |
| `separation.backend`                       | `roformer` (default; the only model) / `residual` (background = master − Σ speaker tracks) / `none` (no background) |
| `separation.model`                         | audio-separator checkpoint name; default `model_bs_roformer_ep_317_sdr_12.9755.ckpt`                                |
| `separation.overlap`, `batch_size`         | RoFormer inference windows (`null` = model defaults, 4 and 1)                                                       |
| `separation.bed.*`                         | the gate above: `threshold_db`, `hold_ms`, `fade_ms`                                                                |
| `separation.write_speech_stem`             | keep `_speech_48k.wav` (the vocals stem) among the outputs                                                          |
| `diarization.num_speakers` / `min` / `max` | constrain pyannote when the cast is known                                                                           |
| `diarization.mode`                         | `full` = one global pyannote pass (default); `chunked` = 45 s chunks + identity re-linking                          |
| `tse.backend`                              | `sepformer` (default) / `adaptive_wiener` / `wesep` / `stub`; `pipeline.run_tse: false` leaves overlaps mixed       |
| `enhancement.enabled`                      | ClearerVoice enhancement (needs `pip install -e ".[clearvoice]"`); off by default                                   |
| `stitching.segment_pad_ms`                 | padding around every diarized turn before it is cut                                                                 |
| `pipeline.run_*`, `pipeline.force`         | skip stages / ignore the stage cache                                                                                |

Anything other than `roformer`, `residual` or `none` as `separation.backend` (a leftover
`demucs`, say) fails the job at once: there is one separator, no fallback.

### Overlap handling

Where two or more people talk at once, the speech stem is split with SpeechBrain SepFormer
(`sepformer-whamr16k`, 16 kHz) and the source whose embedding is closest to the speaker's profile is
kept. That audio is band-limited to 8 kHz (soxr upsampling by default; ClearerVoice super-resolution if
installed and `super_resolution.backend: clearvoice_mossformer2_sr_48k`). Overlaps are usually a small
fraction of a conversation, so the rest of each track stays native 48 kHz.

`adaptive_wiener` is a dependency-free fallback that only attenuates the other speaker;
`pipeline.run_tse: false` leaves the overlap as-is on both tracks (often preferable for dubbing
when overlaps are short).

### Speaker embeddings

`embeddings.backend` in the config picks the voice model behind speaker profiles and overlap
source matching: `speechbrain_ecapa` (192-d; what the thresholds in `default.yaml` were tuned
for) or `wespeaker` (pyannote community-1, 256-d: the space the transcription service's speaker
roster uses, so a track's profile compares directly with the voices the service has stored).

The service sets `wespeaker` through `SEPARATE_EMBEDDINGS` (its default): it already holds that
model for its roster and lends it to the pipeline, so a job loads nothing extra. On six 2–3
speaker interviews both backends gave the same diarization, overlap decisions and tracks; see
[MODEL_SELECTION.md](MODEL_SELECTION.md).

## Outputs

```text
JOBS_DIR/<id>/out/
  _master_48k.wav         decoded master (mono/stereo, 24-bit)
  _speech_48k.wav         BS-RoFormer vocals stem (all speakers) — QA / alternative input for ASR
  background_48k.wav      the gated bed: master where nobody speaks, instrumental under speech
  speaker_00_48k.wav      mono, silence where inactive, sorted by total speech time
  speaker_01_48k.wav
  diarization.rttm        SPEAKER <file> 1 <start> <dur> <NA> <NA> speaker_00 <NA> <NA>
  metadata.json           see below
```

`GET /v1/jobs/{id}/files/{name}` serves them once the job is `done`; the job result's `tracks`
lists the names. (`report.html`, the per-speaker timeline page, is off for jobs:
`output.write_html_report`.)

`metadata.json` keys:

- `segments`: `{speaker_00: [{start, end}, …], …}` — final speech turns per speaker, seconds on the master
  timeline. Feed these to ASR if you want turn-level transcripts.
- `overlaps`: regions where ≥ 2 speakers were active.
- `regions`: what was written where (`single_speaker`, `overlap` = TSE output, `uncertain` = overlap left mixed).
- `speaker_files`, `background`, `speech_stem`, `master`: file map.
- `summaries`: per-speaker totals.

## Caching and resume

Separation stems and diarization results are cached under `cache.dir` keyed by the master file
hash and the model settings (checkpoint, windows, gate, output subtype), so re-running with
different downstream options (TSE, padding, enhancement) skips the expensive stages. The cache
keeps its own copy of the stems (`separation/<key>.speech.wav`, `.background.wav`) and hard-links
them into the output directory of whichever run hits, so a job directory can be deleted (the
service does that when a client deletes its job) without touching the entry; the service
prunes entries unused for `JOBS_RESULT_TTL_H` together with expired jobs. `pipeline.force: true` recomputes everything. The service keeps its cache in `JOBS_DIR/cache`.

## Evaluation suite

```bash
python -c "from audio_intel.separate.evaluation.suite import generate_synthetic_suite; generate_synthetic_suite('./eval_data')"
python -c "from audio_intel.separate.evaluation.suite import run_suite; print(run_suite('./eval_data', './eval_output'))"
```

Scenarios A–I are synthetic tones and only check that the pipeline runs and stays sample-aligned; they
use stub models. Scenario J expects a real 60+ minute recording that you supply.

## Hardware notes

Defaults assume one **RTX 5090** (32 GB VRAM):

- inside the service the separation models are the `separation` model group, unloaded by
  `UNLOAD_IDLE_MODELS` like the others (`device.unload_idle_models` in the YAML is ignored there)
- a 479 s three-speaker interview through `separate=true` (with `wespeaker`): 65 s, 12.1 GB
  peak VRAM
- BS-RoFormer on a 165 s 48 kHz stereo file: 23–25 s, 1.6 GB allocated / 2.9 GB reserved at
  peak; the checkpoint (640 MB) downloads on the first job. pyannote community-1 on an hour:
  about 1–2 min; SepFormer only runs on overlap regions
- RAM: the 16 kHz speech stem is held in memory (about 230 MB per hour); the master is streamed

CPU works for everything but is 20–50x slower.
