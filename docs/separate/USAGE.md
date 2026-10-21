# Speaker separation: usage

`audio-intel separate` splits one recording into sample-aligned 48 kHz tracks: one clean mono
track per speaker (silence where they are not talking), one background track with everything
that is not speech (music, effects, ambience), and `diarization.rttm` + `metadata.json` with
who speaks when on the same timeline. Every file has exactly the length of the master, so the
tracks drop into a DAW or a dubbing pipeline as they are: run ASR per speaker track, replace it
with the dub, keep the background bed.

It is not a transcription tool; it only does the physical separation. The transcription
service is the other half of this package.

## Quick start

```bash
pip install -e ".[separate]"         # Demucs, pyannote, SpeechBrain on top of the service deps
export HF_TOKEN=hf_xxx               # gated pyannote weights: accept the terms on Hugging Face
audio-intel separate recording.mp4 -o ./output
```

Result: `output/speaker_00_48k.wav …`, `output/background_48k.wav`, `output/diarization.rttm`,
`output/metadata.json`, `output/report.html`. All WAVs have identical length.

The Docker image ships the separation stack; inside the container the same command works
(`docker run --rm --gpus all -v $PWD:/work --entrypoint audio-intel audio-intel separate /work/in.mp4 -o /work/out`).

## How it works

```text
input (any audio/video ffmpeg can read)
        |
        v
48 kHz master WAV (mono or stereo; the source is never touched)
        |
        v
Demucs htdemucs_ft  ─────────────────────────►  background_48k.wav
        |  speech (vocals) stem
        v
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
speaker_00_48k.wav, speaker_01_48k.wav, …  +  metadata.json, diarization.rttm, report.html
```

Diarizing the *vocals stem* instead of the raw mix makes pyannote far more robust to music, and
Demucs is the tool actually trained to pull dialogue away from a music/effects bed. Blind fixed-N
separation is used only inside short overlap regions, where it is the right tool; the global
speaker identity always comes from pyannote's clustering plus embeddings.

## Processing a file

```bash
audio-intel separate input.wav -o ./output
audio-intel separate input.mkv -o ./output --num-speakers 3
audio-intel separate input.wav -o ./output --channels stereo --force
python -m audio_intel.separate input.wav -o ./output      # the same entry point
```

### Flags

| Flag | Meaning |
| --- | --- |
| `--num-speakers N` / `--min-speakers` / `--max-speakers` | Constrain the diarizer when you know the cast |
| `--channels keep\|mono\|stereo` | Layout of the master and therefore of `background_48k.wav` (default keep, max 2) |
| `--separator demucs\|residual\|none` | How the background is produced (see below) |
| `--separation-model htdemucs_ft\|htdemucs\|mdx_extra` | Demucs variant; `htdemucs_ft` is best and 4x slower than `htdemucs` |
| `--diarization-mode full\|chunked` | `full` = one global pyannote pass (default) |
| `--tse-model sepformer\|adaptive_wiener\|wesep\|stub` | Overlap handling |
| `--no-tse` | Do not separate overlaps; both speakers get the mixed speech there |
| `--enhance` | Enable ClearerVoice enhancement (needs `pip install -e ".[clearvoice]"`) |
| `--device cuda\|cpu\|auto` | Compute device (default auto) |
| `--cache-dir DIR` / `--force` | Stage cache location / ignore the cache |
| `--stop-after separation\|diarization\|overlap` | Debug: stop early, still write RTTM/metadata |
| `--no-background` / `--no-report` | Skip those outputs |
| `--stub-models` | Offline stub backends (no downloads), for smoke tests |
| `--config my.yaml` | Override any setting of `src/separate/default.yaml` |
| `-v` / `--debug` | Verbose logs / tracebacks |

### Background modes

- **demucs** (default): Demucs separates the whole mix once. `vocals` becomes the speech stem that every
  later stage works on; `drums + bass + other` become `background_48k.wav`. This is what you want for
  film, podcasts with music, anything with a bed.
- **residual**: no separation model. Speaker tracks are cut from the master and
  `background = master − Σ speaker tracks`. Useful for clean studio dialogue or when Demucs is not
  installed. The background will contain whatever the diarizer missed (breaths, laughter between turns).
- **none**: speaker tracks only.

### Overlap handling

Where two or more people talk at once, the speech stem is split with SpeechBrain SepFormer
(`sepformer-whamr16k`, 16 kHz) and the source whose embedding is closest to the speaker's profile is
kept. That audio is band-limited to 8 kHz (soxr upsampling by default; ClearerVoice super-resolution if
installed and `super_resolution.backend: clearvoice_mossformer2_sr_48k`). Overlaps are usually a small
fraction of a conversation, so the rest of each track stays native 48 kHz.

`adaptive_wiener` is a dependency-free fallback that only attenuates the other speaker; `--no-tse`
leaves the overlap as-is on both tracks (often preferable for dubbing when overlaps are short).

### Speaker embeddings

`embeddings.backend` in the config picks the voice model behind speaker profiles and overlap
source matching: `speechbrain_ecapa` (192-d; the default, and what the thresholds in
`default.yaml` were tuned for) or `wespeaker` (pyannote community-1, 256-d: the space the
transcription service's speaker roster uses, so a track's profile compares directly with the
voices the service has stored). See [MODEL_SELECTION.md](MODEL_SELECTION.md).

## Outputs

```text
output/
  _master_48k.wav         decoded master (mono/stereo, 24-bit)
  _speech_48k.wav         Demucs speech stem (all speakers) — QA / alternative input for ASR
  background_48k.wav      music, effects, ambience (same channels as master)
  speaker_00_48k.wav      mono, silence where inactive, sorted by total speech time
  speaker_01_48k.wav
  diarization.rttm        SPEAKER <file> 1 <start> <dur> <NA> <NA> speaker_00 <NA> <NA>
  metadata.json           see below
  report.html             per-speaker / overlap timelines
```

`metadata.json` keys:

- `segments`: `{speaker_00: [{start, end}, …], …}` — final speech turns per speaker, seconds on the master
  timeline. Feed these to ASR if you want turn-level transcripts.
- `overlaps`: regions where ≥ 2 speakers were active.
- `regions`: what was written where (`single_speaker`, `overlap` = TSE output, `uncertain` = overlap left mixed).
- `speaker_files`, `background`, `speech_stem`, `master`: file map.
- `summaries`: per-speaker totals.

## Caching and resume

Separation stems and diarization results are cached under `cache.dir` (default `./cache`) keyed by
the master file hash and the model settings, so re-running with different downstream options (TSE,
padding, enhancement) skips the expensive stages. `--force` recomputes everything.

## Evaluation suite

```bash
python -c "from audio_intel.separate.evaluation.suite import generate_synthetic_suite; generate_synthetic_suite('./eval_data')"
python -c "from audio_intel.separate.evaluation.suite import run_suite; print(run_suite('./eval_data', './eval_output'))"
```

Scenarios A–I are synthetic tones and only check that the pipeline runs and stays sample-aligned; they
use stub models. Scenario J expects a real 60+ minute recording that you supply.

## Hardware notes

Defaults assume one **RTX 5090** (32 GB VRAM):

- all models stay loaded (`device.unload_idle_models: false`); set it to `true` on a 12–16 GB card
- Demucs `htdemucs_ft` on a 1 h stereo file: about 1–2 min; pyannote community-1: about 1–2 min;
  SepFormer only runs on overlap regions
- RAM: the 16 kHz speech stem is held in memory (about 230 MB per hour); the master is streamed

CPU works for everything but is 20–50x slower; `--stub-models` gives an instant smoke run.
