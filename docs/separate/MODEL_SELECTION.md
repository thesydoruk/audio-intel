# Model selection

Goal: per-speaker tracks + a background track, good enough for ASR and for dubbing (the speaker tracks
get replaced, the background is kept). That goal drives every choice below.

## Decision summary

| Stage | Default | Why | Alternatives |
| --- | --- | --- | --- |
| Speech / background separation | **Demucs `htdemucs_ft`** (`demucs` package) | The standard, pip-installable model that actually separates dialogue from music and effects; stereo-aware; `vocals` stem feeds every later stage | `htdemucs` (4x faster, slightly worse), `mdx_extra`; BS-RoFormer / MDX23C via `audio-separator` (better vocals SDR, heavier install) |
| Diarization | **pyannote `speaker-diarization-community-1`**, whole file | Best open diarization; overlap-aware segmentation; global clustering gives stable labels across hours. Run on the vocals stem so music does not confuse it | pyannote 3.1 (older), NeMo Sortformer, pyannoteAI precision-2 (paid) |
| Speaker embeddings | **SpeechBrain ECAPA-TDNN** | Cheap, 192-D, good verification EER; used for enrollment/profile and to pick the right SepFormer source | WeSpeaker ResNet293, pyannote wespeaker embedding |
| Overlap detection | **co-activity of diarization** | community-1 already labels overlapping speech; a second model rarely adds anything | `pyannote/overlapped-speech-detection`, `hybrid` |
| Overlap extraction (TSE) | **SepFormer `sepformer-whamr16k` + embedding matching** | Blind 2-source separation is the right tool for a *local* overlap region; matching each source to the speaker profile removes the permutation problem | `adaptive_wiener` (no model), WeSep BSRNN (needs checkpoint), SoloSpeech / MeanFlow-TSE (research, NC licenses) |
| Enhancement | **off** (optional ClearerVoice MossFormer2_SE_48K / FRCRN_SE_16K) | Demucs already removes the bed; enhancement colours timbre, which voice cloning / dubbing QA dislikes. Enable for noisy field recordings | resemble-enhance, DeepFilterNet |
| 16 → 48 kHz for TSE output | **soxr** (optional ClearerVoice MossFormer2_SR_48K) | Honest upsampling; neural SR invents highs which may or may not be wanted | AudioSR |

## Explicit non-choices

- **Blind fixed-N separation for the whole file** (MossFormer2_SS_16K etc.): cannot keep identities over
  a long recording with an unknown speaker count. Used only inside overlap regions.
- **Chunked diarization + re-linking as the default**: pyannote's own global clustering is better than
  any hand-written Hungarian re-linking of 45 s chunks. `diarization.mode: chunked` stays available for
  recordings too long for one pass.
- **Audio-visual TSE**: needs video of the speaker's face.
- **ASR inside this pipeline**: out of scope; outputs are designed to be fed to Whisper-class models.

## Hardware: RTX 5090

- Blackwell (sm_120) needs PyTorch built against CUDA 12.8 or newer:
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`.
  Plain PyPI wheels on Windows are CPU-only.
- 32 GB VRAM holds Demucs (~2 GB while running), pyannote (~1 GB), ECAPA + SepFormer (~1 GB) at once;
  `device.unload_idle_models: false` is the default. Turn it on for 12–16 GB cards.
- All models run in float32; the `device.dtype` setting is reserved.

## Replacing a backend

Every stage implements an interface in `audio_intel.separate.models.base`:

- `SourceSeparator` — `separate(audio, sr) -> (speech, background)`
- `Diarizer`, `SpeakerEmbedder`, `OverlapDetector`
- `TargetSpeakerExtractor` — `extract(mixture, sr, target_embedding=, reference_audio=)`
- `SpeechEnhancer`, `SpeechSuperResolution`

Register the new class in `audio_intel.separate.models.registry` and select it from `src/separate/default.yaml`
or the CLI.

## Licenses

| Component | Posture |
| --- | --- |
| Demucs | MIT (weights trained on MUSDB-HQ + internal data; check the model card for commercial use) |
| pyannote community-1 | Open weights, gated on Hugging Face; read the model card terms |
| SpeechBrain ECAPA / SepFormer | Apache-2.0 toolkit; check each model card |
| ClearerVoice | Apache-2.0 code; check ModelScope / HF model cards |

Verify upstream licenses before redistributing outputs commercially.
