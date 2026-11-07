# Model selection

Goal: per-speaker tracks + a background track, good enough for ASR and for dubbing (the speaker tracks
get replaced, the background is kept). That goal drives every choice below.

## Decision summary

| Stage                          | Default                                                                                                | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Alternatives                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speech / background separation | **BS-RoFormer `model_bs_roformer_ep_317_sdr_12.9755`** (`audio-separator`), background = gated bed     | Measured against Demucs `htdemucs_ft` on two Fallout 4 cutscenes (48 kHz stereo): its `Instrumental` is exactly `mix − Vocals` (residual at float rounding; Demucs stems summed to −37 dB of the mix, −3.5 dB of bass missing), 12–13.5 dB less voice leaks into the bed under speech (300 Hz–4 kHz projection −39/−40 dB vs −26/−28 dB), and the vocals stem is as clean or cleaner where nobody speaks (−79 dB vs −82 dB on the intro, −54 dB vs −42 dB on the other). The bed is then the original mix wherever the vocals stem is quiet and the instrumental only under speech (`audio/bed.py`), so no re-synthesised music plays where there was no voice to remove; 165 s in 23–25 s, 1.6 GB VRAM | other `audio-separator` checkpoints (Mel-RoFormer, MDX23C) by name in `separation.model`; Demucs was the previous default and is deliberately not kept as a fallback |
| Diarization                    | **pyannote `speaker-diarization-community-1`**, whole file                                             | Best open diarization; overlap-aware segmentation; global clustering gives stable labels across hours. Run on the vocals stem so music does not confuse it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | pyannote 3.1 (older), NeMo Sortformer, pyannoteAI precision-2 (paid)                                                                                                 |
| Speaker embeddings             | **`wespeaker`** (pyannote community-1, 256-D) in the service; SpeechBrain ECAPA-TDNN in `default.yaml` | ECAPA is cheap, 192-D, good verification EER; used for enrollment/profile and to pick the right SepFormer source, and the thresholds in `default.yaml` were tuned for it. The service uses `wespeaker` because it already holds that model for its roster, so a job loads no second model (see below)                                                                                                                                                                                                                                                                                                                                                                                                   | WeSpeaker ResNet293                                                                                                                                                  |
| Overlap detection              | **co-activity of diarization**                                                                         | community-1 already labels overlapping speech; a second model rarely adds anything                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `pyannote/overlapped-speech-detection`, `hybrid`                                                                                                                     |
| Overlap extraction (TSE)       | **SepFormer `sepformer-whamr16k` + embedding matching**                                                | Blind 2-source separation is the right tool for a *local* overlap region; matching each source to the speaker profile removes the permutation problem                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `adaptive_wiener` (no model), WeSep BSRNN (needs checkpoint), SoloSpeech / MeanFlow-TSE (research, NC licenses)                                                      |
| Enhancement                    | **off** (optional ClearerVoice MossFormer2_SE_48K / FRCRN_SE_16K)                                      | The separator already removes the bed; enhancement colours timbre, which voice cloning / dubbing QA dislikes. Enable for noisy field recordings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | resemble-enhance, DeepFilterNet                                                                                                                                      |
| 16 → 48 kHz for TSE output     | **soxr** (optional ClearerVoice MossFormer2_SR_48K)                                                    | Honest upsampling; neural SR invents highs which may or may not be wanted                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | AudioSR                                                                                                                                                              |

### Embeddings: ECAPA vs. wespeaker

Measured with `scripts/eval_separate_embeddings.py` on six 2–3 speaker interviews (8–10 min each,
RTX 5090): both backends give the same diarization (full mode does not depend on the embedder),
the same overlap decisions, and speaker tracks with ≥ 99.3 % sample agreement; leakage into other
speakers' turns differs by at most 0.02. The embedder only matters in overlap regions long enough
for target-speaker extraction, and those interviews had none. So the service defaults to
`SEPARATE_EMBEDDINGS=wespeaker` and shares the model it already holds, while `default.yaml` keeps
`speechbrain_ecapa`, whose profile thresholds were tuned for it. Rerun the script on recordings
with long overlaps before changing either default.

## Explicit non-choices

- **Blind fixed-N separation for the whole file** (MossFormer2_SS_16K etc.): cannot keep identities over
  a long recording with an unknown speaker count. Used only inside overlap regions.
- **Chunked diarization + re-linking as the default**: pyannote's own global clustering is better than
  any hand-written Hungarian re-linking of 45 s chunks. `diarization.mode: chunked` stays available for
  recordings too long for one pass.
- **Audio-visual TSE**: needs video of the speaker's face.
- **ASR inside this pipeline**: out of scope; outputs are designed to be fed to Whisper-class models.
  The service's separation job does exactly that after the pipeline, with its own Whisper.

## Hardware: RTX 5090

- Blackwell (sm_120) needs PyTorch built against CUDA 12.8 or newer:
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`.
  Plain PyPI wheels on Windows are CPU-only.
- 32 GB VRAM holds BS-RoFormer (~3 GB reserved while running), pyannote (~1 GB), ECAPA + SepFormer (~1 GB) at once;
  `device.unload_idle_models: false` is the default. Turn it on for 12–16 GB cards. In the
  service, `UNLOAD_IDLE_MODELS` governs the `separation` model group instead.
- All models run in float32; the `device.dtype` setting is reserved.

## Replacing a backend

Every stage implements an interface in `audio_intel.separate.models.base`:

- `SourceSeparator` — `separate(audio, sr) -> (speech, background)`
- `Diarizer`, `SpeakerEmbedder`, `OverlapDetector`
- `TargetSpeakerExtractor` — `extract(mixture, sr, target_embedding=, reference_audio=)`
- `SpeechEnhancer`, `SpeechSuperResolution`

Register the new class in `audio_intel.separate.models.registry` and select it from `src/separate/default.yaml`
(or a partial YAML passed with `SEPARATE_CONFIG`). The registry rejects unknown backends: a config
that names a removed one (`demucs`) fails instead of falling back.

## Licenses

| Component                       | Posture                                                                                                                                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BS-RoFormer (`audio-separator`) | audio-separator is MIT; the `model_bs_roformer_ep_317` checkpoint comes from the UVR model repository (community weights, no explicit licence) — check before commercial redistribution |
| pyannote community-1            | Open weights, gated on Hugging Face; read the model card terms                                                                                                                          |
| SpeechBrain ECAPA / SepFormer   | Apache-2.0 toolkit; check each model card                                                                                                                                               |
| ClearerVoice                    | Apache-2.0 code; check ModelScope / HF model cards                                                                                                                                      |

Verify upstream licenses before redistributing outputs commercially.
