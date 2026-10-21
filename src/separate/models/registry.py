from __future__ import annotations

from typing import Any

from audio_intel.separate.models.base import (
    Diarizer,
    OverlapDetector,
    SourceSeparator,
    SpeakerEmbedder,
    SpeechEnhancer,
    SpeechSuperResolution,
    TargetSpeakerExtractor,
    resolve_device,
)


def _device_and_dtype(cfg: dict, device: str | None) -> tuple[str, str]:
    dev = resolve_device(device or cfg.get("device", {}).get("device", "auto"))
    dtype = cfg.get("device", {}).get("dtype", "float32")
    return dev, dtype


def build_separator(cfg: dict, device: str | None = None) -> SourceSeparator:
    device, dtype = _device_and_dtype(cfg, device)
    sep = cfg.get("separation", {})
    backend = sep.get("backend", "demucs")
    if backend == "demucs":
        from audio_intel.separate.models.separation.demucs import DemucsSeparator

        return DemucsSeparator(
            model_name=sep.get("model", "htdemucs_ft"),
            shifts=int(sep.get("shifts", 1)),
            overlap=float(sep.get("overlap", 0.25)),
            segment=sep.get("segment"),
            device=device,
            dtype=dtype,
        )
    if backend in {"identity", "residual", "none", "stub"}:
        from audio_intel.separate.models.separation.identity import IdentitySeparator

        return IdentitySeparator(device=device, dtype=dtype)
    raise ValueError(f"Unknown separation backend: {backend}")


def build_diarizer(cfg: dict, device: str | None = None) -> Diarizer:
    device, dtype = _device_and_dtype(cfg, device)
    diar = cfg.get("diarization", {})
    backend = diar.get("backend", "pyannote_community1")
    if backend in {"pyannote_community1", "pyannote"}:
        from audio_intel.separate.models.diarization.pyannote import PyannoteDiarizer

        return PyannoteDiarizer(
            model_id=diar.get("model_id", "pyannote/speaker-diarization-community-1"),
            use_exclusive=bool(diar.get("use_exclusive", False)),
            device=device,
            dtype=dtype,
            pipeline_provider=cfg.get("_providers", {}).get("diarization"),
        )
    if backend == "stub":
        from audio_intel.separate.models.diarization.stub import StubDiarizer

        return StubDiarizer(device=device, dtype=dtype)
    raise ValueError(f"Unknown diarizer backend: {backend}")


def build_embedder(cfg: dict, device: str | None = None) -> SpeakerEmbedder:
    device, dtype = _device_and_dtype(cfg, device)
    emb = cfg.get("embeddings", {})
    backend = emb.get("backend", "speechbrain_ecapa")
    if backend in {"speechbrain_ecapa", "ecapa"}:
        from audio_intel.separate.models.embeddings.speechbrain_ecapa import SpeechBrainECAPA

        cache_dir = cfg.get("cache", {}).get("dir", "./cache")
        return SpeechBrainECAPA(
            model_id=emb.get("model_id", "speechbrain/spkrec-ecapa-voxceleb"),
            savedir=f"{cache_dir}/models/speechbrain_ecapa",
            device=device,
            dtype=dtype,
        )
    if backend in {"wespeaker", "pyannote"}:
        from audio_intel.separate.models.embeddings.wespeaker import (
            DEFAULT_MODEL_ID,
            WeSpeakerEmbedder,
        )

        model_id = emb.get("model_id", "")
        if not model_id or "speechbrain" in model_id:
            model_id = DEFAULT_MODEL_ID
        return WeSpeakerEmbedder(
            model_id=model_id,
            device=device,
            dtype=dtype,
            inference_provider=cfg.get("_providers", {}).get("embedding"),
        )
    if backend == "stub":
        from audio_intel.separate.models.embeddings.stub import StubEmbedder

        return StubEmbedder(device=device, dtype=dtype)
    raise ValueError(f"Unknown embedder backend: {backend}")


def build_overlap_detector(cfg: dict, device: str | None = None) -> OverlapDetector:
    device, dtype = _device_and_dtype(cfg, device)
    ov = cfg.get("overlap", {})
    backend = ov.get("backend", "coactivity")
    if backend in {"hybrid", "pyannote_osd", "coactivity"}:
        from audio_intel.separate.models.overlap.hybrid import HybridOverlapDetector

        return HybridOverlapDetector(
            mode=backend,
            model_id=ov.get("model_id", "pyannote/overlapped-speech-detection"),
            min_overlap_duration=float(ov.get("min_overlap_duration", 0.15)),
            merge_gap=float(ov.get("merge_gap", 0.08)),
            device=device,
            dtype=dtype,
        )
    if backend == "stub":
        from audio_intel.separate.models.overlap.stub import StubOverlapDetector

        return StubOverlapDetector(device=device, dtype=dtype)
    raise ValueError(f"Unknown overlap backend: {backend}")


def build_tse(
    cfg: dict,
    device: str | None = None,
    *,
    embedder: SpeakerEmbedder | None = None,
) -> TargetSpeakerExtractor:
    device, dtype = _device_and_dtype(cfg, device)
    tse_cfg: dict[str, Any] = cfg.get("tse", {})
    backend = tse_cfg.get("backend", "sepformer")
    if backend in {"sepformer", "speechbrain_sepformer"}:
        from audio_intel.separate.models.tse.sepformer import SepFormerTSE

        return SepFormerTSE(cfg=tse_cfg, embedder=embedder, device=device, dtype=dtype)
    if backend in {"adaptive_wiener", "wiener"}:
        from audio_intel.separate.models.tse.adaptive_wiener import AdaptiveWienerTSE

        return AdaptiveWienerTSE(cfg=tse_cfg, device=device, dtype=dtype)
    if backend == "wesep":
        from audio_intel.separate.models.tse.wesep import WeSepTSE

        return WeSepTSE(cfg=tse_cfg, device=device, dtype=dtype)
    if backend == "stub":
        from audio_intel.separate.models.tse.stub import StubTSE

        return StubTSE(device=device, dtype=dtype)
    raise ValueError(f"Unknown TSE backend: {backend}")


def build_enhancer(
    cfg: dict, role: str = "single_speaker", device: str | None = None
) -> SpeechEnhancer:
    device, dtype = _device_and_dtype(cfg, device)
    enh = cfg.get("enhancement", {})
    sr = 48000 if role == "single_speaker" else 16000
    if not enh.get("enabled", False):
        from audio_intel.separate.models.enhancement.identity import IdentityEnhancer

        return IdentityEnhancer(sample_rate=sr, device=device, dtype=dtype)

    if role == "single_speaker":
        backend = enh.get("single_speaker_backend", "clearvoice_mossformer2_se_48k")
        model = enh.get("single_speaker_model", "MossFormer2_SE_48K")
        mix = float(enh.get("single_speaker_mix", 0.85))
    else:
        backend = enh.get("post_tse_backend", "clearvoice_frcrn_se_16k")
        model = enh.get("post_tse_model", "FRCRN_SE_16K")
        mix = float(enh.get("post_tse_mix", 0.70))

    if backend.startswith("clearvoice") or backend in {"MossFormer2_SE_48K", "FRCRN_SE_16K"}:
        from audio_intel.separate.models.enhancement.clearvoice import ClearVoiceEnhancer

        return ClearVoiceEnhancer(
            model_name=model, sample_rate=sr, mix=mix, device=device, dtype=dtype
        )
    if backend in {"identity", "none", "stub"}:
        from audio_intel.separate.models.enhancement.identity import IdentityEnhancer

        return IdentityEnhancer(sample_rate=sr, device=device, dtype=dtype)
    raise ValueError(f"Unknown enhancer backend: {backend}")


def build_super_resolution(cfg: dict, device: str | None = None) -> SpeechSuperResolution:
    device, dtype = _device_and_dtype(cfg, device)
    sr_cfg = cfg.get("super_resolution", {})
    backend = sr_cfg.get("backend", "soxr_hq")
    if not sr_cfg.get("enabled", True):
        backend = "soxr_hq"
    if backend.startswith("clearvoice"):
        from audio_intel.separate.models.super_resolution.clearvoice_sr import (
            ClearVoiceSuperResolution,
        )

        return ClearVoiceSuperResolution(
            model_name=sr_cfg.get("model", "MossFormer2_SR_48K"),
            fallback=sr_cfg.get("fallback", "soxr_hq"),
            device=device,
            dtype=dtype,
        )
    if backend in {"soxr", "soxr_hq", "fallback"}:
        from audio_intel.separate.models.super_resolution.resample_fallback import (
            SoxrSuperResolution,
        )

        return SoxrSuperResolution(output_sample_rate=48000, device=device, dtype=dtype)
    raise ValueError(f"Unknown SR backend: {backend}")
