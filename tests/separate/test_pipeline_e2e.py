from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_intel.separate.config import load_config
from audio_intel.separate.pipeline.orchestrator import SpeakerExtractPipeline


def _synth_mix(path: Path, sr: int = 48000, dur: float = 8.0, stereo: bool = False) -> np.ndarray:
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    audio = np.zeros_like(t)
    audio[int(0.5 * sr) : int(2.5 * sr)] = 0.2 * np.sin(2 * np.pi * 110 * t[: int(2.0 * sr)])
    audio[int(3.0 * sr) : int(5.0 * sr)] = 0.18 * np.sin(2 * np.pi * 180 * t[: int(2.0 * sr)])
    audio[int(5.5 * sr) : int(7.0 * sr)] = 0.2 * np.sin(2 * np.pi * 110 * t[: int(1.5 * sr)])
    # Quiet "music bed" everywhere so the residual background is non-trivial
    audio += 0.01 * np.sin(2 * np.pi * 440 * t)
    if stereo:
        audio = np.stack([audio, 0.5 * audio], axis=1)
    sf.write(path, audio, sr)
    return audio


def _stub_cfg(tmp_path: Path, **extra):
    overrides = {
        "separation": {"backend": "residual"},
        "diarization": {
            "backend": "stub",
            "mode": "full",
            "min_segment_duration": 0.0,
            "merge_gap": 0.0,
        },
        "embeddings": {"backend": "stub", "min_segment_duration": 0.3},
        "overlap": {"backend": "stub"},
        "tse": {"backend": "adaptive_wiener"},
        "enhancement": {"enabled": False},
        "super_resolution": {"backend": "soxr_hq"},
        "device": {"device": "cpu", "dtype": "float32"},
        "cache": {"dir": str(tmp_path / "cache"), "enabled": True},
    }
    for k, v in extra.items():
        overrides.setdefault(k, {}).update(v)
    return load_config(overrides=overrides)


def test_end_to_end_full_mode_writes_aligned_tracks_and_background(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _synth_mix(wav)
    out = tmp_path / "out"
    result = SpeakerExtractPipeline(_stub_cfg(tmp_path)).run(wav, out)

    assert result.metadata_path and Path(result.metadata_path).exists()
    assert result.report_path and Path(result.report_path).exists()
    assert result.rttm_path and Path(result.rttm_path).exists()
    master = sf.info(str(out / "_master_48k.wav"))
    assert result.speaker_files
    for path in result.speaker_files.values():
        info = sf.info(path)
        assert info.samplerate == 48000
        assert info.frames == master.frames
        assert info.channels == 1
    # Residual background exists, same length as master
    assert result.background_path and Path(result.background_path).exists()
    bg = sf.info(result.background_path)
    assert bg.frames == master.frames and bg.samplerate == 48000

    meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert meta["background"] == result.background_path
    assert set(meta["segments"]) == set(result.speaker_files)
    for turns in meta["segments"].values():
        assert turns and all(t["end"] > t["start"] for t in turns)

    # Speech energy lands in speaker tracks, and the sum of speakers + background ≈ master
    m, _ = sf.read(str(out / "_master_48k.wav"), dtype="float32", always_2d=True)
    total = np.zeros(master.frames, dtype=np.float32)
    for path in result.speaker_files.values():
        y, _ = sf.read(path, dtype="float32")
        total += y
    assert float(np.abs(total).max()) > 0.05
    b, _ = sf.read(result.background_path, dtype="float32", always_2d=True)
    recon = total[:, None] + b
    assert np.allclose(recon, m, atol=2e-3)


def test_stereo_master_gives_stereo_background_and_mono_speakers(tmp_path: Path):
    wav = tmp_path / "mix_stereo.wav"
    _synth_mix(wav, stereo=True)
    out = tmp_path / "out"
    result = SpeakerExtractPipeline(_stub_cfg(tmp_path)).run(wav, out)
    master = sf.info(str(out / "_master_48k.wav"))
    assert master.channels == 2
    assert sf.info(result.background_path).channels == 2
    for path in result.speaker_files.values():
        assert sf.info(path).channels == 1


def test_chunked_mode_still_works(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _synth_mix(wav)
    out = tmp_path / "out"
    cfg = _stub_cfg(
        tmp_path,
        diarization={"mode": "chunked"},
        chunking={"duration": 4.0, "overlap": 1.0, "min_chunk_duration": 1.0},
    )
    result = SpeakerExtractPipeline(cfg).run(wav, out)
    master = sf.info(str(out / "_master_48k.wav"))
    assert result.speaker_files
    for path in result.speaker_files.values():
        assert sf.info(path).frames == master.frames
    meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert meta["extra"]["diarization_mode"] == "chunked"
    assert meta["extra"]["chunk_maps"]


def test_resume_reuses_cached_diarization(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _synth_mix(wav)
    out = tmp_path / "out"
    cfg = _stub_cfg(tmp_path)
    SpeakerExtractPipeline(cfg).run(wav, out)
    cache_dir = tmp_path / "cache" / "diarization"
    entries = list(cache_dir.glob("*.json"))
    assert entries
    mtime = entries[0].stat().st_mtime
    SpeakerExtractPipeline(cfg).run(wav, out)
    assert entries[0].stat().st_mtime == mtime


def test_stop_after_diarization_writes_segments_only(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _synth_mix(wav)
    out = tmp_path / "out"
    cfg = _stub_cfg(tmp_path, pipeline={"stop_after": "diarization"})
    result = SpeakerExtractPipeline(cfg).run(wav, out)
    assert not result.speaker_files
    assert result.segments
    assert result.rttm_path and Path(result.rttm_path).read_text(encoding="utf-8").startswith(
        "SPEAKER"
    )
