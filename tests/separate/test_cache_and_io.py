from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from audio_intel.separate.audio.io import (
    materialize_master_wav,
    read_audio,
    read_mono_resampled,
    read_segment,
    resolve_channels,
)
from audio_intel.separate.cache.store import CacheStore


def test_cache_store_roundtrip(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    key = CacheStore.key("stage", 1, 2.5, None, {"b": 1, "a": 2})
    assert key == CacheStore.key("stage", 1, 2.5, None, {"a": 2, "b": 1})
    assert store.load_json("diar", key) is None
    store.save_json("diar", key, {"segments": [{"start": 0.0, "end": 1.0}]})
    assert store.load_json("diar", key) == {"segments": [{"start": 0.0, "end": 1.0}]}
    arr = np.arange(10, dtype=np.float32)
    store.save_array("emb", key, arr)
    assert np.array_equal(store.load_array("emb", key), arr)


def test_cache_store_disabled(tmp_path: Path):
    store = CacheStore(tmp_path / "cache", enabled=False)
    store.save_json("x", "k", {"a": 1})
    assert store.load_json("x", "k") is None
    assert not (tmp_path / "cache").exists()


def test_file_hash_changes_with_content(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"hellp")
    assert CacheStore.file_hash(a) != CacheStore.file_hash(b)
    assert CacheStore.file_hash(a) == CacheStore.file_hash(a)


def test_resolve_channels():
    assert resolve_channels(1, "keep") == 1
    assert resolve_channels(2, "keep") == 2
    assert resolve_channels(6, "keep") == 2
    assert resolve_channels(2, "mono") == 1
    assert resolve_channels(1, "stereo") == 2


def test_materialize_master_resamples_and_keeps_layout(tmp_path: Path):
    sr = 44100
    t = np.arange(sr, dtype=np.float32) / sr
    stereo = np.stack(
        [0.3 * np.sin(2 * np.pi * 440 * t), 0.1 * np.sin(2 * np.pi * 440 * t)], axis=1
    )
    src = tmp_path / "src.wav"
    sf.write(src, stereo, sr)
    dest = tmp_path / "out" / "_master_48k.wav"
    info = materialize_master_wav(src, dest, target_sr=48000, channels="keep")
    assert info.sample_rate == 48000 and info.channels == 2
    assert abs(info.duration - 1.0) < 1e-3
    mono_info = materialize_master_wav(
        src, tmp_path / "out" / "mono.wav", target_sr=48000, channels="mono"
    )
    assert mono_info.channels == 1

    full, fsr = read_audio(dest)
    assert full.shape == (48000, 2) and fsr == 48000
    seg, ssr = read_segment(dest, 0.25, 0.5, sample_rate=16000)
    assert ssr == 16000 and len(seg) == 4000
    mono16 = read_mono_resampled(dest, 16000)
    assert abs(len(mono16) - 16000) <= 2


def test_materialize_refuses_to_overwrite_source(tmp_path: Path):
    src = tmp_path / "src.wav"
    sf.write(src, np.zeros(1000, dtype=np.float32), 48000)
    try:
        materialize_master_wav(src, src)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
