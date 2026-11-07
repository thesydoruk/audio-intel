"""The separation cache owns the stems and materializes them into the current output dir.

Regression: the cache used to store the *paths* of the first job's stems; a later job for the
same source got those paths back, and once the first job's directory was deleted (every client
deletes its job when done) the hit returned files that did not exist, or files in another
job's directory that the second job's file route could not serve.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from audio_intel.separate.cache.store import CacheStore
from audio_intel.separate.config import load_config
from audio_intel.separate.models.separation import IdentitySeparator
from audio_intel.separate.pipeline.orchestrator import SpeakerExtractPipeline

SR = 48000


class CountingSeparator(IdentitySeparator):
    """Identity speech stem, half-scale background; counts the calls."""

    def __init__(self) -> None:
        super().__init__(device="cpu", dtype="float32")
        self.calls = 0

    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        return audio.copy(), 0.5 * audio


def _mix(path: Path) -> None:
    t = np.arange(3 * SR, dtype=np.float32) / SR
    audio = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, SR)


def _cfg(tmp_path: Path, **sections):
    overrides = {
        "separation": {"backend": "roformer"},
        "pipeline": {"stop_after": "separation"},
        "device": {"device": "cpu", "dtype": "float32"},
        "cache": {"dir": str(tmp_path / "cache"), "enabled": True},
    }
    for k, v in sections.items():
        overrides.setdefault(k, {}).update(v)
    return load_config(overrides=overrides)


def _run(cfg, wav: Path, out: Path) -> CountingSeparator:
    pipe = SpeakerExtractPipeline(cfg)
    sep = CountingSeparator()
    pipe._separator = sep
    pipe.run(wav, out)
    return sep


def _read(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    return data


def test_hit_materializes_stems_into_a_fresh_output_dir(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _mix(wav)
    cfg = _cfg(tmp_path)

    first_out = tmp_path / "job-a" / "out"
    sep = _run(cfg, wav, first_out)
    assert sep.calls == 1
    assert (first_out / "_speech_48k.wav").is_file()
    assert (first_out / "background_48k.wav").is_file()
    cached = sorted(p.name for p in (tmp_path / "cache" / "separation").iterdir())
    assert len(cached) == 2
    assert cached[0].endswith(".background.wav") and cached[1].endswith(".speech.wav")
    speech_ref = _read(first_out / "_speech_48k.wav")
    bed_ref = _read(first_out / "background_48k.wav")

    # The client deleted its job: the first directory is gone, the cache entry is not.
    shutil.rmtree(tmp_path / "job-a")

    second_out = tmp_path / "job-b" / "out"
    sep2 = _run(cfg, wav, second_out)
    assert sep2.calls == 0, "a cache hit must not run the model"
    assert (second_out / "_speech_48k.wav").is_file()
    assert (second_out / "background_48k.wav").is_file()
    np.testing.assert_array_equal(_read(second_out / "_speech_48k.wav"), speech_ref)
    np.testing.assert_array_equal(_read(second_out / "background_48k.wav"), bed_ref)


def test_deleting_the_job_stem_does_not_touch_the_cache(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _mix(wav)
    cfg = _cfg(tmp_path, separation={"write_speech_stem": False})

    out = tmp_path / "job-a" / "out"
    sep = _run(cfg, wav, out)
    assert sep.calls == 1
    assert not (out / "_speech_48k.wav").exists(), "write_speech_stem: false removes the job copy"
    cache_dir = tmp_path / "cache" / "separation"
    assert any(p.name.endswith(".speech.wav") for p in cache_dir.iterdir())

    # Same source, now keeping the stem: still a hit, and the stem comes back from the cache.
    out2 = tmp_path / "job-b" / "out"
    sep2 = _run(_cfg(tmp_path), wav, out2)
    assert sep2.calls == 0
    assert (out2 / "_speech_48k.wav").is_file()


def test_force_recomputes_and_refreshes_the_entry(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _mix(wav)
    out = tmp_path / "job-a" / "out"
    assert _run(_cfg(tmp_path), wav, out).calls == 1
    out2 = tmp_path / "job-b" / "out"
    assert _run(_cfg(tmp_path, pipeline={"force": True}), wav, out2).calls == 1
    assert (out2 / "_speech_48k.wav").is_file()


def test_partial_entry_is_a_miss(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _mix(wav)
    out = tmp_path / "job-a" / "out"
    assert _run(_cfg(tmp_path), wav, out).calls == 1
    cache_dir = tmp_path / "cache" / "separation"
    next(p for p in cache_dir.iterdir() if p.name.endswith(".background.wav")).unlink()

    out2 = tmp_path / "job-b" / "out"
    assert _run(_cfg(tmp_path), wav, out2).calls == 1
    assert (out2 / "_speech_48k.wav").is_file()
    assert (out2 / "background_48k.wav").is_file()
    assert sum(1 for _ in cache_dir.iterdir()) == 2, "the entry is complete again"


def test_disabled_cache_still_writes_the_job_stems(tmp_path: Path):
    wav = tmp_path / "mix.wav"
    _mix(wav)
    cfg = _cfg(tmp_path, cache={"enabled": False})
    out = tmp_path / "job-a" / "out"
    assert _run(cfg, wav, out).calls == 1
    assert (out / "_speech_48k.wav").is_file()
    assert not list((tmp_path / "cache" / "separation").glob("*")), "nothing cached"
    assert _run(cfg, wav, tmp_path / "job-b" / "out").calls == 1


# ------------------------------------------------------------------ CacheStore
def test_store_put_get_file_roundtrip(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    src = tmp_path / "src.bin"
    src.write_bytes(b"stems")
    assert not store.has_file("separation", "k", ".speech.wav")
    store.put_file("separation", "k", ".speech.wav", src)
    assert store.has_file("separation", "k", ".speech.wav")
    src.unlink()  # the caller's copy goes away; the cache keeps its own

    dst = tmp_path / "elsewhere" / "out" / "_speech_48k.wav"
    assert store.get_file("separation", "k", ".speech.wav", dst)
    assert dst.read_bytes() == b"stems"
    assert not list(dst.parent.glob("*.tmp"))
    # a second materialization replaces whatever is there (a different file: writing
    # through the hard link would change the cache entry itself)
    dst.unlink()
    dst.write_bytes(b"stale")
    assert store.get_file("separation", "k", ".speech.wav", dst)
    assert dst.read_bytes() == b"stems"


def test_store_get_file_miss_leaves_dst_alone(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    dst = tmp_path / "out" / "x.wav"
    dst.parent.mkdir()
    dst.write_bytes(b"mine")
    assert not store.get_file("separation", "nope", ".speech.wav", dst)
    assert dst.read_bytes() == b"mine"
    disabled = CacheStore(tmp_path / "cache2", enabled=False)
    disabled.put_file("separation", "k", ".speech.wav", dst)
    assert not (tmp_path / "cache2").exists()
    assert not disabled.get_file("separation", "k", ".speech.wav", dst)


def test_store_falls_back_to_a_copy_when_linking_fails(tmp_path: Path, monkeypatch):
    def no_link(*_a, **_k):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", no_link)
    store = CacheStore(tmp_path / "cache")
    src = tmp_path / "src.bin"
    src.write_bytes(b"copied")
    store.put_file("separation", "k", ".speech.wav", src)
    dst = tmp_path / "out" / "x.wav"
    assert store.get_file("separation", "k", ".speech.wav", dst)
    assert dst.read_bytes() == b"copied"
    src.write_bytes(b"changed")
    assert dst.read_bytes() == b"copied", "a copy, not a link to the source"


def test_store_prune_removes_stale_files_and_a_hit_refreshes(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    old_src, hot_src = tmp_path / "old.bin", tmp_path / "hot.bin"  # distinct inodes
    old_src.write_bytes(b"o")
    hot_src.write_bytes(b"h")
    store.put_file("separation", "old", ".speech.wav", old_src)
    store.put_file("separation", "hot", ".speech.wav", hot_src)
    store.save_json("diarization", "d", {"a": 1})
    t0 = 1_000_000.0
    for p in (tmp_path / "cache").rglob("*"):
        if p.is_file():
            os.utime(p, (t0, t0))
    # Refreshing a hit happens through get_file: the entry's mtime moves to "now".
    assert store.get_file("separation", "hot", ".speech.wav", tmp_path / "out" / "s.wav")
    hot = store.file_path("separation", "hot", ".speech.wav")
    assert hot.stat().st_mtime > t0

    assert store.prune(max_age_s=3600, now=t0 + 3601) == 2
    assert not store.has_file("separation", "old", ".speech.wav")
    assert store.load_json("diarization", "d") is None
    assert store.has_file("separation", "hot", ".speech.wav")
    assert store.prune(max_age_s=3600, now=t0 + 3601) == 0
    assert CacheStore(tmp_path / "none", enabled=False).prune(0) == 0


@pytest.mark.skipif(os.name == "nt", reason="hard-link detection via st_nlink is POSIX-reliable")
def test_store_hard_links_on_one_volume(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    src = tmp_path / "src.bin"
    src.write_bytes(b"linked")
    store.put_file("separation", "k", ".speech.wav", src)
    assert src.stat().st_nlink == 2
