"""RoformerSeparator against a fake ``audio_separator`` that writes two stems to disk."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from audio_intel.separate.models.registry import build_separator
from audio_intel.separate.models.separation import IdentitySeparator, RoformerSeparator

SR = 48000


class FakeSeparator:
    """Writes Vocals = input inside [0.8, 1.2) s, Instrumental = input - Vocals, as float WAVs.

    ``extra`` appends that many zero samples to both stems so trimming is exercised;
    ``short`` drops samples so padding is.
    """

    instances: list[FakeSeparator] = []
    extra = 0
    short = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.output_dir = kwargs.get("output_dir")
        self.model_instance = None
        self.loaded: str | None = None
        self.calls: list[tuple[str, dict]] = []
        FakeSeparator.instances.append(self)

    def load_model(self, model_filename):
        self.loaded = model_filename
        self.model_instance = types.SimpleNamespace(output_dir=self.output_dir)

    def separate(self, path, names):
        self.calls.append((path, dict(names)))
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        info = sf.info(path)
        assert info.subtype == "FLOAT", (
            "the adapter must write float input so stems come back float"
        )
        assert sr == self.kwargs["sample_rate"]
        n = len(audio)
        vocals = np.zeros_like(audio)
        sl = slice(int(0.8 * sr), int(1.2 * sr))
        vocals[sl] = audio[sl]
        instrumental = audio - vocals
        if self.extra:
            pad = np.zeros((self.extra, audio.shape[1]), dtype=np.float32)
            vocals, instrumental = np.vstack([vocals, pad]), np.vstack([instrumental, pad])
        if self.short:
            vocals, instrumental = vocals[: n - self.short], instrumental[: n - self.short]
        out = self.model_instance.output_dir
        sf.write(os.path.join(out, names["Vocals"] + ".wav"), vocals, sr, subtype="FLOAT")
        sf.write(
            os.path.join(out, names["Instrumental"] + ".wav"), instrumental, sr, subtype="FLOAT"
        )
        return [names["Instrumental"] + ".wav", names["Vocals"] + ".wav"]


@pytest.fixture
def fake_audio_separator(monkeypatch):
    pkg = types.ModuleType("audio_separator")
    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = FakeSeparator
    pkg.separator = mod
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        ),
    )
    FakeSeparator.instances = []
    FakeSeparator.extra = 0
    FakeSeparator.short = 0
    yield FakeSeparator


def _mix(seconds: float = 2.0, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform(-0.9, 0.9, (int(seconds * SR), channels)).astype(np.float32)


def test_stereo_round_trip_stems_scaled_back_and_bed_gated(fake_audio_separator, tmp_path: Path):
    audio = _mix()
    sep = RoformerSeparator(model_dir=tmp_path / "models", device="cpu")
    speech, background = sep.separate(audio, SR)

    assert speech.shape == audio.shape == background.shape
    assert speech.dtype == np.float32 and background.dtype == np.float32
    # the fake saw the input at half scale and float; the adapter undid the scale
    fake = fake_audio_separator.instances[-1]
    assert fake.loaded == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert fake.kwargs["normalization_threshold"] == 1.0 and fake.kwargs["use_soundfile"]
    assert fake.kwargs["model_file_dir"] == str(tmp_path / "models")
    burst = slice(int(0.8 * SR), int(1.2 * SR))
    assert np.allclose(speech[burst], audio[burst], atol=1e-6)
    assert np.allclose(speech[: int(0.8 * SR)], 0.0)
    # the bed: instrumental (= 0 here) under the voice, the untouched source away from it
    assert np.allclose(background[int(0.9 * SR) : int(1.1 * SR)], 0.0, atol=1e-6)
    assert np.array_equal(background[: int(0.4 * SR)], audio[: int(0.4 * SR)])
    assert np.array_equal(background[int(1.6 * SR) :], audio[int(1.6 * SR) :])


def test_mono_input_gives_mono_stems(fake_audio_separator, tmp_path: Path):
    audio = _mix(channels=1)
    sep = RoformerSeparator(model_dir=tmp_path, device="cpu")
    speech, background = sep.separate(audio, SR)
    assert speech.shape == audio.shape == background.shape
    assert np.allclose(
        speech[int(0.8 * SR) : int(1.2 * SR)], audio[int(0.8 * SR) : int(1.2 * SR)], atol=1e-6
    )
    assert np.array_equal(background[: int(0.4 * SR)], audio[: int(0.4 * SR)])


def test_one_dimensional_input_is_accepted(fake_audio_separator, tmp_path: Path):
    audio = _mix(channels=1)[:, 0]
    speech, background = RoformerSeparator(model_dir=tmp_path, device="cpu").separate(audio, SR)
    assert speech.shape == (len(audio), 1) and background.shape == (len(audio), 1)


@pytest.mark.parametrize("extra, short", [(123, 0), (0, 77)])
def test_stems_are_trimmed_or_padded_to_the_input_length(
    fake_audio_separator, tmp_path: Path, extra, short
):
    fake_audio_separator.extra = extra
    fake_audio_separator.short = short
    audio = _mix()
    speech, background = RoformerSeparator(model_dir=tmp_path, device="cpu").separate(audio, SR)
    assert len(speech) == len(audio) == len(background)
    if short:
        assert np.all(speech[-short:] == 0.0)


def test_unload_drops_the_model_and_separate_reloads_it(fake_audio_separator, tmp_path: Path):
    sep = RoformerSeparator(model_dir=tmp_path, device="cpu")
    sep.separate(_mix(1.0), SR)
    assert sep.loaded and len(fake_audio_separator.instances) == 1
    sep.unload()
    assert not sep.loaded
    sep.separate(_mix(1.0), SR)
    assert sep.loaded and len(fake_audio_separator.instances) == 2


def test_a_new_sample_rate_rebuilds_the_separator(fake_audio_separator, tmp_path: Path):
    sep = RoformerSeparator(model_dir=tmp_path, device="cpu")
    sep.separate(_mix(1.0), SR)
    sep.separate(_mix(1.0), SR)
    assert len(fake_audio_separator.instances) == 1
    sep.separate(np.zeros((44100, 2), dtype=np.float32), 44100)
    assert len(fake_audio_separator.instances) == 2
    assert fake_audio_separator.instances[-1].kwargs["sample_rate"] == 44100


def test_registry_builds_roformer_with_the_bed_settings(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    monkeypatch.delenv("MODEL_CACHE_DIR", raising=False)
    cfg = {
        "separation": {
            "backend": "roformer",
            "model": "some_other_roformer.ckpt",
            "segment_size": 512,
            "overlap": 8,
            "batch_size": 2,
            "bed": {"threshold_db": -50, "hold_ms": 100, "fade_ms": 80},
        },
        "cache": {"dir": str(tmp_path)},
        "device": {"device": "cpu"},
    }
    sep = build_separator(cfg)
    assert isinstance(sep, RoformerSeparator)
    assert sep.model_filename == "some_other_roformer.ckpt"
    assert (sep.segment_size, sep.overlap, sep.batch_size) == (512, 8, 2)
    assert (sep.bed_threshold_db, sep.bed_hold_ms, sep.bed_fade_ms) == (-50.0, 100.0, 80.0)
    assert sep.model_dir == tmp_path / "models" / "audio-separator"


def test_registry_model_dir_follows_the_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "volume"))
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    sep = build_separator({"separation": {"backend": "roformer"}, "device": {"device": "cpu"}})
    assert sep.model_dir == tmp_path / "volume" / "audio-separator"
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "explicit"))
    sep = build_separator({"separation": {"backend": "roformer"}, "device": {"device": "cpu"}})
    assert sep.model_dir == tmp_path / "explicit"


@pytest.mark.parametrize("backend", ["identity", "residual", "none", "stub"])
def test_registry_keeps_the_identity_backends(backend):
    sep = build_separator({"separation": {"backend": backend}, "device": {"device": "cpu"}})
    assert isinstance(sep, IdentitySeparator)


@pytest.mark.parametrize("backend", ["demucs", "htdemucs_ft", "mdx", ""])
def test_registry_rejects_every_other_backend_including_demucs(backend):
    with pytest.raises(ValueError, match="Unknown separation backend"):
        build_separator({"separation": {"backend": backend}, "device": {"device": "cpu"}})


def test_default_config_selects_roformer():
    from audio_intel.separate.config import load_config

    cfg = load_config()
    assert cfg["separation"]["backend"] == "roformer"
    assert cfg["separation"]["model"] == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert cfg["separation"]["bed"] == {"threshold_db": -45, "hold_ms": 150, "fade_ms": 150}
    assert isinstance(build_separator({**cfg, "device": {"device": "cpu"}}), RoformerSeparator)
