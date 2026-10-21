"""Synthetic evaluation suite covering scenarios A–J (J is long-form placeholder)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from audio_intel.separate.config import load_config
from audio_intel.separate.pipeline.orchestrator import SpeakerExtractPipeline

SCENARIOS: list[dict[str, Any]] = [
    {"id": "A", "name": "one_clean", "description": "One speaker + clean room"},
    {"id": "B", "name": "one_noise", "description": "One speaker + noise"},
    {"id": "C", "name": "two_alternating", "description": "Two speakers alternating"},
    {"id": "D", "name": "two_overlap", "description": "Two speakers with overlap"},
    {"id": "E", "name": "three_speakers", "description": "Three+ speakers"},
    {"id": "F", "name": "return_after_gap", "description": "Speaker disappears then returns"},
    {"id": "G", "name": "music_speech", "description": "Music + speech"},
    {"id": "H", "name": "laughter_applause", "description": "Laughter/applause + speech"},
    {"id": "I", "name": "reverb", "description": "Heavy reverberation"},
    {
        "id": "J",
        "name": "long_60min",
        "description": "Long 60+ minute recording (user-supplied; not auto-synthesized)",
    },
]


def _tone(freq: float, duration: float, sr: int, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    # Harmonic stack approximates voiced speech better than a pure sine for embedding tests
    sig = amp * (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.1 * np.sin(2 * np.pi * 3 * freq * t)
    )
    # Amplitude envelope
    env = np.ones_like(sig)
    fade = min(len(sig) // 10, int(0.05 * sr))
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        env[:fade] = ramp
        env[-fade:] = ramp[::-1]
    return (sig * env).astype(np.float32)


def _noise(duration: float, sr: int, amp: float = 0.05) -> np.ndarray:
    return (amp * np.random.randn(int(duration * sr))).astype(np.float32)


def _place(track: np.ndarray, start: float, clip: np.ndarray, sr: int) -> None:
    i = int(start * sr)
    n = min(len(clip), len(track) - i)
    if n > 0:
        track[i : i + n] += clip[:n]


def generate_synthetic_suite(data_dir: str | Path, sr: int = 48000) -> dict[str, str]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.random.seed(0)
    paths: dict[str, str] = {}

    # A: one clean
    dur = 8.0
    a = np.zeros(int(dur * sr), dtype=np.float32)
    _place(a, 0.5, _tone(120, 3.0, sr), sr)
    _place(a, 4.5, _tone(120, 2.5, sr), sr)
    path = data_dir / "A_one_clean.wav"
    sf.write(path, a, sr)
    paths["A"] = str(path)

    # B: one + noise
    b = a.copy() + _noise(dur, sr, 0.08)
    path = data_dir / "B_one_noise.wav"
    sf.write(path, b, sr)
    paths["B"] = str(path)

    # C: two alternating
    dur = 12.0
    c = np.zeros(int(dur * sr), dtype=np.float32)
    _place(c, 0.5, _tone(110, 2.0, sr), sr)
    _place(c, 3.0, _tone(180, 2.0, sr, amp=0.18), sr)
    _place(c, 5.5, _tone(110, 2.0, sr), sr)
    _place(c, 8.0, _tone(180, 2.5, sr, amp=0.18), sr)
    path = data_dir / "C_two_alternating.wav"
    sf.write(path, c, sr)
    paths["C"] = str(path)

    # D: two overlap
    d = np.zeros(int(dur * sr), dtype=np.float32)
    _place(d, 1.0, _tone(110, 4.0, sr), sr)
    _place(d, 3.0, _tone(190, 4.0, sr, amp=0.18), sr)
    path = data_dir / "D_two_overlap.wav"
    sf.write(path, d, sr)
    paths["D"] = str(path)

    # E: three speakers
    dur = 15.0
    e = np.zeros(int(dur * sr), dtype=np.float32)
    _place(e, 0.5, _tone(100, 2.0, sr), sr)
    _place(e, 3.0, _tone(160, 2.0, sr, amp=0.18), sr)
    _place(e, 5.5, _tone(220, 2.0, sr, amp=0.16), sr)
    _place(e, 8.0, _tone(100, 2.0, sr), sr)
    _place(e, 10.5, _tone(160, 2.0, sr) + _tone(220, 2.0, sr, amp=0.14), sr)
    path = data_dir / "E_three_speakers.wav"
    sf.write(path, e, sr)
    paths["E"] = str(path)

    # F: return after gap
    dur = 20.0
    f = np.zeros(int(dur * sr), dtype=np.float32)
    _place(f, 0.5, _tone(115, 2.5, sr), sr)
    _place(f, 4.0, _tone(175, 2.0, sr, amp=0.18), sr)
    # long silence then return
    _place(f, 14.0, _tone(115, 3.0, sr), sr)
    path = data_dir / "F_return_after_gap.wav"
    sf.write(path, f, sr)
    paths["F"] = str(path)

    # G: music + speech
    dur = 10.0
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    music = 0.05 * np.sin(2 * np.pi * 440 * t) + 0.04 * np.sin(2 * np.pi * 554 * t)
    g = music.astype(np.float32)
    _place(g, 2.0, _tone(130, 4.0, sr), sr)
    path = data_dir / "G_music_speech.wav"
    sf.write(path, g, sr)
    paths["G"] = str(path)

    # H: laughter/applause-like bursts + speech
    h = np.zeros(int(dur * sr), dtype=np.float32)
    _place(h, 1.0, _tone(125, 3.0, sr), sr)
    bursts = (0.12 * np.abs(rng.standard_normal(int(1.5 * sr)))).astype(np.float32)
    _place(h, 4.5, bursts, sr)
    _place(h, 6.5, _tone(125, 2.0, sr), sr)
    path = data_dir / "H_laughter_applause.wav"
    sf.write(path, h, sr)
    paths["H"] = str(path)

    # I: reverb (simple FIR tail)
    dry = np.zeros(int(dur * sr), dtype=np.float32)
    _place(dry, 1.0, _tone(140, 3.0, sr), sr)
    _place(dry, 5.0, _tone(200, 3.0, sr, amp=0.17), sr)
    ir = np.zeros(int(0.4 * sr), dtype=np.float32)
    ir[0] = 1.0
    for k, amp in enumerate([0.45, 0.3, 0.2, 0.12, 0.08]):
        ir[int((0.04 + 0.05 * k) * sr)] = amp
    wet = np.convolve(dry, ir)[: len(dry)].astype(np.float32)
    path = data_dir / "I_reverb.wav"
    sf.write(path, wet, sr)
    paths["I"] = str(path)

    # J: placeholder readme for long file
    readme = data_dir / "J_long_60min.README.txt"
    readme.write_text(
        "Place a 60+ minute 48 kHz recording here as J_long_60min.wav for full endurance tests.\n",
        encoding="utf-8",
    )
    paths["J"] = str(readme)
    return paths


def run_suite(data_dir: str | Path, output_dir: str | Path) -> str:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(
        overrides={
            "separation": {"backend": "residual"},
            "diarization": {"backend": "stub", "mode": "chunked"},
            "embeddings": {"backend": "stub"},
            "overlap": {"backend": "stub"},
            "tse": {"backend": "adaptive_wiener"},
            "enhancement": {"enabled": False},
            "super_resolution": {"backend": "soxr_hq"},
            "chunking": {"duration": 6.0, "overlap": 1.5},
            "device": {"device": "cpu", "dtype": "float32", "unload_idle_models": False},
            "cache": {"enabled": False},
        }
    )
    lines = ["scenario,ok,speakers,duration,notes"]
    for sc in SCENARIOS:
        if sc["id"] == "J":
            lines.append("J,skip,0,0,user-supplied long file")
            continue
        wavs = list(data_dir.glob(f"{sc['id']}_*.wav"))
        if not wavs:
            lines.append(f"{sc['id']},missing,0,0,generate first")
            continue
        out = output_dir / sc["id"]
        try:
            # Fresh pipeline per scenario: speaker profiles must not leak between files.
            result = SpeakerExtractPipeline(cfg).run(wavs[0], out)
            # Sync check: all tracks same length
            import soundfile as sf2

            ok = True
            notes = []
            master = sf2.info(str(out / "_master_48k.wav"))
            for path in list(result.speaker_files.values()) + (
                [result.background_path] if result.background_path else []
            ):
                info = sf2.info(path)
                if info.frames != master.frames or info.samplerate != 48000:
                    ok = False
                    notes.append("sync_mismatch")
            lines.append(
                f"{sc['id']},{int(ok)},{len(result.global_speakers)},{result.input_info.duration:.2f},"
                f"{';'.join(notes) or 'ok'}"
            )
        except Exception as exc:  # pragma: no cover
            lines.append(f"{sc['id']},0,0,0,{type(exc).__name__}:{exc}")
    report = "\n".join(lines) + "\n"
    (output_dir / "suite_report.csv").write_text(report, encoding="utf-8")
    return report
