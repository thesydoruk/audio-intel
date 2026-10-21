#!/usr/bin/env python
"""Compare speaker-embedding backends of the separation pipeline on real recordings.

Runs `audio-intel separate` twice per input, once with ``speechbrain_ecapa`` and
once with ``wespeaker``, and reports what differs: the diarization (which does
not depend on the embedder in full mode and must match), the speaker profiles'
within-speaker similarity, how many overlap regions picked a different source,
and the sample agreement of each speaker track between the two runs.

    python scripts/eval_separate_embeddings.py <recording>... --out ./eval_sep

Needs the `separate` extra and a GPU for a sensible runtime. The stage cache
makes the second run skip Demucs and diarization, so each extra backend costs
only profiles and overlap extraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BACKENDS = ("speechbrain_ecapa", "wespeaker")


def run(recording: Path, out: Path, backend: str, cache: Path) -> Path:
    from audio_intel.separate.config import load_config
    from audio_intel.separate.pipeline.orchestrator import SpeakerExtractPipeline

    target = out / recording.stem / backend
    target.mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        overrides={
            "embeddings": {"backend": backend},
            "cache": {"dir": str(cache)},
            "output": {"write_html_report": False},
            "pipeline": {"resume": True},
        }
    )
    SpeakerExtractPipeline(cfg).run(str(recording), str(target))
    return target


def rttm_turns(path: Path) -> list[tuple[str, float, float]]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            turns.append((parts[7], round(start, 3), round(start + dur, 3)))
    return turns


def track_agreement(a: Path, b: Path) -> float:
    """Fraction of samples (over active ones) where two tracks are within 1e-3."""
    x, _ = sf.read(a, dtype="float32")
    y, _ = sf.read(b, dtype="float32")
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    active = (np.abs(x) > 1e-3) | (np.abs(y) > 1e-3)
    if not active.any():
        return 1.0
    return float(np.mean(np.abs(x[active] - y[active]) < 1e-3))


def compare(recording: Path, runs: dict[str, Path]) -> dict:
    report: dict = {"recording": recording.name}
    metas = {
        b: json.loads((p / "metadata.json").read_text(encoding="utf-8")) for b, p in runs.items()
    }
    rttms = {b: rttm_turns(p / "diarization.rttm") for b, p in runs.items()}
    a, b = BACKENDS
    report["diarization_identical"] = rttms[a] == rttms[b]
    report["speakers"] = {k: len(m.get("summaries", {})) for k, m in metas.items()}
    report["overlaps"] = {k: len(m.get("overlaps", [])) for k, m in metas.items()}
    regions = {
        k: sum(1 for r in m.get("regions", []) if r.get("source_type") == "overlap")
        for k, m in metas.items()
    }
    report["overlap_regions_written"] = regions
    sims = {
        k: [
            r["embedding_similarity"]
            for r in m.get("regions", [])
            if r.get("embedding_similarity") is not None
        ]
        for k, m in metas.items()
    }
    report["embedding_similarity_median"] = {
        k: round(float(np.median(v)), 3) if v else None for k, v in sims.items()
    }
    agreement = {}
    for name, rel in metas[a].get("speaker_files", {}).items():
        other = metas[b].get("speaker_files", {}).get(name)
        if other:
            agreement[name] = round(
                track_agreement(runs[a] / Path(rel).name, runs[b] / Path(other).name), 4
            )
    report["track_sample_agreement"] = agreement
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("./eval_sep"))
    args = parser.parse_args()
    cache = args.out / "cache"
    reports = []
    for recording in args.recordings:
        runs = {backend: run(recording, args.out, backend, cache) for backend in BACKENDS}
        report = compare(recording, runs)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
    (args.out / "report.json").write_text(
        json.dumps(reports, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
