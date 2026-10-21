#!/usr/bin/env python
"""Measure the hallucination filter on a labelled corpus.

The corpus is a JSONL manifest, one file per line::

    {"path": "...wav", "label": "speech" | "nospeech", "text": "<reference, optional>",
     "lang": "uk" | null, "source": "free text"}

Every file is transcribed in-process with the filter in dry-run mode, so each
segment carries its features and the verdict the filter *would* give. The
report then shows, per rule and for a grid of thresholds, what the filter
costs on speech (segments and reference characters lost) and what it buys on
non-speech (segments and seconds of text removed).

Run it inside the service image on a GPU host::

    docker run --rm -i --gpus all -v $PWD:/work -v <models>:/var/lib/whisper \\
        -e ASR_HALLUCINATION_FILTER=0 -e MODEL_CACHE_DIR=/var/lib/whisper \\
        --entrypoint python audio-intel:local scripts/eval_hallucination.py \\
        /work/manifest.jsonl --out /work/eval.jsonl

A second pass with ``--replay eval.jsonl`` re-scores the saved segments
without touching the GPU, so thresholds can be explored offline.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_NON_LETTER = re.compile(r"[^\w\s]|_|\d", re.UNICODE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _NON_LETTER.sub(" ", text.lower())).strip()


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def transcribe_corpus(manifest: Path, out: Path, *, limit: int | None) -> list[dict]:
    from audio_intel.config import load_config
    from audio_intel.transcribe.service import Transcriber

    cfg = load_config()
    if cfg.hallucination_filter:
        sys.exit("set ASR_HALLUCINATION_FILTER=0: the corpus pass must be a dry run")
    transcriber = Transcriber(cfg)
    rows: list[dict] = []
    entries = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    if limit:
        entries = entries[:limit]
    with out.open("w", encoding="utf-8") as sink:
        for index, entry in enumerate(entries, 1):
            started = time.perf_counter()
            result = transcriber.transcribe(entry["path"], override_language=entry.get("lang"))
            row = {
                **entry,
                "duration": result["duration"],
                "languages": result["languages"],
                "segments": [s for s in result["segments"] if s.get("kind") == "speech"],
                "seconds": round(time.perf_counter() - started, 2),
            }
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{index}/{len(entries)}] {entry['label']:8s} {Path(entry['path']).name}: "
                f"{len(row['segments'])} segments in {row['seconds']}s",
                file=sys.stderr,
            )
    return rows


def verdict(seg: dict, *, min_coverage: float, max_nsp: float, max_span: float) -> str | None:
    """Re-apply the rules to a saved segment (mirrors assess_segment)."""
    if seg.get("hallucination_reason") == "beyond_audio":
        return "beyond_audio"
    if seg.get("speech_coverage", 1.0) < min_coverage:
        return "no_vad_speech"
    span = float(seg["end"]) - float(seg["start"])
    if float(seg.get("no_speech_prob", 0.0)) >= max_nsp and span <= max_span:
        return "no_speech_prob"
    return None


# Ground truth for the report only: Whisper's caption-credit phrases are hallucinations
# wherever they appear, including inside a file labelled "speech" (end of a video).
ARTEFACT = re.compile(
    r"дякую за перегляд|субтитры создавал|thanks for watching|^thank you\.$", re.I
)
DEFAULTS = {"min_coverage": 0.3, "max_nsp": 0.8, "max_span": 1.0}


def is_hallucination(row: dict, seg: dict) -> bool:
    return row["label"] == "nospeech" or bool(ARTEFACT.search(seg["text"].strip()))


def score(rows: list[dict], **thresholds: float) -> dict:
    halluc = halluc_dropped = real = real_dropped = 0
    ref_sim_before = ref_sim_after = 0.0
    ref_count = 0
    reasons: Counter[str] = Counter()
    dropped_real: list[tuple[str, dict]] = []
    kept_halluc: list[tuple[str, dict]] = []
    for row in rows:
        kept_text: list[str] = []
        for seg in row["segments"]:
            reason = verdict(seg, **thresholds)
            name = Path(row["path"]).name
            if is_hallucination(row, seg):
                halluc += 1
                if reason:
                    halluc_dropped += 1
                    reasons[reason] += 1
                else:
                    kept_halluc.append((name, seg))
            else:
                real += 1
                if reason:
                    real_dropped += 1
                    dropped_real.append((name, seg))
                else:
                    kept_text.append(seg["text"])
        if row["label"] == "speech" and row.get("text"):
            ref_count += 1
            ref_sim_before += similarity(" ".join(s["text"] for s in row["segments"]), row["text"])
            ref_sim_after += similarity(" ".join(kept_text), row["text"])
    return {
        **thresholds,
        "halluc": halluc,
        "halluc_dropped": halluc_dropped,
        "real": real,
        "real_dropped": real_dropped,
        "real_dropped_pct": round(100.0 * real_dropped / max(1, real), 2),
        "ref_similarity_before": round(ref_sim_before / max(1, ref_count), 4),
        "ref_similarity_after": round(ref_sim_after / max(1, ref_count), 4),
        "reasons": dict(reasons),
        "dropped_real": dropped_real,
        "kept_halluc": kept_halluc,
    }


def report(rows: list[dict], *, show: int) -> None:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row.get("source", "?")].append(row)
    print("== corpus")
    for source, items in by_source.items():
        segs = sum(len(r["segments"]) for r in items)
        secs = sum(float(r.get("duration") or 0) for r in items)
        label = items[0]["label"]
        print(f"  {source:18s} {len(items):4d} files {secs:8.0f}s {segs:5d} segments ({label})")

    groups = {"hallucinations": [], "real speech": []}
    for r in rows:
        for s in r["segments"]:
            groups["hallucinations" if is_hallucination(r, s) else "real speech"].append(s)
    for title, segs in groups.items():
        print(f"\n== features on {title} (n={len(segs)})")
        for key in ("speech_coverage", "no_speech_prob", "avg_logprob"):
            values = sorted(float(s.get(key, 0.0)) for s in segs)
            if values:
                q = quantile(values)
                print(
                    f"  {key:16s} p1={q(0.01):6.2f} p10={q(0.1):6.2f} p50={q(0.5):6.2f} "
                    f"p90={q(0.9):6.2f} p99={q(0.99):6.2f}"
                )
        spans = sorted(float(s["end"]) - float(s["start"]) for s in segs)
        if spans:
            q = quantile(spans)
            print(
                f"  {'span_s':16s} p1={q(0.01):6.2f} p10={q(0.1):6.2f} p50={q(0.5):6.2f} "
                f"p90={q(0.9):6.2f}"
            )

    print("\n== threshold grid: hallucinations caught vs real speech lost")
    print("  coverage  nsp  span | caught        | lost            ref-sim before→after")
    for min_coverage in (0.0, 0.3, 0.5):
        for max_nsp in (0.6, 0.7, 0.8, 0.9, 1.01):
            for max_span in (0.5, 1.0, 2.0, 99.0):
                s = score(rows, min_coverage=min_coverage, max_nsp=max_nsp, max_span=max_span)
                caught_pct = 100 * s["halluc_dropped"] / max(1, s["halluc"])
                print(
                    f"  {min_coverage:8.1f} {max_nsp:4.2f} {max_span:5.1f} | "
                    f"{s['halluc_dropped']:3d}/{s['halluc']:<3d} {caught_pct:5.1f}% | "
                    f"{s['real_dropped']:3d}/{s['real']:<4d} {s['real_dropped_pct']:5.2f}%  "
                    f"{s['ref_similarity_before']:.4f}→{s['ref_similarity_after']:.4f}"
                )

    default = score(rows, **DEFAULTS)
    print(f"\n== at the defaults {DEFAULTS}: reasons {default['reasons']}")
    print(f"== real speech the defaults drop (first {show}):")
    for name, seg in default["dropped_real"][:show]:
        print(describe(name, seg))
    print(f"== hallucinations the defaults keep (first {show}):")
    for name, seg in default["kept_halluc"][:show]:
        print(describe(name, seg))


def quantile(values: list[float]):
    """Percentile lookup over a sorted list."""
    return lambda p: values[min(len(values) - 1, int(p * len(values)))]


def describe(name: str, seg: dict) -> str:
    span = float(seg["end"]) - float(seg["start"])
    return (
        f"  {name}: [{seg['start']:.1f}-{seg['end']:.1f}] cov={seg.get('speech_coverage')} "
        f"span={span:.1f} nsp={seg.get('no_speech_prob', 0):.2f} {seg['text'][:90]!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", nargs="?", type=Path, help="JSONL manifest (corpus pass)")
    parser.add_argument(
        "--out", type=Path, default=Path("eval.jsonl"), help="where segments are saved"
    )
    parser.add_argument("--replay", type=Path, help="re-score a saved --out file, no transcription")
    parser.add_argument("--limit", type=int, help="only the first N manifest entries")
    parser.add_argument("--show", type=int, default=25, help="examples to print per list")
    args = parser.parse_args()

    if args.replay:
        rows = [
            json.loads(line)
            for line in args.replay.read_text(encoding="utf-8").splitlines()
            if line
        ]
    elif args.manifest:
        rows = transcribe_corpus(args.manifest, args.out, limit=args.limit)
    else:
        parser.error("give a manifest or --replay")
    report(rows, show=args.show)


if __name__ == "__main__":
    main()
