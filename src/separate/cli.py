from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from audio_intel.separate.config import load_config
from audio_intel.separate.pipeline.orchestrator import SpeakerExtractPipeline
from rich.console import Console
from rich.logging import RichHandler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-intel separate",
        description=(
            "Split a recording into sample-aligned 48 kHz tracks: one per speaker plus one "
            "background (music/effects) track, ready for ASR and dubbing."
        ),
    )
    p.add_argument("input", type=str, help="Input audio or video (anything ffmpeg can decode)")
    p.add_argument("--output", "-o", type=str, default="./output", help="Output directory")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config path (defaults to configs/default.yaml)",
    )
    p.add_argument(
        "--sample-rate", type=int, default=None, help="Master sample rate (default 48000)"
    )
    p.add_argument(
        "--channels",
        type=str,
        default=None,
        choices=["keep", "mono", "stereo"],
        help="Master/background channel layout (default keep, max 2)",
    )

    g = p.add_argument_group("speakers")
    g.add_argument(
        "--num-speakers", type=int, default=None, help="Exact number of speakers if known"
    )
    g.add_argument("--min-speakers", type=int, default=None)
    g.add_argument("--max-speakers", type=int, default=None)
    g.add_argument(
        "--diarization-mode",
        type=str,
        default=None,
        choices=["full", "chunked"],
        help="full = one global pass (default); chunked = 45 s chunks + identity re-linking",
    )
    g.add_argument("--chunk-duration", type=float, default=None, help="Chunk length (chunked mode)")
    g.add_argument("--chunk-overlap", type=float, default=None, help="Chunk overlap (chunked mode)")

    m = p.add_argument_group("models")
    m.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Speech/background separation: demucs (default) | residual | none",
    )
    m.add_argument(
        "--separation-model", type=str, default=None, help="Demucs model name (htdemucs_ft)"
    )
    m.add_argument(
        "--tse-model",
        type=str,
        default=None,
        help="Overlap extraction: sepformer (default) | adaptive_wiener | wesep | stub",
    )
    m.add_argument(
        "--no-tse", action="store_true", help="Do not separate overlaps; write the speech mix there"
    )
    m.add_argument("--enhance", action="store_true", help="Enable ClearerVoice speech enhancement")
    m.add_argument(
        "--enhancer", type=str, default=None, help="Single-speaker enhancer backend/model"
    )
    m.add_argument("--sr-model", type=str, default=None, help="16->48 kHz super-resolution backend")
    m.add_argument(
        "--stub-models", action="store_true", help="Offline stub backends (tests / smoke runs)"
    )

    r = p.add_argument_group("runtime")
    r.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "auto"])
    r.add_argument("--dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"])
    r.add_argument("--cache-dir", type=str, default=None)
    r.add_argument(
        "--resume", action="store_true", default=None, help="Reuse cached stage results (default)"
    )
    r.add_argument(
        "--force", action="store_true", default=False, help="Ignore cache, recompute everything"
    )
    r.add_argument(
        "--stop-after",
        type=str,
        default=None,
        choices=["separation", "diarization", "overlap"],
        help="Stop after this stage (debugging)",
    )
    r.add_argument("--only-diarize", action="store_true", help="Alias for --stop-after diarization")
    r.add_argument("--no-background", action="store_true", help="Do not write background_48k.wav")
    r.add_argument("--no-report", action="store_true", help="Skip report.html")
    r.add_argument("--debug", action="store_true", default=False)
    r.add_argument("-v", "--verbose", action="store_true", default=False)
    return p


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    o: dict[str, Any] = {}

    def set_path(path: list[str], value: Any) -> None:
        cur = o
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = value

    if args.sample_rate is not None:
        set_path(["audio", "target_sample_rate"], args.sample_rate)
    if args.channels is not None:
        set_path(["audio", "master_channels"], args.channels)
    if args.num_speakers is not None:
        set_path(["diarization", "num_speakers"], args.num_speakers)
    if args.min_speakers is not None:
        set_path(["diarization", "min_speakers"], args.min_speakers)
    if args.max_speakers is not None:
        set_path(["diarization", "max_speakers"], args.max_speakers)
    if args.diarization_mode is not None:
        set_path(["diarization", "mode"], args.diarization_mode)
    if args.chunk_duration is not None:
        set_path(["chunking", "duration"], args.chunk_duration)
    if args.chunk_overlap is not None:
        set_path(["chunking", "overlap"], args.chunk_overlap)
    if args.separator is not None:
        set_path(["separation", "backend"], args.separator)
    if args.separation_model is not None:
        set_path(["separation", "model"], args.separation_model)
    if args.tse_model is not None:
        set_path(["tse", "backend"], args.tse_model)
    if args.no_tse:
        set_path(["pipeline", "run_tse"], False)
    if args.enhance:
        set_path(["enhancement", "enabled"], True)
    if args.enhancer is not None:
        set_path(["enhancement", "single_speaker_backend"], args.enhancer)
        set_path(["enhancement", "enabled"], args.enhancer not in {"identity", "none", "stub"})
    if args.sr_model is not None:
        set_path(["super_resolution", "backend"], args.sr_model)
    if args.device is not None:
        set_path(["device", "device"], args.device)
    if args.dtype is not None:
        set_path(["device", "dtype"], args.dtype)
    if args.cache_dir is not None:
        set_path(["cache", "dir"], args.cache_dir)
    if args.resume:
        set_path(["pipeline", "resume"], True)
    if args.force:
        set_path(["pipeline", "force"], True)
    if args.debug:
        set_path(["pipeline", "debug"], True)
    stop_after = args.stop_after or ("diarization" if args.only_diarize else None)
    if stop_after:
        set_path(["pipeline", "stop_after"], stop_after)
    if args.no_background:
        set_path(["output", "write_background"], False)
    if args.no_report:
        set_path(["output", "write_html_report"], False)
    if args.stub_models:
        set_path(["separation", "backend"], "residual")
        set_path(["diarization", "backend"], "stub")
        set_path(["embeddings", "backend"], "stub")
        set_path(["overlap", "backend"], "stub")
        set_path(["tse", "backend"], "adaptive_wiener")
        set_path(["enhancement", "enabled"], False)
        set_path(["super_resolution", "backend"], "soxr_hq")
    return o


def main(argv: list[str] | None = None) -> int:
    console = Console(stderr=True, soft_wrap=True)
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if (args.verbose or args.debug) else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=args.debug)],
    )
    for noisy in ("speechbrain", "pyannote", "torio", "urllib3", "httpx", "filelock", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input not found:[/red] {input_path}")
        return 2

    cfg = load_config(args.config, overrides=_overrides_from_args(args))
    pipeline = SpeakerExtractPipeline(cfg)

    def progress(msg: str) -> None:
        console.print(f"[cyan]{msg}[/cyan]")

    try:
        result = pipeline.run(input_path, args.output, progress=progress)
    except Exception as exc:
        console.print(f"[red]Pipeline failed:[/red] {exc}")
        if args.debug:
            raise
        console.print("Re-run with --debug for a traceback.")
        return 1

    console.print(
        f"[green]Wrote {len(result.speaker_files)} speaker track(s) to[/green] {result.output_dir}"
    )
    for sid, path in sorted(result.speaker_files.items()):
        console.print(f"  {sid}: {path}")
    if result.background_path:
        console.print(f"  background: {result.background_path}")
    if result.metadata_path:
        console.print(f"metadata: {result.metadata_path}")
    if result.rttm_path:
        console.print(f"rttm:     {result.rttm_path}")
    if result.report_path:
        console.print(f"report:   {result.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
