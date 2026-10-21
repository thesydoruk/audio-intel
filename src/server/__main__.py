"""CLI entry point: the HTTP transcription server, or ``separate`` for the CLI.

``audio-intel`` with no arguments (or ``serve``) starts the server;
``audio-intel separate <input> [-o out] ...`` runs speaker separation on one
file without the server (see ``audio-intel separate --help``).
"""

from __future__ import annotations

import logging
import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "separate":
        from audio_intel.separate.cli import main as separate_main

        raise SystemExit(separate_main(args[1:]))
    if args and args[0] != "serve":
        raise SystemExit(f"unknown command {args[0]!r}; use `serve` or `separate`")

    import uvicorn
    from audio_intel.config import load_config
    from audio_intel.server.app import app

    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level=cfg.log_level.lower())


if __name__ == "__main__":
    main()
