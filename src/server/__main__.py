"""CLI entry point for the HTTP transcription server."""

from __future__ import annotations

import logging


def main() -> None:
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
