"""Load variables from a .env file before reading os.environ.

Called at import time from :mod:`audio_intel.config`. Existing environment
variables are never overwritten (``override=False``).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file() -> Path | None:
    """Populate os.environ from `.env` (does not override existing variables)."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return None

    explicit = os.environ.get("ENV_FILE") or os.environ.get("DOTENV_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            load_dotenv(path, override=False)
            return path
        return None

    discovered = find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=False)
    if discovered:
        load_dotenv(discovered, override=False)
        return Path(discovered)
    return None
