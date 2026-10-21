from __future__ import annotations

import os
from typing import Any


def speechbrain_fetch_kwargs() -> dict[str, Any]:
    """Extra kwargs for ``*.from_hparams`` so downloads work without symlink privileges.

    SpeechBrain symlinks files from the Hugging Face cache into ``savedir`` by
    default. On Windows that needs Developer Mode or admin rights, so we copy
    instead (``SPEAKER_EXTRACT_SB_COPY=1`` forces copying on any platform).
    """
    if os.name == "nt" or os.environ.get("SPEAKER_EXTRACT_SB_COPY") == "1":
        try:
            from speechbrain.utils.fetching import LocalStrategy

            return {"local_strategy": LocalStrategy.COPY}
        except ImportError:  # pragma: no cover
            return {}
    return {}
