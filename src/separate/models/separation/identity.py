from __future__ import annotations

import numpy as np
from audio_intel.separate.models.base import SourceSeparator


class IdentitySeparator(SourceSeparator):
    """No-op separator: the whole mix is treated as speech, background is silence.

    Used for tests and for recordings that contain no music or effects. The
    orchestrator can still derive a residual background (master minus speech)
    when ``separation.backend`` is ``residual``.
    """

    name = "identity"

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        _ = sample_rate
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        return audio.copy(), np.zeros_like(audio)
