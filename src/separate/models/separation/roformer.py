from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from audio_intel.separate.audio.bed import gate_bed
from audio_intel.separate.models.base import SourceSeparator

log = logging.getLogger(__name__)

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# The stems are written as float32 and the input is halved first: audio-separator scales
# any stem whose peak exceeds its normalization threshold (1.0 at most), which would break
# "instrumental == source - vocals"; at half scale no stem can get there.
_INPUT_SCALE = 0.5


def default_model_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Where the checkpoints live: ``AUDIO_SEPARATOR_MODEL_DIR``, else the models volume."""
    env = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
    if env:
        return Path(env)
    root = os.environ.get("MODEL_CACHE_DIR")
    if root:
        return Path(root) / "audio-separator"
    return Path(cache_dir or "./cache") / "models" / "audio-separator"


class _ExpectedNoise(logging.Filter):
    """audio-separator warns about two things that are by design here.

    onnxruntime has no CUDA provider in the image (RoFormer runs on torch, not ONNX) and
    the stems are written with soundfile on purpose (the only writer that keeps float).
    """

    _texts = ("CUDAExecutionProvider not available", "Using soundfile for writing")

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(t in record.getMessage() for t in self._texts)


def _quiet_audio_separator_logs() -> None:
    for name in (
        "audio_separator.separator.separator",
        "audio_separator.separator.common_separator",
    ):
        logger = logging.getLogger(name)
        if not any(isinstance(f, _ExpectedNoise) for f in logger.filters):
            logger.addFilter(_ExpectedNoise())


class RoformerSeparator(SourceSeparator):
    """Speech / background separation with BS-RoFormer through ``audio-separator``.

    The model's ``Vocals`` stem is the speech stem every later stage works on. The
    background is not the model's ``Instrumental`` as such but :func:`gate_bed` of it:
    the original mix wherever the vocals stem is quiet, the instrumental only under
    speech, with ramps between. Both come back at the input sample rate (the model runs
    at whatever rate the input has; 48 kHz measured at least as clean as a 44.1 kHz round
    trip), in the input channel layout and length.

    ``audio-separator`` picks the GPU itself (CUDA when available, the first visible
    card); ``device`` is kept for the interface and only decides whether to warn.
    """

    name = "roformer"

    def __init__(
        self,
        model_filename: str = DEFAULT_MODEL,
        *,
        segment_size: int = 256,
        overlap: int | None = None,
        batch_size: int | None = None,
        bed: dict[str, Any] | None = None,
        model_dir: str | os.PathLike[str] | None = None,
        device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_filename = model_filename
        self.segment_size = int(segment_size)
        self.overlap = None if overlap is None else int(overlap)
        self.batch_size = None if batch_size is None else int(batch_size)
        bed = dict(bed or {})
        self.bed_threshold_db = float(bed.get("threshold_db", -45.0))
        self.bed_hold_ms = float(bed.get("hold_ms", 150.0))
        self.bed_fade_ms = float(bed.get("fade_ms", 150.0))
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self._separator: Any = None
        self._separator_sample_rate: int | None = None

    # ------------------------------------------------------------------ model
    def _make_separator(self, output_dir: str, sample_rate: int) -> Any:
        from audio_separator.separator import Separator

        _quiet_audio_separator_logs()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return Separator(
            log_level=logging.WARNING,
            model_file_dir=str(self.model_dir),
            output_dir=output_dir,
            output_format="WAV",
            normalization_threshold=1.0,
            amplification_threshold=0.0,
            sample_rate=int(sample_rate),
            use_soundfile=True,
            mdxc_params={
                "segment_size": self.segment_size,
                "override_model_segment_size": False,
                "batch_size": self.batch_size,
                "overlap": self.overlap,
                "pitch_shift": 0,
            },
        )

    def load(self) -> None:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested for RoFormer but unavailable; running on CPU (slow).")
        # audio-separator fixes the sample rate per instance; the instance is rebuilt on the
        # first input with another rate. Loading here downloads and loads the checkpoint.
        sep = self._make_separator(tempfile.gettempdir(), 48000)
        sep.load_model(model_filename=self.model_filename)
        self._separator = sep
        self._separator_sample_rate = 48000
        self._loaded = True

    def unload(self) -> None:
        self._separator = None
        self._separator_sample_rate = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def _separator_for(self, output_dir: str, sample_rate: int) -> Any:
        if self._separator is None or self._separator_sample_rate != sample_rate:
            sep = self._make_separator(output_dir, sample_rate)
            sep.load_model(model_filename=self.model_filename)
            self._separator = sep
            self._separator_sample_rate = sample_rate
        else:
            self._separator.output_dir = output_dir
            if getattr(self._separator, "model_instance", None) is not None:
                self._separator.model_instance.output_dir = output_dir
        return self._separator

    # --------------------------------------------------------------- separate
    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        self.ensure_loaded()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        n_in, n_ch = audio.shape

        # The model is stereo: duplicate mono, drop channels beyond two.
        work = audio
        if n_ch == 1:
            work = np.repeat(work, 2, axis=1)
        elif n_ch > 2:
            work = work[:, :2]

        with tempfile.TemporaryDirectory(prefix="roformer-") as tmp:
            src_path = os.path.join(tmp, "input.wav")
            sf.write(src_path, work * _INPUT_SCALE, sample_rate, subtype="FLOAT")
            sep = self._separator_for(tmp, sample_rate)
            sep.separate(src_path, {"Vocals": "vocals", "Instrumental": "instrumental"})
            vocals = self._read_stem(os.path.join(tmp, "vocals.wav"), sample_rate)
            instrumental = self._read_stem(os.path.join(tmp, "instrumental.wav"), sample_rate)

        out_ch = min(n_ch, 2)
        vocals = self._to_input_layout(vocals, out_ch, n_in)
        instrumental = self._to_input_layout(instrumental, out_ch, n_in)
        source = np.ascontiguousarray(audio[:, :out_ch], dtype=np.float32)
        background = gate_bed(
            source,
            instrumental,
            vocals,
            sample_rate,
            threshold_db=self.bed_threshold_db,
            hold_ms=self.bed_hold_ms,
            fade_ms=self.bed_fade_ms,
        )
        return vocals, background

    @staticmethod
    def _read_stem(path: str, sample_rate: int) -> np.ndarray:
        stem, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != sample_rate:
            raise RuntimeError(f"audio-separator wrote {path} at {sr} Hz, expected {sample_rate}")
        return stem / _INPUT_SCALE

    @staticmethod
    def _to_input_layout(stem: np.ndarray, n_ch: int, n_in: int) -> np.ndarray:
        if n_ch == 1:
            stem = stem.mean(axis=1, keepdims=True)
        elif stem.shape[1] != n_ch:
            stem = stem[:, :n_ch]
        if len(stem) < n_in:
            stem = np.pad(stem, ((0, n_in - len(stem)), (0, 0)))
        return np.ascontiguousarray(stem[:n_in], dtype=np.float32)
