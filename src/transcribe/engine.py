"""Unified faster-whisper transcription for the HTTP service.

``WhisperEngine`` owns a :class:`~audio_intel.common.model_pool.ModelPool` so
parallel chunk workers each get their own model copy without reloading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from audio_intel.common.model_pool import ModelPool
from audio_intel.config import Config
from audio_intel.transcribe.segments import segment_from_whisper
from audio_intel.types import SpeechSegment
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class WhisperRuntime:
    """Immutable faster-whisper decode settings shared across model copies."""

    model: str
    device: str
    compute_type: str
    download_root: str = ""
    cpu_threads: int = 2
    beam_size: int = 5
    temperatures: tuple[float, ...] = (0.0,)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    hallucination_silence_s: float = 0.0
    word_timestamps: bool = True
    lang_min_prob: float = 0.5
    vad_filter: bool = False

    @classmethod
    def from_server(cls, cfg: Config) -> WhisperRuntime:
        """Build runtime from HTTP service environment config."""
        return cls(
            model=cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            download_root=cfg.download_root,
            cpu_threads=cfg.cpu_threads,
            beam_size=cfg.beam_size,
            temperatures=cfg.temperatures,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            no_speech_threshold=cfg.no_speech_threshold,
            hallucination_silence_s=cfg.hallucination_silence_s,
            lang_min_prob=cfg.lang_min_prob,
        )


@dataclass
class ClipTranscription:
    """Result of transcribing one clip or file (before merging into a transcript)."""

    segments: list[SpeechSegment]
    language: str
    language_probability: float


class WhisperEngine:
    """Shared faster-whisper wrapper used by the HTTP service."""

    def __init__(self, runtime: WhisperRuntime) -> None:
        self.runtime = runtime
        self._pool = ModelPool(self._create_model)

    @staticmethod
    def validate_cuda(device: str) -> None:
        """Fail fast when ``device`` is ``cuda`` but no GPU is available."""
        if device != "cuda":
            return
        try:
            import ctranslate2
        except ImportError as exc:
            raise RuntimeError("ctranslate2 is required for GPU Whisper inference") from exc
        if ctranslate2.get_cuda_device_count() == 0:
            raise RuntimeError("whisper.device is 'cuda' but no CUDA GPU is available")

    def ensure_workers(self, num_workers: int = 1) -> None:
        """Pre-load ``num_workers`` Whisper models (validates CUDA first)."""
        self.validate_cuda(self.runtime.device)
        self._pool.ensure_ready(num_workers)

    @property
    def primary_model(self) -> WhisperModel:
        """Return the first loaded model (for callers that manage lifecycle themselves)."""
        self._pool.ensure_ready(1)
        return self._pool.acquire()

    def _create_model(self) -> WhisperModel:
        kwargs: dict[str, Any] = {
            "device": self.runtime.device,
            "compute_type": self.runtime.compute_type,
            "cpu_threads": self.runtime.cpu_threads,
        }
        if self.runtime.download_root:
            kwargs["download_root"] = self.runtime.download_root
        return WhisperModel(self.runtime.model, **kwargs)

    def transcribe_array(
        self,
        clip: Any,
        *,
        language: str | None,
        time_offset: float = 0.0,
        normalize_text: bool = True,
    ) -> ClipTranscription:
        """Transcribe an in-memory float32 waveform (HTTP chunked path).

        ``time_offset`` shifts segment timestamps when the clip is a slice of a longer file.
        """
        model = self._pool.acquire()
        try:
            return self._decode(
                model,
                clip,
                language=language,
                time_offset=time_offset,
                normalize_text=normalize_text,
            )
        finally:
            self._pool.release(model)

    def _decode(
        self,
        model: WhisperModel,
        clip: Any,
        *,
        language: str | None,
        time_offset: float = 0.0,
        normalize_text: bool = True,
    ) -> ClipTranscription:
        runtime = self.runtime
        seg_iter, info = model.transcribe(
            clip,
            language=language,
            beam_size=runtime.beam_size,
            temperature=runtime.temperatures,
            compression_ratio_threshold=runtime.compression_ratio_threshold,
            log_prob_threshold=runtime.log_prob_threshold,
            no_speech_threshold=runtime.no_speech_threshold,
            vad_filter=False,
            word_timestamps=runtime.word_timestamps,
            hallucination_silence_threshold=(runtime.hallucination_silence_s or None),
            condition_on_previous_text=False,
        )

        trusted = (info.language_probability or 0.0) >= runtime.lang_min_prob
        detected_language = info.language if (info.language and trusted) else "unknown"
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)

        segments: list[SpeechSegment] = []
        for index, seg in enumerate(seg_iter):
            raw_text = seg.text.strip()
            if not raw_text:
                continue

            chunk_lang = detected_language if detected_language != "unknown" else None
            segments.append(
                segment_from_whisper(
                    seg,
                    seg_id=index,
                    time_offset=time_offset,
                    language=chunk_lang,
                    language_probability=language_probability if chunk_lang else None,
                    normalize_text=normalize_text,
                )
            )

        return ClipTranscription(
            segments=segments,
            language=detected_language,
            language_probability=language_probability,
        )
