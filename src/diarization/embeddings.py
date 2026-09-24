"""Speaker embedding extraction via pyannote (optional, with diarization)."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from audio_intel.config import Config

log = logging.getLogger("audio-intel")


def split_model_id(model_id: str) -> tuple[str, str | None]:
    """Split ``owner/repo[/subfolder]`` into a Hugging Face repo id and subfolder."""
    if os.path.exists(model_id):
        return model_id, None  # local checkpoint path
    parts = [part for part in model_id.strip().strip("/").split("/") if part]
    if len(parts) <= 2:
        return "/".join(parts), None
    return "/".join(parts[:2]), "/".join(parts[2:])


class SpeakerEmbedder:
    """Pre-loaded pyannote embedding models shared across concurrent requests."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._per_request_workers = cfg.diarization_embed_workers
        self._count = self._per_request_workers * cfg.max_concurrent_requests
        self._inferences: list[object | None] = [None] * self._count
        self._locks = [threading.Lock() for _ in range(self._count)]
        self._init_lock = threading.Lock()
        self._dimension: int | None = None

    @property
    def model_id(self) -> str:
        """Checkpoint id reported next to every vector; spaces differ per model."""
        return self.cfg.diarization_embedding_model

    @property
    def dimension(self) -> int | None:
        """Vector length of the loaded model (``None`` until the first load)."""
        return self._dimension

    def ensure_ready(self) -> None:
        """Load every pyannote embedding model copy."""
        for index in range(self._count):
            self._load_inference(index)

    def _load_inference(self, index: int) -> None:
        if self._inferences[index] is not None:
            return
        with self._init_lock:
            if self._inferences[index] is not None:
                return
            import torch
            from audio_intel.hf_pretrained import pretrained_auth_kwargs
            from pyannote.audio import Inference, Model

            token = self.cfg.hf_token
            if not token:
                raise ValueError("HF_TOKEN is required when SPEAKERS_ENABLED=1")

            log.info(
                "Loading pyannote embedding model %s %d/%d (device=%s)",
                self.model_id,
                index + 1,
                self._count,
                self.cfg.diarization_device,
            )
            checkpoint, subfolder = split_model_id(self.model_id)
            kwargs = pretrained_auth_kwargs(token, Model.from_pretrained)
            if subfolder:
                kwargs["subfolder"] = subfolder
            model = Model.from_pretrained(checkpoint, **kwargs)
            if model is None:
                raise RuntimeError(
                    f"Could not load {self.model_id}: accept its terms on Hugging Face "
                    "for the account that owns HF_TOKEN"
                )
            dimension = getattr(model, "dimension", None)
            if isinstance(dimension, int) and dimension > 0:
                self._dimension = dimension
            inference = Inference(model, window="whole")
            inference.to(torch.device(self.cfg.diarization_device))
            self._inferences[index] = inference
            log.info(
                "Speaker embedding model %d/%d loaded (%s-d)",
                index + 1,
                self._count,
                self._dimension or "?",
            )

    @staticmethod
    def _build_speaker_clip(
        audio: np.ndarray,
        spans: list[tuple[float, float]],
        *,
        sample_rate: int,
        min_speech_s: float,
        max_clip_s: float,
    ) -> np.ndarray | None:
        """Concatenate up to ``max_clip_s`` of speech for one speaker."""
        total = sum(max(0.0, end - start) for start, end in spans)
        if total < min_speech_s:
            return None

        chunks: list[np.ndarray] = []
        budget = max_clip_s
        for start, end in sorted(spans):
            if budget <= 0:
                break
            take = min(max(0.0, end - start), budget)
            if take <= 0:
                continue
            start_i = int(start * sample_rate)
            end_i = int((start + take) * sample_rate)
            if end_i <= start_i:
                continue
            chunks.append(audio[start_i:end_i])
            budget -= take

        if not chunks:
            return None
        return np.concatenate(chunks)

    def _normalize_embedding(self, vector) -> list[float] | None:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        if self._dimension is None:
            self._dimension = int(arr.size)
        elif arr.size != self._dimension:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            return None
        return [round(float(v), 6) for v in (arr / norm).tolist()]

    def _embed_speaker(
        self,
        model_idx: int,
        sid: str,
        clip: np.ndarray,
        *,
        sample_rate: int,
    ) -> tuple[str, list[float]] | None:
        import torch

        waveform = {
            "waveform": torch.from_numpy(clip).unsqueeze(0),
            "sample_rate": sample_rate,
        }
        try:
            with self._locks[model_idx]:
                self._load_inference(model_idx)
                inference = self._inferences[model_idx]
                assert inference is not None
                vector = inference(waveform)
        except Exception:  # noqa: BLE001
            log.exception(
                "Speaker embedding failed for %s (model %d/%d)",
                sid,
                model_idx + 1,
                self._count,
            )
            return None

        normalized = self._normalize_embedding(vector)
        if normalized is None:
            log.warning(
                "Unusable embedding for %s (model %d/%d, expected %s-d finite vector)",
                sid,
                model_idx + 1,
                self._count,
                self._dimension or "?",
            )
            return None
        return sid, normalized

    def embed_speaker_spans(
        self,
        audio: np.ndarray,
        spans: list[tuple[float, float]],
        *,
        sample_rate: int,
        min_speech_s: float = 0.5,
        max_clip_s: float = 30.0,
    ) -> list[float] | None:
        """Embed one speaker from explicit time spans inside an in-memory waveform."""
        clip = self._build_speaker_clip(
            audio,
            spans,
            sample_rate=sample_rate,
            min_speech_s=min_speech_s,
            max_clip_s=max_clip_s,
        )
        if clip is None:
            return None
        result = self._embed_speaker(0, "link", clip, sample_rate=sample_rate)
        if result is None:
            return None
        return result[1]

    def embed_speaker_spans_from_path(
        self,
        path: str,
        spans: list[tuple[float, float]],
        *,
        sample_rate: int,
        min_speech_s: float = 0.5,
        max_clip_s: float = 30.0,
    ) -> list[float] | None:
        """Embed one speaker by reading only the needed spans from disk."""
        clip = self._build_speaker_clip_from_path(
            path,
            spans,
            sample_rate=sample_rate,
            min_speech_s=min_speech_s,
            max_clip_s=max_clip_s,
        )
        if clip is None:
            return None
        result = self._embed_speaker(0, "link", clip, sample_rate=sample_rate)
        if result is None:
            return None
        return result[1]

    @staticmethod
    def _build_speaker_clip_from_path(
        path: str,
        spans: list[tuple[float, float]],
        *,
        sample_rate: int,
        min_speech_s: float,
        max_clip_s: float,
    ) -> np.ndarray | None:
        """Load at most ``max_clip_s`` of speech for one speaker, never the whole file."""
        from audio_intel.audio.decode import load_audio_spans

        total = sum(max(0.0, end - start) for start, end in spans)
        if total < min_speech_s:
            return None
        clip = load_audio_spans(
            path,
            sorted(spans),
            sample_rate,
            max_duration_s=max_clip_s,
        )
        if clip.size == 0:
            return None
        return clip

    def embed_speakers(
        self,
        path: str,
        intervals: list[dict],
        *,
        sample_rate: int,
        min_speech_s: float = 3.0,
        max_clip_s: float = 30.0,
    ) -> dict[str, list[float]]:
        """Return L2-normalized embedding vectors keyed by ``speaker_id``."""
        if not intervals:
            return {}

        started = time.perf_counter()
        by_speaker: dict[str, list[tuple[float, float]]] = {}
        for item in intervals:
            sid = str(item["speaker_id"])
            by_speaker.setdefault(sid, []).append((float(item["start"]), float(item["end"])))

        tasks: list[tuple[int, str, np.ndarray]] = []
        for task_idx, (sid, spans) in enumerate(by_speaker.items()):
            clip = self._build_speaker_clip_from_path(
                path,
                spans,
                sample_rate=sample_rate,
                min_speech_s=min_speech_s,
                max_clip_s=max_clip_s,
            )
            if clip is not None:
                model_idx = task_idx % self._count
                tasks.append((model_idx, sid, clip))

        if not tasks:
            log.info(
                "Speaker embeddings skipped: no speakers met min speech (%.1fs)",
                min_speech_s,
            )
            return {}

        log.info(
            "Speaker embeddings starting: %d speakers, %d model cop%s",
            len(tasks),
            self._count,
            "y" if self._count == 1 else "ies",
        )

        out: dict[str, list[float]] = {}
        max_workers = min(len(tasks), self._per_request_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._embed_speaker,
                    model_idx,
                    sid,
                    clip,
                    sample_rate=sample_rate,
                )
                for model_idx, sid, clip in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    sid, embedding = result
                    out[sid] = embedding

        log.info(
            "Speaker embeddings finished in %.1fs: %d/%d speakers (%d model cop%s)",
            time.perf_counter() - started,
            len(out),
            len(by_speaker),
            self._count,
            "y" if self._count == 1 else "ies",
        )
        return out
