"""Speaker embedding extraction via pyannote (optional, with diarization)."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from audio_intel.config import Config

log = logging.getLogger("audio-intel")


class SpeakerEmbedder:
    """Pre-loaded pyannote embedding models shared across concurrent requests."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._per_request_workers = cfg.diarization_embed_workers
        self._count = self._per_request_workers * cfg.max_concurrent_requests
        self._inferences: list[object | None] = [None] * self._count
        self._locks = [threading.Lock() for _ in range(self._count)]
        self._init_lock = threading.Lock()

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
                "Loading pyannote embedding model %d/%d (device=%s)",
                index + 1,
                self._count,
                self.cfg.diarization_device,
            )
            model = Model.from_pretrained(
                "pyannote/embedding",
                **pretrained_auth_kwargs(token, Model.from_pretrained),
            )
            inference = Inference(model, window="whole")
            inference.to(torch.device(self.cfg.diarization_device))
            self._inferences[index] = inference
            log.info("Speaker embedding model %d/%d loaded", index + 1, self._count)

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

    @staticmethod
    def _normalize_embedding(vector) -> list[float] | None:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            return None
        normalized = (arr / norm).tolist()
        if len(normalized) != 512:
            return None
        return [round(float(v), 6) for v in normalized]

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
                "Unexpected embedding size for %s (model %d/%d, expected 512)",
                sid,
                model_idx + 1,
                self._count,
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
        from audio_intel.audio.decode import load_audio

        if not intervals:
            return {}

        started = time.perf_counter()
        audio = load_audio(path)
        by_speaker: dict[str, list[tuple[float, float]]] = {}
        for item in intervals:
            sid = str(item["speaker_id"])
            by_speaker.setdefault(sid, []).append((float(item["start"]), float(item["end"])))

        tasks: list[tuple[int, str, np.ndarray]] = []
        for task_idx, (sid, spans) in enumerate(by_speaker.items()):
            clip = self._build_speaker_clip(
                audio,
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
