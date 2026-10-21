from __future__ import annotations

import logging

import numpy as np
from audio_intel.separate.audio.resample import resample_audio
from audio_intel.separate.models._speechbrain import speechbrain_fetch_kwargs
from audio_intel.separate.models.base import (
    SpeakerEmbedder,
    TargetSpeakerExtractor,
    cosine_similarity,
    normalize_audio,
)
from audio_intel.separate.models.tse.adaptive_wiener import AdaptiveWienerTSE

log = logging.getLogger(__name__)


class SepFormerTSE(TargetSpeakerExtractor):
    """Target speaker extraction for overlap regions via blind separation + speaker matching.

    Overlap regions are short and local, so a fixed-N blind separator is a sound
    choice *there* even though it is the wrong tool for the whole recording. The
    region is split into N sources with SpeechBrain SepFormer, every source is
    embedded with the same speaker encoder that built the global profiles, and
    the source closest to the target profile is returned. Long regions are
    processed in overlapping windows with per-window source selection, so
    permutation across windows cannot swap speakers.

    Falls back to :class:`AdaptiveWienerTSE` when SpeechBrain or the embedder is
    unavailable, so the pipeline always produces output.
    """

    name = "sepformer"
    sample_rate = 16000

    def __init__(
        self,
        cfg: dict | None = None,
        *,
        embedder: SpeakerEmbedder | None = None,
        device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        cfg = cfg or {}
        self.cfg = cfg
        self.model_id = str(cfg.get("sepformer_model", "speechbrain/sepformer-whamr16k"))
        self.model_sample_rate = int(cfg.get("sepformer_sample_rate", 16000))
        self.sample_rate = int(cfg.get("sample_rate", 16000))
        self.window_s = float(cfg.get("window_seconds", 8.0))
        self.window_overlap_s = float(cfg.get("window_overlap_seconds", 1.0))
        self.min_similarity = float(cfg.get("min_source_similarity", -1.0))
        self.savedir = str(cfg.get("savedir", "./cache/models/sepformer"))
        self.embedder = embedder
        self._model = None
        self._fallback = AdaptiveWienerTSE(cfg=cfg, device="cpu", dtype="float32")
        self.load_error: str | None = None

    # ----------------------------------------------------------------- lifecycle
    def load(self) -> None:
        self._fallback.load()
        try:
            import torch
            from speechbrain.inference.separation import SepformerSeparation

            dev = (
                self.device
                if (self.device.startswith("cuda") and torch.cuda.is_available())
                else "cpu"
            )
            self._model = SepformerSeparation.from_hparams(
                source=self.model_id,
                savedir=self.savedir,
                run_opts={"device": dev},
                **speechbrain_fetch_kwargs(),
            )
            self._model.eval()
            self.device = dev
        except Exception as exc:  # pragma: no cover - depends on optional deps / network
            self._model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "SepFormer TSE unavailable (%s); falling back to adaptive Wiener TSE for overlaps.",
                self.load_error,
            )
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._fallback.unload()
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ extract
    def extract(
        self,
        mixture: np.ndarray,
        sample_rate: int,
        *,
        target_embedding: np.ndarray | None = None,
        reference_audio: np.ndarray | None = None,
        reference_sample_rate: int | None = None,
    ) -> np.ndarray:
        self.ensure_loaded()
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate:
            mix = resample_audio(mix, sample_rate, self.sample_rate)

        can_match = (
            self._model is not None
            and self.embedder is not None
            and (target_embedding is not None or reference_audio is not None)
        )
        if not can_match:
            return self._fallback.extract(
                mix,
                self.sample_rate,
                target_embedding=target_embedding,
                reference_audio=reference_audio,
                reference_sample_rate=reference_sample_rate,
            )

        if target_embedding is None:
            ref = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
            ref_sr = int(reference_sample_rate or sample_rate)
            target_embedding = self.embedder.embed(ref, ref_sr)
        target = np.asarray(target_embedding, dtype=np.float32).reshape(-1)

        work = mix
        if self.sample_rate != self.model_sample_rate:
            work = resample_audio(work, self.sample_rate, self.model_sample_rate)

        out = self._extract_windowed(work, target)

        if self.model_sample_rate != self.sample_rate:
            out = resample_audio(out, self.model_sample_rate, self.sample_rate)
        if len(out) < len(mix):
            out = np.pad(out, (0, len(mix) - len(out)))
        return normalize_audio(out[: len(mix)])

    def _extract_windowed(self, mix: np.ndarray, target: np.ndarray) -> np.ndarray:
        sr = self.model_sample_rate
        win = max(int(self.window_s * sr), sr)
        hop_overlap = int(self.window_overlap_s * sr)
        n = len(mix)
        if n <= win + hop_overlap:
            return self._separate_and_pick(mix, target)

        out = np.zeros(n, dtype=np.float32)
        norm = np.zeros(n, dtype=np.float32)
        step = win - hop_overlap
        start = 0
        while start < n:
            end = min(n, start + win)
            piece = mix[start:end]
            picked = self._separate_and_pick(piece, target)
            m = len(picked)
            window = np.ones(m, dtype=np.float32)
            fade = min(hop_overlap, m // 2)
            if fade > 1:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                if start > 0:
                    window[:fade] = ramp
                if end < n:
                    window[-fade:] = ramp[::-1]
            out[start : start + m] += picked * window
            norm[start : start + m] += window
            if end >= n:
                break
            start += step
        norm[norm == 0] = 1.0
        return out / norm

    def _separate_and_pick(self, piece: np.ndarray, target: np.ndarray) -> np.ndarray:
        import torch

        assert self._model is not None and self.embedder is not None
        peak = float(np.max(np.abs(piece)) + 1e-8)
        scale = 0.9 / peak if peak > 0.9 else 1.0
        x = torch.from_numpy(np.ascontiguousarray(piece * scale)).float().unsqueeze(0)
        with torch.inference_mode():
            est = self._model.separate_batch(x.to(self.device))  # (1, T, n_src)
        sources = est[0].detach().cpu().numpy().astype(np.float32).T  # (n_src, T)

        best_idx = 0
        best_sim = -2.0
        for i, src in enumerate(sources):
            if not np.isfinite(src).all() or float(np.max(np.abs(src))) < 1e-5:
                continue
            emb = self.embedder.embed(src, self.model_sample_rate)
            sim = cosine_similarity(emb, target)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        picked = sources[best_idx] / scale
        if best_sim < self.min_similarity:
            log.debug(
                "SepFormer: best source similarity %.3f below floor; keeping anyway", best_sim
            )
        m = min(len(picked), len(piece))
        out = np.zeros(len(piece), dtype=np.float32)
        out[:m] = picked[:m]
        return out
