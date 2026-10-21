from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from audio_intel.separate.embeddings.profile import SpeakerProfileBank
from audio_intel.separate.models.base import cosine_similarity
from audio_intel.separate.types import DiarizationSegment, EmbeddingSample, TimeSpan
from scipy.optimize import linear_sum_assignment


@dataclass
class LocalSpeakerStats:
    local_id: str
    embedding: np.ndarray
    quality: float
    duration: float
    span: TimeSpan
    overlap_span_ratio: float = 0.0


@dataclass
class Assignment:
    local_id: str
    global_id: str
    score: float
    embedding_similarity: float
    is_new: bool = False


class GlobalSpeakerTracker:
    """Resolves local diarization IDs to stable global speaker identities across chunks."""

    def __init__(self, bank: SpeakerProfileBank, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.bank = bank
        self.w_emb = float(cfg.get("weight_embedding", 0.55))
        self.w_overlap = float(cfg.get("weight_overlap_consistency", 0.15))
        self.w_temporal = float(cfg.get("weight_temporal_continuity", 0.20))
        self.w_profile = float(cfg.get("weight_profile_confidence", 0.05))
        self.w_quality = float(cfg.get("weight_segment_quality", 0.05))
        self.new_thr = float(cfg.get("new_speaker_threshold", 0.42))
        self.min_emb = float(cfg.get("min_embedding_similarity", 0.28))
        self._prev_assignments: dict[str, str] = {}
        self._prev_chunk_span: TimeSpan | None = None

    def assign_chunk(
        self,
        locals_: list[LocalSpeakerStats],
        *,
        chunk_span: TimeSpan,
        overlap_with_prev: TimeSpan | None,
        create_profiles: bool = True,
    ) -> list[Assignment]:
        if not locals_:
            return []

        global_ids = self.bank.speakers()
        if not global_ids:
            out: list[Assignment] = []
            for loc in locals_:
                sample = EmbeddingSample(
                    speaker_id="",
                    vector=loc.embedding,
                    span=loc.span,
                    quality=loc.quality,
                )
                if create_profiles:
                    gid = self.bank.create(sample)
                else:
                    gid = f"speaker_{len(out):02d}"
                out.append(
                    Assignment(
                        local_id=loc.local_id,
                        global_id=gid,
                        score=1.0,
                        embedding_similarity=1.0,
                        is_new=True,
                    )
                )
            self._prev_assignments = {a.local_id: a.global_id for a in out}
            self._prev_chunk_span = chunk_span
            return out

        # Similarity matrix: rows=local, cols=global
        score = np.full((len(locals_), len(global_ids)), -1e6, dtype=np.float64)
        emb_sims = np.zeros_like(score)
        for i, loc in enumerate(locals_):
            for j, gid in enumerate(global_ids):
                emb = self.bank.centroid(gid)
                if emb is None:
                    continue
                sim = cosine_similarity(loc.embedding, emb)
                emb_sims[i, j] = sim
                temporal = self._temporal_score(loc, gid, overlap_with_prev)
                overlap_c = 1.0 - min(1.0, loc.overlap_span_ratio)
                profile_c = self.bank.profiles[gid].confidence
                combined = (
                    self.w_emb * _sim_to_unit(sim)
                    + self.w_temporal * temporal
                    + self.w_overlap * overlap_c
                    + self.w_profile * profile_c
                    + self.w_quality * loc.quality
                )
                if sim < self.min_emb and temporal < 0.5:
                    combined *= 0.5
                score[i, j] = combined

        # Hungarian on cost = -score
        row_ind, col_ind = linear_sum_assignment(-score)
        used_globals: set[str] = set()
        assignments: list[Assignment] = []
        assigned_locals: set[str] = set()

        for r, c in zip(row_ind, col_ind, strict=True):
            sc = float(score[r, c])
            loc = locals_[r]
            gid = global_ids[c]
            if sc < self.new_thr:
                continue
            used_globals.add(gid)
            assigned_locals.add(loc.local_id)
            assignments.append(
                Assignment(
                    local_id=loc.local_id,
                    global_id=gid,
                    score=sc,
                    embedding_similarity=float(emb_sims[r, c]),
                    is_new=False,
                )
            )
            if create_profiles:
                self.bank.maybe_update(
                    gid,
                    EmbeddingSample(
                        speaker_id=gid,
                        vector=loc.embedding,
                        span=loc.span,
                        quality=loc.quality,
                    ),
                )

        # Unassigned locals become new speakers
        for loc in locals_:
            if loc.local_id in assigned_locals:
                continue
            sample = EmbeddingSample(
                speaker_id="",
                vector=loc.embedding,
                span=loc.span,
                quality=loc.quality,
            )
            if create_profiles:
                gid = self.bank.create(sample)
            else:
                gid = f"speaker_{self.bank._next_id:02d}"
                self.bank._next_id += 1
            assignments.append(
                Assignment(
                    local_id=loc.local_id,
                    global_id=gid,
                    score=0.0,
                    embedding_similarity=0.0,
                    is_new=True,
                )
            )

        self._prev_assignments = {a.local_id: a.global_id for a in assignments}
        self._prev_chunk_span = chunk_span
        return assignments

    def _temporal_score(
        self,
        loc: LocalSpeakerStats,
        global_id: str,
        overlap_with_prev: TimeSpan | None,
    ) -> float:
        if overlap_with_prev is None or self._prev_chunk_span is None:
            return 0.5  # neutral when speaker may return after long silence
        # If this local speaker is active in the overlap zone and a previous
        # local mapped to global_id, reward continuity.
        ov = loc.span.overlap(overlap_with_prev)
        if ov <= 0:
            return 0.5
        # Prefer globals that were recently active
        if global_id in self._prev_assignments.values():
            return 0.85
        return 0.35


def _sim_to_unit(sim: float) -> float:
    # Cosine in [-1,1] -> [0,1]
    return float(np.clip((sim + 1.0) * 0.5, 0.0, 1.0))


def segments_for_local(
    segments: list[DiarizationSegment], local_id: str
) -> list[DiarizationSegment]:
    return [s for s in segments if s.local_speaker == local_id]
