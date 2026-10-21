"""The ``/health`` body: liveness plus what the process holds on the GPU.

Besides the flags clients already read, the body tells an outside scheduler
what this instance is doing: which model groups are resident and for how long
they have been idle, how many requests are in flight or queued, how many GPU
tokens are taken, and how full the card is. The GPU figures come from the
driver (``mem_get_info``), so they count every process on the device, not only
this one; ``torch_reserved_mb`` is this process's share.
"""

from __future__ import annotations

from typing import Any, Protocol


class _Observable(Protocol):
    def snapshot(self) -> dict[str, Any]: ...


def gpu_snapshot() -> dict[str, Any] | None:
    """Device-wide VRAM use and this process's torch pool; ``None`` without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return {
            "device": torch.cuda.get_device_name(0),
            "vram_total_mb": int(total / 2**20),
            "vram_used_mb": int((total - free) / 2**20),
            "torch_reserved_mb": int(torch.cuda.memory_reserved(0) / 2**20),
            "torch_allocated_mb": int(torch.cuda.memory_allocated(0) / 2**20),
        }
    except Exception:  # noqa: BLE001 — health must answer whatever the driver does
        return None


def build_health_body(
    *,
    version: str,
    cfg: Any,
    service: _Observable | None,
    gpu: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "ok",
        "version": version,
        "model": cfg.model,
        "diarization_enabled": cfg.diarization_enabled,
        "speaker_embedding_model": (
            cfg.diarization_embedding_model if cfg.diarization_enabled else None
        ),
        "alignment_enabled": cfg.alignment_enabled,
        "gpu": gpu,
    }
    if service is not None:
        body.update(service.snapshot())
    return body
