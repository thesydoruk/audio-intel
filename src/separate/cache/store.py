from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class CacheStore:
    """Content-addressed on-disk cache for expensive pipeline stages.

    Keys are derived from the source audio hash plus stage parameters, so a
    resumed run with the same input and settings reuses previous results while
    any change in input or model configuration produces fresh entries.
    """

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ keys
    @staticmethod
    def file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
        """SHA-1 over file bytes plus size; stable across runs, cheap enough for long WAVs."""
        path = Path(path)
        h = hashlib.sha1()
        h.update(str(path.stat().st_size).encode())
        with path.open("rb") as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def key(*parts: Any) -> str:
        payload = json.dumps([_jsonable(p) for p in parts], sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _path(self, stage: str, key: str, suffix: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}{suffix}"

    # ------------------------------------------------------------------ json
    def has(self, stage: str, key: str, suffix: str = ".json") -> bool:
        return self.enabled and self._path(stage, key, suffix).exists()

    def load_json(self, stage: str, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        p = self._path(stage, key, ".json")
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def save_json(self, stage: str, key: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        p = self._path(stage, key, ".json")
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, default=_jsonable), encoding="utf-8")
        tmp.replace(p)

    # ---------------------------------------------------------------- arrays
    def load_array(self, stage: str, key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        p = self._path(stage, key, ".npy")
        if not p.exists():
            return None
        try:
            return np.load(p, allow_pickle=False)
        except (OSError, ValueError):
            return None

    def save_array(self, stage: str, key: str, array: np.ndarray) -> None:
        if not self.enabled:
            return
        p = self._path(stage, key, ".npy")
        tmp = p.with_suffix(".npy.tmp")
        with tmp.open("wb") as f:
            np.save(f, np.ascontiguousarray(array), allow_pickle=False)
        tmp.replace(p)

    # ----------------------------------------------------------------- files
    def file_path(self, stage: str, key: str, suffix: str) -> Path:
        """Path for a stage artifact stored as a regular file (e.g. a WAV stem)."""
        return self._path(stage, key, suffix)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
