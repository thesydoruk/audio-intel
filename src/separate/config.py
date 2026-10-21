from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


def _find_default_config() -> Path:
    candidates = [DEFAULT_CONFIG_PATH, Path.cwd() / "configs" / "default.yaml"]
    for path in candidates:
        if path.exists():
            return path
    return DEFAULT_CONFIG_PATH


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg_path = Path(path) if path else _find_default_config()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def ensure_cache_dir(cfg: dict[str, Any]) -> Path:
    path = Path(cfg.get("cache", {}).get("dir", "./cache"))
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("diarization", "embeddings", "overlap", "tse", "enhancement", "sr", "audio"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path
