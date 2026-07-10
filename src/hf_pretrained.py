"""Hugging Face auth kwargs compatible with pyannote 3.x and newer hub APIs."""

from __future__ import annotations

import inspect
from typing import Any


def pretrained_auth_kwargs(token: str, from_pretrained: Any) -> dict[str, str]:
    """Return ``token`` or ``use_auth_token`` depending on the callable signature."""
    params = inspect.signature(from_pretrained).parameters
    if "token" in params:
        return {"token": token}
    if "use_auth_token" in params:
        return {"use_auth_token": token}
    return {}
