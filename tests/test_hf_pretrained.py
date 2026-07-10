"""Tests for Hugging Face auth kwarg selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fake_from_pretrained(*_args, **kwargs):
    return kwargs


class TestPretrainedAuthKwargs(unittest.TestCase):
    def test_prefers_token_when_supported(self):
        from audio_intel.hf_pretrained import pretrained_auth_kwargs

        def from_pretrained(_model, *, token: str):
            return token

        self.assertEqual(
            pretrained_auth_kwargs("hf_test", from_pretrained),
            {"token": "hf_test"},
        )

    def test_falls_back_to_use_auth_token(self):
        from audio_intel.hf_pretrained import pretrained_auth_kwargs

        def from_pretrained(_model, *, use_auth_token: str):
            return use_auth_token

        self.assertEqual(
            pretrained_auth_kwargs("hf_test", from_pretrained),
            {"use_auth_token": "hf_test"},
        )


if __name__ == "__main__":
    unittest.main()
