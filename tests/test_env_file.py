"""Unit tests for .env loading."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.env_file import load_env_file  # noqa: E402


class EnvFileTest(unittest.TestCase):
    def test_load_env_file_does_not_override_existing_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("ASR_MODEL=from-dotenv\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"ASR_MODEL": "from-shell"}, clear=False):
                with mock.patch.dict(os.environ, {"ENV_FILE": str(env_path)}, clear=False):
                    loaded = load_env_file()
                    self.assertEqual(loaded, env_path)
                    self.assertEqual(os.environ["ASR_MODEL"], "from-shell")

    def test_load_env_file_populates_missing_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("ASR_BEAM_SIZE=7\n", encoding="utf-8")

            env = {key: value for key, value in os.environ.items() if key != "ASR_BEAM_SIZE"}
            with mock.patch.dict(os.environ, {**env, "ENV_FILE": str(env_path)}, clear=True):
                loaded = load_env_file()
                self.assertEqual(loaded, env_path)
                self.assertEqual(os.environ.get("ASR_BEAM_SIZE"), "7")


if __name__ == "__main__":
    unittest.main()
