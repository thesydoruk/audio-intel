"""Unit tests for MediaWorkspace preparation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.audio.workspace import MediaWorkspace  # noqa: E402


class MediaWorkspacePrepareTest(unittest.TestCase):
    def test_prepare_converts_whisper_and_aed_in_parallel(self) -> None:
        with (
            mock.patch("audio_intel.audio.workspace.probe_media_duration", return_value=10.0),
            mock.patch("audio_intel.audio.workspace.convert_to_wav") as convert,
            mock.patch("audio_intel.audio.workspace.tempfile.mkdtemp", return_value="/tmp/ws"),
            mock.patch.object(Path, "exists", return_value=True),
        ):
            workspace = MediaWorkspace.prepare("clip.mp3", need_aed=True, aed_sample_rate=32000)

        self.assertEqual(convert.call_count, 2)
        sample_rates = {call.kwargs["sample_rate"] for call in convert.call_args_list}
        self.assertEqual(sample_rates, {16000, 32000})
        self.assertIsNotNone(workspace.aed_32k)
        workspace.close()

    def test_prepare_converts_only_whisper_when_aed_not_needed(self) -> None:
        with (
            mock.patch("audio_intel.audio.workspace.probe_media_duration", return_value=5.0),
            mock.patch("audio_intel.audio.workspace.convert_to_wav") as convert,
            mock.patch("audio_intel.audio.workspace.tempfile.mkdtemp", return_value="/tmp/ws"),
            mock.patch.object(Path, "exists", return_value=True),
        ):
            workspace = MediaWorkspace.prepare("clip.mp3", need_aed=False)

        convert.assert_called_once()
        self.assertEqual(convert.call_args.kwargs["sample_rate"], 16000)
        self.assertIsNone(workspace.aed_32k)
        workspace.close()


if __name__ == "__main__":
    unittest.main()
