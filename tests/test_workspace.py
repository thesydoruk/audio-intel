"""Unit tests for MediaWorkspace preparation."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_intel.audio.workspace import (  # noqa: E402
    MediaWorkspace,
    convert_timeout_s,
    convert_to_wavs,
)


class MediaWorkspacePrepareTest(unittest.TestCase):
    def test_prepare_decodes_whisper_and_aed_in_one_ffmpeg_run(self) -> None:
        with (
            mock.patch("audio_intel.audio.workspace.probe_media_duration", return_value=10.0),
            mock.patch("audio_intel.audio.workspace.convert_to_wavs") as convert,
            mock.patch("audio_intel.audio.workspace.wav_duration_s", return_value=9.98),
            mock.patch("audio_intel.audio.workspace.tempfile.mkdtemp", return_value="/tmp/ws"),
            mock.patch.object(Path, "exists", return_value=True),
        ):
            workspace = MediaWorkspace.prepare("clip.mp3", need_aed=True, aed_sample_rate=32000)

        convert.assert_called_once()
        outputs = convert.call_args.args[1]
        self.assertEqual([rate for _, rate in outputs], [16000, 32000])
        self.assertEqual(convert.call_args.kwargs["timeout_s"], convert_timeout_s(10.0))
        self.assertIsNotNone(workspace.aed_32k)
        # The decoded WAV, not the container probe, defines the timeline.
        self.assertEqual(workspace.duration_s, 9.98)
        workspace.close()

    def test_prepare_converts_only_whisper_when_aed_not_needed(self) -> None:
        with (
            mock.patch("audio_intel.audio.workspace.probe_media_duration", return_value=5.0),
            mock.patch("audio_intel.audio.workspace.convert_to_wavs") as convert,
            mock.patch("audio_intel.audio.workspace.wav_duration_s", return_value=5.0),
            mock.patch("audio_intel.audio.workspace.tempfile.mkdtemp", return_value="/tmp/ws"),
            mock.patch.object(Path, "exists", return_value=True),
        ):
            workspace = MediaWorkspace.prepare("clip.mp3", need_aed=False)

        convert.assert_called_once()
        self.assertEqual([rate for _, rate in convert.call_args.args[1]], [16000])
        self.assertIsNone(workspace.aed_32k)
        workspace.close()

    def test_prepare_survives_a_source_without_container_duration(self) -> None:
        with (
            mock.patch(
                "audio_intel.audio.workspace.probe_media_duration",
                side_effect=ValueError("N/A"),
            ),
            mock.patch("audio_intel.audio.workspace.convert_to_wavs") as convert,
            mock.patch("audio_intel.audio.workspace.wav_duration_s", return_value=3.0),
            mock.patch("audio_intel.audio.workspace.tempfile.mkdtemp", return_value="/tmp/ws"),
            mock.patch.object(Path, "exists", return_value=True),
        ):
            workspace = MediaWorkspace.prepare("stream.ts")

        self.assertEqual(convert.call_args.kwargs["timeout_s"], convert_timeout_s(0.0))
        self.assertEqual(workspace.duration_s, 3.0)

    def test_prepare_removes_workspace_when_decode_fails(self) -> None:
        with (
            mock.patch("audio_intel.audio.workspace.probe_media_duration", return_value=5.0),
            mock.patch(
                "audio_intel.audio.workspace.convert_to_wavs",
                side_effect=RuntimeError("ffmpeg failed"),
            ),
            mock.patch("audio_intel.audio.workspace.tempfile.mkdtemp", return_value="/tmp/ws"),
            mock.patch("audio_intel.audio.workspace.shutil.rmtree") as rmtree,
            mock.patch.object(Path, "exists", return_value=True),
        ):
            with self.assertRaises(RuntimeError):
                MediaWorkspace.prepare("clip.mp3")

        rmtree.assert_called_once_with(Path("/tmp/ws"), ignore_errors=True)


class ConvertToWavsTest(unittest.TestCase):
    def test_timeout_turns_into_runtime_error(self) -> None:
        with mock.patch(
            "audio_intel.audio.workspace.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                convert_to_wavs("clip.mp3", [("/tmp/ws/out.wav", 16000)], timeout_s=300.0)
        self.assertIn("300s", str(ctx.exception))

    def test_one_ffmpeg_invocation_writes_every_output(self) -> None:
        with (
            mock.patch("audio_intel.audio.workspace.Path.mkdir"),
            mock.patch("audio_intel.audio.workspace.subprocess.run") as run,
        ):
            run.return_value = mock.MagicMock(returncode=0, stderr="")
            convert_to_wavs(
                "clip.mp4", [("/tmp/ws/a.wav", 16000), ("/tmp/ws/b.wav", 32000)], timeout_s=42.0
            )

        run.assert_called_once()
        cmd = run.call_args.args[0]
        self.assertEqual(cmd.count("-i"), 1)
        self.assertEqual(cmd.count("-ar"), 2)
        self.assertEqual(cmd.count("-vn"), 2)
        self.assertEqual(run.call_args.kwargs["timeout"], 42.0)

    def test_convert_timeout_scales_with_duration(self) -> None:
        self.assertEqual(convert_timeout_s(10.0), 300.0)
        self.assertEqual(convert_timeout_s(3600.0), 7200.0)


if __name__ == "__main__":
    unittest.main()
