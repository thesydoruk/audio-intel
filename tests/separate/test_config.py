from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from audio_intel.separate.config import DEFAULT_CONFIG_PATH, load_config


def test_defaults_come_from_the_packaged_yaml():
    cfg = load_config()
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as f:
        assert cfg == yaml.safe_load(f)


def test_user_file_is_merged_over_the_defaults(tmp_path: Path):
    user = tmp_path / "my.yaml"
    user.write_text("embeddings:\n  backend: wespeaker\nstitching:\n  crossfade_ms: 10\n")

    cfg = load_config(user)
    defaults = load_config()

    assert cfg["embeddings"]["backend"] == "wespeaker"
    assert cfg["stitching"]["crossfade_ms"] == 10
    # Siblings of the changed keys and untouched sections keep their defaults.
    assert cfg["embeddings"]["model_id"] == defaults["embeddings"]["model_id"]
    assert cfg["stitching"]["segment_pad_ms"] == defaults["stitching"]["segment_pad_ms"]
    assert cfg["separation"] == defaults["separation"]


def test_overrides_win_over_the_user_file(tmp_path: Path):
    user = tmp_path / "my.yaml"
    user.write_text("embeddings:\n  backend: wespeaker\n")

    cfg = load_config(user, overrides={"embeddings": {"backend": "stub"}})

    assert cfg["embeddings"]["backend"] == "stub"


def test_empty_user_file_changes_nothing(tmp_path: Path):
    user = tmp_path / "empty.yaml"
    user.write_text("")

    assert load_config(user) == load_config()


def test_user_file_must_be_a_mapping(tmp_path: Path):
    user = tmp_path / "list.yaml"
    user.write_text("- a\n- b\n")

    with pytest.raises(ValueError):
        load_config(user)
