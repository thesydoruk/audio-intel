# Contributing to audio-intel

Thanks for helping improve the project. This document is the short path from a
clone to a reviewable pull request.

## Development setup

Follow the [Quick start](../README.md#quick-start) in the README, then install
dev tools:

```bash
# Linux / macOS
./scripts/setup.sh

# Windows
.\scripts\setup.ps1
```

`setup` creates `.venv`, installs `.[all,dev]`, copies `.env.example` → `.env`,
and registers the pre-commit hook.

Manual equivalent:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[all,dev]"
pre-commit install
cp .env.example .env
```

You need **Python 3.10+** and **ffmpeg** / **ffprobe** on `PATH`. A GPU is not
required for unit tests.

## Checks

```bash
python -m pytest tests/
pre-commit run --all-files
```

Hooks on each commit:

- **ruff** — lint (`--fix`) and format for `*.py`
- **mdformat** — format `*.md`

Unit tests mock model loads and do not download Whisper, pyannote, or PANNs
weights. Do not commit `.env`, tokens, audio fixtures, or cache directories
under `data/`.

## Pull requests

1. Open an issue first for larger design changes.
2. Keep the diff focused on one problem.
3. Match the existing style (Ruff, 100-character line length, type hints).
4. Add or update tests when behavior changes.
5. Update the README when you change the API, env vars, or extras.

By submitting a contribution you agree that it is licensed under the
[MIT License](../LICENSE), copyright Valerii Sydoruk unless you state otherwise
in the pull request.
