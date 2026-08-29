#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  "$ROOT/scripts/setup.sh"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m audio_intel.server
