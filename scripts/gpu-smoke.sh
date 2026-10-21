#!/usr/bin/env bash
# GPU check of a built image on the host it will run on: torch's kernel list and
# CTranslate2's CUDA view, then the service itself on one media file.
#
#   scripts/gpu-smoke.sh <image> <sample media> [model] [diarize 0|1] [align 0|1]
#
# Needs an NVIDIA GPU, the NVIDIA Container Toolkit, and HF_TOKEN in the
# environment when diarize=1 (gated pyannote weights). Models download into
# GPU_SMOKE_CACHE (default ./data/whisper, the same cache the service uses).
set -uo pipefail

image="${1:?usage: scripts/gpu-smoke.sh <image> <sample media> [model] [diarize] [align]}"
sample="${2:?sample media file (speech, ideally with music or noise)}"
model="${3:-large-v3}"
diarize="${4:-0}"
align="${5:-0}"
cache="$(cd "$(dirname "${GPU_SMOKE_CACHE:-./data/whisper}")" 2>/dev/null && pwd)/$(basename "${GPU_SMOKE_CACHE:-./data/whisper}")"
mkdir -p "$cache"
sample="$(cd "$(dirname "$sample")" && pwd)/$(basename "$sample")"
name="audio-intel-gpu-smoke-$$"
port="${GPU_SMOKE_PORT:-18080}"
status=0

echo "=== torch / ctranslate2 on this GPU"
docker run --rm -i --gpus all --entrypoint python "$image" - <<'PY' || status=1
import ctranslate2
import torch

cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("device", torch.cuda.get_device_name(0), arch)
print("kernels", torch.cuda.get_arch_list())
assert arch in torch.cuda.get_arch_list(), f"torch has no {arch} kernels for this card"
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
assert torch.isfinite((a @ a).float()).all(), "fp16 matmul produced non-finite values"
print("ctranslate2", ctranslate2.__version__, "cuda devices", ctranslate2.get_cuda_device_count())
assert ctranslate2.get_cuda_device_count() > 0, "ctranslate2 sees no CUDA device"
print("compute types", sorted(ctranslate2.get_supported_compute_types("cuda")))
PY

echo "=== service: model=$model diarize=$diarize align=$align"
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all -p "$port:8080" \
  -v "$cache:/var/lib/whisper" \
  -e AUDIO_INTEL_PORT=8080 -e MODEL_CACHE_DIR=/var/lib/whisper -e HF_HOME=/var/lib/whisper \
  -e ASR_MODEL="$model" -e ASR_DEVICE=cuda -e ASR_COMPUTE_TYPE=float16 \
  -e SOUND_EVENTS_ENABLED=1 -e SPEAKERS_ENABLED="$diarize" -e WORD_ALIGN_ENABLED="$align" \
  -e MAX_CONCURRENT_REQUESTS=1 -e LOG_LEVEL=INFO -e HF_TOKEN="${HF_TOKEN:-}" \
  "$image" >/dev/null
for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  if [ "$(docker inspect -f '{{.State.Running}}' "$name")" != "true" ]; then
    echo "container exited:"; docker logs "$name" 2>&1 | tail -20; status=1; break
  fi
  sleep 5
done
curl -s "http://127.0.0.1:$port/health"; echo
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  vram: /'

echo "=== transcribe $(basename "$sample")"
form=(-F "file=@$sample" -F "sound_events=true")
[ "$diarize" = "1" ] && form+=(-F "diarize=true")
[ "$align" = "1" ] && form+=(-F "align=true")
out="$(mktemp)"
code="$(curl -s -o "$out" -w '%{http_code}' "${form[@]}" "http://127.0.0.1:$port/v1/audio/transcriptions")"
if [ "$code" != "200" ]; then echo "HTTP $code"; head -c 600 "$out"; echo; status=1; fi
python3 - "$out" <<'PY' || status=1
import json, sys
r = json.load(open(sys.argv[1]))
speech = [s for s in r.get("segments", []) if s.get("kind") == "speech"]
sounds = [s for s in r.get("segments", []) if s.get("kind") == "sound"]
print(f"duration {r.get('duration')}s languages {r.get('languages')} confidence {r.get('confidence')}")
print(f"speech {len(speech)} sound {len(sounds)} speakers {len(r.get('speakers') or [])} "
      f"dropped {r.get('dropped_segments', 0)} align_applied {r.get('alignment_applied')}")
print("text:", (r.get("text") or "")[:300])
assert speech, "no speech recognised in the sample"
PY
rm -f "$out"
docker logs "$name" 2>&1 | grep -iE "error|traceback|out of memory" | head -5 | sed 's/^/  log: /'
docker rm -f "$name" >/dev/null 2>&1 || true
[ "$status" = 0 ] && echo "=== OK" || echo "=== FAILED"
exit "$status"
