#!/usr/bin/env bash
# Phase 0.5 smoke: prove the Stage 1 mechanics on a real GPU, cheaply.
#
# This is NOT a convergence test. It answers the questions that can only be answered
# on a pod, and nothing else:
#
#   1. Do the StyleTTS 2 dependencies install, and does monotonic_align actually build?
#   2. Does the Kokoro warm start load into StyleTTS 2's model builder without shape
#      errors, and how many tensors land vs. get reinitialized?
#   3. Do training steps produce finite, sane losses?
#   4. How many steps per second — so the plan's extrapolated wall-clock estimates can
#      be replaced with measurements.
#
# Everything below writes to /workspace (the persistent volume), never to / or /root,
# for the same reason scripts/pod_bootstrap.sh does: the container disk does not survive.
set -euo pipefail

WORK=/workspace
SMOKE=$WORK/smoke
N_CLIPS=${N_CLIPS:-500}
MAX_STEPS=${MAX_STEPS:-60}
mkdir -p "$SMOKE"
cd "$WORK"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
trap 'echo "SMOKE_RESULT=FAILED (line $LINENO)"; exit 1' ERR

log "=== 1/6 system ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
           torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

log "=== 2/6 code ==="
# Shipped inline as base64 tar.gz so nothing has to be pushed to a public remote.
if [ -n "${KREYOL_SRC_B64:-}" ]; then
  echo "$KREYOL_SRC_B64" | base64 -d | tar xz -C "$WORK"
  log "unpacked $(find $WORK/src -name '*.py' | wc -l) python files"
fi
export PYTHONPATH="$WORK/src:${PYTHONPATH:-}"

if [ ! -d "$WORK/StyleTTS2" ]; then
  git clone --depth 1 https://github.com/yl4579/StyleTTS2.git "$WORK/StyleTTS2"
fi
cd "$WORK/StyleTTS2"
log "aligner + JDC ship inside the repo: $(ls -la Utils/ASR/epoch_00080.pth Utils/JDC/bst.t7 | wc -l) files present"

log "=== 3/6 deps ==="
pip install -q SoundFile torchaudio munch pydub pyyaml librosa nltk matplotlib accelerate \
    transformers einops einops-exts tqdm typing-guard git+https://github.com/resemble-ai/monotonic_align.git \
    soxr huggingface_hub datasets pyarrow 2>&1 | tail -3
python -c "import monotonic_align, torchaudio, librosa, transformers; print('deps OK')"

log "=== 4/6 warm start ==="
cd "$WORK"
python - <<'PYEOF'
import os, sys, torch
sys.path.insert(0, "/workspace/src")
from kreyol_tts.warmstart import to_styletts2, fetch_kokoro
src = fetch_kokoro(os.environ.get("HF_TOKEN"))
rep = to_styletts2(src, "/workspace/checkpoints/kokoro_ht_init.pth")
assert rep["total_params"] > 80e6, rep["total_params"]
print("WARMSTART_PARAMS=%d" % rep["total_params"])
print("WARMSTART_MISSING=%s" % ",".join(rep["missing"]))
PYEOF

log "=== 5/6 data ==="
python - <<PYEOF
import os, sys
sys.path.insert(0, "/workspace/src")
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
from smoke_data import build
build(n=$N_CLIPS, out="/workspace/smoke/data")
PYEOF

log "=== 6/6 training steps ==="
cd "$WORK/StyleTTS2"
python "$WORK/src/smoke_train.py" --steps "$MAX_STEPS" --data /workspace/smoke/data \
       --init /workspace/checkpoints/kokoro_ht_init.pth --out /workspace/smoke/out

log "SMOKE_RESULT=OK"
