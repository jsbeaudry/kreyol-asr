#!/usr/bin/env bash
# Fresh RunPod / Lambda / Vast GPU pod -> running fine-tune, in one command.
#
#   export HF_TOKEN=hf_...
#   bash scripts/pod_bootstrap.sh
#
# Run it from inside the NeMo container (or a pod started from that image).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/kreyol-asr}"
# Default onto the persistent volume, not the container disk: a pod restart wipes
# /opt but keeps /workspace, and re-cloning NeMo every restart is wasted minutes.
export NEMO_DIR="${NEMO_DIR:-/workspace/NeMo}"
export HF_HOME="${HF_HOME:-/workspace/.hf}"
DATA_CFG="${DATA_CFG:-configs/datasets.ht.yaml}"
FT_CFG="${FT_CFG:-configs/finetune.ht.yaml}"

: "${HF_TOKEN:?HF_TOKEN must be set — private datasets and the push step need it}"

echo "==> GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# The nvcr.io/nvidia/nemo:26.06 container requires driver >= 595.58 and will
# crash-loop on hosts running 570.x (most RunPod secure-cloud machines today).
# So we do NOT depend on that image: the model card's own instructions are to
# pip-install NeMo from main, which works on any CUDA 12.8 PyTorch image.
echo "==> NeMo"
if python -c "import nemo" 2>/dev/null; then
  python -c "import nemo; print('nemo', nemo.__version__, '(preinstalled)')"
else
  echo "NeMo not present — installing from main, per the model card."
  pip install -q Cython packaging
  # The model card gives `git+https://...#egg=nemo_toolkit[asr]`, which pip >= 25
  # rejects outright ("error: invalid-egg-fragment") — egg fragments cannot carry
  # extras any more. The PEP 508 direct reference below is the supported spelling.
  pip install -q "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
  pip install -q "transformers>=5.13.0"
  # NeMo only enforces a numba MINIMUM (0.53) and no upper bound, but numba 0.62
  # moved CUDA support into the separate `numba-cuda` package and changed @cuda.jit
  # signature handling. NeMo's RNNT loss kernels predate that, so a fresh install
  # picks up 0.66 and every training step dies with
  #   TypeError: Signature mismatch: 2 argument types given, but function takes 1
  # 0.61.2 is the last release with the old in-tree CUDA target.
  pip install -q "numba==0.61.2"
  python -c "import nemo; print('nemo', nemo.__version__, '(installed)')"
fi

if [ ! -f "$NEMO_DIR/examples/asr/speech_to_text_finetune.py" ]; then
  echo "NeMo examples/ missing at $NEMO_DIR — cloning (the pip wheel omits them)."
  git clone --depth 1 https://github.com/NVIDIA/NeMo.git "$NEMO_DIR"
fi

echo "==> Install"
cd "$REPO_DIR"
pip install -e ".[dev]" >/dev/null
mkdir -p logs

echo "==> Smoke test first (fails fast on config/manifest problems)"
bash scripts/smoke.sh

echo "==> Full pipeline in tmux session 'ft' (detach with C-b d)"
tmux new-session -d -s ft "
  set -euo pipefail
  cd $REPO_DIR
  export NEMO_DIR=$NEMO_DIR HF_HOME=$HF_HOME HF_TOKEN=$HF_TOKEN
  kreyol-asr prepare --config $DATA_CFG 2>&1 | tee logs/prepare.log
  kreyol-asr patch   --config $FT_CFG   2>&1 | tee logs/patch.log
  kreyol-asr train   --config $FT_CFG   2>&1 | tee logs/train.log
  kreyol-asr bench   --config $FT_CFG   2>&1 | tee logs/bench.log
  echo 'DONE — review benchmarks/latest/report.md, then: kreyol-asr push'
  sleep infinity
"
echo "Attach with:  tmux attach -t ft"
echo "Note: push is intentionally NOT automatic — check the benchmark first."
