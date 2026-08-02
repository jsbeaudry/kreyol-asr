#!/usr/bin/env bash
# ~10 minute end-to-end proof before committing to a multi-hour run.
# Exercises every moving part at tiny scale: HF download -> manifests -> .nemo patch
# -> NeMo training step -> checkpoint write.
set -euo pipefail

FT_CFG="${FT_CFG:-configs/finetune.ht.yaml}"
SMOKE_DIR="${SMOKE_DIR:-data/smoke}"
CLIPS="${CLIPS:-50}"
STEPS="${STEPS:-20}"

echo "==> [1/4] prepare ($CLIPS clips from the public CMU set) -> $SMOKE_DIR"
# --output-dir is essential: without it this writes to the config's output_dir
# (data/ht) and destroys a real prepared corpus.
kreyol-asr prepare \
  --datasets jsbeaudry/cmu_haitian_creole_speech:train \
  --limit "$CLIPS" \
  --config configs/datasets.ht.yaml \
  --output-dir "$SMOKE_DIR"

echo "==> [2/4] patch (register ht-HT, warm-start from fr-FR)"
kreyol-asr patch --config "$FT_CFG" --out checkpoints/smoke-init.nemo

echo "==> [3/4] train ($STEPS steps)"
# --exp-dir is not optional here. exp_manager runs with resume_if_exists=true, so
# a smoke checkpoint sitting in the real exp/ makes the next real run RESUME from
# 20 steps of 50 clips and quietly ignore --init.
kreyol-asr train --config "$FT_CFG" \
  --data-dir "$SMOKE_DIR" \
  --init checkpoints/smoke-init.nemo \
  --exp-dir exp/smoke \
  --max-steps "$STEPS" --devices 1

echo "==> [4/4] baseline inference sanity check"
# Should produce recognizable French-ish text (the base model has no Creole slot,
# so it decodes Creole as French). Gibberish => bad audio paths or a broken
# manifest, not a bad model.
#
# --data-dir keeps this on the smoke corpus. Without it, bench falls back to the
# real test split and takes ~10 min for 391 clips x 4 latencies — a real
# measurement, but not what a smoke test is for. The authoritative baseline comes
# from the full `kreyol-asr bench` run.
kreyol-asr bench --config "$FT_CFG" --baseline-only \
  --data-dir "$SMOKE_DIR" --out-dir benchmarks/smoke

echo
echo "SMOKE OK. Steps/sec from the training log tells you what a full run costs."
