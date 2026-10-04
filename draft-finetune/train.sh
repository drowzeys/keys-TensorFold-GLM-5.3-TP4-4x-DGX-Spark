#!/bin/bash
# Fine-tune incoai/GLM-5.3-DFlash2 on your captured data, on ONE DGX Spark (not serving), ~4 h for 3200 steps.
# The settings of our fine-tune: 900 captured sequences, 3200 steps x batch 2, KL to the target's soft labels.
#   DRAFT=/models/GLM-5.3-DFlash2 TARGET=/models/GLM-5.3-EXL3-2.75BPW CAP=/data/cap OUT=/data/draft-ft ./train.sh
set -eu
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-04}
: "${DRAFT:?incoai/GLM-5.3-DFlash2 directory}" "${TARGET:?the EXL3 checkpoint directory}" "${CAP:?capture directory (seq-*.safetensors)}" "${OUT:?output directory}"
mkdir -p "$OUT"
docker run --rm --gpus all --ipc=host --memory 110g --memory-swap 110g \
  -v "$(cd "$(dirname "$0")" && pwd)":/ft:ro -v "$DRAFT":/draft:ro -v "$TARGET":/target:ro -v "$CAP":/seq:ro -v "$OUT":/out \
  -e DRAFT_DIR=/draft -e TARGET_DIR=/target -e SEQ_DIR=/seq -e OUT_DIR=/out \
  -e TRAIN_SET=full -e LOSS=kl -e TAIL=bucket -e SOFT_TEMP=1.0 -e POS_WEIGHT=dpace -e SEL_TARGET=soft -e SEL_POSW=1 \
  -e STEPS=${STEPS:-3200} -e LR=1e-4 -e WARMUP_STEPS=50 -e BATCH=2 -e EVAL_EVERY=100 -e SAVE_EVERY=400 \
  -e MEM_FRACTION=0.60 -e POOL=8 -e HOLDOUT_MAX=64 \
  --entrypoint python3 "$IMAGE" /ft/dflash2_train.py
cp "$DRAFT/config.json" "$OUT/config.json"
echo "draft: $OUT (model.safetensors + config.json); serve it with DRAFT=$OUT ./one-shot.sh up"
