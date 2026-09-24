#!/usr/bin/env bash
# Subtractive ablations on best model: Feat. + V-JEPA 2.1 ViT-L, TemporalCNNv2.
# Each run changes one hyperparameter from the reference config.
#
# Usage:
#   pixi run train_ablate_best [--device cuda:0] [--dry-run]
#   bash pipeline/train_ablate_best.sh [--device cuda:0] [--dry-run]

set -euo pipefail

DEVICE="cuda:0"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

INPUT="features+embeddings_vjepa21_vitl_temporal"
RUN=0
TOTAL=4

run_train() {
    local LABEL="$1"; shift
    RUN=$((RUN + 1))
    echo "=========================================="
    echo " [${RUN}/${TOTAL}] ${LABEL}"
    echo "=========================================="
    pixi run train \
        --input "${INPUT}" \
        --exclude social \
        --device "${DEVICE}" \
        ${DRY_RUN} \
        "$@"
    echo ""
}

# 1. 1 epoch (K=24)
run_train "1 epoch (K=24)" \
    --model temporal_cnn2 --dropout 0.0 --n-segments 24 --epochs 1

# 2. No class weights (K=24)
run_train "No class weights (K=24)" \
    --model temporal_cnn2 --dropout 0.0 --n-segments 24 --class-weights none

# 3. No label smoothing (K=24)
run_train "No label smoothing (K=24)" \
    --model temporal_cnn2 --dropout 0.0 --n-segments 24 --label-smoothing 0.0

# 4. MLP (no temporal)
run_train "MLP (no temporal)" \
    --model mlp

echo "Done — ${TOTAL} runs."
