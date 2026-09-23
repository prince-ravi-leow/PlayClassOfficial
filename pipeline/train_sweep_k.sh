#!/usr/bin/env bash
# Sweep K (temporal segments) for all backbones.
# Embeddings-only, TemporalCNNv2, excluding social.
#
# Usage:
#   pixi run train_sweep_k [--device cuda:0] [--dry-run]
#   bash pipeline/train_sweep_k.sh [--device cuda:0] [--dry-run]

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

K_VALUES=(4 8 12 16 24 32 48 64)

BACKBONES=(
    "embeddings_dinov3_vitb:DINOv3-ViT-B"
    "embeddings_dinov3_vitl:DINOv3-ViT-L"
    "embeddings_videoprism_vitb_temporal:VP-Base"
    "embeddings_videoprism_vitl_temporal:VP-Large"
    "embeddings_vjepa2_vitl_temporal:V-JEPA2-ViT-L"
    "embeddings_vjepa21_vitb_temporal:V-JEPA2.1-ViT-B"
    "embeddings_vjepa21_vitl_temporal:V-JEPA2.1-ViT-L"
)

TOTAL=$(( ${#BACKBONES[@]} * ${#K_VALUES[@]} ))
RUN=0

for entry in "${BACKBONES[@]}"; do
    IFS=':' read -r EMB LABEL <<< "${entry}"

    for K in "${K_VALUES[@]}"; do
        RUN=$((RUN + 1))
        echo "=========================================="
        echo " [${RUN}/${TOTAL}] ${LABEL}  K=${K}"
        echo "=========================================="

        pixi run python -m pipeline.train \
            --model temporal_cnn2 \
            --input "${EMB}" \
            --exclude social \
            --dropout 0.0 \
            --n-segments "${K}" \
            --device "${DEVICE}" \
            ${DRY_RUN}

        echo ""
    done
done

echo "Done — ${TOTAL} runs."
