#!/usr/bin/env bash
# Usage:
#   pixi run -e tracker setup_hqsam2

set -euo pipefail

HQ_DIR="ext/sam-hq"
HQ_REPO="https://github.com/SysCV/sam-hq"
HQ_COMMIT="e696978d60352dc9a26b12631cd91781502c6546" # main, 2025-09-11

if [[ -d "$HQ_DIR/sam-hq2/sam2" ]]; then
    echo "HQ-SAM 2 already present in $HQ_DIR, skipping."
    exit 0
fi

# Shallow, sparse checkout of the pinned commit: just the sam2 package (~2 MB
# instead of the full ~150 MB repo)
git init -q "$HQ_DIR"
git -C "$HQ_DIR" remote add origin "$HQ_REPO"
git -C "$HQ_DIR" sparse-checkout set sam-hq2/sam2
git -C "$HQ_DIR" fetch -q --depth 1 --filter=blob:none origin "$HQ_COMMIT"
git -C "$HQ_DIR" checkout -q FETCH_HEAD

echo "HQ-SAM 2 code fetched to $HQ_DIR/sam-hq2 (commit ${HQ_COMMIT:0:7})."
