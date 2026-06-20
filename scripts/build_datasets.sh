#!/usr/bin/env bash
# Build and push the HOI HuggingFace datasets used by the repo.
#
# Usage:
#   bash scripts/build_datasets.sh <hf-username> <hicodet-path> <vghoi-path>
#
# Requires HF_TOKEN in the environment.

set -euo pipefail

HF_USER="${1:?Usage: bash scripts/build_datasets.sh <hf-username> <hicodet-path> <vghoi-path>}"
HICODET_PATH="${2:?Path to HICODET dataset root is required}"
VGHOI_PATH="${3:?Path to VGHOI dataset root is required}"

python -m src.data.datasets._builder \
    --hf_username "$HF_USER" \
    --dataset_name hicodet \
    --dataset_path "$HICODET_PATH" \
    --dataset_type instance \
    --dataset_splits train test \
    --detectors gt detr_r50

python -m src.data.datasets._builder \
    --hf_username "$HF_USER" \
    --dataset_name vghoi \
    --dataset_path "$VGHOI_PATH" \
    --dataset_type instance \
    --dataset_splits test \
    --detectors gt gdino
