#!/usr/bin/env bash
set -euo pipefail

python -m lerobot.scripts.cache_puma_features \
  --dataset-root /hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_full \
  --repo-id /hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_full \
  --output-dir /hkfs/work/workspace/scratch/utphd-myspace/puma_features/cylinder_full \
  --camera observation.images.static_cam \
  --target-map /hkfs/work/workspace/scratch/utphd-myspace/lerobot/bash_scripts/cylinder_full_target_map.json \
  --future-steps 4 \
  --future-stride 4 \
  --device cuda \
  --dtype bfloat16
