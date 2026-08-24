#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m lerobot.scripts.cache_puma_features \
  --dataset-root /hkfs/work/workspace/scratch/utphd-myspace/datasets/conveyor_cube \
  --repo-id /hkfs/work/workspace/scratch/utphd-myspace/datasets/conveyor_cube \
  --output-dir /hkfs/work/workspace/scratch/utphd-myspace/puma_features/conveyor_cube \
  --camera observation.images.static_cam \
  --target-map /hkfs/work/workspace/scratch/utphd-myspace/lerobot/bash_scripts/conveyor_cube_target_map.json \
  --future-steps 4 \
  --future-stride 4 \
  --device cuda \
  --dtype bfloat16
