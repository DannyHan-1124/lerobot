#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH -J pi05_moving_can_FASTER_10ksteps
#SBATCH -o logs/pi05_moving_can_FASTER_10ksteps/%x_%j.out
#SBATCH -e logs/pi05_moving_can_FASTER_10ksteps/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot

mkdir -p logs/pi05_moving_can_FASTER_10ksteps

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

. /hkfs/work/workspace/scratch/utphd-myspace/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

source bash_scripts/env_lerobot.sh

export DATASET_DIR=/hkfs/work/workspace/scratch/utphd-myspace/datasets/put_moving_can_in_bowl
export HF_DATASETS_CACHE=$PROJECT_WS/hf_cache/datasets_put_moving_can_in_bowl
mkdir -p "$HF_DATASETS_CACHE"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job ${SLURM_JOB_ID:-unknown} running on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
nvidia-smi
python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch; aborting instead of falling back to CPU.")
PY

python - <<'PY'
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

dataset = Path(os.environ["DATASET_DIR"])
episodes_path = dataset / "meta/episodes/chunk-000/file-000.parquet"
episodes = pd.read_parquet(episodes_path)

parts = []
for parquet_path in sorted((dataset / "data").glob("*/*.parquet")):
    parts.append(pq.read_table(parquet_path, columns=["episode_index", "index"]).to_pandas())

if not parts:
    raise SystemExit(f"No data parquet files found under {dataset / 'data'}")

data = pd.concat(parts, ignore_index=True)
actual = data.groupby("episode_index")["index"].agg(["min", "max", "count"]).reset_index()
merged = episodes.merge(actual, on="episode_index", how="left")
merged["actual_to"] = merged["max"] + 1
bad = merged[
    (merged["dataset_from_index"] != merged["min"])
    | (merged["dataset_to_index"] != merged["actual_to"])
    | (merged["length"] != merged["count"])
]
if len(bad) > 0:
    print(bad[["episode_index", "dataset_from_index", "dataset_to_index", "length", "min", "actual_to", "count"]])
    raise SystemExit("Dataset episode metadata is not repaired; aborting training.")

print(f"Dataset index sanity check passed for {len(episodes)} episodes.")
PY

accelerate launch \
  --use_deepspeed \
  --zero_stage=2 \
  --offload_optimizer_device=none \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id="$DATASET_DIR" \
  --output_dir=/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_moving_can_FASTER_10ksteps \
  --job_name=pi05_moving_can_FASTER_10ksteps \
  --policy.path=/hkfs/work/workspace/scratch/utphd-myspace/models/pi05_base \
  --policy.repo_id=local/pi05-test \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.static_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.optimizer_lr=5e-05 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=20000 \
  --policy.scheduler_decay_lr=2.5e-06 \
  --policy.faster_train_max_delay=10 \
  --policy.faster_train_mix_prob=0.5 \
  --policy.faster_train_alpha=0.6 \
  --policy.faster_train_u0=0.9 \
  --save_freq=10000 \
  --gradient_accumulation_steps=4 \
  --steps=10000 \
  --batch_size=16
