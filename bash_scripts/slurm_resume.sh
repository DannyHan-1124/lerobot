#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH -J pi05_puma_cylinder_resume_5000
#SBATCH -o /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_cylinder_resume_5000/%x_%j.out
#SBATCH -e /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_cylinder_resume_5000/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

mkdir -p logs/pi05_puma_cylinder_resume_5000

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

. /hkfs/work/workspace/scratch/utphd-myspace/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
source bash_scripts/env_lerobot.sh

export CHECKPOINT_DIR=/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_puma_cylinder_full_10k/checkpoints/005000
export TRAIN_CONFIG="$CHECKPOINT_DIR/pretrained_model/train_config.json"
export HF_DATASETS_CACHE=$PROJECT_WS/hf_cache/puma_cylinder_full
mkdir -p "$HF_DATASETS_CACHE"

export MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))

echo "Job ${SLURM_JOB_ID:-unknown} running on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
nvidia-smi

python - <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

checkpoint = Path(os.environ["CHECKPOINT_DIR"])
config = Path(os.environ["TRAIN_CONFIG"])
step_file = checkpoint / "training_step.json"
optimizer_dir = checkpoint / "accelerate_state" / "pytorch_model"

print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch; aborting instead of falling back to CPU.")
if torch.cuda.device_count() != 4:
    sys.exit(f"Expected 4 GPUs for this resume job, found {torch.cuda.device_count()}.")
if not config.is_file() or not step_file.is_file():
    sys.exit(f"Incomplete checkpoint: {checkpoint}")
step = int(json.loads(step_file.read_text())["step"])
if step != 5000:
    sys.exit(f"Expected checkpoint step 5000, found {step}.")
optimizer_shards = sorted(optimizer_dir.glob("bf16_zero_pp_rank_*_optim_states.pt"))
if len(optimizer_shards) != 4:
    sys.exit(f"Expected 4 DeepSpeed optimizer shards, found {len(optimizer_shards)}.")
print(f"Checkpoint validation passed at step {step} with 4 optimizer shards.")
PY

accelerate launch \
  --use_deepspeed \
  --zero_stage=2 \
  --offload_optimizer_device=none \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --config_path="$TRAIN_CONFIG" \
  --resume=true \
  --save_freq=2500 \
  --steps=10000
