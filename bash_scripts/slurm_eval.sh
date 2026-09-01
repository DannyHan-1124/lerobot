#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH -J pi05_puma_conveyor_openloop_1500
#SBATCH -o /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_conveyor_openloop_1500/%x_%j.out
#SBATCH -e /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_conveyor_openloop_1500/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

mkdir -p logs/pi05_puma_conveyor_openloop_1500

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

. /hkfs/work/workspace/scratch/utphd-myspace/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

source bash_scripts/env_lerobot.sh

export CHECKPOINT_DIR=/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_puma_conveyor_cube_2k/checkpoints/001500
export HF_DATASETS_CACHE=$PROJECT_WS/hf_cache/puma_conveyor_cube
mkdir -p "$HF_DATASETS_CACHE"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job ${SLURM_JOB_ID:-unknown} running on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
nvidia-smi
python - <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

checkpoint = Path(os.environ["CHECKPOINT_DIR"])
pretrained = checkpoint / "pretrained_model"
config_path = pretrained / "train_config.json"
step_path = checkpoint / "training_step.json"

print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch; aborting instead of falling back to CPU.")
if torch.cuda.device_count() != 1:
    sys.exit(f"Expected 1 GPU for this evaluation job, found {torch.cuda.device_count()}.")
required = [step_path, config_path, pretrained / "model.safetensors"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    sys.exit(f"Incomplete checkpoint; missing files: {missing}")
step = int(json.loads(step_path.read_text())["step"])
config = json.loads(config_path.read_text())
dataset = Path(config["dataset"]["repo_id"])
feature_cache = Path(config["policy"]["puma_config"]["feature_cache"])
if not dataset.is_dir():
    sys.exit(f"Dataset does not exist: {dataset}")
if not feature_cache.is_dir():
    sys.exit(f"PUMA feature cache does not exist: {feature_cache}")
print(f"Checkpoint validation passed at step {step}.")
print(f"Dataset: {dataset}")
print(f"PUMA feature cache: {feature_cache}")
PY

python /hkfs/work/workspace/scratch/utphd-myspace/lerobot/openloop_eval.py
