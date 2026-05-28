#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH -J test
#SBATCH -o logs/test/%x_%j.out
#SBATCH -e logs/test/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot

mkdir -p logs/test

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

conda activate lerobot
source env_lerobot.sh

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job ${SLURM_JOB_ID:-unknown} running on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT=${MASTER_PORT}"
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

accelerate launch \
  --use_deepspeed \
  --zero_stage=2 \
  --offload_optimizer_device=none \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id=/hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_cube_full \
  --output_dir=/hkfs/work/workspace/scratch/utphd-myspace/outputs/test \
  --job_name=test \
  --policy.path=lerobot/pi05_base \
  --policy.repo_id=local/pi05-test \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.static_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb"}' \
  --policy.empty_cameras=1 \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.optimizer_lr=1e-04 \
  --gradient_accumulation_steps=4 \
  --steps=10 \
  --batch_size=16

