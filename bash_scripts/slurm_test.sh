#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=20
#SBATCH -J pi05_puma_conveyor_smoke
#SBATCH -o /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_conveyor_smoke/%x_%j.out
#SBATCH -e /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_conveyor_smoke/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

mkdir -p logs/pi05_puma_conveyor_smoke

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

. /hkfs/work/workspace/scratch/utphd-myspace/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

source bash_scripts/env_lerobot.sh

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
if torch.cuda.device_count() != 4:
    sys.exit(f"Expected 4 GPUs for this smoke test, found {torch.cuda.device_count()}.")
PY

accelerate launch \
  --use_deepspeed \
  --zero_stage=2 \
  --offload_optimizer_device=none \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id=/hkfs/work/workspace/scratch/utphd-myspace/datasets/conveyor_cube \
  --output_dir=/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_puma_conveyor_cube_smoke \
  --job_name=pi05_puma_conveyor_cube_smoke \
  --policy.path=/hkfs/work/workspace/scratch/utphd-myspace/models/pi05_base \
  --policy.repo_id=local/pi05-puma-conveyor-smoke \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.static_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb"}' \
  --policy.puma_config.enabled=true \
  --policy.puma_config.dataset_flow_camera_key=observation.images.static_cam \
  --policy.puma_config.flow_camera_key=observation.images.base_0_rgb \
  --policy.puma_config.feature_cache=/hkfs/work/workspace/scratch/utphd-myspace/puma_features/conveyor_cube \
  --policy.puma_config.history_steps=4 \
  --policy.puma_config.history_stride=4 \
  --policy.puma_config.future_steps=4 \
  --policy.puma_config.future_stride=4 \
  --policy.puma_config.future_feature_dim=768 \
  --policy.puma_config.world_loss_weight=0.05 \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.optimizer_lr=5e-05 \
  --policy.scheduler_warmup_steps=1 \
  --policy.scheduler_decay_steps=2 \
  --policy.scheduler_decay_lr=2.5e-06 \
  --save_checkpoint=false \
  --log_freq=1 \
  --num_workers=4 \
  --gradient_accumulation_steps=1 \
  --steps=2 \
  --batch_size=1
