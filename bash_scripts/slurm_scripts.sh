#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH -J pi05_puma_cylinder_full_10k
#SBATCH -o /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_cylinder_full_10k/%x_%j.out
#SBATCH -e /hkfs/work/workspace/scratch/utphd-myspace/lerobot/logs/pi05_puma_cylinder_full_10k/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

mkdir -p logs/pi05_puma_cylinder_full_10k

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

. /hkfs/work/workspace/scratch/utphd-myspace/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

source bash_scripts/env_lerobot.sh

export DATASET_DIR=/hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_full
export PUMA_FEATURE_CACHE=/hkfs/work/workspace/scratch/utphd-myspace/puma_features/cylinder_full
export OUTPUT_DIR=/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_puma_cylinder_full_10k
export HF_DATASETS_CACHE=$PROJECT_WS/hf_cache/puma_cylinder_full
mkdir -p "$HF_DATASETS_CACHE"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job ${SLURM_JOB_ID:-unknown} running on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "PUMA_FEATURE_CACHE=${PUMA_FEATURE_CACHE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
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
if torch.cuda.device_count() != 4:
    sys.exit(f"Expected 4 GPUs for this training job, found {torch.cuda.device_count()}.")
PY

accelerate launch \
  --use_deepspeed \
  --zero_stage=2 \
  --offload_optimizer_device=none \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id="$DATASET_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_puma_cylinder_full_10k \
  --policy.path=/hkfs/work/workspace/scratch/utphd-myspace/models/pi05_base \
  --policy.repo_id=local/pi05-puma-cylinder-full \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.static_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb"}' \
  --policy.puma_config.enabled=true \
  --policy.puma_config.dataset_flow_camera_key=observation.images.static_cam \
  --policy.puma_config.flow_camera_key=observation.images.base_0_rgb \
  --policy.puma_config.feature_cache="$PUMA_FEATURE_CACHE" \
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
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=10000 \
  --policy.scheduler_decay_lr=2.5e-06 \
  --save_freq=5000 \
  --log_freq=200 \
  --num_workers=4 \
  --gradient_accumulation_steps=8 \
  --steps=10000 \
  --batch_size=8
