#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH -J pi05_abpolicy
#SBATCH -o logs/pi05_cylinder_full_abpolicy_10ksteps/%x_%j.out
#SBATCH -e logs/pi05_cylinder_full_abpolicy_10ksteps/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot
mkdir -p logs/pi05_cylinder_full_abpolicy_10ksteps
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

. /hkfs/work/workspace/scratch/utphd-myspace/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
source bash_scripts/env_lerobot.sh

python - <<'PY'
from pathlib import Path

import lerobot
from lerobot.policies.pi05.configuration_pi05 import PI05Config

expected_source = (Path.cwd() / "src").resolve()
loaded_source = Path(lerobot.__file__).resolve()
print(f"Using LeRobot from: {loaded_source}")
if not loaded_source.is_relative_to(expected_source):
    raise RuntimeError(f"Expected LeRobot under {expected_source}, loaded {loaded_source}")
if not hasattr(PI05Config, "abpolicy_enabled"):
    raise RuntimeError("Loaded PI05Config does not contain the ABPolicy implementation")
PY

accelerate launch \
  --use_deepspeed \
  --zero_stage=2 \
  --offload_optimizer_device=none \
  --num_processes=4 \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id=/hkfs/work/workspace/scratch/utphd-myspace/datasets/cylinder_full \
  --output_dir=/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_cylinder_full_abpolicy_10ksteps \
  --job_name=pi05_abpolicy \
  --policy.path=/hkfs/work/workspace/scratch/utphd-myspace/models/pi05_base \
  --policy.repo_id=local/pi05-abpolicy \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.static_cam": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb"}' \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.abpolicy_enabled=true \
  --policy.chunk_size=8 \
  --policy.n_action_steps=32 \
  --policy.abpolicy_past_action_steps=8 \
  --policy.abpolicy_future_action_steps=32 \
  --policy.abpolicy_spline_degree=3 \
  --policy.abpolicy_num_control_points=8 \
  --policy.abpolicy_num_free_control_points=4 \
  --policy.abpolicy_action_representation=cartesian_rotvec \
  --policy.use_relative_actions=false \
  --save_freq=5000 \
  --gradient_accumulation_steps=4 \
  --steps=10000 \
  --batch_size=16
