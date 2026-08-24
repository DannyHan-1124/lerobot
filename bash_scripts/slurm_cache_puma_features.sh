#!/bin/bash
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH -J cache_puma_conveyor_cube
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd /hkfs/work/workspace/scratch/utphd-myspace/lerobot
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
mkdir -p logs/cache_puma_conveyor_cube

module purge
module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

. /hkfs/work/workspace/scratch/utphd-myspace/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
source bash_scripts/env_lerobot.sh

echo "Job ${SLURM_JOB_ID:-unknown} running on $(hostname)"
nvidia-smi
bash bash_scripts/cache_puma_features.sh
