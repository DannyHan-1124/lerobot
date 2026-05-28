#!/bin/bash
# Interactive session setup for LeRobot on HoreKa

module use /software/easybuild/modules/all
module load FFmpeg/7.1.2-GCCcore-14.3.0
module load devel/cuda/12.9

# Activate your virtual environment
conda activate lerobot

# Set up LeRobot environment
source env_lerobot.sh
