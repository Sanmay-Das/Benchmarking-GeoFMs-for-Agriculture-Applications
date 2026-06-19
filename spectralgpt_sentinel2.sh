#!/bin/bash
#SBATCH --job-name=spectralgpt_mn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spectralgpt_NorthCentMN_%j.out
#SBATCH --error=logs/spectralgpt_NorthCentMN_%j.err

set -euo pipefail

module purge
module load cuda/12.1

# Activate environment
cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

# Go to work directory
cd /bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich

# Run training (distributed launch with 1 GPU)
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=29547 --use_env train_multi_GPU_new.py