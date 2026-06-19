#!/bin/bash
#SBATCH --job-name=spectralgpt_cd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spectralgpt_cd_%j_IA.out
#SBATCH --error=logs/spectralgpt_cd_%j_IA.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

# Activate environment
cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

# Go to work directory
cd /bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection

# Create logs dir if not exists
mkdir -p logs

# Derive unique port from job ID to avoid conflicts when multiple jobs run on same node
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Run training (distributed launch with 1 GPU)
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env train_cd_spectralgpt.py \
    --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/spectralgpt \
    --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/SpectralGPT+.pth \
    --output-dir ./cd_train_spectralgpt \
    --lr 0.0001 \
    --warmup-epochs 0