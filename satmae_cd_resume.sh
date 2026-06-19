#!/bin/bash
#SBATCH --job-name=satmae_cd_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/satmae_cd_%j_resume.out
#SBATCH --error=logs/satmae_cd_%j_resume.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

# Activate environment
cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

# Go to work directory
cd /bigdata/eldawylab/sdas050/MS_Research/SatMAE/ChangeDetection

# Create logs dir if not exists
mkdir -p logs

# Derive unique port from job ID to avoid conflicts
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Resume training from last_checkpoint.pth (saved at epoch 45)
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env train_cd_satmae.py \
    --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/satmae \
    --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/pretrain-vit-large-e199.pth \
    --output-dir ./cd_train_satmae \
    --lr 0.0001 \
    --warmup-epochs 0 \
    --batch_size 16 \
    --epochs 60 \
    --resume ./cd_train_satmae/last_checkpoint.pth
