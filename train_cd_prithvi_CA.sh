#!/bin/bash
#SBATCH --job-name=train_cd_prithvi_CA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_cd_prithvi_CA_%j.out
#SBATCH --error=logs/train_cd_prithvi_CA_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs prithvi_finetune/ChangeDetection/cd_train_prithvi_CA

echo "==== Prithvi CD Training — California ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi | head -20
echo ""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd prithvi_finetune/ChangeDetection

python train_cd_prithvi_CA.py \
    --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/prithvi \
    --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/Prithvi_EO_V1_100M.pt \
    --output-dir ./cd_train_prithvi_CA \
    --epochs 60 \
    --batch_size 4 \
    --lr 1e-3 \
    --workers 8 \
    --amp True

echo ""
echo "==== Done ===="
ls -lh cd_train_prithvi_CA/
