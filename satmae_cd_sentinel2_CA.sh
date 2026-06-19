#!/bin/bash
#SBATCH --job-name=satmae_cd_CA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/satmae_cd_%j_CA.out
#SBATCH --error=logs/satmae_cd_%j_CA.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

cd /bigdata/eldawylab/sdas050/MS_Research/SatMAE/ChangeDetection

mkdir -p logs

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "==== SatMAE CD Training — California ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Train: NorthCA  |  Val: CentCA  |  Test: SouthCA"
echo ""

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env train_cd_satmae_CA.py \
    --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/satmae \
    --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/pretrain-vit-large-e199.pth \
    --output-dir ./cd_train_satmae_CA \
    --lr 0.0001 \
    --warmup-epochs 0 \
    --batch_size 16 \
    --epochs 60

echo ""
echo "==== Done ===="
ls -lh cd_train_satmae_CA/
