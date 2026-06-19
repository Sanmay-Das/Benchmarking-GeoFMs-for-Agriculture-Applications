#!/bin/bash
#SBATCH --job-name=satmae_cd_MN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/satmae_cd_%j_MN.out
#SBATCH --error=logs/satmae_cd_%j_MN.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

cd /bigdata/eldawylab/sdas050/MS_Research/SatMAE/ChangeDetection

mkdir -p logs

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "==== SatMAE CD Training — Minnesota ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Train: NorthMN  |  Val: CentMN  |  Test: SouthMN"
echo ""

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env train_cd_satmae_MN.py \
    --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/satmae \
    --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/pretrain-vit-large-e199.pth \
    --output-dir ./cd_train_satmae_MN \
    --lr 0.0001 \
    --warmup-epochs 0 \
    --batch_size 16 \
    --epochs 60

echo ""
echo "==== Done ===="
ls -lh cd_train_satmae_MN/
