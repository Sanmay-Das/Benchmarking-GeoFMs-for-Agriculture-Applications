#!/bin/bash
#SBATCH --job-name=prithvi_cd_NC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/prithvi_cd_%j_NC.out
#SBATCH --error=logs/prithvi_cd_%j_NC.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

cd /bigdata/eldawylab/sdas050/MS_Research/prithvi_finetune/ChangeDetection

mkdir -p logs

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "==== Prithvi CD Training — North Carolina ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Train: NENC  |  Val: ECNC  |  Test: EastNC"
echo ""

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env train_cd_prithvi_NC.py \
    --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/prithvi \
    --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/Prithvi_EO_V1_100M.pt \
    --output-dir ./cd_train_prithvi_NC \
    --lr 0.0001 \
    --warmup-epochs 0 \
    --batch_size 16 \
    --epochs 60

echo ""
echo "==== Done ===="
ls -lh cd_train_prithvi_NC/
