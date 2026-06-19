#!/bin/bash
#SBATCH --job-name=sgpt_cd_CA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spectralgpt_cd_%j_CA.out
#SBATCH --error=logs/spectralgpt_cd_%j_CA.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

cd /bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection

mkdir -p logs

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "==== SpectralGPT CD Training — California ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Train: NorthCA  |  Val: CentCA  |  Test: SouthCA"
echo ""

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env train_cd_spectralgpt_CA.py \
    --data-root /bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/spectralgpt \
    --pretrain-path /bigdata/eldawylab/sdas050/MS_Research/weights/SpectralGPT+.pth \
    --output-dir ./cd_train_spectralgpt_CA \
    --lr 0.0001 \
    --warmup-epochs 0

echo ""
echo "==== Done ===="
ls -lh cd_train_spectralgpt_CA/
