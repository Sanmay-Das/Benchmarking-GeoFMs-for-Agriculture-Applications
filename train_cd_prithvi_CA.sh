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

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

# Cluster module system (UCR HPCC). Skipped when unavailable, e.g. on a
# workstation where CUDA is already on the path.
if command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.1
fi

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs" "$MSR_OUTPUT_ROOT/prithvi_finetune/ChangeDetection/cd_train_prithvi_CA"

echo "==== Prithvi CD Training -- California ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
nvidia-smi | head -20
echo ""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd prithvi_finetune/ChangeDetection

python train_cd_prithvi_CA.py \
    --data-root $MSR_ROOT/change_detection_chips/prithvi \
    --pretrain-path $MSR_ROOT/weights/Prithvi_EO_V1_100M.pt \
    --output-dir ./cd_train_prithvi_CA \
    --epochs 60 \
    --batch_size 4 \
    --lr 1e-3 \
    --workers 8 \
    --amp True

echo ""
echo "==== Done ===="
ls -lh cd_train_prithvi_CA/
