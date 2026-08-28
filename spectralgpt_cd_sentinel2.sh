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

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

# Cluster module system (UCR HPCC). Skipped when unavailable, e.g. on a
# workstation where CUDA is already on the path.
if command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.1
fi

# Activate environment
cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

# Go to work directory
cd $MSR_ROOT/IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection

# Create logs dir if not exists
mkdir -p "$MSR_OUTPUT_ROOT/logs"

# Derive unique port from job ID to avoid conflicts when multiple jobs run on same node
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Run training (distributed launch with 1 GPU)
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env train_cd_spectralgpt.py \
    --data-root $MSR_ROOT/change_detection_chips/spectralgpt \
    --pretrain-path $MSR_ROOT/weights/SpectralGPT+.pth \
    --output-dir ./cd_train_spectralgpt \
    --lr 0.0001 \
    --warmup-epochs 0