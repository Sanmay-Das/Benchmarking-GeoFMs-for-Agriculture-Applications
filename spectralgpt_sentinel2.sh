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

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

module purge
module load cuda/12.1

# Activate environment
cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

# Go to work directory
cd $MSR_ROOT/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich

# Run training (distributed launch with 1 GPU)
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=29547 --use_env train_multi_GPU_new.py