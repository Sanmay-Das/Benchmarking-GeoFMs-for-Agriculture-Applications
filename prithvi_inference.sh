#!/bin/bash
#SBATCH --job-name=prithvi_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prithvi_SouthMNinfer_%j.out
#SBATCH --error=logs/prithvi_SouthMNinfer_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

module purge
module load cuda/12.1

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./prithvifmenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/prithvifmenv/bin/activate" ]; then
    source "$MSR_ROOT/prithvifmenv/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs"
mkdir -p predictions/prithvi_NWIA_terratorch

echo "==== Prithvi Sliding Window Inference - EastNC ====="
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Execute the new sliding window script
#python infer_prithvi_SouthMN.py
python infer_prithvi_NWIA.py

echo ""
echo "==== Done ====="
echo ""

ls -lh predictions/prithvi_NWIA_terratorch/