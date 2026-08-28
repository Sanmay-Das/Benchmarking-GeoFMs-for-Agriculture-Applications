#!/bin/bash
#SBATCH --job-name=satmae_fpn_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/satmae_fpn_NWIAinfer_%j.out
#SBATCH --error=logs/satmae_fpn_NWIAinfer_%j.err

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
# requirements/; falls back to ./satmae_env if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/satmae_env/bin/activate" ]; then
    source "$MSR_ROOT/satmae_env/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs"
mkdir -p predictions/satmae_fpn_NWIA

echo "==== SatMAE + FPN Sliding Window Inference - NWIA ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python infer_satmae_fpn_NWIA.py

echo ""
echo "==== Done ===="
ls -lh predictions/satmae_fpn_NWIA/
