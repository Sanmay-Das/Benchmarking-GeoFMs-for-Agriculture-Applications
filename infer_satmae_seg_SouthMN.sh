#!/bin/bash
#SBATCH --job-name=satmae_seg_SouthMN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/satmae_seg_SouthMN_%j.out
#SBATCH --error=logs/satmae_seg_SouthMN_%j.err

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

mkdir -p "$MSR_OUTPUT_ROOT/logs" \
         "$MSR_OUTPUT_ROOT/predictions/satmae_fcn_SouthMN" \
         "$MSR_OUTPUT_ROOT/predictions/satmae_fpn_SouthMN" \
         "$MSR_OUTPUT_ROOT/predictions/satmae_psanet_SouthMN"

echo "==== SatMAE Segmentation Inference -- SouthMN ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

echo "====== [1/3] FCN -- already done, skipping ======"
echo ""

echo "====== [2/3] FPN ======"
python infer_satmae_fpn_SouthMN.py
echo ""

echo "====== [3/3] PSANet ======"
python infer_satmae_psanet_SouthMN.py
echo ""

echo "==== Done ===="
ls -lh predictions/satmae_fcn_SouthMN/
ls -lh predictions/satmae_fpn_SouthMN/
ls -lh predictions/satmae_psanet_SouthMN/
