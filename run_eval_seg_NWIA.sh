#!/bin/bash
#SBATCH --job-name=eval_seg_NWIA
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_seg_NWIA_%j.out
#SBATCH --error=logs/eval_seg_NWIA_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/satmae_env/bin/activate" ]; then
    source "$MSR_ROOT/satmae_env/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs"

echo "==== Segmentation Evaluation -- NWIA Iowa ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

python evaluate_seg_NWIA.py

echo ""
echo "Done: $(date)"
