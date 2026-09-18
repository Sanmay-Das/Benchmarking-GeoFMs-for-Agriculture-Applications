#!/bin/bash
#SBATCH --job-name=class_dist
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=$MSR_ROOT/logs/class_distribution_%j.out
#SBATCH --error=$MSR_ROOT/logs/class_distribution_%j.err

set -euo pipefail

source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/satmae_env/bin/activate" ]; then
    source "$MSR_ROOT/satmae_env/bin/activate"
fi

echo "==== Class Distribution Analysis ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

python scripts/class_distribution.py

echo ""
echo "Done: $(date)"
