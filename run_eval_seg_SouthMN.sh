#!/bin/bash
#SBATCH --job-name=eval_seg_SouthMN
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=$MSR_ROOT/logs/eval_seg_SouthMN_%j.out
#SBATCH --error=$MSR_ROOT/logs/eval_seg_SouthMN_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"
cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/satmae_env/bin/activate" ]; then
    source "$MSR_ROOT/satmae_env/bin/activate"
fi
mkdir -p "$MSR_OUTPUT_ROOT/logs"
echo "Start: $(date)"
python evaluate_seg_SouthMN.py
echo "Done: $(date)"
