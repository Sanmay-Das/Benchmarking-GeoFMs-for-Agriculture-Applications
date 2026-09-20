#!/bin/bash
#SBATCH --job-name=viz_seg_SouthMN
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/viz_seg_SouthMN_%j.out
#SBATCH --error=logs/viz_seg_SouthMN_%j.err

set -euo pipefail

source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/configs/paths.sh"

source /etc/profile.d/modules.sh
module purge

cd "$GFM_ROOT"
# Python environment. Set GFM_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${GFM_VENV:-}" ] && [ -f "$GFM_ROOT/satmae_env/bin/activate" ]; then
    source "$GFM_ROOT/satmae_env/bin/activate"
fi

mkdir -p "$GFM_OUTPUT_ROOT/logs" "$GFM_OUTPUT_ROOT/visualizations/seg_SouthMN"

echo "==== Segmentation Visualization -- SouthMN ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname) (CPU-only)"
echo "Start: $(date)"
echo ""

python scripts/visualize_seg_satmae_SouthMN.py

echo ""
echo "Done: $(date)"
ls -lh visualizations/seg_SouthMN/
