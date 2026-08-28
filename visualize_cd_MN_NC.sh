#!/bin/bash
#SBATCH --job-name=viscd_MN_NC
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/viscd_MN_NC_%j.out
#SBATCH --error=logs/viscd_MN_NC_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs"

echo "==== CD Visualization: SouthMN ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

python visualize_cd_SouthMN.py

echo ""
echo "==== CD Visualization: EastNC ===="
echo ""

python visualize_cd_EastNC.py

echo ""
echo "==== Done ===="
ls -lh visualizations/cd_SouthMN/
ls -lh visualizations/cd_EastNC/
