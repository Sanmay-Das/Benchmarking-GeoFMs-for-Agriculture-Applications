#!/bin/bash
#SBATCH --job-name=viscd_SouthCA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/viscd_SouthCA_%j.out
#SBATCH --error=logs/viscd_SouthCA_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs"

echo "==== CD Visualization: SouthCA (SpectralGPT + Prithvi + SatMAE) ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

python visualize_cd_SouthCA.py

echo ""
echo "==== Done ===="
ls -lh visualizations/cd_SouthCA/
