#!/bin/bash
#SBATCH --job-name=viz_cd_NWIA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/viz_cd_NWIA_%j.out
#SBATCH --error=logs/viz_cd_NWIA_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs" "$MSR_OUTPUT_ROOT/visualizations/cd_NWIA"

echo "Running CD visualization..."
python visualize_cd_NWIA.py

echo "Done. Outputs in: visualizations/cd_NWIA/"
