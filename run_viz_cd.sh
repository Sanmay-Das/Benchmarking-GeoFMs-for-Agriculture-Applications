#!/bin/bash
#SBATCH --job-name=viz_cd
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/viz_cd_%j.out
#SBATCH --error=logs/viz_cd_%j.err
#
# Change-detection visualization launcher.
#
# Replaces the five visualize_cd_*.sh wrappers.
#
#     sbatch run_viz_cd.sh --region SouthMN
#     sbatch run_viz_cd.sh --region NWIA --models Prithvi SatMAE
#
# Arguments are passed through to visualize_cd.py. Outputs land under
# $MSR_OUTPUT_ROOT/predictions/; override the walltime at submit time with
# `sbatch --time=...` when a region needs longer than the 4h default.

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
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

python visualize_cd.py "$@"

echo ""
echo "Outputs under $MSR_OUTPUT_ROOT/visualizations/"
ls -lh "$MSR_OUTPUT_ROOT/visualizations/" 2>/dev/null || true
