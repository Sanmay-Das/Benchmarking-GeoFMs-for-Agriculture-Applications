#!/bin/bash
#SBATCH --job-name=infer_cd_sgpt_SouthCA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/infer_cd_sgpt_SouthCA_%j.out
#SBATCH --error=logs/infer_cd_sgpt_SouthCA_%j.err

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

mkdir -p "$MSR_OUTPUT_ROOT/logs" "$MSR_OUTPUT_ROOT/predictions/cd_spectralgpt_SouthCA"

echo "==== SpectralGPT CD Inference -- SouthCA ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

python infer_cd_spectralgpt_SouthCA.py

echo ""
echo "==== Done ===="
ls -lh predictions/cd_spectralgpt_SouthCA/
