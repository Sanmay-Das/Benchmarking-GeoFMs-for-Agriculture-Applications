#!/bin/bash
#SBATCH --job-name=make_manifests
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/make_manifests_%j.out
#SBATCH --error=logs/make_manifests_%j.err
#
# Rebuild the change-detection chip manifests from the chips themselves.
#
# Manifests are derived data: run this once after downloading and unpacking
# the chip archives. Replaces the per-model/per-region generate_cd_csvs_*.sh
# scripts, which all called the same code with different arguments.
#
#     ./make_manifests.sh              # every model
#     ./make_manifests.sh satmae       # one model
#
# Paths come from configs/paths.sh; set MSR_DATA_ROOT to wherever the chips
# were unpacked.

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(satmae spectralgpt prithvi)
fi

for model in "${MODELS[@]}"; do
    base="$MSR_DATA_ROOT/change_detection_chips/$model"
    if [ ! -d "$base" ]; then
        echo "skip $model: $base not found"
        continue
    fi
    echo "==== manifests for $model ===="
    python scripts/generate_cd_csv.py --base "$base" --region all
done

echo "Done."
