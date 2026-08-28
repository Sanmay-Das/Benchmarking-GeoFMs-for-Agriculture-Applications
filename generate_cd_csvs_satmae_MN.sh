#!/bin/bash
#SBATCH --job-name=gen_csv_satmae_MN
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gen_csv_satmae_MN_%j.out
#SBATCH --error=logs/gen_csv_satmae_MN_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs"

echo "==== Generating SatMAE MN CSVs ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

python generate_cd_csvs_satmae_MN.py

echo ""
echo "==== Done ===="
ls -lh change_detection_chips/satmae/NorthMN_chips.csv
ls -lh change_detection_chips/satmae/CentMN_chips.csv
ls -lh change_detection_chips/satmae/SouthMN_chips.csv
