#!/bin/bash
#SBATCH --job-name=vismap_SouthMN
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=$MSR_ROOT/logs/vismap_SouthMN_%j.out
#SBATCH --error=$MSR_ROOT/logs/vismap_SouthMN_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/satmae_env/bin/activate" ]; then
    source "$MSR_ROOT/satmae_env/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs"

echo "==== SatMAE FPN SouthMN PNG Export ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

python scripts/vismap_satmae_fpn_SouthMN.py

echo ""
echo "==== Done ===="
ls -lh predictions/colored_maps/SouthMN_*
