#!/bin/bash
#SBATCH --job-name=satmae_vismap
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/satmae_vismap_%j.out
#SBATCH --error=logs/satmae_vismap_%j.err

set -euo pipefail

source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/configs/paths.sh"

cd "$GFM_ROOT"
# Python environment. Set GFM_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${GFM_VENV:-}" ] && [ -f "$GFM_ROOT/satmae_env/bin/activate" ]; then
    source "$GFM_ROOT/satmae_env/bin/activate"
fi

mkdir -p "$GFM_OUTPUT_ROOT/logs"

echo "==== SatMAE Colored Map Generation ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo ""

python scripts/vismap_satmae_fpn_NWIA.py
python scripts/vismap_satmae_fcn_NWIA.py
python scripts/vismap_satmae_psanet_NWIA.py

echo ""
echo "==== Done ===="
ls -lh predictions/colored_maps/NWIA_SatMAE_*
