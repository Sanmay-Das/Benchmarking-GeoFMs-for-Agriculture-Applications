#!/bin/bash
#SBATCH --job-name=satmae_vismap
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/satmae_vismap_%j.out
#SBATCH --error=logs/satmae_vismap_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

mkdir -p logs

echo "==== SatMAE Colored Map Generation ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python scripts/vismap_satmae_fpn_NWIA.py
python scripts/vismap_satmae_fcn_NWIA.py
python scripts/vismap_satmae_psanet_NWIA.py

echo ""
echo "==== Done ===="
ls -lh predictions/colored_maps/NWIA_SatMAE_*
