#!/bin/bash
#SBATCH --job-name=viscd_prithvi_SouthCA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/viscd_prithvi_SouthCA_%j.out
#SBATCH --error=logs/viscd_prithvi_SouthCA_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs

echo "==== CD Visualization: SouthCA — Prithvi ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python visualize_cd_prithvi_SouthCA.py

echo ""
echo "==== Done ===="
ls -lh visualizations/cd_SouthCA/
