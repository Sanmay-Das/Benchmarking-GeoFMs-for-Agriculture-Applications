#!/bin/bash
#SBATCH --job-name=viscd_SouthCA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/viscd_SouthCA_%j.out
#SBATCH --error=logs/viscd_SouthCA_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs

echo "==== CD Visualization: SouthCA (SpectralGPT + Prithvi + SatMAE) ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python visualize_cd_SouthCA.py

echo ""
echo "==== Done ===="
ls -lh visualizations/cd_SouthCA/
