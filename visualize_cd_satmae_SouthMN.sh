#!/bin/bash
#SBATCH --job-name=viscd_satmae_SouthMN
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viscd_satmae_SouthMN_%j.out
#SBATCH --error=logs/viscd_satmae_SouthMN_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs

echo "==== CD Visualization: SatMAE SouthMN ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python visualize_cd_satmae_SouthMN.py

echo ""
echo "==== Done ===="
ls -lh visualizations/cd_SouthMN/
