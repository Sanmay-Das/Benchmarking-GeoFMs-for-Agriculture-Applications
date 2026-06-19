#!/bin/bash
#SBATCH --job-name=viz_cd_NWIA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/viz_cd_NWIA_%j.out
#SBATCH --error=logs/viz_cd_NWIA_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs visualizations/cd_NWIA

echo "Running CD visualization..."
python visualize_cd_NWIA.py

echo "Done. Outputs in: visualizations/cd_NWIA/"
