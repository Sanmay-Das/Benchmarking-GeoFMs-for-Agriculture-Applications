#!/bin/bash
#SBATCH --job-name=viz_seg_SouthMN
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/viz_seg_SouthMN_%j.out
#SBATCH --error=logs/viz_seg_SouthMN_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

mkdir -p logs visualizations/seg_SouthMN

echo "==== Segmentation Visualization — SouthMN ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname) (CPU-only)"
echo "Start: $(date)"
echo ""

python visualize_seg_satmae_SouthMN.py

echo ""
echo "Done: $(date)"
ls -lh visualizations/seg_SouthMN/
