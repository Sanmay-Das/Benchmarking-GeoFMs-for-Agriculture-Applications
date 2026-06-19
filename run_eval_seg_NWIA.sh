#!/bin/bash
#SBATCH --job-name=eval_seg_NWIA
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_seg_NWIA_%j.out
#SBATCH --error=logs/eval_seg_NWIA_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

mkdir -p logs

echo "==== Segmentation Evaluation — NWIA Iowa ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

python evaluate_seg_NWIA.py

echo ""
echo "Done: $(date)"
