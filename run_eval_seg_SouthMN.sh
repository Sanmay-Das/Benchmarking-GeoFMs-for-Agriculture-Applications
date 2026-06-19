#!/bin/bash
#SBATCH --job-name=eval_seg_SouthMN
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/bigdata/eldawylab/sdas050/MS_Research/logs/eval_seg_SouthMN_%j.out
#SBATCH --error=/bigdata/eldawylab/sdas050/MS_Research/logs/eval_seg_SouthMN_%j.err

set -euo pipefail
cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate
mkdir -p logs
echo "Start: $(date)"
python evaluate_seg_SouthMN.py
echo "Done: $(date)"
