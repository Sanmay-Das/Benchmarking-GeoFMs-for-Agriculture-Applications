#!/bin/bash
#SBATCH --job-name=prithvi_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prithvi_SouthMNinfer_%j.out
#SBATCH --error=logs/prithvi_SouthMNinfer_%j.err

set -euo pipefail

module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source prithvifmenv/bin/activate

mkdir -p logs
mkdir -p predictions/prithvi_NWIA_terratorch

echo "==== Prithvi Sliding Window Inference - EastNC ====="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Execute the new sliding window script
#python infer_prithvi_SouthMN.py
python infer_prithvi_NWIA.py

echo ""
echo "==== Done ====="
echo ""

ls -lh predictions/prithvi_NWIA_terratorch/