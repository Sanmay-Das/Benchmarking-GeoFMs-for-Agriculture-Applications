#!/bin/bash
#SBATCH --job-name=spectral_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/spectralgpt_SouthMNinfer_%j.out
#SBATCH --error=logs/spectralgpt_SouthMNinfer_%j.err

set -euo pipefail

module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich

mkdir -p logs
mkdir -p /bigdata/eldawylab/sdas050/MS_Research/predictions/spectralgpt_SouthMN_terratorch

source /bigdata/eldawylab/sdas050/MS_Research/spectralgptenv/bin/activate

echo "==== SpectralGPT Sliding Window Inference - SouthMN ====="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python /bigdata/eldawylab/sdas050/MS_Research/infer_spectralgpt_SouthMN.py

echo ""
echo "==== Done ====="
ls -lh /bigdata/eldawylab/sdas050/MS_Research/predictions/spectralgpt_SouthMN_terratorch/