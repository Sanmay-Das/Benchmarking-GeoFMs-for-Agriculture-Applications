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

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

module purge
module load cuda/12.1

cd $MSR_ROOT/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich

mkdir -p "$MSR_OUTPUT_ROOT/logs"
mkdir -p $MSR_ROOT/predictions/spectralgpt_SouthMN_terratorch

source $MSR_ROOT/spectralgptenv/bin/activate

echo "==== SpectralGPT Sliding Window Inference - SouthMN ====="
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python $MSR_ROOT/infer_spectralgpt_SouthMN.py

echo ""
echo "==== Done ====="
ls -lh $MSR_ROOT/predictions/spectralgpt_SouthMN_terratorch/