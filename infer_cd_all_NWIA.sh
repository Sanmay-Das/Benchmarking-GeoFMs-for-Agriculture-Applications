#!/bin/bash
#SBATCH --job-name=infer_cd_NWIA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/infer_cd_NWIA_%j.out
#SBATCH --error=logs/infer_cd_NWIA_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs predictions/cd_spectralgpt_NWIA predictions/cd_prithvi_NWIA predictions/cd_satmae_NWIA

echo "=============================="
echo "SpectralGPT CD Inference"
echo "=============================="
python infer_cd_spectralgpt_NWIA.py

echo ""
echo "=============================="
echo "Prithvi CD Inference"
echo "=============================="
python infer_cd_prithvi_NWIA.py

echo ""
echo "=============================="
echo "SatMAE CD Inference"
echo "=============================="
python infer_cd_satmae_NWIA.py

echo ""
echo "All inference complete."
