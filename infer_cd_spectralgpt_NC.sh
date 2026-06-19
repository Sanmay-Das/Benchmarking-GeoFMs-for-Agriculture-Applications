#!/bin/bash
#SBATCH --job-name=infer_cd_sgpt_EastNC
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/infer_cd_sgpt_EastNC_%j.out
#SBATCH --error=logs/infer_cd_sgpt_EastNC_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs predictions/cd_spectralgpt_EastNC

echo "==== SpectralGPT CD Inference — EastNC ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python infer_cd_spectralgpt_EastNC.py

echo ""
echo "==== Done ===="
ls -lh predictions/cd_spectralgpt_EastNC/
