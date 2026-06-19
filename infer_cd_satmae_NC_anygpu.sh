#!/bin/bash
#SBATCH --job-name=infer_cd_satmae_EastNC_anygpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/infer_cd_satmae_EastNC_anygpu_%j.out
#SBATCH --error=logs/infer_cd_satmae_EastNC_anygpu_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs predictions/cd_satmae_EastNC

echo "==== SatMAE CD Inference — EastNC (any GPU) ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo ""

python infer_cd_satmae_EastNC.py

echo ""
echo "==== Done ===="
ls -lh predictions/cd_satmae_EastNC/
