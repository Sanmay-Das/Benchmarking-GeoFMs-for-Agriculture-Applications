#!/bin/bash
#SBATCH --job-name=satmae_seg_SouthMN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/satmae_seg_SouthMN_%j.out
#SBATCH --error=logs/satmae_seg_SouthMN_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

mkdir -p logs \
         predictions/satmae_fcn_SouthMN \
         predictions/satmae_fpn_SouthMN \
         predictions/satmae_psanet_SouthMN

echo "==== SatMAE Segmentation Inference — SouthMN ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

echo "====== [1/3] FCN — already done, skipping ======"
echo ""

echo "====== [2/3] FPN ======"
python infer_satmae_fpn_SouthMN.py
echo ""

echo "====== [3/3] PSANet ======"
python infer_satmae_psanet_SouthMN.py
echo ""

echo "==== Done ===="
ls -lh predictions/satmae_fcn_SouthMN/
ls -lh predictions/satmae_fpn_SouthMN/
ls -lh predictions/satmae_psanet_SouthMN/
