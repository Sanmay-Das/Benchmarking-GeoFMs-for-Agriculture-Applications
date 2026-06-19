#!/bin/bash
#SBATCH --job-name=infer_cd_SouthMN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/infer_cd_SouthMN_%j.out
#SBATCH --error=logs/infer_cd_SouthMN_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs predictions/cd_prithvi_SouthMN

echo "==== Prithvi CD Inference — SouthMN ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python infer_cd_prithvi_SouthMN.py

echo ""
echo "==== Done ===="
ls -lh predictions/cd_prithvi_SouthMN/
