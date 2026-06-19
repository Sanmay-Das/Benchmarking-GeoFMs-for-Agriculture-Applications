#!/bin/bash
#SBATCH --job-name=satmae_fpn_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/satmae_fpn_NWIAinfer_%j.out
#SBATCH --error=logs/satmae_fpn_NWIAinfer_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

mkdir -p logs
mkdir -p predictions/satmae_fpn_NWIA

echo "==== SatMAE + FPN Sliding Window Inference - NWIA ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python infer_satmae_fpn_NWIA.py

echo ""
echo "==== Done ===="
ls -lh predictions/satmae_fpn_NWIA/
