#!/bin/bash
#SBATCH --job-name=satmae_fcn_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/satmae_fcn_NWIAinfer_%j.out
#SBATCH --error=logs/satmae_fcn_NWIAinfer_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

mkdir -p logs
mkdir -p predictions/satmae_fcn_NWIA

echo "==== SatMAE + FCNHead Sliding Window Inference - NWIA ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python infer_satmae_fcn_NWIA.py

echo ""
echo "==== Done ===="
ls -lh predictions/satmae_fcn_NWIA/
