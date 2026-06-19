#!/bin/bash
#SBATCH --job-name=prithvi_mtcc_1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prithvi_IA_%j.out
#SBATCH --error=logs/prithvi_IA_%j.err

set -euo pipefail

module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source prithvifmenv/bin/activate

echo "==== Node & GPU ===="
hostname
nvidia-smi || true
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
PY
echo "===================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Run MTCC on a single GPU (no DDP)
mim train mmsegmentation \
  prithvi_finetune/configs/multi_temporal_crop_classification.py \
  --launcher none \
  --gpus 1
