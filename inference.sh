#!/bin/bash
#SBATCH --job-name=prithvi_infer_NWIA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prithvi_NWIA_eval_%j.out
#SBATCH --error=logs/prithvi_NWIA_eval_%j.err

set -euo pipefail

module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source prithvifmenv/bin/activate

CHECKPOINT="/bigdata/eldawylab/sdas050/MS_Research/experiments/prithvi_multi_temporal_crop_classification/best_mIoU_epoch_20.pth"

mim test mmsegmentation \
  prithvi_finetune/configs/multi_temporal_crop_classification.py \
  --checkpoint "$CHECKPOINT" \
  --launcher none \
  --eval mIoU