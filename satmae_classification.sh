#!/bin/bash
#SBATCH --job-name=satmae_ms_finetune_1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=13:00:00
#SBATCH --output=logs/satmae_mscl_IA_finetune_%j.out
#SBATCH --error=logs/satmae_mscl_IA_finetune_%j.err

set -euo pipefail

module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH="${PYTHONPATH:-}:/bigdata/eldawylab/sdas050/MS_Research/SatMAE"

# FOR SINGLE GPU:
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

cd /bigdata/eldawylab/sdas050/MS_Research/SatMAE

python main_finetune.py \
  --batch_size 16 --accum_iter 16 --blr 0.0002 \
  --epochs 30 --num_workers 16 \
  --input_size 96 --patch_size 8 \
  --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --model_type group_c \
  --grouped_bands 0 1 2 3 --grouped_bands 4 5 6 7 --grouped_bands 8 9 \
  --dataset_type sentinel \
  --train_path /bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/train.csv \
  --test_path  /bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/test.csv \
  --output_dir /bigdata/eldawylab/sdas050/MS_Research/SatMAE/output \
  --log_dir    /bigdata/eldawylab/sdas050/MS_Research/SatMAE/logs \
  --finetune   /bigdata/eldawylab/sdas050/MS_Research/weights/pretrain-vit-large-e199.pth