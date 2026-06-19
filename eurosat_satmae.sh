#!/bin/bash
#SBATCH --job-name=satmae_eurosat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/satmae_eurosat_%j.out
#SBATCH --error=logs/satmae_eurosat_%j.err

set -euo pipefail

echo "=========================================="
echo "SatMAE EuroSAT Finetuning"
echo "Target: 95.74% (Paper Table 8)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
hostname
echo ""

# Load modules
module purge
module load cuda/11.7

# Navigate to project
cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate
cd SatMAE  # ← ADD THIS LINE

# Single-GPU distributed setup
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0

# Verify Python environment
echo "Verifying environment..."
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
PY
echo ""

echo "Starting training..."
echo ""

python main_finetune.py \
    --output_dir eurosat_output \
    --log_dir eurosat_output \
    --batch_size 64 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 2e-3 \
    --layer_decay 0.75 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --finetune /bigdata/eldawylab/sdas050/MS_Research/weights/fmow_pretrain.pth \
    --num_workers 8 \
    --train_path data/EuroSAT/train.csv \
    --test_path data/EuroSAT/test.csv \
    --nb_classes 10 \
    --input_size 224 \
    --patch_size 16

EXIT_STATUS=$?

echo ""
echo "================================================"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Training Complete!"
    echo ""
    echo "Final Results:"
    if [ -f eurosat_output/log.txt ]; then
        grep "Acc@1" eurosat_output/log.txt | tail -1
    fi
    echo ""
    echo "Expected: 95.74%"
    echo ""
    echo "Output files:"
    ls -lh eurosat_output/
else
    echo "Training Failed!"
fi
echo "Completed: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "================================================"

exit $EXIT_STATUS