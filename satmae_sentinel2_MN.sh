#!/bin/bash
#SBATCH --job-name=satmae_seg_MN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/satmae_seg_MN_%j.out
#SBATCH --error=logs/satmae_seg_MN_%j.err

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

echo "=========================================="
echo "SATMAE + PSANET CROP SEGMENTATION"
echo "main_finetune.py -- Single GPU A100"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo ""

# ============================================================
# PATHS
# ============================================================
SATMAE_DIR="$MSR_ROOT/SatMAE"

# Parent directory -- dataset_seg.py scans subdirs (NorthMN, CentMN, SouthMN)
# to find each chip. No need to copy chips into MN/.
DATA_ROOT="$MSR_ROOT/SatMAE_chips_MN"

# Split files live in SatMAE_chips_multitemporal/MN/
SPLITS_DIR="$MSR_ROOT/SatMAE_chips_multitemporal/MN"

PRETRAIN_WEIGHTS="$MSR_ROOT/weights/pretrain-vit-large-e199.pth"
OUTPUT_DIR="${SATMAE_DIR}/output_seg_MN"
LOG_DIR="${OUTPUT_DIR}/logs"

# ============================================================
# Environment
# ============================================================
# Cluster module system (UCR HPCC). Skipped when unavailable, e.g. on a
# workstation where CUDA is already on the path.
if command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.1
fi

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/satmae_env/bin/activate" ]; then
    source "$MSR_ROOT/satmae_env/bin/activate"
fi

mkdir -p "${SATMAE_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "==== Environment ===="
hostname
nvidia-smi || true
python - <<'PY'
import torch
print("PyTorch:        ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:            ", torch.cuda.get_device_name(0))
    print("VRAM:           ", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
PY
echo "====================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH="${PYTHONPATH:-}:${SATMAE_DIR}"

echo ""
echo "=========================================="
echo "CONFIGURATION"
echo "=========================================="
echo "Data root:       ${DATA_ROOT}"
echo "  NorthMN/       train chips"
echo "  CentMN/        val chips"
echo "  SouthMN/       test chips"
echo "  MN/train.txt -> NorthMN chip names"
echo "  MN/val.txt   -> CentMN chip names"
echo "  MN/test.txt  -> SouthMN chip names"
echo "Pretrain:        ${PRETRAIN_WEIGHTS}"
echo "Output dir:      ${OUTPUT_DIR}"
echo ""
echo "Model:           SatMAE ViT-Large + PSANet"
echo "Encoder:         GroupChannels (3 groups x 6 bands)"
echo "Input size:      96x96"
echo "Patch size:      8"
echo "In channels:     18 (6 bands x 3 timesteps)"
echo "Classes:         14 (0=NoData ignored, 1-13=crops)"
echo ""
echo "Optimizer:       SGD (paper A.10), momentum=0.9"
echo "Encoder LR:      1e-3  (paper A.10)"
echo "Head+PSA LR:     1e-2  (paper A.10)"
echo "LR scheduler:    Polynomial decay, power=0.9 (paper A.10)"
echo "Weight decay:    1e-4  (paper A.10)"
echo "Epochs:          100"
echo "Batch size:      8"
echo "Accum iter:      16"
echo "Effective batch: 128   (8 x 16 x 1 GPU, matches paper)"
echo "Loss:            CrossEntropyLoss(ignore_index=0)"
echo "=========================================="
echo ""

cd ${SATMAE_DIR}

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    --use_env \
    main_finetune.py \
    --model vit_large_patch16 \
    --input_size 96 \
    --patch_size 8 \
    --nb_classes 14 \
    --drop_path 0.2 \
    --epochs 100 \
    --batch_size 8 \
    --accum_iter 16 \
    --finetune ${PRETRAIN_WEIGHTS} \
    --data_path ${DATA_ROOT} \
    --train_path ${SPLITS_DIR}/train.txt \
    --test_path ${SPLITS_DIR}/val.txt \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --optimizer sgd \
    --lr 1e-2 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --poly_power 0.9 \
    --ignore_index 0 \
    --shrink_factor 2 \
    --zoom_factor 8 \
    --save_every 5 \
    --num_workers 8 \
    --pin_mem \
    --seed 42 \
    --dist_url env://

echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "End time: $(date)"
echo "Output:   ${OUTPUT_DIR}"
echo "=========================================="
