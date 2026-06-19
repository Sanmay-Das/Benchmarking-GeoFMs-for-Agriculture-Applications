#!/bin/bash
#SBATCH --job-name=satmae_psanet_NC_pre
#SBATCH --partition=preempt_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/satmae_psanet_NC_pre_%j.out
#SBATCH --error=logs/satmae_psanet_NC_pre_%j.err

set -euo pipefail

SATMAE_DIR="/bigdata/eldawylab/sdas050/MS_Research/SatMAE"
DATA_ROOT="/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal"
SPLITS_DIR="/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/NC"
PRETRAIN_WEIGHTS="/bigdata/eldawylab/sdas050/MS_Research/weights/pretrain-vit-large-e199.pth"
OUTPUT_DIR="${SATMAE_DIR}/output_seg_NC"
LOG_DIR="${OUTPUT_DIR}/logs"

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1

cd /bigdata/eldawylab/sdas050/MS_Research
source satmae_env/bin/activate

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" logs

echo "=========================================="
echo "SATMAE + PSANet — NC CROP SEGMENTATION"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "Train:     NENC  |  Val: ECNC  |  Test: EastNC"
echo ""

hostname
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH="${PYTHONPATH:-}:${SATMAE_DIR}"

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

cd ${SATMAE_DIR}

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
echo "TRAINING COMPLETE — $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
