#!/bin/bash
#SBATCH --job-name=train_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_seg_%j.out
#SBATCH --error=logs/train_seg_%j.err
#
# SatMAE segmentation training launcher.
#
# Replaces the six satmae_sentinel2[_<head>][_MN].sh wrappers, which differed
# only in a job name, which decoder entrypoint they invoked and which chip set
# they read.
#
#     sbatch train_seg.sh --head fpn --split Iowa
#     sbatch train_seg.sh --head psanet --split MN
#
# Heads: fcn and fpn share their hyperparameters; psanet additionally sets
# --optimizer sgd, --shrink_factor and --zoom_factor, as the original
# satmae_sentinel2.sh did.
#
# Checkpoints go to $MSR_OUTPUT_ROOT/checkpoints/seg/, not into SatMAE/ as the
# original wrappers did.

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

HEAD="fpn"
SPLIT="Iowa"
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --head)  HEAD="$2";  shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

case "$HEAD" in
    fcn)    ENTRY="main_finetune_fcn.py"; HEAD_ARGS=() ;;
    fpn)    ENTRY="main_finetune_fpn.py"; HEAD_ARGS=() ;;
    psanet) ENTRY="main_finetune.py"
            HEAD_ARGS=(--optimizer sgd --shrink_factor 2 --zoom_factor 8) ;;
    *) echo "unknown head: $HEAD (expected fcn, fpn or psanet)" >&2; exit 2 ;;
esac

# The Iowa run reads the shared multitemporal chip set; MN has its own.
case "$SPLIT" in
    Iowa) CHIPS="$MSR_DATA_ROOT/SatMAE_chips_multitemporal" ;;
    MN)   CHIPS="$MSR_DATA_ROOT/SatMAE_chips_MN" ;;
    *) echo "unknown split: $SPLIT (expected Iowa or MN)" >&2; exit 2 ;;
esac
SPLITS_DIR="$MSR_DATA_ROOT/SatMAE_chips_multitemporal/$SPLIT"

SUFFIX=""
[ "$HEAD" != "psanet" ] && SUFFIX="_$HEAD"
OUTPUT_DIR="$MSR_OUTPUT_ROOT/checkpoints/seg/output_seg_${SPLIT}${SUFFIX}"
LOG_DIR="$OUTPUT_DIR/logs"

if command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.1
fi

# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./satmae_env if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/satmae_env/bin/activate" ]; then
    source "$MSR_ROOT/satmae_env/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs" "$LOG_DIR"
cd "$MSR_ROOT/SatMAE"

MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))

echo "==== SatMAE segmentation -- head=$HEAD split=$SPLIT ===="
echo "Job:   ${SLURM_JOB_ID:-local}"
echo "Node:  $(hostname)"
echo "Chips: $CHIPS"
echo "Out:   $OUTPUT_DIR"
echo ""

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    --use_env \
    "$ENTRY" \
    --model vit_large_patch16 \
    --input_size 96 \
    --patch_size 8 \
    --nb_classes 14 \
    --drop_path 0.2 \
    --epochs 100 \
    --batch_size 8 \
    --accum_iter 16 \
    --finetune "$MSR_WEIGHTS/pretrain-vit-large-e199.pth" \
    --data_path "$CHIPS" \
    --train_path "$SPLITS_DIR/train.txt" \
    --test_path "$SPLITS_DIR/val.txt" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --lr 1e-2 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --poly_power 0.9 \
    --ignore_index 0 \
    --save_every 5 \
    --num_workers 8 \
    --pin_mem \
    --seed 42 \
    --dist_url env:// \
    "${HEAD_ARGS[@]}" "${EXTRA[@]}"

echo ""
echo "==== Done ===="
ls -lh "$OUTPUT_DIR"
