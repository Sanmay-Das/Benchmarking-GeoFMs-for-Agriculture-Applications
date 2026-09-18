#!/bin/bash
#SBATCH --job-name=train_cd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_cd_%j.out
#SBATCH --error=logs/train_cd_%j.err
#
# Change-detection training launcher.
#
# Replaces the thirteen <model>_cd_sentinel2[_<STATE>].sh wrappers, which
# differed only in a job name, the per-state training script they invoked and
# the output directory. The geographic split is now an argument, resolved from
# registry.CD_SPLITS.
#
#     sbatch train_cd.sh --model prithvi --state california
#     sbatch train_cd.sh --model satmae  --state minnesota
#
# --state is the documented form, matching reproduce.sh and fetch.py. --split
# still takes the internal codes (IA, CA, MN, NC) for existing job scripts.
#
# States: iowa (CentIA/EastIA/NWIA), california (NorthCA/CentCA/SouthCA),
#         minnesota (NorthMN/CentMN/SouthMN), north_carolina (NENC/ECNC/EastNC).
#
# Checkpoints are written under $MSR_OUTPUT_ROOT/checkpoints/cd/, not into the
# source tree as the original wrappers did.

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

MODEL=""
SPLIT="IA"
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --state)
            SPLIT=$(python3 -c "
import sys; sys.path.insert(0, '$MSR_ROOT/configs')
import registry as R; print(R.state('$2')['cd'])") || exit 2
            shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "usage: sbatch train_cd.sh --model {satmae|spectralgpt|prithvi}" \
         "--state {iowa|minnesota|north_carolina|california}" >&2
    exit 2
fi

case "$MODEL" in
    satmae)
        CODE_DIR="SatMAE/ChangeDetection"
        SCRIPT="train_cd_satmae.py"
        PRETRAIN="$MSR_WEIGHTS/pretrain-vit-large-e199.pth"
        HYPER=(--lr 0.0001 --warmup-epochs 0 --batch_size 16 --epochs 60) ;;
    spectralgpt)
        CODE_DIR="IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection"
        SCRIPT="train_cd_spectralgpt.py"
        PRETRAIN="$MSR_WEIGHTS/SpectralGPT+.pth"
        HYPER=(--lr 0.0001 --warmup-epochs 0) ;;
    prithvi)
        CODE_DIR="prithvi_finetune/ChangeDetection"
        SCRIPT="train_cd_prithvi.py"
        PRETRAIN="$MSR_WEIGHTS/Prithvi_EO_V1_100M.pt"
        HYPER=(--lr 0.0001 --warmup-epochs 0 --batch_size 16 --epochs 60) ;;
    *)
        echo "unknown model: $MODEL" >&2; exit 2 ;;
esac

# Cluster module system (UCR HPCC). Skipped when unavailable, e.g. on a
# workstation where CUDA is already on the path.
if command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.1
fi

# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

OUT_DIR="$MSR_OUTPUT_ROOT/checkpoints/cd/cd_train_${MODEL}_${SPLIT}"
mkdir -p "$MSR_OUTPUT_ROOT/logs" "$OUT_DIR"

cd "$MSR_ROOT/$CODE_DIR"

# Derive a unique port from the job id so concurrent jobs on one node do not clash.
MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))

echo "==== ${MODEL} CD Training -- split ${SPLIT} ===="
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Out:  $OUT_DIR"
echo ""

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$MASTER_PORT --use_env "$SCRIPT" \
    --split "$SPLIT" \
    --data-root "$MSR_DATA_ROOT/change_detection_chips/$MODEL" \
    --pretrain-path "$PRETRAIN" \
    --output-dir "$OUT_DIR" \
    "${HYPER[@]}" "${EXTRA[@]}"

echo ""
echo "==== Done ===="
ls -lh "$OUT_DIR"
