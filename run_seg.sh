#!/bin/bash
#SBATCH --job-name=infer_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/infer_seg_%j.out
#SBATCH --error=logs/infer_seg_%j.err
#
# Semantic-segmentation inference launcher.
#
# Replaces the segmentation wrappers (infer_satmae_*.sh, prithvi_inference.sh,
# spectralgpt_inference.sh, infer_satmae_seg_SouthMN.sh), which differed only
# in a job name and the script they invoked.
#
#     sbatch run_seg.sh --model satmae --head fpn --region NWIA
#     sbatch run_seg.sh --model prithvi --region all
#     sbatch run_seg.sh --all --region NWIA      # every backbone, one region
#
# Arguments are passed through to infer_seg.py. Outputs land under
# $MSR_OUTPUT_ROOT/predictions/; override the walltime at submit time with
# `sbatch --time=...` when a region needs longer than the 4h default.

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/configs/paths.sh"

# Cluster module system (UCR HPCC). Skipped when unavailable, e.g. on a
# workstation where CUDA is already on the path.
if command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.1
fi

cd "$MSR_ROOT"
# Python environment. Set MSR_VENV to your venv built from
# requirements/; falls back to ./spectralgptenv if present.
if [ -z "${MSR_VENV:-}" ] && [ -f "$MSR_ROOT/spectralgptenv/bin/activate" ]; then
    source "$MSR_ROOT/spectralgptenv/bin/activate"
fi

# --all runs every backbone over the same region.
if [ "${1:-}" = "--all" ]; then
    shift
    for model in satmae spectralgpt prithvi; do
        echo "=============================="
        echo "$model"
        echo "=============================="
        if [ "$model" = "satmae" ]; then
            # SatMAE has three decoders; run each.
            for head in fcn fpn psanet; do
                python infer_seg.py --model satmae --head "$head" "$@"
            done
        else
            python infer_seg.py --model "$model" "$@"
        fi
    done
else
    python infer_seg.py "$@"
fi

echo ""
echo "Outputs under $MSR_OUTPUT_ROOT/predictions/"
ls -lh "$MSR_OUTPUT_ROOT/predictions/" 2>/dev/null || true
