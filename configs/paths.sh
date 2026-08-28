#!/bin/bash
# Shared path setup for the SLURM wrapper scripts.
#
# Source this from any launcher:
#     source "$(dirname "${BASH_SOURCE[0]}")/configs/paths.sh"
#
# Override any root from the environment before submitting, e.g.
#     export MSR_DATA_ROOT=/scratch/me/geofm-data
#     sbatch infer_cd_spectralgpt_MN.sh

# Repository root: parent of the directory holding this file.
_this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MSR_ROOT="${MSR_ROOT:-$(dirname "$_this_dir")}"

export MSR_DATA_ROOT="${MSR_DATA_ROOT:-$MSR_ROOT/data}"
export MSR_OUTPUT_ROOT="${MSR_OUTPUT_ROOT:-$MSR_ROOT/outputs}"
export MSR_WEIGHTS="${MSR_WEIGHTS:-$MSR_ROOT/weights}"

# Python environment. Set MSR_VENV to the venv created from requirements/,
# or leave unset to use whatever python is already active.
if [ -n "${MSR_VENV:-}" ] && [ -f "$MSR_VENV/bin/activate" ]; then
    source "$MSR_VENV/bin/activate"
fi

mkdir -p "$MSR_OUTPUT_ROOT/logs" "$MSR_OUTPUT_ROOT/predictions"
