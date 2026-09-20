#!/bin/bash
# Shared path setup for the SLURM wrapper scripts.
#
# Source this from any launcher:
#     source "$(dirname "${BASH_SOURCE[0]}")/configs/paths.sh"
#
# Override any root from the environment before submitting, e.g.
#     export GFM_DATA_ROOT=/scratch/me/geofm-data
#     ./benchmark-gfm --sbatch infer --task cd --model spectralgpt --state minnesota

# The GFM_* names replace an older MSR_* set, called after the directory this
# was developed in. The old names are still honoured so existing shells and job
# scripts keep working.
for _v in ROOT DATA_ROOT OUTPUT_ROOT WEIGHTS VENV VENVS PYTHON; do
    eval "_new=\${GFM_$_v:-}"
    eval "_old=\${MSR_$_v:-}"
    if [ -z "$_new" ] && [ -n "$_old" ]; then
        eval "export GFM_$_v=\"\$MSR_$_v\""
    fi
done
unset _v _new _old

# Repository root: parent of the directory holding this file.
_this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GFM_ROOT="${GFM_ROOT:-$(dirname "$_this_dir")}"

export GFM_DATA_ROOT="${GFM_DATA_ROOT:-$GFM_ROOT/data}"
export GFM_OUTPUT_ROOT="${GFM_OUTPUT_ROOT:-$GFM_ROOT/outputs}"
export GFM_WEIGHTS="${GFM_WEIGHTS:-$GFM_ROOT/weights}"

# Python environment. Set GFM_VENV to the venv created from requirements/,
# or leave unset to use whatever python is already active.
if [ -n "${GFM_VENV:-}" ] && [ -f "$GFM_VENV/bin/activate" ]; then
    source "$GFM_VENV/bin/activate"
fi

mkdir -p "$GFM_OUTPUT_ROOT/logs" "$GFM_OUTPUT_ROOT/predictions"
